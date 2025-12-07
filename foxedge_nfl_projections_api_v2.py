#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoxEdge NFL Projections — v6 (real-world ranges)
- EPA-first with multi-season priors and team-specific current-season pace
- Reduced shrinkage, wider EPA caps, smarter variance, stronger pace impact
- Same CLI/QoL as v5.1 so you can filter and export cleanly

Usage examples:
  python foxedge_nfl_projections_api_v2.py \
    --seasons 2020 2021 2022 2023 2024 2025 \
    --only-today \
    --n-sims 50000 --seed 14 \
    --format csv \
    --out ./projections_wk14.csv

  python foxedge_nfl_projections_api_v2.py \
    --seasons 2022 2023 2024 2025 \
    --from-week 12 --to-week 13\
    --lite --format jsonl \
    --out ./proj_w13.jsonl
"""

import os, sys, argparse, warnings, numpy as np, pandas as pd, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 240)

def log(msg: str):
    print(f"[foxedge] {msg}", flush=True)

# -------------------------
# Data fetch (EPA required)
# -------------------------

def fetch_pbp_and_sched(seasons: List[int]) -> Dict[str, pd.DataFrame]:
    try:
        import nfl_data_py as nfl
    except Exception as e:
        raise RuntimeError("nfl_data_py is not installed. Run: pip install nfl_data_py") from e
    tables: Dict[str, pd.DataFrame] = {}
    schedules = nfl.import_schedules(years=seasons)
    if schedules is None or schedules.empty:
        raise RuntimeError("import_schedules returned no rows")
    tables["schedules"] = schedules
    log(f"[api] schedules rows={len(schedules)}")
    pbp = None; errs = []
    for fn in ("import_pbp_data","import_pbp"):
        f = getattr(nfl, fn, None)
        if f is None:
            errs.append(f"nfl_data_py.{fn} missing"); continue
        try:
            tmp = f(years=seasons, downcast=True)
            if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                pbp = tmp; src = fn; break
            errs.append(f"{fn} returned empty")
        except Exception as e:
            errs.append(f"{fn} error: {e}")
    if pbp is None:
        raise RuntimeError("Failed to fetch PBP EPA/play. " + "; ".join(errs))
    tables["pbp"] = pbp
    log(f"[api] pbp rows={len(pbp)} via {src}")
    # injuries optional
    inj = None
    for fn in ("import_injuries","import_injury_reports","import_injury"):
        f = getattr(nfl, fn, None)
        if f is None: continue
        try:
            tmp = f(years=seasons)
            if isinstance(tmp, pd.DataFrame) and not tmp.empty:
                inj = tmp; break
        except Exception:
            pass
    if inj is not None:
        tables["injuries"] = inj
        log(f"[api] injuries rows={len(inj)}")
    else:
        log("[api] injuries unavailable; proceeding without")
    return tables

# -------------------------
# Normalization
# -------------------------

def clean_schedule(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    s.columns = [c.lower() for c in s.columns]
    if "home_team" in s.columns: s.rename(columns={"home_team":"home"}, inplace=True)
    if "away_team" in s.columns: s.rename(columns={"away_team":"away"}, inplace=True)
    for cand in ["gameday","game_date","start_time","kickoff","datetime"]:
        if cand in s.columns:
            s["game_date"] = pd.to_datetime(s[cand], utc=True, errors="coerce")
            break
    s["kickoff_et"] = s.get("game_date", pd.Series([pd.NaT]*len(s))).dt.tz_convert("US/Eastern") if "game_date" in s else pd.NaT
    s["home"] = s["home"].astype(str).str.upper()
    s["away"] = s["away"].astype(str).str.upper()
    return s

def pbp_to_game(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [c.lower() for c in x.columns]
    need = {"posteam","defteam","epa","game_id","season","week"}
    if not need.issubset(set(x.columns)):
        raise RuntimeError(f"PBP missing required columns: {need - set(x.columns)}")
    if "play_type" in x.columns:
        mask = x["play_type"].astype(str).str.lower().isin(["pass","run","rush","sack","scramble","qb_kneel"])
        if mask.any(): x = x[mask]
    g = x.groupby(["season","week","game_id","posteam","defteam"], dropna=False).agg(
        plays=("epa","count"),
        epa_mean=("epa","mean")
    ).reset_index()
    g["off_team"] = g["posteam"].astype(str).str.upper()
    g["def_team"] = g["defteam"].astype(str).str.upper()
    return g[["season","week","game_id","off_team","def_team","plays","epa_mean"]]

# -------------------------
# Ratings machinery (looser)
# -------------------------

def ridge(off_ids, def_ids, rows, y, lam=4.0):
    n_off, n_def = len(off_ids), len(def_ids)
    m = len(rows)
    X = np.zeros((m, n_off + n_def + 1), dtype=float)
    for i, (oi, di, is_home) in enumerate(rows):
        X[i, oi] = 1.0
        X[i, n_off + di] = -1.0
        X[i, n_off + n_def] = 1.0 if is_home else 0.0
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta[:n_off], beta[n_off:n_off+n_def], float(beta[-1])

def ratings_from_history(history: pd.DataFrame, schedule: pd.DataFrame, lam=4.0) -> Tuple[pd.DataFrame, float, pd.Series]:
    df = history.copy()
    df.columns = [c.lower() for c in df.columns]
    s = schedule.copy(); s.columns = [c.lower() for c in s.columns]
    home_map = s.set_index("game_id")["home"].astype(str).str.upper().to_dict() if "game_id" in s.columns else {}
    df["off_team"] = df["off_team"].astype(str).str.upper()
    df["def_team"] = df["def_team"].astype(str).str.upper()
    df["epa_mean"] = pd.to_numeric(df["epa_mean"], errors="coerce").fillna(0.0).clip(-0.8, 0.8)  # wider cap
    teams = sorted(pd.unique(pd.concat([df["off_team"], df["def_team"]])))
    off_ids = {t:i for i,t in enumerate(teams)}
    def_ids = {t:i for i,t in enumerate(teams)}
    gids = df.get("game_id", pd.Series([None]*len(df)))
    home_flags = [str(o).upper() == home_map.get(g, "") for o,g in zip(df["off_team"], gids)]
    y = df["epa_mean"].to_numpy() - np.nanmean(df["epa_mean"].to_numpy())
    off, deff, hfa = ridge(off_ids, def_ids,
                           list(zip([off_ids[o] for o in df["off_team"]],
                                    [def_ids[d] for d in df["def_team"]],
                                    home_flags)),
                           y, lam=lam)
    ratings = pd.DataFrame({"team": teams, "off_rating": off, "def_rating": deff})
    counts = df.groupby("off_team").size().reindex(teams, fill_value=0)
    return ratings, float(hfa), counts

def recency_weighted_prior(history_all: pd.DataFrame, schedule: pd.DataFrame, current_season: int) -> Tuple[pd.DataFrame, float]:
    priors = []; hfas = []
    for season in sorted(history_all["season"].dropna().unique()):
        season = int(season)
        if season >= current_season: continue
        hist_s = history_all[history_all["season"] == season]
        if hist_s.empty: continue
        r_s, hfa_s, _ = ratings_from_history(hist_s, schedule, lam=5.0)
        decay = 0.7 ** (current_season - season)   # slower decay than v5.1
        r_s["off_rating"] *= decay
        r_s["def_rating"] *= decay
        hfas.append(hfa_s * decay)
        r_s["season"] = season
        priors.append(r_s)
    if not priors:
        return pd.DataFrame(columns=["team","off_rating","def_rating"]), 0.0
    R = pd.concat(priors, ignore_index=True)
    prior = R.groupby("team", as_index=False).agg(off_rating=("off_rating","mean"),
                                                  def_rating=("def_rating","mean"))
    prior["off_rating"] = prior["off_rating"].clip(-0.40, 0.40)
    prior["def_rating"] = prior["def_rating"].clip(-0.40, 0.40)
    hfa_prior = float(np.clip(np.nanmean(hfas) if hfas else 0.0, -0.10, 0.10))
    hfa_prior = float(0.5 * hfa_prior + 0.5 * 0.02)
    return prior, hfa_prior

def blend_prior_current(prior: pd.DataFrame, cur: pd.DataFrame, cur_counts: pd.Series) -> pd.DataFrame:
    teams = sorted(set(prior["team"]).union(set(cur["team"])))
    p = prior.set_index("team").reindex(teams).fillna(0.0)
    c = cur.set_index("team").reindex(teams).fillna(0.0)
    n = cur_counts.reindex(teams).fillna(0.0).to_numpy()
    # let current season influence faster
    w_cur = np.clip(n / (n + 60.0), 0.0, 0.90)
    w_prior = 1.0 - w_cur
    out = pd.DataFrame({
        "team": teams,
        "off_rating": (w_prior * p["off_rating"].to_numpy()) + (w_cur * c["off_rating"].to_numpy()),
        "def_rating": (w_prior * p["def_rating"].to_numpy()) + (w_cur * c["def_rating"].to_numpy()),
    })
    out["off_rating"] = out["off_rating"].clip(-0.45, 0.45)
    out["def_rating"] = out["def_rating"].clip(-0.45, 0.45)
    return out

# -------------------------
# Injuries
# -------------------------

OUT_STATUSES = {"out","injured reserve","ir","dnp","inactive","doubtful"}
LIMITED_STATUSES = {"questionable","limited","lp","day-to-day","probable"}

def build_health_index(inj: Optional[pd.DataFrame]) -> pd.Series:
    if inj is None or inj.empty:
        return pd.Series(dtype=float)
    x = inj.copy(); x.columns = [c.lower() for c in x.columns]
    team_col = "team" if "team" in x.columns else ("team_abbr" if "team_abbr" in x.columns else None)
    if team_col is None:
        return pd.Series(dtype=float)
    out = {}
    for t, sub in x.groupby(x[team_col].astype(str).str.upper()):
        status_col = next((c for c in ["status","designation","game_status","practice_status"] if c in sub.columns), None)
        pos_col = next((c for c in ["position","pos","role"] if c in sub.columns), None)
        penalty = 0.0
        if status_col is not None:
            stats = sub[status_col].astype(str).str.lower()
            outs = stats.apply(lambda s: any(k in s for k in OUT_STATUSES))
            lims = stats.apply(lambda s: any(k in s for k in LIMITED_STATUSES))
            penalty += 0.01 * int(lims.sum()) + 0.03 * int(outs.sum())
            if pos_col is not None:
                pos = sub[pos_col].astype(str).str.upper()
                qb_out = ((pos.str.contains("QB")) & outs).sum()
                wr_out = ((pos.str.contains("WR|TE")) & outs).sum()
                ol_out = ((pos.str.contains("OL|LT|RT|C|G")) & outs).sum()
                db_out = ((pos.str.contains("CB|DB|S")) & outs).sum()
                penalty += 0.05*qb_out + 0.02*wr_out + 0.03*ol_out + 0.02*db_out
        out[t] = float(np.clip(1.0 - penalty, 0.70, 1.05))
    return pd.Series(out, name="health_index")

# -------------------------
# Pace modeling (stronger effect)
# -------------------------

def build_current_season_pace(history_all: pd.DataFrame, current_season: int) -> Tuple[float, pd.Series, pd.Series]:
    cur = history_all[history_all["season"] == current_season].copy()
    if cur.empty or "plays" not in cur.columns:
        return 65.0, pd.Series(dtype=float), pd.Series(dtype=float)
    off = cur.groupby(["off_team","game_id"])["plays"].sum().groupby("off_team").median()
    allowed = cur.groupby(["def_team","game_id"])["plays"].sum().groupby("def_team").median()
    league = float(np.nanmedian(cur.groupby(["off_team","game_id"])["plays"].sum().to_numpy()))
    return league, off.rename("off_pace"), allowed.rename("def_pace")

# -------------------------
# Projection
# -------------------------

@dataclass
class GameProjection:
    season: int
    week: int
    game_id: str
    kickoff_et: Optional[pd.Timestamp]
    home: str
    away: str
    home_win_prob: float
    fair_spread_home: float
    fair_total: float
    home_pts_p5: float
    home_pts_p25: float
    home_pts_p50: float
    home_pts_p75: float
    home_pts_p95: float
    away_pts_p5: float
    away_pts_p25: float
    away_pts_p50: float
    away_pts_p75: float
    away_pts_p95: float
    injury_penalty_home: float
    injury_penalty_away: float
    pace_adj: float
    hfa_estimate: float

def simulate(mu_h, mu_a, sd_h, sd_a, rho=0.10, n=50000, seed: Optional[int]=None):
    rng = np.random.default_rng(seed)
    cov = rho * sd_h * sd_a
    cov_mat = np.array([[sd_h**2, cov],[cov, sd_a**2]])
    mean = np.array([mu_h, mu_a])
    draws = rng.multivariate_normal(mean, cov_mat, size=n)
    draws = np.clip(draws, 0.0, None)
    home = draws[:,0]; away = draws[:,1]
    q = lambda a,p: float(np.quantile(a, p))
    return {
        "home_q": [q(home, p) for p in [0.05,0.25,0.50,0.75,0.95]],
        "away_q": [q(away, p) for p in [0.05,0.25,0.50,0.75,0.95]],
        "wp": float((home > away).mean())
    }

def build_projections(tables: Dict[str, pd.DataFrame], seasons: List[int], n_sims: int, seed: Optional[int]) -> pd.DataFrame:
    sched = clean_schedule(tables["schedules"])
    pbp = tables["pbp"]
    injuries = tables.get("injuries")

    games = pbp_to_game(pbp)
    current_season = max([int(s) for s in seasons])
    hist_cur = games[games["season"] == current_season]
    prior_ratings, hfa_prior = recency_weighted_prior(games, sched, current_season)
    if hist_cur.empty:
        cur_ratings = prior_ratings.copy()
        cur_counts = pd.Series(0.0, index=prior_ratings["team"])
        hfa_cur = 0.0
    else:
        cur_ratings, hfa_cur, cur_counts = ratings_from_history(hist_cur, sched, lam=3.5)
    ratings = blend_prior_current(prior_ratings, cur_ratings, cur_counts)
    hfa = float(np.clip(0.6 * hfa_prior + 0.4 * hfa_cur, -0.12, 0.14))

    league_play_med, off_pace, def_pace = build_current_season_pace(games, current_season)
    if off_pace.empty or def_pace.empty:
        league_play_med = float(np.nanmedian(games["plays"])) if "plays" in games.columns else 65.0
    rat = ratings.set_index("team")
    hl = build_health_index(injuries)

    now = pd.Timestamp.now(tz="UTC").tz_convert("US/Eastern")
    up = sched.copy()
    if "kickoff_et" in up.columns and up["kickoff_et"].notna().any():
        up["is_future"] = up["kickoff_et"] > now
    else:
        up["is_future"] = True

    out = []
    for _, r in up.iterrows():
        home = str(r.get("home","")).upper(); away = str(r.get("away","")).upper()
        if not home or not away: continue
        season = int(r.get("season", current_season))
        week = int(r.get("week", 0))
        gid = str(r.get("game_id", f"{season}_{home}_{away}")).lower()
        k = r.get("kickoff_et", pd.NaT)

        off_h = float(rat.loc[home,"off_rating"]) if home in rat.index else 0.0
        def_h = float(rat.loc[home,"def_rating"]) if home in rat.index else 0.0
        off_a = float(rat.loc[away,"off_rating"]) if away in rat.index else 0.0
        def_a = float(rat.loc[away,"def_rating"]) if away in rat.index else 0.0

        mult_h = float(hl.get(home, 1.0))
        mult_a = float(hl.get(away, 1.0))

        # efficiencies: wider band
        eff_h = np.clip((off_h - def_a) * mult_h + 0.12 * hfa, -0.50, 0.50)
        eff_a = np.clip((off_a - def_h) * mult_a, -0.50, 0.50)

        # pace expectations
        off_h_plays = float(off_pace.get(home, league_play_med)) if not off_pace.empty else league_play_med
        def_a_plays = float(def_pace.get(away, league_play_med)) if not def_pace.empty else league_play_med
        off_a_plays = float(off_pace.get(away, league_play_med)) if not off_pace.empty else league_play_med
        def_h_plays = float(def_pace.get(home, league_play_med)) if not def_pace.empty else league_play_med

        exp_plays_home = 0.5 * (off_h_plays + def_a_plays)
        exp_plays_away = 0.5 * (off_a_plays + def_h_plays)

        # stronger pace effect
        pace_factor_home = np.clip(exp_plays_home / league_play_med, 0.85, 1.15)
        pace_factor_away = np.clip(exp_plays_away / league_play_med, 0.85, 1.15)

        # points mapping: baseline 21.5, scale 34.0 per EPA spread
        base = 21.5; scale = 34.0
        mu_h0 = base + scale * eff_h
        mu_a0 = base + scale * eff_a

        # apply pace to the deviation from baseline; slightly stronger exponent
        exponent = 0.9
        mu_h = base + (mu_h0 - base) * (pace_factor_home ** exponent)
        mu_a = base + (mu_a0 - base) * (pace_factor_away ** exponent)

        # clamp to wider realistic band
        mu_h = float(np.clip(mu_h, 9.0, 44.0))
        mu_a = float(np.clip(mu_a, 9.0, 44.0))

        # smarter variance: combine baseline Poisson-ish variance with efficiency noise
        var_h = max(90.0, 1.15 * mu_h + 30.0 * (abs(eff_h)))  # floor + scale with mean and uncertainty
        var_a = max(90.0, 1.15 * mu_a + 30.0 * (abs(eff_a)))
        sd_h = float(np.sqrt(var_h))
        sd_a = float(np.sqrt(var_a))

        sim = simulate(mu_h, mu_a, sd_h, sd_a, rho=0.10, n=n_sims, seed=seed)
        hq, aq, wp = sim["home_q"], sim["away_q"], sim["wp"]

        fair_spread = float(hq[2] - aq[2])
        fair_total = float(hq[2] + aq[2])

        # looser clamps so we can see ±7 favorites and totals into mid/high 40s
        fair_spread = float(np.clip(fair_spread, -9.5, 9.5))
        fair_total = float(np.clip(fair_total, 36.0, 56.0))

        pace_adj = float(0.5 * (exp_plays_home + exp_plays_away))

        out.append({
            "season": season, "week": week, "game_id": gid, "kickoff_et": k,
            "home": home, "away": away, "is_future": bool(r.get("is_future", True)),
            "home_win_prob": wp, "fair_spread_home": fair_spread, "fair_total": fair_total,
            "home_pts_p5": hq[0], "home_pts_p25": hq[1], "home_pts_p50": hq[2], "home_pts_p75": hq[3], "home_pts_p95": hq[4],
            "away_pts_p5": aq[0], "away_pts_p25": aq[1], "away_pts_p50": aq[2], "away_pts_p75": aq[3], "away_pts_p95": aq[4],
            "injury_penalty_home": float(1.0 - mult_h), "injury_penalty_away": float(1.0 - mult_a),
            "pace_adj": pace_adj, "hfa_estimate": hfa
        })
    return pd.DataFrame(out)

# -------------------------
# Filters and output formatting (same as v5.1)
# -------------------------

CORE_COLS = [
    "season","week","game_id","kickoff_et","home","away",
    "home_win_prob","fair_spread_home","fair_total","pace_adj"
]

def apply_filters(df: pd.DataFrame, only_upcoming: bool, from_week: Optional[int], to_week: Optional[int], teams: Optional[List[str]]) -> pd.DataFrame:
    x = df.copy()
    if only_upcoming and "is_future" in x.columns:
        x = x[x["is_future"] == True]
    if from_week is not None:
        x = x[x["week"] >= int(from_week)]
    if to_week is not None:
        x = x[x["week"] <= int(to_week)]
    if teams:
        tset = {t.strip().upper() for t in teams if t.strip()}
        x = x[x["home"].isin(tset) | x["away"].isin(tset)]
    x = x.drop(columns=["is_future"], errors="ignore")
    return x

def write_output(df: pd.DataFrame, out_path: str, fmt: str):
    fmt = fmt.lower()
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "parquet":
        try:
            import pyarrow as pa, pyarrow.parquet as pq  # noqa: F401
            df.to_parquet(out_path, index=False)
        except Exception as e:
            raise RuntimeError(f"Parquet output requires pyarrow: {e}")
    elif fmt == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps({k: (None if pd.isna(v) else v) for k,v in row.items()}, default=str) + "\n")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def main():
    ap = argparse.ArgumentParser(description="FoxEdge NFL Projections — v6 real-world ranges")
    ap.add_argument("--seasons", nargs="+", type=int, required=True, help="Seasons to fetch (include prior years and current)")
    ap.add_argument("--out", type=str, default="./projections.csv", help="Output file path")
    ap.add_argument("--format", type=str, default="csv", choices=["csv","parquet","jsonl"], help="Output format")
    ap.add_argument("--n-sims", type=int, default=50000, help="Monte Carlo samples")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    only = ap.add_mutually_exclusive_group()
    only.add_argument("--only-upcoming", action="store_true", default=True, help="Keep only future games (default)")
    only.add_argument("--include-past", action="store_true", help="Include past games too")
    ap.add_argument("--from-week", type=int, default=None, help="Minimum week (inclusive)")
    ap.add_argument("--to-week", type=int, default=None, help="Maximum week (inclusive)")
    ap.add_argument("--teams", type=str, default=None, help="Comma-separated team filter (keeps games with either team)")
    ap.add_argument("--lite", action="store_true", help="Only core columns")
    args = ap.parse_args()

    tables = fetch_pbp_and_sched(args.seasons)
    df = build_projections(tables, args.seasons, n_sims=args.n_sims, seed=args.seed)

    only_upcoming = False if args.include_past else True
    only_upcoming = args.only_upcoming or only_upcoming

    teams = [t for t in args.teams.split(",")] if args.teams else None
    df = apply_filters(df, only_upcoming=only_upcoming, from_week=args.from_week, to_week=args.to_week, teams=teams)

    if args.lite:
        cols = [c for c in CORE_COLS if c in df.columns]
        df = df[cols]

    write_output(df, args.out, args.format)
    log(f"Wrote projections: {args.out} ({len(df)} games) [fmt={args.format}]")

if __name__ == "__main__":
    main()
