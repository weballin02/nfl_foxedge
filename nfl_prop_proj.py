#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import dirichlet
import unicodedata, re, difflib

# nfl_data_py
from nfl_data_py import (
    import_pbp_data, import_schedules, import_sc_lines,
    import_weekly_data, import_weekly_pfr, import_weekly_rosters,
    import_seasonal_rosters, import_players, import_snap_counts
)

st.set_page_config(page_title="FoxEdge NFL Prop Proj (Fixed Odds Schema)", page_icon="🦊", layout="wide")

# ---------------- Normalization ----------------

SUFFIX_RE = re.compile(r"\b(jr\.?|sr\.?|ii|iii|iv|v)\b", re.IGNORECASE)
UNICODE_MINUS = "\u2212"

TEAM_ALIASES = {
    "san francisco 49ers": "SF", "49ers": "SF", "sf 49ers": "SF", "sf": "SF",
    "dallas cowboys": "DAL", "cowboys": "DAL", "dal": "DAL",
    "philadelphia eagles": "PHI", "eagles": "PHI", "phi": "PHI",
    "kansas city chiefs": "KC", "chiefs": "KC", "kc": "KC", "kansas city": "KC",
    "buffalo bills": "BUF", "bills": "BUF", "buf": "BUF",
    "miami dolphins": "MIA", "dolphins": "MIA", "mia": "MIA",
    "new york jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    "new york giants": "NYG", "giants": "NYG", "nyg": "NYG",
    "los angeles rams": "LA", "rams": "LA", "lar": "LA", "la rams": "LA",
    "los angeles chargers": "LAC", "chargers": "LAC", "lac": "LAC",
    "las vegas raiders": "LV", "raiders": "LV", "lv": "LV",
    "seattle seahawks": "SEA", "seahawks": "SEA", "sea": "SEA",
    "detroit lions": "DET", "lions": "DET", "det": "DET",
    "green bay packers": "GB", "packers": "GB", "gb": "GB",
    "minnesota vikings": "MIN", "vikings": "MIN", "min": "MIN",
    "chicago bears": "CHI", "bears": "CHI", "chi": "CHI",
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "bucs": "TB", "tb": "TB",
    "new orleans saints": "NO", "saints": "NO", "no": "NO", "new orleans": "NO",
    "atlanta falcons": "ATL", "falcons": "ATL", "atl": "ATL",
    "carolina panthers": "CAR", "panthers": "CAR", "car": "CAR",
    "tennessee titans": "TEN", "titans": "TEN", "ten": "TEN",
    "indianapolis colts": "IND", "colts": "IND", "ind": "IND",
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jax": "JAX",
    "houston texans": "HOU", "texans": "HOU", "hou": "HOU",
    "baltimore ravens": "BAL", "ravens": "BAL", "bal": "BAL",
    "pittsburgh steelers": "PIT", "steelers": "PIT", "pit": "PIT",
    "cleveland browns": "CLE", "browns": "CLE", "cle": "CLE",
    "cincinnati bengals": "CIN", "bengals": "CIN", "cin": "CIN",
    "new england patriots": "NE", "patriots": "NE", "ne": "NE",
    "washington commanders": "WAS", "commanders": "WAS", "was": "WAS", "wsh": "WAS",
    "arizona cardinals": "ARI", "cardinals": "ARI", "ari": "ARI",
    "denver broncos": "DEN", "broncos": "DEN", "den": "DEN",
}

# Market keys (exactly as you said your file will contain)
PROP_MAP = {
    "player_pass_attempts": "QB Pass Attempts",
    "player_pass_yds": "QB Pass Yards",
    "player_reception_yds": "WR/TE Receiving Yards",
    "player_receptions": "WR/TE Receptions",
    "player_rush_attempts": "RB Rush Attempts",
    "player_rush_yds": "RB Rush Yards",
}

def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def clean_player_name(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = ascii_fold(s.lower())
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = SUFFIX_RE.sub("", s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonicalize_clean_name(s: str) -> str:
    # add light aliasing if you need specific nicknames later
    return clean_player_name(s)

def clean_team(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    key = ascii_fold(s.lower()).strip()
    return TEAM_ALIASES.get(key, s.upper())

def to_float_clean(x):
    if x is None: return np.nan
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null"): return np.nan
    if s in ("even", "ev", "evs"): return 100.0
    s = s.replace(UNICODE_MINUS, "-")
    s = re.sub(r"^[ou]\s*", "", s)
    s = s.replace(",", ".")
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def american_to_prob(odds):
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o == 0 or not np.isfinite(o): return np.nan
    return (-o)/(100 - o) if o < 0 else 100/(100 + o)

def fair_from_prob(p):
    if p <= 0 or p >= 1 or not np.isfinite(p): return np.inf
    return -100 * p / (1 - p)

def safe_div(n, d):
    try:
        return n / d if d not in (0, 0.0, None) else np.nan
    except Exception:
        return np.nan

import math

def token_ratio(a: str, b: str) -> float:
    """Simple token set overlap score in [0,1]."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def build_name_aliases(clean_name: str) -> list[str]:
    """
    Generate alias keys for matching:
    - full clean name
    - first initial + remaining tokens (covers 'A St Brown' vs 'Amon Ra St Brown')
    - collapse multiple given names to first initial
    """
    if not clean_name:
        return []
    toks = clean_name.split()
    aliases = {clean_name}
    if len(toks) >= 2:
        first_initial = toks[0][0]
        rest = " ".join(toks[1:])
        aliases.add(f"{first_initial} {rest}".strip())
        # if two given names, also initial + second given + rest
        if len(toks) >= 3:
            aliases.add(f"{first_initial} {' '.join(toks[2:])}".strip())
    return list(aliases)

def best_similarity(a: str, b: str) -> float:
    """Combine difflib and token overlap to be tolerant of initials."""
    from difflib import SequenceMatcher
    d = SequenceMatcher(None, a, b).ratio()
    t = token_ratio(a, b)
    return max(d, t)

def last_name_unique_match(target_norm: str, side_map: dict):
    toks = target_norm.split()
    if not toks:
        return None
    last = toks[-1]
    # collect candidates that contain the last-name token
    hits = [k for k in side_map.keys() if last in k.split()]
    if len(hits) == 1:
        k = hits[0]
        return k, side_map[k], "last_name_unique"
    return None


# ---------------- Data loaders ----------------

@st.cache_data(show_spinner=False)
def load_pbp(seasons: list[int]) -> pd.DataFrame:
    pbp = import_pbp_data(seasons)
    for b in ["pass","rush","qb_hit","sack","no_huddle","complete_pass","touchdown"]:
        if b in pbp.columns:
            pbp[b] = pd.to_numeric(pbp[b], errors="coerce").fillna(0.0)
    if "scramble" in pbp.columns:
        pbp["scramble"] = pd.to_numeric(pbp["scramble"], errors="coerce")
    elif "qb_scramble" in pbp.columns:
        pbp["scramble"] = pd.to_numeric(pbp["qb_scramble"], errors="coerce")
    else:
        pbp["scramble"] = np.nan
    if "air_yards" not in pbp.columns: pbp["air_yards"] = np.nan
    if "yardline_100" not in pbp.columns: pbp["yardline_100"] = np.nan
    pbp["rz20"] = (pbp["yardline_100"] <= 20).astype(int)
    for c in ["posteam","defteam","receiver_player_name","rusher_player_name","passer_player_name"]:
        if c not in pbp.columns: pbp[c] = np.nan
    pbp["receiver"] = pbp.get("receiver", pbp["receiver_player_name"])
    pbp["rusher"]   = pbp.get("rusher",   pbp["rusher_player_name"])
    pbp["passer"]   = pbp.get("passer",   pbp["passer_player_name"])
    qtr = pbp.get("qtr", pd.Series(np.nan, index=pbp.index))
    score_diff = pbp.get("score_differential", pbp.get("score_differential_post", pd.Series(np.nan, index=pbp.index)))
    two_min = pbp.get("two_minute_warning", pd.Series(False, index=pbp.index)).fillna(False)
    pbp["neutral_state"] = (qtr.between(1,3, inclusive="both").fillna(False) & score_diff.abs().le(7).fillna(False) & (~two_min)).astype(int)
    return pbp

@st.cache_data(show_spinner=False)
def load_weekly(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for y in seasons:
        df = None
        try:
            df = import_weekly_data([y])
        except Exception:
            pass
        if df is None or df.empty:
            try:
                pfr = import_weekly_pfr([y])
                if pfr is not None and not pfr.empty:
                    df = pfr
            except Exception:
                pass
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_aux(seasons: list[int]):
    sched = import_schedules(seasons)
    try: lines = import_sc_lines(seasons)
    except Exception: lines = pd.DataFrame()
    try: wrost = import_weekly_rosters(seasons)
    except Exception: wrost = pd.DataFrame()
    try: srost = import_seasonal_rosters(seasons)
    except Exception: srost = pd.DataFrame()
    try: players = import_players()
    except Exception: players = pd.DataFrame()
    try: snaps = import_snap_counts(seasons)
    except Exception: snaps = pd.DataFrame()
    return sched, lines, wrost, srost, players, snaps

def build_roster_map(weekly_rosters: pd.DataFrame, seasonal_rosters: pd.DataFrame,
                     seasons: list[int], week_range: tuple[int, int]) -> dict:
    mapping = {}
    def add(df, name_col, team_col):
        for _, r in df.iterrows():
            nm = canonicalize_clean_name(str(r[name_col]))
            tm = clean_team(str(r[team_col]))
            if nm: mapping[nm] = tm
    if weekly_rosters is not None and not weekly_rosters.empty:
        lc = {c.lower(): c for c in weekly_rosters.columns}
        ncol = lc.get("player_name") or lc.get("full_name") or lc.get("name")
        tcol = lc.get("team") or lc.get("team_abbr") or lc.get("team_abbreviation")
        if ncol and tcol:
            df = weekly_rosters.copy()
            if "season" in df.columns: df = df[df["season"].isin(seasons)]
            if "week" in df.columns and week_range:
                df = df[(df["week"]>=week_range[0]) & (df["week"]<=week_range[1])]
                df = df.sort_values([ncol,"week"]).drop_duplicates(subset=[ncol], keep="last")
            add(df, ncol, tcol)
    if not mapping and seasonal_rosters is not None and not seasonal_rosters.empty:
        lc = {c.lower(): c for c in seasonal_rosters.columns}
        ncol = lc.get("player_name") or lc.get("full_name") or lc.get("name")
        tcol = lc.get("team") or lc.get("team_abbr") or lc.get("team_abbreviation")
        if ncol and tcol:
            df = seasonal_rosters.copy()
            if "season" in df.columns: df = df[df["season"].isin(seasons)]
            add(df, ncol, tcol)
    return mapping

def build_alias_map(players_df: pd.DataFrame, wrost: pd.DataFrame, srost: pd.DataFrame) -> dict:
    alias = {}
    def put(name, team=""):
        cn = canonicalize_clean_name(name)
        if not cn: return
        if cn not in alias: alias[cn] = {"display": name, "team": clean_team(team) if team else ""}
        else:
            if not alias[cn]["team"] and team: alias[cn]["team"] = clean_team(team)
            if len(alias[cn]["display"]) < len(name): alias[cn]["display"] = name
    for df in [wrost, srost]:
        if df is None or df.empty: continue
        lc = {c.lower(): c for c in df.columns}
        ncol = lc.get("player_name") or lc.get("full_name") or lc.get("name")
        tcol = lc.get("team") or lc.get("team_abbr") or lc.get("team_abbreviation")
        if ncol and tcol:
            for _, r in df.iterrows(): put(str(r[ncol]), str(r[tcol]))
    if players_df is not None and not players_df.empty:
        lc = {c.lower(): c for c in players_df.columns}
        disp = lc.get("display_name") or lc.get("full_name") or lc.get("name")
        team = lc.get("recent_team") or lc.get("team") or lc.get("team_abbr")
        if disp:
            for _, r in players_df.iterrows(): put(str(r[disp]), str(r[team]) if team in players_df.columns else "")
    return alias

# ---------------- Features & simulation ----------------

def team_context_features(pbp: pd.DataFrame, team: str, window_games: int = 12) -> dict:
    df = pbp[(pbp["posteam"] == team)]
    if "game_id" in df.columns:
        gids = df["game_id"].dropna().unique().tolist()[-window_games:]
        df = df[df["game_id"].isin(gids)]
    if df.empty or "game_id" not in df.columns:
        return {"plays_per_game": 60.0, "seconds_per_play": 28.0, "neutral_pass_rate": 0.58}
    g = df.groupby(["game_id","posteam"]).size().reset_index(name="plays")
    plays_per_game = g["plays"].mean()
    sec_per_play = 3600.0 / g["plays"].mean()
    neutral = df[df["neutral_state"] == 1]
    if neutral.empty:
        neutral_pass = safe_div(df["pass"].sum(), (df["pass"] + df["rush"]).sum())
    else:
        neutral_pass = safe_div(neutral["pass"].sum(), (neutral["pass"] + neutral["rush"]).sum())
    return {"plays_per_game": float(plays_per_game), "seconds_per_play": float(sec_per_play), "neutral_pass_rate": float(neutral_pass)}

def opponent_defense_features(pbp: pd.DataFrame, opp: str, window_games: int = 8) -> dict:
    dfp = pbp[pbp["defteam"] == opp]
    if "game_id" in dfp.columns:
        gids = dfp["game_id"].dropna().unique().tolist()[-window_games:]
        dfp = dfp[dfp["game_id"].isin(gids)]
    p = dfp[dfp.get("pass", 0) == 1.0][["sack","qb_hit"]].copy()
    dropbacks = len(p) if p is not None else 0
    sacks = p["sack"].sum() if p is not None else 0
    hits  = p["qb_hit"].sum() if p is not None else 0
    pressure_rate = safe_div(sacks + hits, dropbacks)
    sack_rate = safe_div(sacks, dropbacks)
    pa = dfp[(dfp.get("pass",0) == 1.0)]
    pr = dfp[(dfp.get("rush",0) == 1.0)]
    exp_pass = safe_div((pa["air_yards"] >= 15).sum(), len(pa)) if len(pa) else np.nan
    exp_rush = safe_div((pr.get("yards_gained",0) >= 12).sum(), len(pr)) if len(pr) else np.nan
    cp = pa[pa.get("complete_pass",0) == 1.0].copy()
    if not cp.empty:
        cp["yac_proxy"] = cp.get("yards_gained",0) - np.clip(cp["air_yards"].fillna(0), 0, None)
        yac_allowed = float(cp["yac_proxy"].mean())
    else:
        yac_allowed = np.nan
    rr = pr.copy()
    rr["success_off"] = (rr.get("epa",0) > 0).astype(int)
    run_succ = float(rr["success_off"].mean()) if len(rr) else np.nan
    return {
        "pressure_rate": float(pressure_rate) if pressure_rate==pressure_rate else 0.22,
        "sack_rate": float(sack_rate) if sack_rate==sack_rate else 0.07,
        "explosive_pass_allowed": float(exp_pass) if exp_pass==exp_pass else 0.12,
        "explosive_rush_allowed": float(exp_rush) if exp_rush==exp_rush else 0.09,
        "yac_allowed": float(yac_allowed) if yac_allowed==yac_allowed else 3.5,
        "run_success_allowed": float(run_succ) if run_succ==run_succ else 0.42,
    }

def player_usage_priors(pbp: pd.DataFrame, team: str, window_games: int = 8):
    dft = pbp[pbp["posteam"] == team]
    if "game_id" in dft.columns:
        gids = dft["game_id"].dropna().unique().tolist()
        dft = dft[dft["game_id"].isin(gids[-window_games:])]
    rec = dft[dft.get("pass",0) == 1.0].dropna(subset=["receiver"])
    tgt_counts = rec.groupby("receiver").size().reset_index(name="targets").sort_values("targets", ascending=False)
    tgt_total = max(tgt_counts["targets"].sum(), 1)
    tgt_counts["tgt_share"] = tgt_counts["targets"] / tgt_total
    ru = dft[dft.get("rush",0) == 1.0].dropna(subset=["rusher"])
    car_counts = ru.groupby("rusher").size().reset_index(name="carries").sort_values("carries", ascending=False)
    car_total = max(car_counts["carries"].sum(), 1)
    car_counts["car_share"] = car_counts["carries"] / car_total
    wrte = tgt_counts.rename(columns={"receiver": "player"}).head(8)
    rbs  = car_counts.rename(columns={"rusher": "player"}).head(4)
    return wrte, rbs

def qb_prior(pbp: pd.DataFrame, team: str, window_games: int = 8):
    dft = pbp[pbp["posteam"] == team]
    if "game_id" in dft.columns:
        gids = dft["game_id"].dropna().unique().tolist()
        dft = dft[dft["game_id"].isin(gids[-window_games:])]
    atts = dft[dft.get("pass",0) == 1.0].groupby(["game_id","posteam"]).size().reset_index(name="atts")
    mean_att = atts["atts"].mean() if not atts.empty else 32.0
    var_att  = atts["atts"].var(ddof=1) if len(atts) > 1 else mean_att * 1.5
    passes = dft[dft.get("pass",0) == 1.0].copy()
    if not passes.empty:
        comp = passes[passes.get("complete_pass",0) == 1.0].copy()
        ypa = comp.get("yards_gained", pd.Series([6.8])).mean() if not comp.empty else 6.8
        ay  = passes["air_yards"].dropna().mean() if passes["air_yards"].notna().any() else 7.4
    else:
        ypa, ay = 6.8, 7.4
    return {"mean_att": float(mean_att), "var_att": float(var_att), "ypa": float(ypa), "adot": float(ay)}

def script_weights_from_market(spread: float, total: float) -> dict:
    s = np.clip(spread, -14, 14) if pd.notna(spread) else 0.0
    lead_weight = 0.5 + (-s/28.0)
    trail_weight = 1.0 - lead_weight
    t = total if pd.notna(total) else 44.0
    plays_scale = np.clip((t - 40.0)/20.0 + 1.0, 0.85, 1.15)
    return {"lead": float(lead_weight), "trail": float(trail_weight), "plays_scale": float(plays_scale)}

def simulate_game_props(pbp, lines, home_team, away_team, n_sims=10000, random_seed=1337):
    rng = np.random.default_rng(random_seed)

    spread = np.nan; total = np.nan
    if lines is not None and not lines.empty:
        for c in ["team","opponent","home_team","away_team","favorite","underdog","spread","total_line"]:
            if c not in lines.columns:
                lines[c] = np.nan
        lr = lines[((lines["team"]==home_team) & (lines["opponent"]==away_team)) | ((lines["team"]==away_team) & (lines["opponent"]==home_team))]
        if not lr.empty:
            r = lr.iloc[0]; spread = r.get("spread", np.nan); total = r.get("total_line", np.nan)

    ctx_home = team_context_features(pbp, home_team)
    ctx_away = team_context_features(pbp, away_team)
    opp_home = opponent_defense_features(pbp, away_team)
    opp_away = opponent_defense_features(pbp, home_team)
    qb_home  = qb_prior(pbp, home_team)
    qb_away  = qb_prior(pbp, away_team)
    wrte_home, rb_home = player_usage_priors(pbp, home_team)
    wrte_away, rb_away = player_usage_priors(pbp, away_team)

    def alphas_from_shares(df, share_col, k):
        if df.empty: return np.array([1.0, 1.0, 1.0])
        shares = df[share_col].to_numpy()
        shares = shares / shares.sum() if shares.sum() > 0 else np.ones_like(shares)/len(shares)
        return np.clip(shares * k, 0.5, None)

    alpha_wr_home = alphas_from_shares(wrte_home, "tgt_share", k=60.0)
    alpha_wr_away = alphas_from_shares(wrte_away, "tgt_share", k=60.0)
    alpha_rb_home = alphas_from_shares(rb_home,  "car_share", k=40.0)
    alpha_rb_away = alphas_from_shares(rb_away,  "car_share", k=40.0)

    sw = script_weights_from_market(spread, total)

    def simulate_team(context, qbctx, oppdef):
        plays_mu = context["plays_per_game"] * sw["plays_scale"]
        plays = rng.normal(loc=plays_mu, scale=max(2.0, 0.06 * plays_mu), size=n_sims).clip(min=45, max=85)
        base_pr = context["neutral_pass_rate"]
        pr_lead = np.clip(base_pr - 0.05, 0.35, 0.70)
        pr_trail= np.clip(base_pr + 0.07, 0.45, 0.80)
        pass_rate = sw["lead"] * pr_lead + sw["trail"] * pr_trail
        pass_rate_draws = rng.beta(a=pass_rate*80, b=(1-pass_rate)*80, size=n_sims)
        attempts = rng.binomial(n=np.round(plays).astype(int), p=pass_rate_draws)
        ypa_base = qbctx["ypa"]
        ypa_adj  = ypa_base * (1 - 0.5*(oppdef["pressure_rate"] - 0.22))
        ypa_draws= rng.normal(loc=ypa_adj, scale=max(0.4, 0.15*ypa_adj), size=n_sims).clip(3.8, 9.5)
        pass_yards = attempts * ypa_draws
        return plays, attempts, pass_yards

    plays_h, atts_h, pass_yds_h = simulate_team(ctx_home, qb_home, opp_home)
    plays_a, atts_a, pass_yds_a = simulate_team(ctx_away, qb_away, opp_away)

    def split_dirichlet(total_counts, alpha_vec):
        if len(alpha_vec) == 0: return np.zeros((n_sims, 0))
        shares = dirichlet(alpha_vec).rvs(n_sims, random_state=rng)
        return (shares * total_counts[:, None]).astype(int)

    t_home = split_dirichlet(atts_h, alpha_wr_home)
    t_away = split_dirichlet(atts_a, alpha_wr_away)

    rush_home = np.clip(plays_h - atts_h, 10, None)
    rush_away = np.clip(plays_a - atts_a, 10, None)
    car_home = split_dirichlet(rush_home, alpha_rb_home)
    car_away = split_dirichlet(rush_away, alpha_rb_away)

    def rec_outcomes(targ_matrix, yac_allowed, explosive_pass_allowed, pressure_rate):
        n_players = targ_matrix.shape[1]
        recs = np.zeros_like(targ_matrix)
        yds  = np.zeros_like(targ_matrix, dtype=float)
        if n_players == 0: return recs, yds
        catch_p = np.clip(0.63 - 0.1*(pressure_rate - 0.22), 0.50, 0.72)
        yac = max(yac_allowed, 2.5)
        deep_p = np.clip(explosive_pass_allowed + 0.02, 0.06, 0.20)
        rngl = np.random.default_rng(12345)
        for j in range(n_players):
            t = targ_matrix[:, j].astype(int)
            cp_draw = rngl.beta(a=catch_p*60, b=(1-catch_p)*60, size=n_sims)
            recs[:, j] = rngl.binomial(n=t, p=cp_draw)
            short_mu = 4.0 + yac
            deep_mu  = 18.0 + yac
            mix = rngl.binomial(1, deep_p, size=n_sims)
            per_target = np.where(
                mix==1,
                rngl.lognormal(mean=np.log(deep_mu),  sigma=0.55, size=n_sims),
                rngl.lognormal(mean=np.log(short_mu), sigma=0.45, size=n_sims)
            )
            yds[:, j] = per_target * t
        return recs, yds

    recs_h, recyds_h = rec_outcomes(t_home, opp_home["yac_allowed"], opp_home["explosive_pass_allowed"], opp_home["pressure_rate"])
    recs_a, recyds_a = rec_outcomes(t_away, opp_away["yac_allowed"], opp_away["explosive_pass_allowed"], opp_away["pressure_rate"])

    def rush_outcomes(carry_matrix, oppdef):
        n_players = carry_matrix.shape[1]
        yards = np.zeros_like(carry_matrix, dtype=float)
        if n_players == 0: return yards
        ypc_base = 4.3 + 1.2*(0.45 - oppdef["run_success_allowed"])
        ypc_base = np.clip(ypc_base, 3.6, 5.2)
        rngl = np.random.default_rng(54321)
        for j in range(n_players):
            c = carry_matrix[:, j].astype(int)
            ypc_draw = rngl.lognormal(mean=np.log(ypc_base), sigma=0.25, size=n_sims)
            yards[:, j] = ypc_draw * c
        return yards

    rushyds_h = rush_outcomes(car_home, opp_home)
    rushyds_a = rush_outcomes(car_away, opp_away)

    qb_home_name = pbp.loc[pbp["posteam"]==home_team, "passer"].dropna().value_counts().idxmax() if (pbp["posteam"]==home_team).any() and pbp["passer"].notna().any() else f"{home_team}_QB"
    qb_away_name = pbp.loc[pbp["posteam"]==away_team, "passer"].dropna().value_counts().idxmax() if (pbp["posteam"]==away_team).any() and pbp["passer"].notna().any() else f"{away_team}_QB"

    return {
        "home": {"qb_atts": atts_h, "qb_yds": pass_yds_h, "wr_recs": recs_h, "wr_yds": recyds_h, "rb_carr": car_home, "rb_yds": rushyds_h,
                 "wr_names": wrte_home["player"].tolist(), "rb_names": rb_home["player"].tolist(), "qb_name": qb_home_name},
        "away": {"qb_atts": atts_a, "qb_yds": pass_yds_a, "wr_recs": recs_a, "wr_yds": recyds_a, "rb_carr": car_away, "rb_yds": rushyds_a,
                 "wr_names": wrte_away["player"].tolist(), "rb_names": rb_away["player"].tolist(), "qb_name": qb_away_name},
    }

def price_line(samples: np.ndarray, line: float, market_odds_over=-110, market_odds_under=-110):
    if samples is None or len(samples)==0 or not np.isfinite(line):
        return {"p_over": np.nan, "p_under": np.nan, "fair_over": np.nan, "fair_under": np.nan, "edge_over": np.nan, "edge_under": np.nan,
                "p50": np.nan, "mean": np.nan}
    p_over = float(np.mean(samples >= line))
    p_under = 1.0 - p_over
    fair_over = fair_from_prob(p_over)
    fair_under = fair_from_prob(p_under)
    ip_over = american_to_prob(market_odds_over)
    ip_under = american_to_prob(market_odds_under)
    edge_over = (p_over - ip_over) if pd.notna(ip_over) else np.nan
    edge_under = (p_under - ip_under) if pd.notna(ip_under) else np.nan
    return {"p_over": p_over, "p_under": p_under, "fair_over": fair_over, "fair_under": fair_under,
            "edge_over": edge_over, "edge_under": edge_under, "p50": float(np.median(samples)), "mean": float(np.mean(samples))}

# ---------------- Fixed-schema odds parser ----------------

REQUIRED_COLS = ["game_id","commence_time","in_play","bookmaker","last_update","home_team","away_team","market","label","description","price","point"]

def parse_fixed_odds(df: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        raise KeyError(f"Odds CSV missing required columns: {missing}")

    # filter to selected matchup (home/away may be swapped in feed; keep both orders)
    df["_home_norm"] = df[cols["home_team"]].astype(str).apply(clean_team)
    df["_away_norm"] = df[cols["away_team"]].astype(str).apply(clean_team)
    mask = ((df["_home_norm"].eq(home) & df["_away_norm"].eq(away)) |
            (df["_home_norm"].eq(away) & df["_away_norm"].eq(home)))
    if mask.any():
        df = df[mask].copy()

    # normalize fields
    df["player"] = df[cols["description"]].astype(str)
    df["player_norm"] = df["player"].apply(canonicalize_clean_name)
    df["prop_norm"] = df[cols["market"]].astype(str).str.strip().map(lambda s: PROP_MAP.get(s, s))
    df["label_norm"] = df[cols["label"]].astype(str).str.strip().str.title()
    df["line"] = df[cols["point"]].apply(to_float_clean)
    df["price"] = df[cols["price"]].apply(to_float_clean)
    df["book"] = df[cols["bookmaker"]].astype(str)

    # pivot Over/Under into a single row
    key = ["book","player","player_norm","prop_norm","line"]
    over  = df[df["label_norm"]=="Over"].groupby(key, as_index=False).agg({"price":"last"}).rename(columns={"price":"over_odds"})
    under = df[df["label_norm"]=="Under"].groupby(key, as_index=False).agg({"price":"last"}).rename(columns={"price":"under_odds"})
    merged = pd.merge(over, under, on=key, how="outer")

    # keep only lines we can price
    merged = merged[merged["line"].notna()].reset_index(drop=True)
    return merged

# ---------------- UI ----------------

st.sidebar.title("FoxEdge NFL Prop Proj — Fixed Odds Schema")
years = st.sidebar.multiselect("Seasons", list(range(1999, 2026)), default=[2023, 2024])
weeks = st.sidebar.slider("Weeks", 1, 22, (1, 18), 1)
n_sims = st.sidebar.number_input("Simulations", min_value=2000, max_value=50000, value=10000, step=1000)
fuzzy_cutoff = st.sidebar.slider("Name match threshold", 0.60, 0.99, 0.75, 0.01)
btn = st.sidebar.button("Fetch / Refresh", type="primary")

for k in ["pbp","sched","lines","wrost","srost","players","roster_map","alias_map"]:
    if k not in st.session_state: st.session_state[k] = None

if btn or st.session_state.get("pbp") is None:
    with st.status("Loading data...", expanded=False):
        _pbp = load_pbp(years)
        if "week" in _pbp.columns:
            _pbp = _pbp[(_pbp["week"]>=weeks[0]) & (_pbp["week"]<=weeks[1])]
        st.session_state["pbp"] = _pbp
        sched, lines, wrost, srost, players, snaps = load_aux(years)
        st.session_state["sched"] = sched
        st.session_state["lines"] = lines
        st.session_state["wrost"] = wrost
        st.session_state["srost"] = srost
        st.session_state["players"] = players
        st.session_state["roster_map"] = build_roster_map(wrost, srost, years, (weeks[0], weeks[1]))
        st.session_state["alias_map"]  = build_alias_map(players, wrost, srost)

pbp   = st.session_state["pbp"]
sched = st.session_state["sched"]
lines = st.session_state["lines"]
roster_map = st.session_state.get("roster_map", {}) or {}
alias_map  = st.session_state.get("alias_map", {}) or {}

if pbp is None or pbp.empty:
    st.stop()

# Game selector
if sched is not None and not sched.empty and {"home_team","away_team"}.issubset(sched.columns):
    sched_view = sched[(sched.get("week", pd.Series(0, index=sched.index))>=weeks[0]) & (sched.get("week", pd.Series(99, index=sched.index))<=weeks[1])]
    game_opts = [f"{row['away_team']} @ {row['home_team']} (W{int(row['week']) if 'week' in sched.columns else '?'})" for _, row in sched_view.iterrows()]
    if len(game_opts)==0:
        st.warning("No games in schedule for selected filters."); st.stop()
    sel_idx = st.selectbox("Select game", options=list(range(len(game_opts))),
                           format_func=lambda i: game_opts[i] if i is not None and i < len(game_opts) else "",
                           index=0)
    sel_row = sched_view.iloc[sel_idx]
    home = sel_row["home_team"]; away = sel_row["away_team"]
else:
    teams = sorted(pbp["posteam"].dropna().unique()[:2])
    home, away = teams[0], teams[1] if len(teams)>1 else teams[0]

st.subheader(f"Matchup: {away} @ {home}")
st.caption("Using spread/total if available; otherwise neutral script priors.")

with st.spinner("Simulating game..."):
    dists = simulate_game_props(pbp, lines, home, away, n_sims=n_sims)

# ---------- Upload odds CSV (fixed schema) ----------
st.markdown("## Upload odds CSV (fixed schema)")
uploaded = st.file_uploader("Upload CSV with columns: game_id, commence_time, in_play, bookmaker, last_update, home_team, away_team, market, label, description, price, point", type=["csv"])

if uploaded is None:
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception:
    uploaded.seek(0); raw = pd.read_csv(uploaded, encoding="latin-1")

try:
    odds = parse_fixed_odds(raw, home, away)
except Exception as e:
    st.error(str(e))
    st.dataframe(raw.head(25), use_container_width=True)
    st.stop()

# attach team hints via roster/alias
odds["roster_team"] = odds["player_norm"].map(roster_map).fillna("")
odds["alias_team"]  = odds["player_norm"].map(lambda k: alias_map.get(k, {}).get("team",""))
odds["team"] = np.where(odds["roster_team"].isin([home, away]) & (odds["roster_team"]!=""),
                        odds["roster_team"],
                        np.where(odds["alias_team"].isin([home, away]) & (odds["alias_team"]!=""),
                                 odds["alias_team"], ""))

st.markdown("### Parsed and normalized odds")
st.dataframe(odds.head(40), use_container_width=True)

# Build normalized sim lookup maps
def nm_map(names): return {canonicalize_clean_name(n): i for i, n in enumerate(names)}
home_wr_map = nm_map(dists["home"]["wr_names"])
away_wr_map = nm_map(dists["away"]["wr_names"])
home_rb_map = nm_map(dists["home"]["rb_names"])
away_rb_map = nm_map(dists["away"]["rb_names"])
home_qb = canonicalize_clean_name(dists["home"]["qb_name"])
away_qb = canonicalize_clean_name(dists["away"]["qb_name"])

def nm_map_with_aliases(names):
    base = {}
    for i, n in enumerate(names):
        cn = canonicalize_clean_name(n)
        for alias in build_name_aliases(cn):
            base[alias] = i
    return base

home_wr_map = nm_map_with_aliases(dists["home"]["wr_names"])
away_wr_map = nm_map_with_aliases(dists["away"]["wr_names"])
home_rb_map = nm_map_with_aliases(dists["home"]["rb_names"])
away_rb_map = nm_map_with_aliases(dists["away"]["rb_names"])

home_qb = canonicalize_clean_name(dists["home"]["qb_name"])
away_qb = canonicalize_clean_name(dists["away"]["qb_name"])


def fuzzy_index(name_map: dict, target_norm: str, cutoff: float):
    # exact first
    if target_norm in name_map:
        return target_norm, name_map[target_norm], "exact"
    # approximate
    best_key, best_score = None, -math.inf
    for k in name_map.keys():
        s = best_similarity(target_norm, k)
        if s > best_score:
            best_key, best_score = k, s
    if best_key is not None and best_score >= cutoff:
        return best_key, name_map[best_key], f"fuzzy({best_score:.2f})"
    return None, None, f"no_match({best_score:.2f})"


def samples_for(row):
    prop = row["prop_norm"]
    pname_norm = row["player_norm"]
    team_hint = row.get("team","") or row.get("roster_team","") or row.get("alias_team","")

    # QB
    if prop == "QB Pass Attempts":
        if pname_norm == home_qb: return dists["home"]["qb_atts"], "home", dists["home"]["qb_name"], "exact qb"
        if pname_norm == away_qb: return dists["away"]["qb_atts"], "away", dists["away"]["qb_name"], "exact qb"
        h_sim = difflib.SequenceMatcher(None, pname_norm, home_qb).ratio()
        a_sim = difflib.SequenceMatcher(None, pname_norm, away_qb).ratio()
        if max(h_sim, a_sim) >= fuzzy_cutoff:
            side = "home" if h_sim >= a_sim else "away"
            return dists[side]["qb_atts"], side, dists[side]["qb_name"], "fuzzy qb"
        side = "home" if team_hint == home else ("away" if team_hint == away else "")
        if side:
            return dists[side]["qb_atts"], side, dists[side]["qb_name"], "team qb"
        return None, "", "", "no_match"

    if prop == "QB Pass Yards":
        if pname_norm == home_qb: return dists["home"]["qb_yds"], "home", dists["home"]["qb_name"], "exact qb"
        if pname_norm == away_qb: return dists["away"]["qb_yds"], "away", dists["away"]["qb_name"], "exact qb"
        h_sim = difflib.SequenceMatcher(None, pname_norm, home_qb).ratio()
        a_sim = difflib.SequenceMatcher(None, pname_norm, away_qb).ratio()
        if max(h_sim, a_sim) >= fuzzy_cutoff:
            side = "home" if h_sim >= a_sim else "away"
            return dists[side]["qb_yds"], side, dists[side]["qb_name"], "fuzzy qb"
        side = "home" if team_hint == home else ("away" if team_hint == away else "")
        if side:
            return dists[side]["qb_yds"], side, dists[side]["qb_name"], "team qb"
        return None, "", "", "no_match"

    # WR/TE
    if prop in ["WR/TE Receptions","WR/TE Receiving Yards"]:
        maps = {"home": home_wr_map, "away": away_wr_map}
        candidate_sides = (["home", "away"] if not team_hint else (["home"] if team_hint==home else ["away"]))
        for side in candidate_sides:
            nm, idx, how = fuzzy_index(maps[side], pname_norm, cutoff=fuzzy_cutoff)
            if idx is not None:
                arr = dists[side]["wr_recs"][:, idx] if prop=="WR/TE Receptions" else dists[side]["wr_yds"][:, idx]
                disp = dists[side]["wr_names"][idx]
                return arr, side, disp, how
        return None, "", "", "no_match"

    # RB
    if prop in ["RB Rush Attempts","RB Rush Yards"]:
        maps = {"home": home_rb_map, "away": away_rb_map}
        candidate_sides = (["home", "away"] if not team_hint else (["home"] if team_hint==home else ["away"]))
        for side in candidate_sides:
            nm, idx, how = fuzzy_index(maps[side], pname_norm, cutoff=fuzzy_cutoff)
            if idx is not None:
                arr = dists[side]["rb_carr"][:, idx] if prop=="RB Rush Attempts" else dists[side]["rb_yds"][:, idx]
                disp = dists[side]["rb_names"][idx]
                return arr, side, disp, how
        return None, "", "", "no_match"

    return None, "", "", "unsupported_prop"

# Price everything
recs = []
for _, r in odds.iterrows():
    samples, side, matched_name, how = samples_for(r)
    priced = price_line(samples, r["line"], r.get("over_odds", np.nan), r.get("under_odds", np.nan)) if samples is not None else {
        "p_over": np.nan, "p_under": np.nan, "fair_over": np.nan, "fair_under": np.nan,
        "edge_over": np.nan, "edge_under": np.nan, "p50": np.nan, "mean": np.nan
    }
    recs.append({
        "book": r.get("book",""),
        "team_hint": r.get("team",""),
        "roster_team": r.get("roster_team",""),
        "player_csv": r["player"],
        "player_norm": r["player_norm"],
        "player_matched": matched_name,
        "match_method": how,
        "team_side_used": side,
        "prop": r["prop_norm"],
        "market_line": r["line"],
        "over_odds": r.get("over_odds", np.nan),
        "under_odds": r.get("under_odds", np.nan),
        "proj_mean": priced["mean"],
        "proj_p50": priced["p50"],
        "delta_p50_minus_line": priced["p50"] - r["line"] if pd.notna(priced["p50"]) else np.nan,
        "p_over": priced["p_over"],
        "p_under": priced["p_under"],
        "fair_over": priced["fair_over"],
        "fair_under": priced["fair_under"],
        "edge_over_pp": priced["edge_over"]*100 if pd.notna(priced["edge_over"]) else np.nan,
        "edge_under_pp": priced["edge_under"]*100 if pd.notna(priced["edge_under"]) else np.nan,
    })

out = pd.DataFrame(recs)

st.markdown("## Edges vs Market")
if out.empty:
    st.info("No odds matched players in this matchup.")
else:
    out["best_edge_pp"] = out[["edge_over_pp","edge_under_pp"]].abs().max(axis=1)
    st.dataframe(out.sort_values(["best_edge_pp","prop"], ascending=[False, True]), use_container_width=True)

    # Unmatched diagnostics
    unmatched = out[out["match_method"].isin(["no_match","unsupported_prop"])]
    if not unmatched.empty:
        from difflib import SequenceMatcher
        def topk(target_norm, m, k=3):
            cand = []
            for nm in m.keys():
                s = SequenceMatcher(None, target_norm, nm).ratio()
                cand.append((nm, s))
            cand.sort(key=lambda x: x[1], reverse=True)
            return cand[:k]
        rows = []
        for _, r in unmatched.iterrows():
            pname = r["player_norm"]
            cands = []
            cands += [("home_wr",)+x for x in topk(pname, home_wr_map)]
            cands += [("away_wr",)+x for x in topk(pname, away_wr_map)]
            cands += [("home_rb",)+x for x in topk(pname, home_rb_map)]
            cands += [("away_rb",)+x for x in topk(pname, away_rb_map)]
            cands.sort(key=lambda x: x[2], reverse=True)
            rows.append({
                "player_csv": r["player_csv"],
                "player_norm": pname,
                "prop": r["prop"],
                "team_hint": r["team_hint"],
                "roster_team": r["roster_team"],
                "candidates": "; ".join([f"{side}:{nm}({sc:.2f})" for side, nm, sc in cands[:3]])
            })
        dbg = pd.DataFrame(rows).drop_duplicates(subset=["player_norm","prop"])
        st.warning("Unmatched diagnostics — top candidates")
        st.dataframe(dbg.head(100), use_container_width=True)

# ===================== Top N Best Bets =====================
st.markdown("## Top N Best Bets")

with st.expander("Filter & ranking", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        min_edge = st.number_input("Min edge %", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    with c2:
        min_delta_yards = st.number_input("Min delta (yards)", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
    with c3:
        min_delta_counts = st.number_input("Min delta (attempts/receptions)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    with c4:
        top_n = st.number_input("Top N", min_value=1, max_value=200, value=10, step=1)
    with c5:
        dedupe_players = st.checkbox("One pick per player+prop", value=True)

# decide which side has the better edge
pick_over_is_better = out["edge_over_pp"].abs() >= out["edge_under_pp"].abs()
out["pick_side"] = np.where(pick_over_is_better,
                            np.where(out["edge_over_pp"] >= 0, "Over", "Under"),
                            np.where(out["edge_under_pp"] >= 0, "Under", "Over"))

# edge and prices for the chosen side
out["edge_pp"] = np.where(out["pick_side"] == "Over", out["edge_over_pp"], out["edge_under_pp"])
out["market_price"] = np.where(out["pick_side"] == "Over", out["over_odds"], out["under_odds"])
out["fair_price"]   = np.where(out["pick_side"] == "Over", out["fair_over"], out["fair_under"])
out["p_pick"]       = np.where(out["pick_side"] == "Over", out["p_over"], out["p_under"])

# yards vs count delta gating
count_props = out["prop"].isin(["QB Pass Attempts", "WR/TE Receptions", "RB Rush Attempts"])
delta_ok = np.where(count_props,
                    out["delta_p50_minus_line"].abs() >= min_delta_counts,
                    out["delta_p50_minus_line"].abs() >= min_delta_yards)

# filter: positive edge and sufficient delta
bb = out[(out["edge_pp"] >= min_edge) & delta_ok].copy()

# optional: dedupe to one pick per player+prop (keep best edge)
if dedupe_players and not bb.empty:
    bb.sort_values(["player_matched", "prop", "edge_pp"], ascending=[True, True, False], inplace=True)
    bb = bb.drop_duplicates(subset=["player_matched", "prop"], keep="first")

# Kelly sizing (fraction of bankroll) at the posted American odds for the chosen side
def kelly_fraction(p_win: float, american_odds: float) -> float:
    # convert American to decimal-1 (b)
    if not np.isfinite(p_win) or not np.isfinite(american_odds):
        return np.nan
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / abs(american_odds)
    k = (p_win * (b + 1) - 1) / b
    return float(k) if np.isfinite(k) else np.nan

bb["kelly_frac"] = bb.apply(lambda r: kelly_fraction(r["p_pick"], r["market_price"]), axis=1)

# tidy columns
cols = [
    "book", "team_side_used", "player_matched", "prop", "pick_side",
    "market_line", "market_price", "proj_p50", "proj_mean",
    "delta_p50_minus_line", "p_pick", "fair_price", "edge_pp", "kelly_frac",
    "match_method", "team_hint", "roster_team"
]
present_cols = [c for c in cols if c in bb.columns]

# rank and cap to Top N
bb = bb.sort_values(["edge_pp", "p_pick"], ascending=[False, False]).head(int(top_n))

st.dataframe(bb[present_cols], use_container_width=True)

# quick export
st.download_button(
    "Download Top N bets (CSV)",
    data=bb[present_cols].to_csv(index=False),
    file_name="foxedge_top_bets.csv",
    mime="text/csv",
)
# ================== End Top N Best Bets =====================


st.download_button("Download priced odds CSV", data=out.to_csv(index=False), file_name="foxedge_priced_odds.csv", mime="text/csv")

st.caption("This app expects the fixed odds schema you specified. It filters to the selected matchup, maps players via weekly/seasonal rosters, and prices QB/WR/RB core props.")
