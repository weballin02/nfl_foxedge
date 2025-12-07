# nfl_predictor_app.py
# NFL predictor — Residual vs Market (closing lines + QB continuity + fold-wise calibration + conformal)

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import streamlit as st

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMRegressor

try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

try:
    import nfl_data_py as nfl
except Exception:
    st.error("`nfl_data_py` is required. Install with: `pip install nfl_data_py`")
    raise

# ----------------------------- Utilities ------------------------------------ #

def _pick_date_col(df: pd.DataFrame) -> str:
    for c in ["gameday", "game_date", "start_time"]:
        if c in df.columns:
            return c
    return df.columns[0]

def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _team_abbr_fix(s: pd.Series) -> pd.Series:
    return s.fillna("").str.upper()

def _ensure_float(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ewma_prior(x: pd.Series, halflife: float = 5.0) -> float:
    if len(x) == 0:
        return np.nan
    return x.ewm(halflife=halflife, min_periods=1).mean().iloc[-1]

# High-accuracy erf approximation; no SciPy/np.erf needed
def _normal_cdf(z: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26
    z = np.asarray(z, dtype=float)
    sign = np.sign(z)
    x = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    a1,a2,a3,a4,a5 = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
    poly = (((a5*t + a4)*t + a3)*t + a2)*t + a1
    erf_approx = 1.0 - poly * np.exp(-x*x)
    return 0.5 * (1.0 + sign * erf_approx)

def load_closing_lines_csv(path: str) -> pd.DataFrame:
    """
    CSV needs: game_id, close_spread_home, close_total
    close_spread_home: home spread (negative = home favorite), close_total: closing total
    """
    cl = pd.read_csv(path)
    need = {"game_id", "close_spread_home", "close_total"}
    missing = need - set(cl.columns)
    if missing:
        raise ValueError(f"Closing-lines CSV missing columns: {missing}")
    cl = cl[["game_id", "close_spread_home", "close_total"]].copy()
    cl["close_spread_home"] = pd.to_numeric(cl["close_spread_home"], errors="coerce")
    cl["close_total"] = pd.to_numeric(cl["close_total"], errors="coerce")
    return cl

# ----------------------------- Odds / Splits Utilities ---------------------- #

ODDS_REQ = [
    "book", "market", "label", "price", "point",
    "home_team", "away_team", "season", "week"
]

def norm_team_simple(x) -> str:
    """
    Minimal team normalizer. For odds CSVs, you should match abbreviations
    to whatever your model uses (e.g. 'KC', 'PHI', etc).
    """
    if pd.isna(x):
        raise ValueError("Team label is NaN")
    return str(x).strip().upper()


def collapse_odds_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a timestamp exists (ts/timestamp/asof/collected_at), keep earliest
    ('open') and latest ('last') per (book,market,label,home_team,away_team,
    season,week) and attach price/point deltas. Else just dedupe.
    """
    if df.empty:
        return df

    df = df.copy()
    ts_col = None
    for c in ["ts", "timestamp", "asof", "collected_at"]:
        if c in df.columns:
            ts_col = c
            break

    base_cols = ["book", "market", "label", "home_team", "away_team", "season", "week"]

    # Ensure required columns exist
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan

    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["point"] = pd.to_numeric(df.get("point"), errors="coerce")
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce").astype("Int64")

    df = df.dropna(subset=["price"])

    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        df = df.sort_values(base_cols + [ts_col])

        first = (
            df.groupby(base_cols, as_index=False)
              .first()
              .rename(columns={"price": "open_price", "point": "open_point"})
        )
        last = df.groupby(base_cols, as_index=False).last()

        merged = last.merge(
            first[base_cols + ["open_price", "open_point"]],
            on=base_cols,
            how="left",
        )
        merged["price_delta"] = merged["price"] - merged["open_price"]
        merged["point_delta"] = merged["point"] - merged["open_point"]
        return merged

    # No timestamp: just dedupe records
    dedup = df.drop_duplicates(
        subset=["book", "market", "label", "price", "point",
                "home_team", "away_team", "season", "week"]
    )
    return dedup


def load_odds_csv(path: str) -> pd.DataFrame:
    """
    Load a full odds CSV of the form expected by the old edge app:
    columns: ODDS_REQ + optional timestamp columns.

    Returns a cleaned, normalized DataFrame with open/last + deltas when
    timestamps exist.
    """
    df = pd.read_csv(path)
    missing = [c for c in ODDS_REQ if c not in df.columns]
    if missing:
        raise ValueError(f"odds CSV missing required columns: {missing}")

    df["home_team"] = df["home_team"].map(norm_team_simple)
    df["away_team"] = df["away_team"].map(norm_team_simple)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["market"] = df["market"].astype(str).str.lower()
    df["label"] = df["label"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    df = df.dropna(subset=["price"])

    df = collapse_odds_snapshots(df)
    return df


def parse_uploaded_odds(df_ou: pd.DataFrame) -> pd.DataFrame:
    """
    Ingest an uploaded odds CSV in either:

      1) Raw schema (book feed):
         - market, point, price, home_team, away_team, [label]

      2) Template schema (hand-built):
         - Matchup, Bookmaker Line, Over Price, Under Price,
           Home ML, Away ML, [Spread Line, Spread Price]

    Returns a DataFrame with one row per matchup:
      ['matchup','book_line','over_price','under_price',
       'home_ml','away_ml','spread_line','spread_price']
    """
    if df_ou is None or df_ou.empty:
        return pd.DataFrame(
            columns=[
                "matchup", "book_line", "over_price", "under_price",
                "home_ml", "away_ml", "spread_line", "spread_price",
            ]
        )

    df_ou = df_ou.copy()
    df_ou.columns = [c.strip().lower() for c in df_ou.columns]

    # Case 1: raw feed with market/point/price/home_team/away_team
    raw_cols = {"market", "point", "price", "home_team", "away_team"}
    if raw_cols.issubset(df_ou.columns):
        df_ou["market"] = df_ou["market"].astype(str).str.lower()
        if "label" in df_ou.columns:
            df_ou["label"] = df_ou["label"].astype(str)
        else:
            df_ou["label"] = ""

        out_rows = []
        for (h, a), grp in df_ou.groupby(["home_team", "away_team"]):
            matchup = f"{a} @ {h}"

            # Totals (over/under)
            tot = grp[grp["market"].isin(["totals", "total"])]
            line = over_p = under_p = None
            if not tot.empty:
                over = tot[tot["label"].str.lower().eq("over")]
                under = tot[tot["label"].str.lower().eq("under")]
                if not over.empty:
                    line = float(over["point"].iloc[0])
                    over_p = float(over["price"].iloc[0])
                if not under.empty:
                    if line is None:
                        line = float(under["point"].iloc[0])
                    under_p = float(under["price"].iloc[0])

            # Moneyline
            ml = grp[grp["market"] == "moneyline"]
            h_ml = a_ml = None
            if not ml.empty:
                # Try matching via labels if provided
                labels = ml["label"].str.upper()
                home_ml_rows = ml[labels.isin([str(h).upper()])]
                away_ml_rows = ml[labels.isin([str(a).upper()])]
                if not home_ml_rows.empty:
                    h_ml = float(home_ml_rows["price"].iloc[0])
                if not away_ml_rows.empty:
                    a_ml = float(away_ml_rows["price"].iloc[0])
                if h_ml is None and len(ml) >= 1:
                    h_ml = float(ml["price"].iloc[0])
                if a_ml is None and len(ml) >= 2:
                    a_ml = float(ml["price"].iloc[1])

            # Spread: home team line & price
            sp = grp[grp["market"] == "spread"]
            spread_line = spread_price = None
            if not sp.empty:
                # Try label match; fallback to first row
                home_sp = sp[sp["label"].str.upper().eq(str(h).upper())]
                if not home_sp.empty:
                    spread_line = float(home_sp["point"].iloc[0])
                    spread_price = float(home_sp["price"].iloc[0])
                else:
                    spread_line = float(sp["point"].iloc[0])
                    spread_price = float(sp["price"].iloc[0])

            out_rows.append(
                {
                    "matchup": matchup,
                    "book_line": float(line or 0),
                    "over_price": float(over_p or 0),
                    "under_price": float(under_p or 0),
                    "home_ml": float(h_ml or 0),
                    "away_ml": float(a_ml or 0),
                    "spread_line": float(spread_line or 0),
                    "spread_price": float(spread_price or 0),
                }
            )

        return pd.DataFrame(out_rows)

    # Case 2: template schema
    template_cols = {
        "matchup",
        "bookmaker line",
        "over price",
        "under price",
        "home ml",
        "away ml",
    }
    if template_cols.issubset(df_ou.columns):
        rows = []
        for _, r in df_ou.iterrows():
            rows.append(
                {
                    "matchup": str(r["matchup"]).strip(),
                    "book_line": float(r["bookmaker line"]),
                    "over_price": float(r["over price"]),
                    "under_price": float(r["under price"]),
                    "home_ml": float(r["home ml"]),
                    "away_ml": float(r["away ml"]),
                    "spread_line": float(r.get("spread line", 0) or 0),
                    "spread_price": float(r.get("spread price", 0) or 0),
                }
            )
        return pd.DataFrame(rows)

    raise ValueError(
        "Unsupported odds CSV schema. Expected either raw book feed "
        "({market, point, price, home_team, away_team[,label]}) or template "
        "({Matchup, Bookmaker Line, Over Price, Under Price, Home ML, Away ML, ...})."
    )

# ----------------------------- Data Loading --------------------------------- #

@st.cache_data(show_spinner=False)
def load_seasons(seasons: List[int], closing_path: Optional[str]) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        sched = nfl.import_schedules([yr]).reset_index(drop=True)
        date_col = _pick_date_col(sched)
        sched["game_date"] = _to_dt(sched[date_col])

        base_cols = [
            "game_id","game_date","season","week",
            "home_team","away_team","home_score","away_score",
            "venue","roof","surface",
            "spread_line","total_line",
            "temp","wind","humidity"
        ]
        have = [c for c in base_cols if c in sched.columns]
        df = sched[have].rename(columns={"home_score":"home_points","away_score":"away_points"})
        frames.append(df)

    g = pd.concat(frames, ignore_index=True).sort_values("game_date")
    g["home_team"] = _team_abbr_fix(g["home_team"])
    g["away_team"] = _team_abbr_fix(g["away_team"])
    g["played"] = g["home_points"].notna() & g["away_points"].notna()

    # environment flags
    g["roof"] = g.get("roof", "outdoors").astype(str).str.lower().fillna("outdoors")
    g["surface"] = g.get("surface", "").astype(str).str.lower().fillna("")
    g["is_dome"] = g["roof"].isin(["dome","closed"]).astype(int)
    g["is_retractable"] = g["roof"].eq("retractable").astype(int)
    g["is_turf"] = g["surface"].str.contains("turf|fieldturf|artificial").astype(int)

    # weather normalize
    for c in ["temp","wind","humidity"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        else:
            g[c] = np.nan
    g["wind_12plus"] = (g["wind"].fillna(0) >= 12).astype(int)
    g["wind_15plus"] = (g["wind"].fillna(0) >= 15).astype(int)

    # market normalize
    if "spread_line" not in g.columns: g["spread_line"] = np.nan
    if "total_line" not in g.columns: g["total_line"] = np.nan
    g["spread_line"] = pd.to_numeric(g["spread_line"], errors="coerce")  # home spread
    g["total_line"]  = pd.to_numeric(g["total_line"], errors="coerce")

    # Optional closing lines override
    if closing_path:
        try:
            cl = load_closing_lines_csv(closing_path)
            g = g.merge(cl, on="game_id", how="left")
            g["spread_line"] = np.where(g["close_spread_home"].notna(), g["close_spread_home"], g["spread_line"])
            g["total_line"]  = np.where(g["close_total"].notna(), g["close_total"], g["total_line"])
        except Exception as e:
            st.warning(f"Closing-lines merge skipped: {e}")

    return g

def to_team_game_logs(games: pd.DataFrame) -> pd.DataFrame:
    home = games[["game_id","game_date","season","week","home_team","away_team","home_points","away_points"]].copy()
    away = games[["game_id","game_date","season","week","home_team","away_team","home_points","away_points"]].copy()

    home = home.rename(columns={"home_team":"team_abbr","away_team":"opp_abbr"})
    away = away.rename(columns={"away_team":"team_abbr","home_team":"opp_abbr"})

    home["points_for"] = home["home_points"]
    home["points_against"] = home["away_points"]
    home["is_home"] = 1

    away["points_for"] = away["away_points"]
    away["points_against"] = away["home_points"]
    away["is_home"] = 0

    cols = ["game_id","game_date","season","week","team_abbr","opp_abbr","is_home","points_for","points_against"]
    team_logs = pd.concat([home[cols], away[cols]], ignore_index=True)
    team_logs = _ensure_float(team_logs, ["points_for","points_against"])
    return team_logs.sort_values(["game_date","team_abbr"]).reset_index(drop=True)

def add_rolling_features(team_logs: pd.DataFrame, windows=(1,3,5)) -> pd.DataFrame:
    out = team_logs.copy().sort_values(["team_abbr","game_date"])
    for w in windows:
        out[f"pf_{w}w"] = out.groupby("team_abbr")["points_for"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        out[f"pa_{w}w"] = out.groupby("team_abbr")["points_against"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    out["pf_ewma"] = out.groupby("team_abbr")["points_for"].transform(lambda s: s.shift(1).ewm(halflife=5, min_periods=1).mean())
    return out

def _one_step_arima_prior(series: pd.Series) -> Optional[float]:
    if not _HAS_ARIMA or series.dropna().shape[0] < 6:
        return np.nan
    try:
        model = ARIMA(series.astype(float), order=(1,0,0))
        res = model.fit(method_kwargs={"warn_convergence": False})
        f = res.forecast(steps=1)
        return float(f.iloc[0])
    except Exception:
        return np.nan

def add_scoring_priors(team_logs: pd.DataFrame) -> pd.DataFrame:
    out = team_logs.copy().sort_values(["team_abbr","game_date"])
    priors = []
    for _, g in out.groupby("team_abbr", sort=False):
        pf_hist = g["points_for"]
        team_priors = []
        for idx in range(len(g)):
            hist = pf_hist.iloc[:idx].dropna()
            if len(hist) == 0:
                team_priors.append(np.nan); continue
            arima_pred = _one_step_arima_prior(hist)
            team_priors.append(_ewma_prior(hist) if np.isnan(arima_pred) else arima_pred)
        priors.extend(team_priors)
    out["pf_prior"] = np.array(priors, dtype=float)
    league_rolling = out.groupby("game_date")["points_for"].transform(lambda s: s.mean())
    out["pf_prior"] = out["pf_prior"].fillna(league_rolling)
    return out

def add_rest_days(team_logs: pd.DataFrame) -> pd.DataFrame:
    tl = team_logs.copy().sort_values(["team_abbr","game_date"])
    tl["prev_game_date"] = tl.groupby("team_abbr")["game_date"].shift(1)
    tl["rest_days"] = (tl["game_date"] - tl["prev_game_date"]).dt.total_seconds() / 86400.0
    tl["rest_days"] = tl["rest_days"].clip(lower=3, upper=20).fillna(7.0)
    tl["short_week"] = (tl["rest_days"] <= 6).astype(int)
    return tl.drop(columns=["prev_game_date"])

def add_games_played(team_logs: pd.DataFrame) -> pd.DataFrame:
    tl = team_logs.copy().sort_values(["team_abbr","game_date"])
    tl["games_played"] = tl.groupby("team_abbr").cumcount()
    return tl

# ---------- PBP-derived form & QB continuity -------- #

@st.cache_data(show_spinner=False)
def load_pbp(seasons: List[int]) -> pd.DataFrame:
    try:
        pbp = nfl.import_pbp_data(seasons)
    except Exception:
        return pd.DataFrame()
    cols = ["game_id","game_date","posteam","epa","pass","qb_dropback","passer_player_id"]
    for c in cols:
        if c not in pbp.columns:
            pbp[c] = np.nan
    pbp["game_date"] = _to_dt(pbp["game_date"])
    pbp = pbp[pbp["posteam"].notna()].copy()
    pbp["posteam"] = _team_abbr_fix(pbp["posteam"])
    pbp["is_play"] = pbp["epa"].notna().astype(int)
    return pbp

def pbp_offense_agg(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp.empty:
        return pd.DataFrame(columns=["game_id","team_abbr","off_epa_per_play","pass_rate"])
    agg = (
        pbp.groupby(["game_id","posteam"], as_index=False)
           .agg(plays=("is_play","sum"),
                epa_sum=("epa","sum"),
                dropbacks=("qb_dropback","sum"),
                passes=("pass","sum"))
    )
    agg["off_epa_per_play"] = np.where(agg["plays"]>0, agg["epa_sum"]/agg["plays"], 0.0)
    agg["pass_rate"] = np.where(agg["plays"]>0, agg["passes"]/agg["plays"], 0.0)
    agg = agg.rename(columns={"posteam":"team_abbr"})
    return agg[["game_id","team_abbr","off_epa_per_play","pass_rate"]]

def qb_continuity_flags(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp.empty:
        return pd.DataFrame(columns=["game_id","team_abbr","qb_first_id"])
    fp = (
        pbp[pbp["pass"] == 1]
        .sort_values(["game_id","game_date"])
        .groupby(["game_id","posteam"], as_index=False)
        .first()[["game_id","posteam","passer_player_id"]]
    )
    fp = fp.rename(columns={"posteam":"team_abbr","passer_player_id":"qb_first_id"})
    fp["team_abbr"] = _team_abbr_fix(fp["team_abbr"])
    return fp

def attach_pbp_form_and_qb(team_logs: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    tl = team_logs.copy()
    # offense form
    agg = pbp_offense_agg(pbp)
    if agg.empty:
        for c in ["off_epa_per_play","pass_rate","off_epa3","off_epa5","pass_rate3","pass_rate5"]:
            tl[c] = np.nan
    else:
        tl = tl.merge(agg, on=["game_id","team_abbr"], how="left")
        tl = tl.sort_values(["team_abbr","game_date"])
        tl["off_epa3"] = tl.groupby("team_abbr")["off_epa_per_play"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        tl["off_epa5"] = tl.groupby("team_abbr")["off_epa_per_play"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        tl["pass_rate3"] = tl.groupby("team_abbr")["pass_rate"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        tl["pass_rate5"] = tl.groupby("team_abbr")["pass_rate"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    # QB continuity
    qbf = qb_continuity_flags(pbp)
    if qbf.empty:
        tl["qb_changed_prev"] = 0
        return tl
    tl = tl.merge(qbf, on=["game_id","team_abbr"], how="left")
    tl = tl.sort_values(["team_abbr","game_date"])
    tl["qb_first_id"] = tl.groupby("team_abbr")["qb_first_id"].fillna(method="ffill").fillna("UNK")
    tl["qb_prev_id"] = tl.groupby("team_abbr")["qb_first_id"].shift(1)
    tl["qb_changed_prev"] = (tl["qb_prev_id"].notna() & (tl["qb_first_id"] != tl["qb_prev_id"])).astype(int)
    tl = tl.drop(columns=["qb_prev_id"])
    return tl

# --------------------------- Matchup Assembly -------------------------------- #

def _latest_before(tl: pd.DataFrame, team: str, date) -> Optional[pd.Series]:
    rows = tl[(tl["team_abbr"] == team) & (tl["game_date"] < date)]
    return rows.iloc[-1] if len(rows) else None

def assemble_training(games: pd.DataFrame, team_logs_feats: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    played = games[games["home_points"].notna()].copy()
    tl = team_logs_feats.copy().sort_values(["team_abbr","game_date"])

    recs = []
    for _, r in played.iterrows():
        h = _latest_before(tl, r["home_team"], r["game_date"])
        a = _latest_before(tl, r["away_team"], r["game_date"])
        if h is None or a is None:
            continue

        wind = r.get("wind", np.nan)
        wind_val = float(wind) if pd.notna(wind) else 0.0
        wind_val = float(np.clip(wind_val, 0.0, 30.0))
        is_turf = int(r.get("is_turf", 0))
        is_dome = int(r.get("is_dome", 0))

        sp = r.get("spread_line", np.nan)
        tlm = r.get("total_line", np.nan)
        abs_sp = float(abs(sp)) if pd.notna(sp) else 0.0
        abs_tl = float(abs(tlm)) if pd.notna(tlm) else 0.0

        rec = {
            "game_id": r["game_id"], "game_date": r["game_date"],
            "season": r["season"], "week": r["week"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "home_points": r["home_points"], "away_points": r["away_points"],

            # priors and rolls
            "home_prior": h["pf_prior"], "away_prior": a["pf_prior"],
            "home_pf1": h.get("pf_1w", np.nan), "home_pf3": h.get("pf_3w", np.nan), "home_pf5": h.get("pf_5w", np.nan),
            "home_pa1": h.get("pa_1w", np.nan), "home_pa3": h.get("pa_3w", np.nan), "home_pa5": h.get("pa_5w", np.nan),
            "home_pf_ewma": h.get("pf_ewma", np.nan),
            "away_pf1": a.get("pf_1w", np.nan), "away_pf3": a.get("pf_3w", np.nan), "away_pf5": a.get("pf_5w", np.nan),
            "away_pa1": a.get("pa_1w", np.nan), "away_pa3": a.get("pa_3w", np.nan), "away_pa5": a.get("pa_5w", np.nan),
            "away_pf_ewma": a.get("pf_ewma", np.nan),

            # PBP offensive form
            "home_off_epa3": h.get("off_epa3", np.nan), "home_off_epa5": h.get("off_epa5", np.nan),
            "home_pass3": h.get("pass_rate3", np.nan), "home_pass5": h.get("pass_rate5", np.nan),
            "away_off_epa3": a.get("off_epa3", np.nan), "away_off_epa5": a.get("off_epa5", np.nan),
            "away_pass3": a.get("pass_rate3", np.nan), "away_pass5": a.get("pass_rate5", np.nan),

            # QB continuity proxy
            "home_qb_change": h.get("qb_changed_prev", 0),
            "away_qb_change": a.get("qb_changed_prev", 0),

            # rest/env/weather
            "home_rest": h["rest_days"], "away_rest": a["rest_days"],
            "home_short": h["short_week"], "away_short": a["short_week"],
            "rest_diff": h["rest_days"] - a["rest_days"],
            "is_dome": is_dome, "is_turf": is_turf,
            "temp": r.get("temp", np.nan), "wind": wind_val,
            "wind_12plus": r.get("wind_12plus", 0), "wind_15plus": r.get("wind_15plus", 0),

            # wind interactions
            "wind_sq": wind_val ** 2,
            "wind_on_turf": is_turf * wind_val,
            "wind_outdoors": (1 - is_dome) * wind_val,

            # market
            "spread_line": sp,
            "total_line": tlm,
            "abs_spread": abs_sp,
            "abs_total": abs_tl,
        }
        recs.append(rec)

    train = pd.DataFrame(recs).dropna(subset=["home_points","away_points"])
    train["margin"] = train["home_points"] - train["away_points"]
    train["total"]  = train["home_points"] + train["away_points"]
    train["margin_resid"] = train["margin"] - (-train["spread_line"])
    train["total_resid"]  = train["total"]  - train["total_line"]

    features = [
        # priors
        "home_prior","away_prior",
        # PF/PA rolls
        "home_pf1","home_pf3","home_pf5","home_pa1","home_pa3","home_pa5","home_pf_ewma",
        "away_pf1","away_pf3","away_pf5","away_pa1","away_pa3","away_pa5","away_pf_ewma",
        # PBP form
        "home_off_epa3","home_off_epa5","home_pass3","home_pass5",
        "away_off_epa3","away_off_epa5","away_pass3","away_pass5",
        # QB continuity
        "home_qb_change","away_qb_change",
        # rest/env/weather
        "home_rest","away_rest","rest_diff","home_short","away_short",
        "is_dome","is_turf","temp","wind","wind_12plus","wind_15plus",
        # wind interactions
        "wind_sq","wind_on_turf","wind_outdoors",
        # market anchors
        "spread_line","total_line","abs_spread","abs_total",
    ]
    trainX = train.reindex(columns=features, fill_value=0).copy()
    return train, trainX, features

def assemble_upcoming(games: pd.DataFrame, team_logs_feats: pd.DataFrame) -> pd.DataFrame:
    future = games[~games["played"]].copy()
    if future.empty:
        return future.assign(msg="no upcoming games")
    tl = team_logs_feats.copy().sort_values(["team_abbr","game_date"])

    recs = []
    for _, r in future.iterrows():
        h = _latest_before(tl, r["home_team"], r["game_date"])
        a = _latest_before(tl, r["away_team"], r["game_date"])
        if h is None or a is None:
            continue

        wind = r.get("wind", np.nan)
        wind_val = float(wind) if pd.notna(wind) else 0.0
        wind_val = float(np.clip(wind_val, 0.0, 30.0))
        is_turf = int(r.get("is_turf", 0))
        is_dome = int(r.get("is_dome", 0))

        sp = r.get("spread_line", np.nan)
        tlm = r.get("total_line", np.nan)
        abs_sp = float(abs(sp)) if pd.notna(sp) else 0.0
        abs_tl = float(abs(tlm)) if pd.notna(tlm) else 0.0

        rec = {
            "game_id": r["game_id"], "game_date": r["game_date"],
            "season": r["season"], "week": r["week"],
            "home_team": r["home_team"], "away_team": r["away_team"],

            "home_prior": h["pf_prior"], "away_prior": a["pf_prior"],
            "home_pf1": h.get("pf_1w", np.nan), "home_pf3": h.get("pf_3w", np.nan), "home_pf5": h.get("pf_5w", np.nan),
            "home_pa1": h.get("pa_1w", np.nan), "home_pa3": h.get("pa_3w", np.nan), "home_pa5": h.get("pa_5w", np.nan),
            "home_pf_ewma": h.get("pf_ewma", np.nan),
            "away_pf1": a.get("pf_1w", np.nan), "away_pf3": a.get("pf_3w", np.nan), "away_pf5": a.get("pf_5w", np.nan),
            "away_pa1": a.get("pa_1w", np.nan), "away_pa3": a.get("pa_3w", np.nan), "away_pa5": a.get("pa_5w", np.nan),
            "away_pf_ewma": a.get("pf_ewma", np.nan),

            "home_off_epa3": h.get("off_epa3", np.nan), "home_off_epa5": h.get("off_epa5", np.nan),
            "home_pass3": h.get("pass_rate3", np.nan), "home_pass5": h.get("pass_rate5", np.nan),
            "away_off_epa3": a.get("off_epa3", np.nan), "away_off_epa5": a.get("off_epa5", np.nan),
            "away_pass3": a.get("pass_rate3", np.nan), "away_pass5": a.get("pass_rate5", np.nan),

            "home_qb_change": h.get("qb_changed_prev", 0),
            "away_qb_change": a.get("qb_changed_prev", 0),

            "home_rest": h["rest_days"], "away_rest": a["rest_days"],
            "home_short": h["short_week"], "away_short": a["short_week"],
            "rest_diff": h["rest_days"] - a["rest_days"],
            "is_dome": is_dome, "is_turf": is_turf,
            "temp": r.get("temp", np.nan), "wind": wind_val,
            "wind_12plus": r.get("wind_12plus", 0), "wind_15plus": r.get("wind_15plus", 0),

            "wind_sq": wind_val ** 2,
            "wind_on_turf": is_turf * wind_val,
            "wind_outdoors": (1 - is_dome) * wind_val,

            "spread_line": sp,
            "total_line": tlm,
            "abs_spread": abs_sp,
            "abs_total": abs_tl,

            "home_gp": h["games_played"], "away_gp": a["games_played"],
        }
        recs.append(rec)
    return pd.DataFrame(recs)

# ------------------------------- Modeling ----------------------------------- #

@dataclass
class Models:
    resid_margin: LGBMRegressor
    resid_total: LGBMRegressor
    winprob: CalibratedClassifierCV

@dataclass
class Diagnostics:
    metrics: Dict[str, float]
    conformal: Dict[str, float]
    tau_overdisp: float

def conformal_quantile(abs_residuals: np.ndarray, alpha: float, default: float = 8.0) -> float:
    arr = np.asarray(abs_residuals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, 1 - alpha))

@st.cache_resource(show_spinner=False)
def train_models_residual(train: pd.DataFrame,
                          trainX: pd.DataFrame,
                          features: List[str],
                          n_splits: int = 5) -> Tuple[Models, Diagnostics]:

    has_mkt_margin = train["spread_line"].notna()
    has_mkt_total  = train["total_line"].notna()

    model_margin = LGBMRegressor(
        objective="mae",
        n_estimators=900, learning_rate=0.03,
        max_depth=-1, num_leaves=63,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, min_data_in_leaf=60,
        random_state=42
    )
    model_total = LGBMRegressor(
        objective="mae",
        n_estimators=900, learning_rate=0.03,
        max_depth=-1, num_leaves=63,
        subsample=0.9, colsample_bytree=0.8,
        reg_lambda=1.5, min_data_in_leaf=90,  # tightened for stability
        random_state=42
    )

    # --------- First pass: OOF point preds (using single models incrementally) ---------
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n = len(trainX)
    oof_margin_pred = np.full(n, np.nan, dtype=float)
    oof_total_pred  = np.full(n, np.nan, dtype=float)

    for tr_idx, va_idx in tscv.split(trainX):
        # Margin residual
        tr_mask_m = has_mkt_margin.iloc[tr_idx].values
        if tr_mask_m.sum() > 10:
            Xtr_m = trainX.iloc[tr_idx][features][tr_mask_m]
            ytr_m = train.iloc[tr_idx]["margin_resid"][tr_mask_m]
            w_m = np.linspace(0.5, 1.0, Xtr_m.shape[0])
            model_margin.fit(Xtr_m, ytr_m, sample_weight=w_m)

            Xva_m = trainX.iloc[va_idx][features]
            resid_m = model_margin.predict(Xva_m)
            sp_va  = train.iloc[va_idx]["spread_line"].values
            sp_va  = np.nan_to_num(sp_va, nan=0.0)
            pred_m = resid_m + (-sp_va)
            oof_margin_pred[va_idx] = pred_m

        # Total residual
        tr_mask_t = has_mkt_total.iloc[tr_idx].values
        if tr_mask_t.sum() > 10:
            Xtr_t = trainX.iloc[tr_idx][features][tr_mask_t]
            ytr_t = train.iloc[tr_idx]["total_resid"][tr_mask_t]
            w_t = np.linspace(0.5, 1.0, Xtr_t.shape[0])
            model_total.fit(Xtr_t, ytr_t, sample_weight=w_t)

            Xva_t = trainX.iloc[va_idx][features]
            resid_t = model_total.predict(Xva_t)
            tl_va  = train.iloc[va_idx]["total_line"].values
            tl_va  = np.nan_to_num(tl_va, nan=0.0)
            pred_t = resid_t + tl_va
            oof_total_pred[va_idx] = pred_t

    # --------- Metrics (OOF points) ---------
    mask_eval = (~np.isnan(oof_margin_pred)) & (~np.isnan(oof_total_pred))
    if mask_eval.any():
        mae_margin = mean_absolute_error(train["margin"][mask_eval], oof_margin_pred[mask_eval])
        ph = (oof_total_pred[mask_eval] + oof_margin_pred[mask_eval]) / 2.0
        pa = (oof_total_pred[mask_eval] - oof_margin_pred[mask_eval]) / 2.0
        mae_home = mean_absolute_error(train["home_points"].values[mask_eval], ph)
        mae_away = mean_absolute_error(train["away_points"].values[mask_eval], pa)
        mae_total = mean_absolute_error(train["total"].values[mask_eval], oof_total_pred[mask_eval])
    else:
        mae_margin = mae_home = mae_away = mae_total = np.nan

    # --------- Fold-wise calibrator for honest OOF win prob ---------
    oof_winprob = np.full(n, np.nan, dtype=float)
    tscv2 = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in tscv2.split(trainX):
        # Train per-fold residual models on TR only (no leakage)
        tr_mask_m = has_mkt_margin.iloc[tr_idx].values
        tr_mask_t = has_mkt_total.iloc[tr_idx].values
        va_mask   = (has_mkt_margin.iloc[va_idx].values) & (has_mkt_total.iloc[va_idx].values)
        if tr_mask_m.sum() < 200 or tr_mask_t.sum() < 200 or va_mask.sum() < 50:
            continue

        # Fit residual models on TR
        m_fold = LGBMRegressor(**model_margin.get_params())
        t_fold = LGBMRegressor(**model_total.get_params())

        Xtr_m = trainX.iloc[tr_idx][features][tr_mask_m]
        ytr_m = train.iloc[tr_idx]["margin_resid"][tr_mask_m]
        w_m = np.linspace(0.5, 1.0, Xtr_m.shape[0])
        m_fold.fit(Xtr_m, ytr_m, sample_weight=w_m)

        Xtr_t = trainX.iloc[tr_idx][features][tr_mask_t]
        ytr_t = train.iloc[tr_idx]["total_resid"][tr_mask_t]
        w_t = np.linspace(0.5, 1.0, Xtr_t.shape[0])
        t_fold.fit(Xtr_t, ytr_t, sample_weight=w_t)

        # TRAIN meta-features for calibrator
        sp_tr = np.nan_to_num(train["spread_line"].values[tr_idx][tr_mask_m & tr_mask_t], nan=0.0)
        tl_tr = np.nan_to_num(train["total_line"].values[tr_idx][tr_mask_m & tr_mask_t],  nan=0.0)
        XtrF  = trainX.iloc[tr_idx][features].iloc[tr_mask_m & tr_mask_t]
        resid_m_tr = m_fold.predict(XtrF)
        resid_t_tr = t_fold.predict(XtrF)
        m_tr = resid_m_tr + (-sp_tr)
        t_tr = resid_t_tr + tl_tr
        d_m_tr = m_tr - sp_tr
        d_t_tr = t_tr - tl_tr
        calib_X_tr = np.c_[m_tr, t_tr, sp_tr, tl_tr, d_m_tr, d_t_tr]
        y_tr = (train["margin"].values[tr_idx][tr_mask_m & tr_mask_t] > 0).astype(int)

        base = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=350,
            l2_regularization=0.5, max_bins=255
        )
        fold_cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
        fold_cal.fit(calib_X_tr, y_tr)

        # VALIDATION meta-features and calibrated OOF
        sp_va = np.nan_to_num(train["spread_line"].values[va_idx][va_mask], nan=0.0)
        tl_va = np.nan_to_num(train["total_line"].values[va_idx][va_mask],  nan=0.0)
        XvaF  = trainX.iloc[va_idx][features].iloc[va_mask]
        resid_m_va = m_fold.predict(XvaF)
        resid_t_va = t_fold.predict(XvaF)
        m_va = resid_m_va + (-sp_va)
        t_va = resid_t_va + tl_va
        d_m_va = m_va - sp_va
        d_t_va = t_va - tl_va
        calib_X_va = np.c_[m_va, t_va, sp_va, tl_va, d_m_va, d_t_va]
        oof_winprob_fold = fold_cal.predict_proba(calib_X_va)[:, 1]

        idx_write = np.array(va_idx)[va_mask]
        oof_winprob[idx_write] = oof_winprob_fold

    mask_prob = ~np.isnan(oof_winprob)
    if mask_prob.any():
        brier = brier_score_loss((train["margin"].values[mask_prob] > 0).astype(int), oof_winprob[mask_prob])
    else:
        brier = np.nan

    # --------- Conformal 80% bands from OOF ---------
    if mask_eval.any():
        ph_all = (oof_total_pred + oof_margin_pred) / 2.0
        pa_all = (oof_total_pred - oof_margin_pred) / 2.0
        res_home  = train["home_points"].values[mask_eval] - ph_all[mask_eval]
        res_away  = train["away_points"].values[mask_eval] - pa_all[mask_eval]
        res_total = train["total"].values[mask_eval] - oof_total_pred[mask_eval]
        q80_home  = conformal_quantile(np.abs(res_home),  0.20, default=7.5)
        q80_away  = conformal_quantile(np.abs(res_away),  0.20, default=7.5)
        q80_total = conformal_quantile(np.abs(res_total), 0.20, default=10.0)
    else:
        q80_home = 7.5; q80_away = 7.5; q80_total = 10.0

    metrics = {
        "MAE_margin": float(mae_margin) if pd.notna(mae_margin) else np.nan,
        "MAE_home": float(mae_home) if pd.notna(mae_home) else np.nan,
        "MAE_away": float(mae_away) if pd.notna(mae_away) else np.nan,
        "MAE_total": float(mae_total) if pd.notna(mae_total) else np.nan,
        "Brier_win": float(brier) if pd.notna(brier) else np.nan,
    }

    # --------- Final full-train refit of residual models ---------
    resid_margin_mask = train["margin_resid"].notna() & train["spread_line"].notna()
    resid_total_mask  = train["total_resid"].notna()  & train["total_line"].notna()

    X_full_m = trainX[features][resid_margin_mask]
    y_full_m = train.loc[resid_margin_mask, "margin_resid"]
    w_full_m = np.linspace(0.5, 1.0, len(X_full_m))
    model_margin.fit(X_full_m, y_full_m, sample_weight=w_full_m)

    X_full_t = trainX[features][resid_total_mask]
    y_full_t = train.loc[resid_total_mask, "total_resid"]
    w_full_t = np.linspace(0.5, 1.0, len(X_full_t))
    model_total.fit(X_full_t, y_full_t, sample_weight=w_full_t)

    # --------- Deployment calibrator on full data ---------
    sp_all = np.nan_to_num(train["spread_line"].values, nan=0.0)
    tl_all = np.nan_to_num(train["total_line"].values,  nan=0.0)
    resid_m_full = model_margin.predict(trainX[features])
    resid_t_full = model_total.predict(trainX[features])
    m_full = resid_m_full + (-sp_all)
    t_full = resid_t_full + tl_all
    delta_m_full = m_full - sp_all
    delta_t_full = t_full - tl_all
    calib_X = np.c_[m_full, t_full, sp_all, tl_all, delta_m_full, delta_t_full]
    y_full = (train["margin"].values > 0).astype(int)
    base = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=350,
        l2_regularization=0.5, max_bins=255
    )
    win_clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    win_clf.fit(calib_X, y_full)

    diags = Diagnostics(metrics=metrics, conformal={"q80_home": q80_home, "q80_away": q80_away, "q80_total": q80_total}, tau_overdisp=1.0)
    return Models(resid_margin=model_margin, resid_total=model_total, winprob=win_clf), diags

def compute_adaptive_blend_weight(h_gp: int, a_gp: int, wind: float, is_dome: int) -> float:
    # Base: 0.30 -> 0.80 between 3 and 12 games (weight on model)
    g = max(int(h_gp), int(a_gp))
    if g <= 3: w = 0.30
    elif g >= 12: w = 0.80
    else: w = 0.30 + (0.80 - 0.30) * ((g - 3) / (12 - 3))
    wind = 0.0 if np.isnan(wind) else float(wind)
    if is_dome == 1: w += 0.05
    if wind >= 12: w -= 0.05
    if wind >= 15: w -= 0.05
    return float(np.clip(w, 0.15, 0.90))

def predict_upcoming(models: Models,
                     diagnostics: Diagnostics,
                     features: List[str],
                     upc: pd.DataFrame,
                     round_display: bool = True) -> pd.DataFrame:

    Xf = upc.reindex(columns=features, fill_value=0).copy()

    resid_margin = models.resid_margin.predict(Xf)
    resid_total  = models.resid_total.predict(Xf)

    sp = upc.get("spread_line", pd.Series(np.zeros(len(upc)))).values
    tl = upc.get("total_line",  pd.Series(np.zeros(len(upc)))).values
    sp = np.nan_to_num(sp, nan=0.0)
    tl = np.nan_to_num(tl, nan=0.0)
    pred_margin_model = resid_margin + (-sp)
    pred_total_model  = resid_total  + tl

    prior_total  = upc["home_prior"].values + upc["away_prior"].values
    prior_margin = upc["home_prior"].values - upc["away_prior"].values
    pred_margin_prior_mkt = (-sp) * 0.75 + prior_margin * 0.25
    pred_total_prior_mkt  = tl * 0.75 + prior_total * 0.25

    h_gp = upc.get("home_gp", pd.Series(np.zeros(len(upc)))).values
    a_gp = upc.get("away_gp", pd.Series(np.zeros(len(upc)))).values
    winds = upc.get("wind", pd.Series(np.zeros(len(upc)))).values
    domes = upc.get("is_dome", pd.Series(np.zeros(len(upc)))).values
    w_vec = np.array([compute_adaptive_blend_weight(h, a, w, d) for h,a,w,d in zip(h_gp,a_gp,winds,domes)], dtype=float)

    has_mkt = np.isfinite(sp) & np.isfinite(tl)
    w_vec = np.where(has_mkt, w_vec, np.clip(w_vec - 0.15, 0.15, 0.90))

    pred_margin = w_vec * pred_margin_model + (1 - w_vec) * pred_margin_prior_mkt
    pred_total  = w_vec * pred_total_model  + (1 - w_vec) * pred_total_prior_mkt

    pred_home_pts = (pred_total + pred_margin) / 2.0
    pred_away_pts = (pred_total - pred_margin) / 2.0

    delta_m = pred_margin - sp
    delta_t = pred_total - tl
    m_pred = np.c_[np.nan_to_num(pred_margin), np.nan_to_num(pred_total), sp, tl, delta_m, delta_t]
    win_prob = models.winprob.predict_proba(m_pred)[:, 1]

    out = upc.copy()
    out["pred_margin"] = pred_margin
    out["pred_total"] = pred_total
    out["pred_home_pts"] = pred_home_pts
    out["pred_away_pts"] = pred_away_pts
    out["win_prob"] = win_prob

    out["pred_margin_model"] = pred_margin_model
    out["pred_total_model"]  = pred_total_model
    out["pred_margin_prior_mkt"] = pred_margin_prior_mkt
    out["pred_total_prior_mkt"]  = pred_total_prior_mkt
    out["blend_w"] = w_vec

    qh = diagnostics.conformal["q80_home"]
    qa = diagnostics.conformal["q80_away"]
    qt = diagnostics.conformal["q80_total"]
    out["home_lo80"] = pred_home_pts - qh
    out["home_hi80"] = pred_home_pts + qh
    out["away_lo80"] = pred_away_pts - qa
    out["away_hi80"] = pred_away_pts + qa
    out["total_lo80"] = pred_total - qt
    out["total_hi80"] = pred_total + qt

    if round_display:
        for c in ["pred_home_pts","pred_away_pts","pred_total"]:
            out[f"{c}_disp"] = (out[c] * 2).round() / 2.0
        for c in ["home_lo80","home_hi80","away_lo80","away_hi80","total_lo80","total_hi80"]:
            out[f"{c}_disp"] = (out[c] * 2).round() / 2.0

    return out


# ------------------------------- Orchestration ------------------------------- #

def attach_uploaded_odds_to_upcoming(upcoming: pd.DataFrame,
                                     odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take upcoming games (home_team, away_team, etc.) and a parsed odds_df from
    parse_uploaded_odds(), and attach total_line / spread_line / moneylines.
    """
    if upcoming.empty or odds_df.empty:
        return upcoming

    up = upcoming.copy()

    # Build a matchup key like "AWAY @ HOME" to mirror uploader convention
    up["matchup"] = up["away_team"].astype(str).str.upper() + " @ " + up["home_team"].astype(str).str.upper()

    # Normalize odds_df
    odds = odds_df.copy()
    odds["matchup"] = odds["matchup"].astype(str).str.upper()

    up = up.merge(odds, on="matchup", how="left", suffixes=("", "_odds"))

    # Attach lines if not already present
    if "total_line" not in up.columns or up["total_line"].isna().all():
        up["total_line"] = up["book_line"].replace(0, np.nan)

    if "spread_line" not in up.columns or up["spread_line"].isna().all():
        # spread_line in odds is from home perspective in the uploader logic
        up["spread_line"] = up["spread_line"].where(up["spread_line"].notna(), up["spread_line_odds"])

    # Optional: attach moneyline fields for later use
    if "home_ml" not in up.columns:
        up["home_ml"] = up.get("home_ml_odds", np.nan)
    if "away_ml" not in up.columns:
        up["away_ml"] = up.get("away_ml_odds", np.nan)

    return up

def run_pipeline(seasons: List[int], closing_path: Optional[str], n_splits: int) -> Tuple[Models, pd.DataFrame, Diagnostics, pd.DataFrame, List[str]]:
    games = load_seasons(seasons, closing_path)

    team_logs = to_team_game_logs(games)
    team_logs = add_rolling_features(team_logs, windows=(1,3,5))
    team_logs = add_scoring_priors(team_logs)
    team_logs = add_rest_days(team_logs)

    pbp = load_pbp(seasons)
    team_logs = attach_pbp_form_and_qb(team_logs, pbp)
    team_logs = add_games_played(team_logs)

    train, trainX, features = assemble_training(games, team_logs)
    models, diags = train_models_residual(train, trainX, features, n_splits=n_splits)

    upcoming = assemble_upcoming(games, team_logs)
    preds = pd.DataFrame()
    if not upcoming.empty and "msg" not in upcoming.columns:
        preds = predict_upcoming(models, diags, features, upcoming, round_display=True)

    return models, preds, diags, upcoming, features

# --------------------------------- Streamlit -------------------------------- #

st.set_page_config(page_title="NFL Predictor — Residual vs Market (Enhanced)", layout="wide")
st.title("NFL Predictor — Residual vs Market (Enhanced)")
st.caption("Closing lines (optional), QB continuity proxy, MAE residual trees with recency weights, fold-wise calibrated win prob, adaptive blend, conformal bands.")

with st.sidebar:
    st.subheader("Controls")
    colA, colB = st.columns(2)
    start_season = colA.number_input("Start season", min_value=2002, max_value=2050, value=2015, step=1)
    end_season   = colB.number_input("End season", min_value=2002, max_value=2050, value=2024, step=1)
    splits = st.slider("TimeSeries splits", min_value=3, max_value=8, value=5, step=1)
    closing_path = st.text_input("Closing lines CSV (optional)", value="")
    odds_upload = st.file_uploader(
        "Upcoming odds CSV (optional)",
        type=["csv"],
        help=(
            "Either full book feed with columns: market, point, price, home_team, away_team[,label] "
            "or template with: Matchup, Bookmaker Line, Over Price, Under Price, Home ML, Away ML[, Spread Line, Spread Price]."
        ),
    )
    run_btn = st.button("Train & Predict", type="primary", use_container_width=True)

if run_btn:
    if end_season < start_season:
        st.error("End season must be >= start season.")
        st.stop()
    seasons = list(range(int(start_season), int(end_season) + 1))
    closing_path = closing_path.strip() or None

    with st.spinner("Loading data, engineering features, training residual models..."):
        models, preds, diags, upcoming, features = run_pipeline(seasons, closing_path, n_splits=int(splits))

    # If user uploaded odds for upcoming games, overlay them and recompute predictions
    if odds_upload is not None:
        try:
            raw_odds = pd.read_csv(odds_upload)
            parsed_odds = parse_uploaded_odds(raw_odds)
            upcoming = attach_uploaded_odds_to_upcoming(upcoming, parsed_odds)
            if not upcoming.empty and "msg" not in upcoming.columns:
                preds = predict_upcoming(models, diags, features, upcoming, round_display=True)
        except Exception as e:
            st.warning(f"Failed to parse uploaded odds CSV: {e}")

    st.subheader("Out-of-fold diagnostics (where market present)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE (margin)", f"{diags.metrics['MAE_margin']:.2f}" if pd.notna(diags.metrics['MAE_margin']) else "—")
    c2.metric("MAE (home pts)", f"{diags.metrics['MAE_home']:.2f}" if pd.notna(diags.metrics['MAE_home']) else "—")
    c3.metric("MAE (away pts)", f"{diags.metrics['MAE_away']:.2f}" if pd.notna(diags.metrics['MAE_away']) else "—")
    c4.metric("MAE (total)", f"{diags.metrics['MAE_total']:.2f}" if pd.notna(diags.metrics['MAE_total']) else "—")
    c5.metric("Brier (win prob)", f"{diags.metrics['Brier_win']:.3f}" if pd.notna(diags.metrics['Brier_win']) else "—")

    st.markdown("**Features used (sent to LightGBM)**")
    st.code(", ".join(features), language="text")

    st.subheader("Upcoming predictions")
    if preds.empty or "msg" in upcoming.columns:
        st.info("No upcoming games detected or insufficient history/market data.")
    else:
        disp_cols = [
            "game_date","season","week","home_team","away_team",
            "pred_margin","pred_total","win_prob",
            "pred_home_pts","pred_away_pts",
            "home_lo80","home_hi80","away_lo80","away_hi80","total_lo80","total_hi80",
            "spread_line","total_line","blend_w",
            "home_prior","away_prior","home_gp","away_gp",
            "is_dome","is_turf","temp","wind","wind_12plus","wind_15plus",
            "pred_margin_model","pred_total_model","pred_margin_prior_mkt","pred_total_prior_mkt"
        ]
        df_show = preds.copy()
        if "game_date" in df_show.columns:
            df_show["game_date"] = pd.to_datetime(df_show["game_date"]).dt.tz_convert("UTC")
        for c in disp_cols:
            if c not in df_show.columns: df_show[c] = np.nan
        df_show = df_show[disp_cols].sort_values(["game_date","home_team"]).reset_index(drop=True)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "nfl_upcoming_predictions.csv", "text/csv", use_container_width=True)

    with st.expander("Advanced outputs (debug)"):
        if not preds.empty:
            st.dataframe(preds, use_container_width=True, hide_index=True)
else:
    st.info("Set your season range, optionally provide a closing-lines CSV, and click **Train & Predict**.")
