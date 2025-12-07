# -*- coding: utf-8 -*-
"""
NFL sides & totals predictor — productionized residual-on-close training.

Key upgrades:
- Historical closing lines ingestion (spread_close, total_close), flexible columns
- Residual modeling (actual - close); reconstruction for metrics/prediction
- Recency decay sample weights (0.97 ** weeks_ago)
- Huber objective to dampen blowouts
- Optional weather merge from nfl_data_py.import_weather
- Prediction-time anchoring from uploaded odds (lines -> anchors)
- Canonical game_id synthesis to make merges robust
- Wide/long odds normalization; team alias normalization
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
import io

try:
    import nfl_data_py  # type: ignore
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False


RANDOM_SEED = 42

# ----------------------------- Team code normalization ------------------- #
TEAM_ALIASES = {
    # Common nonstandard 3-letter variants
    "NEP": "NE", "GBP": "GB", "LVR": "LV", "ARZ": "ARI", "NOS": "NO", "TBB": "TB",
    # Historical/legacy & alt spellings
    "STL": "LAR", "SL": "LAR", "STLR": "LAR",
    "SD": "LAC", "SDC": "LAC",
    "OAK": "LV", "RAI": "LV",
    "JAX": "JAC", "JAGS": "JAC",
    "WSH": "WAS", "WFT": "WAS", "RED": "WAS",
    "LA": "LAR",  # ambiguous older 'LA' -> Rams by default
    # Typos / alt abbrevs occasionally seen
    "BALR": "BAL", "HST": "HOU", "CLV": "CLE", "ARZC": "ARI",
}

STANDARD_TEAMS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAC",
    "KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
}

def build_game_id(season, week, home_team, away_team) -> Optional[str]:
    try:
        if pd.isna(season) or pd.isna(week) or not home_team or not away_team:
            return None
        ht = normalize_team_code(home_team)
        at = normalize_team_code(away_team)
        if not ht or not at or ht not in STANDARD_TEAMS or at not in STANDARD_TEAMS:
            return None
        return f"{int(season)}-{int(week)}-{at}@{ht}"
    except Exception:
        return None

def normalize_team_code(x: str) -> str:
    """
    Normalize any team token to a standard NFL code expected by this app.
    Uppercases, applies alias mapping, and leaves unknowns uppercased.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    s = str(x).strip().upper()
    s = TEAM_ALIASES.get(s, s)
    FULLNAME_ALIASES = {
        "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
        "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
        "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
        "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAC","KANSAS CITY CHIEFS":"KC",
        "LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","LAS VEGAS RAIDERS":"LV","MIAMI DOLPHINS":"MIA",
        "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
        "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SEATTLE SEAHAWKS":"SEA",
        "SAN FRANCISCO 49ERS":"SF","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
        "WASHINGTON FOOTBALL TEAM":"WAS","WASHINGTON REDSKINS":"WAS","ST LOUIS RAMS":"LAR","SAN DIEGO CHARGERS":"LAC",
        "OAKLAND RAIDERS":"LV"
    }
    s = FULLNAME_ALIASES.get(s, s)
    return s

# ----------------------------- Robust tabular reader ---------------------- #
def _read_tabular(file_like) -> pd.DataFrame:
    """
    Robust reader for CSV or Excel-like uploads (including misnamed .csv that are actually xlsx/zip).
    """
    data_bytes = None
    if hasattr(file_like, "getvalue"):
        data_bytes = file_like.getvalue()
    elif isinstance(file_like, (bytes, bytearray)):
        data_bytes = bytes(file_like)
    else:
        try:
            data_bytes = file_like.read()
            if hasattr(file_like, "seek"):
                file_like.seek(0)
        except Exception:
            data_bytes = None

    if data_bytes is not None and len(data_bytes) >= 2 and data_bytes[:2] == b"PK":
        try:
            return pd.read_excel(io.BytesIO(data_bytes))
        except Exception:
            return pd.read_excel(io.BytesIO(data_bytes), engine="openpyxl")
    name = getattr(file_like, "name", "") or ""
    if name.lower().endswith((".xlsx",".xls",".xlsm",".xlsb")) and data_bytes is not None:
        return pd.read_excel(io.BytesIO(data_bytes))

    encodings = ["utf-8", "cp1252", "latin-1", "utf-16", "utf-16le", "utf-16be"]
    last_err = None
    for enc in encodings:
        try:
            if data_bytes is not None:
                return pd.read_csv(io.BytesIO(data_bytes), encoding=enc, engine="python")
            return pd.read_csv(file_like, encoding=enc, engine="python")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not parse uploaded file as CSV or Excel. Last error: {last_err}")

def _standardize_odds_columns(odds_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common sportsbook/export schemas into a single-row-per-game summary:

    Output columns (if available):
      - game_id (optional)
      - season, week, home_team, away_team (optional; used to build game_id)
      - ml               (home moneyline)
      - spread_line      (home spread; negative means home favored)
      - spread_odds      (odds for home spread)
      - total_line       (game total)
      - total_odds       (odds for OVER)
    """
    df = odds_raw.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    def _pick_col_case_insensitive(*names):
        for nm in names:
            if nm in cols_lower:
                return cols_lower[nm]
        return None

    def _pick_first_numeric_col_with_substrings(required_substrings, forbidden_substrings=None):
        forbidden_substrings = forbidden_substrings or []
        cand_close = None
        cand_any = None
        for c in df.columns:
            name = c.lower()
            if all(sub in name for sub in required_substrings) and not any(bad in name for bad in forbidden_substrings):
                try:
                    ser = pd.to_numeric(df[c], errors="coerce")
                    if ser.notna().sum() > 0:
                        if "close" in name and cand_close is None:
                            cand_close = c
                        if cand_any is None:
                            cand_any = c
                except Exception:
                    continue
        return cand_close or cand_any

    cols = cols_lower

    # id/keys
    id_cols = {}
    for cand in ["game_id", "gid", "id"]:
        if cand in cols:
            id_cols["game_id"] = cols[cand]
            break
    for cand in ["season","year"]:
        if cand in cols:
            id_cols["season"] = cols[cand]; break
    if "week" in cols:
        id_cols["week"] = cols["week"]
    home_keys = [k for k in ["home_team","home","home_abbr","home_abbreviation","team_home","team_h","home_name"] if k in cols]
    away_keys = [k for k in ["away_team","away","visitor","vis","away_abbr","away_abbreviation","team_away","team_a","away_name"] if k in cols]
    if home_keys: id_cols["home_team"] = cols[home_keys[0]]
    if away_keys: id_cols["away_team"] = cols[away_keys[0]]

    # -------- historical-close style (wide, lines only) --------
    def pick_ci(*names):
        return _pick_col_case_insensitive(*names)

    sp_close_col = pick_ci("spread_close","closing_spread","close_spread","home_spread_close","home_close_spread","spread_line_close","home_spread")
    to_close_col = pick_ci("total_close","closing_total","close_total","total_line_close","total","ou_close","o_u_close","over_under","ou")

    if sp_close_col is None:
        sp_close_col = _pick_first_numeric_col_with_substrings(["spread"], ["odds","price","american"])
    if to_close_col is None:
        to_close_col = _pick_first_numeric_col_with_substrings(["total"], ["odds","price","american"]) \
                       or _pick_first_numeric_col_with_substrings(["ou"], ["odds","price","american"])

    if sp_close_col is not None or to_close_col is not None:
        out = pd.DataFrame()
        if "game_id" in id_cols:
            out["game_id"] = df[id_cols["game_id"]].astype(str)
        if "season" in id_cols:
            out["season"] = pd.to_numeric(df[id_cols["season"]], errors="coerce").astype("Int64")
        if "week" in id_cols:
            out["week"] = pd.to_numeric(df[id_cols["week"]], errors="coerce").astype("Int64")
        if "home_team" in id_cols:
            df[id_cols["home_team"]] = df[id_cols["home_team"]].apply(normalize_team_code)
            out["home_team"] = df[id_cols["home_team"]]
        if "away_team" in id_cols:
            df[id_cols["away_team"]] = df[id_cols["away_team"]].apply(normalize_team_code)
            out["away_team"] = df[id_cols["away_team"]]
        out["ml"] = np.nan
        out["spread_line"] = pd.to_numeric(df[sp_close_col], errors="coerce") if sp_close_col else np.nan
        out["spread_odds"] = np.nan
        out["total_line"] = pd.to_numeric(df[to_close_col], errors="coerce") if to_close_col else np.nan
        out["total_odds"] = np.nan
        return out

    # -------- long format --------
    long_market = any(k in cols for k in ["market","bet_type","type","bettype"])
    long_label  = any(k in cols for k in ["label","side","selection","runner","competitor","team"])
    long_price  = any(k in cols for k in ["price","odds","american","american_odds"])
    long_point  = any(k in cols for k in ["point","line","handicap","points","number"])

    if long_market and long_label and long_price:
        mcol = cols.get("market") or cols.get("bet_type") or cols.get("type") or cols.get("bettype")
        lcol = cols.get("label") or cols.get("side") or cols.get("selection") or cols.get("runner") or cols.get("competitor") or cols.get("team")
        pcol = cols.get("price") or cols.get("odds") or cols.get("american") or cols.get("american_odds")
        ptcol = cols.get("point") or cols.get("line") or cols.get("handicap") or cols.get("points") or cols.get("number")

        g = df.copy()
        for c in [lcol]:
            g[c] = g[c].astype(str).str.strip()
        g["_market_norm"] = g[mcol].astype(str).str.lower().str.replace(r"[^a-z]+","",regex=True)
        g["_sel_raw"] = g[lcol].astype(str)
        g["_sel_norm"] = g[lcol].astype(str).str.lower().str.replace(r"[^a-z]+","",regex=True)
        g["_sel_code"] = g["_sel_raw"].apply(normalize_team_code)

        for k_out, k_in in id_cols.items():
            g[k_out] = g[k_in]

        def pick_ml(grp):
            ml = np.nan
            sub = grp[(grp["_market_norm"].isin(["moneyline","ml"])) | (grp["_market_norm"].str.contains("moneyline"))]
            if not sub.empty:
                home_row = sub[sub["_sel_norm"].isin(["home","hometeam"])]
                if not home_row.empty:
                    try: return float(home_row.iloc[0][pcol])
                    except Exception: pass
                home_code = normalize_team_code(grp["home_team"].iloc[0]) if "home_team" in grp else None
                if home_code:
                    for _, r in sub.iterrows():
                        if normalize_team_code(r["_sel_raw"]) == home_code or str(r.get("_sel_code","")) == home_code:
                            try: return float(r[pcol])
                            except Exception: pass
            return ml

        def pick_spread(grp):
            sp_line = np.nan; sp_odds = np.nan
            sub = grp[g["_market_norm"].isin(["spreads","spread","handicap"])]
            if sub.empty:
                sub = grp[g["_market_norm"].str.contains("spread")]
            if not sub.empty:
                home_code = normalize_team_code(grp["home_team"].iloc[0]) if "home_team" in grp else None
                cands = sub[sub["_sel_norm"].isin(["home","hometeam"])]
                if cands.empty and home_code:
                    mask_name = [normalize_team_code(x) == home_code for x in sub["_sel_raw"]]
                    mask_code = [str(y) == home_code for y in sub.get("_sel_code", pd.Series(index=sub.index, dtype=object))]
                    cands = sub[[a or b for a,b in zip(mask_name, mask_code)]]
                if not cands.empty:
                    try:
                        sp_line = float(cands.iloc[0][ptcol]) if ptcol in sub.columns else np.nan
                        sel = str(cands.iloc[0]["_sel_norm"])
                        if sel in ["away","awayteam"] and not np.isnan(sp_line):
                            sp_line = -sp_line
                        sp_odds = float(cands.iloc[0][pcol])
                    except Exception:
                        pass
                else:
                    away = sub[sub["_sel_norm"].isin(["away","awayteam"])]
                    if not away.empty:
                        try:
                            an = float(away.iloc[0][ptcol]) if ptcol in sub.columns else np.nan
                            sp_line = -an if not np.isnan(an) else np.nan
                            sp_odds = np.nan if pd.isna(away.iloc[0][pcol]) else float(away.iloc[0][pcol])
                        except Exception:
                            pass
            return sp_line, sp_odds

        def pick_total(grp):
            to_line = np.nan; to_odds = np.nan
            sub = grp[g["_market_norm"].isin(["totals","total","overunder","ou"])]
            if sub.empty:
                sub = grp[g["_market_norm"].str.contains("total")]
            if not sub.empty:
                over = sub[sub["_sel_norm"].isin(["over"])]
                if not over.empty:
                    try:
                        to_line = float(over.iloc[0][ptcol]) if ptcol in sub.columns else np.nan
                        to_odds = float(over.iloc[0][pcol])
                    except Exception:
                        pass
                else:
                    under = sub[sub["_sel_norm"].isin(["under"])]
                    if not under.empty:
                        try:
                            to_line = float(under.iloc[0][ptcol]) if ptcol in sub.columns else np.nan
                            to_odds = np.nan
                        except Exception:
                            pass
            return to_line, to_odds

        summaries = []
        group_keys = []
        if "game_id" in id_cols:
            group_keys.append("game_id")
        for key in ["season","week","home_team","away_team"]:
            if key in id_cols:
                group_keys.append(key)
        if not group_keys:
            group_keys = []

        if group_keys:
            for _, grp in g.groupby(group_keys, dropna=False):
                ml = pick_ml(grp)
                sp_line, sp_odds = pick_spread(grp)
                to_line, to_odds = pick_total(grp)
                rec = {}
                for k in group_keys: rec[k] = grp.iloc[0][k]
                if "home_team" in rec: rec["home_team"] = normalize_team_code(rec["home_team"])
                if "away_team" in rec: rec["away_team"] = normalize_team_code(rec["away_team"])
                rec.update({"ml": ml, "spread_line": sp_line, "spread_odds": sp_odds, "total_line": to_line, "total_odds": to_odds})
                summaries.append(rec)
            return pd.DataFrame(summaries)

    # -------- wide format --------
    def pick_col(*names):
        for nm in names:
            if nm in cols:
                return cols[nm]
        return None

    ml_col = pick_col("home_ml","home_moneyline","ml_home","moneyline_home","homeprice","home_odds","moneyline_h")
    sp_line_col = pick_col(
        "home_spread_line","home_spread","spread_home","handicap_home","linespread_home","spreadh",
        "home_spread_close","home_close_spread","spread_line_close"
    )
    sp_odds_col = pick_col("home_spread_odds","spread_home_odds","odds_spread_home","home_spreadprice","spreadodds_h")
    away_sp_line_col = pick_col("away_spread_line","away_spread","spread_away","handicap_away","linespread_away","spreada")
    away_sp_odds_col = pick_col("away_spread_odds","spread_away_odds","odds_spread_away","away_spreadprice","spreadodds_a")
    tot_line_col = pick_col(
        "total_line","total","over_under","ou","total_points","points_total","game_total",
        "total_close","closing_total","close_total","total_line_close"
    )
    over_odds_col = pick_col("over_odds","total_over_odds","odds_over","overprice","overprice_american")

    if not tot_line_col:
        tot_line_col = _pick_first_numeric_col_with_substrings(["total"], ["odds","price","american"]) \
                       or _pick_first_numeric_col_with_substrings(["ou"], ["odds","price","american"])

    if not sp_line_col and not away_sp_line_col:
        generic_spread = _pick_first_numeric_col_with_substrings(["spread"], ["odds","price","american"])
        if generic_spread:
            sp_line_col = generic_spread

    if any([ml_col, sp_line_col, away_sp_line_col, tot_line_col]):
        out = pd.DataFrame()
        if "game_id" in id_cols: out["game_id"] = df[id_cols["game_id"]].astype(str)
        if "season" in id_cols:  out["season"] = pd.to_numeric(df[id_cols["season"]], errors="coerce").astype("Int64")
        if "week" in id_cols:    out["week"]   = pd.to_numeric(df[id_cols["week"]], errors="coerce").astype("Int64")
        if "home_team" in id_cols: out["home_team"] = df[id_cols["home_team"]].apply(normalize_team_code)
        if "away_team" in id_cols: out["away_team"] = df[id_cols["away_team"]].apply(normalize_team_code)

        out["ml"] = pd.to_numeric(df[ml_col], errors="coerce") if ml_col else np.nan
        if sp_line_col:
            out["spread_line"] = pd.to_numeric(df[sp_line_col], errors="coerce")
            out["spread_odds"] = pd.to_numeric(df[sp_odds_col], errors="coerce") if sp_odds_col else np.nan
        elif away_sp_line_col:
            inv = pd.to_numeric(df[away_sp_line_col], errors="coerce")
            out["spread_line"] = -inv
            out["spread_odds"] = np.nan if not away_sp_odds_col else pd.to_numeric(df[away_sp_odds_col], errors="coerce")
        else:
            out["spread_line"] = np.nan; out["spread_odds"] = np.nan

        out["total_line"] = pd.to_numeric(df[tot_line_col], errors="coerce") if tot_line_col else np.nan
        out["total_odds"] = pd.to_numeric(df[over_odds_col], errors="coerce") if over_odds_col else np.nan

        return out

    # -------- passthrough known schema --------
    if {"game_id","ml","spread_line","spread_odds","total_line","total_odds"}.issubset(df.columns):
        return df[["game_id","ml","spread_line","spread_odds","total_line","total_odds"]].copy()

    ident_ok = any(k in df.columns for k in ["game_id"]) or all(k in df.columns for k in ["season","week","home_team","away_team"])
    has_close_like = any("spread" in c.lower() for c in df.columns) and any(("total" in c.lower()) or ("ou" in c.lower()) for c in df.columns)
    if ident_ok and has_close_like:
        sp_guess = _pick_first_numeric_col_with_substrings(["spread"], ["odds","price","american"]) or _pick_col_case_insensitive("spread","spread_close")
        to_guess = _pick_first_numeric_col_with_substrings(["total"], ["odds","price","american"]) or _pick_col_case_insensitive("total","total_close","ou")
        if sp_guess or to_guess:
            out = pd.DataFrame()
            if "game_id" in df.columns: out["game_id"] = df["game_id"].astype(str)
            for k in ["season","week","home_team","away_team"]:
                if k in df.columns: out[k] = df[k]
            out["ml"] = np.nan
            out["spread_line"] = pd.to_numeric(df[sp_guess], errors="coerce") if sp_guess else np.nan
            out["spread_odds"] = np.nan
            out["total_line"] = pd.to_numeric(df[to_guess], errors="coerce") if to_guess else np.nan
            out["total_odds"] = np.nan
            if "home_team" in out.columns: out["home_team"] = out["home_team"].apply(normalize_team_code)
            if "away_team" in out.columns: out["away_team"] = out["away_team"].apply(normalize_team_code)
            return out

    raise ValueError(f"Unrecognized odds format. Columns seen: {list(df.columns)}")

def _summarize_anchor_cols(df: pd.DataFrame, prefix: str = "") -> str:
    parts = []
    n = len(df)
    for col, label in [("feat_spread_close","spread_close"), ("feat_total_close","total_close")]:
        if col in df.columns:
            cnt = int(df[col].notna().sum())
            pct = (100.0 * cnt / n) if n else 0.0
            parts.append(f"{label}={cnt} ({pct:.1f}%)")
    s = " • ".join(parts) if parts else "no anchors"
    return (prefix + " " + s).strip()

# ----------------------------- Synthetic data ----------------------------- #
def generate_synthetic_data(seasons: Iterable[int]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    teams = [f"T{idx:02d}" for idx in range(16)]
    rows: List[dict] = []
    gid = 0
    for season in seasons:
        for i in range(40):
            week = i // 4 + 1
            h = rng.choice(teams); a = rng.choice([t for t in teams if t != h])
            h_off = float(rng.normal(50, 10)); a_off = float(rng.normal(50, 10))
            h_def = float(rng.normal(50, 10)); a_def = float(rng.normal(50, 10))
            h_rest = int(rng.integers(3, 10)); a_rest = int(rng.integers(3, 10))
            wind = float(rng.integers(0, 21)); precip = int(rng.integers(0, 2))
            temp = float(rng.integers(30, 91)); dome = int(rng.integers(0, 2))
            month = (week * 7) % 12 + 1; day = int(rng.integers(1, 29))
            kickoff = pd.Timestamp(f"{season}-{month:02d}-{day:02d} 13:00")
            rating_diff = (h_off - a_off) + (a_def - h_def)
            off_sum = (h_off + a_off); def_sum = (h_def + a_def)
            margin = 0.5 * rating_diff + rng.normal(0.0, 3.0)
            total  = 40.0 + 0.5 * (off_sum - def_sum) + rng.normal(0.0, 2.0)
            hp = (total + margin) / 2.0
            ap = (total - margin) / 2.0
            rows.append(dict(
                game_id=f"{season}-{gid}", season=season, week=week, kickoff=kickoff,
                home_team=h, away_team=a,
                home_off_rating=h_off, away_off_rating=a_off,
                home_def_rating=h_def, away_def_rating=a_def,
                home_rest=h_rest, away_rest=a_rest,
                wind_mph=wind, precip_flag=precip, temp_f=temp, is_dome=dome,
                home_points=hp, away_points=ap
            ))
            gid += 1
    df = pd.DataFrame(rows).sort_values(["season","week","game_id"]).reset_index(drop=True)
    return df

# ----------------------------- Real data loader --------------------------- #
def load_real_data(seasons: Iterable[int], include_scores: bool = True) -> pd.DataFrame:
    if not HAS_NFL_DATA_PY:
        raise RuntimeError("nfl_data_py is not installed; cannot load real data")
    raw = nfl_data_py.import_schedules(list(seasons))

    try:
        wx = nfl_data_py.import_weather(list(seasons))
        wx = wx.rename(columns={"wind": "wind_mph","temperature": "temp_f","is_indoor": "is_dome"})
        keep = [c for c in ["game_id","wind_mph","temp_f","is_dome"] if c in wx.columns]
        if keep:
            raw = raw.merge(wx[keep], on="game_id", how="left")
    except Exception:
        pass

    score_cols = {
        "home_score": "home_points",
        "away_score": "away_points",
        "score_home": "home_points",
        "score_away": "away_points",
    }
    for src, dst in score_cols.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})

    date_col = None
    for cand in ["kickoff", "game_date", "gameday", "start_time", "start_time_et", "datetime", "date"]:
        if cand in raw.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("Could not find a kickoff/date column in schedule data")

    df = raw[["game_id","season","week","home_team","away_team","home_points","away_points",date_col]].copy()
    df["kickoff"] = pd.to_datetime(df[date_col])
    df = df.drop(columns=[date_col])
    if include_scores:
        df = df.dropna(subset=["home_points","away_points"])
    return df

# ----------------------------- Training close helper ---------------------- #
def _prepare_training_close_lines(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts historical closing lines with flexible schemas.
    Preferred join: game_id; fallback: season+week+home_team+away_team.
    """
    df = close_df.copy()

    def norm(s: str) -> str:
        return "".join(ch.lower() for ch in str(s) if ch.isalnum() or ch == "_").replace("__","_")
    colmap = {c: norm(c) for c in df.columns}
    inv = {v: k for k, v in colmap.items()}

    gid_col = next((inv[k] for k in ["gameid","gid","id","game_id"] if k in inv), None)
    season_col = next((inv[k] for k in ["season","year"] if k in inv), None)
    week_col   = next((inv[k] for k in ["week","wk"] if k in inv), None)

    def find_team(prefixes):
        for p in prefixes:
            for cand in [f"{p}", f"{p}team", f"{p}abbr", f"{p}abbreviation", f"{p}name", f"team_{p}"]:
                if cand in inv:
                    return inv[cand]
        return None
    home_col = find_team(["home","h"])
    away_col = find_team(["away","a","visitor","vis"])

    def pick(cands):
        for k in cands:
            if k in inv: return inv[k]
        for raw, nm in colmap.items():
            if any(c in nm for c in cands):
                return raw
        return None

    spread_candidates = ["spreadclose","closingspread","closespread","spread","homeclosespread","homespreadclose","spreadlineclose"]
    total_candidates  = ["totalclose","closingtotal","closetotal","total","ouclose","o_u_close","totallineclose"]

    spread_col = pick(spread_candidates)
    total_col  = pick(total_candidates)

    out = pd.DataFrame()
    if gid_col is not None:
        out["game_id"] = df[gid_col].astype(str)
    else:
        if not (season_col and week_col and home_col and away_col):
            raise ValueError("Need either game_id or season+week+home_team+away_team to merge close lines.")
        out["season"] = pd.to_numeric(df[season_col], errors="coerce").astype("Int64")
        out["week"]   = pd.to_numeric(df[week_col], errors="coerce").astype("Int64")
        out["home_team"] = df[home_col].apply(normalize_team_code).astype(str)
        out["away_team"] = df[away_col].apply(normalize_team_code).astype(str)

    if spread_col is not None:
        out["feat_spread_close"] = pd.to_numeric(df[spread_col], errors="coerce")
    if total_col is not None:
        out["feat_total_close"]  = pd.to_numeric(df[total_col], errors="coerce")

    if "game_id" not in out.columns and all(k in out.columns for k in ["season","week","home_team","away_team"]):
        out["game_id"] = [
            build_game_id(s, w, h, a)
            for s, w, h, a in zip(out["season"], out["week"], out["home_team"], out["away_team"])
        ]

    if "feat_spread_close" not in out.columns and "feat_total_close" not in out.columns:
        raise ValueError("Could not find close line columns for spread/total.")
    return out

# ----------------------------- Ratings & geo ------------------------------ #
def compute_team_ratings(df: pd.DataFrame, alpha: float = 5.0) -> pd.DataFrame:
    records = []
    for season in sorted(df["season"].unique()):
        hist = df[(df["season"] == season) & df["home_points"].notna() & df["away_points"].notna()]
        if hist.empty: continue
        teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        X = np.zeros((len(hist), len(teams)))
        y = hist["home_points"].values - hist["away_points"].values
        for r, (_, row) in enumerate(hist.iterrows()):
            X[r, idx[row["home_team"]]] = 1.0
            X[r, idx[row["away_team"]]] = -1.0
        X = X - X.mean(axis=0, keepdims=True)
        model = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
        for t, i in idx.items():
            records.append({"season": season, "team": t, "rating_bt": float(model.coef_[i])})
    return pd.DataFrame(records)

NFL_TEAM_LATLON = {
    "ARI": (33.527, -112.262), "ATL": (33.755, -84.401), "BAL": (39.278, -76.622),
    "BUF": (42.773, -78.787), "CAR": (35.225, -80.852), "CHI": (41.862, -87.616),
    "CIN": (39.095, -84.516), "CLE": (41.506, -81.699), "DAL": (32.747, -97.094),
    "DEN": (39.743, -105.020), "DET": (42.340, -83.046), "GB": (44.501, -88.062),
    "HOU": (29.684, -95.410), "IND": (39.760, -86.163), "JAX": (30.323, -81.639),
    "KC": (39.049, -94.484), "LAC": (33.864, -118.261), "LAR": (34.014, -118.287),
    "LV": (36.090, -115.183), "MIA": (25.958, -80.238), "MIN": (44.974, -93.258),
    "NE": (42.091, -71.264), "NO": (29.951, -90.081), "NYG": (40.813, -74.074),
    "NYJ": (40.813, -74.074), "PHI": (39.901, -75.167), "PIT": (40.446, -80.015),
    "SEA": (47.595, -122.331), "SF": (37.403, -121.969), "TB": (27.975, -82.503),
    "TEN": (36.166, -86.771), "WAS": (38.907, -77.007)
}

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_km = 6371.0
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return 0.621371 * R_km * c

def compute_elo_per_game(df: pd.DataFrame,
                         k_base: float = 20.0,
                         margin_scale: float = 400.0,
                         home_adv: float = 45.0,
                         carryover: float = 0.75) -> pd.DataFrame:
    rows = []
    df_sorted = df.sort_values(["season","kickoff"]).reset_index(drop=True)
    elo = {}; season_last = {}; DEFAULT = 1500.0
    for season in sorted(df_sorted["season"].dropna().unique()):
        teams = sorted(set(df_sorted.loc[df_sorted["season"] == season, "home_team"])
                       .union(set(df_sorted.loc[df_sorted["season"] == season, "away_team"])))
        for t in teams:
            elo[t] = carryover * season_last.get(t, DEFAULT) + (1-carryover)*DEFAULT
        seas = df_sorted[df_sorted["season"] == season]
        for _, g in seas.iterrows():
            h, a = g["home_team"], g["away_team"]
            mh = float(g.get("home_points")) if pd.notna(g.get("home_points")) else np.nan
            ma = float(g.get("away_points")) if pd.notna(g.get("away_points")) else np.nan
            e_h = elo.get(h, DEFAULT); e_a = elo.get(a, DEFAULT)
            rows.append({"game_id": g["game_id"], "team": h, "elo_pre": e_h})
            rows.append({"game_id": g["game_id"], "team": a, "elo_pre": e_a})
            if not (pd.notna(mh) and pd.notna(ma)): continue
            diff = (e_h + home_adv) - e_a
            exp_h = 1.0 / (1.0 + 10.0 ** (-diff / margin_scale))
            s_h = 1.0 if mh > ma else (0.5 if mh == ma else 0.0); s_a = 1.0 - s_h
            mov = abs(mh - ma); k = k_base * (1 + min(mov, 28) / 28.0)
            elo[h] = e_h + k * (s_h - exp_h); elo[a] = e_a + k * (s_a - (1 - exp_h))
        for t in teams:
            season_last[t] = elo.get(t, DEFAULT)
    return pd.DataFrame(rows)

# ----------------------------- Feature builder ---------------------------- #
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy().reset_index(drop=True)
    df["margin"] = df["home_points"] - df["away_points"]
    df["total"]  = df["home_points"] + df["away_points"]
    df["home_win"] = (df["margin"] > 0).astype(int)

    try:
        bt = compute_team_ratings(df)
    except Exception:
        bt = pd.DataFrame(columns=["season","team","rating_bt"])

    # Synthetic branch
    if {"home_off_rating","away_off_rating","home_def_rating","away_def_rating","home_rest","away_rest"}.issubset(df.columns):
        df["feat_rating_diff"] = (df["home_off_rating"] - df["away_off_rating"]) + (df["away_def_rating"] - df["home_def_rating"])
        df["feat_off_sum"] = df["home_off_rating"] + df["away_off_rating"]
        df["feat_def_sum"] = df["home_def_rating"] + df["away_def_rating"]
        df["feat_rest_diff"] = df["home_rest"] - df["away_rest"]
        for col in ["wind_mph","precip_flag","temp_f","is_dome"]:
            if col in df.columns: df[f"feat_{col}"] = df[col]
        if not bt.empty:
            bt_home = bt.rename(columns={"team": "home_team", "rating_bt": "_r_h", "season": "_s"})
            df = df.merge(
                bt_home[["home_team", "_r_h", "_s"]],
                on=["home_team"],
                how="left",
            )

            bt_away = bt.rename(columns={"team": "away_team", "rating_bt": "_r_a", "season": "_s"})
            df = df.merge(
                bt_away[["away_team", "_r_a", "_s"]],
                left_on=["away_team", "_s"],
                right_on=["away_team", "_s"],
                how="left",
            )

            df["feat_bt_rating_diff"] = (
                df["_r_h"].fillna(0.0) - df["_r_a"].fillna(0.0)
            )
            df.drop(columns=[c for c in ["_r_h", "_r_a", "_s"] if c in df.columns], inplace=True)
        else:
            df["feat_bt_rating_diff"] = 0.0
        df["feat_form_x_rating"] = (df["feat_off_sum"] - df["feat_def_sum"]) + 0.3 * df["feat_bt_rating_diff"]
        df["feat_total_tendency"] = df["feat_off_sum"] - 0.5 * df["feat_def_sum"]
        df["feat_elo_diff"] = df["feat_bt_rating_diff"] * 10.0
        df["feat_travel_miles_away"] = 500.0
        df["feat_short_week"] = (df["feat_rest_diff"] < 3).astype(int)
        df["feat_bye_week"] = (df["feat_rest_diff"] > 9).astype(int)

        for c in ["feat_spread_close","feat_total_close"]:
            if c in raw_df.columns and c not in df.columns: df[c] = raw_df[c]

        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        return df[["game_id","season","week","kickoff","home_team","away_team"] + feature_cols + ["margin","total","home_win"]]

    # Real-data branch
    df = df.sort_values("kickoff").reset_index(drop=True)
    team_stats = {}; pf_d, pa_d, rest_d = [], [], []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        for t in [h, a]:
            if t not in team_stats: team_stats[t] = {"pf": [], "pa": [], "last": None}
        def avg_last(vals, n=5): return float(np.mean(vals[-n:])) if vals else 0.0
        h_pf = avg_last(team_stats[h]["pf"]); h_pa = avg_last(team_stats[h]["pa"])
        a_pf = avg_last(team_stats[a]["pf"]); a_pa = avg_last(team_stats[a]["pa"])
        pf_d.append(h_pf - a_pf); pa_d.append(h_pa - a_pa)
        last_h = team_stats[h]["last"]; last_a = team_stats[a]["last"]
        rest_h = (row["kickoff"] - last_h).days if isinstance(last_h, pd.Timestamp) else 7.0
        rest_a = (row["kickoff"] - last_a).days if isinstance(last_a, pd.Timestamp) else 7.0
        rest_d.append(rest_h - rest_a)
        hp, ap = row.get("home_points"), row.get("away_points")
        if pd.notna(hp) and pd.notna(ap):
            team_stats[h]["pf"].append(float(hp)); team_stats[h]["pa"].append(float(ap))
            team_stats[a]["pf"].append(float(ap)); team_stats[a]["pa"].append(float(hp))
        team_stats[h]["last"] = row["kickoff"]; team_stats[a]["last"] = row["kickoff"]

    df["feat_points_for_diff"] = pf_d
    df["feat_points_against_diff"] = pa_d
    df["feat_rest_diff"] = rest_d

    if not bt.empty:
        # home side ratings by season
        bt_home = bt.rename(
            columns={"team": "home_team", "rating_bt": "_r_h", "season": "_s"}
        )
        df = df.merge(
            bt_home[["home_team", "_r_h", "_s"]],
            left_on=["home_team", "season"],
            right_on=["home_team", "_s"],
            how="left",
        )

        # away side ratings by season
        bt_away = bt.rename(
            columns={"team": "away_team", "rating_bt": "_r_a", "season": "_s"}
        )
        df = df.merge(
            bt_away[["away_team", "_r_a", "_s"]],
            left_on=["away_team", "season"],
            right_on=["away_team", "_s"],
            how="left",
        )

        df["feat_bt_rating_diff"] = (
            df["_r_h"].fillna(0.0) - df["_r_a"].fillna(0.0)
        )
        df.drop(columns=[c for c in ["_r_h", "_r_a", "_s"] if c in df.columns], inplace=True)
    else:
        df["feat_bt_rating_diff"] = 0.0

    try:
        elo_df = compute_elo_per_game(df)
        elo_h = elo_df.rename(columns={"team":"home_team","elo_pre":"_elo_h"})
        elo_a = elo_df.rename(columns={"team":"away_team","elo_pre":"_elo_a"})
        df = df.merge(elo_h[["game_id","home_team","_elo_h"]], on=["game_id","home_team"], how="left")
        df = df.merge(elo_a[["game_id","away_team","_elo_a"]], on=["game_id","away_team"], how="left")
        df["_elo_h"] = df["_elo_h"].fillna(1500.0); df["_elo_a"] = df["_elo_a"].fillna(1500.0)
        df["feat_elo_diff"] = df["_elo_h"] - df["_elo_a"]
        df.drop(columns=[c for c in ["_elo_h","_elo_a"] if c in df.columns], inplace=True)
    except Exception:
        df["feat_elo_diff"] = 0.0

    travel_miles = []
    for _, r in df.iterrows():
        hl = NFL_TEAM_LATLON.get(str(r["home_team"])); al = NFL_TEAM_LATLON.get(str(r["away_team"]))
        travel_miles.append(_haversine_miles(al[0], al[1], hl[0], hl[1]) if (hl and al) else 0.0)
    df["feat_travel_miles_away"] = travel_miles
    df["feat_short_week"] = (df["feat_rest_diff"] < 6).astype(int)
    df["feat_bye_week"]   = (df["feat_rest_diff"] > 13).astype(int)

    for c in ["feat_spread_close","feat_total_close"]:
        if c in raw_df.columns and c not in df.columns: df[c] = raw_df[c]

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    return df[["game_id","season","week","kickoff","home_team","away_team"] + feature_cols + ["margin","total","home_win"]]

def get_feature_columns(feature_df: pd.DataFrame) -> List[str]:
    return [c for c in feature_df.columns if c.startswith("feat_")]

# Helper: align features for prediction to model's trained feature names
def _align_features_for_predict(model: LGBMRegressor, X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    LightGBM is strict about feature count + order.
    This builds a DataFrame whose columns EXACTLY match what the model was fit on.

    Behavior:
    - Use model.feature_name_ if present (sklearn API attr after fit).
    - Fallback to model.booster_.feature_name(), which returns a list.
    - Any missing columns in X_raw are added with 0.0.
    - Any extra columns in X_raw are dropped.
    - Column order is forced.
    """
    # 1. get trained feature names
    trained_feats = None
    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        trained_feats = list(model.feature_name_)
    else:
        try:
            trained_feats = list(model.booster_.feature_name())
        except Exception:
            trained_feats = list(X_raw.columns)

    # 2. build aligned frame
    aligned = {}
    for col in trained_feats:
        if col in X_raw.columns:
            aligned[col] = X_raw[col].astype(float)
        else:
            aligned[col] = 0.0  # unseen at inference -> 0 fallback
    aligned_df = pd.DataFrame(aligned, index=X_raw.index)

    # 3. ensure no NaN leakage
    aligned_df = aligned_df.fillna(0.0)

    return aligned_df

# ----------------------------- Upcoming games feats ----------------------- #
def compute_team_stats_for_real(df: pd.DataFrame) -> Tuple[dict, float, float]:
    df_sorted = df.sort_values("kickoff").reset_index(drop=True)
    team_stats: dict[str, dict] = {}
    all_pf: List[float] = []; all_pa: List[float] = []
    for _, row in df_sorted.iterrows():
        h, a = row["home_team"], row["away_team"]
        for t in [h, a]:
            if t not in team_stats:
                team_stats[t] = {"points_for": [], "points_against": [], "last_kickoff": None, "avg_pf": 0.0, "avg_pa": 0.0}
        team_stats[h]["last_kickoff"] = row["kickoff"]; team_stats[a]["last_kickoff"] = row["kickoff"]
        hp, ap = row.get("home_points"), row.get("away_points")
        if pd.notna(hp) and pd.notna(ap):
            team_stats[h]["points_for"].append(float(hp)); team_stats[h]["points_against"].append(float(ap))
            team_stats[a]["points_for"].append(float(ap)); team_stats[a]["points_against"].append(float(hp))
            all_pf.extend([float(hp), float(ap)]); all_pa.extend([float(ap), float(hp)])
    league_pf = float(np.mean(all_pf)) if all_pf else 0.0
    league_pa = float(np.mean(all_pa)) if all_pa else 0.0
    for t, stats in team_stats.items():
        pf = stats["points_for"]; pa = stats["points_against"]
        stats["avg_pf"] = float(np.mean(pf)) if pf else league_pf
        stats["avg_pa"] = float(np.mean(pa)) if pa else league_pa
    return team_stats, league_pf, league_pa

def build_features_for_upcoming_games(df_pred: pd.DataFrame, team_stats: dict, league_pf: float, league_pa: float) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in df_pred.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs = team_stats.get(h, {}); as_ = team_stats.get(a, {})
        h_pf = hs.get("avg_pf", league_pf); a_pf = as_.get("avg_pf", league_pf)
        h_pa = hs.get("avg_pa", league_pa); a_pa = as_.get("avg_pa", league_pa)
        pf_diff = h_pf - a_pf; pa_diff = h_pa - a_pa
        kickoff_ts = row["kickoff"]
        last_h = hs.get("last_kickoff"); last_a = as_.get("last_kickoff")
        rest_h = (kickoff_ts - last_h).days if isinstance(last_h, pd.Timestamp) else 7.0
        rest_a = (kickoff_ts - last_a).days if isinstance(last_a, pd.Timestamp) else 7.0
        rows.append({
            "game_id": row["game_id"], "season": row["season"], "week": row["week"], "kickoff": row["kickoff"],
            "home_team": h, "away_team": a,
            "feat_points_for_diff": pf_diff, "feat_points_against_diff": pa_diff,
            "feat_rest_diff": float(rest_h - rest_a),
            "margin": np.nan, "total": np.nan, "home_win": np.nan
        })
    return pd.DataFrame(rows)

# ----------------------------- Training & calibration --------------------- #
def train_models(feat_df: pd.DataFrame, seasons: Iterable[int]) -> Tuple[LGBMRegressor, LGBMRegressor, IsotonicRegression, float, float, dict, dict]:
    feature_cols = get_feature_columns(feat_df)
    years_sorted = sorted(set(int(s) for s in seasons))
    if len(years_sorted) < 3:
        try: st.warning("Training on fewer than 3 seasons – expect unstable validation and MAE.")
        except Exception: pass

    has_spread_close = "feat_spread_close" in feat_df.columns and feat_df["feat_spread_close"].notna().any()
    has_total_close  = "feat_total_close"  in feat_df.columns and feat_df["feat_total_close"].notna().any()
    uses_close_anchor = has_spread_close or has_total_close
    feature_cols_no_anchors = [c for c in feature_cols if c not in ["feat_spread_close","feat_total_close"]]

    def _weeks_ago(series_dt: pd.Series, ref: pd.Timestamp) -> np.ndarray:
        delta_days = (ref - series_dt).dt.days.clip(lower=0)
        return (delta_days // 7).astype(int).to_numpy()

    oof_m_pred, oof_t_pred, oof_m_true, oof_t_true, oof_home_win = [], [], [], [], []
    fold_sigmas_m, fold_sigmas_t = [], []

    base = dict(
        n_estimators=4000, learning_rate=0.02, num_leaves=63, max_depth=-1,
        min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=2.0, random_state=RANDOM_SEED
    )

    for val_year in years_sorted:
        train_years = [y for y in years_sorted if y < val_year]
        if not train_years: continue
        tr = feat_df["season"].isin(train_years) & feat_df["margin"].notna() & feat_df["total"].notna()
        va = (feat_df["season"] == val_year) & feat_df["margin"].notna() & feat_df["total"].notna()
        if not tr.any() or not va.any(): continue

        X_tr_all, X_va_all = feat_df.loc[tr, feature_cols], feat_df.loc[va, feature_cols]
        ref_dt = feat_df.loc[tr, "kickoff"].max()
        w_weeks = _weeks_ago(feat_df.loc[tr, "kickoff"], ref_dt)
        sample_w = (0.97 ** w_weeks)

        if uses_close_anchor:
            y_m_tr = (feat_df.loc[tr, "margin"] - (feat_df.loc[tr, "feat_spread_close"].fillna(0) if has_spread_close else 0)).values
            y_t_tr = (feat_df.loc[tr, "total"]  - (feat_df.loc[tr, "feat_total_close"].fillna(0)  if has_total_close  else 0)).values
            y_m_va_abs = feat_df.loc[va, "margin"].values
            y_t_va_abs = feat_df.loc[va, "total"].values
            X_tr = feat_df.loc[tr, feature_cols_no_anchors]
            X_va = feat_df.loc[va, feature_cols_no_anchors]
        else:
            y_m_tr = feat_df.loc[tr, "margin"].values
            y_t_tr = feat_df.loc[tr, "total"].values
            y_m_va_abs = feat_df.loc[va, "margin"].values
            y_t_va_abs = feat_df.loc[va, "total"].values
            X_tr = X_tr_all; X_va = X_va_all

        m = LGBMRegressor(objective="huber", **base)
        t = LGBMRegressor(objective="huber", **base)
        y_m_va_eval = y_m_va_abs - (feat_df.loc[va, "feat_spread_close"].fillna(0).values if uses_close_anchor and has_spread_close else 0)
        y_t_va_eval = y_t_va_abs - (feat_df.loc[va, "feat_total_close"].fillna(0).values  if uses_close_anchor and has_total_close  else 0)

        m.fit(X_tr, y_m_tr, sample_weight=sample_w, eval_set=[(X_va, y_m_va_eval)],
              eval_metric="l1", callbacks=[lgb.early_stopping(200, verbose=False)])
        t.fit(X_tr, y_t_tr, sample_weight=sample_w, eval_set=[(X_va, y_t_va_eval)],
              eval_metric="l1", callbacks=[lgb.early_stopping(200, verbose=False)])

        m_res = m.predict(X_va); t_res = t.predict(X_va)
        if uses_close_anchor:
            base_spread = feat_df.loc[va, "feat_spread_close"].fillna(0).to_numpy() if has_spread_close else np.zeros_like(m_res)
            base_total  = feat_df.loc[va, "feat_total_close"].fillna(0).to_numpy()  if has_total_close  else np.zeros_like(t_res)
            m_pred = m_res + base_spread; t_pred = t_res + base_total
        else:
            m_pred = m_res; t_pred = t_res

        oof_m_pred.extend(m_pred.tolist()); oof_t_pred.extend(t_pred.tolist())
        oof_m_true.extend(y_m_va_abs.tolist()); oof_t_true.extend(y_t_va_abs.tolist())
        oof_home_win.extend((y_m_va_abs > 0).astype(int).tolist())

        m_q84 = LGBMRegressor(objective="quantile", alpha=0.84, **{k: v for k, v in base.items() if k != "lambda_l2"})
        m_q16 = LGBMRegressor(objective="quantile", alpha=0.16, **{k: v for k, v in base.items() if k != "lambda_l2"})
        t_q84 = LGBMRegressor(objective="quantile", alpha=0.84, **{k: v for k, v in base.items() if k != "lambda_l2"})
        t_q16 = LGBMRegressor(objective="quantile", alpha=0.16, **{k: v for k, v in base.items() if k != "lambda_l2"})
        m_q84.fit(X_tr, y_m_tr); m_q16.fit(X_tr, y_m_tr)
        t_q84.fit(X_tr, y_t_tr); t_q16.fit(X_tr, y_t_tr)
        sig_m = float(np.mean((m_q84.predict(X_va) - m_q16.predict(X_va)) / 2.0))
        sig_t = float(np.mean((t_q84.predict(X_va) - t_q16.predict(X_va)) / 2.0))
        fold_sigmas_m.append(max(sig_m, 1.0)); fold_sigmas_t.append(max(sig_t, 1.0))

    if not oof_m_pred:
        df_sorted = feat_df.sort_values(["season","kickoff"])
        mask = df_sorted["margin"].notna() & df_sorted["total"].notna()
        df_sorted = df_sorted[mask]
        n = len(df_sorted)
        if n >= 10:
            cut = int(0.8 * n)
            X_tr_all = df_sorted.iloc[:cut][feature_cols]
            X_va_all = df_sorted.iloc[cut:][feature_cols]
            uses_close_anchor = has_spread_close or has_total_close
            if uses_close_anchor:
                X_tr = df_sorted.iloc[:cut][feature_cols_no_anchors]
                X_va = df_sorted.iloc[cut:][feature_cols_no_anchors]
                y_m_tr = (df_sorted.iloc[:cut]["margin"] - df_sorted.iloc[:cut]["feat_spread_close"].fillna(0)).values if has_spread_close else df_sorted.iloc[:cut]["margin"].values
                y_t_tr = (df_sorted.iloc[:cut]["total"]  - df_sorted.iloc[:cut]["feat_total_close"].fillna(0)).values  if has_total_close  else df_sorted.iloc[:cut]["total"].values
                y_m_va_abs = df_sorted.iloc[cut:]["margin"].values
                y_t_va_abs = df_sorted.iloc[cut:]["total"].values
            else:
                X_tr, X_va = X_tr_all, X_va_all
                y_m_tr = df_sorted.iloc[:cut]["margin"].values
                y_t_tr = df_sorted.iloc[:cut]["total"].values
                y_m_va_abs = df_sorted.iloc[cut:]["margin"].values
                y_t_va_abs = df_sorted.iloc[cut:]["total"].values

            m = LGBMRegressor(objective="huber", **base).fit(X_tr, y_m_tr)
            t = LGBMRegressor(objective="huber", **base).fit(X_tr, y_t_tr)
            m_res = m.predict(X_va); t_res = t.predict(X_va)
            if uses_close_anchor:
                base_spread = df_sorted.iloc[cut:]["feat_spread_close"].fillna(0).to_numpy() if has_spread_close else np.zeros_like(m_res)
                base_total  = df_sorted.iloc[cut:]["feat_total_close"].fillna(0).to_numpy()  if has_total_close  else np.zeros_like(t_res)
                m_pred = m_res + base_spread; t_pred = t_res + base_total
            else:
                m_pred = m_res; t_pred = t_res
            oof_m_pred = m_pred.tolist(); oof_t_pred = t_pred.tolist()
            oof_m_true = y_m_va_abs.tolist(); oof_t_true = y_t_va_abs.tolist()
            oof_home_win = (y_m_va_abs > 0).astype(int).tolist()

    calibrator = IsotonicRegression(out_of_bounds="clip")
    if oof_m_pred and len(set(oof_home_win)) > 1:
        calibrator.fit(np.array(oof_m_pred), np.array(oof_home_win))
    else:
        calibrator.fit([0, 1], [0.5, 0.5])

    mae_margin = float(np.mean(np.abs(np.array(oof_m_true) - np.array(oof_m_pred)))) if oof_m_pred else float("nan")
    mae_total  = float(np.mean(np.abs(np.array(oof_t_true) - np.array(oof_t_pred)))) if oof_t_pred else float("nan")
    brier = float(np.mean((np.array(oof_home_win) - calibrator.predict(np.array(oof_m_pred)))**2)) if oof_m_pred else float("nan")

    mask_all = feat_df["season"].isin(years_sorted) & feat_df["margin"].notna() & feat_df["total"].notna()
    X_all = feat_df.loc[mask_all, feature_cols]
    if has_spread_close or has_total_close:
        y_m_all = (feat_df.loc[mask_all, "margin"] - (feat_df.loc[mask_all, "feat_spread_close"].fillna(0) if has_spread_close else 0)).values
        y_t_all = (feat_df.loc[mask_all, "total"]  - (feat_df.loc[mask_all, "feat_total_close"].fillna(0)  if has_total_close  else 0)).values
        X_all_fit = feat_df.loc[mask_all, [c for c in feature_cols if c not in ["feat_spread_close","feat_total_close"]]]
    else:
        y_m_all = feat_df.loc[mask_all, "margin"].values
        y_t_all = feat_df.loc[mask_all, "total"].values
        X_all_fit = X_all

    m_final = LGBMRegressor(objective="huber", **base).fit(X_all_fit, y_m_all)
    t_final = LGBMRegressor(objective="huber", **base).fit(X_all_fit, y_t_all)

    sigma_margin = max(float(np.mean(fold_sigmas_m)) if fold_sigmas_m else 5.0, 3.0)
    sigma_total  = max(float(np.mean(fold_sigmas_t)) if fold_sigmas_t else 6.0, 3.5)

    metrics = {"MAE_margin": mae_margin, "MAE_total": mae_total, "Brier_home": brier}
    config = {
        "uses_close_anchor": bool(has_spread_close or has_total_close),
        "has_spread_close": bool(has_spread_close),
        "has_total_close": bool(has_total_close),
        "feature_cols_no_anchors": [c for c in feature_cols if c not in ["feat_spread_close","feat_total_close"]]
    }
    return m_final, t_final, calibrator, sigma_margin, sigma_total, metrics, config

# ----------------------------- Pricing utils ------------------------------ #
def american_to_prob(line: float) -> float:
    try:
        if line > 0: return 100.0 / (line + 100.0)
        else: return -line / (-line + 100.0)
    except Exception:
        return 0.5

def expected_value(p: float, line: float) -> float:
    decimal_odds = 1.0 + (line / 100.0) if line > 0 else 1.0 + (100.0 / (-line))
    return p * (decimal_odds - 1.0) - (1.0 - p)

def kelly(p: float, line: float, cap: float = 0.15) -> float:
    decimal_odds = 1.0 + (line / 100.0) if line > 0 else 1.0 + (100.0 / (-line))
    f = (p * (decimal_odds - 1.0) - (1.0 - p)) / (decimal_odds - 1.0)
    return max(0.0, min(cap, f))

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ----------------------------- App ---------------------------------------- #
st.set_page_config(page_title="NFL Sides & Totals Predictor", layout="wide", page_icon="🏈")
st.title("NFL Sides & Totals Predictor – Residual on Close")

@st.cache_data(show_spinner=False)
def load_data(seasons: Iterable[int], use_real: bool, include_scores: bool = True) -> pd.DataFrame:
    seasons = list(seasons)
    if use_real and HAS_NFL_DATA_PY:
        try:
            return load_real_data(seasons, include_scores=include_scores)
        except Exception as e:
            st.warning(f"Falling back to synthetic data due to: {e}")
    return generate_synthetic_data(seasons)

@st.cache_resource(show_spinner=True)
def load_or_train_model(start_year: int, end_year: int, use_real: bool):
    df = load_data(range(start_year, end_year + 1), use_real)
    # Ensure canonical game_id exists for robust merges
    if "game_id" not in df.columns or df["game_id"].isna().any():
        if all(k in df.columns for k in ["season","week","home_team","away_team"]):
            df["home_team"] = df["home_team"].apply(normalize_team_code).astype(str)
            df["away_team"] = df["away_team"].apply(normalize_team_code).astype(str)
            df["game_id"] = [
                build_game_id(s, w, h, a)
                for s, w, h, a in zip(df["season"], df["week"], df["home_team"], df["away_team"])
            ]

    # Attach historical close anchors if provided
    close_train = st.session_state.get("odds_close_train_df", None)
    if close_train is not None:
        try:
            ready_raw = _read_tabular(close_train)
            ready = _prepare_training_close_lines(ready_raw)
            # normalize teams in df too (merge safety)
            df["home_team"] = df["home_team"].apply(normalize_team_code).astype(str)
            df["away_team"] = df["away_team"].apply(normalize_team_code).astype(str)
            if "game_id" in ready.columns and "game_id" in df.columns:
                df = df.merge(ready, on="game_id", how="left")
            else:
                ready["home_team"] = ready["home_team"].apply(normalize_team_code).astype(str)
                ready["away_team"] = ready["away_team"].apply(normalize_team_code).astype(str)
                merge_keys = ["season","week","home_team","away_team"]
                if all(k in df.columns for k in merge_keys) and all(k in ready.columns for k in merge_keys):
                    df = df.merge(ready, on=merge_keys, how="left")
                else:
                    st.warning("Historical close file missing merge keys; skipping.")
                    st.session_state["anchor_train_stats"] = "Training anchors: skipped (missing merge keys)"
            try:
                st.session_state["anchor_train_stats"] = _summarize_anchor_cols(df, "Training anchors:")
            except Exception:
                pass
            try:
                _teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True).dropna()))
                st.caption(f"Teams in training after normalization: {len(_teams)} unique")
            except Exception:
                pass
        except Exception as e:
            st.warning(f"Ignoring historical close file: {e}")
            st.session_state["anchor_train_stats"] = f"Training anchors: error ({e})"
    else:
        st.session_state["anchor_train_stats"] = None

    feat_df = build_features(df)
    return train_models(feat_df, range(start_year, end_year + 1))

def main():
    st.sidebar.header("Training & Prediction Settings")
    start_year = st.sidebar.number_input("Train start year", min_value=2000, max_value=2025, value=2016, step=1)
    end_year   = st.sidebar.number_input("Train end year",   min_value=start_year, max_value=2025, value=2023, step=1)
    predict_year = st.sidebar.number_input("Predict season", min_value=2015, max_value=2030, value=2024, step=1)
    predict_week = st.sidebar.number_input("Predict week (0=all)", min_value=0, max_value=20, value=1, step=1)

    odds_close_train_file = st.sidebar.file_uploader("Upload historical CLOSE lines (training) – CSV or Excel", type=["csv","xlsx","xls"])
    if odds_close_train_file is not None:
        st.session_state["odds_close_train_df"] = odds_close_train_file
        st.sidebar.success("Historical close file attached (parsed automatically at train time).")
    else:
        st.session_state["odds_close_train_df"] = None

    odds_file = st.sidebar.file_uploader("Upload odds file for prediction/pricing (CSV or Excel)", type=["csv","xlsx","xls"])

    min_edge = st.sidebar.slider("Minimum edge", 0.0, 0.1, 0.0, 0.01)
    kelly_cap = st.sidebar.slider("Kelly cap", 0.0, 0.5, 0.15, 0.01)
    st.sidebar.caption("Tip: Train on ≥ 6 seasons. Residual-on-close requires close lines in training.")

    use_real_data = st.sidebar.checkbox("Use real NFL data (nfl_data_py)", value=HAS_NFL_DATA_PY and True, disabled=not HAS_NFL_DATA_PY)
    if not HAS_NFL_DATA_PY and use_real_data:
        st.sidebar.write("(Real data unavailable – using synthetic)")

    if st.sidebar.button("Train or Load Model"):
        with st.spinner("Training model..."):
            m_model, t_model, calibrator, sigma_m, sigma_t, metrics, config = load_or_train_model(int(start_year), int(end_year), use_real_data)
        st.success(f"Model ready. MAE_margin={metrics.get('MAE_margin', float('nan')):.3f}, MAE_total={metrics.get('MAE_total', float('nan')):.3f}")
        if st.session_state.get("anchor_train_stats"):
            st.caption(st.session_state["anchor_train_stats"])
        st.session_state["model"] = {
            "m_model": m_model, "t_model": t_model, "calibrator": calibrator,
            "sigma_m": sigma_m, "sigma_t": sigma_t,
            "use_real_data": use_real_data,
            "config": config, "uses_close_anchor": config.get("uses_close_anchor", False)
        }
        st.subheader("Cross-validated metrics")
        st.write({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})

    if "model" not in st.session_state:
        st.info("Train a model to generate predictions.")
        return

    model = st.session_state["model"]
    with st.spinner("Computing predictions..."):
        df_pred = load_data([int(predict_year)], model.get("use_real_data", False), include_scores=False)
        if predict_week != 0:
            df_pred = df_pred[df_pred["week"] == int(predict_week)]
        if df_pred.empty:
            st.warning("No games found for the selected season/week.")
            return

        # Ensure canonical IDs on prediction set
        df_pred["home_team"] = df_pred["home_team"].apply(normalize_team_code).astype(str)
        df_pred["away_team"] = df_pred["away_team"].apply(normalize_team_code).astype(str)
        if "game_id" not in df_pred.columns or df_pred["game_id"].isna().any():
            if all(k in df_pred.columns for k in ["season","week","home_team","away_team"]):
                df_pred["game_id"] = [
                    build_game_id(s, w, h, a)
                    for s, w, h, a in zip(df_pred["season"], df_pred["week"], df_pred["home_team"], df_pred["away_team"])
                ]

        # Features for upcoming games
        if model.get("use_real_data", False):
            df_train_stats = load_data(range(int(start_year), int(end_year) + 1), True, include_scores=True)
            team_stats, league_pf, league_pa = compute_team_stats_for_real(df_train_stats)
            feat_df = build_features_for_upcoming_games(df_pred, team_stats, league_pf, league_pa)
        else:
            feat_df = build_features(df_pred)

        # Inject anchors from odds file (if provided)
        anchors_df = None
        if odds_file is not None:
            tmp_odds = _read_tabular(odds_file)
            try:
                std_odds = _standardize_odds_columns(tmp_odds)
                cand_keys = [c for c in ["game_id","season","week","home_team","away_team"] if c in std_odds.columns]
                anchors_df = std_odds[cand_keys + [c for c in ["spread_line","total_line"] if c in std_odds.columns]].copy()
                anchors_df = anchors_df.rename(columns={"spread_line":"feat_spread_close","total_line":"feat_total_close"})
            except Exception as e:
                st.warning(f"Could not standardize odds file for anchors: {e}")

        if anchors_df is not None and not anchors_df.empty:
            feat_df["home_team"] = feat_df["home_team"].apply(normalize_team_code).astype(str)
            feat_df["away_team"] = feat_df["away_team"].apply(normalize_team_code).astype(str)
            if "home_team" in anchors_df.columns:
                anchors_df["home_team"] = anchors_df["home_team"].apply(normalize_team_code).astype(str)
            if "away_team" in anchors_df.columns:
                anchors_df["away_team"] = anchors_df["away_team"].apply(normalize_team_code).astype(str)

            if ("game_id" not in feat_df.columns) or feat_df["game_id"].isna().any():
                if all(k in feat_df.columns for k in ["season","week","home_team","away_team"]):
                    feat_df["game_id"] = [
                        build_game_id(s, w, h, a)
                        for s, w, h, a in zip(feat_df["season"], feat_df["week"], feat_df["home_team"], feat_df["away_team"])
                    ]
            if ("game_id" not in anchors_df.columns) or anchors_df["game_id"].isna().any():
                if all(k in anchors_df.columns for k in ["season","week","home_team","away_team"]):
                    anchors_df["game_id"] = [
                        build_game_id(s, w, h, a)
                        for s, w, h, a in zip(anchors_df["season"], anchors_df["week"], anchors_df["home_team"], anchors_df["away_team"])
                    ]

            if "game_id" in anchors_df.columns and "game_id" in feat_df.columns:
                feat_df = feat_df.merge(anchors_df[["game_id","feat_spread_close","feat_total_close"]], on="game_id", how="left")
            elif all(k in anchors_df.columns for k in ["season","week","home_team","away_team"]):
                feat_df = feat_df.merge(anchors_df, on=["season","week","home_team","away_team"], how="left")
            else:
                st.warning("Prediction odds file missing merge keys for anchors; skipping anchor merge.")

        try:
            st.caption(_summarize_anchor_cols(feat_df, "Prediction anchors:"))
        except Exception:
            pass
        try:
            _teams = sorted(pd.unique(pd.concat([feat_df["home_team"], feat_df["away_team"]], ignore_index=True).dropna()))
            st.caption(f"Teams in prediction after normalization: {len(_teams)} unique")
        except Exception:
            pass

        if feat_df.empty:
            st.warning("No features could be constructed for the selected games.")
            return

        feature_cols = get_feature_columns(feat_df)
        cfg = model.get("config", {})
        cols_no_anchors = cfg.get("feature_cols_no_anchors", [c for c in feature_cols if c not in ["feat_spread_close","feat_total_close"]])
        X_for_pred = feat_df[cols_no_anchors] if cfg.get("uses_close_anchor", False) else feat_df[feature_cols]

        # Align inference columns to what each LightGBM model was actually trained on.
        X_margin_in = _align_features_for_predict(model["m_model"], X_for_pred)
        X_total_in  = _align_features_for_predict(model["t_model"], X_for_pred)

        # run model preds on aligned frames
        margin_res_pred = model["m_model"].predict(X_margin_in)
        total_res_pred  = model["t_model"].predict(X_total_in)
        spread_base = feat_df.get("feat_spread_close", pd.Series(np.zeros(len(feat_df)))).fillna(0).to_numpy()
        total_base  = feat_df.get("feat_total_close",  pd.Series(np.zeros(len(feat_df)))).fillna(0).to_numpy()
        if cfg.get("uses_close_anchor", False):
            margin_pred = margin_res_pred + spread_base
            total_pred  = total_res_pred  + total_base
        else:
            margin_pred = margin_res_pred
            total_pred  = total_res_pred

        sigma_margin = model["sigma_m"]
        sigma_total  = model["sigma_t"]
        p_home = model["calibrator"].predict(margin_pred)
        home_pred = (total_pred + margin_pred) / 2.0
        away_pred = (total_pred - margin_pred) / 2.0

        out_df = feat_df[["game_id","season","week","kickoff","home_team","away_team"]].copy()
        out_df["home_pred"] = home_pred
        out_df["away_pred"] = away_pred
        out_df["margin_pred"] = margin_pred
        out_df["total_pred"] = total_pred
        out_df["p_home"] = p_home
        out_df["sigma_margin"] = sigma_margin
        out_df["sigma_total"] = sigma_total

        # Pricing block (merge standardized odds, compute edges)
        if odds_file is not None:
            odds_df_raw = _read_tabular(odds_file)
            try:
                odds_std = _standardize_odds_columns(odds_df_raw)
            except Exception as e:
                cols_seen = list(odds_df_raw.columns)
                st.error(f"Unrecognised odds CSV format. {e}\nColumns in file: {cols_seen}")
                odds_std = pd.DataFrame()

            if not odds_std.empty:
                mk = [c for c in ["game_id","season","week","home_team","away_team"] if c in odds_std.columns]
                st.caption(f"Odds file standardised. Keys present: {mk} • coverage: spread_line={int(odds_std.get('spread_line', pd.Series(dtype=float)).notna().sum())}, total_line={int(odds_std.get('total_line', pd.Series(dtype=float)).notna().sum())}")

                # Normalize & ensure game_id both sides
                out_df["home_team"] = out_df["home_team"].apply(normalize_team_code).astype(str)
                out_df["away_team"] = out_df["away_team"].apply(normalize_team_code).astype(str)
                if "home_team" in odds_std.columns:
                    odds_std["home_team"] = odds_std["home_team"].apply(normalize_team_code).astype(str)
                if "away_team" in odds_std.columns:
                    odds_std["away_team"] = odds_std["away_team"].apply(normalize_team_code).astype(str)
                if "game_id" not in out_df.columns or out_df["game_id"].isna().any():
                    if all(k in out_df.columns for k in ["season","week","home_team","away_team"]):
                        out_df["game_id"] = [
                            build_game_id(s, w, h, a)
                            for s, w, h, a in zip(out_df["season"], out_df["week"], out_df["home_team"], out_df["away_team"])
                        ]
                if "game_id" not in odds_std.columns or odds_std["game_id"].isna().any():
                    if all(k in odds_std.columns for k in ["season","week","home_team","away_team"]):
                        odds_std["game_id"] = [
                            build_game_id(s, w, h, a)
                            for s, w, h, a in zip(odds_std["season"], odds_std["week"], odds_std["home_team"], odds_std["away_team"])
                        ]

                if "game_id" in odds_std.columns and "game_id" in out_df.columns:
                    merged = pd.merge(out_df, odds_std, on="game_id", how="left")
                elif all(k in odds_std.columns for k in ["season","week","home_team","away_team"]):
                    merged = pd.merge(out_df, odds_std, on=["season","week","home_team","away_team"], how="left")
                else:
                    st.warning("Standardized odds missing merge keys; pricing skipped.")
                    merged = out_df.copy()

                try:
                    st.caption(f"Merged odds: spread_line rows={int(merged.get('spread_line', pd.Series(dtype=float)).notna().sum())}, total_line rows={int(merged.get('total_line', pd.Series(dtype=float)).notna().sum())}")
                except Exception:
                    pass
            else:
                merged = out_df.copy()

            # Compute edges
            if 'merged' not in locals():
                merged = out_df.copy()
            edges_ml, ev_ml, kelly_ml = [], [], []
            edges_spread, ev_spread, kelly_spread = [], [], []
            edges_total, ev_total, kelly_total = [], [], []
            for _, row in merged.iterrows():
                prob_home = row["p_home"]

                ml_line = row.get("ml", np.nan)
                if pd.notna(ml_line):
                    implied_prob_ml = american_to_prob(ml_line)
                    edge_ml = prob_home - implied_prob_ml
                    if abs(edge_ml) < min_edge:
                        edges_ml.append(np.nan); ev_ml.append(np.nan); kelly_ml.append(0.0)
                    else:
                        edges_ml.append(edge_ml)
                        ev_ml.append(expected_value(prob_home, ml_line))
                        kelly_ml.append(kelly(prob_home, ml_line, cap=kelly_cap))
                else:
                    edges_ml.append(np.nan); ev_ml.append(np.nan); kelly_ml.append(0.0)

                spread_line = row.get("spread_line", np.nan)
                spread_odds = row.get("spread_odds", np.nan)
                if pd.notna(spread_line) and pd.notna(spread_odds):
                    z = (spread_line - row["margin_pred"]) / (sigma_margin if sigma_margin > 0 else 1.0)
                    prob_cover = 1.0 - norm_cdf(z)
                    implied_prob_spread = american_to_prob(spread_odds)
                    edge_sp = prob_cover - implied_prob_spread
                    if abs(edge_sp) < min_edge:
                        edges_spread.append(np.nan); ev_spread.append(np.nan); kelly_spread.append(0.0)
                    else:
                        edges_spread.append(edge_sp)
                        ev_spread.append(expected_value(prob_cover, spread_odds))
                        kelly_spread.append(kelly(prob_cover, spread_odds, cap=kelly_cap))
                else:
                    edges_spread.append(np.nan); ev_spread.append(np.nan); kelly_spread.append(0.0)

                total_line = row.get("total_line", np.nan)
                total_odds = row.get("total_odds", np.nan)
                if pd.notna(total_line) and pd.notna(total_odds):
                    zt = (total_line - row["total_pred"]) / (sigma_total if sigma_total > 0 else 1.0)
                    prob_over = 1.0 - norm_cdf(zt)
                    implied_prob_total = american_to_prob(total_odds)
                    edge_tot = prob_over - implied_prob_total
                    if abs(edge_tot) < min_edge:
                        edges_total.append(np.nan); ev_total.append(np.nan); kelly_total.append(0.0)
                    else:
                        edges_total.append(edge_tot)
                        ev_total.append(expected_value(prob_over, total_odds))
                        kelly_total.append(kelly(prob_over, total_odds, cap=kelly_cap))
                else:
                    edges_total.append(np.nan); ev_total.append(np.nan); kelly_total.append(0.0)

            merged["edge_ml"] = edges_ml; merged["ev_ml"] = ev_ml; merged["kelly_ml"] = kelly_ml
            merged["edge_spread"] = edges_spread; merged["ev_spread"] = ev_spread; merged["kelly_spread"] = kelly_spread
            merged["edge_total"] = edges_total; merged["ev_total"] = ev_total; merged["kelly_total"] = kelly_total
            out_df = merged

        st.subheader("Predictions")
        display_df = out_df.copy()
        edge_cols = [c for c in display_df.columns if c.startswith("edge_")]
        if edge_cols and display_df[edge_cols].notna().any().any():
            mask = np.any(display_df[edge_cols].abs() >= min_edge, axis=1)
            display_df = display_df[mask]
        st.dataframe(display_df)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="predictions.csv", mime="text/csv")

if __name__ == "__main__":
    main()
