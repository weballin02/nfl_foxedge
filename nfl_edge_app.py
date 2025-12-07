
"""
NFL Edge App — production‑grade single file with auto ingestion

This script needs no CSVs. It fetches historical results and upcoming schedules
using `nfl_data_py`, trains a deterministic points-for model, calibrates win
probabilities on predicted margins, and provides a Streamlit UI with EV math
for moneyline, spreads, and totals.

Guarantees
----------
- Symmetric scoring: both sides predicted with the same points_for target.
- Proper uncertainty: margin sigma for spreads, total sigma for totals.
- Deterministic: fixed seeds, time-aware validation split.
- Strict schema: fail fast on missing fields.
- Odds-safe: team alias normalization and schema validation.

Dependencies
------------
pip install nfl_data_py pandas numpy scikit-learn joblib streamlit requests beautifulsoup4

Quickstart
----------
# Train (auto-ingest historical games)
python nfl_edge_app.py train --out models/nfl_model.pkl --min_season 2020

# Serve UI (auto-ingest games and upcoming matchups)
streamlit run nfl_edge_app.py -- serve --model models/nfl_model.pkl --min_season 2020

Optional odds CSV (for EV calculations)
---------------------------------------
Required columns:
  book,market,label,price,point,home_team,away_team,season,week
  - market in {'moneyline','spread','total'}
  - label: for ML/SPREAD use a team; for TOTAL use 'Over' or 'Under'
  - price: American odds (e.g., -120, +135)
  - point: spread or total line (NaN for moneyline)

"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
# ---------------------------
# Time zone heuristics (for travel)
# ---------------------------
TEAM_TZ: Dict[str, str] = {
    'ARI':'MT','ATL':'ET','BAL':'ET','BUF':'ET','CAR':'ET','CHI':'CT','CIN':'ET','CLE':'ET','DAL':'CT',
    'DEN':'MT','DET':'ET','GB':'CT','HOU':'CT','IND':'ET','JAX':'ET','KC':'CT','LAC':'PT','LAR':'PT',
    'LV':'PT','MIA':'ET','MIN':'CT','NE':'ET','NO':'CT','NYG':'ET','NYJ':'ET','PHI':'ET','PIT':'ET',
    'SEA':'PT','SF':'PT','TB':'ET','TEN':'CT','WAS':'ET'
}
TZ_IDX = {'PT': -3, 'MT': -2, 'CT': -1, 'ET': 0}
def _tz_idx(team: str) -> int:
    return TZ_IDX.get(TEAM_TZ.get(team, 'ET'), 0)
CACHE_DIR = "./data"
HIST_CACHE = os.path.join(CACHE_DIR, "nfl_games.parquet")
# ---------------------------
# Weather ingestion (from pbp metadata) + caching
# ---------------------------
WEATHER_CACHE_DIR = os.path.join(CACHE_DIR, "weather_cache")
os.makedirs(WEATHER_CACHE_DIR, exist_ok=True)

def _parse_precip_flag(text: str) -> float:
    s = str(text).lower()
    if any(k in s for k in ['rain', 'drizzle', 'shower', 'snow', 'sleet', 'flurries']):
        return 1.0
    return 0.0

def get_weather_table_for_years(years: List[int]) -> pd.DataFrame:
    """
    Returns per-game pre metrics:
    ['game_id','season','week','date','home_team','away_team','temp_f','wind_mph','is_dome','precip_flag']
    Derived from nflfastR pbp where available; falls back to schedule with neutral values.
    """
    years = sorted(set(int(y) for y in years))
    if not years:
        return pd.DataFrame(columns=['game_id','season','week','date','home_team','away_team','temp_f','wind_mph','is_dome','precip_flag'])
    key = f"{min(years)}_{max(years)}"
    cache_path = os.path.join(WEATHER_CACHE_DIR, f"weather_{key}.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    sched = _import_sched_with_refs(years).copy()
    pbp = nfl.import_pbp_data(years)

    # Map available pbp fields defensively
    col_temp = next((c for c in ['temp', 'temperature', 'temp_f'] if c in pbp.columns), None)
    col_wind = next((c for c in ['wind', 'wind_mph'] if c in pbp.columns), None)
    col_roof = 'roof' if 'roof' in pbp.columns else None
    col_weather = next((c for c in ['weather', 'game_weather'] if c in pbp.columns), None)

    if 'game_id' in pbp.columns:
        grp = pbp.groupby('game_id')
        def first_valid(series: pd.Series):
            s = series.dropna()
            return s.iloc[0] if len(s) else np.nan
        agg = pd.DataFrame({'game_id': list(grp.groups.keys())})
        agg['temp_f'] = grp[col_temp].median() if col_temp else np.nan
        agg['wind_mph'] = grp[col_wind].median() if col_wind else np.nan
        agg['roof'] = grp[col_roof].apply(first_valid) if col_roof else np.nan
        if col_weather:
            wtxt = grp[col_weather].apply(first_valid)
            agg['precip_flag'] = wtxt.apply(_parse_precip_flag)
        else:
            agg['precip_flag'] = np.nan
    else:
        agg = pd.DataFrame(columns=['game_id','temp_f','wind_mph','roof','precip_flag'])

    df = sched.merge(agg, on='game_id', how='left')
    df['date'] = pd.to_datetime(df['gameday']).dt.tz_localize(None)
    df['is_dome'] = df['roof'].astype(str).str.lower().isin(['dome','closed','indoors','inside']).astype(float)

    out = df[['game_id','season','week','date','home_team','away_team','temp_f','wind_mph','is_dome','precip_flag']].copy()
    try:
        out.to_parquet(cache_path)
    except Exception:
        pass
    return out

def compute_weather_adjustments(row: pd.Series) -> Tuple[float, float, float]:
    """
    Returns (delta_total, delta_sigma_total, delta_sigma_margin).
    Heuristic based on wind/temp/precip; domes neutral.
    """
    try:
        if float(row.get('is_dome', 0)) >= 0.5:
            return 0.0, 0.0, 0.0
    except Exception:
        pass
    total_adj = 0.0
    sig_tot_up = 0.0
    sig_mar_up = 0.0
    wind = row.get('wind_mph', np.nan)
    temp = row.get('temp_f', np.nan)
    precip = row.get('precip_flag', 0.0)

    try:
        if pd.notna(wind):
            wind = float(wind)
            if wind >= 20:
                total_adj -= 6.0; sig_tot_up += 1.5
            elif wind >= 15:
                total_adj -= 3.0; sig_tot_up += 0.7
    except Exception:
        pass

    try:
        if pd.notna(temp) and float(temp) < 25:
            total_adj -= 2.0; sig_mar_up += 0.3
    except Exception:
        pass

    try:
        if pd.notna(precip) and float(precip) >= 0.5:
            total_adj -= 4.0; sig_tot_up += 0.5
    except Exception:
        pass

    total_adj = max(total_adj, -12.0)
    return total_adj, sig_tot_up, sig_mar_up

# ---------------------------
# Injuries adjustments (produced by injuries.py) + travel heuristics
# ---------------------------
def load_injury_adjustments(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ['home_team','away_team']:
        if c in df.columns:
            df[c] = df[c].map(norm_team)
    if 'season' in df.columns:
        df['season'] = df['season'].astype(int)
    if 'week' in df.columns:
        df['week'] = df['week'].astype(int)
    return df

def compute_travel_adjustments(home: str, away: str, home_rest: float, away_rest: float) -> Tuple[float, float]:
    """
    Returns (delta_margin_for_home, delta_sigma_margin).
    Penalize short weeks and large eastbound time-zone jumps for away.
    """
    adj_margin = 0.0
    sig_mar_up = 0.0
    tz_diff = _tz_idx(home) - _tz_idx(away)
    if away_rest <= 5:
        adj_margin += 0.5
        sig_mar_up += 0.1
    if home_rest <= 5:
        adj_margin -= 0.5
        sig_mar_up += 0.1
    if tz_diff >= 2:
        adj_margin += 0.3
    elif tz_diff <= -2:
        adj_margin -= 0.1
    return adj_margin, sig_mar_up

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import joblib

import re
try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

# Optional UI
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # noqa: F401

# Auto-ingestion
try:
    import nfl_data_py as nfl  # type: ignore
except Exception as e:
    raise ImportError("Auto ingestion requires nfl_data_py. Install with: pip install nfl_data_py") from e

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------
# Team aliases and validation
# ---------------------------
TEAM_ALIASES: Dict[str, str] = {
    # NFC
    'ARI':'ARI','ARIZONA':'ARI','ARIZONA CARDINALS':'ARI','CARDINALS':'ARI','AZ':'ARI',
    'ATL':'ATL','ATLANTA':'ATL','ATLANTA FALCONS':'ATL','FALCONS':'ATL',
    'CAR':'CAR','CAROLINA':'CAR','CAROLINA PANTHERS':'CAR','PANTHERS':'CAR',
    'CHI':'CHI','CHICAGO':'CHI','CHICAGO BEARS':'CHI','BEARS':'CHI',
    'DAL':'DAL','DALLAS':'DAL','DALLAS COWBOYS':'DAL','COWBOYS':'DAL',
    'DET':'DET','DETROIT':'DET','DETROIT LIONS':'DET','LIONS':'DET',
    'GB':'GB','GNB':'GB','GREEN BAY':'GB','GREEN BAY PACKERS':'GB','PACKERS':'GB',
    'LAR':'LAR','LA RAMS':'LAR','LOS ANGELES RAMS':'LAR','RAMS':'LAR','STL RAMS':'LAR',
    'MIN':'MIN','MINNESOTA':'MIN','MINNESOTA VIKINGS':'MIN','VIKINGS':'MIN',
    'NO':'NO','NOR':'NO','NEW ORLEANS':'NO','NEW ORLEANS SAINTS':'NO','SAINTS':'NO',
    'NYG':'NYG','NEW YORK GIANTS':'NYG','GIANTS':'NYG',
    'PHI':'PHI','PHILADELPHIA':'PHI','PHILADELPHIA EAGLES':'PHI','EAGLES':'PHI',
    'SEA':'SEA','SEATTLE':'SEA','SEATTLE SEAHAWKS':'SEA','SEAHAWKS':'SEA',
    'SF':'SF','SFO':'SF','SAN FRANCISCO':'SF','SAN FRANCISCO 49ERS':'SF','49ERS':'SF',
    'TB':'TB','TAMPA BAY':'TB','TAMPA BAY BUCCANEERS':'TB','BUCCANEERS':'TB','BUCS':'TB',
    'WAS':'WAS','WSH':'WAS','WASHINGTON':'WAS','WASHINGTON COMMANDERS':'WAS','COMMANDERS':'WAS','REDSKINS':'WAS',
    # AFC
    'BAL':'BAL','BALTIMORE':'BAL','BALTIMORE RAVENS':'BAL','RAVENS':'BAL',
    'BUF':'BUF','BUFFALO':'BUF','BUFFALO BILLS':'BUF','BILLS':'BUF',
    'CIN':'CIN','CINCINNATI':'CIN','CINCINNATI BENGALS':'CIN','BENGALS':'CIN',
    'CLE':'CLE','CLEVELAND':'CLE','CLEVELAND BROWNS':'CLE','BROWNS':'CLE',
    'DEN':'DEN','DENVER':'DEN','DENVER BRONCOS':'DEN','BRONCOS':'DEN',
    'HOU':'HOU','HOUSTON':'HOU','HOUSTON TEXANS':'HOU','TEXANS':'HOU',
    'IND':'IND','INDIANAPOLIS':'IND','INDIANAPOLIS COLTS':'IND','COLTS':'IND',
    'JAX':'JAX','JAC':'JAX','JACKSONVILLE':'JAX','JACKSONVILLE JAGUARS':'JAX','JAGUARS':'JAX','JAGS':'JAX',
    'KC':'KC','KAN':'KC','KANSAS CITY':'KC','KANSAS CITY CHIEFS':'KC','CHIEFS':'KC',
    'LAC':'LAC','LA CHARGERS':'LAC','LOS ANGELES CHARGERS':'LAC','CHARGERS':'LAC','SD':'LAC','SDG':'LAC','SAN DIEGO CHARGERS':'LAC',
    'LV':'LV','LVR':'LV','LAS VEGAS':'LV','LAS VEGAS RAIDERS':'LV','OAK':'LV','RAIDERS':'LV','OAKLAND RAIDERS':'LV',
    'MIA':'MIA','MIAMI':'MIA','MIAMI DOLPHINS':'MIA','DOLPHINS':'MIA',
    'NE':'NE','NWE':'NE','NEW ENGLAND':'NE','NEW ENGLAND PATRIOTS':'NE','PATRIOTS':'NE',
    'NYJ':'NYJ','NEW YORK JETS':'NYJ','JETS':'NYJ',
    'PIT':'PIT','PITTSBURGH':'PIT','PITTSBURGH STEELERS':'PIT','STEELERS':'PIT',
    'TEN':'TEN','TENNESSEE':'TEN','TENNESSEE TITANS':'TEN','TITANS':'TEN','HOU OILERS':'TEN','OILERS':'TEN',
}


def norm_team(x: str) -> str:
    if pd.isna(x):
        raise ValueError("Team label is NaN")
    k = str(x).strip().upper()
    return TEAM_ALIASES.get(k, TEAM_ALIASES.get(k.replace('.', ''), k))

# Extra nickname/city fallbacks for DK parsing
NICKNAME_ALIASES: Dict[str, str] = {
    'EAGLES':'PHI','COMMANDERS':'WAS','FOOTBALL TEAM':'WAS',
    'GIANTS':'NYG','JETS':'NYJ',
    'COWBOYS':'DAL','WASHINGTON':'WAS',
    'SAINTS':'NO','PACKERS':'GB','BEARS':'CHI','LIONS':'DET','VIKINGS':'MIN',
    'BUCCANEERS':'TB','BUCS':'TB','FALCONS':'ATL','PANTHERS':'CAR','SEAHAWKS':'SEA',
    '49ERS':'SF','NINERS':'SF','RAMS':'LAR','CARDINALS':'ARI',
    'PATRIOTS':'NE','DOLPHINS':'MIA','BILLS':'BUF','JETS':'NYJ',
    'STEELERS':'PIT','BROWNS':'CLE','BENGALS':'CIN','RAVENS':'BAL',
    'COLTS':'IND','TEXANS':'HOU','JAGUARS':'JAX','JAGS':'JAX','TITANS':'TEN',
    'CHIEFS':'KC','BRONCOS':'DEN','RAIDERS':'LV','CHARGERS':'LAC'
}

def _normalize_team_text(txt: str) -> str:
    """
    Aggressive normalizer for scraped team text: strip punctuation, collapse spaces,
    try full string, then last token, then any token match against nickname aliases.
    """
    u = re.sub(r'[^A-Za-z ]+', ' ', str(txt)).upper().strip()
    u = re.sub(r'\s+', ' ', u)
    if not u:
        return u
    # Direct alias hits
    if u in TEAM_ALIASES:
        return TEAM_ALIASES[u]
    if u in NICKNAME_ALIASES:
        return NICKNAME_ALIASES[u]
    parts = [p for p in u.split(' ') if p]
    # Prefer the last token (nickname) if present
    if parts:
        last = parts[-1]
        if last in TEAM_ALIASES: return TEAM_ALIASES[last]
        if last in NICKNAME_ALIASES: return NICKNAME_ALIASES[last]
    # Scan any token
    for p in parts:
        if p in TEAM_ALIASES: return TEAM_ALIASES[p]
        if p in NICKNAME_ALIASES: return NICKNAME_ALIASES[p]
    # Special-cases for LA/NY ambiguity if no nickname token present
    if 'LOS ANGELES' in u and 'CHARGERS' in u: return 'LAC'
    if 'LOS ANGELES' in u and 'RAMS' in u: return 'LAR'
    if 'NEW YORK' in u and 'JETS' in u: return 'NYJ'
    if 'NEW YORK' in u and 'GIANTS' in u: return 'NYG'
    # Fallback to best-effort city match
    for city, code in [('ARIZONA','ARI'),('ATLANTA','ATL'),('CAROLINA','CAR'),('CHICAGO','CHI'),
                       ('DALLAS','DAL'),('DETROIT','DET'),('GREEN BAY','GB'),('MINNESOTA','MIN'),
                       ('NEW ORLEANS','NO'),('PHILADELPHIA','PHI'),('SEATTLE','SEA'),('SAN FRANCISCO','SF'),
                       ('TAMPA BAY','TB'),('WASHINGTON','WAS'),
                       ('BALTIMORE','BAL'),('BUFFALO','BUF'),('CINCINNATI','CIN'),('CLEVELAND','CLE'),
                       ('DENVER','DEN'),('HOUSTON','HOU'),('INDIANAPOLIS','IND'),('JACKSONVILLE','JAX'),
                       ('KANSAS CITY','KC'),('LAS VEGAS','LV'),('MIAMI','MIA'),('NEW ENGLAND','NE'),
                       ('NEW YORK JETS','NYJ'),('NEW YORK GIANTS','NYG'),('PITTSBURGH','PIT'),('TENNESSEE','TEN')]:
        if city in u:
            return code
    return TEAM_ALIASES.get(u, u)

# ---------------------------
# Odds utilities
# ---------------------------
def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def prob_to_american(p: float) -> int:
    p = max(min(float(p), 0.9999), 0.0001)
    if p >= 0.5:
        return int(round(-p / (1 - p) * 100))
    else:
        return int(round((1 - p) / p * 100))

def ev_from_prob_and_american(p: float, odds: float) -> float:
    # Expected value per unit stake
    if odds < 0:
        b = 100.0 / (-float(odds))
    else:
        b = float(odds) / 100.0
    q = 1 - float(p)
    return float(p) * b - q

# ---- Spread/cover probability helpers ----
SQRT2 = math.sqrt(2.0)
def Phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / SQRT2))

def P_home_cover(margin: float, line: float, sigma: float) -> float:
    # Home covers if margin > -line
    return 1 - Phi((-line - margin) / sigma)

def P_away_cover(margin: float, line: float, sigma: float) -> float:
    # Away covers if margin < line
    return Phi((line - margin) / sigma)

# ---- Kelly sizing, line hygiene, reliability bins ----
def kelly_fraction(p: float, american_odds: float, max_kelly: float = 0.15, min_edge: float = 0.0) -> float:
    """
    Return capped Kelly fraction for probability p and American odds.
    max_kelly is the cap per bet (e.g., 0.15 = 15% of bankroll).
    min_edge filters tiny expected edges; if net edge < min_edge, return 0.
    """
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    if american_odds < 0:
        b = 100.0 / (-float(american_odds))  # net profit per 1 staked
    else:
        b = float(american_odds) / 100.0
    q = 1.0 - p
    edge = b * p - q
    if edge < min_edge:
        return 0.0
    k = edge / b
    return float(max(0.0, min(k, max_kelly)))

def standardize_total_line(x: float) -> float:
    """Round total lines to nearest half-point to reduce duplicate clutter."""
    return float(round(x * 2.0) / 2.0)

def collapse_odds_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a timestamp exists (ts/timestamp/asof/collected_at), keep both earliest ('open') and latest ('last')
    per (book,market,label,home_team,away_team,season,week) and attach deltas. Else just dedupe.
    """
    ts_col = None
    for c in ['ts', 'timestamp', 'asof', 'collected_at']:
        if c in df.columns:
            ts_col = c
            break
    base_cols = ['book','market','label','home_team','away_team','season','week']
    if ts_col:
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['point'] = pd.to_numeric(df['point'], errors='coerce')
        sort_df = df.sort_values(ts_col)
        open_df = sort_df.groupby(base_cols, as_index=False).first().add_prefix('open_')
        last_df = sort_df.groupby(base_cols, as_index=False).last()
        merged = last_df.merge(open_df, left_on=base_cols, right_on=['open_'+c for c in base_cols], how='left')
        merged['open_price'] = merged['open_price']
        merged['open_point'] = merged['open_point']
        merged['price_delta'] = merged['price'] - merged['open_price']
        merged['point_delta'] = merged['point'] - merged['open_point']
        # drop duplicated key cols from open_
        for c in base_cols:
            oc = 'open_'+c
            if oc in merged.columns:
                merged.drop(columns=[oc], inplace=True)
        return merged
    return df.drop_duplicates(subset=['book','market','label','price','point','home_team','away_team','season','week'])

def reliability_by_bins(probs: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Per-bin calibration: bin edges, count, mean_pred, empirical, Brier; includes scalar ECE replicated on each row.
    """
    probs = np.asarray(probs, dtype=float)
    actual = np.asarray(actual, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows, ece = [], 0.0
    N = len(probs)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append({'bin': b, 'left': bins[b], 'right': bins[b+1], 'n': 0, 'mean_pred': np.nan, 'empirical': np.nan, 'brier': np.nan})
            continue
        mp = float(probs[m].mean())
        emp = float(actual[m].mean())
        br = float(np.mean((probs[m] - actual[m])**2))
        rows.append({'bin': b, 'left': bins[b], 'right': bins[b+1], 'n': int(m.sum()), 'mean_pred': mp, 'empirical': emp, 'brier': br})
        ece += (m.sum() / N) * abs(emp - mp)
    df = pd.DataFrame(rows)
    df['ece'] = ece
    return df

# ---------------------------
# DraftKings Network betting splits (NFL) scraper
# ---------------------------
DK_EG_NFL = 88808  # entity group code for NFL as used by DK Network
DK_SPLITS_BASE = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"


def _dk_guess_team_code(name: str) -> str:
    return _normalize_team_text(name)

def _clean_percent(x: str) -> float:
    try:
        return float(str(x).replace('%','').strip()) / 100.0
    except Exception:
        return float('nan')

def _clean_american(x: str) -> int:
    # normalize unicode minus
    s = str(x).replace('−','-').strip()
    m = re.match(r'^-?\+?\d{3,4}$', s)
    if not m:
        return int(float('nan'))
    return int(s)

def _clean_point(x: str) -> float:
    s = str(x).replace('−','-').strip()
    try:
        return float(s)
    except Exception:
        # try to extract last numeric token
        m = re.search(r'(-?\d+(?:\.\d+)?)', s)
        return float(m.group(1)) if m else float('nan')

def build_dk_splits_url(window: str = "n7days", market: str | int = 0) -> str:
    """
    window: 'today' | 'tomorrow' | 'n7days' | 'n30days'
    market: 0 = All, or strings 'Moneyline'|'Spread'|'Total'
    """
    if isinstance(market, str):
        m = market.strip().lower()
        if m.startswith('money'):
            market = 'Moneyline'
        elif m.startswith('spread'):
            market = 'Spread'
        elif m.startswith('total'):
            market = 'Total'
        else:
            market = 0
    return f"{DK_SPLITS_BASE}?tb_eg={DK_EG_NFL}&tb_edate={window}&tb_emt={market}"

def fetch_dk_splits_nfl(window: str = "n7days", market: str | int = 0) -> pd.DataFrame:
    """
    Scrape DK Network betting splits for NFL and return a tidy dataframe:
    columns: ['home','away','market','label','price','point','handle_pct','bets_pct','game_time']
    - market in {'moneyline','spread','total'}
    - label: team code for ML/SPREAD; 'Over'/'Under' for TOTAL
    """
    if requests is None or BeautifulSoup is None:
        raise ImportError("Fetching DK splits requires requests and beautifulsoup4. Install them first.")
    url = build_dk_splits_url(window=window, market=market)
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    rows = []
    # Each matchup appears as a heading with "Away @ Home"
    headers = soup.find_all(['h5','h4'])
    for h in headers:
        title = re.sub(r'\s+', ' ', h.get_text(" ", strip=True)).strip()
        if '@' not in title:
            continue
        try:
            away_name, home_name = [t.strip() for t in title.split('@', 1)]
        except Exception:
            continue
        away_code = _dk_guess_team_code(away_name)
        home_code = _dk_guess_team_code(home_name)

        # Next siblings until next header comprise the block
        block_texts = []
        for sib in h.next_siblings:
            if getattr(sib, 'name', None) in ('h5','h4'):
                break
            txt = sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else str(sib).strip()
            if txt:
                block_texts.extend([t for t in txt.split('\n') if t.strip()])

        # Track current section
        section = None  # 'moneyline'|'spread'|'total'
        i = 0
        # Useful labels to ignore
        ignore_tokens = {'ODDS','% HANDLE','% BETS','ODDS % HANDLE % BETS','SPREAD','TOTAL','MONEYLINE'}
        while i < len(block_texts):
            t = block_texts[i].strip()
            u = t.upper()
            if not t or u in ignore_tokens:
                i += 1
                continue
            # switch section
            if 'MONEYLINE' == u:
                section = 'moneyline'; i += 1; continue
            if 'SPREAD' == u:
                section = 'spread'; i += 1; continue
            if 'TOTAL' == u:
                section = 'total'; i += 1; continue

            # Identify odds token
            if re.match(r'^[−-]?\d{3,4}$', u):
                # Odds line without a preceding label is useless; skip
                i += 1
                continue

            # Assume current token is the label; next non-empty should be American odds, then %handle and %bets
            label_text = t
            # find odds
            j = i + 1
            # skip any header junk
            while j < len(block_texts) and not re.match(r'^[−-]?\d{3,4}$', block_texts[j].strip().replace(' ', '')):
                # Might be time string like "9/4, 08:20PM" → store but don't break the sequence
                j += 1
            if j >= len(block_texts):
                i += 1
                continue
            price_tok = block_texts[j].strip()
            price = _clean_american(price_tok)
            # next two percent tokens
            handle_tok = block_texts[j+1].strip() if j+1 < len(block_texts) else ''
            bets_tok = block_texts[j+2].strip() if j+2 < len(block_texts) else ''
            handle = _clean_percent(handle_tok)
            bets = _clean_percent(bets_tok)

            # Parse per section
            if section == 'spread':
                # e.g., "PHI Eagles -8.5"
                # split last numeric as point
                m = re.search(r'(.*)\s+(-?\d+(?:\.\d+)?)$', label_text.replace('−','-'))
                if m:
                    team_str = m.group(1).strip()
                    point = _clean_point(m.group(2))
                else:
                    team_str, point = label_text, float('nan')
                code = _dk_guess_team_code(team_str)
                rows.append({
                    'home': home_code, 'away': away_code,
                    'market': 'spread', 'label': code,
                    'price': price, 'point': point,
                    'handle_pct': handle, 'bets_pct': bets,
                    'game_time': None
                })
            elif section == 'total':
                # e.g., "Over 44.5" / "Under 44.5"
                m = re.search(r'^(Over|Under)\s+(-?\d+(?:\.\d+)?)$', label_text, flags=re.IGNORECASE)
                if m:
                    ou = m.group(1).capitalize()
                    point = _clean_point(m.group(2))
                    rows.append({
                        'home': home_code, 'away': away_code,
                        'market': 'total', 'label': ou,
                        'price': price, 'point': point,
                        'handle_pct': handle, 'bets_pct': bets,
                        'game_time': None
                    })
            elif section == 'moneyline':
                # label is the team name
                code = _dk_guess_team_code(label_text)
                rows.append({
                    'home': home_code, 'away': away_code,
                    'market': 'moneyline', 'label': code,
                    'price': price, 'point': float('nan'),
                    'handle_pct': handle, 'bets_pct': bets,
                    'game_time': None
                })

            # advance index past label + odds + two percent tokens
            i = j + 3
        # end while
    # end for headers

    df = pd.DataFrame(rows)
    # Deduplicate if page shows pagination repeats
    if not df.empty:
        df = df.drop_duplicates(subset=['home','away','market','label','point','price'])
    return df

# ---------------------------
# Data ingestion via nfl_data_py
# ---------------------------

def fetch_games(min_season: int | None = None) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(HIST_CACHE):
        df = pd.read_parquet(HIST_CACHE)
    else:
        years = list(range(1999, pd.Timestamp.today().year + 1))
        sched = nfl.import_schedules(years)
        # Keep completed games
        done = sched[(sched['home_score'].notna()) & (sched['away_score'].notna())].copy()
        df = done[['season','week','gameday','home_team','away_team','home_score','away_score']].copy()
        df.to_parquet(HIST_CACHE)
    df = df.rename(columns={'gameday':'date','home_score':'home_points','away_score':'away_points'})
    df['date'] = pd.to_datetime(df['date'])
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)
    df['home_team'] = df['home_team'].map(norm_team)
    df['away_team'] = df['away_team'].map(norm_team)
    if min_season is not None:
        df = df[df['season'] >= int(min_season)]
    return df.sort_values(['season','week','date']).reset_index(drop=True)

def load_upcoming_matchups_auto(min_season: int | None = None) -> pd.DataFrame:
    today = pd.Timestamp('today').normalize()
    years = list(range(1999 if min_season is None else int(min_season), today.year + 1))
    sched = _import_sched_with_refs(years)
    fut = sched[(sched['gameday'].dt.tz_localize(None) >= today) &
                (sched['home_team'].notna()) & (sched['away_team'].notna())].copy()
    df = pd.DataFrame({
        'game_id': fut['game_id'],
        'date': pd.to_datetime(fut['gameday']).dt.tz_localize(None),
        'season': fut['season'].astype(int),
        'week': fut['week'].astype(int),
        'home_team': fut['home_team'].map(norm_team),
        'away_team': fut['away_team'].map(norm_team),
        'referee': fut['referee'],
    }).sort_values(['season','week','date']).reset_index(drop=True)
    return df

# ---------------------------
# Officiating: staged, leak-free crew metrics
# ---------------------------
CREW_CACHE_DIR = os.path.join(CACHE_DIR, "crew_cache")
os.makedirs(CREW_CACHE_DIR, exist_ok=True)

def _extract_referee_column(sched: pd.DataFrame) -> pd.Series:
    for c in ['referee', 'referee_full_name', 'referee_name', 'official', 'crew']:
        if c in sched.columns:
            s = sched[c].astype(str)
            s = s.where(s.str.strip().astype(bool), other=np.nan)
            return s
    return pd.Series([np.nan] * len(sched))

def _import_sched_with_refs(years: List[int]) -> pd.DataFrame:
    s = nfl.import_schedules(years).copy()
    s['gameday'] = pd.to_datetime(s['gameday'], errors='coerce')
    s['season'] = s['season'].astype(int)
    s['week'] = s['week'].astype(int)
    s['home_team'] = s['home_team'].map(norm_team)
    s['away_team'] = s['away_team'].map(norm_team)
    if 'game_id' not in s.columns:
        s['game_id'] = (
            s['season'].astype(str) + '-' + s['week'].astype(str) + '-' + s['home_team'] + '-' + s['away_team']
        )
    s['referee'] = _extract_referee_column(s)
    # game_type exists in nfl_data_py schedules; if not, add REG as default
    if 'game_type' not in s.columns:
        s['game_type'] = 'REG'
    return s[['game_id','season','week','gameday','home_team','away_team','referee','game_type']]

def compute_ref_metrics_officiating_staged(sched: pd.DataFrame, pbp: pd.DataFrame, k_prior: float = 80.0) -> pd.DataFrame:
    """
    For each scheduled game, compute pre-game officiating metrics for its referee using
    only prior games of that same referee. Output one row per game with pre metrics:
    ['game_id','season','week','date','home_team','away_team','referee',
     'ref_pen_pg_pre','ref_home_bias_pre','ref_dpi_rate_pre','ref_hold_rate_pre'].
    """
    sch = sched.sort_values(['gameday','season','week']).copy()
    sch['date'] = pd.to_datetime(sch['gameday']).dt.tz_localize(None)
    pbp = pbp.copy()
    if 'season_type' in pbp.columns:
        pbp = pbp[pbp['season_type'].astype(str).str.upper() == 'REG']
    # penalties only
    if 'penalty' in pbp.columns:
        pbp = pbp[pbp['penalty'] == 1]
    # Ensure needed cols
    keep_cols = ['game_id','penalty_team','home_team','away_team','penalty_type']
    for c in keep_cols:
        if c not in pbp.columns:
            pbp[c] = np.nan
    pen = pbp[keep_cols].copy()
    pen['on_home'] = (pen['penalty_team'] == pen['home_team']).astype('float')
    pen['on_away'] = (pen['penalty_team'] == pen['away_team']).astype('float')

    # League priors
    per_game_counts = pen.groupby('game_id', dropna=True).size()
    league_pen_pg = float(per_game_counts.mean()) if len(per_game_counts) else 12.0

    out = []
    sch_ref = sch[['game_id','date','referee']].copy()
    for _, row in sch.iterrows():
        gid = row['game_id']; ref = row.get('referee'); dt = row['date']
        if not isinstance(ref, str) or not ref:
            out.append({
                'game_id': gid, 'season': int(row['season']), 'week': int(row['week']), 'date': dt,
                'home_team': row['home_team'], 'away_team': row['away_team'], 'referee': np.nan,
                'ref_pen_pg_pre': league_pen_pg, 'ref_home_bias_pre': 0.0,
                'ref_dpi_rate_pre': 0.0, 'ref_hold_rate_pre': 0.0,
            })
            continue
        prior_ids = sch_ref[(sch_ref['referee'] == ref) & (sch_ref['date'] < dt)]['game_id'].tolist()
        if len(prior_ids) == 0:
            out.append({
                'game_id': gid, 'season': int(row['season']), 'week': int(row['week']), 'date': dt,
                'home_team': row['home_team'], 'away_team': row['away_team'], 'referee': ref,
                'ref_pen_pg_pre': league_pen_pg, 'ref_home_bias_pre': 0.0,
                'ref_dpi_rate_pre': 0.0, 'ref_hold_rate_pre': 0.0,
            })
            continue
        pen_prior = pen[pen['game_id'].isin(prior_ids)]
        total_calls = int(len(pen_prior))
        games_calls = int(pen_prior['game_id'].nunique())
        pen_pg = float(total_calls / games_calls) if games_calls else league_pen_pg

        home_calls = float(pen_prior['on_home'].sum())
        away_calls = float(pen_prior['on_away'].sum())
        denom = home_calls + away_calls
        bias = float((away_calls - home_calls) / denom) if denom > 0 else 0.0

        pt = pen_prior['penalty_type'].astype(str)
        dpi = float((pt == 'Defensive Pass Interference').sum())
        hold = float(pt.str.contains('Holding', na=False).sum())
        # Shrink rates toward 0 by k_prior pseudo-events
        dpi_rate = (dpi + 0.0 * k_prior) / (total_calls + k_prior) if (total_calls + k_prior) > 0 else 0.0
        hold_rate = (hold + 0.0 * k_prior) / (total_calls + k_prior) if (total_calls + k_prior) > 0 else 0.0

        out.append({
            'game_id': gid, 'season': int(row['season']), 'week': int(row['week']), 'date': dt,
            'home_team': row['home_team'], 'away_team': row['away_team'], 'referee': ref,
            'ref_pen_pg_pre': float(pen_pg), 'ref_home_bias_pre': float(bias),
            'ref_dpi_rate_pre': float(dpi_rate), 'ref_hold_rate_pre': float(hold_rate),
        })
    return pd.DataFrame(out)

def get_crew_pre_table_for_years(years: List[int]) -> pd.DataFrame:
    years = sorted(set(int(y) for y in years))
    if not years:
        return pd.DataFrame(columns=[
            'game_id','season','week','date','home_team','away_team','referee',
            'ref_pen_pg_pre','ref_home_bias_pre','ref_dpi_rate_pre','ref_hold_rate_pre'
        ])
    key = f"{min(years)}_{max(years)}"
    cache_path = os.path.join(CREW_CACHE_DIR, f"crew_pre_{key}.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass
    sched = _import_sched_with_refs(years)
    pbp = nfl.import_pbp_data(years)
    crew = compute_ref_metrics_officiating_staged(sched, pbp)
    try:
        crew.to_parquet(cache_path)
    except Exception:
        pass
    return crew

# ---------------------------
# Feature engineering
# ---------------------------
FEATURES = [
    'is_home', 'opp_is_home',
    'team_pts_for_l5', 'team_pts_against_l5', 'team_margin_l5',
    'opp_pts_for_l5', 'opp_pts_against_l5', 'opp_margin_l5',
    'off_adv_l5', 'pace_proxy_l5',
    'team_elo_pre', 'opp_elo_pre', 'elo_diff_pre',
    'team_off_pre', 'opp_def_pre', 'off_minus_def_pre',
    'ref_pen_pg_pre', 'ref_home_bias_pre', 'ref_dpi_rate_pre', 'ref_hold_rate_pre',
    'team_days_rest', 'opp_days_rest',
]

# ---------------------------
# Simple Elo for team-strength feature (no leakage)
# ---------------------------
def compute_elo_on_games(df_games: pd.DataFrame, base: float = 1500.0) -> pd.DataFrame:
    g = df_games.sort_values(['date','season','week']).reset_index(drop=True).copy()
    teams = pd.unique(g[['home_team','away_team']].values.ravel('K'))
    elo = {t: base for t in teams}
    HFA = 55.0   # Elo points of home-field advantage
    K = 20.0

    pre_home, pre_away = [], []
    for _, row in g.iterrows():
        h, a = row['home_team'], row['away_team']
        eh, ea = elo.get(h, base), elo.get(a, base)
        pre_home.append(eh)
        pre_away.append(ea)

        diff = (eh + HFA) - ea
        exp_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        exp_away = 1.0 - exp_home

        margin = float(row['home_points'] - row['away_points'])
        if margin == 0:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = (1.0, 0.0) if margin > 0 else (0.0, 1.0)

        mult = np.log(abs(margin) + 1.0) * (2.2 / ((diff * 0.001) + 2.2))
        elo[h] = eh + K * mult * (s_home - exp_home)
        elo[a] = ea + K * mult * (s_away - exp_away)

    g['pre_elo_home'] = pre_home
    g['pre_elo_away'] = pre_away
    return g[['date','season','week','home_team','away_team','pre_elo_home','pre_elo_away']]

from sklearn.linear_model import Ridge

def compute_off_def_ratings(df_games: pd.DataFrame, alpha: float = 10.0) -> pd.DataFrame:
    # Build design matrix: points_for ~ HFA*is_home + Off(team) - Def(opponent)
    g = df_games.sort_values(['date']).copy()
    teams = sorted(pd.unique(g[['home_team','away_team']].values.ravel('K')))
    idx = {t:i for i,t in enumerate(teams)}
    n = len(g)
    m = len(teams)

    # Home rows
    Xh = np.zeros((n, 1 + 2*m))  # [HFA | Off(m) | Def(m)]
    yh = g['home_points'].values.astype(float)
    Xh[:,0] = 1.0  # HFA bias
    for i,(h,a) in enumerate(zip(g['home_team'], g['away_team'])):
        Xh[i, 1 + idx[h]] = 1.0
        Xh[i, 1 + m + idx[a]] = -1.0

    # Away rows (no HFA)
    Xa = np.zeros((n, 1 + 2*m))
    ya = g['away_points'].values.astype(float)
    for i,(h,a) in enumerate(zip(g['home_team'], g['away_team'])):
        Xa[i, 1 + idx[a]] = 1.0
        Xa[i, 1 + m + idx[h]] = -1.0

    X = np.vstack([Xh, Xa])
    y = np.concatenate([yh, ya])

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y)
    HFA = float(model.coef_[0])
    Off = model.coef_[1:1+m]
    Def = model.coef_[1+m:1+2*m]

    df_off = pd.DataFrame({'team': teams, 'off_rating': Off})
    df_def = pd.DataFrame({'team': teams, 'def_rating': Def})
    return df_off.merge(df_def, on='team').assign(hfa=HFA)

def compute_off_def_ratings_staged(df_games: pd.DataFrame, alpha: float = 10.0) -> pd.DataFrame:
    """
    Compute schedule-adjusted offense/defense ratings for each team before each game,
    by solving a ridge system on season-to-date games strictly before that game.
    Returns a row per actual game with pre-game ratings for both teams.
    """
    out_rows = []
    for season, g_seas in df_games.sort_values(['date']).groupby('season', sort=False):
        g = g_seas.reset_index(drop=True)
        teams = sorted(pd.unique(g[['home_team','away_team']].values.ravel('K')))
        idx = {t: i for i, t in enumerate(teams)}
        m = len(teams)

        for i in range(len(g)):
            # No prior games in season → zero priors
            if i == 0:
                out_rows.append({
                    'season': int(season),
                    'date': g.loc[i, 'date'],
                    'week': int(g.loc[i, 'week']),
                    'home_team': g.loc[i, 'home_team'],
                    'away_team': g.loc[i, 'away_team'],
                    'off_pre_home': 0.0, 'def_pre_home': 0.0,
                    'off_pre_away': 0.0, 'def_pre_away': 0.0,
                })
                continue

            sub = g.iloc[:i]
            n = len(sub)

            # Design: points_for ≈ HFA*is_home + Off(team) − Def(opponent)
            Xh = np.zeros((n, 1 + 2*m))
            yh = sub['home_points'].values.astype(float)
            Xh[:, 0] = 1.0
            for r, (h, a) in enumerate(zip(sub['home_team'], sub['away_team'])):
                Xh[r, 1 + idx[h]] = 1.0
                Xh[r, 1 + m + idx[a]] = -1.0

            Xa = np.zeros((n, 1 + 2*m))
            ya = sub['away_points'].values.astype(float)
            for r, (h, a) in enumerate(zip(sub['home_team'], sub['away_team'])):
                Xa[r, 1 + idx[a]] = 1.0
                Xa[r, 1 + m + idx[h]] = -1.0

            X = np.vstack([Xh, Xa])
            y = np.concatenate([yh, ya])

            try:
                ridge = Ridge(alpha=alpha, fit_intercept=False)
                ridge.fit(X, y)
                Off = ridge.coef_[1:1+m]
                Def = ridge.coef_[1+m:1+2*m]
                off_map = {t: float(Off[idx[t]]) for t in teams}
                def_map = {t: float(Def[idx[t]]) for t in teams}
            except Exception:
                # If the system is singular early in season, fall back to zeros
                off_map = {t: 0.0 for t in teams}
                def_map = {t: 0.0 for t in teams}

            h, a = g.loc[i, 'home_team'], g.loc[i, 'away_team']
            out_rows.append({
                'season': int(season),
                'date': g.loc[i, 'date'],
                'week': int(g.loc[i, 'week']),
                'home_team': h,
                'away_team': a,
                'off_pre_home': off_map.get(h, 0.0),
                'def_pre_home': def_map.get(h, 0.0),
                'off_pre_away': off_map.get(a, 0.0),
                'def_pre_away': def_map.get(a, 0.0),
            })
    return pd.DataFrame(out_rows)

def make_team_week(df_games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df_games.iterrows():
        rows.append({
            'date': g['date'], 'season': g['season'], 'week': g['week'],
            'team': g['home_team'], 'opponent': g['away_team'], 'is_home': 1,
            'points_for': g['home_points'], 'points_against': g['away_points']
        })
        rows.append({
            'date': g['date'], 'season': g['season'], 'week': g['week'],
            'team': g['away_team'], 'opponent': g['home_team'], 'is_home': 0,
            'points_for': g['away_points'], 'points_against': g['home_points']
        })
    tw = pd.DataFrame(rows)
    tw = tw.sort_values(['team','season','week','date']).reset_index(drop=True)

    # Merge pregame Elo without leakage
    elo_df = compute_elo_on_games(df_games)

    # Map pregame Elo for the row orientation
    _tw = tw.merge(
        elo_df.rename(columns={
            'home_team':'team', 'away_team':'opponent',
            'pre_elo_home':'team_elo_pre', 'pre_elo_away':'opp_elo_pre'
        }),
        on=['date','season','week','team','opponent'], how='left'
    )

    # Also handle swapped orientation (away rows)
    _tw = _tw.merge(
        elo_df.rename(columns={
            'home_team':'opponent', 'away_team':'team',
            'pre_elo_home':'opp_elo_pre_swap', 'pre_elo_away':'team_elo_pre_swap'
        }),
        on=['date','season','week','team','opponent'], how='left'
    )

    # Coalesce whichever side applied
    _tw['team_elo_pre'] = _tw['team_elo_pre'].fillna(_tw['team_elo_pre_swap'])
    _tw['opp_elo_pre'] = _tw['opp_elo_pre'].fillna(_tw['opp_elo_pre_swap'])
    _tw = _tw.drop(columns=['team_elo_pre_swap','opp_elo_pre_swap'])
    _tw['elo_diff_pre'] = _tw['team_elo_pre'] - _tw['opp_elo_pre']

    tw = _tw

    # Leak-free schedule-adjusted ratings (season-to-date before each game)
    od = compute_off_def_ratings_staged(df_games)

    # Orientation A: team == home, opponent == away
    map_a = od.rename(columns={
        'home_team': 'team',
        'away_team': 'opponent',
        'off_pre_home': 'team_off_pre',
        'def_pre_home': 'team_def_pre',
        'def_pre_away': 'opp_def_pre',
    })[['date','season','week','team','opponent','team_off_pre','team_def_pre','opp_def_pre']]

    # Orientation B: team == away, opponent == home
    map_b = od.rename(columns={
        'home_team': 'opponent',
        'away_team': 'team',
        'off_pre_away': 'team_off_pre_swap',
        'def_pre_away': 'team_def_pre_swap',
        'def_pre_home': 'opp_def_pre_swap',
    })[['date','season','week','team','opponent','team_off_pre_swap','team_def_pre_swap','opp_def_pre_swap']]

    tw = tw.merge(map_a, on=['date','season','week','team','opponent'], how='left')
    tw = tw.merge(map_b, on=['date','season','week','team','opponent'], how='left')

    # Coalesce orientation
    tw['team_off_pre'] = tw['team_off_pre'].fillna(tw['team_off_pre_swap'])
    tw['team_def_pre'] = tw['team_def_pre'].fillna(tw['team_def_pre_swap'])
    tw['opp_def_pre']  = tw['opp_def_pre'].fillna(tw['opp_def_pre_swap'])
    tw = tw.drop(columns=['team_off_pre_swap','team_def_pre_swap','opp_def_pre_swap'])

    # Composite feature
    tw['off_minus_def_pre'] = tw['team_off_pre'] - tw['opp_def_pre']

    # Officiating crew features: leak-free per game, same for both teams
    crew_years = list(sorted(pd.unique(df_games['season'].astype(int))))
    crew_pre = get_crew_pre_table_for_years(crew_years)

    crew_a = crew_pre.rename(columns={
        'home_team':'team', 'away_team':'opponent'
    })[['date','season','week','team','opponent',
        'ref_pen_pg_pre','ref_home_bias_pre','ref_dpi_rate_pre','ref_hold_rate_pre']]

    crew_b = crew_pre.rename(columns={
        'home_team': 'opponent',
        'away_team': 'team'
    })[['date','season','week','team','opponent',
        'ref_pen_pg_pre','ref_home_bias_pre','ref_dpi_rate_pre','ref_hold_rate_pre']].copy()
    _feat_cols = ['ref_pen_pg_pre','ref_home_bias_pre','ref_dpi_rate_pre','ref_hold_rate_pre']
    crew_b = pd.concat(
        [crew_b[['date','season','week','team','opponent']],
         crew_b[_feat_cols].add_suffix('_swap')],
        axis=1
    )

    tw = tw.merge(crew_a, on=['date','season','week','team','opponent'], how='left')
    tw = tw.merge(crew_b, on=['date','season','week','team','opponent'], how='left')

    # Coalesce orientation (features are identical per game)
    for col in ['ref_pen_pg_pre','ref_home_bias_pre','ref_dpi_rate_pre','ref_hold_rate_pre']:
        swap_col = col + '_swap'
        if swap_col in tw.columns:
            tw[col] = tw[col].fillna(tw[swap_col])
            tw.drop(columns=[swap_col], inplace=True)

    # Fill remaining officiating NaNs with neutral priors
    tw['ref_pen_pg_pre'] = tw['ref_pen_pg_pre'].fillna(tw['ref_pen_pg_pre'].median() if 'ref_pen_pg_pre' in tw.columns else 12.0)
    tw['ref_home_bias_pre'] = tw['ref_home_bias_pre'].fillna(0.0)
    tw['ref_dpi_rate_pre'] = tw['ref_dpi_rate_pre'].fillna(0.0)
    tw['ref_hold_rate_pre'] = tw['ref_hold_rate_pre'].fillna(0.0)

    # Rolling stats
    tw['team_pts_for_l5'] = tw.groupby('team')['points_for'].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    tw['team_pts_against_l5'] = tw.groupby('team')['points_against'].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    tw['margin'] = tw['points_for'] - tw['points_against']
    tw['team_margin_l5'] = tw.groupby('team')['margin'].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())

    # Opponent rolling via self-join
    opp = tw[['team','season','week','team_pts_for_l5','team_pts_against_l5','team_margin_l5']].rename(columns={
        'team':'opponent',
        'team_pts_for_l5':'opp_pts_for_l5',
        'team_pts_against_l5':'opp_pts_against_l5',
        'team_margin_l5':'opp_margin_l5'
    })
    tw = tw.merge(opp, on=['opponent','season','week'], how='left')

    # Recompute off_adv_l5 and pace_proxy_l5 after opponent merge
    tw['off_adv_l5'] = tw['team_pts_for_l5'] - tw['opp_pts_against_l5']
    tw['pace_proxy_l5'] = tw['team_pts_for_l5'] + tw['opp_pts_for_l5']

    # Rest days
    tw['prev_date'] = tw.groupby('team')['date'].shift(1)
    tw['team_days_rest'] = (tw['date'] - tw['prev_date']).dt.days.clip(lower=3)
    tw['team_days_rest'] = tw['team_days_rest'].fillna(7)

    opp_dates = tw[['team','season','week','date']].rename(columns={'team':'opponent','date':'opp_date'})
    tw = tw.merge(opp_dates, on=['opponent','season','week'], how='left')
    tw['opp_prev_date'] = tw.groupby('opponent')['opp_date'].shift(1)
    tw['opp_days_rest'] = (tw['opp_date'] - tw['opp_prev_date']).dt.days.clip(lower=3)
    tw['opp_days_rest'] = tw['opp_days_rest'].fillna(7)

    tw['opp_is_home'] = 1 - tw['is_home']
    tw = tw.drop(columns=['prev_date','opp_prev_date','opp_date','margin'])
    return tw

@dataclass
class ModelArtifacts:
    pts_model: PoissonRegressor
    scaler: StandardScaler
    feat_cols: List[str]
    cal_model: IsotonicRegression
    sigma_points: float
    sigma_margin: float
    sigma_total: float
    meta: Dict

def fit_models(team_week: pd.DataFrame, min_season: int) -> ModelArtifacts:
    # Build supervised dataset with valid features only
    feat_rows = []
    for _, row in team_week.iterrows():
        if any(pd.isna(row.get(c)) for c in FEATURES):
            continue
        feat_rows.append({
            'season': row['season'], 'week': row['week'], 'date': row['date'],
            'team': row['team'], 'opponent': row['opponent'],
            **{c: row[c] for c in FEATURES},
            'y': row['points_for']
        })
    ds = pd.DataFrame(feat_rows)
    ds = ds[ds['season'] >= int(min_season)].sort_values(['season','week','date']).reset_index(drop=True)

    X = ds[FEATURES].values
    y = ds['y'].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Time-aware split: last season for validation when possible
    last_season = int(ds['season'].max())
    train_mask = ds['season'] < last_season
    if train_mask.sum() < 100:
        tscv = TimeSeriesSplit(n_splits=5)
        tr_idx, va_idx = list(tscv.split(Xs))[-1]
    else:
        tr_idx = np.where(train_mask)[0]
        va_idx = np.where(~train_mask)[0]

    # Poisson regression for points (nonnegative, better for count-like targets)
    model = PoissonRegressor(alpha=1.0, max_iter=2000, tol=1e-6)

    # Recency weights: half-life = 8 weeks
    weeks = (ds['date'].max() - ds['date']).dt.days / 7.0
    sample_w = np.power(0.5, weeks / 8.0).values

    # Fit with sample weights
    model.fit(Xs[tr_idx], y[tr_idx], sample_weight=sample_w[tr_idx])

    # Validation predictions
    y_pred_va = model.predict(Xs[va_idx])
    y_pred_va = np.clip(y_pred_va, 0, None)
    sigma_points = float(np.std(y[va_idx] - y_pred_va))

    # Build paired margins on validation
    va = ds.iloc[va_idx].copy()
    va['pred_points'] = y_pred_va
    key = va['date'].astype(str) + '|' + va['team'] + '|' + va['opponent']
    pair_key = va['date'].astype(str) + '|' + va['opponent'] + '|' + va['team']
    pairs = va.set_index(key).join(va.set_index(pair_key)[['pred_points','y']].add_prefix('opp_'), how='inner')

    if len(pairs) >= 50:
        pred_margin = (pairs['pred_points'] - pairs['opp_pred_points']).values
        pred_total  = (pairs['pred_points'] + pairs['opp_pred_points']).values

        actual_margin = (pairs['y'] - pairs['opp_y']).values
        actual_total  = (pairs['y'] + pairs['opp_y']).values

        # Isotonic calibration for P(home win) as a function of predicted margin
        cal_model = IsotonicRegression(out_of_bounds='clip')
        cal_model.fit(pred_margin, (actual_margin > 0).astype(int))

        # Empirical residual spreads
        resid_margin = actual_margin - pred_margin
        resid_total  = actual_total  - pred_total

        sigma_margin = float(np.std(resid_margin))
        sigma_total  = float(np.std(resid_total))

        # Brier with isotonic transform
        brier = brier_score_loss((actual_margin > 0).astype(int), cal_model.transform(pred_margin))
    else:
        # Fallback: nearly identity mapping on margins
        cal_model = IsotonicRegression(out_of_bounds='clip')
        xs = np.linspace(-30, 30, 300)
        ys = (xs > 0).astype(int)
        cal_model.fit(xs, ys)
        sigma_margin = max(7.0, sigma_points * math.sqrt(2.0))
        sigma_total  = max(10.0, sigma_points * math.sqrt(2.0))
        brier = None

    meta = {
        'features': FEATURES,
        'sigma_points': sigma_points,
        'sigma_margin': sigma_margin,
        'sigma_total': sigma_total,
        'last_season': last_season,
        'min_season': int(min_season),
        'model': 'PoissonRegressor(points_for) + Isotonic(margin→P)',
        'metrics': {
            'val_points_mae': float(mean_absolute_error(y[va_idx], y_pred_va)),
            'val_brier_home_win': None if brier is None else float(brier),
        }
    }

    return ModelArtifacts(
        pts_model=model,
        scaler=scaler,
        feat_cols=list(FEATURES),
        cal_model=cal_model,
        sigma_points=sigma_points,
        sigma_margin=sigma_margin,
        sigma_total=sigma_total,
        meta=meta,
    )

def save_artifacts(art: ModelArtifacts, out_path: str) -> None:
    obj = {
        'pts_model': art.pts_model,
        'scaler': art.scaler,
        'feat_cols': art.feat_cols,
        'cal_model': art.cal_model,
        'sigma_points': art.sigma_points,
        'sigma_margin': art.sigma_margin,
        'sigma_total': art.sigma_total,
        'meta': art.meta,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(obj, out_path)

def load_artifacts(path: str) -> ModelArtifacts:
    obj = joblib.load(path)
    return ModelArtifacts(
        pts_model=obj['pts_model'],
        scaler=obj['scaler'],
        feat_cols=obj['feat_cols'],
        cal_model=obj['cal_model'],
        sigma_points=float(obj['sigma_points']),
        sigma_margin=float(obj['sigma_margin']),
        sigma_total=float(obj['sigma_total']),
        meta=obj['meta']
    )

# ---------------------------
# Prediction for a slate
# ---------------------------
@dataclass
class GamePred:
    season: int
    week: int
    date: pd.Timestamp
    home: str
    away: str
    home_pts: float
    away_pts: float
    total: float
    margin: float
    p_home_win: float
    sigma_total: float
    sigma_margin: float

def predict_week(art, hist_tw: pd.DataFrame, matchups: pd.DataFrame, injury_adj: Optional[pd.DataFrame] = None, weather_tbl: Optional[pd.DataFrame] = None) -> List['GamePred']:
    """
    Produce predictions for upcoming matchups using the trained artifacts (art),
    historical team-week table (hist_tw), and a matchups DataFrame with at least:
    ['season','week','date','home_team','away_team'].

    Returns a list[GamePred].
    """
    # Safety: enforce column names we rely on
    required_matchup_cols = {'season','week','date','home_team','away_team'}
    missing = required_matchup_cols - set(matchups.columns)
    if missing:
        raise ValueError(f"matchups missing required columns: {sorted(missing)}")

    # Prepare officiating table for relevant seasons (history + upcoming)
    years = sorted(set(pd.concat([matchups['season'], hist_tw['season']]).astype(int).tolist()))
    crew_pre = get_crew_pre_table_for_years(years)

    inj_df = None
    if injury_adj is not None and not injury_adj.empty:
        inj_df = injury_adj.copy()
    wtbl = None
    if weather_tbl is not None and not weather_tbl.empty:
        wtbl = weather_tbl.copy()

    # Local helpers
    def last5(s: pd.Series) -> float:
        return float(s.tail(5).mean()) if len(s) else 0.0

    # Make sure we only use history strictly before game date
    hist_tw = hist_tw.sort_values('date').reset_index(drop=True)

    results: List['GamePred'] = []

    for _, g in matchups.iterrows():
        game_date = pd.to_datetime(g['date'])
        home = norm_team(g['home_team'])
        away = norm_team(g['away_team'])

        # Historical windows up to the day before this game
        hist_home = hist_tw[(hist_tw['team'] == home) & (hist_tw['date'] < game_date)].sort_values('date')
        hist_away = hist_tw[(hist_tw['team'] == away) & (hist_tw['date'] < game_date)].sort_values('date')

        # Officiating metrics for this game (same for both teams)
        c_row = crew_pre[
            (crew_pre['season'] == int(g['season'])) &
            (crew_pre['week'] == int(g['week'])) &
            (crew_pre['home_team'] == home) &
            (crew_pre['away_team'] == away)
        ]
        if c_row.empty:
            c_row = crew_pre[
                (crew_pre['season'] == int(g['season'])) &
                (crew_pre['week'] == int(g['week'])) &
                (crew_pre['home_team'] == away) &
                (crew_pre['away_team'] == home)
            ]
        if c_row.empty or c_row[['ref_pen_pg_pre','ref_home_bias_pre','ref_dpi_rate_pre','ref_hold_rate_pre']].isna().all(axis=None):
            # Neutral priors if unknown
            ref_pen_pg_pre = float(crew_pre['ref_pen_pg_pre'].median()) if 'ref_pen_pg_pre' in crew_pre.columns and not crew_pre.empty else 12.0
            ref_home_bias_pre = 0.0
            ref_dpi_rate_pre = 0.0
            ref_hold_rate_pre = 0.0
        else:
            rr = c_row.iloc[0]
            ref_pen_pg_pre = float(rr.get('ref_pen_pg_pre', np.nan)) if not pd.isna(rr.get('ref_pen_pg_pre', np.nan)) else float(crew_pre['ref_pen_pg_pre'].median())
            ref_home_bias_pre = float(rr.get('ref_home_bias_pre', 0.0))
            ref_dpi_rate_pre = float(rr.get('ref_dpi_rate_pre', 0.0))
            ref_hold_rate_pre = float(rr.get('ref_hold_rate_pre', 0.0))

        # Elo pregame from most recent history
        home_elo = float(hist_home['team_elo_pre'].iloc[-1]) if 'team_elo_pre' in hist_home.columns and not hist_home.empty else 1500.0
        away_elo = float(hist_away['team_elo_pre'].iloc[-1]) if 'team_elo_pre' in hist_away.columns and not hist_away.empty else 1500.0

        # Schedule-adjusted Off/Def pre ratings from most recent history
        home_off = float(hist_home['team_off_pre'].iloc[-1]) if 'team_off_pre' in hist_home.columns and not hist_home.empty else 0.0
        home_def = float(hist_home['team_def_pre'].iloc[-1]) if 'team_def_pre' in hist_home.columns and not hist_home.empty else 0.0
        away_off = float(hist_away['team_off_pre'].iloc[-1]) if 'team_off_pre' in hist_away.columns and not hist_away.empty else 0.0
        away_def = float(hist_away['team_def_pre'].iloc[-1]) if 'team_def_pre' in hist_away.columns and not hist_away.empty else 0.0

        # Home feature row
        home_row = {
            'is_home': 1, 'opp_is_home': 0,
            'team_pts_for_l5': last5(hist_home['points_for'] if 'points_for' in hist_home.columns else hist_home.get('team_points', pd.Series(dtype=float))),
            'team_pts_against_l5': last5(hist_home['points_against'] if 'points_against' in hist_home.columns else hist_home.get('opp_points', pd.Series(dtype=float))),
            'team_margin_l5': last5((hist_home['points_for'] - hist_home['points_against']) if all(k in hist_home.columns for k in ['points_for','points_against']) else (hist_home.get('team_points', pd.Series(dtype=float)) - hist_home.get('opp_points', pd.Series(dtype=float)))),
            'opp_pts_for_l5': last5(hist_away['points_for'] if 'points_for' in hist_away.columns else hist_away.get('team_points', pd.Series(dtype=float))),
            'opp_pts_against_l5': last5(hist_away['points_against'] if 'points_against' in hist_away.columns else hist_away.get('opp_points', pd.Series(dtype=float))),
            'opp_margin_l5': last5((hist_away['points_for'] - hist_away['points_against']) if all(k in hist_away.columns for k in ['points_for','points_against']) else (hist_away.get('team_points', pd.Series(dtype=float)) - hist_away.get('opp_points', pd.Series(dtype=float)))),
            'team_days_rest': float((game_date - hist_home['date'].iloc[-1]).days) if not hist_home.empty else 7.0,
            'opp_days_rest': float((game_date - hist_away['date'].iloc[-1]).days) if not hist_away.empty else 7.0,
            # Elo
            'team_elo_pre': home_elo,
            'opp_elo_pre': away_elo,
            'elo_diff_pre': home_elo - away_elo,
            # Staged Off/Def
            'team_off_pre': home_off,
            'opp_def_pre': away_def,
            'off_minus_def_pre': home_off - away_def,
            # Crew (same both rows)
            'ref_pen_pg_pre': ref_pen_pg_pre,
            'ref_home_bias_pre': ref_home_bias_pre,
            'ref_dpi_rate_pre': ref_dpi_rate_pre,
            'ref_hold_rate_pre': ref_hold_rate_pre,
        }
        # Derived matchup features
        home_row['off_adv_l5'] = home_row['team_pts_for_l5'] - home_row['opp_pts_against_l5']
        home_row['pace_proxy_l5'] = home_row['team_pts_for_l5'] + home_row['opp_pts_for_l5']

        # Away feature row
        away_row = {
            'is_home': 0, 'opp_is_home': 1,
            'team_pts_for_l5': last5(hist_away['points_for'] if 'points_for' in hist_away.columns else hist_away.get('team_points', pd.Series(dtype=float))),
            'team_pts_against_l5': last5(hist_away['points_against'] if 'points_against' in hist_away.columns else hist_away.get('opp_points', pd.Series(dtype=float))),
            'team_margin_l5': last5((hist_away['points_for'] - hist_away['points_against']) if all(k in hist_away.columns for k in ['points_for','points_against']) else (hist_away.get('team_points', pd.Series(dtype=float)) - hist_away.get('opp_points', pd.Series(dtype=float)))),
            'opp_pts_for_l5': last5(hist_home['points_for'] if 'points_for' in hist_home.columns else hist_home.get('team_points', pd.Series(dtype=float))),
            'opp_pts_against_l5': last5(hist_home['points_against'] if 'points_against' in hist_home.columns else hist_home.get('opp_points', pd.Series(dtype=float))),
            'opp_margin_l5': last5((hist_home['points_for'] - hist_home['points_against']) if all(k in hist_home.columns for k in ['points_for','points_against']) else (hist_home.get('team_points', pd.Series(dtype=float)) - hist_home.get('opp_points', pd.Series(dtype=float)))),
            'team_days_rest': float((game_date - hist_away['date'].iloc[-1]).days) if not hist_away.empty else 7.0,
            'opp_days_rest': float((game_date - hist_home['date'].iloc[-1]).days) if not hist_home.empty else 7.0,
            # Elo
            'team_elo_pre': away_elo,
            'opp_elo_pre': home_elo,
            'elo_diff_pre': away_elo - home_elo,
            # Staged Off/Def
            'team_off_pre': away_off,
            'opp_def_pre': home_def,
            'off_minus_def_pre': away_off - home_def,
            # Crew (same both rows)
            'ref_pen_pg_pre': ref_pen_pg_pre,
            'ref_home_bias_pre': ref_home_bias_pre,
            'ref_dpi_rate_pre': ref_dpi_rate_pre,
            'ref_hold_rate_pre': ref_hold_rate_pre,
        }
        away_row['off_adv_l5'] = away_row['team_pts_for_l5'] - away_row['opp_pts_against_l5']
        away_row['pace_proxy_l5'] = away_row['team_pts_for_l5'] + away_row['opp_pts_for_l5']

        # Build feature matrix in trained column order
        def row_to_X(row: Dict[str, float]) -> np.ndarray:
            return np.array([row.get(c, 0.0) for c in art.feat_cols], dtype=float)

        X_home = row_to_X(home_row)[None, :]
        X_away = row_to_X(away_row)[None, :]

        # Scale then predict base points
        Xh_s = art.scaler.transform(X_home)
        Xa_s = art.scaler.transform(X_away)
        base_home_pts = float(max(0.0, art.pts_model.predict(Xh_s)[0]))
        base_away_pts = float(max(0.0, art.pts_model.predict(Xa_s)[0]))
        base_total = base_home_pts + base_away_pts
        base_margin = base_home_pts - base_away_pts

        # Start with base means and sigmas
        total = base_total
        margin = base_margin
        sigma_total = float(art.sigma_total)
        sigma_margin = float(art.sigma_margin)

        # Apply injuries (if provided). Assume injury_delta_spread is AWAY-centric; convert to home margin.
        if inj_df is not None:
            inj_row = None
            if 'game_id' in matchups.columns:
                inj_row = inj_df[inj_df.get('game_id', pd.Series(dtype=str)) == g.get('game_id')].head(1)
            if inj_row is None or inj_row.empty:
                if all(c in inj_df.columns for c in ['season','week','home_team','away_team']):
                    inj_row = inj_df[
                        (inj_df['season'] == int(g['season'])) &
                        (inj_df['week'] == int(g['week'])) &
                        (inj_df['home_team'] == home) &
                        (inj_df['away_team'] == away)
                    ].head(1)
            if inj_row is not None and not inj_row.empty:
                try:
                    ds = float(inj_row.iloc[0].get('injury_delta_spread', 0.0))
                    dt = float(inj_row.iloc[0].get('injury_delta_total', 0.0))
                    margin += -ds  # convert away-centric to home-centric
                    total  += dt
                except Exception:
                    pass

        # Apply weather (if provided)
        if wtbl is not None:
            w_row = None
            if 'game_id' in matchups.columns:
                w_row = wtbl[wtbl.get('game_id', pd.Series(dtype=str)) == g.get('game_id')].head(1)
            if w_row is None or w_row.empty:
                w_row = wtbl[
                    (wtbl['season'] == int(g['season'])) &
                    (wtbl['week'] == int(g['week'])) &
                    (wtbl['home_team'] == home) &
                    (wtbl['away_team'] == away)
                ].head(1)
            if w_row is not None and not w_row.empty:
                dtot, dsig_tot, dsig_mar = compute_weather_adjustments(w_row.iloc[0])
                total += dtot
                sigma_total += dsig_tot
                sigma_margin += dsig_mar

        # Apply travel (timezone + short rest) using computed rest days from rows
        home_rest = float(home_row.get('team_days_rest', 7.0))
        away_rest = float(away_row.get('team_days_rest', 7.0))
        dmar, dsmar = compute_travel_adjustments(home, away, home_rest, away_rest)
        margin += dmar
        sigma_margin += dsmar

        # Reconcile points to adjusted total/margin
        home_pts = max(0.0, (total + margin) / 2.0)
        away_pts = max(0.0, total - home_pts)

        # Calibrated home win probability via isotonic on adjusted margin
        p_home = float(art.cal_model.transform(np.array([margin]))[0])
        p_home = float(min(max(p_home, 1e-4), 1 - 1e-4))

        results.append(GamePred(
            season=int(g['season']), week=int(g['week']), date=game_date,
            home=home, away=away,
            home_pts=home_pts, away_pts=away_pts,
            total=total, margin=margin,
            p_home_win=p_home,
            sigma_total=sigma_total,
            sigma_margin=sigma_margin,
        ))

    return results

# ---------------------------
# Odds ingestion + EV
# ---------------------------
ODDS_REQ = ['book','market','label','price','point','home_team','away_team','season','week']

def load_odds_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ODDS_REQ if c not in df.columns]
    if missing:
        raise ValueError(f"odds CSV missing columns: {missing}")
    df['home_team'] = df['home_team'].map(norm_team)
    df['away_team'] = df['away_team'].map(norm_team)
    df['season'] = df['season'].astype(int)
    df['week'] = df['week'].astype(int)
    df['market'] = df['market'].str.lower()
    df['label'] = df['label'].astype(str)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['point'] = pd.to_numeric(df['point'], errors='coerce')
    df = df.dropna(subset=['price'])
    # collapse duplicate snapshots and add open/last deltas if timestamp present
    df = collapse_odds_snapshots(df)
    return df

def attach_odds(preds: List[GamePred], odds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gp in preds:
        mask = (odds['season']==gp.season) & (odds['week']==gp.week) & (odds['home_team']==gp.home) & (odds['away_team']==gp.away)
        o = odds[mask]

        # Moneyline
        ml = o[o['market']=='moneyline']
        if not ml.empty:
            for side in [gp.home, gp.away]:
                side_mask = ml['label'].str.upper().map(norm_team) == side
                for _, r in ml[side_mask].iterrows():
                    p = gp.p_home_win if side==gp.home else 1-gp.p_home_win
                    ev = ev_from_prob_and_american(p, r['price'])
                    kelly = kelly_fraction(p, r['price'], max_kelly=0.15, min_edge=0.0)
                    rows.append({
                        'season': gp.season,'week': gp.week,'date': gp.date,
                        'home': gp.home,'away': gp.away,
                        'market': 'moneyline','label': side,'book': r['book'],
                        'price': int(r['price']), 'point': np.nan,
                        'model_prob': round(p,4), 'edge_ev': round(ev,4),
                        'kelly': round(kelly,4),
                        'open_price': r.get('open_price', np.nan),
                        'price_delta': r.get('price_delta', np.nan),
                    })

        # Spread
        sp = o[o['market']=='spread']
        if not sp.empty:
            for side in [gp.home, gp.away]:
                side_mask = sp['label'].str.upper().map(norm_team) == side
                for _, r in sp[side_mask].iterrows():
                    line = float(r['point'])
                    # Home cover: margin > -line; Away cover: margin < line
                    if side == gp.home:
                        p_cover = P_home_cover(gp.margin, line, gp.sigma_margin)
                    else:
                        p_cover = P_away_cover(gp.margin, line, gp.sigma_margin)
                    ev = ev_from_prob_and_american(p_cover, r['price'])
                    kelly = kelly_fraction(p_cover, r['price'], max_kelly=0.15, min_edge=0.0)
                    rows.append({
                        'season': gp.season,'week': gp.week,'date': gp.date,
                        'home': gp.home,'away': gp.away,
                        'market': 'spread','label': side,'book': r['book'],
                        'price': int(r['price']), 'point': line,
                        'model_prob': round(p_cover,4), 'edge_ev': round(ev,4),
                        'kelly': round(kelly,4),
                        'open_point': r.get('open_point', np.nan),
                        'point_delta': r.get('point_delta', np.nan),
                        'open_price': r.get('open_price', np.nan),
                        'price_delta': r.get('price_delta', np.nan),
                    })

        # Total
        tot = o[o['market']=='total']
        if not tot.empty:
            for ou in ['Over','Under']:
                ou_mask = tot['label'].str.capitalize()==ou
                for _, r in tot[ou_mask].iterrows():
                    line = standardize_total_line(float(r['point']))
                    if ou=='Over':
                        z = (line - gp.total) / gp.sigma_total
                    else:
                        z = (gp.total - line) / gp.sigma_total
                    p = float(1 - 0.5*(1+math.erf(z/math.sqrt(2))))
                    ev = ev_from_prob_and_american(p, r['price'])
                    kelly = kelly_fraction(p, r['price'], max_kelly=0.15, min_edge=0.0)
                    rows.append({
                        'season': gp.season,'week': gp.week,'date': gp.date,
                        'home': gp.home,'away': gp.away,
                        'market': 'total','label': ou,'book': r['book'],
                        'price': int(r['price']), 'point': line,
                        'model_prob': round(p,4), 'edge_ev': round(ev,4),
                        'kelly': round(kelly,4),
                        'open_point': r.get('open_point', np.nan),
                        'point_delta': r.get('point_delta', np.nan),
                        'open_price': r.get('open_price', np.nan),
                        'price_delta': r.get('price_delta', np.nan),
                    })

    return pd.DataFrame(rows)

# ---------------------------
# Merge DK splits into EV table
# ---------------------------
def merge_dk_splits_into_ev(ev_df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join DK splits onto EV rows by (home, away, market, label).
    For totals, we join on (market,label) ignoring point mismatches; lines move often.
    """
    if splits is None or splits.empty:
        return ev_df
    # Ensure consistent dtypes
    splits = splits.copy()
    splits['home'] = splits['home'].astype(str)
    splits['away'] = splits['away'].astype(str)
    splits['market'] = splits['market'].astype(str)
    splits['label'] = splits['label'].astype(str)

    out = ev_df.merge(
        splits[['home','away','market','label','handle_pct','bets_pct']],
        on=['home','away','market','label'],
        how='left'
    )
    return out

# ---------------------------
# Helper: Convert DK splits to odds-like DataFrame
# ---------------------------
def odds_from_dk_splits(preds: List[GamePred], splits: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DK Network splits into an odds dataframe with the required schema:
    book,market,label,price,point,home_team,away_team,season,week
    - book hardcoded to 'DKNET'
    - season/week derived from the predictions list for matching (home,away)
    """
    if splits is None or splits.empty:
        return pd.DataFrame(columns=ODDS_REQ)
    # Build a quick lookup from (home,away) to (season,week)
    key_rows = []
    for gp in preds:
        key_rows.append({'home': gp.home, 'away': gp.away, 'season': gp.season, 'week': gp.week})
    key_df = pd.DataFrame(key_rows)
    if key_df.empty:
        return pd.DataFrame(columns=ODDS_REQ)

    out = []
    for _, r in splits.iterrows():
        try:
            # Stronger normalization for alignment
            home = norm_team(_normalize_team_text(r['home']))
            away = norm_team(_normalize_team_text(r['away']))
            mkt  = str(r['market']).lower()
            label = str(r['label'])
            price = int(r['price']) if not pd.isna(r['price']) else np.nan
            point = float(r['point']) if not pd.isna(r['point']) else np.nan
        except Exception:
            continue
        # match to season/week via (home,away)
        k = key_df[(key_df['home']==home) & (key_df['away']==away)]
        if k.empty:
            # try swapped orientation just in case; DK page should be away @ home though
            k = key_df[(key_df['home']==away) & (key_df['away']==home)]
        # If still not found, try matching when either side matches and report unmatched later
        if k.empty:
            k = key_df[(key_df['home']==home) | (key_df['away']==away) | (key_df['home']==away) | (key_df['away']==home)]
            if k.empty:
                continue
        season = int(k['season'].iloc[0])
        week   = int(k['week'].iloc[0])
        # Normalize totals label capitalization
        if mkt == 'total':
            label = label.capitalize()
        # Compose odds row
        out.append({
            'book': 'DKNET',
            'market': mkt,
            'label': label,
            'price': price,
            'point': point if mkt != 'moneyline' else np.nan,
            'home_team': home,
            'away_team': away,
            'season': season,
            'week': week,
        })
    if not out:
        return pd.DataFrame(columns=ODDS_REQ)
    df = pd.DataFrame(out)
    # Standardize types
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['point'] = pd.to_numeric(df['point'], errors='coerce')
    df = df.dropna(subset=['price'])
    return df[ODDS_REQ]

# Diagnostic helper for DK splits alignment
def _diagnose_splits_alignment(preds: List[GamePred], splits: pd.DataFrame) -> pd.DataFrame:
    pred_pairs = {(p.home, p.away) for p in preds}
    rows = []
    for _, r in splits.iterrows():
        h = norm_team(_normalize_team_text(r['home']))
        a = norm_team(_normalize_team_text(r['away']))
        ok = (h, a) in pred_pairs or (a, h) in pred_pairs
        rows.append({'home_scraped': r['home'], 'away_scraped': r['away'], 'home_norm': h, 'away_norm': a, 'aligned': ok})
    return pd.DataFrame(rows)

# ---------------------------
# CLI + TRAIN / SERVE modes
# ---------------------------
def cmd_train(args: argparse.Namespace) -> None:
    games = fetch_games(min_season=args.min_season)
    tw = make_team_week(games)
    art = fit_models(tw, min_season=args.min_season or int(games['season'].min()))
    # ML reliability (validation = last season)
    try:
        last_year = int(art.meta['last_season'])
        games_va = games[games['season'] == last_year].copy()
        if not games_va.empty:
            # Build preds on validation slate
            wk_va = games_va[['date','season','week','home_team','away_team']].rename(
                columns={'home_team':'home_team','away_team':'away_team'}
            )
            # predict_week expects columns: date, season, week, home_team, away_team
            preds_va = predict_week(art, make_team_week(games[games['season'] <= last_year]), wk_va)
            # Align actuals in same order as preds via key
            if preds_va:
                key = games_va['date'].astype(str) + '|' + games_va['home_team'] + '|' + games_va['away_team']
                games_va['_key'] = key.values
                probs, actual = [], []
                for p in preds_va:
                    k = str(p.date) + '|' + p.home + '|' + p.away
                    row = games_va[games_va['_key']==k]
                    if not row.empty:
                        probs.append(p.p_home_win)
                        actual.append(int((float(row['home_points'].iloc[0]) - float(row['away_points'].iloc[0])) > 0))
                if probs:
                    rel = reliability_by_bins(np.array(probs), np.array(actual), n_bins=10)
                    art.meta['reliability_ml'] = rel.to_dict(orient='records')
    except Exception:
        pass
    save_artifacts(art, args.out)
    print(json.dumps({'saved': args.out, **art.meta}, indent=2, default=str))

def serve_ui(args: argparse.Namespace) -> None:
    if st is None:
        print("Streamlit is required. Install and run: streamlit run nfl_edge_app.py -- serve --model models/nfl_model.pkl", file=sys.stderr)
        sys.exit(2)

    st.set_page_config(page_title="NFL Edge App", layout="wide")
    st.title("NFL Edge App")

    @st.cache_data(ttl=180)
    def _load_artifacts_cached(p: str) -> ModelArtifacts:
        return load_artifacts(p)

    @st.cache_data(ttl=600)
    def _hist() -> Tuple[pd.DataFrame, pd.DataFrame]:
        g = fetch_games(min_season=args.min_season)
        return g, make_team_week(g)

    @st.cache_data(ttl=180)
    def _load_odds(p: str) -> pd.DataFrame:
        return load_odds_csv(p)

    @st.cache_data(ttl=180)
    def _load_inj(path: str) -> Optional[pd.DataFrame]:
        if not path:
            return None
        try:
            return load_injury_adjustments(path)
        except Exception:
            return None

    @st.cache_data(ttl=600)
    def _weather_tbl(years: List[int]) -> pd.DataFrame:
        return get_weather_table_for_years(years)

    try:
        art = _load_artifacts_cached(args.model)
        st.success("Model loaded")
        st.json(art.meta)
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

    games, team_week = _hist()
    st.caption(f"Historical games: {len(games)} | team-weeks: {len(team_week)}")

    st.subheader("Select slate")
    slate_choice = st.radio("Slate source", ["Upcoming (auto)", "Historical week"], horizontal=True)

    if slate_choice == "Upcoming (auto)":
        try:
            wk = load_upcoming_matchups_auto(min_season=args.min_season)
        except Exception as e:
            st.error(f"Failed to load upcoming: {e}")
            st.stop()
    else:
        # Historical week selection (use full schedule, not just completed games)
        years_all = sorted(games['season'].dropna().astype(int).unique().tolist())
        sched_all = _import_sched_with_refs(years_all)
        seasons_available = sorted(sched_all['season'].dropna().astype(int).unique().tolist(), reverse=True)
        sel_season = st.selectbox("Season", seasons_available, index=0)
        # Regular season weeks present in schedule for this season
        week_options = sorted(sched_all.loc[(sched_all['season'] == int(sel_season)) & (sched_all['game_type'].astype(str).str.upper() == 'REG'), 'week']
                              .dropna().astype(int).unique().tolist())
        sel_week = st.selectbox("Week", week_options, index=0)
        wk_src = sched_all[(sched_all['season'] == int(sel_season)) & (sched_all['week'] == int(sel_week))].copy()
        if wk_src.empty:
            st.warning("No scheduled games found for that season/week.")
            st.stop()
        wk = wk_src[['gameday','season','week','home_team','away_team']].rename(columns={
            'gameday':'date',
            'season':'season',
            'week':'week',
            'home_team':'home_team',
            'away_team':'away_team'
        })
        wk['date'] = pd.to_datetime(wk['date']).dt.tz_localize(None)

    # Weather table for relevant years (history + current)
    years_needed = sorted(set(games['season'].astype(int).tolist() + wk['season'].astype(int).tolist()))
    wtbl = _weather_tbl(years_needed)

    # Optional injuries parquet path (CLI default plumbed into UI)
    inj_path_default = getattr(args, 'injury_adjustments', '') or ''
    inj_path = st.text_input("Optional injuries adjustments parquet (from injuries.py)", value=inj_path_default)

    preds = predict_week(art, team_week, wk, injury_adj=_load_inj(inj_path), weather_tbl=wtbl)
    if not preds:
        st.warning("No predictions produced. Ensure teams have sufficient prior history.")
        st.stop()

    pred_rows = [{
        'season': p.season, 'week': p.week, 'date': p.date.date(),
        'home': p.home, 'away': p.away,
        'Home_Pts': round(p.home_pts,1), 'Away_Pts': round(p.away_pts,1),
        'Total': round(p.total,1), 'Margin': round(p.margin,1),
        'P(Home Win)': round(p.p_home_win,4),
        'Sigma_Total': round(p.sigma_total,2), 'Sigma_Margin': round(p.sigma_margin,2)
    } for p in preds]
    pred_df = pd.DataFrame(pred_rows)
    st.dataframe(pred_df, use_container_width=True)

    # Show ML calibration bins from training artifacts if present
    if isinstance(art.meta.get('reliability_ml'), list):
        st.subheader("ML Calibration (validation bins)")
        st.dataframe(pd.DataFrame(art.meta['reliability_ml']), use_container_width=True)

    st.subheader("Odds source")
    use_dk = st.checkbox("Use DraftKings Network betting splits as primary odds source", value=True)
    odds_path = st.text_input("Backup odds CSV path (optional)", value="")
    ev_df = None

    # Try DK splits first if selected
    if use_dk:
        try:
            splits_df = fetch_dk_splits_nfl(window="n7days", market=0)
            st.caption(f"Fetched DK splits rows: {len(splits_df)}")
            st.dataframe(splits_df.head(20), use_container_width=True)
            # Convert splits to odds-like schema
            odds_from_splits = odds_from_dk_splits(preds, splits_df)
            if odds_from_splits.empty:
                st.warning("DK splits fetched but could not align to upcoming matchups.")
                try:
                    diag = _diagnose_splits_alignment(preds, splits_df)
                    st.dataframe(diag.head(30), use_container_width=True)
                except Exception:
                    pass
            else:
                # Attach EVs using DK-derived odds
                ev_df = attach_odds(preds, odds_from_splits)
                # Merge splits onto EV for handle/bets percentages
                ev_df = merge_dk_splits_into_ev(ev_df, splits_df)
        except Exception as e:
            st.warning(f"DK splits failed: {e}")

    # Fallback to CSV if no EVs yet and a path is provided
    if ev_df is None and odds_path:
        try:
            odds = _load_odds(odds_path)
            ev_df = attach_odds(preds, odds)
        except Exception as e:
            st.error(f"Failed to load/attach odds CSV: {e}")

    if ev_df is not None and not ev_df.empty:
        # rank within each market by EV and display
        ev_df['rank'] = ev_df.groupby(['market'])['edge_ev'].rank(ascending=False, method='first')
        st.dataframe(ev_df.sort_values(['market','rank']), use_container_width=True)

        # Confluence flags helper, scoped to see pred_df
        def _flags(row):
            flags = []
            m = pred_df[(pred_df['home'] == row['home']) & (pred_df['away'] == row['away'])]
            if not m.empty:
                model_total = float(m['Total'].iloc[0])
                model_margin = float(m['Margin'].iloc[0])
                if row['market'] == 'spread':
                    if abs(model_margin + float(row['point'])) >= 2.5:
                        flags.append('MarginVsLine≥2.5')
                elif row['market'] == 'total':
                    if abs(model_total - float(row['point'])) >= 2.0:
                        flags.append('TotalDelta≥2')
                elif row['market'] == 'moneyline':
                    imp = american_to_prob(row['price'])
                    if abs(float(row['model_prob']) - imp) >= 0.05:
                        flags.append('PriceMisprice≥5%')
            # movement flags if available (from collapsed snapshots via CSV)
            try:
                if pd.notna(row.get('point_delta', np.nan)) and abs(float(row['point_delta'])) >= 0.5:
                    flags.append('LineMoved≥0.5')
            except Exception:
                pass
            try:
                if pd.notna(row.get('price_delta', np.nan)) and abs(float(row['price_delta'])) >= 20:
                    flags.append('PriceMoved≥20')
            except Exception:
                pass
            # public splits confluence if available
            try:
                hp = row.get('handle_pct', np.nan)
                if not pd.isna(hp):
                    if hp >= 0.60:
                        flags.append('Public≥60')
                    elif hp <= 0.40:
                        flags.append('Contrarian≤40')
            except Exception:
                pass
            return ','.join(flags)

        ev_df['flags'] = ev_df.apply(_flags, axis=1)

        st.subheader("Top plays (export-ready)")
        top_n = st.number_input("Top N per market", min_value=1, max_value=20, value=5, step=1)
        top_df = ev_df.sort_values(['market','edge_ev'], ascending=[True, False]) \
                      .groupby('market').head(int(top_n)).reset_index(drop=True)
        st.dataframe(top_df, use_container_width=True)
        st.download_button(
            "Download Top Plays CSV",
            data=top_df.to_csv(index=False).encode('utf-8'),
            file_name="top_plays.csv",
            mime="text/csv"
        )
    elif use_dk:
        st.info("Enable CSV backup or try again if the splits page lags behind the official schedule.")

    st.caption("v1.3 — deterministic, symmetric, variance-aware; Kelly + flags + open/last deltas.")

def render(state: dict | None = None, *, default_model: str | None = None, default_min_season: int | None = None) -> None:
    """Router-friendly Streamlit UI. Mirrors serve_ui() without set_page_config or argparse.
    Use from router:  import nfl_edge_app as nea; nea.render(page_state("NFL Edge"))
    """
    if st is None:
        return

    st.title("NFL Edge App")
    st.caption("Router mode: state persists until tab/app closes.")

    # Persistent bucket
    ss = state if state is not None else st.session_state
    ss.setdefault("model_path", default_model or "models/nfl_model.pkl")
    ss.setdefault("min_season", int(default_min_season) if default_min_season is not None else 2012)
    ss.setdefault("injury_parquet", "")
    ss.setdefault("odds_csv_path", "")

    # Controls
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        ss["model_path"] = st.text_input("Model artifacts (.pkl)", value=str(ss["model_path"]))
    with c2:
        ss["min_season"] = st.number_input("Min season", min_value=2000, max_value=2100, step=1, value=int(ss["min_season"]))
    with c3:
        ss["injury_parquet"] = st.text_input("Injuries parquet (optional)", value=str(ss["injury_parquet"]))

    # Cached helpers (adapted from serve_ui)
    @st.cache_data(ttl=180)
    def _load_artifacts_cached(p: str) -> ModelArtifacts:
        return load_artifacts(p)

    @st.cache_data(ttl=600)
    def _hist(min_season_val: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        g = fetch_games(min_season=min_season_val)
        return g, make_team_week(g)

    @st.cache_data(ttl=600)
    def _weather_tbl(years: list[int]) -> pd.DataFrame:
        return get_weather_table_for_years(years)

    @st.cache_data(ttl=180)
    def _load_inj(path: str) -> Optional[pd.DataFrame]:
        if not path:
            return None
        try:
            return load_injury_adjustments(path)
        except Exception:
            return None

    # Load model
    try:
        art = _load_artifacts_cached(ss["model_path"])
        st.success("Model loaded")
        with st.expander("Artifacts meta", expanded=False):
            st.json(art.meta)
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return

    # History and upcoming
    games, team_week = _hist(int(ss["min_season"]))
    st.caption(f"Historical games: {len(games)} • team-weeks: {len(team_week)}")

    st.subheader("Select slate")
    ss.setdefault("slate_choice", "Upcoming (auto)")
    slate_choice = st.radio("Slate source", ["Upcoming (auto)", "Historical week"], index=0 if ss["slate_choice"] == "Upcoming (auto)" else 1, horizontal=True)
    ss["slate_choice"] = slate_choice

    if slate_choice == "Upcoming (auto)":
        try:
            wk = load_upcoming_matchups_auto(min_season=int(ss["min_season"]))
        except Exception as e:
            st.error(f"Failed to load upcoming: {e}")
            return
    else:
        years_all = sorted(games['season'].dropna().astype(int).unique().tolist())
        sched_all = _import_sched_with_refs(years_all)
        seasons_available = sorted(sched_all['season'].dropna().astype(int).unique().tolist(), reverse=True)
        if "sel_season" not in ss:
            ss["sel_season"] = seasons_available[0]
        ss["sel_season"] = st.selectbox("Season", seasons_available, index=0)
        week_options = sorted(sched_all.loc[(sched_all['season'] == int(ss["sel_season"])) & (sched_all['game_type'].astype(str).str.upper() == 'REG'), 'week']
                              .dropna().astype(int).unique().tolist())
        if "sel_week" not in ss:
            ss["sel_week"] = week_options[0] if week_options else 1
        ss["sel_week"] = st.selectbox("Week", week_options, index=0 if week_options else 0)
        wk_src = sched_all[(sched_all['season'] == int(ss["sel_season"])) & (sched_all['week'] == int(ss["sel_week"]))].copy()
        if wk_src.empty:
            st.warning("No scheduled games found for that season/week.")
            return
        wk = wk_src[['gameday','season','week','home_team','away_team']].rename(columns={
            'gameday':'date',
            'season':'season',
            'week':'week',
            'home_team':'home_team',
            'away_team':'away_team'
        })
        wk['date'] = pd.to_datetime(wk['date']).dt.tz_localize(None)

    years_needed = sorted(set(games['season'].astype(int).tolist() + wk['season'].astype(int).tolist()))
    wtbl = _weather_tbl(years_needed)

    preds = predict_week(art, team_week, wk, injury_adj=_load_inj(ss.get("injury_parquet", "")), weather_tbl=wtbl)
    if not preds:
        st.warning("No predictions produced. Ensure teams have sufficient prior history.")
        return

    pred_rows = [{
        'season': p.season, 'week': p.week, 'date': p.date.date(),
        'home': p.home, 'away': p.away,
        'Home_Pts': round(p.home_pts,1), 'Away_Pts': round(p.away_pts,1),
        'Total': round(p.total,1), 'Margin': round(p.margin,1),
        'P(Home Win)': round(p.p_home_win,4),
        'σ_total': round(p.sigma_total,2), 'σ_margin': round(p.sigma_margin,2),
    } for p in preds]
    df_preds = pd.DataFrame(pred_rows)
    st.subheader("Model slate")
    st.dataframe(df_preds, use_container_width=True, hide_index=True)

    # Optional odds CSV for EV
    st.divider()
    st.subheader("Attach odds CSV (optional)")
    odds_up = st.file_uploader("Upload odds CSV (schema in docstring)", type=["csv"], key="odds_csv_router")
    if odds_up is not None:
        try:
            odds_df = pd.read_csv(odds_up)
            odds_df = collapse_odds_snapshots(odds_df)
            ev_df = attach_odds(preds, odds_df)
            st.success(f"Attached odds: {len(ev_df)} rows")
            st.dataframe(ev_df.head(300), use_container_width=True)
            st.download_button("Download EV CSV", ev_df.to_csv(index=False), file_name="nfl_ev.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to process odds CSV: {e}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NFL Edge App — train or serve")
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='Train models and write artifacts')
    t.add_argument('--out', required=True, help='Output artifact .pkl path')
    t.add_argument('--min_season', type=int, default=2012, help='Filter to seasons >= this')
    t.set_defaults(func=cmd_train)

    s = sub.add_parser('serve', help='Run Streamlit UI (must be invoked via streamlit)')
    s.add_argument('--model', required=True, help='Artifacts .pkl to load')
    s.add_argument('--min_season', type=int, default=2012, help='Seasons to ingest when auto')
    s.add_argument('--injury_adjustments', type=str, default="", help='Parquet path produced by injuries.py predict (optional)')
    s.set_defaults(func=serve_ui)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])
    args.func(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL Edge App")
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--out", required=True)
    p_train.add_argument("--min_season", type=int, default=2012)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--model", required=True)
    p_serve.add_argument("--min_season", type=int, default=2012)
    p_serve.add_argument("--injury_adjustments", default="")

    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "serve":
        serve_ui(args)
    else:
        parser.print_help()