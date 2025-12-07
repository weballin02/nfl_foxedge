import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime, timedelta
import pytz
import altair as alt
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import json
import logging
import math
from streamlit_autorefresh import st_autorefresh
import numpy as np

# ============================================================================
# NFL-SPECIFIC KEY NUMBER DISTRIBUTIONS (Enhanced)
# ============================================================================
NFL_KEY_MASS = {
    -21: 0.003, -20: 0.003, -17: 0.004, -16: 0.005, -14: 0.006, -13: 0.007,
    -12: 0.008, -11: 0.012, -10: 0.018, -9: 0.022, -8: 0.028, -7: 0.095,
    -6: 0.055, -5: 0.038, -4: 0.062, -3: 0.160,
    -2: 0.048, -1: 0.038, 0: 0.012,
    1: 0.038, 2: 0.048, 3: 0.160, 4: 0.062, 5: 0.038, 6: 0.055,
    7: 0.095, 8: 0.028, 9: 0.022, 10: 0.018, 11: 0.012, 12: 0.008,
    13: 0.007, 14: 0.006, 16: 0.005, 17: 0.004, 20: 0.003, 21: 0.003
}
NFL_TOTAL_KEYS = {
    37: 0.025, 38: 0.022, 39: 0.028, 40: 0.032, 41: 0.048, 42: 0.038,
    43: 0.055, 44: 0.062, 45: 0.058, 46: 0.048, 47: 0.055, 48: 0.045,
    49: 0.042, 50: 0.038, 51: 0.048, 52: 0.035, 53: 0.032, 54: 0.035
}
HALF_POINT_VALUE = {
    2.5: 0.8, 3.0: 2.5, 3.5: 1.2, 6.5: 0.9, 7.0: 2.0, 7.5: 0.8,
    9.5: 0.5, 10.0: 0.7, 10.5: 0.4, 13.5: 0.4, 14.0: 0.5
}
WONG_TEASER_RANGES = {
    "PRIME_FAV": (7.5, 8.5),
    "GOOD_FAV": (8.5, 9.5),
    "PRIME_DOG": (1.5, 2.5),
    "GOOD_DOG": (2.5, 3.5),
    "AVOID_FAV": (4.5, 6.5),
    "AVOID_DOG": (-1.5, 1.5),
}
NFL_KEY_CROSS_VALUE = {
    3: 2.50, 7: 2.00, 6: 1.00, 10: 0.80, 4: 0.75,
    14: 0.60, 2: 0.50, 1: 0.40,
}
POPULAR_TEAMS = [
    'Cowboys', 'Chiefs', '49ers', 'Eagles', 'Packers', 'Patriots',
    'Steelers', 'Raiders', 'Broncos', 'Giants'
]

# ============================================================================
# CONFIGURATION
# ============================================================================
HISTORY_FILE = "mood_history.csv"
GATE_THRESHOLD = 60
EXPOSURE_THRESHOLD = 3000
ODDLOT_THRESHOLD = 0.12
CORR_THRESHOLD = 0.025
RLM_BET_THRESHOLD = 6.0
STEAM_ODDS_THRESHOLD = 3
DEFAULT_W_EV = 0.40
DEFAULT_W_IRR = 0.25
DEFAULT_W_RLM = 0.20
DEFAULT_W_STEAM = 0.15
DEFAULT_KELLY_CAP = 0.04
LEAGUE_GROUPS = {"NFL": 88807, "NCAAF": 88806}

st.set_page_config(
    page_title="FoxEdge NFL Pro Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BETLOGS_ROOT = os.getenv("BETLOGS_ROOT", os.getcwd())
BETLOGS_DB = os.path.join(BETLOGS_ROOT, "bet_logs_nfl.db")

def _ensure_parent(path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

# ============================================================================
# PROBABILITY MODELS
# ============================================================================
def nfl_cover_prob_enhanced(spread_line: float, is_home: bool = False,
                           is_divisional: bool = False) -> float | None:
    if spread_line is None:
        return None
    try:
        L = float(spread_line)
    except Exception:
        return None
    
    sign = 1 if L >= 0 else -1
    L = abs(L)
    lo, hi = math.floor(L), math.ceil(L)
    w_hi = L - lo
    w_lo = 1 - w_hi

    def tail_prob(x):
        return sum(v for k, v in NFL_KEY_MASS.items() if k >= x)

    if sign > 0:
        p_lo = tail_prob(lo)
        p_hi = tail_prob(hi)
    else:
        p_lo = 1 - tail_prob(-lo + 1e-9)
        p_hi = 1 - tail_prob(-hi + 1e-9)

    p = w_lo * p_lo + w_hi * p_hi

    if is_home and L < 0:
        p = min(0.95, p * 1.02)
    if is_divisional:
        p = max(0.05, min(0.95, p + 0.01))

    return max(0.0, min(1.0, p))

def nfl_total_prob_enhanced(total_line: float, is_outdoor: bool = True,
                           temp: float = None, wind: int = None,
                           is_primetime: bool = False) -> float | None:
    if total_line is None:
        return None
    try:
        tl = float(total_line)
    except Exception:
        return None

    sd = 9.8
    mu = tl

    if is_outdoor and temp is not None:
        if temp < 32:
            mu -= 1.5
        elif temp < 40:
            mu -= 0.8
        elif temp > 85:
            mu -= 0.5

    if wind is not None and wind > 15:
        mu -= (wind - 15) * 0.15

    if is_primetime:
        mu -= 0.8

    from math import erf, sqrt
    z = (tl - mu) / sd
    p_over = 0.5 * (1 - erf(z / sqrt(2)))

    bump = sum(v for k, v in NFL_TOTAL_KEYS.items() if abs(tl - k) < 0.6)
    p_over = p_over + 0.5 * bump

    return max(0.0, min(1.0, p_over))

def implied_prob_from_odds(odds):
    if odds is None:
        return None
    try:
        odds = int(odds)
    except Exception:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 0.5

# ============================================================================
# KEY NUMBER UTILITIES
# ============================================================================
def _key_weighted_spread_move(entry_spread, close_spread):
    if entry_spread is None or close_spread is None:
        return None
    try:
        es = float(entry_spread)
        cs = float(close_spread)
    except Exception:
        return None

    lo, hi = sorted([es, cs])
    base = cs - es
    bonus = 0.0

    for k, val in NFL_KEY_CROSS_VALUE.items():
        if lo < k <= hi:
            bonus += val * (1 if base > 0 else -1)

    for half_pt, premium in HALF_POINT_VALUE.items():
        if lo < half_pt <= hi:
            bonus += premium * 0.3 * (1 if base > 0 else -1)

    return round(base + bonus, 3)

def _crossed_key(es, cs, key):
    if es is None or cs is None:
        return False
    lo, hi = sorted([es, cs])
    return lo < key <= hi

def wong_teaser_tag(spread_line: float) -> str | None:
    if spread_line is None:
        return None
    try:
        s = abs(float(spread_line))
    except Exception:
        return None

    for tag, (low, high) in WONG_TEASER_RANGES.items():
        if low <= s <= high:
            return tag
    return None

# ============================================================================
# PARSING UTILITIES
# ============================================================================
def clean_odds(odds_str):
    try:
        return int(str(odds_str).replace("−", "-").strip())
    except Exception:
        try:
            return int(float(odds_str))
        except Exception:
            return None

def _parse_total_from_side(side_text: str):
    try:
        s = str(side_text).strip()
        if s.lower().startswith("over ") or s.lower().startswith("under "):
            parts = s.split()
            if len(parts) >= 2:
                return parts[0].title(), float(parts[1])
    except Exception:
        pass
    return None, None

def _parse_spread_from_side(side_text: str):
    try:
        s = str(side_text).strip()
        parts = s.split()
        if len(parts) >= 2:
            last = parts[-1].replace("−", "-")
            if last.startswith(("+", "-")):
                return float(last)
    except Exception:
        pass
    return None

def _dec(american):
    try:
        american = int(american)
    except Exception:
        return None
    return (1 + american/100) if american > 0 else (1 + 100/abs(american))

def _parse_game_time(time_str: str) -> datetime | None:
    """Parse game time string to datetime."""
    try:
        pac = pytz.timezone("America/Los_Angeles")
        time_str = str(time_str).strip()
        for fmt in ["%a %I:%M %p", "%m/%d %I:%M %p", "%I:%M %p", "%a %m/%d %I:%M %p"]:
            try:
                parsed = datetime.strptime(time_str, fmt)
                if parsed.year == 1900:
                    parsed = parsed.replace(year=datetime.now().year)
                return pac.localize(parsed)
            except:
                continue
        return None
    except:
        return None

# ============================================================================
# ADVANCED SIGNAL DETECTION
# ============================================================================
def detect_public_bias(row: pd.Series) -> list[str]:
    biases = []
    matchup = str(row.get('matchup', ''))
    bets = row.get('%bets', 0)
    handle = row.get('%handle', 0)
    game_time = str(row.get('game_time', ''))

    is_popular = any(team in matchup for team in POPULAR_TEAMS)
    if is_popular and bets > 70:
        biases.append("POPULAR_OVERBET")

    is_primetime = any(x in game_time for x in ['SNF', 'MNF', 'TNF'])
    if is_primetime and bets > 65:
        biases.append("PRIMETIME_PUBLIC")

    if bets < 35 and handle < 40:
        biases.append("SMALL_MARKET_VALUE")

    if bets > 80:
        biases.append("EXTREME_PUBLIC")

    return biases

def detect_sharp_patterns(row: pd.Series, opens: pd.DataFrame) -> list[str]:
    patterns = []
    matchup = row.get('matchup')
    market = row.get('market')
    side = row.get('side')
    current_odds = row.get('odds')
    bets_pct = row.get('%bets', 0)
    handle_pct = row.get('%handle', 0)

    open_row = opens[
        (opens['matchup'] == matchup) &
        (opens['market'] == market) &
        (opens['side'] == side)
    ]

    if not open_row.empty:
        open_odds = open_row.iloc[0].get('odds')
        open_spread = open_row.iloc[0].get('spread')
        current_spread = row.get('spread')

        if open_odds is not None and current_odds is not None:
            odds_move = abs(current_odds - open_odds)
            if bets_pct > 65 and odds_move < 3:
                patterns.append("LINE_FREEZE")
            if odds_move >= 5 and bets_pct < 50:
                patterns.append("SHARP_STEAM")

        if open_spread is not None and current_spread is not None:
            spread_move = abs(current_spread - open_spread)
            if spread_move >= 1.5 and bets_pct < 45:
                patterns.append("SHARP_MOVE")

    if bets_pct < 40 and handle_pct > 60:
        patterns.append("SHARP_MONEY_HEAVY")

    if current_odds is not None and -115 <= current_odds <= 115:
        patterns.append("LOW_HOLD")

    return patterns

def detect_market_inefficiencies(spread_row: pd.Series, ml_row: pd.Series | None,
                                total_row: pd.Series | None) -> list[dict]:
    inefficiencies = []

    if ml_row is not None:
        spread_implied = nfl_cover_prob_enhanced(spread_row.get('spread'))
        ml_implied = implied_prob_from_odds(ml_row.get('odds'))
        if spread_implied and ml_implied and abs(spread_implied - ml_implied) > 0.05:
            inefficiencies.append({
                'type': 'SPREAD_ML_DIVERGENCE',
                'spread_prob': spread_implied,
                'ml_prob': ml_implied,
                'edge': abs(spread_implied - ml_implied)
            })

    if total_row is not None:
        ou, total_line = _parse_total_from_side(total_row.get('side'))
        spread = spread_row.get('spread')
        if total_line and spread:
            fav_total = (total_line + abs(spread)) / 2
            dog_total = (total_line - abs(spread)) / 2
            if fav_total > 35 or fav_total < 10 or dog_total < 10:
                inefficiencies.append({
                    'type': 'UNREALISTIC_TEAM_TOTAL',
                    'fav_total': fav_total,
                    'dog_total': dog_total
                })

    return inefficiencies

def analyze_line_movement_quality(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df
    df = history_df.sort_values('timestamp').copy()
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600
    df['line_diff'] = df['line'].diff()
    df['line_velocity'] = df['line_diff'] / df['time_diff']
    df['handle_diff'] = df['handle_pct'].diff()
    df['movement_efficiency'] = df['line_diff'] / df['handle_diff']
    for key in [3.0, 7.0, 10.0]:
        df[f'stuck_at_{key}'] = (
            (df['line'].shift(1) < key) &
            (df['line'] >= key) &
            (df['line'].shift(-1).fillna(key) == key)
        )
    df['acceleration'] = df['line_velocity'].diff()
    return df

# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

def analyze_bet_timing_performance(clv_history: pd.DataFrame) -> dict:
    if clv_history.empty:
        return {}
    df = clv_history.copy()
    df['entry_time'] = pd.to_datetime(df.get('entry_time', df.get('computed_utc')))
    df['entry_dow'] = df['entry_time'].dt.dayofweek
    df['entry_hour'] = df['entry_time'].dt.hour
    results = {}
    dow_perf = df.groupby('entry_dow')['clv_primary'].agg(['mean', 'count', 'std'])
    if not dow_perf.empty:
        results['best_dow'] = dow_perf['mean'].idxmax()
        results['dow_breakdown'] = dow_perf
    hour_perf = df.groupby('entry_hour')['clv_primary'].agg(['mean', 'count'])
    if not hour_perf.empty:
        results['best_hour'] = hour_perf['mean'].idxmax()
        results['hour_breakdown'] = hour_perf
    return results

def analyze_performance_by_bet_type(bet_history: pd.DataFrame) -> dict:
    if bet_history.empty:
        return {}
    df = bet_history.copy()
    results = {}
    if 'market' in df.columns:
        results['by_market'] = df.groupby('market').agg({
            'clv_primary': ['mean', 'median', 'count'],
            'edge_score': 'mean'
        })
    if 'odds' in df.columns:
        df['side_type'] = df['odds'].apply(lambda x: 'FAVORITE' if x and x < 0 else 'UNDERDOG')
        results['by_side'] = df.groupby('side_type')['clv_primary'].agg(['mean', 'count'])
    if 'edge_score' in df.columns:
        df['edge_decile'] = pd.qcut(df['edge_score'], q=10, labels=False, duplicates='drop')
        results['by_edge_decile'] = df.groupby('edge_decile')['clv_primary'].agg(['mean', 'count'])
    if '%bets' in df.columns:
        df['public_bucket'] = pd.cut(df['%bets'], bins=[0, 40, 60, 100], labels=['<40%', '40-60%', '>60%'])
        results['by_public'] = df.groupby('public_bucket')['clv_primary'].agg(['mean', 'count'])
    return results

def calibrate_contrarian_signals(bet_history: pd.DataFrame) -> dict:
    if bet_history.empty or 'clv_primary' not in bet_history.columns:
        return {}
    df = bet_history.copy()
    thresholds = range(55, 85, 5)
    results = []
    for threshold in thresholds:
        fades = df[df['%bets'] >= threshold]
        if len(fades) >= 10:
            results.append({
                'threshold': threshold,
                'avg_clv': fades['clv_primary'].mean(),
                'count': len(fades),
                'pct_positive_clv': (fades['clv_primary'] > 0).mean() * 100
            })
    optimal_df = pd.DataFrame(results) if results else pd.DataFrame()
    if '%handle' in df.columns:
        df['irrationality'] = abs(df['%bets'] - df['%handle'])
        irr_thresholds = range(5, 30, 5)
        irr_results = []
        for irr_thresh in irr_thresholds:
            high_irr = df[df['irrationality'] >= irr_thresh]
            if len(high_irr) >= 10:
                irr_results.append({
                    'threshold': irr_thresh,
                    'avg_clv': high_irr['clv_primary'].mean(),
                    'count': len(high_irr),
                    'pct_positive_clv': (high_irr['clv_primary'] > 0).mean() * 100
                })
        irr_df = pd.DataFrame(irr_results) if irr_results else pd.DataFrame()
    else:
        irr_df = pd.DataFrame()
    return {
        'public_threshold_analysis': optimal_df,
        'irrationality_analysis': irr_df
    }

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def init_database(db_path: str = BETLOGS_DB):
    _ensure_parent(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS line_snapshots (
                game_id TEXT,
                matchup TEXT,
                market TEXT,
                side TEXT,
                book TEXT,
                snapshot_type TEXT,
                odds INTEGER,
                total REAL,
                spread REAL,
                bets_pct REAL,
                handle_pct REAL,
                game_time TEXT,
                timestamp_utc TEXT,
                PRIMARY KEY (game_id, market, side, book, snapshot_type)
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS clv_logs (
                game_id TEXT,
                matchup TEXT,
                market TEXT,
                side TEXT,
                book TEXT,
                entry_odds INTEGER,
                close_odds INTEGER,
                entry_total REAL,
                close_total REAL,
                entry_spread REAL,
                close_spread REAL,
                entry_bets_pct REAL,
                close_bets_pct REAL,
                entry_handle_pct REAL,
                close_handle_pct REAL,
                bets_delta REAL,
                handle_delta REAL,
                clv_prob_pp REAL,
                clv_line_move REAL,
                clv_spread_move REAL,
                clv_quality_total REAL,
                clv_quality_spread REAL,
                elasticity_total REAL,
                elasticity_spread REAL,
                primary_market TEXT,
                clv_primary REAL,
                elasticity_primary REAL,
                edge_label TEXT,
                edge_badge TEXT,
                crossed_3 INTEGER,
                crossed_7 INTEGER,
                crossed_6 INTEGER,
                crossed_10 INTEGER,
                wong_tag_entry TEXT,
                wong_tag_close TEXT,
                lost_wong_value INTEGER,
                move_quality_spread_keyed REAL,
                half_point_value REAL,
                entry_time TEXT,
                game_time TEXT,
                computed_utc TEXT,
                PRIMARY KEY (game_id, market, side, book)
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TEXT,
                analysis_type TEXT,
                metric_name TEXT,
                metric_value REAL,
                details TEXT,
                computed_utc TEXT
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS game_schedule (
                game_id TEXT PRIMARY KEY,
                matchup TEXT,
                game_time_utc TEXT,
                game_time_local TEXT,
                is_closed INTEGER DEFAULT 0,
                close_logged_utc TEXT
            );
            """
        )


def log_snapshot_row(game_id, matchup, market, side, book, snapshot_type,
                    odds_val=None, total_val=None, spread_val=None,
                    bets_pct=None, handle_pct=None, game_time=None,
                    db_path: str = BETLOGS_DB):
    _ensure_parent(db_path)
    with sqlite3.connect(db_path) as con:
        ts = datetime.utcnow().isoformat()
        con.execute(
            """
            INSERT OR REPLACE INTO line_snapshots
            (game_id, matchup, market, side, book, snapshot_type, odds, total, spread,
             bets_pct, handle_pct, game_time, timestamp_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            (
                str(game_id), str(matchup), str(market), str(side), str(book), str(snapshot_type),
                None if odds_val is None else int(odds_val),
                None if total_val is None else float(total_val),
                None if spread_val is None else float(spread_val),
                None if bets_pct is None else float(bets_pct),
                None if handle_pct is None else float(handle_pct),
                game_time,
                ts
            )
        )


def log_snapshot(df, snapshot_type="OPEN"):
    for _, r in df.iterrows():
        matchup = r.get('matchup')
        market = r.get('market')
        side = r.get('side')
        odds = clean_odds(r.get('odds'))
        game_time = r.get('game_time')
        total_val = None
        spread_val = None
        mkt = str(market).lower()
        if mkt.startswith("total"):
            ou, tot = _parse_total_from_side(side)
            total_val = tot
        elif mkt.startswith("spread"):
            spread_val = _parse_spread_from_side(side)
        date_iso = datetime.now(pytz.timezone("America/Los_Angeles")).date().isoformat()
        gid = f"{date_iso}_{str(matchup).replace(' ','_')}_{market}_{str(side).replace(' ','_')}"[:128]
        log_snapshot_row(
            gid, matchup, market, side, "DK", snapshot_type,
            odds_val=odds, total_val=total_val, spread_val=spread_val,
            bets_pct=r.get('%bets'), handle_pct=r.get('%handle'),
            game_time=game_time
        )


def update_game_schedule(df, db_path: str = BETLOGS_DB):
    with sqlite3.connect(db_path) as con:
        for matchup in df['matchup'].unique():
            game_df = df[df['matchup'] == matchup]
            if not game_df.empty:
                game_time_str = game_df.iloc[0].get('game_time')
                game_time_dt = _parse_game_time(game_time_str)
                if game_time_dt:
                    date_iso = datetime.now(pytz.timezone("America/Los_Angeles")).date().isoformat()
                    gid = f"{date_iso}_{str(matchup).replace(' ','_')}"[:128]
                    con.execute(
                        """
                        INSERT OR IGNORE INTO game_schedule
                        (game_id, matchup, game_time_utc, game_time_local, is_closed)
                        VALUES (?,?,?,?,0)
                        """,
                        (
                            gid,
                            matchup,
                            game_time_dt.astimezone(pytz.UTC).isoformat(),
                            game_time_dt.isoformat(),
                        )
                    )


def auto_log_close_lines(db_path: str = BETLOGS_DB):
    """Mark games as closed when start time passes. Actual closing snapshots
    should be logged via the UI 'Log Current as CLOSE' once markets pull.
    """
    try:
        with sqlite3.connect(db_path) as con:
            now_utc = datetime.now(pytz.UTC)
            games = pd.read_sql_query(
                """
                SELECT * FROM game_schedule
                WHERE is_closed = 0 AND game_time_utc IS NOT NULL
                """,
                con
            )
            if games.empty:
                return 0
            closed = 0
            for _, game in games.iterrows():
                game_time = pd.to_datetime(game['game_time_utc'])
                # add 5-minute buffer to avoid early closure
                if now_utc >= game_time + timedelta(minutes=5):
                    con.execute(
                        """
                        UPDATE game_schedule
                        SET is_closed = 1, close_logged_utc = ?
                        WHERE game_id = ?
                        """,
                        (now_utc.isoformat(), game['game_id'])
                    )
                    closed += 1
            return closed
    except Exception as e:
        return 0

# ============================================================================
# DATA INGEST: FILE UPLOAD OR SCRAPE
# ============================================================================
REQUIRED_COLUMNS = ['matchup', 'market', 'side', 'odds', '%bets', '%handle', 'game_time']

@st.cache_data(show_spinner=False)
def load_uploaded_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # normalize naming
    rename_map = {
        'bets_pct': '%bets', 'handle_pct': '%handle', 'odds_american': 'odds',
        'linetype': 'market', 'wager': 'market', 'selection': 'side', 'event': 'matchup'
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # ensure columns exist
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = None
    # derive parsed fields
    df['odds'] = df['odds'].apply(clean_odds)
    df['spread'] = df.apply(lambda r: _parse_spread_from_side(r['side']) if str(r['market']).lower().startswith('spread') else None, axis=1)
    df['total'] = df.apply(lambda r: _parse_total_from_side(r['side'])[1] if str(r['market']).lower().startswith('total') else None, axis=1)
    return df

@st.cache_data(show_spinner=False)
def demo_dataset() -> pd.DataFrame:
    data = [
        {"matchup": "Cowboys @ Eagles", "market": "Spread", "side": "Eagles -3.0", "odds": -110, "%bets": 74, "%handle": 66, "game_time": "Sun 5:20 PM"},
        {"matchup": "Cowboys @ Eagles", "market": "Moneyline", "side": "Eagles", "odds": -160, "%bets": 68, "%handle": 62, "game_time": "Sun 5:20 PM"},
        {"matchup": "Cowboys @ Eagles", "market": "Total", "side": "Over 47.5", "odds": -108, "%bets": 58, "%handle": 61, "game_time": "Sun 5:20 PM"},
        {"matchup": "Chiefs @ Raiders", "market": "Spread", "side": "Raiders +7.5", "odds": -112, "%bets": 36, "%handle": 44, "game_time": "Sun 1:25 PM"},
        {"matchup": "Chiefs @ Raiders", "market": "Moneyline", "side": "Raiders", "odds": +285, "%bets": 24, "%handle": 28, "game_time": "Sun 1:25 PM"},
        {"matchup": "Chiefs @ Raiders", "market": "Total", "side": "Under 45.0", "odds": -105, "%bets": 48, "%handle": 52, "game_time": "Sun 1:25 PM"},
    ]
    df = pd.DataFrame(data)
    df['spread'] = df.apply(lambda r: _parse_spread_from_side(r['side']) if r['market'].lower().startswith('spread') else None, axis=1)
    df['total'] = df.apply(lambda r: _parse_total_from_side(r['side'])[1] if r['market'].lower().startswith('total') else None, axis=1)
    return df

# ============================================================================
# EDGE ENGINE
# ============================================================================

def compute_edge_columns(curr_df: pd.DataFrame, open_df: pd.DataFrame,
                         w_ev=DEFAULT_W_EV, w_irr=DEFAULT_W_IRR,
                         w_rlm=DEFAULT_W_RLM, w_steam=DEFAULT_W_STEAM) -> pd.DataFrame:
    if curr_df is None or curr_df.empty:
        return pd.DataFrame()
    df = curr_df.copy()

    # Join to open snapshots for deltas
    if open_df is None:
        open_df = pd.DataFrame(columns=df.columns)
    key_cols = ['matchup', 'market', 'side']
    open_df_slim = open_df[key_cols + ['odds', 'spread', 'total']].rename(columns={
        'odds': 'open_odds', 'spread': 'open_spread', 'total': 'open_total'
    })
    df = df.merge(open_df_slim, on=key_cols, how='left')

    # EV vs implied
    df['implied'] = df['odds'].apply(implied_prob_from_odds)
    df['open_implied'] = df['open_odds'].apply(implied_prob_from_odds)

    # Irrationality
    df['irrationality'] = (df['%bets'] - df['%handle']).abs()

    # Reverse line move (RLM) heuristic
    # Tickets heavier but price moved against public
    df['rlm'] = np.where((df['%bets'] > 55) & (df['implied'].notna()) & (df['open_implied'].notna()),
                         (df['open_implied'] - df['implied']) * 100,
                         0.0)

    # Steam: change in American odds
    df['steam'] = np.where(df['open_odds'].notna() & df['odds'].notna(),
                           (df['open_odds'] - df['odds']).abs(), 0.0)

    # Spread key-number quality (if spread market)
    df['spread_move_keyed'] = np.where(df['market'].str.lower().str.startswith('spread') &
                                       df['open_spread'].notna() & df['spread'].notna(),
                                       _key_weighted_spread_move(df['open_spread'], df['spread']),
                                       0.0)

    # Normalize components to 0..1 for scoring
    def _norm(s):
        s = s.fillna(0)
        if s.max() == s.min():
            return s * 0
        return (s - s.min()) / (s.max() - s.min())

    ev_comp = 1 - _norm(df['implied'])  # cheaper price is higher EV given fixed true prob assumption
    irr_comp = _norm(df['irrationality'])
    rlm_comp = _norm(df['rlm'])
    steam_comp = _norm(df['steam'])

    df['edge_score'] = (
        w_ev * ev_comp +
        w_irr * irr_comp +
        w_rlm * rlm_comp +
        w_steam * steam_comp
    ) * 100

    # Tags
    df['bias_tags'] = df.apply(detect_public_bias, axis=1)
    df['sharp_tags'] = df.apply(lambda r: detect_sharp_patterns(r, open_df), axis=1)

    return df

# ============================================================================
# CLV ENGINE FROM DB OPEN/CLOSE
# ============================================================================

def _pull_snapshot(con, snap_type: str) -> pd.DataFrame:
    q = """
        SELECT game_id, matchup, market, side, book, odds, total, spread, bets_pct, handle_pct, game_time, timestamp_utc
        FROM line_snapshots WHERE snapshot_type = ?
    """
    df = pd.read_sql_query(q, con, params=(snap_type,))
    df.rename(columns={'bets_pct': '%bets', 'handle_pct': '%handle'}, inplace=True)
    return df


def compute_clv_from_db(db_path: str = BETLOGS_DB) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        open_df = _pull_snapshot(con, 'OPEN')
        close_df = _pull_snapshot(con, 'CLOSE')
        if open_df.empty or close_df.empty:
            return pd.DataFrame()
        key = ['game_id', 'market', 'side', 'book']
        m = open_df.merge(close_df, on=key, suffixes=("_open", "_close"))
        # primary market selection
        def _primary(row):
            mk = row['market']
            return 'Spread' if str(mk).lower().startswith('spread') else ('Total' if str(mk).lower().startswith('total') else 'Moneyline')
        m['primary_market'] = m['market'].apply(_primary)

        # CLV components
        m['bets_delta'] = m['%bets_close'] - m['%bets_open']
        m['handle_delta'] = m['%handle_close'] - m['%handle_open']
        m['clv_line_move'] = (m['odds_close'] - m['odds_open']).astype(float)
        m['clv_spread_move'] = m.apply(lambda r: _key_weighted_spread_move(r['spread_open'], r['spread_close']) if str(r['market']).lower().startswith('spread') else 0.0, axis=1)
        m['clv_quality_total'] = 0.0
        m['clv_quality_spread'] = np.where(m['clv_spread_move'].notna(), m['clv_spread_move'], 0.0)
        # implied prob change in percentage points
        m['clv_prob_pp'] = m.apply(lambda r: (implied_prob_from_odds(r['odds_close']) - implied_prob_from_odds(r['odds_open'])) * 100 if pd.notna(r['odds_open']) and pd.notna(r['odds_close']) else 0.0, axis=1)
        m['elasticity_total'] = 0.0
        m['elasticity_spread'] = np.where(m['bets_delta'] != 0, m['clv_spread_move'] / m['bets_delta'], np.nan)
        m['clv_primary'] = m.apply(lambda r: r['clv_prob_pp'] if r['primary_market'] == 'Moneyline' else (r['clv_spread_move'] if r['primary_market'] == 'Spread' else r['clv_quality_total']), axis=1)
        m['elasticity_primary'] = m.apply(lambda r: r['elasticity_spread'] if r['primary_market'] == 'Spread' else np.nan, axis=1)
        # keys crossed
        m['crossed_3'] = m.apply(lambda r: int(_crossed_key(r['spread_open'], r['spread_close'], 3)), axis=1)
        m['crossed_7'] = m.apply(lambda r: int(_crossed_key(r['spread_open'], r['spread_close'], 7)), axis=1)
        m['crossed_6'] = m.apply(lambda r: int(_crossed_key(r['spread_open'], r['spread_close'], 6)), axis=1)
        m['crossed_10'] = m.apply(lambda r: int(_crossed_key(r['spread_open'], r['spread_close'], 10)), axis=1)
        m['wong_tag_entry'] = m['spread_open'].apply(wong_teaser_tag)
        m['wong_tag_close'] = m['spread_close'].apply(wong_teaser_tag)
        m['lost_wong_value'] = np.where((m['wong_tag_entry'].notna()) & (m['wong_tag_close'].isna()), 1, 0)
        m['move_quality_spread_keyed'] = m['clv_spread_move']
        m['half_point_value'] = m.apply(lambda r: next((HALF_POINT_VALUE[k] for k in HALF_POINT_VALUE if abs((r['spread_close'] or 0) - k) < 0.01), 0.0), axis=1)
        m['entry_time'] = m['timestamp_utc_open']
        m['game_time'] = m['game_time_open']
        m['computed_utc'] = datetime.utcnow().isoformat()
        return m

# ============================================================================
# STREAMLIT APP
# ============================================================================

def sidebar_controls():
    st.sidebar.header("Weights & Thresholds")
    w_ev = st.sidebar.slider("Weight: EV/Price", 0.0, 1.0, DEFAULT_W_EV, 0.05)
    w_irr = st.sidebar.slider("Weight: Irrationality", 0.0, 1.0, DEFAULT_W_IRR, 0.05)
    w_rlm = st.sidebar.slider("Weight: RLM", 0.0, 1.0, DEFAULT_W_RLM, 0.05)
    w_steam = st.sidebar.slider("Weight: Steam", 0.0, 1.0, DEFAULT_W_STEAM, 0.05)
    kelly_cap = st.sidebar.slider("Kelly Cap (fraction)", 0.0, 0.20, DEFAULT_KELLY_CAP, 0.005)
    st.sidebar.caption("Tune scoring to your appetite. Yes, math is still king.")
    return w_ev, w_irr, w_rlm, w_steam, kelly_cap


def kelly_sizing(edge_prob_delta_pp: float, price_american: int, cap: float = DEFAULT_KELLY_CAP) -> float:
    if price_american is None:
        return 0.0
    b = _dec(price_american) - 1
    if b is None or b <= 0:
        return 0.0
    # Use delta in implied probability as proxy for edge
    p_implied = implied_prob_from_odds(price_american)
    if p_implied is None:
        return 0.0
    p_true = min(max(p_implied + edge_prob_delta_pp/100.0, 0.001), 0.999)
    q = 1 - p_true
    f_star = (b*p_true - q) / b
    if pd.isna(f_star):
        return 0.0
    return float(max(0.0, min(cap, f_star)))


def main():
    st.title("🦊 FoxEdge NFL Pro Analyzer")
    st.caption("Public delusions measured. Sharp signals surfaced. Decisions enforced.")

    init_database(BETLOGS_DB)
    closed_count = auto_log_close_lines(BETLOGS_DB)
    if closed_count:
        st.info(f"Auto-marked {closed_count} games as closed based on start times.")

    w_ev, w_irr, w_rlm, w_steam, kcap = sidebar_controls()

    st.subheader("Ingest Markets")
    colA, colB = st.columns([1,1])
    with colA:
        up = st.file_uploader("Upload current market CSV (columns: matchup, market, side, odds, %bets, %handle, game_time)", type=["csv"], key="up_curr")
        if up is not None:
            curr_df = load_uploaded_csv(up)
        else:
            st.caption("No file? Using demo data. Your bankroll will be fine.")
            curr_df = demo_dataset()
        st.dataframe(curr_df, use_container_width=True)
    with colB:
        up_open = st.file_uploader("Upload OPEN market CSV (optional, to compute deltas)", type=["csv"], key="up_open")
        open_df = load_uploaded_csv(up_open) if up_open is not None else pd.DataFrame(columns=curr_df.columns)
        st.dataframe(open_df if not open_df.empty else pd.DataFrame({'note': ['No OPEN file provided']}), use_container_width=True)

    # Log helpers
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Log Current as OPEN", type="primary"):
            log_snapshot(curr_df, snapshot_type="OPEN")
            update_game_schedule(curr_df)
            st.success("Logged current markets as OPEN and updated schedule.")
    with col2:
        if st.button("Log Current as CLOSE", type="secondary"):
            log_snapshot(curr_df, snapshot_type="CLOSE")
            st.success("Logged current markets as CLOSE.")
    with col3:
        if st.button("Show CLV from DB"):
            clvdf = compute_clv_from_db()
            if clvdf.empty:
                st.warning("Need both OPEN and CLOSE snapshots to compute CLV.")
            else:
                st.dataframe(clvdf.sort_values('clv_primary', ascending=False), use_container_width=True)
                chart = alt.Chart(clvdf).mark_bar().encode(
                    x=alt.X('clv_primary:Q', title='CLV (primary units)'),
                    y=alt.Y('matchup:N', sort='-x', title='Matchup'),
                    color=alt.Color('primary_market:N')
                )
                st.altair_chart(chart, use_container_width=True)

    st.subheader("Edge Scoring")
    edges = compute_edge_columns(curr_df, open_df, w_ev, w_irr, w_rlm, w_steam)
    if edges.empty:
        st.stop()

    # Kelly sizing based on price edge proxy (prob delta between open and now)
    edges['edge_pp'] = (edges['open_implied'].fillna(edges['implied']) - edges['implied']) * 100
    edges['kelly_frac'] = edges.apply(lambda r: kelly_sizing(r['edge_pp'], r['odds'], cap=kcap), axis=1)
    edges['stake_units'] = (edges['kelly_frac'] * 100).round(2)

    # Pretty tags
    edges['tags'] = edges.apply(lambda r: list(set(r['bias_tags'] + r['sharp_tags'])), axis=1)

    # Rank and show
    edges = edges.sort_values(['edge_score', 'irrationality', 'rlm', 'steam'], ascending=False)

    st.dataframe(
        edges[['matchup','market','side','odds','%bets','%handle','implied','edge_score','irrationality','rlm','steam','spread','open_spread','edge_pp','kelly_frac','stake_units','tags']],
        use_container_width=True
    )

    st.subheader("Visuals")
    c1, c2 = st.columns(2)
    with c1:
        chart1 = alt.Chart(edges).mark_circle(size=90).encode(
            x=alt.X('%bets:Q', title='% Bets'),
            y=alt.Y('%handle:Q', title='% Handle'),
            color=alt.Color('edge_score:Q', title='Edge Score'),
            tooltip=['matchup','market','side','odds','edge_score','irrationality','rlm','steam']
        )
        st.altair_chart(chart1, use_container_width=True)
    with c2:
        chart2 = alt.Chart(edges).mark_bar().encode(
            x=alt.X('edge_score:Q', title='Edge Score'),
            y=alt.Y('matchup:N', sort='-x'),
            color=alt.Color('market:N'),
            tooltip=['side','odds','%bets','%handle']
        )
        st.altair_chart(chart2, use_container_width=True)

    st.subheader("Export Picks")
    export_cols = ['matchup','market','side','odds','edge_score','irrationality','rlm','steam','edge_pp','kelly_frac','stake_units','%bets','%handle','tags']
    exp = edges[export_cols].copy()
    exp['generated_utc'] = datetime.utcnow().isoformat()
    st.download_button("Download Picks CSV", data=exp.to_csv(index=False).encode('utf-8'), file_name='foxedge_nfl_edges.csv', mime='text/csv')

    st.caption("No hype. No ghost picks. This is your edge map. Use it or punt it.")


if __name__ == '__main__':
    main()
