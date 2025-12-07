import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sqlite3
import io
from datetime import datetime, timedelta
import pytz
import altair as alt
from PIL import Image, ImageDraw, ImageFont
import os
import re
from pathlib import Path
import json
import qrcode
import logging
import time
import math
from streamlit_autorefresh import st_autorefresh

# ============================================================================
# NFL-SPECIFIC KEY NUMBER DISTRIBUTIONS (Enhanced)
# ============================================================================

# Expanded NFL margin distribution with sub-point granularity
NFL_KEY_MASS = {
    -21: 0.003, -20: 0.003, -17: 0.004, -16: 0.005, -14: 0.006, -13: 0.007,
    -12: 0.008, -11: 0.012, -10: 0.018, -9: 0.022, -8: 0.028, -7: 0.095,  # 7 is CRITICAL
    -6: 0.055, -5: 0.038, -4: 0.062, -3: 0.160,  # 3 is KING
    -2: 0.048, -1: 0.038, 0: 0.012,
    1: 0.038, 2: 0.048, 3: 0.160, 4: 0.062, 5: 0.038, 6: 0.055,
    7: 0.095, 8: 0.028, 9: 0.022, 10: 0.018, 11: 0.012, 12: 0.008,
    13: 0.007, 14: 0.006, 16: 0.005, 17: 0.004, 20: 0.003, 21: 0.003
}

# NFL total key numbers with historical frequency
NFL_TOTAL_KEYS = {
    37: 0.025, 38: 0.022, 39: 0.028, 40: 0.032, 41: 0.048, 42: 0.038,
    43: 0.055, 44: 0.062, 45: 0.058, 46: 0.048, 47: 0.055, 48: 0.045,
    49: 0.042, 50: 0.038, 51: 0.048, 52: 0.035, 53: 0.032, 54: 0.035
}

# Half-point premium values (how much value crossing gains)
HALF_POINT_VALUE = {
    2.5: 0.8, 3.0: 2.5, 3.5: 1.2, 6.5: 0.9, 7.0: 2.0, 7.5: 0.8,
    9.5: 0.5, 10.0: 0.7, 10.5: 0.4, 13.5: 0.4, 14.0: 0.5
}

# Wong teaser sweet spots (expanded)
WONG_TEASER_RANGES = {
    "PRIME_FAV": (7.5, 8.5),    # Tease down through 7 & 3
    "GOOD_FAV": (8.5, 9.5),     # Tease down through 7 & 3  
    "PRIME_DOG": (1.5, 2.5),    # Tease up through 3 & 7
    "GOOD_DOG": (2.5, 3.5),     # Tease up through 3 & 7
    "AVOID_FAV": (4.5, 6.5),    # Dead zone - wastes teaser points
    "AVOID_DOG": (-1.5, 1.5),   # Dead zone
}

# ============================================================================
# NFL SITUATIONAL SPOTS (Sharp Betting Angles)
# ============================================================================

NFL_SITUATIONAL_SPOTS = {
    "HOME_DOG": {"desc": "Home underdog", "edge_boost": 0.02},
    "DIV_DOG": {"desc": "Division underdog", "edge_boost": 0.015},
    "PRIMETIME_UNDER": {"desc": "Primetime under", "edge_boost": 0.01},
    "LOOK_AHEAD": {"desc": "Look-ahead spot", "edge_boost": 0.02},
    "REVENGE": {"desc": "Revenge game", "edge_boost": 0.01},
    "OFF_BYE": {"desc": "Off bye week", "edge_boost": 0.015},
    "SHORT_WEEK": {"desc": "Short rest", "edge_penalty": -0.01},
    "WEST_EARLY": {"desc": "West coast early", "edge_penalty": -0.015},
    "TRAP_GAME": {"desc": "Trap game", "edge_boost": 0.025},
}

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

HISTORY_FILE = "mood_history.csv"
SPLITS_HISTORY_FILE = "splits_history.csv"
GATE_THRESHOLD = 60

# Professional signal thresholds (NFL-tuned)
EXPOSURE_THRESHOLD = 3000       # Higher for NFL
ODDLOT_THRESHOLD = 0.12         # 12% ticket/handle divergence for NFL
CORR_THRESHOLD = 0.025          # 2.5% correlation breach
RLM_BET_THRESHOLD = 6.0         # Higher threshold for NFL (more public action)
STEAM_ODDS_THRESHOLD = 3        # Odds change threshold

# Composite weights
DEFAULT_W_EV = 0.40
DEFAULT_W_IRR = 0.25
DEFAULT_W_RLM = 0.20
DEFAULT_W_STEAM = 0.15
DEFAULT_KELLY_CAP = 0.04  # More conservative for NFL

# DraftKings event group for NFL
LEAGUE_GROUPS = {"NFL": 88808, "NCAAF": 88806}

st.set_page_config(
    page_title="FoxEdge NFL Market Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Database setup
BETLOGS_ROOT = os.getenv("BETLOGS_ROOT", os.getcwd())
BETLOGS_DB = os.path.join(BETLOGS_ROOT, "bet_logs_nfl.db")

def _ensure_parent(path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

# ============================================================================
# NFL-SPECIFIC PROBABILITY MODELS
# ============================================================================

def nfl_cover_prob_enhanced(spread_line: float, is_home: bool = False, 
                           is_divisional: bool = False) -> float | None:
    """
    Enhanced NFL spread cover probability with home/divisional adjustments.
    """
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
    
    # NFL-specific adjustments
    if is_home and L < 0:  # Home underdog
        p = min(0.95, p * 1.02)  # 2% boost
    if is_divisional:
        p = max(0.05, min(0.95, p + 0.01))  # Divisional games tighter
    
    return max(0.0, min(1.0, p))

def nfl_total_prob_enhanced(total_line: float, is_outdoor: bool = True,
                           temp: float = None, wind: int = None,
                           is_primetime: bool = False) -> float | None:
    """
    Enhanced NFL total probability with weather and game context.
    """
    if total_line is None:
        return None
    try:
        tl = float(total_line)
    except Exception:
        return None
    
    sd = 9.8  # NFL total standard deviation
    mu = tl  # Use line as baseline
    
    # Weather adjustments
    if is_outdoor and temp is not None:
        if temp < 32:  # Freezing
            mu -= 1.5
        elif temp < 40:
            mu -= 0.8
        elif temp > 85:  # Hot
            mu -= 0.5
    
    if wind is not None and wind > 15:
        mu -= (wind - 15) * 0.15  # Heavy wind penalty
    
    # Primetime adjustment (historically lower scoring)
    if is_primetime:
        mu -= 0.8
    
    from math import erf, sqrt
    z = (tl - mu) / sd
    p_over = 0.5 * (1 - erf(z / sqrt(2)))
    
    # Key number bumps
    bump = sum(v for k, v in NFL_TOTAL_KEYS.items() if abs(tl - k) < 0.6)
    p_over = p_over + 0.5 * bump
    
    return max(0.0, min(1.0, p_over))

# ============================================================================
# KEY NUMBER CROSSING DETECTION (Enhanced)
# ============================================================================

NFL_KEY_CROSS_VALUE = {
    3: 2.50,   # MOST CRITICAL
    7: 2.00,   # SECOND MOST CRITICAL
    6: 1.00,
    10: 0.80,
    4: 0.75,
    14: 0.60,
    2: 0.50,
    1: 0.40,
}

def _key_weighted_spread_move(entry_spread, close_spread):
    """NFL-aware spread movement with critical key crossing penalties."""
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
    
    # Heavy penalties for crossing critical numbers
    for k, val in NFL_KEY_CROSS_VALUE.items():
        if lo < k <= hi:
            bonus += val * (1 if base > 0 else -1)
    
    # Half-point crossing value
    for half_pt, premium in HALF_POINT_VALUE.items():
        if lo < half_pt <= hi:
            bonus += premium * 0.3 * (1 if base > 0 else -1)
    
    return round(base + bonus, 3)

def _crossed_key(es, cs, key):
    """Check if line crossed a key number."""
    if es is None or cs is None:
        return False
    lo, hi = sorted([es, cs])
    return lo < key <= hi

def wong_teaser_tag(spread_line: float) -> str | None:
    """Identify Wong teaser spots."""
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
# NFL SITUATIONAL ANALYSIS
# ============================================================================

def detect_nfl_situation(row: pd.Series) -> list[str]:
    """Detect NFL situational betting angles."""
    situations = []
    
    matchup = str(row.get("matchup", ""))
    spread = row.get("spread")
    market = str(row.get("market", "")).lower()
    
    # Parse teams (rough heuristic)
    if " @ " in matchup:
        away, home = matchup.split(" @ ")
    elif " vs " in matchup:
        parts = matchup.split(" vs ")
        home, away = parts[0], parts[1] if len(parts) > 1 else ""
    else:
        return situations
    
    # Home dog detection
    if spread is not None and spread > 0 and market == "spread":
        situations.append("HOME_DOG")
    
    # Could add more situational detection here based on metadata
    # (divisional, primetime, etc. - would need additional data sources)
    
    return situations

# ============================================================================
# TIME BUCKET ANALYSIS (NFL-Specific)
# ============================================================================

def _nfl_time_bucket(ts):
    """NFL-specific time windows for line movement analysis."""
    tz = pytz.timezone("America/Los_Angeles")
    if ts is None:
        return "UNKNOWN"
    if getattr(ts, "tzinfo", None) is None:
        ts = tz.localize(pd.to_datetime(ts))
    pt = ts.astimezone(tz)
    dow = pt.weekday()
    h = pt.hour
    
    # NFL-specific windows
    if dow == 1:  # Tuesday
        return "OPEN"  # Lines typically open Tuesday
    if dow in (2, 3):  # Wed-Thu
        return "EARLY_WEEK"
    if dow == 4:  # Friday
        return "LATE_WEEK"
    if dow == 5 and h < 18:  # Saturday day
        return "SATURDAY"
    if dow == 5 and h >= 18:  # Saturday night
        return "SHARP_HOUR"
    if dow == 6 and h < 9:  # Sunday morning
        return "LOCK_90m"
    if dow == 6 and h >= 9:  # Sunday (gameday)
        return "GAMEDAY"
    if dow == 0:  # Monday
        return "MNF"
    return "MIDWEEK"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_odds(odds_str):
    """Clean odds string to integer."""
    try:
        return int(str(odds_str).replace("−", "-").strip())
    except Exception:
        try:
            return int(float(odds_str))
        except Exception:
            return None

def _parse_total_from_side(side_text: str):
    """Extract over/under and total from side text."""
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
    """Extract spread value from side text."""
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
    """Convert American odds to decimal."""
    try:
        american = int(american)
    except Exception:
        return None
    return (1 + american/100) if american > 0 else (1 + 100/abs(american))

def implied_prob_from_odds(odds):
    """Convert American odds to implied probability."""
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
# DATABASE LOGGING
# ============================================================================

def log_snapshot_row(game_id, matchup, market, side, book, snapshot_type, 
                    odds_val=None, total_val=None, spread_val=None, 
                    bets_pct=None, handle_pct=None, db_path: str = BETLOGS_DB):
    """Log line snapshot to database."""
    _ensure_parent(db_path)
    try:
        with sqlite3.connect(db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS line_snapshots (
                    game_id        TEXT,
                    matchup        TEXT,
                    market         TEXT,
                    side           TEXT,
                    book           TEXT,
                    snapshot_type  TEXT,
                    odds           INTEGER,
                    total          REAL,
                    spread         REAL,
                    bets_pct       REAL,
                    handle_pct     REAL,
                    timestamp_utc  TEXT,
                    PRIMARY KEY (game_id, market, side, book, snapshot_type)
                );
            """)
            ts = datetime.utcnow().isoformat()
            con.execute("""
                INSERT OR REPLACE INTO line_snapshots
                (game_id, matchup, market, side, book, snapshot_type, odds, total, spread, bets_pct, handle_pct, timestamp_utc)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                str(game_id), str(matchup), str(market), str(side), str(book), str(snapshot_type),
                None if odds_val is None else int(odds_val),
                None if total_val is None else float(total_val),
                None if spread_val is None else float(spread_val),
                None if bets_pct is None else float(bets_pct),
                None if handle_pct is None else float(handle_pct),
                ts
            ))
    except Exception as e:
        logging.error(f"[log_snapshot_row] {e}")

def log_snapshot(df, snapshot_type: str = "OPEN"):
    """
    Log line snapshots for a DataFrame.

    snapshot_type examples:
      - 'OPEN'       : default open snapshot (per-load)
      - 'CLOSE'      : closing line snapshot (use log_close_snapshot for that flow)
      - 'YOUR_REC'   : your recommended line snapshot
      - 'WEEK_OPEN'  : Tuesday morning weekly open snapshot for the slate
    """
    for _, r in df.iterrows():
        matchup = r.get('matchup')
        market = r.get('market')
        side = r.get('side')
        odds = clean_odds(r.get('odds'))
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
            bets_pct=r.get('%bets'), handle_pct=r.get('%handle')
        )

def log_close_snapshot(df):
    """Log CLOSE snapshots for DataFrame."""
    for _, r in df.iterrows():
        matchup = r.get('matchup')
        market = r.get('market')
        side = r.get('side')
        odds = clean_odds(r.get('odds'))
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
            gid, matchup, market, side, "DK", "CLOSE",
            odds_val=odds, total_val=total_val, spread_val=spread_val,
            bets_pct=r.get('%bets'), handle_pct=r.get('%handle')
        )

def log_your_rec_snapshot(matchup, market, side, odds_val=None, total_val=None, 
                         spread_val=None, bets_pct=None, handle_pct=None):
    """Log YOUR_REC snapshot."""
    date_iso = datetime.now(pytz.timezone("America/Los_Angeles")).date().isoformat()
    gid = f"{date_iso}_{str(matchup).replace(' ','_')}_{market}_{str(side).replace(' ','_')}"[:128]
    log_snapshot_row(
        gid, matchup, market, side, "DK", "YOUR_REC",
        odds_val=odds_val, total_val=total_val, spread_val=spread_val,
        bets_pct=bets_pct, handle_pct=handle_pct
    )


# ============================================================================
# AUTO-LOG WEEKLY OPEN SNAPSHOT (Tuesday morning helper)
# ============================================================================
def auto_log_tuesday_open(df, db_path: str = BETLOGS_DB, tz_name: str = "America/Los_Angeles"):
    """
    If it's Tuesday morning (local timezone) and we haven't logged this week's
    WEEK_OPEN snapshot yet, log a WEEK_OPEN snapshot for the given DataFrame.

    Designed to be cheap and idempotent:
    - Safe to call on every app refresh.
    - Uses a tiny 'meta' table in bet_logs_nfl.db to avoid duplicate logs
      on the same Tuesday.
    """
    try:
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)

        # Tuesday = 1 (Monday = 0)
        if now.weekday() != 1:
            return

        # "Morning" window; adjust if you want (e.g., 6:00–12:00 local time)
        if not (6 <= now.hour < 12):
            return

        _ensure_parent(db_path)
        with sqlite3.connect(db_path) as con:
            # Tiny meta table to track last weekly snapshot timestamp
            con.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

            row = con.execute(
                "SELECT value FROM meta WHERE key = 'weekly_open_snapshot'"
            ).fetchone()

            if row:
                try:
                    last = datetime.fromisoformat(row[0])
                    # Already logged for this Tuesday
                    if last.date() == now.date():
                        return
                except Exception:
                    # If parsing fails, we just treat it as "not logged"
                    pass

            # Log WEEK_OPEN snapshot for the current slate
            log_snapshot(df, snapshot_type="WEEK_OPEN")

            # Mark this Tuesday as completed
            con.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("weekly_open_snapshot", now.isoformat())
            )

    except Exception as e:
        logging.error(f"[auto_log_tuesday_open] {e}")

def auto_log_midweek_open(df, db_path: str = BETLOGS_DB, tz_name: str = "America/Los_Angeles"):
    """
    If it's midweek (Thursday morning) and we haven't logged this week's
    MIDWEEK snapshot yet, log a MIDWEEK snapshot for the given DataFrame.

    Pattern mirrors auto_log_tuesday_open:
    - Safe to call on every app refresh.
    - Uses the same 'meta' table in bet_logs_nfl.db to avoid duplicate logs
      on the same midweek day.
    """
    try:
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)

        # Thursday = 3 (Monday = 0)
        if now.weekday() != 3:
            return

        # "Morning" window; adjust if you want (6:00–12:00 local time)
        if not (6 <= now.hour < 12):
            return

        _ensure_parent(db_path)
        with sqlite3.connect(db_path) as con:
            # Make sure meta table exists
            con.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

            row = con.execute(
                "SELECT value FROM meta WHERE key = 'midweek_open_snapshot'"
            ).fetchone()

            if row:
                try:
                    last = datetime.fromisoformat(row[0])
                    # Already logged for this midweek day
                    if last.date() == now.date():
                        return
                except Exception:
                    # If parsing fails, treat it as "not logged"
                    pass

            # Log MIDWEEK snapshot for the current slate
            log_snapshot(df, snapshot_type="MIDWEEK")

            # Mark this midweek day as completed
            con.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("midweek_open_snapshot", now.isoformat())
            )

    except Exception as e:
        logging.error(f"[auto_log_midweek_open] {e}")

# ============================================================================
# CLV COMPUTATION (Enhanced for NFL)
# ============================================================================

def compute_and_log_clv(db_path: str = BETLOGS_DB) -> pd.DataFrame:
    """Compute Closing Line Value with NFL-specific metrics."""
    with sqlite3.connect(db_path) as con:
        con.execute("""
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
                computed_utc TEXT,
                PRIMARY KEY (game_id, market, side, book)
            );
        """)
        snaps = pd.read_sql_query("SELECT * FROM line_snapshots", con)
    
    if snaps.empty:
        return pd.DataFrame()
    
    snaps = snaps[snaps["book"] == "DK"].copy()
    keys = ["game_id", "matchup", "market", "side", "book"]
    
    entry = snaps[snaps["snapshot_type"] == "YOUR_REC"][keys + ["odds","total","spread","bets_pct","handle_pct"]].rename(columns={
        "odds":"entry_odds","total":"entry_total","spread":"entry_spread",
        "bets_pct":"entry_bets_pct","handle_pct":"entry_handle_pct"
    })
    close = snaps[snaps["snapshot_type"] == "CLOSE"][keys + ["odds","total","spread","bets_pct","handle_pct"]].rename(columns={
        "odds":"close_odds","total":"close_total","spread":"close_spread",
        "bets_pct":"close_bets_pct","handle_pct":"close_handle_pct"
    })
    
    merged = entry.merge(close, on=keys, how="inner")
    
    if merged.empty:
        return pd.DataFrame()
    
    # Flow deltas
    merged["bets_delta"] = (merged.get("close_bets_pct") - merged.get("entry_bets_pct"))
    merged["handle_delta"] = (merged.get("close_handle_pct") - merged.get("entry_handle_pct"))
    
    # CLV metrics
    def _clv_prob_pp(row):
        eo = row.get("entry_odds")
        co = row.get("close_odds")
        if eo is None or co is None:
            return None
        pe = implied_prob_from_odds(eo)
        pc = implied_prob_from_odds(co)
        if pe is None or pc is None:
            return None
        return round(pc - pe, 3)
    
    def _clv_line_move(row):
        if str(row.get("market","")).lower().startswith("total"):
            et, ct = row.get("entry_total"), row.get("close_total")
            if et is None or ct is None:
                return None
            side = str(row.get("side","")).lower()
            if side.startswith("over"):
                return round(et - ct, 3)
            if side.startswith("under"):
                return round(ct - et, 3)
        return None
    
    def _clv_spread_move(row):
        if str(row.get("market","")).lower().startswith("spread"):
            es, cs = row.get("entry_spread"), row.get("close_spread")
            if es is None or cs is None:
                return None
            return _key_weighted_spread_move(es, cs)
        return None
    
    merged["clv_prob_pp"] = merged.apply(_clv_prob_pp, axis=1)
    merged["clv_line_move"] = merged.apply(_clv_line_move, axis=1)
    merged["clv_spread_move"] = merged.apply(_clv_spread_move, axis=1)
    
    # Key crossings (NFL-specific)
    merged["crossed_3"] = merged.apply(lambda r: int(_crossed_key(r.get("entry_spread"), r.get("close_spread"), 3)), axis=1)
    merged["crossed_7"] = merged.apply(lambda r: int(_crossed_key(r.get("entry_spread"), r.get("close_spread"), 7)), axis=1)
    merged["crossed_6"] = merged.apply(lambda r: int(_crossed_key(r.get("entry_spread"), r.get("close_spread"), 6)), axis=1)
    merged["crossed_10"] = merged.apply(lambda r: int(_crossed_key(r.get("entry_spread"), r.get("close_spread"), 10)), axis=1)
    
    # Wong teaser tags
    merged["wong_tag_entry"] = merged["entry_spread"].apply(wong_teaser_tag)
    merged["wong_tag_close"] = merged["close_spread"].apply(wong_teaser_tag)
    merged["lost_wong_value"] = merged.apply(
        lambda r: int((r.get("wong_tag_entry") is not None) and (r.get("wong_tag_close") is None)), 
        axis=1
    )
    
    # Half-point value calculation
    def _calc_half_point_value(row):
        es, cs = row.get("entry_spread"), row.get("close_spread")
        if es is None or cs is None:
            return None
        for half_pt, premium in HALF_POINT_VALUE.items():
            if _crossed_key(es, cs, half_pt):
                return premium
        return 0.0
    
    merged["half_point_value"] = merged.apply(_calc_half_point_value, axis=1)
    
    # Primary CLV and edge labels
    def _primary_clv(row):
        if pd.notna(row.get("clv_line_move")):
            return ("total", row.get("clv_line_move"))
        if pd.notna(row.get("clv_spread_move")):
            return ("spread", row.get("clv_spread_move"))
        return (None, None)
    
    def _edge_label(row):
        mkt, clv_val = _primary_clv(row)
        if clv_val is None:
            return None
        hd = row.get("handle_delta")
        
        try:
            if float(clv_val) <= 0:
                return "Wrong-way"
            if hd is None:
                return "Signal"
            hdv = abs(float(hd))
            if hdv <= 3.0:
                return "Signal"
            if hdv >= 10.0:
                return "Herd"
            return "Mixed"
        except Exception:
            return None
    
    def _edge_badge(lbl):
        return {
            "Signal": "🟢 Sharp Signal",
            "Herd": "🟡 Public Herd",
            "Mixed": "🟠 Mixed Flow",
            "Wrong-way": "🔴 Wrong Way"
        }.get(lbl, None)
    
    merged["primary_market"] = merged.apply(lambda r: _primary_clv(r)[0], axis=1)
    merged["clv_primary"] = merged.apply(lambda r: _primary_clv(r)[1], axis=1)
    merged["edge_label"] = merged.apply(_edge_label, axis=1)
    merged["edge_badge"] = merged["edge_label"].apply(_edge_badge)
    
    # Elasticity and quality metrics
    def _safe_div(num, den):
        try:
            if num is None or den is None:
                return None
            d = float(den)
            if d == 0:
                return None
            return round(float(num)/d, 4)
        except Exception:
            return None
    
    merged["elasticity_total"] = merged.apply(
        lambda r: _safe_div((r.get("close_total") - r.get("entry_total")) if (r.get("close_total") is not None and r.get("entry_total") is not None) else None, r.get("handle_delta")), 
        axis=1
    )
    merged["elasticity_spread"] = merged.apply(
        lambda r: _safe_div((r.get("close_spread") - r.get("entry_spread")) if (r.get("close_spread") is not None and r.get("entry_spread") is not None) else None, r.get("handle_delta")), 
        axis=1
    )
    merged["elasticity_primary"] = merged.apply(
        lambda r: r.get("elasticity_total") if pd.notna(r.get("elasticity_total")) else r.get("elasticity_spread"),
        axis=1
    )
    
    merged["clv_quality_total"] = merged.apply(
        lambda r: _safe_div(r.get("clv_line_move"), abs(r.get("handle_delta")) if r.get("handle_delta") is not None else None),
        axis=1
    )
    merged["clv_quality_spread"] = merged.apply(
        lambda r: _safe_div(r.get("clv_spread_move"), abs(r.get("handle_delta")) if r.get("handle_delta") is not None else None),
        axis=1
    )
    merged["move_quality_spread_keyed"] = merged["clv_quality_spread"]
    
    merged["computed_utc"] = datetime.utcnow().isoformat()
    
    # Write to database
    with sqlite3.connect(db_path) as con:
        rows = merged[[
            "game_id","matchup","market","side","book",
            "entry_odds","close_odds","entry_total","close_total","entry_spread","close_spread",
            "entry_bets_pct","close_bets_pct","entry_handle_pct","close_handle_pct",
            "bets_delta","handle_delta",
            "clv_prob_pp","clv_line_move","clv_spread_move",
            "clv_quality_total","clv_quality_spread","elasticity_total","elasticity_spread",
            "primary_market","clv_primary","elasticity_primary","edge_label","edge_badge",
            "crossed_3","crossed_7","crossed_6","crossed_10",
            "wong_tag_entry","wong_tag_close","lost_wong_value","move_quality_spread_keyed",
            "half_point_value","computed_utc"
        ]].to_records(index=False)
        
        con.executemany("""
            INSERT INTO clv_logs (
                game_id, matchup, market, side, book,
                entry_odds, close_odds, entry_total, close_total, entry_spread, close_spread,
                entry_bets_pct, close_bets_pct, entry_handle_pct, close_handle_pct,
                bets_delta, handle_delta,
                clv_prob_pp, clv_line_move, clv_spread_move,
                clv_quality_total, clv_quality_spread, elasticity_total, elasticity_spread,
                primary_market, clv_primary, elasticity_primary, edge_label, edge_badge,
                crossed_3, crossed_7, crossed_6, crossed_10,
                wong_tag_entry, wong_tag_close, lost_wong_value, move_quality_spread_keyed,
                half_point_value, computed_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(game_id, market, side, book) DO UPDATE SET
                entry_odds=excluded.entry_odds,
                close_odds=excluded.close_odds,
                entry_total=excluded.entry_total,
                close_total=excluded.close_total,
                entry_spread=excluded.entry_spread,
                close_spread=excluded.close_spread,
                entry_bets_pct=excluded.entry_bets_pct,
                close_bets_pct=excluded.close_bets_pct,
                entry_handle_pct=excluded.entry_handle_pct,
                close_handle_pct=excluded.close_handle_pct,
                bets_delta=excluded.bets_delta,
                handle_delta=excluded.handle_delta,
                clv_prob_pp=excluded.clv_prob_pp,
                clv_line_move=excluded.clv_line_move,
                clv_spread_move=excluded.clv_spread_move,
                clv_quality_total=excluded.clv_quality_total,
                clv_quality_spread=excluded.clv_quality_spread,
                elasticity_total=excluded.elasticity_total,
                elasticity_spread=excluded.elasticity_spread,
                primary_market=excluded.primary_market,
                clv_primary=excluded.clv_primary,
                elasticity_primary=excluded.elasticity_primary,
                edge_label=excluded.edge_label,
                edge_badge=excluded.edge_badge,
                crossed_3=excluded.crossed_3,
                crossed_7=excluded.crossed_7,
                crossed_6=excluded.crossed_6,
                crossed_10=excluded.crossed_10,
                wong_tag_entry=excluded.wong_tag_entry,
                wong_tag_close=excluded.wong_tag_close,
                lost_wong_value=excluded.lost_wong_value,
                move_quality_spread_keyed=excluded.move_quality_spread_keyed,
                half_point_value=excluded.half_point_value,
                computed_utc=excluded.computed_utc
        """, list(rows))
    
    return merged

# ============================================================================
# LINE EVOLUTION VIEW (OPEN / WEEK_OPEN / MIDWEEK / CLOSE)
# ============================================================================

def load_line_evolution(db_path: str = BETLOGS_DB) -> pd.DataFrame:
    """
    Load line snapshots and pivot them into a comparison table by snapshot_type.

    Expected snapshot_type values:
      - 'OPEN'       : per-load open snapshot
      - 'WEEK_OPEN'  : Tuesday weekly open snapshot
      - 'MIDWEEK'    : Thursday midweek snapshot
      - 'CLOSE'      : close snapshot (from log_close_snapshot / CLV flow)
    """
    try:
        _ensure_parent(db_path)
        with sqlite3.connect(db_path) as con:
            df = pd.read_sql_query(
                """
                SELECT
                    matchup,
                    market,
                    side,
                    book,
                    snapshot_type,
                    odds,
                    spread,
                    total,
                    timestamp_utc
                FROM line_snapshots
                WHERE book = 'DK'
                """,
                con,
            )
    except Exception as e:
        logging.error(f"[load_line_evolution] {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Pivot snapshot_type into columns for odds / spread / total
    evo = df.pivot_table(
        index=["matchup", "market", "side"],
        columns="snapshot_type",
        values=["odds", "spread", "total"],
        aggfunc="first",
    ).reset_index()

    # Flatten MultiIndex columns like ('odds','OPEN') -> 'odds_OPEN'
    new_cols = []
    for col in evo.columns:
        if isinstance(col, tuple):
            base, snap = col
            if base in ("odds", "spread", "total") and snap:
                new_cols.append(f"{base}_{snap}")
            else:
                new_cols.append(base or snap)
        else:
            new_cols.append(col)
    evo.columns = new_cols

    return evo

# ============================================================================
# DATA FETCHING FROM DRAFTKINGS
# ============================================================================

def fetch_dk_splits(event_group: int | None = None, date_range: str = "today", 
                   market: str = "All", league: str = "NFL") -> pd.DataFrame:
    """Fetch betting splits from DraftKings Network with enhanced parsing."""
    import re
    from urllib.parse import urlencode, urlparse, parse_qs

    pac = pytz.timezone("America/Los_Angeles")
    now = datetime.now(pac)
    
    if event_group is None:
        event_group = LEAGUE_GROUPS.get(league, 88807)

    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
    first_url = f"{base}?{urlencode(params)}"

    UA = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    def _get_html(url: str) -> str:
        """Fetch HTML with retry logic."""
        for attempt in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=20)
                r.raise_for_status()
                if len(r.text) > 1000:  # Sanity check
                    return r.text
            except Exception as e:
                if attempt == 2:
                    st.error(f"Failed to fetch after 3 attempts: {e}")
                continue
        return ""

    def _discover_pages(html: str, first: str) -> list[str]:
        """Discover all pagination URLs."""
        if not html:
            return [first]
        soup = BeautifulSoup(html, "html.parser")
        urls = {first}
        
        # Look for pagination links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "tb_page=" in href and "draftkings-sportsbook-betting-splits" in href:
                if not href.startswith("http"):
                    href = f"https://dknetwork.draftkings.com{href}"
                urls.add(href)
        
        # Regex fallback
        for m in re.finditer(r'href="([^"]+tb_page=\d+[^"]*)"', html):
            url = m.group(1)
            if not url.startswith("http"):
                url = f"https://dknetwork.draftkings.com{url}"
            urls.add(url)
        
        # Sort by page number
        def pnum(u: str) -> int:
            try:
                return int(parse_qs(urlparse(u).query).get("tb_page", ["1"])[0])
            except:
                return 1
        
        return sorted(urls, key=pnum)

    def _clean_text(x: str) -> str:
        x = x or ""
        x = re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", x, flags=re.I)
        x = re.sub(r"\s+", " ", x).strip()
        return x

    def _parse_game_blocks(soup: BeautifulSoup) -> list[dict]:
        """Parse game blocks from HTML."""
        records = []
        
        # Try multiple selector strategies
        blocks = soup.select("div.tb-se")
        if not blocks:
            blocks = soup.select("section.tb-section div[class*='tb-se']")
        if not blocks:
            blocks = soup.select("div[class*='game'], div[class*='event']")
        
        for g in blocks:
            # Find title (matchup)
            title_el = (
                g.select_one("div.tb-se-title h5") or
                g.select_one("h5.tb-se-title") or
                g.select_one("h5") or
                g.select_one("div[class*='title'] h5")
            )
            
            if not title_el:
                continue
            
            title = _clean_text(title_el.get_text(strip=True))
            if not title or len(title) < 5:
                continue
            
            # Find game time
            time_el = (
                g.select_one("div.tb-se-title span") or
                g.select_one("span.tb-time") or
                g.select_one("div[class*='time'] span")
            )
            game_time = _clean_text(time_el.get_text(strip=True)) if time_el else ""

            # Find markets
            market_sections = (
                g.select(".tb-market-wrap > div") or
                g.select("div.tb-market-wrap") or
                g.select("div[class*='market']")
            )
            
            for section in market_sections:
                # Market name
                head = (
                    section.select_one(".tb-se-head > div") or
                    section.select_one(".tb-se-head") or
                    section.select_one("div[class*='head']")
                )
                
                if not head:
                    continue
                
                market_name = _clean_text(head.get_text(strip=True))
                
                # Only process main markets
                if not any(m in market_name for m in ("Moneyline", "Total", "Spread")):
                    continue
                
                # Normalize market name
                if "Moneyline" in market_name:
                    market_name = "Moneyline"
                elif "Total" in market_name:
                    market_name = "Total"
                elif "Spread" in market_name:
                    market_name = "Spread"

                # Find rows
                rows = (
                    section.select(".tb-sodd") or
                    section.select(".tb-odd") or
                    section.select(".tb-row") or
                    section.select("tr") or
                    section.select("div[class*='row']")
                )
                
                for row in rows:
                    # Side/team
                    side_el = (
                        row.select_one(".tb-slipline") or
                        row.select_one(".tb-side") or
                        row.select_one("td:nth-child(1)") or
                        row.select_one("div[class*='side']")
                    )
                    
                    # Odds
                    odds_el = (
                        row.select_one("a.tb-odd-s") or
                        row.select_one(".tb-odd-s") or
                        row.select_one("td:nth-child(2)") or
                        row.select_one("div[class*='odd']")
                    )
                    
                    if not side_el or not odds_el:
                        continue
                    
                    side_txt = _clean_text(side_el.get_text(strip=True))
                    odds_txt = _clean_text(odds_el.get_text(strip=True))
                    
                    # Extract percentages (look for % symbols)
                    pct_texts = []
                    for elem in row.find_all(string=lambda t: isinstance(t, str) and "%" in t):
                        clean = elem.strip().replace("%", "").replace(",", "")
                        if clean.replace(".", "", 1).isdigit():
                            pct_texts.append(float(clean))
                    
                    # Heuristic: first % is handle, second is bets (DK format)
                    handle_pct = pct_texts[0] if len(pct_texts) >= 1 else 0.0
                    bets_pct = pct_texts[1] if len(pct_texts) >= 2 else 0.0

                    # Parse spread if applicable
                    spread_val = None
                    if market_name == "Spread":
                        spread_val = _parse_spread_from_side(side_txt)

                    records.append({
                        "matchup": title,
                        "game_time": game_time,
                        "market": market_name,
                        "side": side_txt,
                        "odds": clean_odds(odds_txt),
                        "spread": spread_val,
                        "%handle": handle_pct,
                        "%bets": bets_pct,
                        "update_time": now,
                    })
        
        return records

    # Fetch first page
    html = _get_html(first_url)
    
    if not html:
        st.error("Failed to fetch data from DraftKings. The site may be down or blocking requests.")
        return pd.DataFrame()
    
    # Discover all pages
    page_urls = _discover_pages(html, first_url)
    
    st.info(f"Found {len(page_urls)} page(s) to parse...")
    
    # Parse all pages
    all_records = []
    for i, url in enumerate(page_urls):
        page_html = html if i == 0 else _get_html(url)
        if not page_html:
            continue
        
        soup = BeautifulSoup(page_html, "html.parser")
        records = _parse_game_blocks(soup)
        all_records.extend(records)
        
        if records:
            st.success(f"Page {i+1}: Found {len(records)} market rows")
    
    if not all_records:
        st.warning(f"No data found for {date_range}. Trying fallback strategies...")
        
        # Fallback: try different date ranges
        if date_range == "today":
            st.info("Trying 'tomorrow'...")
            return fetch_dk_splits(event_group, "tomorrow", market, league)
        elif date_range == "tomorrow":
            st.info("Trying 'n7days'...")
            return fetch_dk_splits(event_group, "n7days", market, league)
        else:
            st.error("No splits found across all date ranges. The slate may be empty or the site format changed.")
            return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    st.success(f"✅ Successfully parsed {len(df)} total market rows from {df['matchup'].nunique()} games")
    
    return df

# ============================================================================
# PLAYER PROP FETCHING (TD SCORERS)
# ============================================================================


# DKNetwork player props base URL (same network we scrape splits from)
PROPS_BASE = "https://dknetwork.draftkings.com/draftkings-sportsbook-player-props/"
UA_PROPS = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

ODDS_TOKEN_RE = re.compile(r"^[+-]\d{2,4}$")

def _ws_clean(s: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", str(s or "")).strip()


# Helper: Convert arbitrary string/cell containing American odds to int (e.g. "+140" -> 140, "-115" -> -115, "Odds: +250" -> 250)
def _odds_to_int_token(val):
    """
    Convert an arbitrary cell/string containing American odds into an int.
    Examples:
      "+140" -> 140
      "-115" -> -115
      "Odds: +250" -> 250
    Returns None if nothing that looks like +/-### is found.
    """
    if val is None:
        return None
    try:
        s = str(val).strip().replace("−", "-")  # handle weird minus glyph
        m = re.search(r"[+-]\d{2,4}", s)
        if not m:
            return None
        return int(m.group(0))
    except Exception:
        return None




# ============================================================================
# TD RAW MARKET SCRAPER (new helper, preserves DK order, no early return)
# ============================================================================
def _fetch_td_raw_markets():
    """
    Fetches 'Anytime TD Scorer', 'First TD Scorer', and '2+ TDs' markets
    from DraftKings Network's Most Bet Player Props (NFL).
    Tries 'today', 'tomorrow', then 'next 7 days' until results found.
    Returns 3 DataFrames (anytime_df, first_df, multi_df) OR None if nothing.
    NOTE: We intentionally DO NOT re-sort or re-rank. We keep DK's surface order
    and we only trim to .head(10) per market type at the very end.
    """
    base_url = "https://dknetwork.draftkings.com/draftkings-sportsbook-player-props/"
    params = {
        "tb_view": "2",       # Most bet props
        "tb_eg": "88808",     # NFL event group
    }
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    date_windows = ["today", "tomorrow", "n7days"]
    all_data = []

    for date_key in date_windows:
        params["tb_edate"] = date_key
        # print(f"\nChecking {date_key}…")  # suppress console spam in Streamlit

        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        if not response.ok:
            # print(f"Request failed for {date_key}: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if not table:
            # print(f"No table found for {date_key}.")
            continue

        rows = table.find_all("tr")[1:]  # skip header
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) != 5:
                continue
            market = cols[2]
            if any(x in market for x in ["TD", "Touchdown"]):
                all_data.append({
                    "Event": cols[0],
                    "Date": cols[1],
                    "Market": market,
                    "Player": cols[3],
                    "Odds": cols[4],
                    "Range": date_key
                })

    if not all_data:
        # print("No TD-related props found in any window.")
        return None, None, None

    df = pd.DataFrame(all_data)

    # Split by market type
    anytime_df = df[df["Market"].str.contains("Anytime", case=False, na=False)]
    first_df   = df[df["Market"].str.contains("First",   case=False, na=False)]
    multi_df   = df[df["Market"].str.contains("2",       case=False, na=False)]

    return anytime_df.head(10), first_df.head(10), multi_df.head(10)


# ============================================================================
# TD MARKET WRAPPER (wraps raw, builds boards, adds implied prob, returns debug)
# ============================================================================
def _fetch_td_markets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Wrapper around `_fetch_td_raw_markets()` that:
    1. pulls raw TD prop rows in DK's native surfaced order
    2. converts to screenshot-ready boards for:
       - Anytime TD Scorer
       - First TD Scorer
       - 2+ TD Scorers
    3. computes implied % from American odds
    4. preserves Range (today / tomorrow / n7days)

    Returns (anytime_board, first_board, multi_board, debug)
    Each board has columns:
      ["Player", "Matchup", "Odds", "Implied %", "Range"]
    """

    anytime_df, first_df, multi_df = _fetch_td_raw_markets()

    debug = {
        "err": None,
        # We can't easily expose html_len / row_count here without re-scraping,
        # so we just expose counts which is still useful for debugging.
        "any_rows": 0 if anytime_df is None else len(anytime_df),
        "first_rows": 0 if first_df is None else len(first_df),
        "multi_rows": 0 if multi_df is None else len(multi_df),
    }

    def _mk_board(df_src: pd.DataFrame | None) -> pd.DataFrame:
        cols = ["Player","Matchup","Odds","Implied %","Range"]
        if df_src is None or df_src.empty:
            return pd.DataFrame(columns=cols)
        board_rows = []
        for _, r in df_src.iterrows():
            odds_txt = str(r.get("Odds", "")).strip()
            # odds as int (e.g. "+120" -> 120)
            odds_val = _odds_to_int_token(odds_txt)
            if odds_val is None:
                continue
            prob = implied_prob_from_odds(odds_val)
            if prob is None:
                continue
            board_rows.append({
                "Player": r.get("Player"),
                "Matchup": r.get("Event"),
                "Odds": odds_val,
                "Implied %": round(float(prob) * 100.0, 1),
                "Range": r.get("Range"),
            })
        if not board_rows:
            return pd.DataFrame(columns=cols)
        # DO NOT re-sort. Keep the order they were scraped / appended from DK.
        return pd.DataFrame(board_rows, columns=cols)

    anytime_board = _mk_board(anytime_df)
    first_board   = _mk_board(first_df)
    multi_board   = _mk_board(multi_df)

    return anytime_board, first_board, multi_board, debug

def build_props_url(event_group: int = 88808, edate: str = "today") -> str:
    """
    Construct DKNetwork Player Props URL.
    tb_view=2 forces the props table view.
    tb_eg is the event group (88808 = NFL).
    tb_edate controls date scope: today | tomorrow | n7days, etc.
    """
    from urllib.parse import urlencode
    params = {"tb_view": "2", "tb_eg": str(event_group), "tb_edate": edate}
    return f"{PROPS_BASE}?{urlencode(params)}"

def fetch_props_html(event_group: int = 88808, edate: str = "today") -> str:
    """
    Fetch raw HTML for the DKNetwork Player Props page for the given slate.
    This uses plain requests (no Playwright). We just sanity check length.
    Returns empty string if we can't get a usable page.
    """
    url = build_props_url(event_group, edate)
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": UA_PROPS}, timeout=20)
            r.raise_for_status()
            html = r.text
            if html and len(html) > 1000:
                return html
        except Exception as e:
            last_err = e
            continue
    logging.error(f"[fetch_props_html] failed: {last_err}")
    return ""

def parse_player_props(html: str) -> list[dict[str, any]]:
    """
    Parse DKNetwork Player Props HTML into normalized rows.
    Output rows have at least:
      - event / matchup
      - player
      - market (ex: 'Anytime Touchdown Scorer')
      - odds (American int)
    We intentionally keep this lightweight (covers the two common table layouts + a fallback block list).
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, any]] = []
    now_epoch = int(time.time())

    def _guess_player_from_td(td_node, fallback_text: str) -> str:
        """
        Try to pull the player name directly from the prop cell td_node.
        We look for obvious player-bearing elements (strong, a, spans with 'player' in class).
        If we can't find anything structured, fall back to the first 2-3 capitalized tokens
        from fallback_text.
        """
        # 1. Check common child selectors
        for sel in [
            ".tb-player",
            ".player",
            "a[href*='/players/']",
            "strong",
            "h3",
            "h4",
            "span"
        ]:
            el = getattr(td_node, "select_one", lambda *_a, **_k: None)(sel)
            if el:
                cand = _ws_clean(el.get_text(" ", strip=True))
                # basic sanity: must look like a name (2+ words starting capitalized)
                if re.search(r"^[A-Z][a-zA-Z.'-]+\s+[A-Z][a-zA-Z.'-]+", cand):
                    return cand
        # 2. Fallback: extract first two capitalized words from fallback_text
        m = re.search(r"([A-Z][a-zA-Z.'-]+\s+[A-Z][a-zA-Z.'-]+)", fallback_text)
        if m:
            return m.group(1).strip()
        return fallback_text.strip()

    # Pass 0: explicit DKNetwork props table under .pp-table-wrapper
    specific_rows = soup.select("div.pp-table-wrapper table tbody tr")
    if specific_rows:
        for tr in specific_rows:
            tds = tr.select("td")
            if len(tds) != 5:
                continue
            matchup = _ws_clean(tds[0].get_text(" ", strip=True))
            date_time = _ws_clean(tds[1].get_text(" ", strip=True))
            prop_text = _ws_clean(tds[2].get_text(" ", strip=True))
            line_text = _ws_clean(tds[3].get_text(" ", strip=True))
            odds_cell = tds[4]
            odds_txt = _ws_clean(odds_cell.get_text(" ", strip=True))
            odds_val = _odds_to_int_token(odds_txt)
            if odds_val is None:
                continue

            link_el = odds_cell.select_one("a[href]")
            href = link_el["href"].strip() if link_el and link_el.has_attr("href") else None

            # Split "Player Market" text into (player, market)
            def _split_player_market(s: str):
                markets = [
                    "Anytime Touchdown Scorer", "Anytime TD Scorer", "Anytime TD",
                    "Receiving Yards", "Receptions",
                    "Rushing Yards", "Passing Yards", "Passing Attempts", "Completions",
                    "Interceptions Thrown", "Rush Attempts", "Targets", "Passing TDs",
                    "Alt Receptions", "Alt Receiving Yards", "Alt Rushing Yards", "Alt Passing Yards",
                    "Longest Reception", "Longest Rush", "Longest Completion",
                    "Rush + Rec Yards", "Rushing + Receiving Yards",
                    # MLB etc, harmless to keep
                    "Hits + Runs + RBIs", "Strikeouts Thrown O/U", "Strikeouts Thrown",
                    "Home Runs", "Total Bases", "Stolen Bases O/U", "Stolen Bases",
                    "Walks Allowed", "RBIs", "Hits", "Runs",
                ]
                markets.sort(key=len, reverse=True)
                for mkt in markets:
                    if s.endswith(mkt):
                        player_name = s[: -len(mkt)].strip()
                        return player_name, mkt
                    if mkt in s:
                        parts = s.split(mkt, 1)
                        player_name = parts[0].strip()
                        return player_name, mkt
                toks = s.split()
                if len(toks) >= 3:
                    return " ".join(toks[:2]), " ".join(toks[2:])
                return s, None

            player, market = _split_player_market(prop_text)
            # If we failed to confidently grab a player (None / empty / looks like just the market),
            # try to infer it first from the prop cell (tds[2]) and then from the Betslip Line cell (tds[3]),
            # because DKNetwork often sticks the player name under the 'Betslip Line' column instead of
            # the prop description column.
            needs_guess = (
                (not player)
                or (market and player == market)
                or (player is not None and len(player.split()) <= 1)
            )
            if needs_guess:
                guessed = _guess_player_from_td(tds[2], prop_text)
                if (not guessed or guessed == prop_text or (len(guessed.split()) <= 1)) and len(tds) >= 4:
                    # try Betslip Line column (tds[3])
                    guessed_alt = _guess_player_from_td(tds[3], line_text)
                    if guessed_alt:
                        guessed = guessed_alt
                if guessed:
                    player = guessed

            out.append({
                "event": matchup or None,
                "date": date_time or None,
                "game_time": None,
                "player": player or None,
                "market": market or None,
                "line": line_text or None,
                "outcome": None,
                "odds": odds_val,
                "team": None,
                "matchup": matchup or None,
                "update_time": now_epoch,
                "dk_url": href,
            })
        if out:
            return out

    # Pass 1: generic 5-col table (matchup, datetime, prop_text, line, odds)
    for tbl in soup.select("table"):
        first_body_tr = tbl.select_one("tbody tr") or None
        if first_body_tr:
            tds = first_body_tr.select("td")
            if len(tds) == 5 and not tbl.select_one("thead"):
                for tr in tbl.select("tbody tr"):
                    cells = [_ws_clean(td.get_text(" ", strip=True)) for td in tr.select("td")]
                    if len(cells) != 5:
                        continue
                    matchup = cells[0]
                    date_time = cells[1]
                    prop_text = cells[2]
                    line_text = cells[3]
                    odds_cell = tr.select_one("td:nth-child(5)")
                    odds_txt = _ws_clean(odds_cell.get_text(" ", strip=True)) if odds_cell else cells[4]
                    odds_val = _odds_to_int_token(odds_txt)
                    if odds_val is None:
                        continue

                    def _split_player_market(s: str):
                        markets = [
                            "Anytime Touchdown Scorer", "Anytime TD Scorer", "Anytime TD",
                            "Receiving Yards", "Receptions",
                            "Rushing Yards", "Passing Yards", "Passing Attempts", "Completions",
                            "Interceptions Thrown", "Rush Attempts", "Targets", "Passing TDs",
                            "Alt Receptions", "Alt Receiving Yards", "Alt Rushing Yards", "Alt Passing Yards",
                            "Longest Reception", "Longest Rush", "Longest Completion",
                            "Rush + Rec Yards", "Rushing + Receiving Yards",
                            "Hits + Runs + RBIs", "Strikeouts Thrown O/U", "Strikeouts Thrown",
                            "Home Runs", "Total Bases", "Stolen Bases O/U", "Stolen Bases",
                            "Walks Allowed", "RBIs", "Hits", "Runs",
                        ]
                        markets.sort(key=len, reverse=True)
                        for mkt in markets:
                            if s.endswith(mkt):
                                player_name = s[: -len(mkt)].strip()
                                return player_name, mkt
                            if mkt in s:
                                parts = s.split(mkt, 1)
                                player_name = parts[0].strip()
                                return player_name, mkt
                        toks = s.split()
                        if len(toks) >= 3:
                            return " ".join(toks[:2]), " ".join(toks[2:])
                        return s, None

                    player, market = _split_player_market(prop_text)
                    needs_guess = (
                        (not player)
                        or (market and player == market)
                        or (player is not None and len(player.split()) <= 1)
                    )
                    if needs_guess:
                        td_nodes = tr.select("td")
                        guessed = None
                        # try prop/description cell first (index 2)
                        if len(td_nodes) >= 3:
                            guessed = _guess_player_from_td(td_nodes[2], prop_text)
                        # if that fails or is junk, try Betslip Line cell (index 3)
                        if (not guessed or guessed == prop_text or (len(guessed.split()) <= 1)) and len(td_nodes) >= 4:
                            guessed_alt = _guess_player_from_td(td_nodes[3], line_text)
                            if guessed_alt:
                                guessed = guessed_alt
                        if guessed:
                            player = guessed

                    out.append({
                        "event": matchup or None,
                        "date": date_time or None,
                        "game_time": None,
                        "player": player or None,
                        "market": market or None,
                        "line": line_text or None,
                        "outcome": None,
                        "odds": odds_val,
                        "team": None,
                        "matchup": matchup or None,
                        "update_time": now_epoch,
                    })
                # finished fast-path table
                continue

    # Pass 2: very loose div-row fallback
    rows = soup.select("div.tb-row, li, article, div[class*='row']")
    for row in rows:
        txt = _ws_clean(row.get_text(" ", strip=True))
        if not txt:
            continue
        odds_tokens = [t for t in txt.split() if ODDS_TOKEN_RE.match(t)]
        if not odds_tokens:
            continue
        # pull player name heuristically
        player_guess = ""
        for sel in [".tb-player", ".player", "a[href*='/players/']", "strong", "h3", "h4"]:
            el = row.select_one(sel)
            if el:
                player_guess = _ws_clean(el.get_text(" ", strip=True)); break
        if not player_guess:
            m = re.search(r"[A-Z][a-z]+ [A-Z][a-zA-Z.\-']+", txt)
            player_guess = m.group(0) if m else ""
        market_guess = ""
        for sel in [".tb-market", ".market", ".prop-type", ".tb-prop"]:
            el = row.select_one(sel)
            if el:
                market_guess = _ws_clean(el.get_text(" ", strip=True)); break
        line_guess = ""
        for sel in [".tb-line", ".line", ".threshold", ".target"]:
            el = row.select_one(sel)
            if el:
                line_guess = _ws_clean(el.get_text(" ", strip=True)); break
        matchup_guess = ""
        for sel in [".tb-game", ".matchup", ".vs", ".tb-event", ".event"]:
            el = row.select_one(sel)
            if el:
                matchup_guess = _ws_clean(el.get_text(" ", strip=True)); break
        out.append({
            "event": matchup_guess or None,
            "date": None,
            "game_time": None,
            "player": player_guess or None,
            "market": market_guess or None,
            "line": line_guess or None,
            "outcome": None,
            "odds": _odds_to_int_token(odds_tokens[0]),
            "team": None,
            "matchup": matchup_guess or None,
            "update_time": now_epoch,
        })

    return out

def build_td_scorer_board(props_rows: list[dict[str, any]] | None, max_players: int = 10) -> pd.DataFrame:
    """
    Build "MOST BET TD SCORERS" style board from player props.
    We approximate "most bet" using the Anytime TD market odds (shorter odds = everyone is piling on).

    Returns a DataFrame with columns:
      Player, Matchup, Odds, Implied %
    sorted by highest implied probability and deduped per player/matchup.
    """
    cols = ["Player", "Matchup", "Odds", "Implied %"]
    if not props_rows:
        return pd.DataFrame(columns=cols)

    rows_out = []
    for r in props_rows:
        market_txt = str(r.get("market", "")).lower()
        # accept multiple DK label variants for TD scorer markets
        if (
            "anytime touchdown" not in market_txt
            and "anytime td" not in market_txt
            and "td scorer" not in market_txt
        ):
            continue
        odds_val = clean_odds(r.get("odds"))
        if odds_val is None:
            continue
        prob = implied_prob_from_odds(odds_val)
        if prob is None:
            continue
        rows_out.append({
            "Player": r.get("player"),
            "Matchup": r.get("matchup") or r.get("event"),
            "Odds": odds_val,
            "Implied %": round(float(prob) * 100.0, 1),
        })

    if not rows_out:
        return pd.DataFrame(columns=cols)

    df_td = pd.DataFrame(rows_out)
    # highest implied % first
    df_td = df_td.sort_values("Implied %", ascending=False)
    # drop dups so each player/matchup only shows once, keep highest prob
    df_td = df_td.drop_duplicates(subset=["Player", "Matchup"], keep="first")
    df_td = df_td.reset_index(drop=True)
    return df_td.head(max_players)


# ============================================================================
# TD BOARD DEBUG HELPER
# ============================================================================

def fetch_td_boards_with_debug() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    High-level wrapper for TD scorer markets.

    Uses `_fetch_td_markets()` to pull:
    - Most Bet Anytime TD Scorers
    - Most Bet First TD Scorers
    - Most Bet 2+ TD Scorers

    Returns (anytime_board, first_board, multi_board, debug)
    where each board already has columns:
      ["Player", "Matchup", "Odds", "Implied %", "Range"]

    debug exposes scrape metadata for troubleshooting.
    """
    anytime_board, first_board, multi_board, dbg = _fetch_td_markets()

    def _ensure(df_in: pd.DataFrame) -> pd.DataFrame:
        cols = ["Player","Matchup","Odds","Implied %","Range"]
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=cols)
        return df_in[cols] if all(c in df_in.columns for c in cols) else pd.DataFrame(columns=cols)

    return _ensure(anytime_board), _ensure(first_board), _ensure(multi_board), {
        "err": dbg.get("err"),
        "any_rows": dbg.get("any_rows"),
        "first_rows": dbg.get("first_rows"),
        "multi_rows": dbg.get("multi_rows"),
    }

# ============================================================================
# PROBABILITY & EV CALCULATIONS
# ============================================================================

def approx_true_prob(row: pd.Series) -> float | None:
    """Approximate true probability with NFL adjustments."""
    odds = clean_odds(row.get("odds"))
    market = str(row.get("market", ""))
    side = str(row.get("side", ""))
    
    implied = implied_prob_from_odds(odds) if odds is not None else None

    if implied is None:
        if market.lower().startswith("total"):
            ou, tot = _parse_total_from_side(side)
            implied = nfl_total_prob_enhanced(tot) if tot is not None else None
        elif market.lower().startswith("spread"):
            s = _parse_spread_from_side(side)
            implied = nfl_cover_prob_enhanced(s) if s is not None else None

    if implied is None:
        return None

    # Handle/bets skew adjustment
    bets = row.get("%bets", 0.0) or 0.0
    handle = row.get("%handle", 0.0) or 0.0
    skew = (handle - bets) / 100.0
    adjust = 0.30 * skew  # NFL has sharper money influence
    p = max(0.001, min(0.999, implied + adjust))

    # NFL key number micro-adjustments
    if market.lower().startswith("spread"):
        s = _parse_spread_from_side(side)
        if s is not None:
            abs_s = abs(s)
            if abs_s in (3, 7):  # Critical keys
                p = max(0.001, min(0.999, p + 0.015 * (1 if s >= 0 else -1)))
            elif abs_s in (6, 10, 4):  # Secondary keys
                p = max(0.001, min(0.999, p + 0.008 * (1 if s >= 0 else -1)))
    
    if market.lower().startswith("total"):
        ou, tot = _parse_total_from_side(side)
        if tot in NFL_TOTAL_KEYS:
            key_boost = 0.5 * NFL_TOTAL_KEYS[tot]
            p = max(0.001, min(0.999, p + key_boost * (1 if ou and ou.lower() == "over" else -1)))
    
    return float(p)

def prob_to_decimal_edge(p_true: float, odds_american: int) -> float:
    """Calculate expected value."""
    d = _dec(odds_american)
    if d is None or p_true is None:
        return 0.0
    b = d - 1.0
    return float(p_true * b - (1.0 - p_true))

def kelly_fraction(p_true: float, odds_american: int) -> float:
    """Calculate Kelly fraction."""
    d = _dec(odds_american)
    if d is None or p_true is None:
        return 0.0
    b = d - 1.0
    if b <= 0:
        return 0.0
    f = (p_true * b - (1 - p_true)) / b
    return max(0.0, float(f))

# ============================================================================
# EDGE SCORING SYSTEM
# ============================================================================

def build_edge_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Build comprehensive edge scores with NFL analytics."""
    if df.empty:
        return df
    
    out = df.copy()

    # Auto-log a weekly open snapshot on Tuesday mornings (idempotent).
    auto_log_tuesday_open(out)

    # Auto-log a midweek snapshot on Thursday mornings (idempotent).
    auto_log_midweek_open(out)

    out["irrationality"] = (out["%bets"] - out["%handle"]).abs()

    # Calculate probabilities and EV
    probs = []
    evs = []
    kellys = []
    
    for _, r in out.iterrows():
        p_true = approx_true_prob(r)
        odds = clean_odds(r.get("odds"))
        
        if p_true is None or odds is None:
            probs.append(None)
            evs.append(0.0)
            kellys.append(0.0)
            continue
        
        ev = prob_to_decimal_edge(p_true, odds)
        kf = kelly_fraction(p_true, odds)
        
        probs.append(p_true)
        evs.append(ev)
        kellys.append(kf)
    
    out["p_true"] = probs
    out["ev_units"] = evs
    out["kelly_raw"] = kellys
    out["kelly_cap"] = out["kelly_raw"].apply(
        lambda x: min(max(x or 0.0, 0.0), float(st.session_state.get("kelly_cap", DEFAULT_KELLY_CAP)))
    )

    # Add situational tags
    out["situations"] = out.apply(detect_nfl_situation, axis=1)
    out["wong_tag"] = out["spread"].apply(wong_teaser_tag)

    # RLM/Steam detection from snapshots
    try:
        with sqlite3.connect(BETLOGS_DB) as con:
            opens = pd.read_sql_query(
                "SELECT * FROM line_snapshots WHERE snapshot_type='OPEN' AND book='DK'", 
                con
            )
    except Exception:
        opens = pd.DataFrame()

    if not opens.empty:
        keys = ["matchup", "market", "side"]
        j = out.merge(
            opens[keys + ["odds", "bets_pct"]].rename(
                columns={"odds": "open_odds", "bets_pct": "open_bets"}
            ),
            on=keys, 
            how="left"
        )
    else:
        j = out.copy()
        j["open_odds"] = None
        j["open_bets"] = None

    def _rlm_flag(row):
        """Reverse Line Movement detection."""
        try:
            if row["open_bets"] is None or row["open_odds"] is None:
                return 0.0
            
            bet_delta = float(row["%bets"]) - float(row["open_bets"])
            cur = clean_odds(row["odds"])
            op = clean_odds(row["open_odds"])
            
            if cur is None or op is None:
                return 0.0
            
            p_cur = implied_prob_from_odds(cur)
            p_op = implied_prob_from_odds(op)
            imp_delta = (p_cur - p_op) if p_cur is not None and p_op is not None else 0.0
            
            # RLM: public bets increase, but line moves against them
            if bet_delta >= RLM_BET_THRESHOLD and imp_delta < 0:
                return min(1.0, max(0.0, (bet_delta - RLM_BET_THRESHOLD) / 15.0))
            
            return 0.0
        except Exception:
            return 0.0

    def _steam_flag(row):
        """Steam move detection."""
        try:
            if row["open_odds"] is None:
                return 0.0
            
            cur = clean_odds(row["odds"])
            op = clean_odds(row["open_odds"])
            
            if cur is None or op is None:
                return 0.0
            
            if abs(cur - op) >= STEAM_ODDS_THRESHOLD:
                return min(1.0, abs(cur - op) / 30.0)
            
            return 0.0
        except Exception:
            return 0.0

    j["rlm"] = j.apply(_rlm_flag, axis=1)
    j["steam"] = j.apply(_steam_flag, axis=1)

    # Normalize components
    irr = j["irrationality"].fillna(0.0).clip(0, 50) / 50.0
    evn = (j["ev_units"].fillna(0.0) + 0.10).clip(0, 0.35) / 0.35
    rlm = j["rlm"].fillna(0.0).clip(0, 1)
    stm = j["steam"].fillna(0.0).clip(0, 1)

    # Get weights from session state
    w_ev = float(st.session_state.get("w_ev", DEFAULT_W_EV))
    w_irr = float(st.session_state.get("w_irr", DEFAULT_W_IRR))
    w_rlm = float(st.session_state.get("w_rlm", DEFAULT_W_RLM))
    w_steam = float(st.session_state.get("w_steam", DEFAULT_W_STEAM))

    # Composite edge score
    comp = 100.0 * (w_ev * evn + w_irr * irr + w_rlm * rlm + w_steam * stm) / max(1e-9, (w_ev + w_irr + w_rlm + w_steam))
    j["edge_score"] = comp.round(1)

    return j

def recommend_bets(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and rank recommendations."""
    if df.empty:
        return df
    
    min_edge = int(st.session_state.get("edge_alert_min", 70))
    min_ev = float(st.session_state.get("ev_alert_min", 0.05))
    
    rec = df[(df["edge_score"] >= min_edge) & (df["ev_units"] >= min_ev)].copy()
    rec["stake_pct"] = (100.0 * rec["kelly_cap"]).round(2)
    
    # Sort by edge quality
    rec = rec.sort_values(["edge_score", "ev_units"], ascending=False)
    
    return rec


# ============================================================================
# VIRAL CONTENT BLOCK BUILDERS (PUBLIC WARNING, MONEY DIVERGENCE, BOOK NEEDS, RLM)
# ============================================================================

# PUBLIC BETTING WARNING IMAGE CONFIG & HELPERS (NFL)

PUBLIC_WARNING_TEMPLATE_PATH = "public_betting.png"  # template file in app working dir

PB_WIDTH, PB_HEIGHT = 1080, 1920
PB_GREEN = (0, 196, 79)
PB_OFFWHITE = (235, 232, 221)

PB_PCT_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
PB_TEXT_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

PB_PCT_FONT_SIZE = 75
PB_TEXT_FONT_SIZE = 75
PB_LINE_SPACING_MULT = 0.9
PB_BLOCK_SPACING = 50
PB_START_Y = 650
PB_LEFT_PCT_X = 120
PB_TEXT_BLOCK_X = 420
PB_LAST_ROW_EXTRA = 40

PB_MAX_TEXT_CHARS = 32  # soft cap for matchup/detail text to avoid overflow


def line_height(font: ImageFont.FreeTypeFont) -> int:
    """Approximate line height for text layout."""
    ascent, descent = font.getmetrics()
    return ascent + descent


def abbreviate_matchup(matchup: str) -> str:
    """
    Best-effort convert a raw matchup string into a compact 'AAA @ BBB' label.

    Generic (works fine for NFL):
    - Splits on ' @ ', ' vs ', ' vs.', or plain '@'
    - Falls back to first and last tokens (e.g. 'DET @ KC')
    """
    s = " ".join(str(matchup or "").upper().split())
    if not s:
        return ""

    for sep in [" @ ", " VS ", " VS.", " VS. ", " V "]:
        if sep in s:
            left, right = s.split(sep, 1)
            return f"{left.strip()[:3]} @ {right.strip()[:3]}"

    if "@" in s:
        left, right = s.split("@", 1)
        return f"{left.strip()[:3]} @ {right.strip()[:3]}"

    parts = s.split()
    if len(parts) >= 2:
        return f"{parts[0][:3]} @ {parts[-1][:3]}"

    return s[:7]


def _truncate_text(text: str, max_chars: int = PB_MAX_TEXT_CHARS) -> str:
    """Simple character-count truncation with ellipsis to keep lines readable."""
    t = str(text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def parse_public_side_for_template(
    market_src: str,
    public_side_txt: str,
    line_now: str | None = None,
) -> tuple[str, str]:
    """
    Normalize a public side like 'DET -3.5' or 'Over 44.5' into (label, line).

    Returns:
      team_label, line_label
    where team_label is usually TEAM / OVER / UNDER and line_label is the numeric edge.
    """
    s = str(public_side_txt or "").strip()
    m = (market_src or "").lower()

    # Totals: "Over 44.5" / "Under 44.5"
    if "total" in m:
        ou, tot = _parse_total_from_side(s)
        if ou and tot is not None:
            return ou.upper(), f"{tot:g}"
        return s.upper(), ""

    # Spreads: "DET -3.5" or "KC +2.5"
    if "spread" in m:
        spr = _parse_spread_from_side(s)
        team = s.split()[0].upper() if s.split() else ""
        if spr is not None:
            return (team or s.upper()), f"{spr:+g}"
        return s.upper(), ""

    # Moneyline: "DET ML -135" or similar book junk
    if "moneyline" in m or "money" in m:
        parts = s.split()
        if len(parts) >= 2 and parts[-1].startswith(("+", "-")):
            team = " ".join(parts[:-1]).upper()
            return team, parts[-1]
        return s.upper(), str(line_now or "")

    # Fallback for anything else
    return s.upper(), str(line_now or "")


def build_public_warning_rows_from_tables(
    spread_df: pd.DataFrame | None,
    total_df: pd.DataFrame | None,
    ml_df: pd.DataFrame | None,
    max_rows: int = 4,
) -> list[dict]:
    """
    Convert spread/total/moneyline public warning tables into generic row dicts
    for the image renderer.

    Each input DataFrame is expected to have at least:
      - 'matchup'
      - '%Bets' or '%bets' or '% Bets'
      - 'side' or 'Public Side'
      - 'Line' or 'Line Now'
    """
    records: list[dict] = []

    def _add(df: pd.DataFrame | None, market_src: str):
        if df is None or df.empty:
            return
        for _, r in df.iterrows():
            rec = dict(r)
            rec["__market_src"] = market_src
            records.append(rec)

    _add(spread_df, "spread")
    _add(total_df, "total")
    _add(ml_df, "moneyline")

    if not records:
        return []

    combined = pd.DataFrame(records)

    # Pick a % column in priority order
    pct_col = None
    for cand in ["%Bets", "%bets", "% Bets", "Public %", "Public % Bets"]:
        if cand in combined.columns:
            pct_col = cand
            break

    if pct_col is None:
        combined["__pct"] = 0.0
    else:
        combined["__pct"] = pd.to_numeric(combined[pct_col], errors="coerce").fillna(0.0)

    combined = combined.sort_values("__pct", ascending=False).head(max_rows)
    if combined.empty:
        return []

    # Side / line columns can come from either the warning builder or other helpers
    side_col = "Public Side" if "Public Side" in combined.columns else "side"
    if side_col not in combined.columns:
        side_col = "side"

    line_col = "Line Now" if "Line Now" in combined.columns else "Line"
    if line_col not in combined.columns:
        line_col = "Line"

    rows_out: list[dict] = []
    for _, r in combined.iterrows():
        pct = float(r.get("__pct", 0.0))
        matchup = r.get("matchup", "")
        market_src = r.get("__market_src", "")
        side_txt = r.get(side_col, "")
        line_now = r.get(line_col, "")

        team, line = parse_public_side_for_template(
            market_src,
            side_txt,
            line_now=str(line_now or ""),
        )

        rows_out.append(
            {
                "pct": pct,
                "team": team,
                "line": line,
                "matchup": matchup,
                "market_src": market_src,
            }
        )

    return rows_out


def render_public_betting_template(rows: list[dict]) -> Image.Image:
    """
    Draw rows of public-betting info onto the FoxEdge template for NFL.

    Each row dict should have:
      - 'pct' (float)
      - 'team' (str)
      - 'line' (str)
      - optional 'matchup' (full matchup string) and 'market_src' ("spread"/"total"/"moneyline")
    """
    base = Image.open(PUBLIC_WARNING_TEMPLATE_PATH).convert("RGBA").resize((PB_WIDTH, PB_HEIGHT))
    draw = ImageDraw.Draw(base)

    pct_font = ImageFont.truetype(PB_PCT_FONT_PATH, PB_PCT_FONT_SIZE)
    text_font = ImageFont.truetype(PB_TEXT_FONT_PATH, PB_TEXT_FONT_SIZE)

    lh = line_height(text_font)
    max_lines_per_row = 3  # "OF BETS ON", matchup, detail line

    for idx, row in enumerate(rows):
        # Vertical placement per row
        y = PB_START_Y + idx * (lh * PB_LINE_SPACING_MULT * max_lines_per_row + PB_BLOCK_SPACING)
        if idx == len(rows) - 1:
            y += PB_LAST_ROW_EXTRA

        # Left-side % label
        pct_val = float(row.get("pct", 0.0))
        pct_text = f"{int(round(pct_val))}%"

        matchup_raw = str(row.get("matchup", "")).upper()
        matchup_abbr = abbreviate_matchup(matchup_raw) if matchup_raw else ""

        team_label = str(row.get("team", "")).upper()
        line_label = str(row.get("line", "")).strip()
        market_src = str(row.get("market_src", "")).lower()

        line1 = "OF BETS ON"
        if "total" in market_src and matchup_abbr:
            line2 = matchup_abbr
            detail_core = f"{team_label} {line_label}".strip()
        elif "spread" in market_src:
            line2 = matchup_abbr or team_label
            detail_core = f"{team_label} {line_label}".strip()
        else:
            line2 = matchup_abbr or team_label
            detail_core = line_label or team_label

        line2 = _truncate_text(line2, PB_MAX_TEXT_CHARS)
        line3 = _truncate_text(detail_core, PB_MAX_TEXT_CHARS) if detail_core else ""

        # Draw text: % on left, then 1–3 stacked lines in the text block
        draw.text((PB_LEFT_PCT_X, y), pct_text, font=pct_font, fill=PB_GREEN)
        draw.text((PB_TEXT_BLOCK_X, y), line1, font=text_font, fill=PB_OFFWHITE)

        if line2:
            y2 = y + lh * PB_LINE_SPACING_MULT
            draw.text((PB_TEXT_BLOCK_X, y2), line2, font=text_font, fill=PB_OFFWHITE)

        if line3:
            y3 = y + 2 * lh * PB_LINE_SPACING_MULT
            draw.text((PB_TEXT_BLOCK_X, y3), line3, font=text_font, fill=PB_OFFWHITE)

    return base

def build_public_betting_warning(df: pd.DataFrame, pct_threshold: float = 70.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 'PUBLIC BETTING WARNING' blocks split by Spread vs Total.

    Rules:
    - Identify sides where %bets >= pct_threshold.
    - Only consider Spread and Total (no Moneyline).
    - Build two tables:
        1. Top 5 most publicly bet spreads
        2. Top 5 most publicly bet totals

    Output columns per table:
      matchup, market, side, %Bets, Line
    """
    cols = ["matchup","market","side","%Bets","Line"]
    if df.empty:
        return (
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
        )

    spread_rows = []
    total_rows = []

    for _, r in df.iterrows():
        bets = r.get("%bets")
        if bets is None:
            continue
        try:
            bets_f = float(bets)
        except Exception:
            continue
        if bets_f < pct_threshold:
            continue

        mkt = str(r.get("market",""))
        mkt_lower = mkt.lower()
        # we only keep spread / total here
        if not (mkt_lower.startswith("spread") or mkt_lower.startswith("total")):
            continue

        side = str(r.get("side",""))
        matchup = str(r.get("matchup",""))

        # Build Line info
        line_info = None
        if mkt_lower.startswith("spread"):
            spr = _parse_spread_from_side(side)
            line_info = f"{spr:+}" if spr is not None else None
        elif mkt_lower.startswith("total"):
            ou, tot = _parse_total_from_side(side)
            if ou and tot:
                line_info = f"{ou} {tot}"
            elif tot:
                line_info = f"Total {tot}"

        row_out = {
            "matchup": matchup,
            "market": mkt,
            "side": side,
            "%Bets": round(bets_f,1),
            "Line": line_info,
        }

        if mkt_lower.startswith("spread"):
            spread_rows.append(row_out)
        elif mkt_lower.startswith("total"):
            total_rows.append(row_out)

    def _mk(rows_list: list[dict]) -> pd.DataFrame:
        if not rows_list:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame(rows_list)
        # Sort highest %Bets first
        out = out.sort_values("%Bets", ascending=False)
        # Drop duplicates caused by multi-page scrape
        out = out.drop_duplicates(subset=["matchup", "market", "side", "Line"], keep="first")
        out = out.reset_index(drop=True)
        return out.head(5)

    spread_df = _mk(spread_rows)
    total_df = _mk(total_rows)

    return spread_df, total_df


def build_money_divergence_report(df: pd.DataFrame, sharp_gap: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 'MONEY DIVERGENCE REPORT' split into Spread vs Total.

    We compute handle-vs-bets delta for each row. Positive big delta = sharp money.
    Negative big delta = public money.

    We return two DataFrames:
      - spread_top5: top 5 spread markets by |delta|
      - total_top5:  top 5 total markets by |delta|

    Each DataFrame has columns:
      matchup, side, Bets %, Handle %, Δ Handle-Bets, Signal
    """
    cols = ["matchup","market","side","Bets %","Handle %","Δ Handle-Bets","Signal"]
    if df.empty:
        return (
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
        )

    recs_spread = []
    recs_total = []

    for _, r in df.iterrows():
        mkt = str(r.get("market", ""))
        mkt_lower = mkt.lower()
        # only spreads and totals
        if not (mkt_lower.startswith("spread") or mkt_lower.startswith("total")):
            continue

        bets = r.get("%bets")
        handle = r.get("%handle")
        if bets is None or handle is None:
            continue
        try:
            bets_f = float(bets)
            handle_f = float(handle)
        except Exception:
            continue

        delta = handle_f - bets_f
        signal = "EVEN"
        if delta >= sharp_gap:
            signal = "🔥 SHARP MONEY"
        elif delta <= -sharp_gap:
            signal = "💀 PUBLIC MONEY"

        row_out = {
            "matchup": str(r.get("matchup", "")),
            "market": mkt,
            "side": str(r.get("side", "")),
            "Bets %": round(bets_f, 1),
            "Handle %": round(handle_f, 1),
            "Δ Handle-Bets": round(delta, 1),
            "Signal": signal,
        }

        if mkt_lower.startswith("spread"):
            recs_spread.append(row_out)
        elif mkt_lower.startswith("total"):
            recs_total.append(row_out)

    def _mk(df_rows: list[dict]) -> pd.DataFrame:
        if not df_rows:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame(df_rows)
        # drop duplicate market entries (multi-page scrape can echo same row)
        out = out.drop_duplicates(
            subset=["matchup", "market", "side", "Bets %", "Handle %"],
            keep="first"
        )
        # strongest disagreement first (absolute delta desc)
        out = out.sort_values(
            "Δ Handle-Bets",
            key=lambda s: s.abs(),
            ascending=False
        ).reset_index(drop=True)
        return out.head(5)

    spread_top5 = _mk(recs_spread)
    total_top5 = _mk(recs_total)

    return spread_top5, total_top5


def build_book_needs_board(df: pd.DataFrame, public_threshold: float = 65.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 'BOOK NEEDS' boards split by market type (Spread / Total / Moneyline).

    We infer book liability by assuming:
    - If %bets on Side A > threshold, public is heavy on Side A.
    - The book "needs" the other side of that exact market.

    We generate three tables:
      1. Spread liability (top 5 by % public)
      2. Total liability (top 5 by % public)
      3. Moneyline liability (top 5 by % public)

    Each table has columns:
      Matchup, Public Side, Public %, Book Needs, Line Now

    For spread:
      If public is on "DET -3.5", book needs the opponent at the flipped spread (e.g. "LAC +3.5").
    For total:
      If public is on "Over 44.5", book needs "Under 44.5".
    For moneyline:
      If public is on "DET ML", book needs the opponent "LAC ML".
    """

    cols = ["Matchup","Public Side","Public %","Book Needs","Line Now"]
    if df.empty:
        return (
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
        )

    def _split_matchup(matchup_raw: str) -> tuple[str, str]:
        """Return (away, home) best-effort from matchup string."""
        matchup_raw = str(matchup_raw)
        if " @ " in matchup_raw:
            away, home = matchup_raw.split(" @ ", 1)
            return away.strip(), home.strip()
        if " vs " in matchup_raw:
            left, right = matchup_raw.split(" vs ", 1)
            # "X vs Y" is ambiguous for home/away, but we just call them (left,right)
            return left.strip(), right.strip()
        parts = matchup_raw.split()
        if len(parts) >= 2:
            return parts[0].strip(), parts[-1].strip()
        return (matchup_raw.strip(), "Opponent")

    spread_rows = []
    total_rows = []
    money_rows = []

    for _, r in df.iterrows():
        mkt = str(r.get("market", ""))
        mkt_lower = mkt.lower()
        bets = r.get("%bets")
        if bets is None:
            continue
        try:
            bets_f = float(bets)
        except Exception:
            continue
        if bets_f < public_threshold:
            continue

        matchup = str(r.get("matchup", ""))
        side_txt = str(r.get("side", ""))
        odds_now = r.get("odds")
        spread_val = r.get("spread")

        away_team, home_team = _split_matchup(matchup)

        # We'll build the opposite side label per market type.
        book_needs = ""
        line_now = None

        if mkt_lower.startswith("spread"):
            # side_txt usually like "DET -3.5" or "LAC +3.5"
            spr_val = _parse_spread_from_side(side_txt)
            # figure which team this side is (first token in side_txt usually the team)
            team_public = side_txt.split()[0] if side_txt.split() else ""
            # pick opponent
            opponent = home_team if team_public and team_public in away_team else away_team
            # flip spread sign for opponent (DET -3.5 => opp +3.5)
            if spr_val is not None:
                flipped = -spr_val
                book_needs = f"{opponent} {flipped:+}"
                line_now = f"{spr_val:+}"
            else:
                # fallback if we couldn't parse number
                book_needs = f"{opponent} (opp side)"
                line_now = None

            spread_rows.append({
                "Matchup": matchup,
                "Public Side": side_txt,
                "Public %": round(bets_f,1),
                "Book Needs": book_needs,
                "Line Now": line_now,
            })

        elif mkt_lower.startswith("total"):
            # side_txt usually like "Over 44.5" or "Under 44.5"
            ou, tot = _parse_total_from_side(side_txt)
            if ou and tot:
                if ou.lower().startswith("over"):
                    opp_ou = "Under"
                else:
                    opp_ou = "Over"
                book_needs = f"{opp_ou} {tot}"
                line_now = f"{ou} {tot}"
            else:
                book_needs = "Opposite total"
                line_now = None

            total_rows.append({
                "Matchup": matchup,
                "Public Side": side_txt,
                "Public %": round(bets_f,1),
                "Book Needs": book_needs,
                "Line Now": line_now,
            })

        elif mkt_lower.startswith("money"):
            # moneyline: side_txt is usually team name / moneyline
            team_public = side_txt.split()[0] if side_txt.split() else ""
            # infer opponent (if public team matches away, opponent is home; else opponent is away)
            if team_public and team_public in away_team:
                opponent = home_team
            else:
                opponent = away_team
            # Book needs opponent ML
            book_needs = f"{opponent} ML"
            line_now = f"{odds_now}" if odds_now is not None else None

            money_rows.append({
                "Matchup": matchup,
                "Public Side": side_txt,
                "Public %": round(bets_f,1),
                "Book Needs": book_needs,
                "Line Now": line_now,
            })

    def _mk(rows_out: list[dict]) -> pd.DataFrame:
        if not rows_out:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame(rows_out)
        # drop duplicate liability rows caused by multi-page scrape echoes
        out = out.drop_duplicates(
            subset=["Matchup", "Public Side", "Book Needs", "Line Now"],
            keep="first"
        )
        out = out.sort_values("Public %", ascending=False).reset_index(drop=True)
        return out.head(5)

    spread_df = _mk(spread_rows)
    total_df = _mk(total_rows)
    money_df = _mk(money_rows)

    return spread_df, total_df, money_df

# ============================================================================
# BOOK PAIN BAR ("Vegas does NOT want this") IMPLEMENTATION
# ============================================================================

def _payout_multiplier(american_odds: int | float | None) -> float | None:
    """
    Convert American odds to approximate profit-per-$1-risked.
    -210  => ~0.476  (risk 210 to win 100 -> 100/210 per $1)
    +175  => 1.75    (risk 100 to win 175 -> 175/100 per $1)
    If odds is None or can't parse, return None.
    """
    if american_odds is None:
        return None
    try:
        o = int(american_odds)
    except Exception:
        return None
    if o < 0:
        return 100.0 / abs(o)
    if o > 0:
        return o / 100.0
    return None  # odds==0 shouldn't really exist

def _pick_public_share(row: pd.Series) -> float | None:
    """
    Return best available 'public share' for a side.
    Prefer %handle. Fallback to %bets.
    Output is a decimal fraction (0.74 for 74%).
    """
    h = row.get("%handle")
    b = row.get("%bets")
    try:
        if h is not None and not pd.isna(h):
            return float(h) / 100.0
        if b is not None and not pd.isna(b):
            return float(b) / 100.0
    except Exception:
        pass
    return None

def _extract_matchup_book_pain_payload(df_match: pd.DataFrame) -> dict | None:
    """
    Build the structured input for Book Pain Bar for ONE matchup.
    We expect rows like:
      market == "Moneyline"  (two sides)
      market == "Spread"     (two sides)

    For Moneyline:
      - favorite = odds < 0
      - underdog = odds > 0

    For Spread:
      - favorite spread = negative spread number
      - underdog spread = positive spread number

    Returns dict with everything needed for scoring, or None if we can't get it.
    """
    if df_match.empty:
        return None

    game_name = str(df_match["matchup"].iloc[0])

    # MONEYLINE
    ml_rows = df_match[df_match["market"].str.lower() == "moneyline"].copy()
    fav_ml_team = None
    fav_ml_odds = None
    fav_ml_share = None
    dog_ml_team = None
    dog_ml_odds = None
    dog_ml_share = None

    for _, r in ml_rows.iterrows():
        team_name = str(r.get("side", ""))
        odds_val = clean_odds(r.get("odds"))
        share_val = _pick_public_share(r)
        if odds_val is None or share_val is None:
            continue
        if odds_val < 0:
            fav_ml_team = team_name
            fav_ml_odds = odds_val
            fav_ml_share = share_val
        else:
            dog_ml_team = team_name
            dog_ml_odds = odds_val
            dog_ml_share = share_val

    # SPREAD
    sp_rows = df_match[df_match["market"].str.lower() == "spread"].copy()
    fav_spread_team = None
    fav_spread_line = None
    fav_spread_price = None
    fav_spread_share = None
    dog_spread_team = None
    dog_spread_line = None
    dog_spread_price = None
    dog_spread_share = None

    for _, r in sp_rows.iterrows():
        team_name = str(r.get("side", ""))
        spread_val = _parse_spread_from_side(team_name)
        price_odds = clean_odds(r.get("odds"))
        share_val = _pick_public_share(r)
        if spread_val is None or share_val is None:
            continue
        # negative spread => favorite
        if spread_val < 0:
            fav_spread_team = team_name
            fav_spread_line = spread_val
            fav_spread_price = price_odds
            fav_spread_share = share_val
        else:
            dog_spread_team = team_name
            dog_spread_line = spread_val
            dog_spread_price = price_odds
            dog_spread_share = share_val

    # Require all four sides to exist so we can score fairly
    if any(x is None for x in [
        fav_ml_team, fav_ml_odds, fav_ml_share,
        dog_ml_team, dog_ml_odds, dog_ml_share,
        fav_spread_team, fav_spread_line, fav_spread_price, fav_spread_share,
        dog_spread_team, dog_spread_line, dog_spread_price, dog_spread_share,
    ]):
        return None

    return {
        "game": game_name,

        "favorite_team": fav_ml_team,
        "favorite_ml_odds": fav_ml_odds,
        "favorite_ml_public_share": fav_ml_share,
        "favorite_spread": fav_spread_line,
        "favorite_spread_price": fav_spread_price,
        "favorite_spread_public_share": fav_spread_share,

        "underdog_team": dog_ml_team,
        "underdog_ml_odds": dog_ml_odds,
        "underdog_ml_public_share": dog_ml_share,
        "underdog_spread": dog_spread_line,
        "underdog_spread_price": dog_spread_price,
        "underdog_spread_public_share": dog_spread_share,
    }

def _compute_book_pain_scores(payload: dict) -> dict | None:
    """
    Take the payload and compute ExposureScore for:
      - favorite ML
      - underdog ML
      - favorite spread
      - underdog spread

    ExposureScore_side = PublicShare_side * PayoutMultiplier_side

    Then pick the highest ExposureScore. That's the 'book pain' side.
    """
    if payload is None:
        return None

    # payout multipliers for ML
    fav_ml_mult = _payout_multiplier(payload.get("favorite_ml_odds"))
    dog_ml_mult = _payout_multiplier(payload.get("underdog_ml_odds"))

    # spreads: use actual juice if we have it; fallback ~0.91 if missing
    fav_sp_mult = _payout_multiplier(payload.get("favorite_spread_price"))
    if fav_sp_mult is None:
        fav_sp_mult = 0.91
    dog_sp_mult = _payout_multiplier(payload.get("underdog_spread_price"))
    if dog_sp_mult is None:
        dog_sp_mult = 0.91

    try:
        ExposureScore_favML = float(payload["favorite_ml_public_share"]) * float(fav_ml_mult)
        ExposureScore_dogML = float(payload["underdog_ml_public_share"]) * float(dog_ml_mult)
        ExposureScore_favSpread = float(payload["favorite_spread_public_share"]) * float(fav_sp_mult)
        ExposureScore_dogSpread = float(payload["underdog_spread_public_share"]) * float(dog_sp_mult)
    except Exception:
        return None

    # pick the max
    scores = [
        (ExposureScore_favML,     "ml",     "fav"),
        (ExposureScore_dogML,     "ml",     "dog"),
        (ExposureScore_favSpread, "spread", "fav"),
        (ExposureScore_dogSpread, "spread", "dog"),
    ]
    scores = [s for s in scores if s[0] is not None]
    if not scores:
        return None
    scores.sort(key=lambda x: x[0], reverse=True)
    top_score, top_type, top_side = scores[0]

    # build social labels + CTA opposite
    if top_type == "ml":
        if top_side == "fav":
            pain_label = f"{payload['favorite_team']} ML"
            pain_odds = payload["favorite_ml_odds"]
            pain_share = payload["favorite_ml_public_share"]
            house_needs = f"{payload['underdog_team']} ML"
        else:
            pain_label = f"{payload['underdog_team']} ML"
            pain_odds = payload["underdog_ml_odds"]
            pain_share = payload["underdog_ml_public_share"]
            house_needs = f"{payload['favorite_team']} ML"
    else:  # spread
        if top_side == "fav":
            line_val = payload["favorite_spread"]
            pain_label = f"{payload['favorite_team']} {line_val:+g}"
            pain_odds = payload["favorite_spread_price"]
            pain_share = payload["favorite_spread_public_share"]
            opp_line = payload["underdog_spread"]
            house_needs = f"{payload['underdog_team']} {opp_line:+g}"
        else:
            line_val = payload["underdog_spread"]
            pain_label = f"{payload['underdog_team']} {line_val:+g}"
            pain_odds = payload["underdog_spread_price"]
            pain_share = payload["underdog_spread_public_share"]
            opp_line = payload["favorite_spread"]
            house_needs = f"{payload['favorite_team']} {opp_line:+g}"

    enriched = dict(payload)
    enriched.update({
        "ExposureScore_favML": ExposureScore_favML,
        "ExposureScore_dogML": ExposureScore_dogML,
        "ExposureScore_favSpread": ExposureScore_favSpread,
        "ExposureScore_dogSpread": ExposureScore_dogSpread,
        "book_pain_market": pain_label,      # e.g. "Lions -4.5"
        "book_pain_type": top_type,          # "ml" or "spread"
        "book_pain_score": top_score,        # max exposure
        "book_pain_odds": pain_odds,         # -110 or +175 etc
        "book_pain_share": pain_share,       # decimal fraction
        "house_needs": house_needs,          # "Vikings +4.5"
    })
    return enriched

def compute_book_pain_all_games(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    For the full DK splits dataframe:
    - build payload per matchup,
    - score exposure,
    - choose the single highest 'book pain' side across the slate.

    Returns:
      summary_df: 1-row DataFrame with the worst pain side overall
      logs: list of dict snapshots (one per matchup) for persistence.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "game","book_pain_market","book_pain_score","book_pain_odds",
            "book_pain_share","house_needs","book_pain_type"
        ]), []

    out_rows = []
    logs = []

    for matchup_val, sub in df.groupby("matchup"):
        payload = _extract_matchup_book_pain_payload(sub)
        enriched = _compute_book_pain_scores(payload) if payload else None
        if not enriched:
            continue

        logs.append({
            "timestamp_utc": datetime.utcnow().isoformat(),
            "game": enriched.get("game"),

            "favorite_team": enriched.get("favorite_team"),
            "favorite_ml_odds": enriched.get("favorite_ml_odds"),
            "favorite_ml_public_share": enriched.get("favorite_ml_public_share"),
            "favorite_spread": enriched.get("favorite_spread"),
            "favorite_spread_price": enriched.get("favorite_spread_price"),
            "favorite_spread_public_share": enriched.get("favorite_spread_public_share"),

            "underdog_team": enriched.get("underdog_team"),
            "underdog_ml_odds": enriched.get("underdog_ml_odds"),
            "underdog_ml_public_share": enriched.get("underdog_ml_public_share"),
            "underdog_spread": enriched.get("underdog_spread"),
            "underdog_spread_price": enriched.get("underdog_spread_price"),
            "underdog_spread_public_share": enriched.get("underdog_spread_public_share"),

            "ExposureScore_favML": enriched.get("ExposureScore_favML"),
            "ExposureScore_dogML": enriched.get("ExposureScore_dogML"),
            "ExposureScore_favSpread": enriched.get("ExposureScore_favSpread"),
            "ExposureScore_dogSpread": enriched.get("ExposureScore_dogSpread"),

            "book_pain_market": enriched.get("book_pain_market"),
            "book_pain_type": enriched.get("book_pain_type"),
            "book_pain_score": enriched.get("book_pain_score"),
            "book_pain_odds": enriched.get("book_pain_odds"),
            "book_pain_share": enriched.get("book_pain_share"),
            "house_needs": enriched.get("house_needs"),
        })

        out_rows.append({
            "game": enriched.get("game"),
            "book_pain_market": enriched.get("book_pain_market"),
            "book_pain_score": enriched.get("book_pain_score"),
            "book_pain_odds": enriched.get("book_pain_odds"),
            "book_pain_share": enriched.get("book_pain_share"),
            "house_needs": enriched.get("house_needs"),
            "book_pain_type": enriched.get("book_pain_type"),
        })

    if not out_rows:
        return pd.DataFrame(columns=[
            "game","book_pain_market","book_pain_score","book_pain_odds",
            "book_pain_share","house_needs","book_pain_type"
        ]), logs

    df_out = pd.DataFrame(out_rows).sort_values("book_pain_score", ascending=False).reset_index(drop=True)
    top_row = df_out.head(1)
    return top_row, logs

def log_book_pain_snapshots(log_rows: list[dict], db_path: str = BETLOGS_DB):
    """
    Persist Book Pain snapshots so we can do postgame accountability content.
    Table: book_pain_logs
    """
    if not log_rows:
        return
    _ensure_parent(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS book_pain_logs (
                game TEXT,
                timestamp_utc TEXT,

                favorite_team TEXT,
                favorite_ml_odds INTEGER,
                favorite_ml_public_share REAL,
                favorite_spread REAL,
                favorite_spread_price INTEGER,
                favorite_spread_public_share REAL,

                underdog_team TEXT,
                underdog_ml_odds INTEGER,
                underdog_ml_public_share REAL,
                underdog_spread REAL,
                underdog_spread_price INTEGER,
                underdog_spread_public_share REAL,

                ExposureScore_favML REAL,
                ExposureScore_dogML REAL,
                ExposureScore_favSpread REAL,
                ExposureScore_dogSpread REAL,

                book_pain_market TEXT,
                book_pain_type TEXT,
                book_pain_score REAL,
                book_pain_odds INTEGER,
                book_pain_share REAL,
                house_needs TEXT,

                PRIMARY KEY (game, timestamp_utc)
            );
        """)

        rows_to_write = []
        for r in log_rows:
            rows_to_write.append((
                r.get("game"),
                r.get("timestamp_utc"),

                r.get("favorite_team"),
                r.get("favorite_ml_odds"),
                r.get("favorite_ml_public_share"),
                r.get("favorite_spread"),
                r.get("favorite_spread_price"),
                r.get("favorite_spread_public_share"),

                r.get("underdog_team"),
                r.get("underdog_ml_odds"),
                r.get("underdog_ml_public_share"),
                r.get("underdog_spread"),
                r.get("underdog_spread_price"),
                r.get("underdog_spread_public_share"),

                r.get("ExposureScore_favML"),
                r.get("ExposureScore_dogML"),
                r.get("ExposureScore_favSpread"),
                r.get("ExposureScore_dogSpread"),

                r.get("book_pain_market"),
                r.get("book_pain_type"),
                r.get("book_pain_score"),
                r.get("book_pain_odds"),
                r.get("book_pain_share"),
                r.get("house_needs"),
            ))

        con.executemany("""
            INSERT OR REPLACE INTO book_pain_logs (
                game, timestamp_utc,

                favorite_team, favorite_ml_odds, favorite_ml_public_share,
                favorite_spread, favorite_spread_price, favorite_spread_public_share,

                underdog_team, underdog_ml_odds, underdog_ml_public_share,
                underdog_spread, underdog_spread_price, underdog_spread_public_share,

                ExposureScore_favML, ExposureScore_dogML,
                ExposureScore_favSpread, ExposureScore_dogSpread,

                book_pain_market, book_pain_type, book_pain_score,
                book_pain_odds, book_pain_share, house_needs
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows_to_write)

def render_book_pain_bar_card(pain_row: pd.Series) -> Image.Image:
    """
    Build the IG-ready 'BOOK PAIN BAR' graphic.
    1080x1350 black background.

    Top: "🔻 BOOK PAIN BAR 🔻"
         "If THIS cashes, the book bleeds."
    Center: big red bar with the pain side ("LIONS -4.5")
    Under it: 2 context lines:
        "61% of public on LIONS -4.5"
        "-110 payout multiplies across the room if it hits"
    Bottom strip: "The house is praying for VIKINGS +4.5"
    """
    W, H = 1080, 1350
    img = Image.new("RGB", (W, H), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # fonts
    try:
        font_header = ImageFont.truetype("Arial.ttf", 64)
        font_sub = ImageFont.truetype("Arial.ttf", 36)
        font_bar = ImageFont.truetype("Arial.ttf", 80)
        font_context = ImageFont.truetype("Arial.ttf", 40)
        font_footer = ImageFont.truetype("Arial.ttf", 44)
    except Exception:
        font_header = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_bar = ImageFont.load_default()
        font_context = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    def _text_center(y, text, font, fill=(255,255,255)):
        bbox = draw.textbbox((0,0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((W - tw)/2, y), text, fill=fill, font=font)
        return th

    y = 60
    y += _text_center(y, "🔻 BOOK PAIN BAR 🔻", font_header, fill=(255,255,255)) + 10
    y += _text_center(y, "If THIS cashes, the book bleeds.", font_sub, fill=(200,200,200)) + 60

    # red bar
    bar_h = 220
    bar_x0 = 80
    bar_x1 = W - 80
    bar_y0 = y
    bar_y1 = y + bar_h
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(200,0,0))

    pain_label = str(pain_row.get("book_pain_market", "")).upper()
    bbox_bar = draw.textbbox((0,0), pain_label, font=font_bar)
    tw_bar = bbox_bar[2] - bbox_bar[0]
    th_bar = bbox_bar[3] - bbox_bar[1]
    draw.text(
        ((W - tw_bar)/2, bar_y0 + (bar_h - th_bar)/2),
        pain_label,
        fill=(255,255,255),
        font=font_bar
    )

    y = bar_y1 + 40

    # context lines
    try:
        pub_share_pct = round(float(pain_row.get("book_pain_share", 0.0))*100.0, 1)
    except Exception:
        pub_share_pct = None

    odds_txt = pain_row.get("book_pain_odds")
    try:
        odds_txt = f"{int(odds_txt):+d}" if odds_txt is not None else ""
    except Exception:
        odds_txt = str(odds_txt)

    line1 = f"{pub_share_pct}% of public on {pain_label}" if pub_share_pct is not None else f"Public loading {pain_label}"
    line2 = f"{odds_txt} payout multiplies across the room if it hits"

    y += _text_center(y, line1, font_context, fill=(150,255,150)) + 10
    y += _text_center(y, line2, font_context, fill=(150,255,150)) + 80

    # footer strip
    bottom_txt = f"The house is praying for {pain_row.get('house_needs','the other side')}".upper()
    bbox_footer = draw.textbbox((0,0), bottom_txt, font=font_footer)
    ftw = bbox_footer[2] - bbox_footer[0]
    fth = bbox_footer[3] - bbox_footer[1]

    footer_h = fth + 40
    foot_y0 = H - footer_h - 60
    foot_y1 = foot_y0 + footer_h
    draw.rectangle([0, foot_y0, W, foot_y1], fill=(20,20,20))

    draw.text(
        ((W - ftw)/2, foot_y0 + (footer_h - fth)/2),
        bottom_txt,
        fill=(255,255,255),
        font=font_footer
    )

    return img

def export_book_pain_card(df_splits: pd.DataFrame) -> dict:
    """
    High-level wrapper for Streamlit:
      1. compute_book_pain_all_games(df_splits)
      2. log snapshots to sqlite
      3. render Book Pain Bar card as PIL
      4. return { "img": PIL.Image, "png_bytes": bytes, "top_row": {...} }

    On failure, return { "error": "..." }.
    """
    summary_df, logs = compute_book_pain_all_games(df_splits)
    if summary_df is None or summary_df.empty:
        return {"error": "No valid Book Pain data"}

    # log for receipts / postgame accountability
    try:
        log_book_pain_snapshots(logs)
    except Exception as e:
        logging.error(f"[export_book_pain_card] log_book_pain_snapshots error: {e}")

    top_row = summary_df.iloc[0]
    card_img = render_book_pain_bar_card(top_row)

    buf = io.BytesIO()
    card_img.save(buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.getvalue()

    return {
        "img": card_img,
        "png_bytes": png_bytes,
        "top_row": top_row.to_dict()
    }

import json  # make sure this is imported once at top of file

def _payout_multiplier_from_american(odds_val) -> float | None:
    """
    Convert an American odds price (e.g. -110, +175) into 'profit per $1 risked'.
    -210 => risk 210 to win 100 -> ~0.476
    +175 => risk 100 to win 175 -> 1.75
    If invalid, return None.
    """
    if odds_val is None:
        return None
    try:
        o = float(odds_val)
    except Exception:
        return None
    if o < 0:
        # favorite-style price
        return 100.0 / abs(o)
    else:
        # dog-style price
        return o / 100.0


def _public_pct_from_row(r: pd.Series) -> float | None:
    """
    Best available 0-100 public % from a DK row.
    Prefer %handle, fallback to %bets.
    Returns float like 61.0 or None.
    """
    cand = r.get("%handle")
    if cand is None or pd.isna(cand):
        cand = r.get("%bets")
    if cand is None or pd.isna(cand):
        return None
    try:
        return float(cand)
    except Exception:
        return None


def build_book_pain_json(df: pd.DataFrame) -> list[dict]:
    """
    Build BOOK PAIN BAR JSON objects per matchup.

    Output per matchup:
    {
      "template": "book_pain_bar_v1",
      "title": "BOOK PAIN BAR — IF THIS HITS, THE HOUSE HURTS",

      "game": "Vikings @ Lions",
      "kickoff_et": "2025-11-02T20:20:00",

      "pain_market_type": "spread",
      "pain_team": "Lions",
      "pain_line": "-4.5",
      "pain_price": -110,
      "pain_public_share_pct": 61,
      "pain_score": 0.555,

      "book_is_praying_for": "Vikings +4.5 cover",

      "narrator_line_top": "Most dangerous public outcome: LIONS -4.5.",
      "narrator_line_bottom": "61% of bettors are on Detroit -4.5 at -110. If they cover, everybody gets paid. The book does not want this."
    }

    Rules:
    - We consider 4 candidate bombs:
        * Favorite ML
        * Underdog ML
        * Favorite spread
        * Underdog spread
    - We compute ExposureScore = PublicShare * PayoutMultiplier
      and pick the highest.
    - We also build the "house is praying for" opposite outcome.
    """
    if df is None or df.empty:
        return []

    payloads: list[dict] = []

    # Work matchup by matchup
    for matchup, g in df.groupby("matchup"):
        # -------------------------
        # 1. MONEYLINE SIDES
        # -------------------------
        money = g[g["market"].str.lower().str.startswith("money")].copy()
        ml_fav = None
        ml_dog = None
        if not money.empty:
            ml_candidates = []
            for _, r in money.iterrows():
                odds_val = clean_odds(r.get("odds"))
                team_label = _extract_team_from_side_label(str(r.get("side", "")))
                pub_pct = _public_pct_from_row(r)  # 0-100
                if odds_val is None or pub_pct is None:
                    continue
                try:
                    o_f = float(odds_val)
                except Exception:
                    continue
                ml_candidates.append({
                    "team": team_label,
                    "odds": odds_val,
                    "public_pct": pub_pct,
                    "odds_f": o_f,
                })
            if ml_candidates:
                # favorite ML = most negative odds
                ml_sorted = sorted(ml_candidates, key=lambda x: x["odds_f"])
                ml_fav = ml_sorted[0]
                ml_dog = ml_sorted[-1]

        # -------------------------
        # 2. SPREAD SIDES
        # -------------------------
        spreads = g[g["market"].str.lower().str.startswith("spread")].copy()
        sp_fav = None
        sp_dog = None
        sp_candidates = []
        if not spreads.empty:
            for _, r in spreads.iterrows():
                side_txt = str(r.get("side", ""))
                spread_num = _parse_spread_from_side(side_txt)  # e.g. -4.5 / +4.5
                pub_pct = _public_pct_from_row(r)                # 0-100
                price_val = clean_odds(r.get("odds"))            # e.g. -110
                if spread_num is None or pub_pct is None:
                    continue
                # crude team extraction: first token before the number is usually team abbrev/name
                team_token = side_txt.split()[0] if side_txt else ""
                sp_candidates.append({
                    "team": team_token,
                    "spread_num": spread_num,
                    "public_pct": pub_pct,
                    "price": price_val,
                })

            if sp_candidates:
                # favorite spread = laying points (negative spread_num)
                negs = [c for c in sp_candidates if c["spread_num"] < 0]
                poss = [c for c in sp_candidates if c["spread_num"] > 0]

                # pick the "main" favorite spread: closest-to-zero negative
                if negs:
                    sp_fav = sorted(negs, key=lambda x: x["spread_num"], reverse=True)[0]
                # pick the "main" dog spread: biggest positive
                if poss:
                    sp_dog = sorted(poss, key=lambda x: x["spread_num"], reverse=True)[0]

                # fallback: if weird scrape where both are negative/positive, use extremes
                if sp_fav is None:
                    sp_fav = sorted(sp_candidates, key=lambda x: x["spread_num"])[0]
                if sp_dog is None:
                    sp_dog = sorted(sp_candidates, key=lambda x: x["spread_num"])[-1]

        # -------------------------
        # 3. EXPOSURE SCORES
        # -------------------------

        def _ml_exposure(side):
            if side is None:
                return None
            mult = _payout_multiplier_from_american(side["odds"])
            if mult is None:
                return None
            try:
                pub_frac = float(side["public_pct"]) / 100.0
            except Exception:
                return None
            return pub_frac * mult

        def _spread_exposure(side):
            if side is None:
                return None
            # compute multiplier off the spread's price if available, fallback 0.91
            mult = _payout_multiplier_from_american(side.get("price"))
            if mult is None:
                mult = 0.91  # default -110 style
            try:
                pub_frac = float(side["public_pct"]) / 100.0
            except Exception:
                return None
            return pub_frac * mult

        favML_exposure = _ml_exposure(ml_fav)
        dogML_exposure = _ml_exposure(ml_dog)
        favSP_exposure = _spread_exposure(sp_fav)
        dogSP_exposure = _spread_exposure(sp_dog)

        # Build candidate set of public bombs
        pain_options = []

        # Favorite ML
        if ml_fav is not None and favML_exposure is not None:
            pain_options.append({
                "market_type": "moneyline",
                "side_type": "favorite",
                "team": ml_fav["team"],
                "line_desc": "ML",
                "price": ml_fav["odds"],
                "public_pct": ml_fav["public_pct"],
                "exposure": favML_exposure,
                "opp_team": ml_dog["team"] if ml_dog else None,
                "opp_odds": ml_dog["odds"] if ml_dog else None,
            })

        # Dog ML
        if ml_dog is not None and dogML_exposure is not None:
            pain_options.append({
                "market_type": "moneyline",
                "side_type": "underdog",
                "team": ml_dog["team"],
                "line_desc": "ML",
                "price": ml_dog["odds"],
                "public_pct": ml_dog["public_pct"],
                "exposure": dogML_exposure,
                "opp_team": ml_fav["team"] if ml_fav else None,
                "opp_odds": ml_fav["odds"] if ml_fav else None,
            })

        # Favorite spread
        if sp_fav is not None and favSP_exposure is not None:
            pain_options.append({
                "market_type": "spread",
                "side_type": "favorite",
                "team": sp_fav["team"],
                "line_desc": f"{sp_fav['spread_num']:+g}",  # "-4.5"
                "price": sp_fav["price"],
                "public_pct": sp_fav["public_pct"],
                "exposure": favSP_exposure,
                "opp_team": sp_dog["team"] if sp_dog else None,
                "opp_line_desc": (f"{sp_dog['spread_num']:+g}" if sp_dog else None),
                "opp_price": sp_dog["price"] if sp_dog else None,
            })

        # Dog spread
        if sp_dog is not None and dogSP_exposure is not None:
            pain_options.append({
                "market_type": "spread",
                "side_type": "underdog",
                "team": sp_dog["team"],
                "line_desc": f"{sp_dog['spread_num']:+g}",  # "+4.5"
                "price": sp_dog["price"],
                "public_pct": sp_dog["public_pct"],
                "exposure": dogSP_exposure,
                "opp_team": sp_fav["team"] if sp_fav else None,
                "opp_line_desc": (f"{sp_fav['spread_num']:+g}" if sp_fav else None),
                "opp_price": sp_fav["price"] if sp_fav else None,
            })

        if not pain_options:
            # can't build anything for this matchup
            continue

        # pick the worst-case side for the book
        pain_pick = max(pain_options, key=lambda x: x["exposure"])

        # Relief string: what the house is rooting for instead
        if pain_pick["market_type"] == "spread":
            # if public bomb is Lions -4.5, book_is_praying_for is Vikings +4.5 cover
            relief_txt = f"{pain_pick.get('opp_team','Other')} {pain_pick.get('opp_line_desc','')} cover"
        else:
            # moneyline case
            if pain_pick["side_type"] == "favorite":
                # public smash on fav ML => book praying for dog upset
                relief_txt = f"{pain_pick.get('opp_team','Other')} moneyline hits"
            else:
                # public smash on dog ML => book praying favorite wins outright
                relief_txt = f"{pain_pick.get('opp_team','Other')} win outright"

        # Narrator lines
        team_upper = str(pain_pick["team"]).upper()

        if pain_pick["market_type"] == "spread":
            narrator_line_top = f"Most dangerous public outcome: {team_upper} {pain_pick['line_desc']}."
            narrator_line_bottom = (
                f"{round(float(pain_pick['public_pct']),0):.0f}% of bettors are on "
                f"{pain_pick['team']} {pain_pick['line_desc']} at "
                f"{int(pain_pick['price']) if pain_pick['price'] is not None else -110}. "
                "If they cover, everybody gets paid. The book does not want this."
            )
            pain_line_for_json = pain_pick["line_desc"]  # "-4.5"
        else:
            narrator_line_top = f"Most dangerous public outcome: {team_upper} ML."
            narrator_line_bottom = (
                f"{round(float(pain_pick['public_pct']),0):.0f}% of bettors are on "
                f"{pain_pick['team']} moneyline at "
                f"{int(pain_pick['price']) if pain_pick['price'] is not None else ''}. "
                "If they win outright, everybody gets paid. The book does not want this."
            )
            # for moneyline we want "+175 ML" style pain_line
            try:
                price_int = int(pain_pick["price"])
            except Exception:
                price_int = pain_pick["price"]
            if price_int is None:
                pain_line_for_json = f"{team_upper} ML"
            else:
                # ensure explicit + sign for dogs
                if isinstance(price_int, int) and price_int > 0:
                    pain_line_for_json = f"+{price_int} ML"
                else:
                    pain_line_for_json = f"{price_int} ML"

        kickoff_iso = _pick_kickoff_et(g)

        # Assemble final payload for this matchup
        payloads.append({
            "template": "book_pain_bar_v1",
            "title": "BOOK PAIN BAR — IF THIS HITS, THE HOUSE HURTS",

            "game": str(matchup),
            "kickoff_et": kickoff_iso,

            "pain_market_type": pain_pick["market_type"],
            "pain_team": str(pain_pick["team"]),
            "pain_line": pain_line_for_json,
            "pain_price": int(pain_pick["price"]) if pain_pick["price"] is not None else None,
            "pain_public_share_pct": round(float(pain_pick["public_pct"]), 1),
            "pain_score": round(float(pain_pick["exposure"]), 3),

            "book_is_praying_for": relief_txt,

            "narrator_line_top": narrator_line_top,
            "narrator_line_bottom": narrator_line_bottom,
        })

    return payloads

def _label_resilience(score: float) -> str:
    """
    Bucket numeric resilience score into label bands.

    70–100 => DEFENDED LINE
    40–69  => RESILIENT
    0–39   => SOFT OPEN / FAKE
    """
    if score is None:
        return "UNKNOWN"
    try:
        s = float(score)
    except Exception:
        return "UNKNOWN"
    if s >= 70:
        return "DEFENDED LINE"
    if s >= 40:
        return "RESILIENT"
    return "SOFT OPEN / FAKE"


def _pick_kickoff_et(group_df: pd.DataFrame) -> str:
    """
    Best-effort kickoff ET timestamp string for that matchup, ISO-style.
    Right now we don't have clean kickoff parsing from DKNetwork scraped HTML,
    so we fall back to 'TBD'. You can upgrade this later.
    """
    return "TBD"


def _collect_spread_snapshots(db_path: str = BETLOGS_DB) -> pd.DataFrame:
    """
    Pull stored spread snapshots from sqlite and normalize them.

    We assume table line_snapshots exists (you already read from it in build_edge_scores):
        matchup TEXT,
        market TEXT,
        side TEXT,
        book TEXT,
        odds INTEGER,
        bets_pct REAL,
        snapshot_type TEXT,  -- e.g. 'OPEN','EARLY','FINAL'
        ts_utc TEXT

    We only care about Spread / DK.

    For each (matchup, snapshot_type) we pick the favorite side = most negative spread,
    and we track:
        team_fav      (string from that side label)
        team_dog      (string from the opposite side label)
        spread_fav    (float, e.g. -4.5)
        price_fav     (current juice/odds for that favorite side)
        price_dog     (juice for dog side)
    """
    try:
        with sqlite3.connect(db_path) as con:
            raw = pd.read_sql_query(
                "SELECT * FROM line_snapshots WHERE market='Spread' AND book='DK';",
                con
            )
    except Exception:
        return pd.DataFrame(columns=[
            "matchup","snapshot_type","team_fav","team_dog",
            "spread_fav","price_fav","price_dog","ts_utc"
        ])

    if raw.empty:
        return pd.DataFrame(columns=[
            "matchup","snapshot_type","team_fav","team_dog",
            "spread_fav","price_fav","price_dog","ts_utc"
        ])

    out_rows = []

    for matchup_val, chunk in raw.groupby("matchup"):
        for snap_type, snapdf in chunk.groupby("snapshot_type"):
            # parse spread numbers off each row's side text
            sp_only = snapdf.copy()
            sp_only["spread_num"] = sp_only["side"].apply(_parse_spread_from_side)
            sp_only = sp_only[sp_only["spread_num"].notnull()]
            if sp_only.empty:
                continue

            # favorite = most negative spread
            fav_row = sp_only.sort_values("spread_num").iloc[0]
            fav_spread = float(fav_row["spread_num"])
            fav_price = clean_odds(fav_row.get("odds"))
            fav_team = str(fav_row.get("side","")).split()[0] if str(fav_row.get("side","")).split() else ""

            # dog = most positive spread
            dog_row = sp_only.sort_values("spread_num").iloc[-1]
            dog_price = clean_odds(dog_row.get("odds"))
            dog_team = str(dog_row.get("side","")).split()[0] if str(dog_row.get("side","")).split() else ""

            out_rows.append({
                "matchup": matchup_val,
                "snapshot_type": snap_type,
                "team_fav": fav_team,
                "team_dog": dog_team,
                "spread_fav": fav_spread,
                "price_fav": fav_price,
                "price_dog": dog_price,
                "ts_utc": fav_row.get("ts_utc"),
            })

    return pd.DataFrame(out_rows)


def _window_resilience_metrics(lines: list[float]) -> tuple[float,float,float]:
    """
    Given [start_line, end_line] in a window:
      MoveSize  = abs(last - first)
      MoveCount = number of distinct values
      Resilience_raw = 1 / (MoveSize * MoveCount)
    We guard against /0.
    Returns (MoveSize, MoveCount, Resilience_raw).
    """
    if not lines:
        return (0.0, 0.0, 0.0)

    uniq_vals = []
    for v in lines:
        if v is None:
            continue
        if v not in uniq_vals:
            uniq_vals.append(v)

    if not uniq_vals:
        return (0.0, 0.0, 0.0)

    first_val = uniq_vals[0]
    last_val = uniq_vals[-1]
    try:
        move_size = abs(float(last_val) - float(first_val))
    except Exception:
        move_size = 0.0

    move_count = float(len(uniq_vals))
    denom = max(move_size * move_count, 1e-6)
    resilience_raw = 1.0 / denom
    return (move_size, move_count, resilience_raw)


def _normalize_resilience_scores(early_raw: float, late_raw: float, all_raws: list[float]) -> tuple[float,float]:
    """
    Turn raw early/late resilience values into 0–100 scores via max scaling:
        score = min(100, max(0, raw / max_raw * 100))
    Returns (early_score, late_score) as rounded ints.
    """
    raws = [r for r in all_raws if r is not None] + [early_raw, late_raw]
    raws = [float(x) for x in raws if x is not None and x >= 0]
    if not raws:
        return (0.0, 0.0)

    max_raw = max(raws)
    if max_raw <= 0:
        return (0.0, 0.0)

    def _scale(val: float) -> int:
        if val is None:
            return 0
        s = (val / max_raw) * 100.0
        if s < 0: s = 0
        if s > 100: s = 100
        return int(round(s))

    return _scale(early_raw), _scale(late_raw)


def _resilience_narrative(
    early_score: float,
    late_score: float,
    open_line: float,
    early_line: float,
    final_line: float,
    fav_team: str
) -> str:
    """
    Generate the human-facing one-liner for the graphic.
    Rules:
    - late >> early and late >=70  => "froze, defended"
    - both <40                     => "still soft"
    - early >=70 then late <40     => "leaked late"
    - late > early                 => "cleaned up late"
    - else                         => "still drifting"
    """
    try:
        e = float(early_score)
        l = float(late_score)
    except Exception:
        e = 0.0
        l = 0.0

    traj = f"{open_line:+g} → {final_line:+g}"

    if l >= e and l >= 70:
        return f"{fav_team} got hit and the number climbed, then froze at {final_line:+g}. This is a defended {fav_team} line."
    if e < 40 and l < 40:
        return f"Open was theater. It kept wobbling ({traj}) and never really settled."
    if e >= 70 and l < 40:
        return f"{fav_team} opened heavy and books tried to hold it, but it leaked late. Not fully trusted."
    if l > e:
        return f"Early move got cleaned up. By final, that {fav_team} number stopped moving."
    return f"Still drifting ({traj}). This line never truly stabilized."


def build_line_resilience_tables_and_json(db_path: str = BETLOGS_DB) -> tuple[pd.DataFrame, dict]:
    """
    Main builder for LINE RESILIENCE.

    We expect that you've been snapshotting spreads into sqlite with snapshot_type
    values like 'OPEN', 'EARLY' (Sat 10 PM ET), and 'FINAL' (30 min pre-kick).

    We:
    - pull those snapshots
    - compute 'early window' (OPEN -> EARLY)
    - compute 'late window'  (EARLY -> FINAL)
    - generate normalized 0–100 resilience scores
    - label them into DEFENDED LINE / RESILIENT / SOFT OPEN / FAKE
    - narrate the matchup
    - output:
        * resilience_df (full table, all games)
        * resilience_json (top 3 volatile games, matches your contract)
    """
    snap = _collect_spread_snapshots(db_path)
    if snap.empty:
        return (
            pd.DataFrame(columns=[
                "game","kickoff_et","team_fav","team_dog",
                "open_spread_fav","early_spread_fav","final_spread_fav",
                "early_resilience_score","late_resilience_score",
                "label_early","label_late","note",
            ]),
            {
                "template": "line_resilience_v1",
                "title": "LINE RESILIENCE — WHICH NUMBERS ARE REAL",
                "slate_timestamp_et": datetime.utcnow().isoformat(),
                "games": [],
            }
        )

    per_game_raws = {}
    meta_store = {}

    for matchup_val, g in snap.groupby("matchup"):
        def _get(snap_type, col):
            try:
                return g[g["snapshot_type"] == snap_type][col].iloc[0]
            except Exception:
                return None

        fav_team = _get("FINAL","team_fav") or _get("EARLY","team_fav") or _get("OPEN","team_fav") or ""
        dog_team = _get("FINAL","team_dog") or _get("EARLY","team_dog") or _get("OPEN","team_dog") or ""

        open_spread  = _get("OPEN","spread_fav")
        early_spread = _get("EARLY","spread_fav")
        final_spread = _get("FINAL","spread_fav")

        # early window: OPEN -> EARLY
        e_move_size, e_move_count, e_raw = _window_resilience_metrics([open_spread, early_spread])

        # late window: EARLY -> FINAL
        l_move_size, l_move_count, l_raw = _window_resilience_metrics([early_spread, final_spread])

        per_game_raws[matchup_val] = {
            "team_fav": fav_team,
            "team_dog": dog_team,
            "open_spread": open_spread,
            "early_spread": early_spread,
            "final_spread": final_spread,
            "early_raw": e_raw,
            "late_raw": l_raw,
        }

        meta_store[matchup_val] = {
            "kickoff_et": _pick_kickoff_et(g),
        }

    # global normalization across slate
    all_raw_vals = []
    for vals in per_game_raws.values():
        all_raw_vals.append(vals["early_raw"])
        all_raw_vals.append(vals["late_raw"])

    rows_out = []

    for matchup_val, vals in per_game_raws.items():
        early_score, late_score = _normalize_resilience_scores(
            vals["early_raw"],
            vals["late_raw"],
            all_raw_vals
        )

        label_early = _label_resilience(early_score)
        label_late  = _label_resilience(late_score)

        note_txt = _resilience_narrative(
            early_score,
            late_score,
            vals["open_spread"],
            vals["early_spread"],
            vals["final_spread"],
            vals["team_fav"],
        )

        rows_out.append({
            "game": matchup_val,
            "kickoff_et": meta_store[matchup_val]["kickoff_et"],
            "team_fav": vals["team_fav"],
            "team_dog": vals["team_dog"],
            "open_spread_fav": vals["open_spread"],
            "early_spread_fav": vals["early_spread"],
            "final_spread_fav": vals["final_spread"],
            "early_resilience_score": early_score,
            "late_resilience_score": late_score,
            "label_early": label_early,
            "label_late": label_late,
            "note": note_txt,
        })

    resilience_df = pd.DataFrame(rows_out)

    # rank by volatility for export JSON (most movement from open -> final)
    def _volatility(r):
        try:
            return abs(float(r["final_spread_fav"]) - float(r["open_spread_fav"]))
        except Exception:
            return 0.0

    games_sorted = sorted(rows_out, key=_volatility, reverse=True)
    top3 = games_sorted[:3]

    resilience_json = {
        "template": "line_resilience_v1",
        "title": "LINE RESILIENCE — WHICH NUMBERS ARE REAL",
        "slate_timestamp_et": datetime.utcnow().isoformat(),
        "games": []
    }

    for grow in top3:
        resilience_json["games"].append({
            "game": grow["game"],
            "kickoff_et": grow["kickoff_et"],
            "team_fav": grow["team_fav"],
            "team_dog": grow["team_dog"],
            "open_spread_fav": grow["open_spread_fav"],
            "early_spread_fav": grow["early_spread_fav"],
            "final_spread_fav": grow["final_spread_fav"],
            "early_resilience_score": grow["early_resilience_score"],
            "late_resilience_score": grow["late_resilience_score"],
            "label_early": grow["label_early"],
            "label_late": grow["label_late"],
            "note": grow["note"],
        })

    return resilience_df, resilience_json

def implied_prob_from_moneyline(odds_american: int | float | None) -> float | None:
    """
    Convert American moneyline odds to implied win probability (no vig removal).
    -210 -> 210 / (210 + 100)
    +175 -> 100 / (175 + 100)
    Returns value in [0,1] or None if invalid.
    """
    if odds_american is None:
        return None
    try:
        o = float(odds_american)
    except Exception:
        return None

    if o < 0:
        # favorite
        ab = abs(o)
        return ab / (ab + 100.0)
    else:
        # dog / plus money
        return 100.0 / (o + 100.0)


def _extract_team_from_side_label(side_txt: str) -> str:
    """
    side_txt from DK for moneyline usually looks like:
      "DET ML" or "LAC ML" or "New England Patriots ML"
      sometimes "DET Moneyline" etc.
    We strip the trailing token (ML / Moneyline) and return the team string.
    Fallback: first token(s) before last token.
    """
    s = str(side_txt or "").strip()
    # Try common suffixes
    for suf in [" ML", " Moneyline", " moneyline", " ML (incl. OT)"]:
        if s.endswith(suf):
            return s[: -len(suf)].strip()
    # Generic fallback: drop last word if it looks like ML-ish
    parts = s.split()
    if len(parts) > 1 and parts[-1].lower().startswith("ml"):
        return " ".join(parts[:-1]).strip()
    return s

def _gravity_label(mult: float) -> str:
    """
    Build the human-facing gravity label:
    >= 1.00 -> 'X.XX× Overweight'
    <  1.00 -> 'X.XX× Belief'
    """
    try:
        m = float(mult)
    except Exception:
        return ""
    if m >= 1.0:
        return f"{m:.2f}× Overweight"
    return f"{m:.2f}× Belief"


def _pick_kickoff_et(df_match: pd.DataFrame) -> str:
    """
    Try to extract an ET kickoff timestamp string for this matchup.
    We'll look for any plausible time column in the DK splits data.
    If we can't find one, we return '' (empty string).
    """
    candidate_cols = [
        "kickoff_et",
        "kickoff_time_et",
        "start_et",
        "start_time_et",
        "commence_time",
        "start_time",
    ]
    for col in candidate_cols:
        if col in df_match.columns:
            val = df_match[col].iloc[0]
            # if it's already a string, assume it's ET in ISO form or close enough
            if isinstance(val, str):
                return val
            # if it's datetime-like, turn it into ISO
            if isinstance(val, (datetime, pd.Timestamp)):
                try:
                    return pd.Timestamp(val).isoformat()
                except Exception:
                    pass
    return ""


def build_handle_gravity_json(df: pd.DataFrame) -> list[dict]:
    """
    Build Handle Gravity JSON objects per matchup.

    Output schema per matchup (each element of the list):
    {
      "template": "handle_gravity_v1",
      "title": "HANDLE GRAVITY: WHO AMERICA IS RIDING TONIGHT",

      "game": "Vikings @ Lions",
      "kickoff_et": "2025-11-02T20:20:00",

      "favorite_team": "Lions",
      "favorite_ml_odds": -210,
      "favorite_public_share_pct": 82,
      "favorite_gravity_multiplier": 1.21,
      "favorite_gravity_label": "1.21× Overweight",

      "underdog_team": "Vikings",
      "underdog_ml_odds": 175,
      "underdog_public_share_pct": 18,
      "underdog_gravity_multiplier": 0.49,
      "underdog_gravity_label": "0.49× Belief",

      "leader_team": "Lions",
      "leader_phrase": "Detroit is carrying the country. The Lions moneyline is emotionally inflated.",
      "narrator_line": "Public is 1.21× heavier on Detroit than the odds say they should be."
    }
    """
    if df is None or df.empty:
        return []

    # We only care about moneyline rows
    money = df[df["market"].str.lower().str.startswith("money")].copy()
    if money.empty:
        return []

    payloads: list[dict] = []

    # Work per matchup
    for matchup, g in money.groupby("matchup"):
        # Step 1. Extract both moneyline sides
        sides_parsed = []
        for _, r in g.iterrows():
            odds_val = clean_odds(r.get("odds"))
            team_label = _extract_team_from_side_label(str(r.get("side", "")))

            # choose %handle first; fallback to %bets
            pub_handle = r.get("%handle")
            pub_bets = r.get("%bets")
            if pub_handle is not None and pub_handle == pub_handle:
                pub_pct_raw = pub_handle
            else:
                pub_pct_raw = pub_bets
            if pub_pct_raw is None:
                continue

            try:
                pub_pct_float = float(pub_pct_raw)
            except Exception:
                continue

            sides_parsed.append({
                "team": team_label,
                "odds": odds_val,
                "public_pct": pub_pct_float,  # 0-100
            })

        # Need two usable sides
        if len(sides_parsed) < 2:
            continue

        # Step 2. Identify favorite vs underdog
        # Rule: negative odds => favorite. positive odds => dog.
        fav_side = None
        dog_side = None
        for s in sides_parsed:
            o = s["odds"]
            if o is None:
                continue
            try:
                o_f = float(o)
            except Exception:
                continue
            if o_f < 0 and fav_side is None:
                fav_side = s
            elif o_f >= 0 and dog_side is None:
                dog_side = s

        # Fallback if book labeled both minus or both plus (edge cases / bad scrape)
        if fav_side is None or dog_side is None:
            tmp_sorted = sorted(
                sides_parsed,
                key=lambda x: (float(x["odds"]) if x["odds"] is not None else 0.0)
            )
            # most negative / smallest number = favorite
            fav_side = tmp_sorted[0]
            dog_side = tmp_sorted[-1]

        if fav_side is None or dog_side is None:
            continue

        fav_odds = fav_side["odds"]
        dog_odds = dog_side["odds"]

        fav_imp = implied_prob_from_moneyline(fav_odds)
        dog_imp = implied_prob_from_moneyline(dog_odds)
        if fav_imp is None or fav_imp <= 0 or dog_imp is None or dog_imp <= 0:
            continue

        fav_share_frac = float(fav_side["public_pct"]) / 100.0
        dog_share_frac = float(dog_side["public_pct"]) / 100.0

        fav_mult = fav_share_frac / fav_imp
        dog_mult = dog_share_frac / dog_imp

        # Step 3. Decide leader (who "owns gravity")
        if fav_mult >= dog_mult:
            leader_team = fav_side["team"]
            leader_mult = fav_mult
        else:
            leader_team = dog_side["team"]
            leader_mult = dog_mult

        # Step 4. Narration strings
        narrator_line = (
            f"Public is {leader_mult:.2f}× heavier on {leader_team} "
            "than the odds say they should be."
        )

        leader_phrase = (
            f"{leader_team} is carrying the country. "
            f"The {leader_team} moneyline is emotionally inflated."
        )

        kickoff_iso = _pick_kickoff_et(g)

        # Step 5. Assemble final payload
        payloads.append({
            "template": "handle_gravity_v1",
            "title": "HANDLE GRAVITY: WHO AMERICA IS RIDING TONIGHT",

            "game": str(matchup),
            "kickoff_et": kickoff_iso,

            "favorite_team": str(fav_side["team"]),
            "favorite_ml_odds": int(fav_odds) if fav_odds is not None else None,
            "favorite_public_share_pct": round(float(fav_side["public_pct"]), 1),
            "favorite_gravity_multiplier": round(float(fav_mult), 2),
            "favorite_gravity_label": _gravity_label(fav_mult),

            "underdog_team": str(dog_side["team"]),
            "underdog_ml_odds": int(dog_odds) if dog_odds is not None else None,
            "underdog_public_share_pct": round(float(dog_side["public_pct"]), 1),
            "underdog_gravity_multiplier": round(float(dog_mult), 2),
            "underdog_gravity_label": _gravity_label(dog_mult),

            "leader_team": str(leader_team),
            "leader_phrase": leader_phrase,
            "narrator_line": narrator_line,
        })

    return payloads

def build_handle_gravity(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Compute HANDLE GRAVITY for each matchup's moneyline market.

    Concept:
    - Pull only Moneyline rows for each matchup.
    - For each side in that matchup:
        * Get moneyline odds (american int)
        * Get public share (prefer %handle, fallback %bets) / 100 -> decimal
        * Convert odds -> implied probability
        * Gravity = public_share / implied_prob
    - Identify which side has higher Gravity. That side is "market_gravity_team".

    We output two things:
    1. summary_df: top matchups ranked by the leader gravity score.
       Columns:
         ["Matchup","Gravity Team","Gravity x","Public %","ML Odds","Note"]
       Where "Note" is f"{public_pct}% money on {odds}" for that team.

    2. raw_rows: a list of dict rows (one per matchup) with full detail
       needed for logging / downstream graphics, including both sides.

    If a matchup doesn't have BOTH sides of the moneyline with usable data,
    we skip it.
    """
    cols_summary = ["Matchup","Gravity Team","Gravity x","Public %","ML Odds","Note"]
    if df.empty:
        return pd.DataFrame(columns=cols_summary), []

    # isolate moneyline rows
    money = df[df["market"].str.lower().str.startswith("money")].copy()
    if money.empty:
        return pd.DataFrame(columns=cols_summary), []

    out_rows_summary: list[dict] = []
    out_rows_log: list[dict] = []

    # group by matchup so we can compare two sides
    for matchup, g in money.groupby("matchup"):
        # parse each side into structured info
        side_info = []
        for _, r in g.iterrows():
            odds_val = clean_odds(r.get("odds"))
            pub_handle = r.get("%handle")
            pub_bets = r.get("%bets")
            # prefer handle% if available, else bets%
            if pub_handle is not None and pub_handle == pub_handle:
                pub_share_pct = pub_handle
            else:
                pub_share_pct = pub_bets
            if pub_share_pct is None:
                continue
            try:
                pub_share = float(pub_share_pct) / 100.0
            except Exception:
                continue

            team_label = _extract_team_from_side_label(str(r.get("side","")))
            imp_prob = implied_prob_from_moneyline(odds_val)
            if imp_prob is None or imp_prob <= 0:
                continue
            gravity = pub_share / imp_prob

            side_info.append({
                "team": team_label,
                "odds": odds_val,
                "public_share": pub_share,
                "public_share_pct": float(pub_share_pct),
                "imp_prob": imp_prob,
                "gravity": gravity,
            })

        if len(side_info) < 2:
            # need both sides to compare (favorite & dog); skip otherwise
            continue

        # pick leader (who's warping harder)
        leader = max(side_info, key=lambda x: x["gravity"])
        # also grab the other side for logging completeness
        if side_info[0] is leader:
            other = side_info[1]
        else:
            other = side_info[0]

        out_rows_summary.append({
            "Matchup": matchup,
            "Gravity Team": leader["team"],
            "Gravity x": round(leader["gravity"], 2),
            "Public %": round(leader["public_share_pct"], 1),
            "ML Odds": leader["odds"],
            "Note": f"{round(leader['public_share_pct'],1)}% money on {leader['odds']}",
        })

        out_rows_log.append({
            "matchup": matchup,
            "team_fav": leader["team"],  # we'll store under leader here; detailed split below
            "odds_leader": leader["odds"],
            "public_leader": leader["public_share"],
            "imp_leader": leader["imp_prob"],
            "gravity_leader": leader["gravity"],
            "team_other": other["team"],
            "odds_other": other["odds"],
            "public_other": other["public_share"],
            "imp_other": other["imp_prob"],
            "gravity_other": other["gravity"],
        })

    if not out_rows_summary:
        return pd.DataFrame(columns=cols_summary), []

    summary_df = pd.DataFrame(out_rows_summary)
    # sort by Gravity x desc to surface strongest narrative first
    summary_df = summary_df.sort_values("Gravity x", ascending=False).reset_index(drop=True)

    # take top 5 for screenshot/posting
    return summary_df.head(5), out_rows_log


def log_handle_gravity_snapshot(rows_log: list[dict]):
    """
    Persist Handle Gravity snapshot rows into sqlite for time-series tracking.
    Table: handle_gravity_snapshots
    Columns:
      ts TEXT ISO,
      matchup TEXT,
      team_leader TEXT,
      odds_leader REAL,
      public_leader REAL,
      imp_leader REAL,
      gravity_leader REAL,
      team_other TEXT,
      odds_other REAL,
      public_other REAL,
      imp_other REAL,
      gravity_other REAL
    """
    if not rows_log:
        return
    ts_now = datetime.now(pytz.timezone("America/Los_Angeles")).isoformat()
    try:
        with sqlite3.connect(BETLOGS_DB) as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS handle_gravity_snapshots (
                    ts TEXT,
                    matchup TEXT,
                    team_leader TEXT,
                    odds_leader REAL,
                    public_leader REAL,
                    imp_leader REAL,
                    gravity_leader REAL,
                    team_other TEXT,
                    odds_other REAL,
                    public_other REAL,
                    imp_other REAL,
                    gravity_other REAL
                )
                """
            )
            con.commit()

            cur.executemany(
                """
                INSERT INTO handle_gravity_snapshots (
                    ts,
                    matchup,
                    team_leader,
                    odds_leader,
                    public_leader,
                    imp_leader,
                    gravity_leader,
                    team_other,
                    odds_other,
                    public_other,
                    imp_other,
                    gravity_other
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        ts_now,
                        row.get("matchup"),
                        row.get("team_fav"),
                        row.get("odds_leader"),
                        row.get("public_leader"),
                        row.get("imp_leader"),
                        row.get("gravity_leader"),
                        row.get("team_other"),
                        row.get("odds_other"),
                        row.get("public_other"),
                        row.get("imp_other"),
                        row.get("gravity_other"),
                    )
                    for row in rows_log
                ],
            )
            con.commit()
    except Exception:
        # snapshot logging failure should never kill UI
        pass

def build_rlm_tracker(df: pd.DataFrame, open_df: pd.DataFrame | None = None, min_move_pts: float = 1.0) -> pd.DataFrame:
    """Create 'REVERSE LINE MOVEMENT' tracker.
    Definition:
    - Majority (%bets > 50) is on Side X now.
    - Line moved TOWARD the other side vs open by >= min_move_pts.

    We approximate line move only for spread markets, because that's what NFL Twitter actually fights over.

    Inputs:
    - df: current splits dataframe
    - open_df: snapshot of OPEN lines (from DB line_snapshots with snapshot_type='OPEN'), optional. If not provided, returns empty.

    Output columns: matchup, side_public, bets%, open_spread, current_spread, move_pts, rlm_flag
    """
    cols = ["matchup","side_public","Bets %","Open Spread","Current Spread","Move (pts)","RLM"]
    if df.empty or open_df is None or open_df.empty:
        return pd.DataFrame(columns=cols)

    # prep open spread by matchup+side text
    odf = open_df.copy()
    odf = odf[odf["market"].str.lower().str.startswith("spread")].copy()
    odf["open_spread"] = odf["side"].apply(_parse_spread_from_side)
    odf = odf[["matchup","side","open_spread","bets_pct"]]
    odf = odf.rename(columns={"side":"side_open_lbl","bets_pct":"open_bets"})

    cur = df[df["market"].str.lower().str.startswith("spread")].copy()
    if cur.empty:
        return pd.DataFrame(columns=cols)

    cur["cur_spread"] = cur["side"].apply(_parse_spread_from_side)

    # We'll merge on matchup + side label text. It's heuristic but good enough for daily content.
    merged = cur.merge(
        odf,
        left_on=["matchup","side"],
        right_on=["matchup","side_open_lbl"],
        how="left"
    )

    rlm_rows = []
    for _, r in merged.iterrows():
        matchup = str(r.get("matchup",""))
        side_lbl = str(r.get("side",""))
        bets_now = r.get("%bets")
        cur_spread = r.get("cur_spread")
        open_spread = r.get("open_spread")
        if bets_now is None or cur_spread is None or open_spread is None:
            continue
        try:
            bets_now_f = float(bets_now)
        except Exception:
            continue

        # who's public NOW
        public_now = bets_now_f > 50.0

        # movement direction in points (signed)
        move_pts = cur_spread - open_spread
        # We say RLM if majority is on this side but line moved AGAINST this side (i.e. made this side cheaper/more attractive for other side).
        # For favorites (negative spreads): moving from -3.5 to -2.5 is TOWARD that favorite (not RLM). Going -3.5 to -4.5 is AWAY (RLM if public is on favorite).
        # We'll handle sign:
        rlm_flag = False
        try:
            # if this side is the favorite (spread < 0) and public is hammering it,
            # then RLM if the spread number got WORSE for them (more negative).
            if cur_spread < 0 and public_now:
                if cur_spread < open_spread and abs(cur_spread - open_spread) >= min_move_pts:
                    rlm_flag = True
            # if this side is the dog (spread > 0) and public is hammering it,
            # then RLM if the dog got LESS points (line moved toward favorite).
            if cur_spread > 0 and public_now:
                if cur_spread > open_spread and abs(cur_spread - open_spread) >= min_move_pts:
                    # dog went from +7.5 to +6.0 -> fewer points -> moving against public dog
                    rlm_flag = True
        except Exception:
            rlm_flag = False

        if rlm_flag:
            rlm_rows.append({
                "matchup": matchup,
                "side_public": side_lbl,
                "Bets %": round(bets_now_f,1),
                "Open Spread": open_spread,
                "Current Spread": cur_spread,
                "Move (pts)": round(move_pts,2),
                "RLM": "RLM 🔥",
            })

    out = pd.DataFrame(rlm_rows)
    if out.empty:
        return out
    # Sort by abs(move)
    out = out.sort_values("Move (pts)", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out.head(5)

# ============================================================================
# LINE RESILIENCE INDEX (EARLY vs LATE WINDOW DEFENSE)
# ============================================================================

def _public_share_pct(row: pd.Series) -> float | None:
    """
    Return best available public share % for this row as a 0-100 float.
    We prefer %handle. If that's missing, we fall back to %bets.
    """
    h = row.get("%handle")
    b = row.get("%bets")
    try:
        if h is not None and not pd.isna(h):
            return float(h)
        if b is not None and not pd.isna(b):
            return float(b)
    except Exception:
        pass
    return None


def _extract_matchup_spread_snapshot(df_match: pd.DataFrame) -> dict | None:
    """
    Take all markets for one matchup. We only care about Spread.

    We pick ONE anchor spread number to track for that matchup.
    Logic:
    - Look at rows where market starts with "Spread"
    - Parse numeric spread with _parse_spread_from_side("DET -3.5") -> -3.5
    - Grab the side with the *most negative* spread (the favorite), because
      that's the headline number the public argues about.
    - Also pull current public % (handle first, else bets).

    Returns:
      {
        "matchup": "Lions @ Vikings",
        "spread":  -3.5,
        "handle_pct": 68.2    # 68.2% of handle/bets, not decimal
      }
    or None if we couldn't get clean data.
    """
    sub = df_match[df_match["market"].str.lower().str.startswith("spread")].copy()
    if sub.empty:
        return None

    picks = []
    for _, r in sub.iterrows():
        spr_val = _parse_spread_from_side(str(r.get("side", "")))
        if spr_val is None:
            continue
        pub_pct = _public_share_pct(r)
        if pub_pct is None:
            continue
        picks.append({
            "spread": spr_val,
            "handle_pct": pub_pct,
        })

    if not picks:
        return None

    # Sort by spread first (more negative first), then absolute value fallback
    # Example: [-7.5, -3.5, +3.5] -> we take -7.5.
    picks_sorted = sorted(picks, key=lambda x: (x["spread"], abs(x["spread"])))
    chosen = picks_sorted[0]

    return {
        "matchup": str(df_match["matchup"].iloc[0]),
        "spread": float(chosen["spread"]),
        "handle_pct": float(chosen["handle_pct"]),
    }


def capture_line_resilience_snapshot(df: pd.DataFrame, snapshot_type: str):
    """
    You manually hit this from the sidebar at different windows:
    - "OPEN"  (Friday ~5pm ET)
    - "EARLY" (Sunday AM or Sat AM for island games)
    - "MID"   (Sat night if you care, optional)
    - "FINAL" (30 min before kick)

    We log the favorite's current spread and public split at that moment.

    Schema stored in sqlite table line_resilience_snapshots:
        matchup TEXT
        snapshot_type TEXT  -- 'OPEN', 'EARLY', 'MID', 'FINAL'
        ts_utc TEXT         -- ISO timestamp
        spread REAL         -- e.g. -3.5
        handle_pct REAL     -- e.g. 68.2 (% handle or % bets we saw)

    PRIMARY KEY (matchup, snapshot_type, ts_utc) lets you store history.
    Later we pull the latest row per (matchup, snapshot_type).
    """
    if df.empty:
        return
    snapshot_type = str(snapshot_type).upper().strip()
    ts_now_utc = datetime.utcnow().isoformat()

    rows_to_write = []
    for matchup_val, sub in df.groupby("matchup"):
        snap = _extract_matchup_spread_snapshot(sub)
        if not snap:
            continue
        rows_to_write.append((
            snap["matchup"],
            snapshot_type,
            ts_now_utc,
            snap["spread"],
            snap["handle_pct"],
        ))

    if not rows_to_write:
        return

    _ensure_parent(BETLOGS_DB)
    with sqlite3.connect(BETLOGS_DB) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS line_resilience_snapshots (
                matchup TEXT,
                snapshot_type TEXT,
                ts_utc TEXT,
                spread REAL,
                handle_pct REAL,
                PRIMARY KEY (matchup, snapshot_type, ts_utc)
            )
            """
        )
        con.executemany(
            """
            INSERT OR REPLACE INTO line_resilience_snapshots (
                matchup, snapshot_type, ts_utc, spread, handle_pct
            ) VALUES (?,?,?,?,?)
            """,
            rows_to_write
        )


def _latest_snapshots_by_type() -> pd.DataFrame:
    """
    Pull the most recent snapshot for each matchup+snapshot_type.
    We'll use this to build the Line Resilience Index.

    Returns DataFrame columns:
      matchup, snapshot_type, ts_utc, spread, handle_pct
    If table doesn't exist yet or is empty, returns empty df.
    """
    try:
        with sqlite3.connect(BETLOGS_DB) as con:
            snap = pd.read_sql_query(
                """
                SELECT matchup, snapshot_type, ts_utc, spread, handle_pct
                FROM line_resilience_snapshots
                """,
                con
            )
    except Exception:
        return pd.DataFrame(columns=["matchup","snapshot_type","ts_utc","spread","handle_pct"])

    if snap.empty:
        return snap

    # keep only the latest timestamp per matchup+snapshot_type
    snap = (
        snap.sort_values("ts_utc")
            .groupby(["matchup", "snapshot_type"], as_index=False)
            .tail(1)
            .reset_index(drop=True)
    )
    return snap


def _categorize_resilience(score: float) -> str:
    """
    Map RI_score → category label.
    Mirrors the spec:
    80–100   🧱 Defended Line
    60–79    Resilient
    40–59    Neutral
    20–39    Soft
    0–19     Fake
    """
    if score >= 80:
        return "🧱 Defended Line"
    if score >= 60:
        return "Resilient"
    if score >= 40:
        return "Neutral"
    if score >= 20:
        return "Soft"
    return "Fake"


def compute_line_resilience_index() -> pd.DataFrame:
    """
    Build the Line Resilience Index table for dashboard + social.

    We expect you to have logged at least these snapshot types for a matchup:
      - OPEN  (baseline spread, early in cycle)
      - EARLY (public starts piling in)
      - FINAL (30 min pre-kick)

    We interpret:
      early_handle = % handle (or bets) on that spread at EARLY
      final_handle = % handle (or bets) at FINAL
      handle_delta = final_handle - early_handle

      open_spread  = spread at OPEN
      early_spread = spread at EARLY
      final_spread = spread at FINAL

      move_early   = early_spread - open_spread
      move_late    = final_spread - early_spread
      total_move   = final_spread - open_spread

    Raw Resilience score:
      RI_raw = (abs(move_early) + 0.5*abs(move_late)) / (abs(total_move) + 0.01)

      if move_late * handle_delta < 0:
          RI_raw *= 1.25   # book held against pressure
      elif move_late * handle_delta > 0:
          RI_raw *= 0.75   # line yielded to pressure

    Then normalize to 0–100 across matchups:
      RI_score = 100 * RI_raw / max(RI_raw)

    Output columns:
      Matchup
      Open→Final     e.g. "-2.5 → -3.5"
      PublicFlow     e.g. "68%→74%"
      RI_score       float 0–100
      Tag            bucket label
      Note           short interpretation ("Book held vs public", etc.)
    """
    snap = _latest_snapshots_by_type()
    if snap.empty:
        return pd.DataFrame(columns=["Matchup","Open→Final","PublicFlow","RI_score","Tag","Note"])

    rows_out = []
    for matchup_val in snap["matchup"].unique():
        ms = snap[snap["matchup"] == matchup_val]

        def _get_val(stype: str, col: str):
            sub = ms[ms["snapshot_type"] == stype]
            if sub.empty:
                return None
            return sub.iloc[0].get(col)

        open_spread  = _get_val("OPEN",  "spread")
        early_spread = _get_val("EARLY", "spread")
        final_spread = _get_val("FINAL", "spread")

        early_handle = _get_val("EARLY", "handle_pct")
        final_handle = _get_val("FINAL", "handle_pct")

        # Need OPEN, EARLY, FINAL at minimum or we skip this matchup.
        if any(v is None for v in [open_spread, early_spread, final_spread, early_handle, final_handle]):
            continue

        try:
            handle_delta = float(final_handle) - float(early_handle)
            move_early   = float(early_spread) - float(open_spread)
            move_late    = float(final_spread) - float(early_spread)
            total_move   = float(final_spread) - float(open_spread)
        except Exception:
            continue

        RI_raw = (abs(move_early) + 0.5 * abs(move_late)) / (abs(total_move) + 0.01)

        prod = move_late * handle_delta
        if prod < 0:
            RI_raw *= 1.25
            pressure_note = "Book held vs public"
        elif prod > 0:
            RI_raw *= 0.75
            pressure_note = "Line yielding to public"
        else:
            pressure_note = "Stable / balanced"

        rows_out.append({
            "matchup": matchup_val,
            "open_spread": open_spread,
            "final_spread": final_spread,
            "early_handle": early_handle,
            "final_handle": final_handle,
            "RI_raw": RI_raw,
            "pressure_note": pressure_note,
        })

    if not rows_out:
        return pd.DataFrame(columns=["Matchup","Open→Final","PublicFlow","RI_score","Tag","Note"])

    tmp = pd.DataFrame(rows_out)

    max_val = tmp["RI_raw"].max()
    if max_val and max_val > 0:
        tmp["RI_score"] = 100.0 * (tmp["RI_raw"] / max_val)
    else:
        tmp["RI_score"] = 0.0

    tmp["Tag"] = tmp["RI_score"].apply(_categorize_resilience)

    def _format_flow(r):
        try:
            eh = float(r["early_handle"])
            fh = float(r["final_handle"])
            return f"{eh:.0f}%→{fh:.0f}%"
        except Exception:
            return ""

    def _format_spreads(r):
        ospr = r["open_spread"]
        fspr = r["final_spread"]
        try:
            return f"{ospr:+g} → {fspr:+g}"
        except Exception:
            return f"{ospr} → {fspr}"

    tmp["Open→Final"] = tmp.apply(_format_spreads, axis=1)
    tmp["PublicFlow"] = tmp.apply(_format_flow, axis=1)

    out_cols = [
        "matchup",
        "Open→Final",
        "PublicFlow",
        "RI_score",
        "Tag",
        "pressure_note",
    ]
    tmp = tmp[out_cols].rename(columns={
        "matchup": "Matchup",
        "pressure_note": "Note",
    })

    tmp = tmp.sort_values("RI_score", ascending=False).reset_index(drop=True)
    return tmp.head(5)

# ============================================================================
# MARKET FINGERPRINT — FULL SLATE MOOD (WEEKLY SLATE PERSONALITY)
# ============================================================================

def _safe_float(x, default=None):
    """Safely convert a value to float, returning `default` on failure."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default

def _pick_side_rows_for_matchup_spread(df_match: pd.DataFrame) -> list[dict]:
    """
    From a single game's rows, pull all spread sides that have:
      - current spread number
      - %bets and %handle
    Also try to attach OPEN spread (from the latest stored OPEN snapshot) later,
    so this helper only extracts 'current' side state.

    Returns a list of dicts like:
      {
        "side_txt": "Lions -4.5",
        "spread_now": -4.5,
        "bets_pct": 61.0,
        "handle_pct": 68.0
      }
    Skips rows we can't parse.
    """
    out = []
    sp = df_match[df_match["market"].str.lower().str.startswith("spread")].copy()
    for _, r in sp.iterrows():
        side_txt = str(r.get("side", ""))
        spread_now = _parse_spread_from_side(side_txt)
        bets_pct = _safe_float(r.get("%bets"))
        handle_pct = _safe_float(r.get("%handle"))
        if spread_now is None:
            continue
        if bets_pct is None or handle_pct is None:
            continue
        out.append({
            "side_txt": side_txt,
            "spread_now": spread_now,
            "bets_pct": bets_pct,
            "handle_pct": handle_pct,
        })
    return out

def _get_open_spread_for_side(open_df: pd.DataFrame,
                              matchup: str,
                              side_txt: str) -> float | None:
    """
    Try to recover the OPEN spread line for a specific matchup+side label,
    using the 'line_snapshots' table snapshot_type='OPEN' (already pulled into open_df).
    We assume open_df columns: matchup, side, spread.
    """
    if open_df is None or open_df.empty:
        return None
    sub = open_df[(open_df["matchup"] == matchup) & (open_df["side"] == side_txt)]
    if sub.empty:
        return None
    sp_open = _safe_float(sub.iloc[0].get("spread"))
    return sp_open

def compute_market_fingerprint(df: pd.DataFrame,
                               open_df: pd.DataFrame | None = None
                               ) -> tuple[pd.DataFrame, dict]:
    """
    Build the 'Market Fingerprint' for the slate.

    Per matchup we compute:
      skew:
          abs(handle_pct_dominant - 50)
          (dominant side = side with higher handle_pct)

      HC (herd conviction):
          HC = 100 - abs(handle_pct_dominant - bets_pct_dominant)

      line_drift:
          abs(current_spread - open_spread) for that dominant side.
          If we don't have open_spread, drift = 0.0.

      MFS_raw:
          0.4*skew + 0.4*(100-HC) + 0.2*(line_drift * 10)

    Then normalize each game's MFS_raw -> 0–100 within slate:
        MFS_score = 100 * (MFS_raw / max(MFS_raw))

    We also assemble slate-level mood metadata:
      - avg_mfs
      - num_high (games with MFS_score > 90)
      - lopsided game (max skew)
      - disagreement game (min HC)
      - volatile game (max line_drift)
      - mood label bucket

    Returns:
      table_df: columns [
         "Matchup","Team","Skew","HC","Drift","MFS_score"
      ]
      meta: dict with keys [
         "avg_mfs","num_high","mood_label",
         "lopsided_desc","disagreement_desc","volatile_desc"
      ]
    """
    cols = ["Matchup","Team","Skew","HC","Drift","MFS_raw","MFS_score"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols), {}

    out_rows = []

    for matchup_val, g in df.groupby("matchup"):
        # pull spread sides for this matchup
        sides_now = _pick_side_rows_for_matchup_spread(g)
        if len(sides_now) < 2:
            # we want both sides of spread to consider this game
            continue

        # pick dominant public side by handle_pct
        dom = max(sides_now, key=lambda r: r["handle_pct"])

        side_txt = dom["side_txt"]
        spread_now = _safe_float(dom["spread_now"])
        bets_pct = _safe_float(dom["bets_pct"])
        handle_pct = _safe_float(dom["handle_pct"])

        # sanity guard
        if spread_now is None or bets_pct is None or handle_pct is None:
            continue

        # skew: how lopsided is money
        skew = abs(handle_pct - 50.0)

        # herd conviction
        # HC high => crowd unified; HC low => books vs public split
        HC_val = 100.0 - abs(handle_pct - bets_pct)

        # open spread for THIS SIDE LABEL, for drift calc
        open_spread = _get_open_spread_for_side(open_df, matchup_val, side_txt)
        if open_spread is None:
            line_drift = 0.0
        else:
            line_drift = abs(_safe_float(spread_now, 0.0) - _safe_float(open_spread, 0.0))

        # MFS_raw
        # (100 - HC_val) is disagreement / chaos: bigger = more chaos
        chaos_component = 100.0 - HC_val
        mfs_raw = (0.4 * skew) + (0.4 * chaos_component) + (0.2 * (line_drift * 10.0))

        out_rows.append({
            "Matchup": matchup_val,
            "Team": side_txt,
            "Skew": skew,
            "HC": HC_val,
            "Drift": line_drift,
            "MFS_raw": mfs_raw,
        })

    if not out_rows:
        return pd.DataFrame(columns=cols), {}

    tmp = pd.DataFrame(out_rows)

    # normalize to 0–100
    max_raw = tmp["MFS_raw"].max()
    if max_raw and max_raw > 0:
        tmp["MFS_score"] = 100.0 * (tmp["MFS_raw"] / max_raw)
    else:
        tmp["MFS_score"] = 0.0

    # slate-level stats
    avg_mfs = float(tmp["MFS_score"].mean())
    num_high = int((tmp["MFS_score"] > 90.0).sum())

    # anomalies
    # most lopsided: max Skew
    lop_row = tmp.loc[tmp["Skew"].idxmax()]
    # biggest disagreement: min HC
    dis_row = tmp.loc[tmp["HC"].idxmin()]
    # most volatile line: max Drift
    vol_row = tmp.loc[tmp["Drift"].idxmax()]

    lopsided_desc = f"{lop_row['Team']} ({lop_row['Matchup']}) — {lop_row['Skew']:.1f} skew"
    disagreement_desc = f"{dis_row['Team']} ({dis_row['Matchup']}) — HC {dis_row['HC']:.1f}"
    volatile_desc = f"{vol_row['Team']} ({vol_row['Matchup']}) — drift {vol_row['Drift']:.1f} pts"

    # mood label bucket by avg_mfs
    if avg_mfs <= 25:
        mood_label = "😴 Boring Slate"
    elif avg_mfs <= 45:
        mood_label = "🎉 Public Party"
    elif avg_mfs <= 65:
        mood_label = "😠 Revenge Week"
    elif avg_mfs <= 80:
        mood_label = "⚡ Whiplash Slate"
    else:
        mood_label = "🩸 Blood Week Incoming"

    meta = {
        "avg_mfs": avg_mfs,
        "num_high": num_high,
        "mood_label": mood_label,
        "lopsided_desc": lopsided_desc,
        "disagreement_desc": disagreement_desc,
        "volatile_desc": volatile_desc,
    }

    # final table cleanup / ordering
    table_df = tmp[["Matchup","Team","Skew","HC","Drift","MFS_score"]].copy()
    table_df = table_df.sort_values("MFS_score", ascending=False).reset_index(drop=True)

    return table_df, meta

def _extract_matchup_core_stats(df_match: pd.DataFrame,
                                open_snap_df: pd.DataFrame | None) -> dict | None:
    """
    For a single matchup, pull the core pieces we need for Market Fingerprint:
    - favorite vs underdog
    - open vs now spread on the favorite
    - favorite/underdog handle %
    - favorite/underdog ticket %
    - absolute move and disagreement

    We’ll infer 'favorite' off:
    - spread side with the most negative spread_now
    - OR moneyline side with the most negative odds
    """

    if df_match.empty:
        return None

    matchup_name = str(df_match["matchup"].iloc[0])

    # --- current SPREAD rows for this matchup ---
    spread_rows = df_match[df_match["market"].str.lower().str.startswith("spread")].copy()
    spread_rows["spread_val"] = spread_rows["side"].apply(_parse_spread_from_side)
    spread_rows = spread_rows[spread_rows["spread_val"].notnull()].copy()

    spread_rows["bets_pct"] = spread_rows["%bets"].apply(_safe_float)
    spread_rows["handle_pct"] = spread_rows["%handle"].apply(_safe_float)

    fav_spread_row = None
    dog_spread_row = None
    if not spread_rows.empty:
        # most negative spread -> favorite
        spread_sorted = spread_rows.sort_values("spread_val")
        fav_spread_row = spread_sorted.iloc[0]
        dog_spread_row = spread_sorted.iloc[-1]

    spread_now_fav = _safe_float(fav_spread_row["spread_val"]) if fav_spread_row is not None else None

    # pull OPEN spread for that same side label from line_snapshots('OPEN')
    spread_open_fav = None
    if (
        open_snap_df is not None
        and fav_spread_row is not None
        and isinstance(open_snap_df, pd.DataFrame)
        and not open_snap_df.empty
        and "matchup" in open_snap_df.columns
        and "side" in open_snap_df.columns
    ):
        side_lbl = str(fav_spread_row.get("side", ""))
        open_row = open_snap_df[
            (open_snap_df["matchup"] == matchup_name) &
            (open_snap_df["side"] == side_lbl)
        ]
        if not open_row.empty:
            spread_open_fav = _safe_float(open_row.iloc[0].get("spread"))

    if spread_open_fav is None:
        spread_open_fav = spread_now_fav

    # favorite side split % (handle / bets) from spread view
    fav_spread_handle_pct = _safe_float(fav_spread_row["%handle"]) if fav_spread_row is not None else None
    fav_spread_bets_pct   = _safe_float(fav_spread_row["%bets"])   if fav_spread_row is not None else None

    dog_spread_handle_pct = _safe_float(dog_spread_row["%handle"]) if dog_spread_row is not None else None
    dog_spread_bets_pct   = _safe_float(dog_spread_row["%bets"])   if dog_spread_row is not None else None

    # --- current MONEYLINE rows for this matchup ---
    ml_rows = df_match[df_match["market"].str.lower().str.startswith("money")].copy()
    ml_rows["odds_val"] = ml_rows["odds"].apply(clean_odds)
    ml_rows["team_label"] = ml_rows["side"].apply(_extract_team_from_side_label)
    ml_rows["bets_pct"] = ml_rows["%bets"].apply(_safe_float)
    ml_rows["handle_pct"] = ml_rows["%handle"].apply(_safe_float)

    fav_ml_row = None
    dog_ml_row = None
    if not ml_rows.empty:
        # most negative odds -> favorite
        ml_sorted = ml_rows.sort_values("odds_val")
        fav_ml_row = ml_sorted.iloc[0]
        dog_ml_row = ml_sorted.iloc[-1]

    fav_ml_handle_pct = _safe_float(fav_ml_row["handle_pct"]) if fav_ml_row is not None else None
    fav_ml_bets_pct   = _safe_float(fav_ml_row["bets_pct"])   if fav_ml_row is not None else None
    dog_ml_handle_pct = _safe_float(dog_ml_row["handle_pct"]) if dog_ml_row is not None else None
    dog_ml_bets_pct   = _safe_float(dog_ml_row["bets_pct"])   if dog_ml_row is not None else None

    # Attempt clean team strings from ML rows first (they’re cleaner text)
    fav_team = str(fav_ml_row["team_label"]) if fav_ml_row is not None else None
    dog_team = str(dog_ml_row["team_label"]) if dog_ml_row is not None else None

    # fallback team names from spread labels like "Lions -4.5"
    if fav_team is None and fav_spread_row is not None:
        fav_team = str(fav_spread_row.get("side","")).split()[0]
    if dog_team is None and dog_spread_row is not None:
        dog_team = str(dog_spread_row.get("side","")).split()[0]

    # favorite_handle_pct = max(handle on fav spread side, handle on fav ML side)
    fav_handle_candidates = [fav_spread_handle_pct, fav_ml_handle_pct]
    fav_handle_candidates = [x for x in fav_handle_candidates if x is not None]
    favorite_handle_pct = max(fav_handle_candidates) if fav_handle_candidates else None

    # favorite_ticket_pct = max(bets % on fav spread, bets % on fav ML)
    fav_ticket_candidates = [fav_spread_bets_pct, fav_ml_bets_pct]
    fav_ticket_candidates = [x for x in fav_ticket_candidates if x is not None]
    favorite_ticket_pct = max(fav_ticket_candidates) if fav_ticket_candidates else None

    # underdog_handle_pct for spotlight
    underdog_handle_pct = None
    for v in [dog_spread_handle_pct, dog_ml_handle_pct]:
        if v is None:
            continue
        if underdog_handle_pct is None or v > underdog_handle_pct:
            underdog_handle_pct = v

    # absolute disagreement between public tickets and handle on fav
    disagreement_abs = None
    if favorite_handle_pct is not None and favorite_ticket_pct is not None:
        disagreement_abs = abs(favorite_handle_pct - favorite_ticket_pct)

    # absolute move from open to now on the favorite
    move_abs = None
    if spread_open_fav is not None and spread_now_fav is not None:
        move_abs = abs(spread_now_fav - spread_open_fav)

    return {
        "matchup": matchup_name,
        "favorite_team": fav_team,
        "underdog_team": dog_team,
        "spread_open_fav": spread_open_fav,
        "spread_now_fav": spread_now_fav,
        "favorite_handle_pct": favorite_handle_pct,
        "favorite_ticket_pct": favorite_ticket_pct,
        "underdog_handle_pct": underdog_handle_pct,
        "move_abs": move_abs,
        "disagreement_abs": disagreement_abs,
    }


def build_market_fingerprint_json(df: pd.DataFrame,
                                  open_snap_df: pd.DataFrame | None,
                                  slate_week: int | None = None
                                  ) -> dict:
    """
    Build the full MARKET FINGERPRINT JSON contract.

    You will hand this JSON directly to your image generator.

    Fields match the spec:
    - mood_label / mood_description
    - favorite_bias_pct / steam_aggression_pct / prop_obsession_pct
    - spotlight bullets (lopsided / move / sharp_conflict)
    - narrator_line
    """

    # if no data, return an "empty shell" with required keys
    def _empty_payload():
        return {
            "template": "market_fingerprint_v1",
            "title": "NFL MARKET FINGERPRINT",
            "slate_week": slate_week,
            "slate_timestamp_et": datetime.utcnow().isoformat(),

            "mood_label": "",
            "mood_description": "",

            "favorite_bias_pct": 0,
            "prop_obsession_pct": None,
            "steam_aggression_pct": 0,

            "most_lopsided_game": "",
            "most_lopsided_statline": "",

            "biggest_line_move_game": "",
            "biggest_line_move_detail": "",

            "sharp_conflict_game": "",
            "sharp_conflict_detail": "",

            "narrator_line": ""
        }

    if df is None or df.empty:
        return _empty_payload()

    # build per-game stats
    games = []
    for matchup_val, sub in df.groupby("matchup"):
        core = _extract_matchup_core_stats(sub, open_snap_df)
        if core:
            games.append(core)

    if not games:
        return _empty_payload()

    total_games = len(games)

    # FAVORITE BIAS %
    chalk_games = 0
    for g in games:
        fh = g.get("favorite_handle_pct")
        if fh is not None and fh > 60.0:
            chalk_games += 1
    favorite_bias_pct = int(round((chalk_games / total_games) * 100.0)) if total_games > 0 else 0

    # STEAM AGGRESSION %
    moved_games = 0
    for g in games:
        mv = g.get("move_abs")
        if mv is not None and mv >= 1.0:
            moved_games += 1
    steam_aggression_pct = int(round((moved_games / total_games) * 100.0)) if total_games > 0 else 0

    # PROP OBSESSION %
    # you’re not classifying props vs sides/totals yet → set None, never drop key
    prop_obsession_pct = None

    # CHAOS / MOOD SCORE
    chaos_raw_list = []
    chaos_pairs = []  # (g, chaos_raw)

    for g in games:
        fav_handle = g.get("favorite_handle_pct")
        fav_tickets = g.get("favorite_ticket_pct", fav_handle)
        if fav_tickets is None:
            fav_tickets = fav_handle

        chalk_weight = 0.0
        if fav_handle is not None and fav_tickets is not None:
            chalk_weight = max(fav_handle, fav_tickets) - 50.0

        move_abs = g.get("move_abs") or 0.0
        movement_weight = abs(move_abs) * 10.0

        disagreement_abs = g.get("disagreement_abs") or 0.0
        disagreement_weight = abs(disagreement_abs)

        chaos_raw = chalk_weight + movement_weight + disagreement_weight
        chaos_raw_list.append(chaos_raw)
        chaos_pairs.append((g, chaos_raw))

    if chaos_raw_list:
        max_chaos = max(chaos_raw_list)
        chaos_scores = []
        for (_, raw_val) in chaos_pairs:
            if max_chaos > 0:
                chaos_scores.append((raw_val / max_chaos) * 100.0)
            else:
                chaos_scores.append(0.0)
        avg_mood_score = sum(chaos_scores) / len(chaos_scores)
    else:
        avg_mood_score = 0.0

    # map avg_mood_score → mood_label + mood_description
    if avg_mood_score <= 25:
        mood_label = "SLEEP WALK"
        mood_description = "Low movement, public calm, no panic. Books relaxed."
    elif avg_mood_score <= 45:
        mood_label = "PUBLIC PARTY"
        mood_description = "Everybody thinks they’re smart and they’re mostly aligned. Chalk Sunday."
    elif avg_mood_score <= 65:
        mood_label = "REVENGE WEEK"
        mood_description = "There’s memory money. Bettors are trying to get it back from last week."
    elif avg_mood_score <= 80:
        mood_label = "WHIPLASH SLATE"
        mood_description = "Market is fighting with itself. Lines are jumping. Everybody thinks they’re early."
    else:
        mood_label = "BLOOD WEEK"
        mood_description = "Public and book are in open conflict. Nothing is safe."

    # SPOTLIGHT 1: Most lopsided public side
    most_lop_val = -1.0
    most_lop_game = ""
    most_lop_team = ""
    for g in games:
        fav_handle_pct = g.get("favorite_handle_pct")
        dog_handle_pct = g.get("underdog_handle_pct")
        for side, pct in [("fav", fav_handle_pct), ("dog", dog_handle_pct)]:
            if pct is None:
                continue
            if pct > most_lop_val:
                most_lop_val = pct
                most_lop_game = g.get("matchup","")
                most_lop_team = g.get("favorite_team") if side == "fav" else g.get("underdog_team")
    most_lopsided_game = most_lop_game
    most_lopsided_statline = (
        f"{most_lop_val:.0f}% of money on {most_lop_team} ML" if most_lop_val >= 0 else ""
    )

    # SPOTLIGHT 2: Biggest line move
    biggest_mv = -1.0
    biggest_line_move_game = ""
    biggest_line_move_detail = ""
    for g in games:
        mv = g.get("move_abs")
        if mv is None:
            continue
        if mv > biggest_mv:
            biggest_mv = mv
            biggest_line_move_game = g.get("matchup","")
            so = g.get("spread_open_fav")
            sn = g.get("spread_now_fav")
            try:
                biggest_line_move_detail = f"{so:+g} → {sn:+g} on {g.get('favorite_team')}"
            except Exception:
                biggest_line_move_detail = f"{so} → {sn} on {g.get('favorite_team')}"

    # SPOTLIGHT 3: Sharp conflict (handle vs tickets split)
    sharp_conflict_val = -1.0
    sharp_conflict_game = ""
    sharp_conflict_detail = ""
    for g in games:
        disc = g.get("disagreement_abs")
        if disc is None:
            continue
        if disc > sharp_conflict_val:
            sharp_conflict_val = disc
            sharp_conflict_game = g.get("matchup","")
            fh = g.get("favorite_handle_pct")
            ft = g.get("favorite_ticket_pct")
            if fh is not None and ft is not None:
                sharp_conflict_detail = (
                    f"{ft:.0f}% of tickets on {g.get('favorite_team')}, "
                    f"but {fh:.0f}% of handle"
                )

    # Narrator line (tone depends on chaos)
    if mood_label in ["WHIPLASH SLATE", "BLOOD WEEK", "REVENGE WEEK"]:
        narrator_line = "This slate is ego money. Everyone thinks they’re early. Most of them are late."
    else:
        narrator_line = "Public thinks it’s easy. Books disagree."

    return {
        "template": "market_fingerprint_v1",
        "title": "NFL MARKET FINGERPRINT",
        "slate_week": slate_week,
        "slate_timestamp_et": datetime.utcnow().isoformat(),

        "mood_label": mood_label,
        "mood_description": mood_description,

        "favorite_bias_pct": favorite_bias_pct,
        "prop_obsession_pct": prop_obsession_pct,
        "steam_aggression_pct": steam_aggression_pct,

        "most_lopsided_game": most_lopsided_game,
        "most_lopsided_statline": most_lopsided_statline,

        "biggest_line_move_game": biggest_line_move_game,
        "biggest_line_move_detail": biggest_line_move_detail,

        "sharp_conflict_game": sharp_conflict_game,
        "sharp_conflict_detail": sharp_conflict_detail,

        "narrator_line": narrator_line
    }

# ============================================================================
# HANDLE GRAVITY CARD RENDERING (SOCIAL ASSET GENERATOR)
# ============================================================================

def _pick_font(size: int, bold: bool = False):
    """
    Try to load a normal/bold system font. If we can't (we're in some weird env),
    fall back to PIL's default bitmap font so Streamlit doesn't crash.
    """
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/HelveticaNeueDeskInterface-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/HelveticaNeueDeskInterface-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_handle_gravity_card(row: dict) -> Image.Image:
    """
    Build a 1080x1350 IG-style static card for Handle Gravity using one matchup row.

    Expected keys in `row`:
      matchup
      team_leader
      gravity_leader
      public_leader        (0-1 decimal)
      odds_leader
      team_other
      gravity_other
      public_other         (0-1 decimal)
      odds_other
    """
    W, H = 1080, 1350
    BG = (10, 10, 12)           # black/near-black
    FG_WHITE = (240, 240, 240)  # off-white
    FG_SUB = (180, 180, 180)    # gray
    NEON = (0, 255, 120)        # neon green
    CIRCLE_BG = (25, 25, 30)

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Fonts
    title_font    = _pick_font(54, bold=True)
    sub_font      = _pick_font(32, bold=False)
    matchup_font  = _pick_font(36, bold=False)
    team_font     = _pick_font(44, bold=True)
    mult_font     = _pick_font(88, bold=True)
    footer_font   = _pick_font(34, bold=True)

    # Header
    header_text = "TONIGHT'S HANDLE GRAVITY"
    header_sub  = "WHO AMERICA IS RIDING"

    draw.text((60, 60), header_text, fill=FG_WHITE, font=title_font)
    draw.text((60, 60 + 60), header_sub,  fill=FG_SUB,   font=sub_font)

    matchup_str = str(row.get("matchup","")).upper()
    draw.text((60, 60 + 60 + 50), matchup_str, fill=FG_WHITE, font=matchup_font)

    # Leader side (left) vs other side (right)
    left_team   = str(row.get("team_leader","")).upper()
    right_team  = str(row.get("team_other","")).upper()

    left_mult   = f"{row.get('gravity_leader',0):.2f}×"
    right_mult  = f"{row.get('gravity_other',0):.2f}×"

    left_pub_pct  = round(float(row.get("public_leader",0)) * 100.0, 1)
    right_pub_pct = round(float(row.get("public_other",0)) * 100.0, 1)

    left_odds   = row.get("odds_leader","")
    right_odds  = row.get("odds_other","")

    # Circle geometry
    circle_d = 400
    left_cx, left_cy   = 300, 650
    right_cx, right_cy = 780, 650

    def _circle(xc, yc, d, team_txt, mult_txt, pct_txt, odds_txt):
        bbox = [xc - d//2, yc - d//2, xc + d//2, yc + d//2]
        draw.ellipse(bbox, fill=CIRCLE_BG, outline=FG_WHITE, width=4)

        # Team label
        bbox_team = draw.textbbox((0, 0), team_txt, font=team_font)
        tw, th = bbox_team[2] - bbox_team[0], bbox_team[3] - bbox_team[1]
        draw.text((xc - tw/2, yc - d//2 + 40), team_txt, fill=FG_WHITE, font=team_font)

        # Gravity multiplier big in the middle
        bbox_mult = draw.textbbox((0, 0), mult_txt, font=mult_font)
        mw, mh = bbox_mult[2] - bbox_mult[0], bbox_mult[3] - bbox_mult[1]
        draw.text((xc - mw/2, yc - mh/2), mult_txt, fill=FG_WHITE, font=mult_font)

        # Subtext
        sub_line = f"{pct_txt}% of public money on {odds_txt}"
        bbox_sub = draw.textbbox((0, 0), sub_line, font=sub_font)
        sw, sh = bbox_sub[2] - bbox_sub[0], bbox_sub[3] - bbox_sub[1]
        draw.text((xc - sw/2, yc + mh/2 + 10), sub_line, fill=FG_SUB, font=sub_font)

    _circle(
        left_cx, left_cy, circle_d,
        left_team,
        left_mult,
        left_pub_pct,
        left_odds
    )
    _circle(
        right_cx, right_cy, circle_d,
        right_team,
        right_mult,
        right_pub_pct,
        right_odds
    )

    # Footer strip
    footer_h  = 180
    footer_y0 = H - footer_h
    draw.rectangle([0, footer_y0, W, H], fill=NEON)

    punch = (
        f"Public is {row.get('gravity_leader',0):.2f}× heavier on {left_team.title()} "
        "than the odds say they should be."
    )
    bbox_footer = draw.textbbox((0, 0), punch, font=footer_font)
    pw, ph = bbox_footer[2] - bbox_footer[0], bbox_footer[3] - bbox_footer[1]
    draw.text(
        ((W - pw)/2, footer_y0 + (footer_h - ph)/2),
        punch,
        fill=(0,0,0),
        font=footer_font
    )

    return img


def export_handle_gravity_card(summary_df: pd.DataFrame,
                               log_rows: list[dict]) -> tuple[bytes, dict] | None:
    """
    Take the Handle Gravity summary_df (top 5 leaderboard) plus the raw log_rows
    (with both sides of each matchup) and build a social card for the #1 game.

    Returns (png_bytes, meta) or None.
    meta = {
        "matchup": ...,
        "team": ...,
        "gravity_x": ...,
    }
    """
    if summary_df is None or summary_df.empty or not log_rows:
        return None

    # Take the top row from summary_df
    top = summary_df.iloc[0]
    matchup_key   = str(top["Matchup"])
    gravity_team  = str(top["Gravity Team"])

    # Find that matchup in raw_rows
    chosen = None
    for r in log_rows:
        # We logged "team_fav" as the leader side label
        if str(r.get("matchup","")) == matchup_key and str(r.get("team_fav","")) == gravity_team:
            chosen = r
            break
        if str(r.get("matchup","")) == matchup_key and str(r.get("team_fav","")).lower() == gravity_team.lower():
            chosen = r
            break
    if chosen is None:
        chosen = log_rows[0]  # fallback

    # Build the payload the renderer expects
    card_payload = {
        "matchup":        chosen.get("matchup"),
        "team_leader":    chosen.get("team_fav"),
        "gravity_leader": chosen.get("gravity_leader"),
        "public_leader":  chosen.get("public_leader"),
        "odds_leader":    chosen.get("odds_leader"),
        "team_other":     chosen.get("team_other"),
        "gravity_other":  chosen.get("gravity_other"),
        "public_other":   chosen.get("public_other"),
        "odds_other":     chosen.get("odds_other"),
    }

    img = render_handle_gravity_card(card_payload)

    # Encode PNG to bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.getvalue()

    meta = {
        "matchup":    chosen.get("matchup"),
        "team":       chosen.get("team_fav"),
        "gravity_x":  round(float(chosen.get("gravity_leader",0)), 2),
    }
    return (png_bytes, meta)

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def update_history(score: float) -> pd.DataFrame:
    """Update mood history."""
    today = datetime.now(pytz.timezone("America/Los_Angeles")).date().isoformat()
    
    try:
        if os.path.exists(HISTORY_FILE):
            hist = pd.read_csv(HISTORY_FILE)
        else:
            hist = pd.DataFrame(columns=["date", "score"])
    except Exception:
        hist = pd.DataFrame(columns=["date", "score"])
    
    if not (hist["date"] == today).any():
        hist = pd.concat([hist, pd.DataFrame([{"date": today, "score": float(score)}])], ignore_index=True)
        try:
            hist.to_csv(HISTORY_FILE, index=False)
        except Exception:
            pass
    
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")
    
    return hist.tail(7)[["date", "score"]]

def market_mood_ring(df):
    """Display mood trend chart."""
    df["irrationality"] = abs(df["%bets"] - df["%handle"])
    mood_score = df["irrationality"].mean()
    hist = update_history(mood_score)
    
    sparkline = (
        alt.Chart(hist)
        .mark_line(point=True, color="#ff3c78")
        .encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('score:Q', title='Irrationality %', scale=alt.Scale(domain=[0, 30]))
        )
        .properties(width=700, height=120)
    )
    
    st.subheader("📈 7-Day Mood Trend")
    st.altair_chart(sparkline, use_container_width=True)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    .css-1lcbmhc {background: linear-gradient(90deg, #0e0f11, #1f1f23);}
    .stMetric {
        border: 2px solid #ff3c78; 
        border-radius: 8px; 
        padding: 12px; 
        background-color: #1a1a1d;
    }
    .stProgress > div > div > div > div {background: #ff3c78;}
    .streamlit-expanderHeader {
        background-color: #1f1f23 !important;
        border-radius: 4px;
        padding: 8px;
    }
    #MainMenu, footer {visibility: hidden;}
    .nfl-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
        margin: 2px;
    }
    .wong-badge {background: #ff3c78; color: white;}
    .key-badge {background: #ffa500; color: black;}
    .signal-badge {background: #00ff00; color: black;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.title("🏈 NFL Controls")

slate = st.sidebar.selectbox("Slate Window", ["today", "tomorrow", "n7days"], index=2)
custom_eg = st.sidebar.number_input("Override Event Group (0 = auto)", min_value=0, step=1, value=0)

refresh_interval = st.sidebar.slider(
    "Auto-refresh (sec)", min_value=15, max_value=3600, value=900, step=5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Signal Thresholds")

EXPOSURE_THRESHOLD = st.sidebar.slider(
    "Exposure Liability ($)", min_value=0, max_value=15000, value=3000, step=500
)
ODDLOT_THRESHOLD = st.sidebar.slider(
    "Odd-Lot Divergence (%)", min_value=0.0, max_value=0.5, value=0.12, step=0.01, format="%.2f"
)
CORR_THRESHOLD = st.sidebar.slider(
    "Correlation Breach (%)", min_value=0.0, max_value=0.1, value=0.025, step=0.005, format="%.3f"
)
RLM_BET_THRESHOLD = st.sidebar.slider(
    "RLM % Bets Δ", min_value=0.0, max_value=25.0, value=6.0, step=0.5, format="%.1f"
)
STEAM_ODDS_THRESHOLD = st.sidebar.slider(
    "Steam Odds Δ", min_value=0, max_value=15, value=3, step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Edge Weights")

st.session_state["w_ev"] = st.sidebar.slider(
    "EV Weight", 0.0, 1.0, st.session_state.get("w_ev", DEFAULT_W_EV), 0.05
)
st.session_state["w_irr"] = st.sidebar.slider(
    "Irrationality Weight", 0.0, 1.0, st.session_state.get("w_irr", DEFAULT_W_IRR), 0.05
)
st.session_state["w_rlm"] = st.sidebar.slider(
    "RLM Weight", 0.0, 1.0, st.session_state.get("w_rlm", DEFAULT_W_RLM), 0.05
)
st.session_state["w_steam"] = st.sidebar.slider(
    "Steam Weight", 0.0, 1.0, st.session_state.get("w_steam", DEFAULT_W_STEAM), 0.05
)
st.session_state["kelly_cap"] = st.sidebar.slider(
    "Kelly Cap (% bankroll)", 0.0, 0.20, st.session_state.get("kelly_cap", DEFAULT_KELLY_CAP), 0.01, format="%.2f"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Alerts")

st.session_state["edge_alert_min"] = st.sidebar.slider(
    "Min Edge Score", 0, 100, int(st.session_state.get("edge_alert_min", 70)), 1
)
st.session_state["ev_alert_min"] = st.sidebar.slider(
    "Min EV (units)", -1.0, 2.0, float(st.session_state.get("ev_alert_min", 0.05)), 0.01
)

# ============================================================================
# AUTO-REFRESH
# ============================================================================

refresh_count = st_autorefresh(interval=refresh_interval * 1000, key="nfl_refresh")

# Manual TD prop scrape button
force_td = st.sidebar.button("Force TD Prop Scrape")
generate_gravity_card = st.sidebar.button("Generate Handle Gravity Visual")
generate_gravity_json = st.sidebar.button("Generate Handle Gravity JSON")
generate_pain_json = st.sidebar.button("Generate Book Pain JSON")
generate_resilience_json = st.sidebar.button("Generate Line Resilience JSON")
generate_fingerprint_json = st.sidebar.button("Generate Market Fingerprint JSON")

st.sidebar.markdown("### 🧱 Line Resilience Snapshots")
res_snap_type = st.sidebar.selectbox(
    "Snapshot Label",
    ["OPEN","EARLY","MID","FINAL"],
    index=0
)
capture_resilience_snapshot = st.sidebar.button("Capture Line Resilience Snapshot")

st.sidebar.markdown(f"🔄 Refresh: {refresh_count}")

if "loaded_snapshot" not in st.session_state:
    st.session_state["loaded_snapshot"] = False

# ============================================================================
# MAIN RENDERING
# ============================================================================

def render_overview(df: pd.DataFrame,
                    td_any: pd.DataFrame | None = None,
                    td_first: pd.DataFrame | None = None,
                    td_multi: pd.DataFrame | None = None,
                    td_debug: dict | None = None,
                    capture_resilience_snapshot: bool = False,
                    res_snap_type: str = "FINAL"):
    """Render main overview page."""
    if df.empty:
        st.warning("⚠️ No splits data available for this slate.")
        return

    # Pull OPEN snapshot rows from DB (for RLM logic)
    try:
        with sqlite3.connect(BETLOGS_DB) as con:
            open_snap = pd.read_sql_query(
                "SELECT matchup, market, side, spread, bets_pct FROM line_snapshots WHERE snapshot_type='OPEN' AND book='DK'",
                con
            )
    except Exception as e:
        logging.error(f"[render_overview] failed to load OPEN snapshots: {e}")
        open_snap = None
    # Optionally capture a new Line Resilience snapshot for this moment (OPEN / EARLY / MID / FINAL)
    if capture_resilience_snapshot:
        try:
            capture_line_resilience_snapshot(df, res_snap_type)
        except Exception as e:
            logging.error(f"[capture_line_resilience_snapshot] {e}")

    # Build current Line Resilience Index table from stored snapshots
    resilience_df = compute_line_resilience_index()

    # MARKET FINGERPRINT (Full Slate Mood / Weekly Personality)
    fingerprint_df, fingerprint_meta = compute_market_fingerprint(df, open_snap)

    # Build the Market Fingerprint JSON (full-slate ritual card)
    fingerprint_json_payload = build_market_fingerprint_json(
        df,
        open_snap,
        slate_week=None  # you can pass real NFL week once you’re tracking it
    )
    fingerprint_json_selected = fingerprint_json_payload if     generate_fingerprint_json else None

    # Build core viral-style content blocks
    public_warn_spread_df, public_warn_total_df = build_public_betting_warning(df)
    money_div_spread_df, money_div_total_df = build_money_divergence_report(df)
    book_spread_df, book_total_df, book_ml_df = build_book_needs_board(df)
    rlm_df = build_rlm_tracker(df, open_snap)

    # Handle Gravity (moneyline sentiment vs implied odds)
    gravity_summary_df, gravity_log_rows = build_handle_gravity(df)

    # Log Handle Gravity snapshot once per run
    if not st.session_state.get("gravity_logged", False):
        log_handle_gravity_snapshot(gravity_log_rows)
        st.session_state["gravity_logged"] = True

    # Optional Handle Gravity card render (PNG preview)
    gravity_card_imgbytes = None
    gravity_card_meta = None
    if generate_gravity_card:
        card_res = export_handle_gravity_card(gravity_summary_df, gravity_log_rows)
        if card_res is not None:
            gravity_card_imgbytes, gravity_card_meta = card_res

    # Handle Gravity JSON payloads (image contract)
    gravity_json_payloads = build_handle_gravity_json(df)
    gravity_json_selected = gravity_json_payloads[0] if (generate_gravity_json and         gravity_json_payloads) else None

    # Book Pain Bar JSON payloads (image contract)
    pain_json_payloads = build_book_pain_json(df)
    pain_json_selected = pain_json_payloads[0] if (generate_pain_json and     pain_json_payloads) else None

    # ---------------------------------------------------------------------
    # MARKET FINGERPRINT DISPLAY (Top-of-page weekly ritual content)
    # ---------------------------------------------------------------------
    st.markdown("## 🩸 Market Fingerprint — Full Slate Mood")

    # If the sidebar button was pressed, surface the Market Fingerprint JSON + download
    if generate_fingerprint_json and fingerprint_json_selected:
        st.markdown("#### Market Fingerprint JSON (Spec Contract)")
        st.json(fingerprint_json_selected)

        mf_json_bytes = json.dumps(fingerprint_json_selected, indent=2).encode("utf-8")
        st.download_button(
            "📥 Download Market Fingerprint JSON",
            data=mf_json_bytes,
            file_name="market_fingerprint_payload.json",
            mime="application/json"
        )

    if fingerprint_df.empty or not fingerprint_meta:
        st.caption("Not enough data yet to fingerprint the slate.")
    else:
        mood_label = fingerprint_meta.get("mood_label","Slate Mood")
        avg_mfs = fingerprint_meta.get("avg_mfs", 0.0)
        num_high = fingerprint_meta.get("num_high", 0)

        lopsided_desc = fingerprint_meta.get("lopsided_desc","")
        disagreement_desc = fingerprint_meta.get("disagreement_desc","")
        volatile_desc = fingerprint_meta.get("volatile_desc","")

        st.markdown(
            f"**{mood_label}**  \n"
            f"Avg MFS: {avg_mfs:.1f} | {num_high} games > 90 MFS (chaos)\n\n"
            f"**Most lopsided:** {lopsided_desc}  \n"
            f"**Biggest fight:** {disagreement_desc}  \n"
            f"**Most volatile:** {volatile_desc}"
        )

        st.dataframe(
            fingerprint_df.head(10),
            use_container_width=True
        )

    st.markdown("---")

    # Display the four blocks (tables will double as screenshot-ready content for social)
    st.markdown("## 📢 NFL Market Pulse (Public vs Book)")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 🚨 Public Betting Warning")
        tab_pb_spread, tab_pb_total = st.tabs(["Spread", "Total"])

        with tab_pb_spread:
            if public_warn_spread_df.empty:
                st.caption("No public-heavy spreads above threshold.")
            else:
                st.dataframe(public_warn_spread_df, use_container_width=True)

        with tab_pb_total:
            if public_warn_total_df.empty:
                st.caption("No public-heavy totals above threshold.")
            else:
                st.dataframe(public_warn_total_df, use_container_width=True)

        # Public Betting Warning Image (Spreads / Totals / Combo)
        with st.expander("📸 Public Betting Warning Image", expanded=False):
            if public_warn_spread_df.empty and public_warn_total_df.empty:
                st.caption("Not enough public-overexposed markets to build an image yet.")
            else:
                # Build row sets for template rendering (spread / total / combo)
                rows_combo = build_public_warning_rows_from_tables(
                    public_warn_spread_df,
                    public_warn_total_df,
                    None,
                    max_rows=4,
                )
                rows_spreads_only = build_public_warning_rows_from_tables(
                    public_warn_spread_df,
                    None,
                    None,
                    max_rows=4,
                )
                rows_totals_only = build_public_warning_rows_from_tables(
                    None,
                    public_warn_total_df,
                    None,
                    max_rows=4,
                )

                source_choice = st.selectbox(
                    "Use which public list for the image?",
                    [
                        "Spreads and totals",
                        "Spreads only",
                        "Totals only",
                    ],
                    key="public_warning_nfl_source_choice",
                )

                if source_choice == "Spreads only":
                    warning_rows = rows_spreads_only
                elif source_choice == "Totals only":
                    warning_rows = rows_totals_only
                else:
                    warning_rows = rows_combo

                if not warning_rows:
                    st.warning(
                        "Not enough public-overexposed sides in the selected category to build an image right now."
                    )
                else:
                    generate_btn = st.button(
                        "Render Public Betting Warning Image",
                        key="render_public_warning_image_nfl",
                    )
                    if generate_btn:
                        try:
                            img = render_public_betting_template(warning_rows)
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            img_bytes = buf.getvalue()

                            st.image(img_bytes, use_container_width=True)
                            st.download_button(
                                label="Download Public Betting Warning Image",
                                data=img_bytes,
                                file_name="nfl_public_betting_warning.png",
                                mime="image/png",
                            )
                        except FileNotFoundError:
                            st.error(
                                f"Template image '{PUBLIC_WARNING_TEMPLATE_PATH}' not found. "
                                "Place the FoxEdge public_betting.png file in the app working directory."
                            )
                        except Exception as e:
                            st.error(f"Failed to render public betting warning image: {e}")

    with colB:
        st.markdown("### 💸 Money Divergence (Sharp vs Public)")
        tab1, tab2 = st.tabs(["Spread", "Total"])

        with tab1:
            if money_div_spread_df.empty:
                st.caption("No major handle/bet splits on spreads.")
            else:
                st.dataframe(money_div_spread_df, use_container_width=True)

        with tab2:
            if money_div_total_df.empty:
                st.caption("No major handle/bet splits on totals.")
            else:
                st.dataframe(money_div_total_df, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("### 🧾 Book Needs Board")
        tab_spread, tab_total, tab_ml = st.tabs(["Spread", "Total", "Moneyline"])

        with tab_spread:
            if book_spread_df.empty:
                st.caption("No obvious spread liability yet.")
            else:
                st.dataframe(book_spread_df, use_container_width=True)

        with tab_total:
            if book_total_df.empty:
                st.caption("No obvious total liability yet.")
            else:
                st.dataframe(book_total_df, use_container_width=True)

        with tab_ml:
            if book_ml_df.empty:
                st.caption("No obvious moneyline liability yet.")
            else:
                st.dataframe(book_ml_df, use_container_width=True)

    with colD:
        st.markdown("### 🌍 Handle Gravity (Moneyline Sentiment)")
        if gravity_summary_df.empty:
            st.caption("No moneyline sentiment strong enough to rank.")
        else:
            st.dataframe(gravity_summary_df, use_container_width=True)

        # If we rendered a Handle Gravity card, show it and allow download
        if generate_gravity_card and gravity_card_imgbytes:
            st.markdown("#### Handle Gravity Card Preview")
            st.image(
                gravity_card_imgbytes,
                caption=f"Handle Gravity: {gravity_card_meta['team']} {gravity_card_meta['gravity_x']}× ({gravity_card_meta['matchup']})",
                use_column_width=True
            )
            st.download_button(
                "📥 Download Handle Gravity PNG",
                data=gravity_card_imgbytes,
                file_name=f"handle_gravity_{gravity_card_meta['team'].replace(' ','_')}.png",
                mime="image/png"
            )

        # If user hit the JSON button, expose the Handle Gravity JSON contract
        if generate_gravity_json:
            if not gravity_json_payloads:
                st.warning("No valid Handle Gravity data to export as JSON.")
            else:
                st.markdown("#### Handle Gravity JSON (Spec Contract)")
                st.json(gravity_json_selected)

                hg_json_bytes = json.dumps(gravity_json_selected, indent=2).encode("utf-8")
                st.download_button(
                    "📥 Download Handle Gravity JSON",
                    data=hg_json_bytes,
                    file_name="handle_gravity_payload.json",
                    mime="application/json"
                )

        # --- BOOK PAIN BAR ("If THIS hits, the house hurts.") ---
        st.subheader("Book Pain Bar — 'If THIS hits, the house hurts.'")

        pain_card_res = export_book_pain_card(df)
        if isinstance(pain_card_res, dict) and ("error" not in pain_card_res):
            st.image(
                pain_card_res["img"],
                caption="Book Pain Bar",
                use_column_width=True
            )
            st.download_button(
                label="📥 Download Book Pain Bar PNG",
                data=pain_card_res["png_bytes"],
                file_name="book_pain_bar.png",
                mime="image/png"
            )
        else:
            st.warning(pain_card_res.get("error", "Book Pain Bar unavailable"))

        # If user hit the JSON button, expose the Book Pain Bar JSON contract
        if generate_pain_json:
            if not pain_json_payloads:
                st.warning("No valid Book Pain data to export as JSON.")
            else:
                st.markdown("#### Book Pain Bar JSON (Spec Contract)")
                st.json(pain_json_selected)

                pain_json_bytes = json.dumps(pain_json_selected, indent=2).encode("utf-8")
                st.download_button(
                    "📥 Download Book Pain Bar JSON",
                    data=pain_json_bytes,
                    file_name="book_pain_bar_payload.json",
                    mime="application/json"
                )

    # --- LINE RESILIENCE ("Which numbers are real vs theater") ---
    st.subheader("Line Resilience — Which numbers are real")

    if resilience_df.empty:
        st.caption("Not enough snapshots (need OPEN / EARLY / FINAL spread logs in sqlite).")
    else:
        st.dataframe(
            resilience_df[[
                "game",
                "open_spread_fav",
                "early_spread_fav",
                "final_spread_fav",
                "early_resilience_score",
                "late_resilience_score",
                "label_early",
                "label_late",
                "note",
            ]],
            use_container_width=True
        )

    if generate_resilience_json:
        st.markdown("#### Line Resilience JSON (Spec Contract)")
        st.json(resilience_json_selected)

        lr_json_bytes = json.dumps(resilience_json_selected, indent=2).encode("utf-8")
        st.download_button(
            "📥 Download Line Resilience JSON",
            data=lr_json_bytes,
            file_name="line_resilience_payload.json",
            mime="application/json"
        )

        st.markdown("### 🔄 Reverse Line Movement Tracker")
        if rlm_df.empty:
            st.caption("No clear RLM signals (1+ pt against public).")
        else:
            st.dataframe(rlm_df, use_container_width=True)

        # --- LINE RESILIENCE INDEX (BOOK DEFENSE VS PUBLIC FLOW) ---
        st.markdown("### 🧱 Line Resilience Index (Book Defense)")
        if resilience_df.empty:
            st.caption("Not enough snapshots yet. Capture OPEN / EARLY / FINAL to enable.")
        else:
            st.dataframe(resilience_df, use_container_width=True)

    # TD Scorer popularity boards (Anytime / First TD / 2+ TD)
    st.markdown("### 🏆 Most Bet TD Scorers")

    col_td1, col_td2, col_td3 = st.columns(3)

    with col_td1:
        st.markdown("**Anytime TD**")
        if td_any is None or td_any.empty:
            st.caption("No data.")
        else:
            st.dataframe(td_any, use_container_width=True)

    with col_td2:
        st.markdown("**First TD**")
        if td_first is None or td_first.empty:
            st.caption("No data.")
        else:
            st.dataframe(td_first, use_container_width=True)

    with col_td3:
        st.markdown("**2+ TDs**")
        if td_multi is None or td_multi.empty:
            st.caption("No data.")
        else:
            st.dataframe(td_multi, use_container_width=True)

    # Debug panel so we can see if DK blocked raw HTML in Streamlit env
    with st.expander("TD Prop Debug", expanded=False):
        if td_debug is None:
            st.write({})
        else:
            st.write(td_debug)

    st.markdown("---")

    # Calculate metrics
    df["irrationality"] = (df["%bets"] - df["%handle"]).abs()
    mood = df["irrationality"].mean()

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Avg Irrationality", f"{mood:.1f}%")
    with col2:
        st.metric("🏈 Games Tracked", f"{df['matchup'].nunique()}")
    with col3:
        st.metric("📈 Total Markets", f"{len(df)}")
    with col4:
        st.metric("🎯 Edge Threshold", f"{int(st.session_state.get('edge_alert_min', 70))}")

    # Mood trend
    market_mood_ring(df)

    # Build edge scores
    scored = build_edge_scores(df)

    # Log snapshots
    if not st.session_state.get("loaded_snapshot", False):
        log_snapshot(scored)
        st.session_state["loaded_snapshot"] = True

    log_close_snapshot(scored)
    clv = compute_and_log_clv()

    # ----------------------------------------------------------------------
    # Line Evolution: OPEN → WEEK_OPEN → MIDWEEK → CLOSE
    # ----------------------------------------------------------------------
    st.subheader("Line Evolution: Open → Midweek → Close")

    evo_df = load_line_evolution()
    if evo_df.empty:
        st.info(
            "No line evolution data yet. The app needs to be opened on:\n"
            "- Tuesday morning (for WEEK_OPEN)\n"
            "- Thursday morning (for MIDWEEK)\n"
            "- Near close (for CLOSE via CLV)"
        )
    else:
        # Allow filtering by matchup
        matchups = sorted([m for m in evo_df["matchup"].dropna().unique()])
        selected_matchup = st.selectbox("Select matchup", ["All"] + matchups)

        if selected_matchup == "All":
            to_show = evo_df.copy()
        else:
            to_show = evo_df[evo_df["matchup"] == selected_matchup].copy()

        # Dynamic list of columns to show so it doesn't break if some snapshot types
        # are missing for a given week/slate.
        base_cols = ["matchup", "market", "side"]
        metric_cols = [
            "odds_OPEN", "odds_WEEK_OPEN", "odds_MIDWEEK", "odds_CLOSE",
            "spread_OPEN", "spread_WEEK_OPEN", "spread_MIDWEEK", "spread_CLOSE",
            "total_OPEN", "total_WEEK_OPEN", "total_MIDWEEK", "total_CLOSE",
        ]
        cols_to_show = base_cols + [c for c in metric_cols if c in to_show.columns]

        to_show = to_show[cols_to_show].sort_values(["matchup", "market", "side"])

        st.dataframe(to_show, use_container_width=True)

    # CLV Analysis
    with st.expander("📉 Closing Line Value (CLV) Analysis", expanded=False):
        if clv is None or clv.empty:
            st.info("CLV data not yet available. Make picks to start tracking.")
        else:
            display_cols = [
                "matchup", "market", "side", "edge_badge", "clv_primary",
                "elasticity_primary", "bets_delta", "handle_delta",
                "crossed_3", "crossed_7", "wong_tag_entry", "lost_wong_value",
                "half_point_value"
            ]
            clv_display = clv[display_cols].sort_values(
                ["edge_badge", "clv_primary"],
                ascending=[True, False]
            )
            st.dataframe(clv_display, use_container_width=True)

            # CLV Summary stats
            avg_clv = clv["clv_primary"].mean()
            wins = len(clv[clv["clv_primary"] > 0])
            total = len(clv)

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg CLV", f"{avg_clv:.3f}" if pd.notna(avg_clv) else "N/A")
            col2.metric("Positive CLV", f"{wins}/{total}")
            col3.metric("Win Rate", f"{(wins/total*100):.1f}%" if total > 0 else "N/A")

    # Recommendations
    st.markdown("---")
    st.subheader("🎯 Recommended NFL Angles")

    rec = recommend_bets(scored)

    if rec.empty:
        st.info("No angles currently meet your thresholds. Adjust settings or wait for better spots.")
    else:
        # Log recommendations
        for _, r in rec.iterrows():
            ou = tot = spr = None
            mkt = r["market"].lower()

            if mkt.startswith("total"):
                ou, tot = _parse_total_from_side(r["side"])
            if mkt.startswith("spread"):
                spr = _parse_spread_from_side(r["side"])

            log_your_rec_snapshot(
                r["matchup"], r["market"], r["side"],
                odds_val=r["odds"], total_val=tot, spread_val=spr,
                bets_pct=r.get("%bets"), handle_pct=r.get("%handle")
            )

        # Display recommendations
        for idx, r in rec.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                with col1:
                    st.markdown(f"**{r['matchup']}**")
                    st.caption(f"{r['market']}: {r['side']}")

                    # Add badges
                    badges = []
                    if r.get('wong_tag'):
                        badges.append(f'<span class="nfl-badge wong-badge">{r["wong_tag"]}</span>')
                    if len(r.get('situations', [])) > 0:
                        for sit in r['situations']:
                            badges.append(f'<span class="nfl-badge key-badge">{sit}</span>')
                    if r.get('rlm', 0) > 0.5:
                        badges.append('<span class="nfl-badge signal-badge">RLM</span>')
                    if r.get('steam', 0) > 0.5:
                        badges.append('<span class="nfl-badge signal-badge">STEAM</span>')

                    if badges:
                        st.markdown(" ".join(badges), unsafe_allow_html=True)

                with col2:
                    st.metric("Edge Score", f"{r['edge_score']:.1f}")
                    st.caption(f"EV: {r['ev_units']:.3f}u")

                with col3:
                    st.metric("Kelly Stake", f"{r['stake_pct']:.2f}%")
                    st.caption(f"Odds: {r['odds']}")

                with col4:
                    st.metric("Public Split", f"{r['%bets']:.0f}%")
                    st.caption(f"Handle: {r['%handle']:.0f}%")

                st.markdown("---")

        # Export recommendations
        view = rec[[
            "matchup", "market", "side", "odds", "%bets", "%handle",
            "irrationality", "p_true", "ev_units", "edge_score",
            "stake_pct", "rlm", "steam", "wong_tag", "situations"
        ]].rename(columns={
            "%bets": "bets_%",
            "%handle": "handle_%",
            "p_true": "true_prob_est"
        })

        csv = view.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Recommendations CSV",
            csv,
            file_name=f"foxedge_nfl_recs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    # Market distributions
    st.markdown("---")
    st.subheader("📊 Market Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Irrationality Distribution**")
        irr_hist = (
            alt.Chart(scored)
            .mark_bar(color="#ff3c78")
            .encode(
                x=alt.X("irrationality:Q", bin=alt.Bin(maxbins=25), title="Irrationality %"),
                y=alt.Y("count()", title="Count")
            )
            .properties(height=250)
        )
        st.altair_chart(irr_hist, use_container_width=True)

    with col2:
        st.markdown("**Expected Value Distribution**")
        ev_hist = (
            alt.Chart(scored)
            .mark_bar(color="#00ff88")
            .encode(
                x=alt.X("ev_units:Q", bin=alt.Bin(maxbins=25), title="EV (units)"),
                y=alt.Y("count()", title="Count")
            )
            .properties(height=250)
        )
        st.altair_chart(ev_hist, use_container_width=True)

    # NFL-specific insights
    st.markdown("---")
    st.subheader("🏈 NFL Key Number Analysis")

    spread_markets = scored[scored["market"].str.lower() == "spread"].copy()

    if not spread_markets.empty:
        col1, col2, col3 = st.columns(3)

        with col1:
            key_3_count = spread_markets["spread"].apply(lambda x: abs(x) if x else 0).between(2.5, 3.5).sum()
            st.metric("Lines Near Key 3", key_3_count)

        with col2:
            key_7_count = spread_markets["spread"].apply(lambda x: abs(x) if x else 0).between(6.5, 7.5).sum()
            st.metric("Lines Near Key 7", key_7_count)

        with col3:
            wong_count = spread_markets["wong_tag"].notna().sum()
            st.metric("Wong Teaser Spots", wong_count)

        # Spread distribution chart
        spread_dist = (
            alt.Chart(spread_markets)
            .mark_bar(color="#ffa500")
            .encode(
                x=alt.X("spread:Q", bin=alt.Bin(step=0.5), title="Spread Line"),
                y=alt.Y("count()", title="Count")
            )
            .properties(height=200)
        )
        st.altair_chart(spread_dist, use_container_width=True)

    # Raw data table
    with st.expander("🔍 Raw Splits Data", expanded=False):
        display_df = scored[[
            "matchup", "market", "side", "odds", "spread",
            "%bets", "%handle", "irrationality", "edge_score",
            "ev_units", "kelly_cap", "wong_tag"
        ]].sort_values(["matchup", "market"])

        st.dataframe(display_df, use_container_width=True)

def render_game_details(df: pd.DataFrame):
    """Render detailed game-by-game analysis."""
    st.header("🏈 NFL Game Details")
    
    if df.empty:
        st.warning("No game data available.")
        return
    
    df["irrationality"] = abs(df["%bets"] - df["%handle"])
    
    # Top games summary
    game_scores = df.groupby("matchup").agg({
        "irrationality": "mean",
        "%bets": "max"
    }).reset_index()
    
    game_scores.columns = ["matchup", "avg_irr", "max_public"]
    game_scores = game_scores.sort_values("avg_irr", ascending=False)
    
    st.markdown("### 🔥 Top Market Movers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Irrational Markets**")
        for idx, row in game_scores.head(3).iterrows():
            icon = "🔴" if row["avg_irr"] >= 20 else "🟠" if row["avg_irr"] >= 15 else "🟡"
            st.markdown(f"{icon} **{row['matchup']}** — {row['avg_irr']:.1f}%")
    
    with col2:
        st.markdown("**Highest Public Exposure**")
        fade_games = game_scores.sort_values("max_public", ascending=False).head(3)
        for idx, row in fade_games.iterrows():
            icon = "🔴" if row["max_public"] >= 75 else "🟠" if row["max_public"] >= 65 else "🟡"
            st.markdown(f"{icon} **{row['matchup']}** — {row['max_public']:.0f}% public")
    
    st.markdown("---")
    
    # Game-by-game breakdown
    for matchup in sorted(df["matchup"].unique()):
        sub = df[df["matchup"] == matchup].copy()
        sub["irrationality"] = (sub["%bets"] - sub["%handle"]).abs()
        game_score = sub["irrationality"].mean()
        max_public = sub["%bets"].max()
        
        # Icon selection
        if game_score >= 20:
            icon = "🔴"
            status = "CRITICAL"
        elif game_score >= 15:
            icon = "🟠"
            status = "HIGH"
        elif game_score >= 10:
            icon = "🟡"
            status = "MODERATE"
        else:
            icon = "🟢"
            status = "LOW"
        
        with st.expander(f"{icon} {matchup} — Irrationality: {status}", expanded=False):
            # Game metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Irrationality", f"{game_score:.1f}%")
            with col2:
                st.metric("Max Public %", f"{max_public:.0f}%")
            with col3:
                spread_markets = sub[sub["market"].str.lower() == "spread"]
                if not spread_markets.empty:
                    avg_spread = spread_markets["spread"].mean()
                    st.metric("Avg Spread", f"{avg_spread:.1f}" if pd.notna(avg_spread) else "N/A")
                else:
                    st.metric("Avg Spread", "N/A")
            with col4:
                total_markets = sub[sub["market"].str.lower() == "total"]
                if not total_markets.empty:
                    # Extract total value from side text
                    totals = []
                    for side in total_markets["side"]:
                        ou, tot = _parse_total_from_side(side)
                        if tot:
                            totals.append(tot)
                    avg_total = sum(totals) / len(totals) if totals else None
                    st.metric("Avg Total", f"{avg_total:.1f}" if avg_total else "N/A")
                else:
                    st.metric("Avg Total", "N/A")
            
            # Market breakdown table
            st.markdown("**Market Breakdown**")
            display_df = sub[[
                "market", "side", "odds", "spread", "%bets", "%handle", "irrationality"
            ]].rename(columns={
                "%bets": "Bets %",
                "%handle": "Handle %",
                "irrationality": "Irr %"
            })
            st.dataframe(display_df, use_container_width=True)
            
            # Visual gauge
            gauge_data = pd.DataFrame({"value": [game_score]})
            gauge = (
                alt.Chart(gauge_data)
                .mark_arc(innerRadius=50, outerRadius=70)
                .encode(
                    theta=alt.Theta('value:Q', scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        'value:Q',
                        scale=alt.Scale(
                            domain=[0, 10, 20, 30, 50],
                            range=['#00ff00', '#ffff00', '#ffa500', '#ff0000', '#8b0000']
                        ),
                        legend=None
                    )
                )
                .properties(width=150, height=150)
            )
            st.altair_chart(gauge, use_container_width=False)
            
            # Insights
            st.markdown("**📊 Insights:**")
            insights = []
            
            if game_score > 15:
                insights.append("⚠️ High irrationality — sharp money likely present")
            if max_public > 70:
                insights.append("🎯 Heavy public lean — prime fade candidate")
            
            spread_data = sub[sub["market"].str.lower() == "spread"]
            if not spread_data.empty:
                for _, row in spread_data.iterrows():
                    spr = row.get("spread")
                    if spr and abs(spr) in (3, 7):
                        insights.append(f"🔑 Critical key number at {spr}")
                    
                    wong = wong_teaser_tag(spr)
                    if wong and "PRIME" in wong:
                        insights.append(f"🎲 {wong} teaser spot identified")
            
            if not insights:
                insights.append("✅ Market appears balanced")
            
            for insight in insights:
                st.markdown(f"- {insight}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run():
    """Main application logic."""
    st.title("🏈 FoxEdge NFL Market Analyzer")
    st.caption("Professional-grade NFL betting analytics with key number awareness and CLV tracking")
    
    # Page selection
    page = st.radio(
        "Select View",
        ["Overview", "Game Details"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Fetch data
    eg = int(custom_eg) if int(custom_eg) > 0 else LEAGUE_GROUPS.get("NFL", 88807)
    
    with st.spinner("Fetching NFL betting splits..."):
        df = fetch_dk_splits(event_group=eg, date_range=slate, league="NFL")
    
    # Data cleaning
    if not df.empty:
        df["odds"] = df["odds"].apply(clean_odds)
        df = df.dropna(subset=["matchup", "market", "side"]).reset_index(drop=True)

    # Fetch TD scorer props across Anytime / First TD / 2+ TD markets
    td_any, td_first, td_multi, td_debug = fetch_td_boards_with_debug()

    # Manual override button just re-pulls the same scrape logic
    if force_td:
        td_any, td_first, td_multi, td_debug = fetch_td_boards_with_debug()
    
    # Route to appropriate page
    if page == "Overview":
        render_overview(
            df,
            td_any,
            td_first,
            td_multi,
            td_debug,
            capture_resilience_snapshot,
            res_snap_type
        )
    else:
        render_game_details(df)
    
    # Footer
    st.markdown("---")
    st.caption(
        "**FoxEdge NFL Analyzer** uses enhanced key-number distributions, situational spot detection, "
        "and professional CLV tracking. All snapshots logged to `bet_logs_nfl.db` for accountability. "
        "Composite edge scores blend EV, irrationality, RLM, and steam signals using your configured weights."
    )
    
    st.caption(f"📊 Database: `{BETLOGS_DB}` | 🔄 Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run()