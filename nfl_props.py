#!/usr/bin/env python3
"""
nfl_props.py

Standalone Streamlit app for FoxEdge NFL player prop recommendations.
Monolithic file: data loading, model training, injury parsing, simulation, edge calc, UI, logging.
Supports both CSV-driven picks and historical-only mode.
"""
import os
import re
import json
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as ndp
import feedparser
from scipy.stats import poisson
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression

# ---------------------- Configuration Constants ---------------------- #
SUPPORTED_MARKETS = [
    "player_pass_attempts",
    "player_receptions",
    "player_rush_attempts"
]
DEFAULT_N_SIMULATIONS = 10000

# Recency weighting parameters
HALF_LIFE_DAYS = 28
DECAY_RATE = np.log(2) / HALF_LIFE_DAYS

EDGE_THRESHOLD = 0.04           # 4% edge
VOLATILITY_RATIO_THRESHOLD = 1.25
Z_MIN = 0.50  # minimum |z| vs line to consider an edge
LOG_DIR = "./logs"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

NFL_TEAMS_3 = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"}

# ---------------------- Utility Functions ---------------------- #

def odds_to_prob(odds):
    """Convert American odds to implied probability."""
    try:
        o = float(odds)
        return (100.0 / (o + 100.0)) if o > 0 else (-o / (-o + 100.0))
    except:
        return np.nan

# --- Team extraction from free text ---
TEAM_RE = re.compile(r"\b([A-Z]{2,4})\b")
def infer_team_from_text(txt: str) -> str | float:
    if not isinstance(txt, str) or not txt:
        return np.nan
    # prefer parenthetical code e.g. "(DAL)"
    m = re.search(r"\(([A-Z]{2,4})\)", txt)
    if m and m.group(1) in NFL_TEAMS_3:
        return m.group(1)
    # fallback: any standalone 2-4 letter token
    for tok in TEAM_RE.findall(txt.upper()):
        if tok in NFL_TEAMS_3:
            return tok
    return np.nan

# --- Robust CSV reader that resets file pointer between attempts ---
from pandas.errors import EmptyDataError, ParserError

def _seek0(f):
    try:
        f.seek(0)
    except Exception:
        pass
    return f

def read_csv_safe(uploaded_file, **kwargs) -> pd.DataFrame:
    """Read a CSV from Streamlit's UploadedFile or file-like object safely.
    - Resets the pointer before reading.
    - Uses the python engine with automatic delimiter inference by default.
    - Returns empty DataFrame on EmptyDataError instead of exploding.
    """
    defaults = dict(engine='python', sep=None)
    defaults.update(kwargs or {})
    try:
        f = _seek0(uploaded_file)
        return pd.read_csv(f, **defaults)
    except EmptyDataError:
        return pd.DataFrame()
    except ParserError:
        # Sometimes files are actually TSV; try a tab retry
        try:
            f = _seek0(uploaded_file)
            return pd.read_csv(f, engine='python', sep='\t')
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ---------------------- Debug Helpers ---------------------- #
DEBUG_DEFAULT = False

def set_debug(enabled: bool):
    st.session_state['__debug_enabled'] = bool(enabled)

def is_debug() -> bool:
    return bool(st.session_state.get('__debug_enabled', DEBUG_DEFAULT))

def dbg(msg: str):
    if is_debug():
        try:
            st.write(msg)
        except Exception:
            pass

def dbg_df(df: pd.DataFrame, title: str, cols: list[str] | None = None, n: int = 10, key: str = ""):
    if not is_debug():
        return
    try:
        st.markdown(f"**DEBUG:** {title}  ")
        if cols:
            view = df[cols].head(n)
        else:
            view = df.head(n)
        st.dataframe(view, use_container_width=True, key=(key or f"dbg_{title.replace(' ','_')}"))
        st.caption(f"shape={df.shape}, cols={list(df.columns)}")
    except Exception:
        pass

# ---------------------- Name Normalization ---------------------- #
def short_name(name: str) -> str:
    """Return a normalized short name key like 'D. Prescott' for matching.
    Handles inputs like 'Dak Prescott', 'D Prescott', 'D. Prescott', extra spaces, suffixes, and 'Last, First' formats."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.strip()
    # normalize comma-separated 'Last, First [Middle]' → 'First Middle Last'
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            # parts[0] = Last, parts[1] = First [Middle]
            s = f"{parts[1]} {parts[0]}".strip()
    # remove common suffixes and commas
    s = re.sub(r",?(\s+(Jr\.|Sr\.|II|III|IV))$", "", s, flags=re.IGNORECASE)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    if len(parts) == 1:
        token = parts[0].replace("..", ".").strip()
        if re.match(r"^[A-Z]\.[ ]?[A-Za-z'\-]+$", token, flags=re.IGNORECASE):
            first_initial = token.split('.')[0][0].upper()
            last = token.split('.')[-1].strip().title()
            return f"{first_initial}. {last}"
        return token.title()
    first = parts[0].replace('.', '')
    last = parts[-1]
    if not first:
        return s.title()
    return f"{first[0].upper()}. {last.title()}"
# --- Last-name-only key for fallback matching ---
def last_name_key(name: str) -> str:
    """Extract letters-only lowercase last name from a full name like 'Dak Prescott' or 'Prescott, Dak'."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.strip()
    # swap comma format
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            s = f"{parts[1]} {parts[0]}"
    s = re.sub(r",?(\s+(Jr\.|Sr\.|II|III|IV))$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    last = s.split(" ")[-1]
    return re.sub(r"[^a-z]", "", last.lower())

# --- Canonical full-name key for exact matching ---
def full_name_key(name: str) -> str:
    """Return 'firstname lastname' lowercase letters-only for robust exact matching.
    Handles 'Last, First' inputs and removes suffixes like Jr., Sr., II, III, IV."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.strip()
    # swap comma format
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            s = f"{parts[1]} {parts[0]}"
    # remove common suffixes
    s = re.sub(r",?(\s+(Jr\.|Sr\.|II|III|IV))$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    toks = s.split(" ")
    if len(toks) >= 2:
        first, last = toks[0], toks[-1]
    else:
        first, last = toks[0], ""
    canon = f"{first} {last}".strip().lower()
    return re.sub(r"[^a-z ]", "", canon)

# --- Robust merge key for player matching ---
def merge_key_from_short(s: str) -> str:
    """Build a robust merge key from a short name like 'D. Prescott' → 'dprescott'."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z]", "", s.lower())

# ---------------------- Model Training ---------------------- #
@st.cache_data(show_spinner=False)
def train_models():
    """
    Train simple volume-based Poisson models: compute historical mean counts per game.
    Returns dict: { market_key: { player_name: mean_count } }
    """
    hist_means = {}
    raw_means = {}
    ensemble_models = {}
    current_year = datetime.date.today().year
    years = list(range(2020, current_year + 1))
    # Pull year by year to tolerate missing current season endpoints
    frames = []
    for y in years:
        try:
            df_y = ndp.import_pbp_data(years=[y])
            if df_y is not None and len(df_y):
                frames.append(df_y)
                dbg(f"{y} done.")
        except Exception as e:
            dbg(f"Skipping {y}: {e}")
            continue
    if not frames:
        raise RuntimeError("No PBP data available for requested years.")
    pbp_df = pd.concat(frames, ignore_index=True)
    pbp_df['game_date'] = pd.to_datetime(pbp_df['game_date'])
    current_ts = pd.Timestamp(datetime.date.today())

    for market in SUPPORTED_MARKETS:
        # select events + player_col as before...
        if market == "player_pass_attempts":
            events = pbp_df[pbp_df.get("pass_attempt") == 1]
            player_col = "passer_player_name"
        elif market == "player_receptions":
            events = pbp_df[(pbp_df.get("play_type") == "pass") & (pbp_df.get("complete_pass") == 1)]
            player_col = "receiver_player_name"
        elif market == "player_rush_attempts":
            events = pbp_df[pbp_df.get("rush_attempt") == 1]
            player_col = "rusher_player_name"
        else:
            continue

        if events.empty:
            hist_means[market] = {}
            raw_means[market] = {}
        else:
            counts = (
                events.groupby([player_col, 'game_id', 'game_date'])
                      .size()
                      .reset_index(name='count')
            )
            # raw mean per game
            raw = counts.groupby(player_col)['count'].mean().to_dict()
            raw_means[market] = raw
            # recency-weighted mean
            counts['days_diff'] = (current_ts - counts['game_date']).dt.days
            counts['weight'] = np.exp(-DECAY_RATE * counts['days_diff'])
            counts['wprod'] = counts['count'] * counts['weight']
            agg = counts.groupby(player_col, as_index=True)[['wprod','weight']].sum()
            weighted = (agg['wprod'] / agg['weight']).to_dict()
            hist_means[market] = weighted

    # build ensemble per market
    for market in SUPPORTED_MARKETS:
        raw = raw_means.get(market, {})
        weighted = hist_means.get(market, {})
        # prepare training data
        data = [(raw.get(p, w), w) for p, w in weighted.items()]
        if len(data) >= 5:
            X = np.array(data)
            y = np.array([w for _, w in data])
            rf = RandomForestRegressor(n_estimators=100, random_state=0)
            rf.fit(X, y)
            ensemble_models[market] = rf
        else:
            ensemble_models[market] = None

    st.session_state['raw_means'] = raw_means
    st.session_state['ensemble_models'] = ensemble_models
    st.session_state['model_means'] = hist_means
    return hist_means


@st.cache_data(show_spinner=False)
def backtest_and_calibrate():
    """
    Run back-test on historical PBP data to train per-market isotonic calibrators.
    Stores models in st.session_state['calibrators'].
    """
    calibrators = {}
    current_year = datetime.date.today().year
    years = list(range(2020, current_year))  # hold out current season
    frames = []
    for y in years:
        try:
            df_y = ndp.import_pbp_data(years=[y])
            if df_y is not None and len(df_y):
                frames.append(df_y)
                dbg(f"{y} done.")
        except Exception as e:
            dbg(f"Skipping {y}: {e}")
            continue
    if not frames:
        raise RuntimeError("No historical PBP data available for backtest.")
    pbp_df = pd.concat(frames, ignore_index=True)
    pbp_df['game_date'] = pd.to_datetime(pbp_df['game_date'])
    # ensure models trained
    if 'model_means' not in st.session_state:
        train_models()
    hist_means = st.session_state['model_means']
    for market in SUPPORTED_MARKETS:
        # pick correct player column and events as in train_models
        if market == "player_pass_attempts":
            events = pbp_df[pbp_df.get("pass_attempt") == 1]
            player_col = "passer_player_name"
        elif market == "player_receptions":
            events = pbp_df[(pbp_df.get("play_type") == "pass") & (pbp_df.get("complete_pass") == 1)]
            player_col = "receiver_player_name"
        elif market == "player_rush_attempts":
            events = pbp_df[pbp_df.get("rush_attempt") == 1]
            player_col = "rusher_player_name"
        else:
            continue
        # compute count per player-game
        counts = (
            events.groupby([player_col, 'game_id'])
                  .size()
                  .reset_index(name='count')
        )
        preds = []
        labels = []
        for _, row in counts.iterrows():
            player = row[player_col]
            actual = row['count']
            lam = hist_means.get(market, {}).get(player, np.nan)
            if np.isnan(lam):
                continue
            # predicted probability of exceeding lambda threshold
            p_over = 1 - poisson.cdf(lam, lam)
            preds.append(p_over)
            labels.append(int(actual > lam))
        if len(preds) >= 20:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(np.array(preds), np.array(labels))
            calibrators[market] = iso
        else:
            calibrators[market] = None
    st.session_state['calibrators'] = calibrators
    return calibrators

# ---------------------- Rosters ---------------------- #
@st.cache_data(show_spinner=False)
def load_rosters(years: list[int] | None = None) -> pd.DataFrame:
    """Load NFL rosters using nfl_data_py seasonal + weekly APIs.
    Strategy:
      1) Determine target season (max of `years` if provided, else current year).
      2) Pull weekly rosters for that season, pick the most recent week available.
      3) Build player→team map from weekly (most accurate snapshot),
         and backfill any missing teams from seasonal rosters.
      4) If both are empty, fallback to building from PBP most recent posteam per player.
    Returns columns: ['player_fullkey','player','team_abbr','season','week'] (week may be NaN if weekly empty).
    """
    try:
        # Determine season to use
        if years is None or (isinstance(years, list) and len(years) == 0):
            season = datetime.date.today().year
        elif isinstance(years, list):
            season = max(years)
        else:
            # tolerate accidental int passed in
            season = int(years)
        dbg(f"load_rosters season={season}")

        # Pull rosters
        wk = pd.DataFrame()
        try:
            wk = ndp.import_weekly_rosters([season])
        except Exception as e:
            dbg(f"weekly_rosters {season} failed: {e}")
        se = pd.DataFrame()
        try:
            se = ndp.import_seasonal_rosters([season])
        except Exception as e:
            dbg(f"seasonal_rosters {season} failed: {e}")

        # Normalize a name column
        def build_name(df: pd.DataFrame) -> pd.Series:
            if df is None or df.empty:
                return pd.Series([], dtype=str)
            if 'player_name' in df.columns:
                return df['player_name'].astype(str)
            if 'first_name' in df.columns and 'last_name' in df.columns:
                return (df['first_name'].astype(str) + ' ' + df['last_name'].astype(str)).str.strip()
            name_cols = [c for c in df.columns if 'name' in c.lower()]
            return df[name_cols[0]].astype(str) if name_cols else pd.Series([], dtype=str)

        # Normalize a team column
        def pick_team(df: pd.DataFrame) -> pd.Series:
            if df is None or df.empty:
                return pd.Series([], dtype=str)
            for c in ['team', 'recent_team', 'team_abbr']:
                if c in df.columns:
                    return df[c].astype(str)
            return pd.Series([np.nan] * len(df))

        # Most recent week snapshot
        weekly_map = pd.DataFrame(columns=['player','team_abbr','season','week','player_fullkey'])
        if not wk.empty and 'week' in wk.columns:
            max_week = pd.to_numeric(wk['week'], errors='coerce').max()
            cur = wk[wk['week'] == max_week].copy()
            cur_player = build_name(cur)
            cur_team = pick_team(cur)
            weekly_map = pd.DataFrame({
                'player': cur_player,
                'team_abbr': cur_team,
                'season': season,
                'week': max_week
            })
            weekly_map['player_fullkey'] = weekly_map['player'].apply(full_name_key)
            weekly_map = weekly_map[weekly_map['player_fullkey'] != '']

        # Seasonal fallback/backfill
        seasonal_map = pd.DataFrame(columns=['player','team_abbr','season','player_fullkey'])
        if not se.empty:
            se_player = build_name(se)
            se_team = pick_team(se)
            seasonal_map = pd.DataFrame({
                'player': se_player,
                'team_abbr': se_team,
                'season': season
            })
            seasonal_map['player_fullkey'] = seasonal_map['player'].apply(full_name_key)
            seasonal_map = seasonal_map[seasonal_map['player_fullkey'] != '']

        # Prefer weekly snapshot, backfill from seasonal
        base = None
        if not weekly_map.empty:
            base = weekly_map
            if not seasonal_map.empty:
                base = base.merge(seasonal_map[['player_fullkey','team_abbr']].rename(columns={'team_abbr':'team_abbr_se'}),
                                  on='player_fullkey', how='left')
                # if weekly team missing, use seasonal
                mask = base['team_abbr'].isna()
                base.loc[mask, 'team_abbr'] = base.loc[mask, 'team_abbr_se']
                base = base.drop(columns=['team_abbr_se'])
        elif not seasonal_map.empty:
            base = seasonal_map
            base['week'] = np.nan

        # If both weekly and seasonal are empty, build a fallback from PBP: most recent posteam per player
        try:
            if ('base' not in locals()) or base is None or base.empty:
                years_fb = [season, max(season-1, season)]
                pbp_fb_frames = []
                for y in years_fb:
                    try:
                        df_y = ndp.import_pbp_data(years=[y])
                        if df_y is not None and len(df_y):
                            pbp_fb_frames.append(df_y)
                    except Exception as e:
                        dbg(f"pbp fallback {y} failed: {e}")
                if pbp_fb_frames:
                    pbp_fb = pd.concat(pbp_fb_frames, ignore_index=True)
                    pbp_fb['game_date'] = pd.to_datetime(pbp_fb['game_date'])
                    # collect player-team pairs across roles using posteam as team
                    rows = []
                    def add_pairs(col_name):
                        if col_name in pbp_fb.columns:
                            sub = pbp_fb[[col_name,'posteam','game_date']].dropna()
                            sub = sub.rename(columns={col_name:'player', 'posteam':'team_abbr'})
                            rows.append(sub)
                    for c in ['passer_player_name','receiver_player_name','rusher_player_name']:
                        add_pairs(c)
                    if rows:
                        all_pairs = pd.concat(rows, ignore_index=True)
                        all_pairs['player_fullkey'] = all_pairs['player'].apply(full_name_key)
                        # pick most recent team for each player
                        all_pairs = all_pairs.sort_values(['player_fullkey','game_date'])
                        last_team = all_pairs.groupby('player_fullkey').tail(1)[['player_fullkey','player','team_abbr']]
                        base = last_team.drop_duplicates(subset=['player_fullkey'])
                        base['season'] = season
                        base['week'] = np.nan
        except Exception as e:
            dbg(f"roster pbp fallback error: {e}")

        if base is None or base.empty:
            dbg("load_rosters produced EMPTY roster map")
            return pd.DataFrame(columns=['player_fullkey','player','team_abbr','season','week'])

        # Deduplicate by player_fullkey; keep weekly row if present
        base = base.sort_values(['player_fullkey', 'week'], ascending=[True, False])
        base = base.drop_duplicates(subset=['player_fullkey'], keep='first')
        if base.empty:
            dbg("load_rosters produced EMPTY roster map")
        return base.reset_index(drop=True)
    except Exception as e:
        dbg(f"load_rosters error: {e}")
        dbg("load_rosters produced EMPTY roster map")
        return pd.DataFrame(columns=['player_fullkey','player','team_abbr','season','week'])

# ---------------------- Injury Status Parsing ---------------------- #
@st.cache_data(show_spinner=False)
def load_status():
    """
    Fetch and parse Rotowire RSS for player status updates.
    Returns DataFrame with ['player','status'] or empty.
    """
    try:
        feed = feedparser.parse('https://www.rotowire.com/rss/latest-football.htm')
        rows = []
        for entry in feed.entries:
            title = entry.get('title', '')
            m = re.match(r'^(.*?):\s*(Out|Questionable|Probable|Doubtful|Limited|Active)', title)
            if m:
                rows.append({'player': m.group(1).strip(), 'status': m.group(2)})
        df = pd.DataFrame(rows).drop_duplicates(subset=['player'])
        return df
    except:
        return pd.DataFrame(columns=['player', 'status'])

def normalize_market_key(market_raw: str) -> str:
    """Map assorted market strings into our SUPPORTED_MARKETS keys.
    Accepts variants like 'Player Pass Attempts', 'Pass Attempts (Player)', 'Receptions (Player)', etc.
    Returns one of SUPPORTED_MARKETS or an empty string if not mappable.
    """
    if not isinstance(market_raw, str):
        return ""
    m = market_raw.strip().lower()
    # common variants
    mapping = {
        "player pass attempts": "player_pass_attempts",
        "pass attempts (player)": "player_pass_attempts",
        "pass attempts": "player_pass_attempts",
        "player receptions": "player_receptions",
        "receptions (player)": "player_receptions",
        "receptions": "player_receptions",
        "player rush attempts": "player_rush_attempts",
        "rush attempts (player)": "player_rush_attempts",
        "rush attempts": "player_rush_attempts",
    }
    # direct mapping
    if m in mapping:
        return mapping[m]
    # fuzzy contains checks
    if "pass" in m and "attempt" in m and "player" in m:
        return "player_pass_attempts"
    if "reception" in m and ("player" in m or "(player)" in m):
        return "player_receptions"
    if "rush" in m and "attempt" in m and ("player" in m or "(player)" in m):
        return "player_rush_attempts"
    return ""

def process_odds_csv_oddsapi_format(uploaded_file) -> pd.DataFrame:
    """Parse generic odds CSVs with columns like
    ['game_id','commence_time','in_play','bookmaker','last_update','home_team','away_team','market','label','description','price','point']
    and convert to our normalized schema: ['player','team','market_key','line','odds','book_name'].
    """
    df = read_csv_safe(uploaded_file)
    dbg("Parsing odds CSV (generic format) ...")
    dbg_df(df, "Raw odds CSV (first 12 rows)", n=12, key="dbg_raw_odds")
    # ensure required columns exist
    required_cols = {'market','label','price','point','bookmaker'}
    if not required_cols.issubset(set(df.columns)):
        return pd.DataFrame()  # not this format

    # Normalize column names we care about
    out = pd.DataFrame()
    out['book_name'] = df['bookmaker'].astype(str)
    out['odds'] = pd.to_numeric(df['price'], errors='coerce')
    out['line'] = pd.to_numeric(df['point'], errors='coerce')

    # Detect side (Over/Under) from description/label
    def extract_side(row):
        txt = f"{row.get('description','')} {row.get('label','')}".strip()
        m = re.search(r"\b(Over|Under)\b", txt, flags=re.IGNORECASE)
        return m.group(1).title() if m else np.nan
    out['side'] = df.apply(extract_side, axis=1)

    # Extract player strictly from 'description' per user requirement
    def extract_player(row):
        desc = str(row.get('description', '')).strip()
        if not desc:
            return np.nan
        txt = desc
        # Remove trailing Over/Under phrasing
        txt = re.sub(r"\s+(Over|Under)\b.*$", "", txt, flags=re.IGNORECASE)
        # If comma format 'Last, First'
        if "," in txt and not " - " in txt:
            parts = [p.strip() for p in txt.split(",")]
            if len(parts) >= 2:
                cand = f"{parts[1]} {parts[0]}".strip()
            else:
                cand = txt
        else:
            cand = txt
        # If there is a hyphen separator, pick the chunk that looks like a person name
        if " - " in cand:
            chunks = [c.strip() for c in cand.split(" - ")]
            name_like = []
            name_re = re.compile(r"^[A-Za-z\.\'\-]+(?:\s+[A-Za-z\.\'\-]+)+$")
            for ch in chunks:
                # Exclude obvious team phrases with all-caps 2-4 letters in parens
                if re.search(r"\([A-Z]{2,4}\)$", ch):
                    # strip team code
                    ch = re.sub(r"\s*\([A-Z]{2,4}\)$", "", ch)
                if name_re.match(ch):
                    name_like.append(ch)
            if name_like:
                cand = name_like[-1]  # usually player is last chunk
            else:
                # fallback: take the longest chunk
                cand = max(chunks, key=len)
        # Remove team code in parentheses at the end
        cand = re.sub(r"\s*\([A-Z]{2,4}\)$", "", cand)
        # Remove leading 'Alt' or 'Alternate'
        cand = re.sub(r"^(Alt\s+|Alternate\s+)", "", cand, flags=re.IGNORECASE)
        return cand.strip()

    out['player'] = df.apply(extract_player, axis=1)

    # Add player_short column for matching
    out['player_short'] = out['player'].apply(short_name)
    out['player_merge'] = out['player_short'].apply(merge_key_from_short)
    out['last_name_merge'] = out['player'].apply(last_name_key)
    # Canonical full-name key
    out['player_fullkey'] = out['player'].apply(full_name_key)

    # --- Persist event timestamps for walk-forward backtesting ---
    # Prefer explicit game start if available; otherwise use last_update as a proxy
    if 'commence_time' in df.columns:
        out['event_time'] = pd.to_datetime(df['commence_time'], errors='coerce')
    elif 'last_update' in df.columns:
        out['event_time'] = pd.to_datetime(df['last_update'], errors='coerce')
    else:
        out['event_time'] = pd.NaT
    # Normalize to a date for grouping; keep a normalized midnight-naive timestamp
    out['event_date'] = pd.to_datetime(out['event_time']).dt.normalize()

    # Team is optional; try to glean from label/description using robust inference
    def extract_team(row):
        # Prefer explicit home/away matching from the source row if available
        txt = f"{row.get('label','')} {row.get('description','')}".strip()
        t = infer_team_from_text(txt)
        return t
    out['team'] = df.apply(extract_team, axis=1)

    # Map market string to our internal key
    out['market_key'] = df['market'].apply(normalize_market_key)

    dbg_df(out, "Parsed odds (pre-filter)",
           ['player','player_short','player_merge','last_name_merge','team','market_key','line','odds','book_name'],
       n=12, key="dbg_parsed_pre")
    dbg(f"Unique markets in odds (raw): {sorted(df['market'].astype(str).str.lower().unique().tolist())[:10]} ...")
    dbg(f"Unique mapped market_keys (pre-filter): {sorted(out['market_key'].dropna().unique().tolist())}")

    # Filter to supported markets and drop missing essentials
    out = out[out['market_key'].isin(SUPPORTED_MARKETS)].copy()
    out = out.dropna(subset=['player','line','odds'])

    # Enrich missing team from roster, but only fill where team is NA
    roster = load_rosters()
    if not roster.empty:
        out = out.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
        out['team'] = out.get('team').astype('string') if 'team' in out.columns else pd.Series([pd.NA]*len(out), dtype='string')
        out['team_abbr'] = out['team_abbr'].astype('string')
        mask = out['team'].isna() | (out['team'] == '')
        out.loc[mask, 'team'] = out.loc[mask, 'team_abbr']
        out = out.drop(columns=['team_abbr'])

    dbg(f"After filter: rows={len(out)}, supported market rows kept, dropped missing player/line/odds rows.")
    if is_debug():
        miss_player = out['player'].isna().sum() if 'player' in out else 0
        miss_line = out['line'].isna().sum() if 'line' in out else 0
        miss_odds = out['odds'].isna().sum() if 'odds' in out else 0
        st.caption(f"Missing counts -> player:{miss_player}, line:{miss_line}, odds:{miss_odds}")

    # Additional audit: missing side
    miss_side = out['side'].isna().sum() if 'side' in out else 0
    if is_debug():
        st.caption(f"Missing side: {miss_side}")

    # Compute implied probability
    out['implied_prob'] = out['odds'].apply(odds_to_prob)

    # Final column order to match downstream expectation
    cols = ['player','player_short','player_merge','team','market_key','line','odds','implied_prob','book_name','side','event_time','event_date','player_fullkey','last_name_merge']
    return out[cols]

# ---------------------- Odds CSV Processing: Prop Details Format ---------------------- #
def process_odds_csv_propdetails_format(uploaded_file) -> pd.DataFrame:
    """Parse Player Prop Detail CSVs where Over/Under appear on the same row.
    This tries to be schema-agnostic by probing common column aliases.
    Returns normalized columns compatible with downstream code:
    ['player','team','market_key','line','odds','book_name','side','implied_prob','player_short','player_merge','last_name_merge','player_fullkey','event_time','event_date']
    """
    df_raw = read_csv_safe(uploaded_file)
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    # Build a case-insensitive column map
    cmap = {c.lower().strip(): c for c in df_raw.columns}
    def pick(*aliases):
        for a in aliases:
            if a in cmap:
                return cmap[a]
        return None

    # Required-ish columns
    col_player = pick('player','playername','name','athlete','athlete_name')
    col_market = pick('market','prop','bet type','bet','wager type','stat')
    col_line   = pick('line','point','number','threshold')
    col_team   = pick('team','team_abbr','teamabbr','team code')
    col_book   = pick('book','bookmaker','sportsbook','sports book')
    col_date   = pick('date','event_date','game_date','commence_time','timestamp','last_update')

    # Over/Under odds may have many aliases
    over_cols = [c for c in df_raw.columns if re.search(r'\bover\b.*(odds|price|american|us)', c, flags=re.IGNORECASE)]
    under_cols = [c for c in df_raw.columns if re.search(r'\bunder\b.*(odds|price|american|us)', c, flags=re.IGNORECASE)]

    # If we can't find minimally required columns, bail
    if not col_player or not col_market or not col_line or (not over_cols and not under_cols):
        return pd.DataFrame()

    # Normalize base frame
    base = pd.DataFrame()
    base['player'] = df_raw[col_player].astype(str)
    base['team'] = df_raw[col_team].astype(str) if col_team else np.nan
    base['market_key'] = df_raw[col_market].astype(str).apply(normalize_market_key)
    base['line'] = pd.to_numeric(df_raw[col_line], errors='coerce')
    base['book_name'] = df_raw[col_book].astype(str) if col_book else 'Unknown'

    # Date/time handling
    if col_date:
        base['event_time'] = pd.to_datetime(df_raw[col_date], errors='coerce')
    else:
        base['event_time'] = pd.NaT
    base['event_date'] = pd.to_datetime(base['event_time']).dt.normalize()

    # Player name keys
    base['player_short'] = base['player'].apply(short_name)
    base['player_merge'] = base['player_short'].apply(merge_key_from_short)
    base['last_name_merge'] = base['player'].apply(last_name_key)
    base['player_fullkey'] = base['player'].apply(full_name_key)

    # Infer team from any text columns if missing
    if 'team' not in base.columns or base['team'].isna().all():
        # Try to infer from any text columns present in the raw frame
        text_cols = [c for c in df_raw.columns if isinstance(c, str) and any(k in c.lower() for k in ['desc','label','team','market','prop','bet'])]
        team_guess = []
        for i in range(len(base)):
            guess = np.nan
            for c in text_cols:
                guess = infer_team_from_text(str(df_raw.iloc[i][c]))
                if pd.notna(guess):
                    break
            team_guess.append(guess)
        base['team'] = pd.Series(team_guess, index=base.index)

    # Build long rows for Over/Under sides
    rows = []
    for idx, r in base.iterrows():
        # Over
        over_price = np.nan
        for c in over_cols:
            v = pd.to_numeric(df_raw.loc[idx, c], errors='coerce')
            if pd.notna(v):
                over_price = v
                break
        # Under
        under_price = np.nan
        for c in under_cols:
            v = pd.to_numeric(df_raw.loc[idx, c], errors='coerce')
            if pd.notna(v):
                under_price = v
                break
        # Build two rows if present
        common = r.to_dict()
        if pd.notna(over_price):
            rows.append({**common, 'side': 'Over', 'odds': over_price})
        if pd.notna(under_price):
            rows.append({**common, 'side': 'Under', 'odds': under_price})
    out = pd.DataFrame(rows)

    # Filter to supported markets and drop empties
    out = out[out['market_key'].isin(SUPPORTED_MARKETS)]
    out = out.dropna(subset=['player','line','odds'])

    # Compute implied prob from American odds (or convert decimals heuristically)
    def prob_from_any(o):
        # If looks like decimal odds (>1 and <20) and not American magnitude, convert
        try:
            val = float(o)
        except:
            return np.nan
        if val > 1.0 and val < 20.0 and not (val >= 100 or val <= -100):
            # decimal odds to implied
            return 1.0 / val
        return odds_to_prob(val)
    out['implied_prob'] = out['odds'].apply(prob_from_any)

    # Backfill team from roster if missing, but only fill where team is NA
    roster = load_rosters()
    if not roster.empty:
        out = out.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
        out['team'] = out.get('team').astype('string') if 'team' in out.columns else pd.Series([pd.NA]*len(out), dtype='string')
        out['team_abbr'] = out['team_abbr'].astype('string')
        mask = out['team'].isna() | (out['team'] == '')
        out.loc[mask, 'team'] = out.loc[mask, 'team_abbr']
        out = out.drop(columns=['team_abbr'])

    # Final column order
    cols = ['player','player_short','player_merge','team','market_key','line','odds','implied_prob','book_name','side','event_time','event_date','player_fullkey','last_name_merge']
    return out[cols]

# ---------------------- Odds CSV Processing ---------------------- #
def process_odds_csv(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV and normalize odds to our schema regardless of source format.
    Supported formats in order: PropDetails, GenericOddsAPI, In-house.
    """
    # Try Prop Details format (Over/Under on same row)
    try:
        df_prop = process_odds_csv_propdetails_format(uploaded_file)
        if not df_prop.empty:
            return df_prop
    except Exception:
        pass

    # Try generic odds export next
    try:
        df_generic = process_odds_csv_oddsapi_format(uploaded_file)
        if not df_generic.empty:
            return df_generic
    except Exception:
        pass

    # Fallback: original in-house schema
    df = read_csv_safe(uploaded_file)
    required = {'player','team','market_key','line','odds','book_name'}
    missing = required - set(df.columns)
    if missing:
        st.error(f"CSV missing columns: {', '.join(missing)}")
        return pd.DataFrame()
    df = df[df['market_key'].isin(SUPPORTED_MARKETS)].copy()
    df['implied_prob'] = df['odds'].apply(odds_to_prob)
    df['line'] = pd.to_numeric(df['line'], errors='coerce')
    df['player_short'] = df['player'].apply(short_name)
    df['player_merge'] = df['player_short'].apply(merge_key_from_short)
    df['last_name_merge'] = df['player'].apply(last_name_key)
    df['player_fullkey'] = df['player'].apply(full_name_key)
    # Enrich missing team from roster, but only fill where team is NA
    roster = load_rosters()
    if not roster.empty:
        df = df.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
        df['team'] = df.get('team').astype('string') if 'team' in df.columns else pd.Series([pd.NA]*len(df), dtype='string')
        df['team_abbr'] = df['team_abbr'].astype('string')
        mask = df['team'].isna() | (df['team'] == '')
        df.loc[mask, 'team'] = df.loc[mask, 'team_abbr']
        df = df.drop(columns=['team_abbr'])
    return df

# ---------------------- Edge Computation (CSV) ---------------------- #
def compute_edges_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute edge and recommendation for CSV-driven picks."""
    # thresholds from sidebar with fallbacks
    EDGE_THR = float(st.session_state.get('edge_threshold', EDGE_THRESHOLD))
    VOL_THR = float(st.session_state.get('volatility_ratio', VOLATILITY_RATIO_THRESHOLD))
    Z_THR = float(st.session_state.get('z_min', Z_MIN))
    results = []
    models = st.session_state.get('model_means', {})
    raw_means = st.session_state.get('raw_means', {})
    ensemble_models = st.session_state.get('ensemble_models', {})
    global_means = {m: np.mean(list(models.get(m, {}).values())) if models.get(m) else 0
                    for m in SUPPORTED_MARKETS}

    for _, row in df.iterrows():
        try:
            market, player = row['market_key'], row['player']
            lam_base = models.get(market, {}).get(player, global_means.get(market, 0))
            # ensemble adjustment
            if ensemble_models.get(market):
                r = raw_means.get(market, {}).get(player, lam_base)
                lam_adj = ensemble_models[market].predict([[r, lam_base]])[0]
            else:
                lam_adj = lam_base
            if lam_adj <= 0:
                raise ValueError('No data')
            # Simulate
            sim = np.random.poisson(lam_adj, DEFAULT_N_SIMULATIONS)
            mu, sigma = sim.mean(), sim.std()

            # Use ONLY the authoritative book line from CSV `point`
            line_used = row.get('book_line', np.nan)
            if pd.isna(line_used):
                # skip rows without a concrete bookmaker line
                raise ValueError('No line')

            # Probabilities against the same line
            p_over = np.mean(sim > line_used)
            p_under = np.mean(sim < line_used)

            # Optional calibration
            cal = st.session_state.get('calibrators', {}).get(market)
            if cal is not None:
                p_over = cal.predict(np.array([p_over]))[0]
                p_under = cal.predict(np.array([p_under]))[0]

            # z-score vs line
            z_vs_line = (mu - line_used) / (sigma + 1e-9)

            # Decide recommendation respecting the CSV 'side' if provided
            side_csv = row.get('side') if 'side' in row else np.nan
            if isinstance(side_csv, str) and side_csv in ('Over','Under'):
                side = side_csv
                p_side = p_over if side == 'Over' else p_under
                implied_prob_used = row.get('implied_prob_fair', row.get('implied_prob', np.nan))
                if (pd.isna(implied_prob_used)) and ('odds' in row and pd.notna(row['odds'])):
                    implied_prob_used = odds_to_prob(row['odds'])
                edge_prob = (p_side - implied_prob_used) if pd.notna(implied_prob_used) else np.nan
                edge_pct = edge_prob * 100.0 if pd.notna(edge_prob) else np.nan
                # guardrails
                ok_vol = (sigma / max(mu, 1e-9)) <= VOL_THR
                ok_z = abs(z_vs_line) >= Z_THR
                pass_edge = (pd.notna(edge_prob) and edge_prob >= EDGE_THR)
                pass_vol = ok_vol
                pass_z = ok_z
                rec = side if (pass_edge and pass_vol and pass_z) else 'No Action'
            else:
                # Without a declared side we do not take action
                side = 'Unknown'
                edge_prob = np.nan
                edge_pct = np.nan
                implied_prob_used = np.nan
                rec = 'No Action'
                pass_edge = False
                pass_vol = False
                pass_z = False
                p_over = np.nan
                p_under = np.nan
                p_side = np.nan
        except:
            lam_adj, mu, sigma, rec, side, line_used, z_vs_line = np.nan, np.nan, np.nan, 'No Action', np.nan, np.nan, np.nan
            pass_edge = False
            pass_vol = False
            pass_z = False
            p_over = np.nan
            p_under = np.nan
            p_side = np.nan
            edge_prob = np.nan
            edge_pct = np.nan
            implied_prob_used = np.nan
        results.append({
            **row.to_dict(),
            'lambda': lam_adj,
            'model_mean': mu,
            'model_std': sigma,
            'line': line_used,
            'delta_vs_line': (mu - line_used) if pd.notna(mu) and pd.notna(line_used) else np.nan,
            'z_vs_line': z_vs_line,
            'edge_prob': edge_prob,
            'edge_pct': edge_pct,
            'recommendation': rec,
            'side': side,
            'p_over': p_over,
            'p_under': p_under,
            'pass_edge': pass_edge,
            'pass_vol': pass_vol,
            'pass_z': pass_z,
            'p_side': p_side,
            'implied_prob_used': implied_prob_used
        })
    return pd.DataFrame(results)

# ---------------------- Edge Computation (Historical) ---------------------- #
# ---------------------- Backtesting (Walk-Forward vs Real Lines) ---------------------- #
@st.cache_data(show_spinner=False)
def load_pbp_years_for_backtest(start_year: int, end_year: int) -> pd.DataFrame:
    """Load play-by-play for an inclusive year range with normalized game_date."""
    frames = []
    for y in range(int(start_year), int(end_year) + 1):
        try:
            df_y = ndp.import_pbp_data(years=[y])
            if df_y is not None and len(df_y):
                frames.append(df_y)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    pbp = pd.concat(frames, ignore_index=True)
    pbp['game_date'] = pd.to_datetime(pbp['game_date'])
    return pbp

def _counts_from_pbp(pbp: pd.DataFrame, market: str) -> pd.DataFrame:
    """Return per-player per-game counts for the given market with canonical name keys."""
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=['player','player_fullkey','game_id','game_date','count'])
    if market == 'player_pass_attempts':
        events = pbp[pbp.get('pass_attempt') == 1]
        player_col = 'passer_player_name'
    elif market == 'player_receptions':
        events = pbp[(pbp.get('play_type') == 'pass') & (pbp.get('complete_pass') == 1)]
        player_col = 'receiver_player_name'
    elif market == 'player_rush_attempts':
        events = pbp[pbp.get('rush_attempt') == 1]
        player_col = 'rusher_player_name'
    else:
        return pd.DataFrame(columns=['player','player_fullkey','game_id','game_date','count'])
    if events.empty:
        return pd.DataFrame(columns=['player','player_fullkey','game_id','game_date','count'])
    counts = (
        events.groupby([player_col, 'game_id', 'game_date'])
              .size()
              .reset_index(name='count')
              .rename(columns={player_col: 'player'})
    )
    counts['player_fullkey'] = counts['player'].apply(full_name_key)
    counts['game_date'] = pd.to_datetime(counts['game_date']).dt.normalize()
    return counts[['player','player_fullkey','game_id','game_date','count']]

@st.cache_data(show_spinner=False)
def build_counts_frames(pbp: pd.DataFrame) -> dict:
    """Build and cache per-market counts frames once for backtesting."""
    frames = {}
    for mk in SUPPORTED_MARKETS:
        frames[mk] = _counts_from_pbp(pbp, mk)
    return frames

def _weighted_lambdas_asof(counts_df: pd.DataFrame, as_of_date: pd.Timestamp) -> dict:
    """Compute recency-weighted per-player means using only games strictly before as_of_date."""
    if counts_df is None or counts_df.empty:
        return {}
    mask = counts_df['game_date'] < pd.to_datetime(as_of_date).normalize()
    hist = counts_df.loc[mask].copy()
    if hist.empty:
        return {}
    # recency weights
    hist['days_diff'] = (pd.to_datetime(as_of_date).normalize() - hist['game_date']).dt.days
    hist['weight'] = np.exp(-DECAY_RATE * hist['days_diff'])
    hist['wprod'] = hist['count'] * hist['weight']
    agg = hist.groupby('player_fullkey', as_index=True)[['wprod','weight']].sum()
    weighted = (agg['wprod'] / agg['weight']).to_dict()
    return weighted

def _american_profit_per_risk_unit(odds_val: float) -> float:
    """Return profit (not payout) per 1.0 risk unit if the bet wins."""
    try:
        o = float(odds_val)
    except Exception:
        return np.nan
    if o >= 100:
        return o / 100.0
    elif o <= -100:
        return 100.0 / abs(o)
    return np.nan

def _resolve_outcome(side: str, actual_count: float, line: float) -> str:
    if pd.isna(actual_count) or pd.isna(line) or side not in ('Over','Under'):
        return 'No Result'
    if abs(line - round(line)) < 1e-8:  # integer line push check
        if actual_count == line:
            return 'Push'
    if side == 'Over':
        return 'Win' if actual_count > line else 'Loss'
    else:
        return 'Win' if actual_count < line else 'Loss'

def run_walkforward_backtest(odds_hist: pd.DataFrame, counts_frames: dict, use_calibration: bool = False) -> tuple[pd.DataFrame, dict]:
    """Walk-forward backtest vs real bookmaker lines.
    - odds_hist must include: player_fullkey, market_key, side, line, odds, implied_prob or implied_prob_fair, event_date
    - counts_frames: dict of market -> counts_df with actuals and historical training data
    Returns (results_df, summary_dict)
    """
    # thresholds
    EDGE_THR = float(st.session_state.get('edge_threshold', EDGE_THRESHOLD))
    VOL_THR = float(st.session_state.get('volatility_ratio', VOLATILITY_RATIO_THRESHOLD))
    Z_THR = float(st.session_state.get('z_min', Z_MIN))

    df = odds_hist.copy()
    # Ensure required keys exist
    if 'player_fullkey' not in df.columns and 'player' in df.columns:
        df['player_fullkey'] = df['player'].apply(full_name_key)
    if 'event_date' not in df.columns:
        # best-effort fallback
        if 'event_time' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_time'], errors='coerce').dt.normalize()
        else:
            st.warning('Historical odds missing event_date; cannot run walk-forward backtest.')
            return pd.DataFrame(), {}
    df['event_date'] = pd.to_datetime(df['event_date']).dt.normalize()
    df = df[df['market_key'].isin(SUPPORTED_MARKETS)]
    df = df.dropna(subset=['player_fullkey','market_key','side','line','odds','event_date'])

    # De-vig fair implied prob when both sides exist for the same (player, market, book, line, date)
    if {'side','implied_prob','player_fullkey','market_key','book_name','line','event_date'}.issubset(df.columns):
        grp_keys = ['player_fullkey','market_key','book_name','line','event_date']
        side_count = df.groupby(grp_keys)['side'].transform('nunique')
        sum_imp = df.groupby(grp_keys)['implied_prob'].transform('sum')
        df['implied_prob_fair'] = np.where(side_count >= 2, df['implied_prob'] / sum_imp, df['implied_prob'])

    results = []
    all_dates = sorted(df['event_date'].dropna().unique().tolist())
    calibrators = st.session_state.get('calibrators', {}) if use_calibration else {}

    for d in all_dates:
        d_ts = pd.to_datetime(d).normalize()
        # Precompute lambdas per market at date d (using only games before d)
        lambda_maps = {mk: _weighted_lambdas_asof(counts_frames.get(mk, pd.DataFrame()), d_ts)
                       for mk in SUPPORTED_MARKETS}
        # Global fallback mean per market if unseen player
        global_means = {}
        for mk, cdf in counts_frames.items():
            if cdf is None or cdf.empty:
                global_means[mk] = 0.0
            else:
                mask_hist = cdf['game_date'] < d_ts
                global_means[mk] = cdf.loc[mask_hist, 'count'].mean() if mask_hist.any() else 0.0

        day_rows = df[df['event_date'] == d_ts]
        for _, row in day_rows.iterrows():
            mk = row['market_key']
            pkey = row['player_fullkey']
            lam = lambda_maps.get(mk, {}).get(pkey, global_means.get(mk, 0.0))
            if lam is None or lam <= 0:
                continue
            # Poisson simulation
            sim = np.random.poisson(lam, DEFAULT_N_SIMULATIONS)
            mu, sigma = sim.mean(), sim.std()
            line = float(row['line'])
            side = row['side']
            p_over = np.mean(sim > line)
            p_under = np.mean(sim < line)
            if use_calibration and calibrators.get(mk) is not None:
                p_over = calibrators[mk].predict(np.array([p_over]))[0]
                p_under = calibrators[mk].predict(np.array([p_under]))[0]
            p_side = p_over if side == 'Over' else p_under
            implied_prob_used = row.get('implied_prob_fair', row.get('implied_prob', np.nan))
            z_vs_line = (mu - line) / (sigma + 1e-9)
            ok_vol = (sigma / max(mu, 1e-9)) <= VOL_THR
            ok_z = abs(z_vs_line) >= Z_THR
            edge_prob = p_side - implied_prob_used if pd.notna(implied_prob_used) else np.nan
            edge_pct = edge_prob * 100.0 if pd.notna(edge_prob) else np.nan
            take = (pd.notna(edge_prob) and edge_prob >= EDGE_THR and ok_vol and ok_z)

            # Actual result from counts on the same calendar date
            counts_df = counts_frames.get(mk, pd.DataFrame())
            actual_row = counts_df[(counts_df['player_fullkey'] == pkey) & (counts_df['game_date'] == d_ts)]
            actual = actual_row['count'].iloc[0] if len(actual_row) else np.nan
            outcome = _resolve_outcome(side, actual, line) if take else 'No Bet'
            pnl = 0.0
            risk = 0.0
            if outcome in ('Win','Loss','Push'):
                risk = 1.0
                if outcome == 'Win':
                    pnl = _american_profit_per_risk_unit(row['odds'])
                elif outcome == 'Loss':
                    pnl = -1.0
                else:
                    pnl = 0.0

            results.append({
                'event_date': d_ts,
                'player': row.get('player', ''),
                'player_fullkey': pkey,
                'market_key': mk,
                'side': side,
                'book_line': line,
                'odds': row['odds'],
                'implied_prob_fair': row.get('implied_prob_fair', np.nan),
                'model_lambda': lam,
                'model_mean': mu,
                'model_std': sigma,
                'p_side': p_side,
                'edge_prob': edge_prob,
                'edge_pct': edge_pct,
                'implied_prob_used': implied_prob_used,
                'z_vs_line': z_vs_line,
                'vol_ratio': (sigma / max(mu, 1e-9)),
                'actual_count': actual,
                'bet_taken': bool(take),
                'outcome': outcome,
                'pnl_risk1u': pnl,
                'risked_units': risk
            })

    res_df = pd.DataFrame(results)
    # Aggregate summary on bets actually taken
    taken = res_df[res_df['bet_taken'] == True]
    summary = {}
    if not taken.empty:
        wins = (taken['outcome'] == 'Win').sum()
        losses = (taken['outcome'] == 'Loss').sum()
        pushes = (taken['outcome'] == 'Push').sum()
        n = len(taken)
        roi = taken['pnl_risk1u'].sum() / max(taken['risked_units'].sum(), 1e-9)
        summary = {
            'bets': int(n),
            'wins': int(wins),
            'losses': int(losses),
            'pushes': int(pushes),
            'win_rate': (wins / n) if n else 0.0,
            'roi_per_unit_risked': roi,
            'avg_edge': float(taken['edge'].mean(skipna=True)) if 'edge' in taken else np.nan,
        }
    return res_df, summary
def compute_edges_hist() -> pd.DataFrame:
    """Compute picks using historical means only (no odds)."""
    results = []
    models = st.session_state.get('model_means', train_models())
    raw_means = st.session_state.get('raw_means', {})
    ensemble_models = st.session_state.get('ensemble_models', {})

    for market, players in models.items():
        for player, lam in players.items():
            try:
                lam_base = lam
                if ensemble_models.get(market):
                    r = raw_means.get(market, {}).get(player, lam_base)
                    lam_adj = ensemble_models[market].predict([[r, lam_base]])[0]
                else:
                    lam_adj = lam_base
                sim = np.random.poisson(lam_adj, DEFAULT_N_SIMULATIONS)
                mu, sigma = sim.mean(), sim.std()
                # Approximate fair odds
                implied_p = 0.5
                p_over = np.mean(sim > lam_adj)
                # apply calibration if available
                cal = st.session_state.get('calibrators', {}).get(market)
                if cal is not None:
                    p_over = cal.predict(np.array([p_over]))[0]
                p_under = np.mean(sim < lam_adj)
                edge = p_over - implied_p if p_over >= p_under else p_under - implied_p
                side = 'Over' if p_over >= p_under else 'Under'
                rec = side if (abs(edge) >= EDGE_THRESHOLD and sigma/mu <= VOLATILITY_RATIO_THRESHOLD) else 'No Action'
            except:
                lam_adj, mu, sigma, edge, rec = np.nan, np.nan, np.nan, np.nan, 'Error'
            results.append({
                'player': player,
                'team': None,
                'market_key': market,
                'line': lam_adj,
                'odds': None,
                'implied_prob': implied_p,
                'status': 'N/A',
                'book_name': 'Historical',
                'lambda': lam_adj,
                'model_mean': mu,
                'model_std': sigma,
                'edge_pct': edge,
                'recommendation': rec
            })
    df_out = pd.DataFrame(results)
    df_out['player_short'] = df_out['player'].apply(short_name)
    df_out['player_merge'] = df_out['player_short'].apply(merge_key_from_short)
    df_out['last_name_merge'] = df_out['player'].apply(last_name_key)
    df_out['player_fullkey'] = df_out['player'].apply(full_name_key)
    # Enrich team from roster
    roster = load_rosters()
    if not roster.empty:
        df_out = df_out.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
        df_out['team'] = df_out.get('team').astype('string') if 'team' in df_out.columns else pd.Series([pd.NA]*len(df_out), dtype='string')
        df_out['team_abbr'] = df_out['team_abbr'].astype('string')
        mask = df_out['team'].isna() | (df_out['team'] == '')
        df_out.loc[mask, 'team'] = df_out.loc[mask, 'team_abbr']
        df_out = df_out.drop(columns=['team_abbr'])
        df_out['team'] = df_out['team'].astype('string')
    return df_out

# ---------------------- Display & Download ---------------------- #
def display_table(df: pd.DataFrame, key_prefix: str = ""):
    if 'edge_pct' not in df.columns and 'edge_prob' in df.columns:
        df = df.assign(edge_pct=df['edge_prob'] * 100.0)
    cols_base = ['player','team','market_key','side','line','delta_vs_line','z_vs_line','odds','edge_pct','model_mean','recommendation','book_name']
    # include status column if present
    cols = [c for c in (cols_base + ['status']) if c in df.columns]
    if 'team' not in df.columns:
        df = df.assign(team=np.nan)
    st.dataframe(df[cols], use_container_width=True)
    # Downloads
    st.download_button('Download CSV', df.to_csv(index=False), 'props.csv', 'text/csv', key=f"{key_prefix}_dl_csv")
    st.download_button('Download JSON', df.to_json(orient='records', indent=2), 'props.json', 'application/json', key=f"{key_prefix}_dl_json")

# ---------------------- Audit Logging ---------------------- #
def log_picks(df: pd.DataFrame):
    path = os.path.join(LOG_DIR, f'picks_{datetime.date.today().isoformat()}.json')
    try:
        with open(path, 'w') as f:
            json.dump(df.to_dict(orient='records'), f, indent=2)
    except:
        pass

# ---------------------- Main App ---------------------- #
def main():
    st.set_page_config(page_title="FoxEdge NFL Props", layout="wide")
    st.title("FoxEdge NFL Player Prop Recommendations")

    # Debug toggle
    dbg_toggle = st.sidebar.checkbox("Verbose debug", value=st.session_state.get('__debug_enabled', DEBUG_DEFAULT))
    set_debug(dbg_toggle)

    # Sidebar controls
    if 'model_means' not in st.session_state:
        st.session_state['model_means'] = {}
    # Historical-only toggle
    hist_only = st.sidebar.checkbox("Use historical predictions only", key='hist_only')
    # Train models button
    if st.sidebar.button("Train Models"):
        with st.spinner("Training models..."):
            train_models()
            st.sidebar.success("Models trained.")

    if st.sidebar.button("Run Back-Test & Calibrate"):
        with st.spinner("Running back-test and calibration..."):
            backtest_and_calibrate()
            st.sidebar.success("Calibration complete.")

    # ---- Threshold controls ----
    st.sidebar.markdown("### Thresholds")
    edge_thr = st.sidebar.slider(
        "Min edge (prob)",
        min_value=0.00, max_value=0.10,
        value=float(st.session_state.get('edge_threshold', EDGE_THRESHOLD)),
        step=0.005
    )
    vol_thr = st.sidebar.slider(
        "Max volatility ratio (sigma/mean)",
        min_value=0.50, max_value=2.00,
        value=float(st.session_state.get('volatility_ratio', VOLATILITY_RATIO_THRESHOLD)),
        step=0.05
    )
    zmin_thr = st.sidebar.slider(
        "Min |z| vs line",
        min_value=0.00, max_value=3.00,
        value=float(st.session_state.get('z_min', Z_MIN)),
        step=0.05
    )
    st.session_state['edge_threshold'] = edge_thr
    st.session_state['volatility_ratio'] = vol_thr
    st.session_state['z_min'] = zmin_thr

    st.markdown('---')
    st.subheader('Walk-Forward Backtest vs Real Lines')
    with st.expander('Run historical walk-forward backtest (no leakage)', expanded=False):
        col_a, col_b, col_c = st.columns([1,1,1])
        with col_a:
            year_start = st.number_input('Start Year', min_value=2015, max_value=datetime.date.today().year, value=2020, step=1)
        with col_b:
            year_end = st.number_input('End Year', min_value=2015, max_value=datetime.date.today().year, value=datetime.date.today().year, step=1)
        with col_c:
            use_cal = st.checkbox('Use calibration (if available)', value=False)
        hist_odds_files = st.file_uploader('Upload Historical Odds CSVs (Prop Detail or OddsAPI)', type=['csv'], key='hist_odds_csv', accept_multiple_files=True)
        run_bt = st.button('Run Walk-Forward Backtest', type='primary', use_container_width=False)
        if run_bt:
            if not hist_odds_files:
                st.error('Please upload at least one historical odds CSV.')
            else:
                with st.spinner('Loading PBP and building counts...'):
                    pbp_bt = load_pbp_years_for_backtest(int(year_start), int(year_end))
                    if pbp_bt.empty:
                        st.error('No PBP data loaded for the selected years.')
                    else:
                        counts_frames = build_counts_frames(pbp_bt)
                        # Parse all uploaded odds CSVs and concatenate
                        frames = []
                        for f in hist_odds_files:
                            df_i = process_odds_csv(f)
                            if not df_i.empty:
                                frames.append(df_i)
                        if not frames:
                            st.error('None of the uploaded CSVs matched a supported odds schema.')
                        else:
                            df_hist_odds = pd.concat(frames, ignore_index=True)
                            # optional sanity: drop exact dupes
                            df_hist_odds = df_hist_odds.drop_duplicates()
                            if df_hist_odds.empty:
                                st.error('Historical odds CSVs could not be parsed. Check required columns.')
                            else:
                                res_df, summary = run_walkforward_backtest(df_hist_odds, counts_frames, use_calibration=use_cal)
                                if res_df.empty:
                                    st.warning('Backtest produced no results (check filters and CSV contents).')
                                else:
                                    st.markdown('**Backtest Summary (bets actually taken):**')
                                    if summary:
                                        st.write({
                                            'bets': summary.get('bets', 0),
                                            'wins': summary.get('wins', 0),
                                            'losses': summary.get('losses', 0),
                                            'pushes': summary.get('pushes', 0),
                                            'win_rate': round(summary.get('win_rate', 0.0), 3),
                                            'roi_per_unit_risked': round(summary.get('roi_per_unit_risked', 0.0), 3),
                                            'avg_edge': round(summary.get('avg_edge', 0.0), 3),
                                        })
                                    st.dataframe(res_df, use_container_width=True)
                                    st.download_button('Download Backtest Results (CSV)', res_df.to_csv(index=False), 'walkforward_results.csv', 'text/csv', key='dl_bt_csv')

    if hist_only:
        df_hist = compute_edges_hist()
        # Merge roster team info for historical baseline
        roster = load_rosters()
        if not roster.empty and 'player_fullkey' in df_hist.columns:
            df_hist = df_hist.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
            if 'team' not in df_hist.columns:
                df_hist['team'] = df_hist['team_abbr']
            else:
                mask_h = df_hist['team'].isna()
                df_hist.loc[mask_h, 'team'] = df_hist.loc[mask_h, 'team_abbr']
            df_hist = df_hist.drop(columns=['team_abbr'])
        df_recs = df_hist[df_hist['recommendation'].isin(['Over','Under'])]
        display_table(df_recs, key_prefix="hist_only")
        log_picks(df_recs)
    else:
        # Always compute and show historical baseline first
        df_hist = compute_edges_hist()
        # Merge roster team info for historical baseline
        roster = load_rosters()
        if not roster.empty and 'player_fullkey' in df_hist.columns:
            df_hist = df_hist.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
            if 'team' not in df_hist.columns:
                df_hist['team'] = df_hist['team_abbr']
            else:
                mask_h = df_hist['team'].isna()
                df_hist.loc[mask_h, 'team'] = df_hist.loc[mask_h, 'team_abbr']
            df_hist = df_hist.drop(columns=['team_abbr'])
        st.subheader("Historical Baseline (no odds)")
        display_table(df_hist, key_prefix="hist_baseline")

        # Then allow odds upload to INSERT odds/lines into the baseline and compute edges
        odds_file = st.sidebar.file_uploader("Upload Odds CSV (optional)", type=['csv'])
        if not odds_file:
            st.info("Optionally upload an odds CSV to insert prices/lines and compute market edges against the historical baseline.")
            return

        df_odds = process_odds_csv(odds_file)
        if df_odds.empty:
            st.warning("Uploaded odds CSV could not be parsed. Check required columns.")
            return

        # ---- De-vig: fair implied probability when both sides exist ----
        if {'side','implied_prob','player_fullkey','market_key','book_name','line'}.issubset(df_odds.columns):
            grp_keys = ['player_fullkey','market_key','book_name','line']
            side_count = df_odds.groupby(grp_keys)['side'].transform('nunique')
            sum_imp = df_odds.groupby(grp_keys)['implied_prob'].transform('sum')
            df_odds['implied_prob_fair'] = np.where(side_count >= 2,
                                                    df_odds['implied_prob'] / sum_imp,
                                                    df_odds['implied_prob'])
            st.caption(f"De-vig coverage: fair probabilities computed for {int((side_count >= 2).sum())} rows.")

        # Ensure short keys exist
        if 'player_short' not in df_odds.columns:
            df_odds['player_short'] = df_odds['player'].apply(short_name)
        if 'player_short' not in df_hist.columns:
            df_hist['player_short'] = df_hist['player'].apply(short_name)
        # Build merge frames with robust key
        if 'player_merge' not in df_odds.columns:
            df_odds['player_merge'] = df_odds['player_short'].apply(merge_key_from_short)
        if 'player_merge' not in df_hist.columns:
            df_hist['player_merge'] = df_hist['player_short'].apply(merge_key_from_short)
        if 'player_fullkey' not in df_odds.columns:
            df_odds['player_fullkey'] = df_odds['player'].apply(full_name_key)
        if 'player_fullkey' not in df_hist.columns:
            df_hist['player_fullkey'] = df_hist['player'].apply(full_name_key)

        # ---- Diagnostics: name & market overlap ----
        try:
            names_hist = set(df_hist.get('player_merge', pd.Series(dtype=str)).dropna().unique())
            names_odds = set(df_odds.get('player_merge', pd.Series(dtype=str)).dropna().unique())
            overlap = len(names_hist & names_odds)
            st.caption(f"Name-key overlap: {overlap} hist vs odds (hist={len(names_hist)}, odds={len(names_odds)})")
            mk_hist = set(df_hist.get('market_key', pd.Series(dtype=str)).dropna().unique())
            mk_odds = set(df_odds.get('market_key', pd.Series(dtype=str)).dropna().unique())
            st.caption(f"Market overlap: {len(mk_hist & mk_odds)} keys; hist={sorted(list(mk_hist))[:6]}..., odds={sorted(list(mk_odds))[:6]}...")
            # Roster coverage note
            roster = load_rosters()
            if not roster.empty:
                cov = df_odds['player_fullkey'].isin(roster['player_fullkey']).sum()
                st.caption(f"Roster coverage on odds players: {cov}/{len(df_odds)}")
            if overlap == 0:
                # show a few samples to help debugging
                st.warning("No player name overlap detected. Check that odds CSV 'description' contains player names in a parseable format.")
                st.write("Sample hist keys:", list(sorted(names_hist))[:10])
                st.write("Sample odds keys:", list(sorted(names_odds))[:10])
        except Exception:
            pass

        odds_keep = list(dict.fromkeys([
            'player_merge', 'player_short', 'last_name_merge', 'market_key',
            'team', 'line', 'odds', 'implied_prob', 'implied_prob_fair', 'book_name', 'side'
        ]))
        odds_keep = [c for c in odds_keep if c in df_odds.columns]

        # Start from historical frame
        df_merge = df_hist.copy()

        # 1) Full-name exact match merge
        if 'player_fullkey' in df_odds.columns and 'player_fullkey' in df_merge.columns:
            odds_full = df_odds[odds_keep + ['player_fullkey']].copy()
            df_full = df_merge.merge(odds_full, on=['player_fullkey','market_key'], how='left', suffixes=('', '_full'))
            # OVERWRITE historical line with odds line when available
            if 'line_full' in df_full.columns:
                m_line = df_full['line_full'].notna()
                df_full.loc[m_line, 'line'] = df_full.loc[m_line, 'line_full']
            # Backfill other fields only where missing
            for col in ['team','odds','implied_prob','book_name','side']:
                full_col = f"{col}_full"
                if full_col in df_full.columns:
                    if col not in df_full.columns:
                        df_full[col] = np.nan
                    mask = df_full[col].isna()
                    df_full.loc[mask, col] = df_full.loc[mask, full_col]
            # drop helper columns from this stage
            drop_cols = [c for c in df_full.columns if c.endswith('_full')]
            df_merge = df_full.drop(columns=drop_cols)

        # 2) Short-name key merge for anything still missing
        odds_short = df_odds[odds_keep].copy()
        df_short = df_merge.merge(odds_short, on=['player_merge','market_key'], how='left', suffixes=('', '_short'))
        # OVERWRITE historical line with odds line when available
        if 'line_short' in df_short.columns:
            m_line = df_short['line_short'].notna()
            df_short.loc[m_line, 'line'] = df_short.loc[m_line, 'line_short']
        # Backfill other fields only where missing
        for col in ['team','odds','implied_prob','book_name','side']:
            s_col = f"{col}_short"
            if s_col in df_short.columns:
                if col not in df_short.columns:
                    df_short[col] = np.nan
                mask = df_short[col].isna()
                df_short.loc[mask, col] = df_short.loc[mask, s_col]
        drop_cols = [c for c in df_short.columns if c.endswith('_short')]
        df_merge = df_short.drop(columns=drop_cols)

        # Backfill team from roster if still missing
        roster = load_rosters()
        if not roster.empty and 'player_fullkey' in df_merge.columns:
            df_merge = df_merge.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
            if 'team' not in df_merge.columns:
                df_merge['team'] = df_merge['team_abbr']
            else:
                mask_t = df_merge['team'].isna()
                df_merge.loc[mask_t, 'team'] = df_merge.loc[mask_t, 'team_abbr']
            df_merge = df_merge.drop(columns=['team_abbr'])

        # Fallback: try last-name-only merge for unique last names per market
        try:
            total = len(df_hist)
            matched = df_merge['odds'].notna().sum() if 'odds' in df_merge.columns else 0
            if total and (matched / total) < 0.10:
                # keep only unique last names per market on both sides to avoid collisions
                hist_uni = df_hist.copy()
                odds_uni = df_odds[odds_keep].copy()
                cnt_h = hist_uni.groupby(['last_name_merge','market_key']).size().reset_index(name='n_h')
                cnt_o = odds_uni.groupby(['last_name_merge','market_key']).size().reset_index(name='n_o')
                uniq_h = cnt_h[(cnt_h['n_h'] == 1) & (cnt_h['last_name_merge'] != '')]
                uniq_o = cnt_o[(cnt_o['n_o'] == 1) & (cnt_o['last_name_merge'] != '')]
                hist_join = hist_uni.merge(uniq_h[['last_name_merge','market_key']], on=['last_name_merge','market_key'], how='inner')
                odds_join = odds_uni.merge(uniq_o[['last_name_merge','market_key']], on=['last_name_merge','market_key'], how='inner')
                df_fallback = hist_join.merge(odds_join, on=['last_name_merge','market_key'], how='left', suffixes=('', '_fb'))
                # fill only where primary merge failed
                for col in ['team','line','odds','implied_prob','book_name','side']:
                    src = col if col in df_fallback.columns else f"{col}_fb"
                    if src in df_fallback.columns:
                        df_merge[col] = df_merge[col].combine_first(df_fallback[src])
                # Ensure unified team column exists after fallback application
                if 'team' not in df_merge.columns and 'team_fb' in df_fallback.columns:
                    df_merge['team'] = df_fallback['team_fb']
                elif 'team' in df_merge.columns and 'team_fb' in df_fallback.columns:
                    df_merge['team'] = df_merge['team'].combine_first(df_fallback['team_fb'])
                # re-evaluate match rate after fallback
                matched2 = df_merge['odds'].notna().sum() if 'odds' in df_merge.columns else 0
                rate2 = (matched2 / total * 100) if total else 0
                if matched2 > matched:
                    st.caption(f"Fallback last-name merge improved matches: {matched} → {matched2} ({rate2:.1f}%).")
        except Exception:
            pass

        # Authoritative book line from CSV (column `point`) is already mapped to `line` during parsing.
        # Preserve explicitly as `book_line`; do NOT substitute historical into book_line.
        if 'line' in df_merge.columns:
            df_merge['book_line'] = df_merge['line']
        else:
            df_merge['book_line'] = np.nan
        # For market-adjusted display, set row 'line' equal to the CSV point
        df_merge['line'] = df_merge['book_line']

        if 'odds' not in df_merge.columns and 'odds_hist' in df_merge.columns:
            df_merge['odds'] = df_merge['odds_hist']
        if 'implied_prob' not in df_merge.columns and 'implied_prob_hist' in df_merge.columns:
            df_merge['implied_prob'] = df_merge['implied_prob_hist']
        # If implied_prob missing but odds present, compute it
        if 'implied_prob' in df_merge.columns and df_merge['implied_prob'].isna().any() and 'odds' in df_merge.columns:
            df_merge['implied_prob'] = df_merge['implied_prob'].fillna(df_merge['odds'].apply(odds_to_prob))

        # If implied_prob column is entirely missing, build it from odds now
        if 'implied_prob' not in df_merge.columns and 'odds' in df_merge.columns:
            df_merge['implied_prob'] = df_merge['odds'].apply(odds_to_prob)

        # Recompute fair probabilities on the merged set so candidates always have implied_prob_fair when both sides exist
        grp_keys = ['player_fullkey','market_key','book_name','book_line']
        missing_keys = [k for k in grp_keys if k not in df_merge.columns]
        if not missing_keys and 'side' in df_merge.columns and 'implied_prob' in df_merge.columns:
            side_count_m = df_merge.groupby(grp_keys)['side'].transform('nunique')
            sum_imp_m = df_merge.groupby(grp_keys)['implied_prob'].transform('sum')
            df_merge['implied_prob_fair'] = np.where(side_count_m >= 2, df_merge['implied_prob'] / sum_imp_m, df_merge['implied_prob'])

        # If team still missing, default to historical team or keep as NaN
        if 'team' not in df_merge.columns and 'team_hist' in df_merge.columns:
            df_merge['team'] = df_merge['team_hist']

        # Default book name for historical rows
        if 'book_name' not in df_merge.columns:
            df_merge['book_name'] = 'Historical'
        df_merge['book_name'] = df_merge['book_name'].fillna('Historical')

        # Merge status and compute final recommendations
        status_df = load_status()
        if not status_df.empty and 'player' in df_merge.columns:
            df_merge = df_merge.merge(status_df, on='player', how='left')
            df_merge['status'] = df_merge['status'].fillna('Unknown')
        else:
            if 'status' not in df_merge.columns:
                df_merge['status'] = 'Unknown'

        # Score only rows that truly have market context (book odds + declared side)
        required_cols = ['odds','side','book_line']
        for rc in required_cols:
            if rc not in df_merge.columns:
                df_merge[rc] = np.nan
        df_candidates = df_merge[df_merge['odds'].notna() & df_merge['side'].isin(['Over','Under']) & df_merge['book_line'].notna()].copy()
        if 'implied_prob' not in df_candidates.columns and 'odds' in df_candidates.columns:
            df_candidates['implied_prob'] = df_candidates['odds'].apply(odds_to_prob)
        if df_candidates.empty:
            st.warning('No candidate rows with odds, side, and line after merge. Check CSV mapping or lower-case column names.')
        df_results = compute_edges_csv(df_candidates)

        # Diagnostics summary for why no picks
        total_rows = len(df_candidates)
        st.caption(f"Candidates scored: {total_rows}")
        if total_rows == 0:
            st.warning('No rows produced by edge computation. Check parsing and thresholds.')
        else:
            taken = df_results[df_results['recommendation'].isin(['Over','Under'])]
            if taken.empty:
                st.info('No bets passed the guardrails. Showing diagnostics for the top candidates by edge.')
            # Show counts by failure reason
            try:
                fail_edge = (~df_results['pass_edge']).sum() if 'pass_edge' in df_results else 0
                fail_vol = (~df_results['pass_vol']).sum() if 'pass_vol' in df_results else 0
                fail_z = (~df_results['pass_z']).sum() if 'pass_z' in df_results else 0
                st.caption(f"Diagnostics — total rows: {total_rows}, pass_edge fails: {fail_edge}, pass_vol fails: {fail_vol}, pass_z fails: {fail_z}")
            except Exception:
                pass

            # Optional view: top N by edge regardless of guardrails
            with st.expander('View top candidates by edge (ignoring guardrails)', expanded=False):
                cols_diag = ['player','team','market_key','side','book_line','line','edge_pct','edge_prob','p_over','p_under','implied_prob_used','z_vs_line','model_mean','model_std','odds','book_name']
                cols_diag = [c for c in cols_diag if c in df_results.columns]
                topN = df_results.sort_values('edge_pct', ascending=False, na_position='last').head(25)
                if not topN.empty:
                    st.dataframe(topN[cols_diag], use_container_width=True)
                else:
                    st.write('No candidates to show.')

        df_recs = df_results[df_results['recommendation'].isin(['Over','Under'])]
        st.subheader("Market-Adjusted Picks (after inserting odds)")
        display_table(df_recs, key_prefix="market_adjusted")
        log_picks(df_recs)

if __name__ == '__main__':
    main()
