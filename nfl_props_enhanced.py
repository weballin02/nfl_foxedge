#!/usr/bin/env python3
"""
nfl_props_enhanced_v2.py

FoxEdge NFL player prop recommendations — enhanced version with unit alignment.
Major additions vs prior version:
- Automatic bookmaker LINE UNIT RESCALING (detects 10x/100x mismatches vs model scale)
- Records raw vs scaled line and chosen scale_factor for auditability
- Uses Negative Binomial simulation, EB shrink, team context, line-aware calibration (as before)
- Tighter guardrails and bootstrap edge CIs (as before)

Run:
  pip install streamlit nfl_data_py feedparser scikit-learn pandas numpy
  streamlit run nfl_props_enhanced_v2.py
"""

import os
import re
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import nfl_data_py as ndp
import feedparser
from sklearn.isotonic import IsotonicRegression

# ------------- Configuration ------------- #
SUPPORTED_MARKETS = [
    "player_pass_attempts",
    "player_receptions",
    "player_rush_attempts"
]

DEFAULT_N_SIMULATIONS = 20000
HALF_LIFE_DAYS = 28
DECAY_RATE = np.log(2) / HALF_LIFE_DAYS

# Guardrails
EDGE_THRESHOLD = 0.06
VOLATILITY_RATIO_THRESHOLD = 1.00
Z_MIN = 1.00

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

DEBUG_DEFAULT = False
NFL_TEAMS_3 = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"}

# Determinism
np.random.seed(1337)

# ------------- Utilities ------------- #
def odds_to_prob(odds):
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)

TEAM_RE = re.compile(r"\b([A-Z]{2,4})\b")
def infer_team_from_text(txt: str) -> str | float:
    if not isinstance(txt, str) or not txt:
        return np.nan
    m = re.search(r"\(([A-Z]{2,4})\)", txt)
    if m and m.group(1) in NFL_TEAMS_3:
        return m.group(1)
    for tok in TEAM_RE.findall(txt.upper()):
        if tok in NFL_TEAMS_3:
            return tok
    return np.nan

from pandas.errors import EmptyDataError, ParserError
def _seek0(f):
    try:
        f.seek(0)
    except Exception:
        pass
    return f

def read_csv_safe(uploaded_file, **kwargs) -> pd.DataFrame:
    defaults = dict(engine="python", sep=None)
    defaults.update(kwargs or {})
    try:
        f = _seek0(uploaded_file)
        return pd.read_csv(f, **defaults)
    except EmptyDataError:
        return pd.DataFrame()
    except ParserError:
        try:
            f = _seek0(uploaded_file)
            return pd.read_csv(f, engine="python", sep="\t")
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

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
        st.markdown(f"**DEBUG:** {title}")
        view = df[cols].head(n) if cols else df.head(n)
        st.dataframe(view, use_container_width=True, key=(key or f"dbg_{title.replace(' ','_')}"))
        st.caption(f"shape={df.shape}, cols={list(df.columns)}")
    except Exception:
        pass

# Name normalization
def short_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}".strip()
    s = re.sub(r",?(\s+(Jr\.|Sr\.|II|III|IV))$", "", s, flags=re.IGNORECASE)
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

def last_name_key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            s = f"{parts[1]} {parts[0]}"
    s = re.sub(r",?(\s+(Jr\.|Sr\.|II|III|IV))$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    last = s.split(" ")[-1]
    return re.sub(r"[^a-z]", "", last.lower())

def full_name_key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            s = f"{parts[1]} {parts[0]}"
    s = re.sub(r",?(\s+(Jr\.|Sr\.|II|III|IV))$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    toks = s.split(" ")
    if len(toks) >= 2:
        first, last = toks[0], toks[-1]
    else:
        first, last = toks[0], ""
    canon = f"{first} {last}".strip().lower()
    return re.sub(r"[^a-z ]", "", canon)

def merge_key_from_short(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z]", "", s.lower())

# ------------- Roster & Status ------------- #
@st.cache_data(show_spinner=False)
def load_rosters(years: list[int] | None = None) -> pd.DataFrame:
    try:
        if years is None or (isinstance(years, list) and len(years) == 0):
            season = datetime.date.today().year
        elif isinstance(years, list):
            season = max(years)
        else:
            season = int(years)

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

        def build_name(df: pd.DataFrame) -> pd.Series:
            if df is None or df.empty:
                return pd.Series([], dtype=str)
            if 'player_name' in df.columns:
                return df['player_name'].astype(str)
            if 'first_name' in df.columns and 'last_name' in df.columns:
                return (df['first_name'].astype(str) + ' ' + df['last_name'].astype(str)).str.strip()
            name_cols = [c for c in df.columns if 'name' in c.lower()]
            return df[name_cols[0]].astype(str) if name_cols else pd.Series([], dtype=str)

        def pick_team(df: pd.DataFrame) -> pd.Series:
            if df is None or df.empty:
                return pd.Series([], dtype=str)
            for c in ['team', 'recent_team', 'team_abbr']:
                if c in df.columns:
                    return df[c].astype(str)
            return pd.Series([np.nan] * len(df))

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

        base = None
        if not weekly_map.empty:
            base = weekly_map
            if not seasonal_map.empty:
                base = base.merge(seasonal_map[['player_fullkey','team_abbr']].rename(columns={'team_abbr':'team_abbr_se'}),
                                  on='player_fullkey', how='left')
                mask = base['team_abbr'].isna()
                base.loc[mask, 'team_abbr'] = base.loc[mask, 'team_abbr_se']
                base = base.drop(columns=['team_abbr_se'])
        elif not seasonal_map.empty:
            base = seasonal_map
            base['week'] = np.nan

        if base is None or base.empty:
            return pd.DataFrame(columns=['player_fullkey','player','team_abbr','season','week'])

        base = base.sort_values(['player_fullkey', 'week'], ascending=[True, False])
        base = base.drop_duplicates(subset=['player_fullkey'], keep='first')
        return base.reset_index(drop=True)
    except Exception as e:
        dbg(f"load_rosters error: {e}")
        return pd.DataFrame(columns=['player_fullkey','player','team_abbr','season','week'])

@st.cache_data(show_spinner=False)
def load_status():
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
    except Exception:
        return pd.DataFrame(columns=['player','status'])

def normalize_market_key(market_raw: str) -> str:
    if not isinstance(market_raw, str):
        return ""
    m = market_raw.strip().lower()
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
    if m in mapping:
        return mapping[m]
    if "pass" in m and "attempt" in m and "player" in m:
        return "player_pass_attempts"
    if "reception" in m and ("player" in m or "(player)" in m):
        return "player_receptions"
    if "rush" in m and "attempt" in m and ("player" in m or "(player)" in m):
        return "player_rush_attempts"
    return ""

# ------------- PBP helpers ------------- #
@st.cache_data(show_spinner=False)
def load_pbp_years(start_year: int, end_year: int) -> pd.DataFrame:
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
    pbp['game_date'] = pd.to_datetime(pbp['game_date']).dt.normalize()
    return pbp

def _counts_from_pbp(pbp: pd.DataFrame, market: str) -> pd.DataFrame:
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
    return {mk: _counts_from_pbp(pbp, mk) for mk in SUPPORTED_MARKETS}

def _weighted_lambdas_asof(counts_df: pd.DataFrame, as_of_date: pd.Timestamp) -> dict:
    if counts_df is None or counts_df.empty:
        return {}
    mask = counts_df['game_date'] < pd.to_datetime(as_of_date).normalize()
    hist = counts_df.loc[mask].copy()
    if hist.empty:
        return {}
    hist['days_diff'] = (pd.to_datetime(as_of_date).normalize() - hist['game_date']).dt.days
    hist['weight'] = np.exp(-DECAY_RATE * hist['days_diff'])
    hist['wprod'] = hist['count'] * hist['weight']
    agg = hist.groupby('player_fullkey', as_index=True)[['wprod','weight']].sum()
    weighted = (agg['wprod'] / agg['weight']).to_dict()
    return weighted

# ------------- Negative Binomial ------------- #
def estimate_nb_dispersion(counts: pd.Series, min_k: float = 0.25, max_k: float = 100.0) -> float:
    c = counts.dropna().astype(float)
    if len(c) < 100:
        return 100.0
    mu = c.mean()
    var = c.var(ddof=1)
    excess = max(var - mu, 1e-9)
    k = (mu * mu) / excess
    return float(np.clip(k, min_k, max_k))

@st.cache_data(show_spinner=False)
def estimate_market_dispersion(counts_frames: dict) -> dict:
    ks = {}
    for mk, cdf in counts_frames.items():
        if cdf is None or cdf.empty:
            ks[mk] = 100.0
        else:
            ks[mk] = estimate_nb_dispersion(cdf['count'])
    st.session_state['dispersion_k'] = ks
    return ks

def nb_rvs(mu: float, k: float, size: int) -> np.ndarray:
    n = max(k, 1e-6)
    p = n / (n + max(mu, 1e-9))
    return np.random.negative_binomial(n, p, size=size)

# ------------- Team context from PBP ------------- #
@st.cache_data(show_spinner=False)
def compute_team_context(pbp: pd.DataFrame, half_life_days:int=28) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    df = pbp.copy()
    df['game_date'] = pd.to_datetime(df['game_date']).dt.normalize()
    per_game = df.groupby(['posteam','game_id','game_date']).size().reset_index(name='plays')
    per_game_pass = df[df.get('play_type')=='pass'].groupby(['posteam','game_id']).size().reset_index(name='pass_plays')
    g = per_game.merge(per_game_pass, on=['posteam','game_id'], how='left').fillna({'pass_plays':0})
    current_ts = pd.Timestamp(datetime.date.today())
    decay = np.log(2)/half_life_days
    g['days_diff'] = (current_ts - pd.to_datetime(g['game_date'])).dt.days
    g['w'] = np.exp(-decay * g['days_diff'])
    agg = g.groupby('posteam').apply(
        lambda x: pd.Series({
            'recent_plays_per_game': (x['plays']*x['w']).sum()/max(x['w'].sum(),1e-9),
            'recent_pass_rate': (x['pass_plays']*x['w']).sum()/max((x['plays']*x['w']).sum(),1e-9)
        })
    ).reset_index().rename(columns={'posteam':'team'})
    agg['recent_rush_rate'] = 1.0 - agg['recent_pass_rate']
    return agg

def apply_team_context_lambda(lam_base: float, market: str, team: str, ctx_df: pd.DataFrame) -> float:
    try:
        row = ctx_df[ctx_df['team']==team].iloc[0]
    except Exception:
        return lam_base
    plays_mult = row['recent_plays_per_game'] / max(ctx_df['recent_plays_per_game'].mean(), 1e-9)
    pass_mult = row['recent_pass_rate'] / max(ctx_df['recent_pass_rate'].mean(), 1e-9)
    rush_mult = row['recent_rush_rate'] / max(ctx_df['recent_rush_rate'].mean(), 1e-9)
    if market == "player_pass_attempts":
        return lam_base * plays_mult * pass_mult
    elif market == "player_rush_attempts":
        return lam_base * plays_mult * rush_mult
    elif market == "player_receptions":
        return lam_base * (0.5*plays_mult + 0.5*pass_mult)
    return lam_base

# ------------- Empirical Bayes shrinkage ------------- #
def eb_shrink_lambda(player_mu: float, market_mu: float, n_games: int, tau: float = 6.0) -> float:
    w = n_games / float(n_games + tau)
    return w*player_mu + (1.0 - w)*market_mu

# ------------- Model training (means & games played) ------------- #
@st.cache_data(show_spinner=False)
def train_models():
    hist_means = {}
    games_counts = {}
    current_year = datetime.date.today().year
    years = list(range(2020, current_year + 1))
    frames = []
    for y in years:
        try:
            df_y = ndp.import_pbp_data(years=[y])
            if df_y is not None and len(df_y):
                frames.append(df_y)
                dbg(f"{y} loaded")
        except Exception as e:
            dbg(f"Skipping {y}: {e}")
            continue
    if not frames:
        raise RuntimeError("No PBP data available.")
    pbp_df = pd.concat(frames, ignore_index=True)
    pbp_df['game_date'] = pd.to_datetime(pbp_df['game_date']).dt.normalize()
    current_ts = pd.Timestamp(datetime.date.today())

    for market in SUPPORTED_MARKETS:
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
            games_counts[market] = {}
        else:
            counts = (
                events.groupby([player_col, 'game_id', 'game_date'])
                      .size()
                      .reset_index(name='count')
            )
            gp = counts.groupby(player_col)['game_id'].nunique().to_dict()
            games_counts[market] = gp

            counts['days_diff'] = (current_ts - counts['game_date']).dt.days
            counts['weight'] = np.exp(-DECAY_RATE * counts['days_diff'])
            counts['wprod'] = counts['count'] * counts['weight']
            agg = counts.groupby(player_col, as_index=True)[['wprod','weight']].sum()
            weighted = (agg['wprod'] / agg['weight']).to_dict()
            hist_means[market] = weighted

    st.session_state['model_means'] = hist_means
    st.session_state['games_counts'] = games_counts
    return hist_means, games_counts

# ------------- Odds CSV processing (Prop Detail + Generic) ------------- #
def process_odds_csv_propdetails_format(uploaded_file) -> pd.DataFrame:
    df_raw = read_csv_safe(uploaded_file)
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    cmap = {c.lower().strip(): c for c in df_raw.columns}
    def pick(*aliases):
        for a in aliases:
            if a in cmap:
                return cmap[a]
        return None

    col_player = pick('player','playername','name','athlete','athlete_name')
    col_market = pick('market','prop','bet type','bet','wager type','stat')
    col_line   = pick('line','point','number','threshold')
    col_team   = pick('team','team_abbr','teamabbr','team code')
    col_book   = pick('book','bookmaker','sportsbook','sports book')
    col_date   = pick('date','event_date','game_date','commence_time','timestamp','last_update')

    over_cols = [c for c in df_raw.columns if re.search(r'\bover\b.*(odds|price|american|us)', c, flags=re.IGNORECASE)]
    under_cols = [c for c in df_raw.columns if re.search(r'\bunder\b.*(odds|price|american|us)', c, flags=re.IGNORECASE)]

    if not col_player or not col_market or not col_line or (not over_cols and not under_cols):
        return pd.DataFrame()

    base = pd.DataFrame()
    base['player'] = df_raw[col_player].astype(str)
    base['team'] = df_raw[col_team].astype(str) if col_team else np.nan
    base['market_key'] = df_raw[col_market].astype(str).apply(normalize_market_key)
    base['line_raw'] = pd.to_numeric(df_raw[col_line], errors='coerce')
    base['book_name'] = df_raw[col_book].astype(str) if col_book else 'Unknown'

    if col_date:
        base['event_time'] = pd.to_datetime(df_raw[col_date], errors='coerce')
    else:
        base['event_time'] = pd.NaT
    base['event_date'] = pd.to_datetime(base['event_time']).dt.normalize()

    base['player_short'] = base['player'].apply(short_name)
    base['player_merge'] = base['player_short'].apply(merge_key_from_short)
    base['last_name_merge'] = base['player'].apply(last_name_key)
    base['player_fullkey'] = base['player'].apply(full_name_key)

    if 'team' not in base.columns or base['team'].isna().all():
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

    rows = []
    for idx, r in base.iterrows():
        over_price = np.nan
        for c in over_cols:
            v = pd.to_numeric(df_raw.loc[idx, c], errors='coerce')
            if pd.notna(v):
                over_price = v; break
        under_price = np.nan
        for c in under_cols:
            v = pd.to_numeric(df_raw.loc[idx, c], errors='coerce')
            if pd.notna(v):
                under_price = v; break
        common = r.to_dict()
        if pd.notna(over_price):
            rows.append({**common, 'side': 'Over', 'odds': over_price})
        if pd.notna(under_price):
            rows.append({**common, 'side': 'Under', 'odds': under_price})
    out = pd.DataFrame(rows)

    out = out[out['market_key'].isin(SUPPORTED_MARKETS)]
    out = out.dropna(subset=['player','line_raw','odds'])

    def prob_from_any(o):
        try:
            val = float(o)
        except:
            return np.nan
        if 1.0 < val < 20.0 and not (val >= 100 or val <= -100):
            return 1.0 / val
        return odds_to_prob(val)
    out['implied_prob'] = out['odds'].apply(prob_from_any)

    roster = load_rosters()
    if not roster.empty:
        out = out.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
        out['team'] = out.get('team').astype('string') if 'team' in out.columns else pd.Series([pd.NA]*len(out), dtype='string')
        out['team_abbr'] = out['team_abbr'].astype('string')
        mask = out['team'].isna() | (out['team'] == '')
        out.loc[mask, 'team'] = out.loc[mask, 'team_abbr']
        out = out.drop(columns=['team_abbr'])

    cols = ['player','player_short','player_merge','team','market_key','line_raw','odds','implied_prob','book_name','side','event_time','event_date','player_fullkey','last_name_merge']
    return out[cols]

def process_odds_csv_oddsapi_format(uploaded_file) -> pd.DataFrame:
    df = read_csv_safe(uploaded_file)
    dbg("Parsing odds CSV (generic format) ...")
    dbg_df(df, "Raw odds CSV (first 12 rows)", n=12, key="dbg_raw_odds")
    required_cols = {'market','label','price','point','bookmaker'}
    if not isinstance(df, pd.DataFrame) or df.empty or not required_cols.issubset(set(df.columns)):
        return pd.DataFrame()

    out = pd.DataFrame()
    out['book_name'] = df['bookmaker'].astype(str)
    out['odds'] = pd.to_numeric(df['price'], errors='coerce')
    out['line_raw'] = pd.to_numeric(df['point'], errors='coerce')

    def extract_side(row):
        txt = f"{row.get('description','')} {row.get('label','')}".strip()
        m = re.search(r"\b(Over|Under)\b", txt, flags=re.IGNORECASE)
        return m.group(1).title() if m else np.nan
    out['side'] = df.apply(extract_side, axis=1)

    def extract_player(row):
        desc = str(row.get('description', '')).strip()
        if not desc:
            return np.nan
        txt = re.sub(r"\s+(Over|Under)\b.*$", "", desc, flags=re.IGNORECASE)
        if "," in txt and not " - " in txt:
            parts = [p.strip() for p in txt.split(",")]
            cand = f"{parts[1]} {parts[0]}".strip() if len(parts)>=2 else txt
        else:
            cand = txt
        if " - " in cand:
            chunks = [c.strip() for c in cand.split(" - ")]
            name_like = []
            name_re = re.compile(r"^[A-Za-z\.\'\-]+(?:\s+[A-Za-z\.\'\-]+)+$")
            for ch in chunks:
                if re.search(r"\([A-Z]{2,4}\)$", ch):
                    ch = re.sub(r"\s*\([A-Z]{2,4}\)$", "", ch)
                if name_re.match(ch):
                    name_like.append(ch)
            cand = name_like[-1] if name_like else max(chunks, key=len)
        cand = re.sub(r"\s*\([A-Z]{2,4}\)$", "", cand)
        cand = re.sub(r"^(Alt\s+|Alternate\s+)", "", cand, flags=re.IGNORECASE)
        return cand.strip()

    out['player'] = df.apply(extract_player, axis=1)
    out['player_short'] = out['player'].apply(short_name)
    out['player_merge'] = out['player_short'].apply(merge_key_from_short)
    out['last_name_merge'] = out['player'].apply(last_name_key)
    out['player_fullkey'] = out['player'].apply(full_name_key)

    if 'commence_time' in df.columns:
        out['event_time'] = pd.to_datetime(df['commence_time'], errors='coerce')
    elif 'last_update' in df.columns:
        out['event_time'] = pd.to_datetime(df['last_update'], errors='coerce')
    else:
        out['event_time'] = pd.NaT
    out['event_date'] = pd.to_datetime(out['event_time']).dt.normalize()

    def extract_team(row):
        txt = f"{row.get('label','')} {row.get('description','')}".strip()
        return infer_team_from_text(txt)
    out['team'] = df.apply(extract_team, axis=1)

    out['market_key'] = df['market'].apply(normalize_market_key)
    out = out[out['market_key'].isin(SUPPORTED_MARKETS)].copy()
    out = out.dropna(subset=['player','line_raw','odds'])

    roster = load_rosters()
    if not roster.empty:
        out = out.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
        out['team'] = out.get('team').astype('string') if 'team' in out.columns else pd.Series([pd.NA]*len(out), dtype='string')
        out['team_abbr'] = out['team_abbr'].astype('string')
        mask = out['team'].isna() | (out['team'] == '')
        out.loc[mask, 'team'] = out.loc[mask, 'team_abbr']
        out = out.drop(columns=['team_abbr'])

    out['implied_prob'] = out['odds'].apply(odds_to_prob)
    cols = ['player','player_short','player_merge','team','market_key','line_raw','odds','implied_prob','book_name','side','event_time','event_date','player_fullkey','last_name_merge']
    return out[cols]

def process_odds_csv(uploaded_file) -> pd.DataFrame:
    """Normalize odds CSV; preserve raw line, we'll scale later after seeing model lambda."""
    try:
        df_prop = process_odds_csv_propdetails_format(uploaded_file)
        if not df_prop.empty:
            return df_prop
    except Exception:
        pass

    try:
        df_generic = process_odds_csv_oddsapi_format(uploaded_file)
        if not df_generic.empty:
            return df_generic
    except Exception:
        pass

    df = read_csv_safe(uploaded_file)
    required = {'player','team','market_key','line','odds','book_name'}
    if not isinstance(df, pd.DataFrame) or df.empty or (required - set(df.columns)):
        st.error("CSV missing required columns or could not be parsed.")
        return pd.DataFrame()

    df = df[df['market_key'].isin(SUPPORTED_MARKETS)].copy()
    df['line_raw'] = pd.to_numeric(df['line'], errors='coerce')
    df.drop(columns=['line'], inplace=True, errors='ignore')
    df['implied_prob'] = df['odds'].apply(odds_to_prob)
    df['player_short'] = df['player'].apply(short_name)
    df['player_merge'] = df['player_short'].apply(merge_key_from_short)
    df['last_name_merge'] = df['player'].apply(last_name_key)
    df['player_fullkey'] = df['player'].apply(full_name_key)

    roster = load_rosters()
    if not roster.empty:
        df = df.merge(roster[['player_fullkey','team_abbr']], on='player_fullkey', how='left')
        df['team'] = df.get('team').astype('string') if 'team' in df.columns else pd.Series([pd.NA]*len(df), dtype='string')
        df['team_abbr'] = df['team_abbr'].astype('string')
        mask = df['team'].isna() | (df['team'] == '')
        df.loc[mask, 'team'] = df.loc[mask, 'team_abbr']
        df = df.drop(columns=['team_abbr'])
    return df

# ------------- Line-aware calibration ------------- #
@st.cache_data(show_spinner=False)
def train_line_aware_calibration(odds_hist: pd.DataFrame, counts_frames: dict, nb_dispersion: dict) -> dict:
    calibrators = {}
    if odds_hist is None or odds_hist.empty:
        return calibrators
    df = odds_hist.dropna(subset=['player_fullkey','market_key','side','line_raw','event_date']).copy()
    df['event_date'] = pd.to_datetime(df['event_date']).dt.normalize()

    for mk in SUPPORTED_MARKETS:
        sub = df[df['market_key'] == mk]
        if sub.empty:
            calibrators[mk] = None
            continue
        preds, labels = [], []
        cdf = counts_frames.get(mk, pd.DataFrame())
        if cdf is None or cdf.empty:
            calibrators[mk] = None
            continue
        k = float(nb_dispersion.get(mk, 100.0))
        dates = sorted(sub['event_date'].unique().tolist())
        for d in dates:
            lam_map = _weighted_lambdas_asof(cdf, d)
            mask_hist = cdf['game_date'] < d
            global_mu = cdf.loc[mask_hist, 'count'].mean() if mask_hist.any() else 0.0
            rows = sub[sub['event_date'] == d]
            for _, r in rows.iterrows():
                lam = lam_map.get(r['player_fullkey'], global_mu)
                if lam <= 0:
                    continue
                line = float(r['line_raw'])
                sim = nb_rvs(lam, k, 20000)
                p_over = float(np.mean(sim > line))
                p_under = float(np.mean(sim < line))
                p_side = p_over if r['side'] == 'Over' else p_under

                actual_row = cdf[(cdf['player_fullkey']==r['player_fullkey']) & (cdf['game_date']==d)]
                actual = actual_row['count'].iloc[0] if len(actual_row) else np.nan
                if pd.isna(actual):
                    continue
                label = int(actual > line) if r['side']=='Over' else int(actual < line)
                preds.append(p_side)
                labels.append(label)
        if len(preds) >= 200:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(np.array(preds), np.array(labels))
            calibrators[mk] = iso
        else:
            calibrators[mk] = None
    st.session_state['calibrators_line'] = calibrators
    return calibrators

# ------------- Historical baseline build ------------- #
@st.cache_data(show_spinner=False)
def compute_hist_baseline() -> pd.DataFrame:
    models, games_counts = st.session_state.get('model_means', {}), st.session_state.get('games_counts', {})
    if not models:
        models, games_counts = train_models()
    roster = load_rosters()
    rows = []
    for market, players in models.items():
        market_mu = np.mean(list(players.values())) if players else 0.0
        for player, lam in players.items():
            n_g = games_counts.get(market, {}).get(player, 0)
            lam_eb = eb_shrink_lambda(lam, market_mu, n_g, tau=6.0)
            player_fk = full_name_key(player)
            team = None
            if not roster.empty:
                r = roster[roster['player_fullkey']==player_fk]
                team = r['team_abbr'].iloc[0] if len(r) else None
            rows.append({
                'player': player,
                'team': team,
                'market_key': market,
                'lambda_base': lam,
                'lambda_eb': lam_eb,
                'player_fullkey': player_fk
            })
    df = pd.DataFrame(rows)
    df['player_short'] = df['player'].apply(short_name)
    df['player_merge'] = df['player_short'].apply(merge_key_from_short)
    df['last_name_merge'] = df['player'].apply(last_name_key)
    return df

# ------------- Line scaling logic ------------- #
def _plausible_range(market: str) -> tuple[float,float]:
    if market == "player_receptions":
        return (0.5, 15.0)
    if market == "player_rush_attempts":
        return (3.0, 35.0)
    if market == "player_pass_attempts":
        return (10.0, 60.0)
    return (0.1, 100.0)

def choose_scale_factor(market: str, line_raw: float, lam_adj: float) -> float:
    if not np.isfinite(line_raw) or not np.isfinite(lam_adj) or lam_adj <= 0:
        return 1.0
    candidates = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    low, high = _plausible_range(market)
    scaled = line_raw * candidates
    z_like = np.abs((scaled - lam_adj) / max(np.sqrt(lam_adj), 1e-6))
    penalty = np.where((scaled < low) | (scaled > high), 5.0, 0.0)
    score = z_like + penalty
    best = candidates[np.argmin(score)]
    return float(best)

# ------------- Edge computation (CSV with odds) ------------- #
def compute_edges_csv(df: pd.DataFrame) -> pd.DataFrame:
    EDGE_THR = float(st.session_state.get('edge_threshold', EDGE_THRESHOLD))
    VOL_THR = float(st.session_state.get('volatility_ratio', VOLATILITY_RATIO_THRESHOLD))
    Z_THR = float(st.session_state.get('z_min', Z_MIN))

    models = st.session_state.get('model_means', {})
    games_counts = st.session_state.get('games_counts', {})
    ks = st.session_state.get('dispersion_k', {})
    team_ctx = st.session_state.get('team_ctx', pd.DataFrame())
    calibrators_line = st.session_state.get('calibrators_line', {})

    results = []
    for _, row in df.iterrows():
        try:
            market = row['market_key']
            player = row['player']
            player_fk = row.get('player_fullkey', full_name_key(player))
            team = row.get('team', None)
            market_mu = np.mean(list(models.get(market, {}).values())) if models.get(market) else 0.0
            lam_base = models.get(market, {}).get(player, market_mu)
            n_g = games_counts.get(market, {}).get(player, 0)
            lam_eb = eb_shrink_lambda(lam_base, market_mu, n_g, tau=6.0)
            lam_ctx = apply_team_context_lambda(lam_eb, market, team, team_ctx) if isinstance(team, str) else lam_eb

            k = float(ks.get(market, 100.0))
            lam_for_scale = max(lam_ctx, 1e-9)

            line_raw = float(row.get('line_raw', np.nan))
            scale_factor = choose_scale_factor(market, line_raw, lam_for_scale)
            line_scaled = line_raw * scale_factor

            lam_adj = 0.85*lam_ctx + 0.15*line_scaled

            sim = nb_rvs(max(lam_adj, 1e-9), k, DEFAULT_N_SIMULATIONS)
            mu, sigma = sim.mean(), sim.std()

            side = row.get('side', 'Unknown')
            if side not in ('Over','Under') or not np.isfinite(line_scaled):
                raise ValueError("Missing side or line")

            p_over = float(np.mean(sim > line_scaled))
            p_under = float(np.mean(sim < line_scaled))
            cal = calibrators_line.get(market)
            if cal is not None:
                p_over = float(cal.predict(np.array([p_over]))[0])
                p_under = float(cal.predict(np.array([p_under]))[0])

            p_side = p_over if side=='Over' else p_under
            implied_prob_used = row.get('implied_prob_fair', row.get('implied_prob', np.nan))
            if pd.isna(implied_prob_used) and pd.notna(row.get('odds', np.nan)):
                implied_prob_used = odds_to_prob(row['odds'])

            boots = 200
            idx = np.random.randint(0, DEFAULT_N_SIMULATIONS, size=(boots, DEFAULT_N_SIMULATIONS))
            side_mask = (sim > line_scaled) if side == 'Over' else (sim < line_scaled)
            p_boot = side_mask[idx].mean(axis=1)
            if cal is not None:
                p_boot = cal.predict(p_boot.reshape(-1,1)).ravel()
            edge_boot = p_boot - implied_prob_used if pd.notna(implied_prob_used) else np.full_like(p_boot, np.nan)
            edge_lo, edge_hi = (np.nan, np.nan)
            if np.isfinite(edge_boot).any():
                edge_lo, edge_hi = np.quantile(edge_boot[~np.isnan(edge_boot)], [0.1, 0.9])

            z_vs_line = (mu - line_scaled) / (sigma + 1e-9)
            ok_vol = (sigma / max(mu, 1e-9)) <= VOL_THR
            ok_z = abs(z_vs_line) >= Z_THR
            pass_edge = (pd.notna(implied_prob_used) and (p_side - implied_prob_used) >= EDGE_THR) and (edge_lo >= EDGE_THR if pd.notna(edge_lo) else True)

            rec = side if (pass_edge and ok_vol and ok_z) else 'No Action'

            results.append({
                **row.to_dict(),
                'line': line_scaled,
                'line_raw': line_raw,
                'scale_factor': scale_factor,
                'lambda': lam_adj,
                'model_mean': mu,
                'model_std': sigma,
                'delta_vs_line': (mu - line_scaled),
                'z_vs_line': z_vs_line,
                'p_over': p_over,
                'p_under': p_under,
                'p_side': p_side,
                'edge_prob': (p_side - implied_prob_used) if pd.notna(implied_prob_used) else np.nan,
                'edge_pct': ((p_side - implied_prob_used) * 100.0) if pd.notna(implied_prob_used) else np.nan,
                'edge_lo': edge_lo*100.0 if pd.notna(edge_lo) else np.nan,
                'edge_hi': edge_hi*100.0 if pd.notna(edge_hi) else np.nan,
                'pass_edge': pass_edge,
                'pass_vol': ok_vol,
                'pass_z': ok_z,
                'recommendation': rec
            })
        except Exception:
            results.append({
                **row.to_dict(),
                'line': np.nan,
                'line_raw': row.get('line_raw', np.nan),
                'scale_factor': np.nan,
                'lambda': np.nan,
                'model_mean': np.nan,
                'model_std': np.nan,
                'delta_vs_line': np.nan,
                'z_vs_line': np.nan,
                'p_over': np.nan,
                'p_under': np.nan,
                'p_side': np.nan,
                'edge_prob': np.nan,
                'edge_pct': np.nan,
                'edge_lo': np.nan,
                'edge_hi': np.nan,
                'pass_edge': False,
                'pass_vol': False,
                'pass_z': False,
                'recommendation': 'No Action'
            })
    return pd.DataFrame(results)

# ------------- Backtest (walk-forward vs real lines) ------------- #
def _american_profit_per_risk_unit(odds_val: float) -> float:
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
    if abs(line - round(line)) < 1e-8:
        if actual_count == line:
            return 'Push'
    if side == 'Over':
        return 'Win' if actual_count > line else 'Loss'
    else:
        return 'Win' if actual_count < line else 'Loss'

def run_walkforward_backtest(odds_hist: pd.DataFrame, counts_frames: dict, team_ctx: pd.DataFrame, use_calibration: bool = True) -> tuple[pd.DataFrame, dict]:
    EDGE_THR = float(st.session_state.get('edge_threshold', EDGE_THRESHOLD))
    VOL_THR = float(st.session_state.get('volatility_ratio', VOLATILITY_RATIO_THRESHOLD))
    Z_THR = float(st.session_state.get('z_min', Z_MIN))

    df = odds_hist.copy()
    if 'player_fullkey' not in df.columns and 'player' in df.columns:
        df['player_fullkey'] = df['player'].apply(full_name_key)
    if 'event_date' not in df.columns:
        if 'event_time' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_time'], errors='coerce').dt.normalize()
        else:
            st.warning('Historical odds missing event_date; cannot run backtest.')
            return pd.DataFrame(), {}
    df['event_date'] = pd.to_datetime(df['event_date']).dt.normalize()
    df = df[df['market_key'].isin(SUPPORTED_MARKETS)]
    df = df.dropna(subset=['player_fullkey','market_key','side','line_raw','odds','event_date'])

    if {'side','implied_prob','player_fullkey','market_key','book_name','line_raw','event_date'}.issubset(df.columns):
        grp_keys = ['player_fullkey','market_key','book_name','line_raw','event_date']
        side_count = df.groupby(grp_keys)['side'].transform('nunique')
        sum_imp = df.groupby(grp_keys)['implied_prob'].transform('sum')
        df['implied_prob_fair'] = np.where(side_count >= 2, df['implied_prob'] / sum_imp, df['implied_prob'])

    calibrators = st.session_state.get('calibrators_line', {}) if use_calibration else {}
    ks = st.session_state.get('dispersion_k', {})
    models = st.session_state.get('model_means', {})
    games_counts = st.session_state.get('games_counts', {})

    results = []
    all_dates = sorted(df['event_date'].dropna().unique().tolist())
    for d in all_dates:
        d_ts = pd.to_datetime(d).normalize()
        lambda_maps = {mk: _weighted_lambdas_asof(counts_frames.get(mk, pd.DataFrame()), d_ts)
                       for mk in SUPPORTED_MARKETS}
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
            team = row.get('team', None)
            lam = lambda_maps.get(mk, {}).get(pkey, global_means.get(mk, 0.0))
            cdf = counts_frames.get(mk, pd.DataFrame())
            n_g = 0
            if cdf is not None and not cdf.empty:
                n_g = cdf[(cdf['player_fullkey']==pkey) & (cdf['game_date']<d_ts)]['game_id'].nunique()
            market_mu = global_means.get(mk, 0.0)
            lam_eb = eb_shrink_lambda(lam, market_mu, n_g, tau=6.0)
            lam_ctx = apply_team_context_lambda(lam_eb, mk, team, team_ctx) if isinstance(team, str) else lam_eb

            line_raw = float(row['line_raw'])
            scale_factor = choose_scale_factor(mk, line_raw, max(lam_ctx,1e-9))
            line = line_raw * scale_factor
            lam_adj = 0.85*lam_ctx + 0.15*line

            k = float(ks.get(mk, 100.0))
            sim = nb_rvs(max(lam_adj, 1e-9), k, DEFAULT_N_SIMULATIONS)
            mu, sigma = sim.mean(), sim.std()

            side = row['side']
            p_over = float(np.mean(sim > line))
            p_under = float(np.mean(sim < line))
            if use_calibration and calibrators.get(mk) is not None:
                p_over = float(calibrators[mk].predict(np.array([p_over]))[0])
                p_under = float(calibrators[mk].predict(np.array([p_under]))[0])
            p_side = p_over if side == 'Over' else p_under
            implied_prob_used = row.get('implied_prob_fair', row.get('implied_prob', np.nan))

            z_vs_line = (mu - line) / (sigma + 1e-9)
            ok_vol = (sigma / max(mu, 1e-9)) <= VOL_THR
            ok_z = abs(z_vs_line) >= Z_THR
            edge_prob = p_side - implied_prob_used if pd.notna(implied_prob_used) else np.nan
            take = (pd.notna(edge_prob) and edge_prob >= EDGE_THR and ok_vol and ok_z)

            actual_row = counts_frames.get(mk, pd.DataFrame())
            actual_row = actual_row[(actual_row['player_fullkey'] == pkey) & (actual_row['game_date'] == d_ts)]
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

            results.append({
                'event_date': d_ts,
                'player': row.get('player', ''),
                'player_fullkey': pkey,
                'market_key': mk,
                'team': team,
                'side': side,
                'book_line_raw': line_raw,
                'scale_factor': scale_factor,
                'book_line': line,
                'odds': row['odds'],
                'implied_prob_fair': row.get('implied_prob_fair', np.nan),
                'model_lambda': lam_adj,
                'model_mean': mu,
                'model_std': sigma,
                'p_side': p_side,
                'edge_prob': edge_prob,
                'z_vs_line': z_vs_line,
                'vol_ratio': (sigma / max(mu, 1e-9)),
                'actual_count': actual,
                'bet_taken': bool(take),
                'outcome': outcome,
                'pnl_risk1u': pnl,
                'risked_units': risk
            })

    res_df = pd.DataFrame(results)
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
        }
    return res_df, summary

# ------------- Display / Logging ------------- #
def display_table(df: pd.DataFrame, key_prefix: str = ""):
    if 'edge_pct' not in df.columns and 'edge_prob' in df.columns:
        df = df.assign(edge_pct=df['edge_prob'] * 100.0)
    cols_base = ['player','team','market_key','side','line','line_raw','scale_factor','delta_vs_line','z_vs_line','odds','edge_pct','edge_lo','edge_hi','model_mean','recommendation','book_name']
    cols = [c for c in cols_base if c in df.columns]
    if 'team' not in df.columns:
        df = df.assign(team=np.nan)
    st.dataframe(df[cols], use_container_width=True)
    st.download_button('Download CSV', df.to_csv(index=False), 'props.csv', 'text/csv', key=f"{key_prefix}_dl_csv")
    st.download_button('Download JSON', df.to_json(orient='records', indent=2), 'props.json', 'application/json', key=f"{key_prefix}_dl_json")

def log_picks(df: pd.DataFrame):
    path = os.path.join(LOG_DIR, f'picks_{datetime.date.today().isoformat()}.json')
    try:
        with open(path, 'w') as f:
            json.dump(df.to_dict(orient='records'), f, indent=2)
    except Exception:
        pass

# ------------- Main App ------------- #
def main():
    st.set_page_config(page_title="FoxEdge NFL Props — Enhanced v2", layout="wide")
    st.title("FoxEdge NFL Player Prop Recommendations — Enhanced v2 (unit-aligned)")

    dbg_toggle = st.sidebar.checkbox("Verbose debug", value=st.session_state.get('__debug_enabled', DEBUG_DEFAULT))
    set_debug(dbg_toggle)

    # Thresholds
    st.sidebar.markdown("### Thresholds")
    edge_thr = st.sidebar.slider("Min edge (prob)", 0.00, 0.15, float(st.session_state.get('edge_threshold', EDGE_THRESHOLD)), 0.005)
    vol_thr  = st.sidebar.slider("Max volatility ratio (σ/μ)", 0.50, 2.00, float(st.session_state.get('volatility_ratio', VOLATILITY_RATIO_THRESHOLD)), 0.05)
    zmin_thr = st.sidebar.slider("Min |z| vs line", 0.00, 3.00, float(st.session_state.get('z_min', Z_MIN)), 0.05)
    st.session_state['edge_threshold'] = edge_thr
    st.session_state['volatility_ratio'] = vol_thr
    st.session_state['z_min'] = zmin_thr

    # Bootstrap
    if st.sidebar.button("Train / Bootstrap Models"):
        with st.spinner("Loading PBP, training means, estimating dispersion, computing team context..."):
            train_models()
            years = list(range(2020, datetime.date.today().year + 1))
            pbp_all = load_pbp_years(min(years), max(years))
            frames = build_counts_frames(pbp_all)
            estimate_market_dispersion(frames)
            st.session_state['team_ctx'] = compute_team_context(pbp_all)
            st.success("Bootstrap complete.")

    st.markdown('---')
    st.subheader('Walk-Forward Backtest vs Real Lines')
    with st.expander('Run historical walk-forward backtest (line-aware, NB, unit-correct)', expanded=False):
        col_a, col_b, col_c = st.columns([1,1,1])
        with col_a:
            year_start = st.number_input('Start Year', min_value=2015, max_value=datetime.date.today().year, value=2020, step=1)
        with col_b:
            year_end = st.number_input('End Year', min_value=2015, max_value=datetime.date.today().year, value=datetime.date.today().year, step=1)
        with col_c:
            use_cal = st.checkbox('Use line-aware calibration (if trained)', value=True)
        hist_odds_files = st.file_uploader('Upload Historical Odds CSVs (Prop Detail or OddsAPI)', type=['csv'], key='hist_odds_csv', accept_multiple_files=True)
        run_bt = st.button('Run Walk-Forward Backtest', type='primary')
        if run_bt:
            if not hist_odds_files:
                st.error('Please upload at least one historical odds CSV.')
            else:
                with st.spinner('Loading PBP and building counts...'):
                    pbp_bt = load_pbp_years(int(year_start), int(year_end))
                    if pbp_bt.empty:
                        st.error('No PBP data loaded for the selected years.')
                    else:
                        counts_frames = build_counts_frames(pbp_bt)
                        ks = estimate_market_dispersion(counts_frames)
                        frames = []
                        for f in hist_odds_files:
                            df_i = process_odds_csv(f)
                            if not df_i.empty:
                                frames.append(df_i)
                        if not frames:
                            st.error('None of the uploaded CSVs matched a supported odds schema.')
                        else:
                            df_hist_odds = pd.concat(frames, ignore_index=True).drop_duplicates()
                            if df_hist_odds.empty:
                                st.error('Historical odds CSVs could not be parsed.')
                            else:
                                train_line_aware_calibration(df_hist_odds, counts_frames, ks)
                                team_ctx = compute_team_context(pbp_bt)
                                res_df, summary = run_walkforward_backtest(df_hist_odds, counts_frames, team_ctx, use_calibration=use_cal)
                                if res_df.empty:
                                    st.warning('Backtest produced no results.')
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
                                        })
                                    st.dataframe(res_df, use_container_width=True)
                                    st.download_button('Download Backtest Results (CSV)', res_df.to_csv(index=False), 'walkforward_results.csv', 'text/csv', key='dl_bt_csv')

    st.markdown('---')
    st.subheader("Live Picks")
    if 'model_means' not in st.session_state or 'games_counts' not in st.session_state:
        st.info("Click 'Train / Bootstrap Models' in the sidebar first.")
        return

    odds_file = st.sidebar.file_uploader("Upload Odds CSV", type=['csv'])
    if not odds_file:
        st.info("Upload an odds CSV to insert prices/lines and compute market edges (with unit rescaling).")
        return

    df_odds = process_odds_csv(odds_file)
    if df_odds.empty:
        st.warning("Uploaded odds CSV could not be parsed. Check required columns.")
        return

    # De-vig when both sides are present
    if {'side','implied_prob','player_fullkey','market_key','book_name','line_raw'}.issubset(df_odds.columns):
        grp_keys = ['player_fullkey','market_key','book_name','line_raw']
        side_count = df_odds.groupby(grp_keys)['side'].transform('nunique')
        sum_imp = df_odds.groupby(grp_keys)['implied_prob'].transform('sum')
        df_odds['implied_prob_fair'] = np.where(side_count >= 2, df_odds['implied_prob'] / sum_imp, df_odds['implied_prob'])
        st.caption(f"De-vig coverage: fair probabilities computed for {(side_count >= 2).sum()} rows.")

    # Ensure bootstrap resources
    if 'dispersion_k' not in st.session_state or 'team_ctx' not in st.session_state:
        with st.spinner("Estimating dispersion and computing team context..."):
            years = list(range(2020, datetime.date.today().year + 1))
            pbp_all = load_pbp_years(min(years), max(years))
            frames = build_counts_frames(pbp_all)
            estimate_market_dispersion(frames)
            st.session_state['team_ctx'] = compute_team_context(pbp_all)

    # Score candidates
    required_cols = ['odds','side','line_raw']
    for rc in required_cols:
        if rc not in df_odds.columns:
            df_odds[rc] = np.nan
    candidates = df_odds[df_odds['odds'].notna() & df_odds['side'].isin(['Over','Under']) & df_odds['line_raw'].notna()].copy()
    if 'implied_prob' not in candidates.columns and 'odds' in candidates.columns:
        candidates['implied_prob'] = candidates['odds'].apply(odds_to_prob)

    if candidates.empty:
        st.warning('No candidate rows with odds, side, and line after parsing.')
        return

    df_results = compute_edges_csv(candidates)

    total_rows = len(candidates)
    st.caption(f"Candidates scored: {total_rows}")
    taken = df_results[df_results['recommendation'].isin(['Over','Under'])]
    if taken.empty:
        st.info('No bets passed the guardrails. Showing diagnostics for the top candidates by edge.')
    try:
        fail_edge = (~df_results['pass_edge']).sum() if 'pass_edge' in df_results else 0
        fail_vol = (~df_results['pass_vol']).sum() if 'pass_vol' in df_results else 0
        fail_z = (~df_results['pass_z']).sum() if 'pass_z' in df_results else 0
        st.caption(f"Diagnostics — total rows: {total_rows}, pass_edge fails: {fail_edge}, pass_vol fails: {fail_vol}, pass_z fails: {fail_z}")
    except Exception:
        pass

    with st.expander('View top candidates by edge (ignoring guardrails)', expanded=False):
        cols_diag = ['player','team','market_key','side','line','line_raw','scale_factor','edge_pct','edge_lo','edge_hi','p_over','p_under','z_vs_line','model_mean','model_std','odds','book_name']
        cols_diag = [c for c in cols_diag if c in df_results.columns]
        topN = df_results.sort_values('edge_pct', ascending=False, na_position='last').head(25)
        if not topN.empty:
            st.dataframe(topN[cols_diag], use_container_width=True)
        else:
            st.write('No candidates to show.')

    df_recs = df_results[df_results['recommendation'].isin(['Over','Under'])]
    st.subheader("Market-Adjusted Picks (after unit-aligned odds)")
    display_table(df_recs, key_prefix="market_adjusted")
    log_picks(df_recs)

if __name__ == '__main__':
    main()
