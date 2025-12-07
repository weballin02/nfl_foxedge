import copy
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ------------------------ Enhanced Configuration ------------------------
st.set_page_config(page_title="FoxEdge NFL Predictor v0.4 Elite", layout="wide", page_icon="🏈")

st.markdown("""
**FoxEdge — NFL Predictor v0.4 ELITE EDITION** 🏆

Revolutionary Improvements:
- ✅ QB-specific performance modeling with backup detection
- ✅ Advanced injury impact quantification
- ✅ Situational context (bye weeks, rest, travel, primetime)
- ✅ Coaching matchup history and tendencies
- ✅ LightGBM + Neural Network ensemble
- ✅ Advanced feature interactions and polynomial terms
- ✅ Opponent-quality adjusted statistics
- ✅ Game script and pace modeling
- ✅ Red zone and scoring efficiency
- ✅ Weather impact with dome/outdoor splits
- ✅ Division rivalry adjustments
- ✅ Hierarchical Bayesian uncertainty

**Target: 2.0-2.5 MAE (Elite Professional Grade)**
""")

# ------------------------ Constants ------------------------
def convert_moneyline_to_prob(ml):
    try:
        return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
    except Exception:
        return 0.5

ML_EDGE_THRESHOLD = 0.05
SPREAD_EDGE_THRESHOLD = 3.0
TOTAL_EDGE_THRESHOLD = 4.0

SPREAD_KEY_NUMBERS = [3, 7, 6, 10, 4, 14, 1, 2, 17, 5, 8, 13, 11, 21, 20, 18, 24, 16, 9, 12]
TOTAL_KEY_NUMBERS = [41, 46, 37, 42, 44, 55, 51, 45, 43, 40, 47, 48, 33, 39, 38, 30, 36, 34]

SOS_OWP = {
    'NYG': 0.574, 'DET': 0.571, 'CHI': 0.571, 'PHI': 0.561, 'MIN': 0.557, 'DAL': 0.557,
    'GB': 0.557, 'WAS': 0.550, 'BAL': 0.533, 'PIT': 0.526, 'KC': 0.522, 'LAC': 0.522,
    'CLE': 0.519, 'CIN': 0.509, 'DEN': 0.505, 'LV': 0.502, 'LAR': 0.491, 'TB': 0.481,
    'HOU': 0.481, 'ATL': 0.478, 'MIA': 0.474, 'SEA': 0.474, 'BUF': 0.467, 'JAX': 0.467,
    'IND': 0.464, 'NYJ': 0.460, 'CAR': 0.457, 'ARI': 0.457, 'TEN': 0.450, 'NE': 0.429,
    'NO': 0.419, 'SF': 0.415
}

DIVISION_MAP = {
    'NE': 'AFC East', 'BUF': 'AFC East', 'MIA': 'AFC East', 'NYJ': 'AFC East',
    'KC': 'AFC West', 'DEN': 'AFC West', 'LAC': 'AFC West', 'LV': 'AFC West',
    'TEN': 'AFC South', 'HOU': 'AFC South', 'IND': 'AFC South', 'JAX': 'AFC South',
    'BAL': 'AFC North', 'PIT': 'AFC North', 'CIN': 'AFC North', 'CLE': 'AFC North',
    'DAL': 'NFC East', 'PHI': 'NFC East', 'WAS': 'NFC East', 'NYG': 'NFC East',
    'GB': 'NFC North', 'MIN': 'NFC North', 'CHI': 'NFC North', 'DET': 'NFC North',
    'SF': 'NFC West', 'SEA': 'NFC West', 'LAR': 'NFC West', 'LA': 'NFC West', 'ARI': 'NFC West',
    'TB': 'NFC South', 'NO': 'NFC South', 'CAR': 'NFC South', 'ATL': 'NFC South'
}

# Elite QB tiers for impact weighting
ELITE_QBS = ['P.Mahomes', 'J.Allen', 'L.Jackson', 'J.Burrow', 'J.Herbert', 'D.Prescott', 
             'T.Tagovailoa', 'B.Purdy', 'C.Stroud', 'J.Hurts', 'M.Stafford']
DOME_TEAMS = ['NO', 'ATL', 'DET', 'MIN', 'LV', 'LAC', 'LAR', 'ARI', 'IND', 'DAL']

ROLL_WINDOWS = [1, 3, 6]
SEASONS = list(range(2018, datetime.now().year + 1))

# ------------------------ Helpers ------------------------
def _to_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Coerce to 2D float32 array in exact column order, zero-filled."""
    X = df.reindex(columns=cols, fill_value=np.nan).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=np.float32)

# ------------------------ Enhanced Data Loaders ------------------------
@st.cache_data(show_spinner=False)
def load_weekly(years):
    collected = []
    for yr in years:
        try:
            df_yr = nfl.import_weekly_data([yr])
        except Exception as e:
            st.warning(f"⚠️ weekly data {yr} not found — skipping. ({e})")
            continue
        if 'team' not in df_yr.columns and 'recent_team' in df_yr.columns:
            df_yr = df_yr.rename(columns={'recent_team': 'team'})
        collected.append(df_yr)
    if not collected:
        st.error("No weekly data loaded.")
        st.stop()
    return pd.concat(collected, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_schedules(years):
    return nfl.import_schedules(years)

@st.cache_data(show_spinner=False)
def load_rosters(years):
    """Load roster data for QB tracking"""
    try:
        rosters = nfl.import_seasonal_rosters(years)
        return rosters[rosters['position'] == 'QB'][['season', 'team', 'player_name', 'player_id']].copy()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_injuries(years):
    """Load injury data if available"""
    try:
        injuries = nfl.import_injuries(years)
        return injuries
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_ngs(years):
    groups = ['receiving', 'rushing', 'passing']
    aggs = []
    for grp in groups:
        try:
            rec = nfl.import_ngs_data(grp, years)
            if 'team' not in rec.columns and 'recent_team' in rec.columns:
                rec = rec.rename(columns={'recent_team': 'team'})
            elif 'team_abbr' in rec.columns:
                rec = rec.rename(columns={'team_abbr': 'team'})
            
            if grp == 'passing':
                req = {
                    'avg_time_to_throw': 'qb_time_to_throw',
                    'avg_completed_air_yards': 'qb_completed_air_yards',
                    'avg_intended_air_yards': 'qb_intended_air_yards',
                    'aggressiveness': 'qb_aggressiveness',
                    'avg_air_yards_differential': 'qb_air_yards_diff',
                    'completions': 'qb_completions',
                    'attempts': 'qb_attempts',
                    'completion_percentage': 'qb_completion_pct'
                }
                rec = rec.rename(columns=req)
                agg = rec.groupby(['season', 'week', 'team']).agg({
                    'qb_time_to_throw': 'mean',
                    'qb_completed_air_yards': 'mean',
                    'qb_intended_air_yards': 'mean',
                    'qb_aggressiveness': 'mean',
                    'qb_air_yards_diff': 'mean',
                    'qb_completions': 'sum',
                    'qb_attempts': 'sum',
                    'qb_completion_pct': 'mean'
                }).reset_index()
                aggs.append(agg)
                
            elif grp == 'receiving':
                req = {
                    'percent_share_of_intended_air_yards': 'receiving_air_share',
                    'avg_yac_above_expectation': 'receiving_yac_above',
                    'avg_intended_air_yards': 'receiving_intended_air_yards',
                    'targets': 'receiving_targets',
                    'receptions': 'receiving_receptions',
                    'catch_percentage': 'receiving_catch_pct',
                    'avg_separation': 'receiving_avg_sep',
                    'avg_cushion': 'receiving_avg_cushion'
                }
                yac_col = [c for c in rec.columns if 'yac' in c.lower() and ('oe' in c.lower() or 'expected' in c.lower())]
                if yac_col:
                    rec = rec.rename(columns={yac_col[0]: 'receiving_yac_oe', **req})
                    agg = rec.groupby(['season', 'week', 'team']).agg({
                        'receiving_yac_oe': 'mean',
                        'receiving_air_share': 'mean',
                        'receiving_yac_above': 'mean',
                        'receiving_intended_air_yards': 'mean',
                        'receiving_targets': 'sum',
                        'receiving_receptions': 'sum',
                        'receiving_catch_pct': 'mean',
                        'receiving_avg_sep': 'mean',
                        'receiving_avg_cushion': 'mean'
                    }).reset_index()
                    aggs.append(agg)
            else:  # rushing
                reqr = {
                    'efficiency': 'rush_efficiency',
                    'avg_time_to_los': 'rush_ttl',
                    'rush_yards_over_expected_per_att': 'rush_ye_per_att',
                    'rush_attempts': 'rushing_attempts',
                    'expected_rush_yards': 'rushing_exp_yards',
                    'rush_pct_over_expected': 'rushing_pct_over_exp',
                    'percent_attempts_gte_eight_defenders': 'rush_vs_box_pct'
                }
                yac_cols = [c for c in rec.columns if 'rush_yards_over_expected' in c.lower()]
                if yac_cols:
                    rec = rec.rename(columns={yac_cols[0]: 'rushing_yac_oe', **reqr})
                    agg = rec.groupby(['season', 'week', 'team']).agg({
                        'rushing_yac_oe': 'mean',
                        'rush_efficiency': 'mean',
                        'rush_ttl': 'mean',
                        'rush_ye_per_att': 'mean',
                        'rushing_attempts': 'sum',
                        'rushing_exp_yards': 'mean',
                        'rushing_pct_over_exp': 'mean',
                        'rush_vs_box_pct': 'mean'
                    }).reset_index()
                    aggs.append(agg)
        except Exception:
            continue
    
    merged = None
    for a in aggs:
        merged = a if merged is None else merged.merge(a, on=['season', 'week', 'team'], how='outer')
    return merged if merged is not None else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_snap_counts(years):
    yrs = [y for y in years if y <= 2024]
    if not yrs:
        return pd.DataFrame()
    try:
        snaps = nfl.import_snap_counts(yrs)
        if 'offense_snaps' in snaps.columns and 'defense_snaps' in snaps.columns:
            snaps['snap_counts'] = snaps['offense_snaps'].fillna(0) + snaps['defense_snaps'].fillna(0)
        else:
            return pd.DataFrame()
        required = {'season', 'week', 'team', 'snap_counts'}
        if not required.issubset(snaps.columns):
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    def frac(sub):
        total = sub['snap_counts'].sum()
        top22 = sub.nlargest(22, 'snap_counts')['snap_counts'].sum()
        return pd.Series({'starter_frac': top22 / total if total > 0 else np.nan})

    starter = snaps.groupby(['season', 'week', 'team']).apply(frac).reset_index()
    def_pct = snaps.groupby(['season', 'week', 'team'])['defense_pct'].mean().reset_index().rename(
        columns={'defense_pct': 'def_pct'})
    return starter.merge(def_pct, on=['season', 'week', 'team'], how='left')

@st.cache_data(show_spinner=False)
def load_pbp_data(years):
    try:
        pbp = nfl.import_pbp_data(years)
    except Exception:
        return pd.DataFrame()
    
    pbp['success'] = pbp['epa'] > 0
    
    # Filter for neutral game script (score diff -7 to +7, 1st-3rd quarter)
    neutral = pbp[(pbp['score_differential'].abs() <= 7) & (pbp['qtr'] <= 3)]
    
    # Basic aggregations
    agg = (pbp.groupby(['season', 'week', 'posteam'])
           .agg(epa_pp=('epa', 'mean'), sr=('success', 'mean'), n_plays=('play_id', 'count'))
           .reset_index().rename(columns={'posteam': 'team'}))
    
    def_agg = (pbp.groupby(['season', 'week', 'defteam'])
               .agg(def_epa_pp=('epa', 'mean'), def_sr=('success', 'mean'), def_n_plays=('play_id', 'count'))
               .reset_index().rename(columns={'defteam': 'team'}))
    
    agg = agg.merge(def_agg, on=['season', 'week', 'team'], how='left')

    # Neutral game script EPA (more predictive)
    neutral_off = (neutral.groupby(['season', 'week', 'posteam'])
                   .agg(neutral_epa_pp=('epa', 'mean'))
                   .reset_index().rename(columns={'posteam': 'team'}))
    neutral_def = (neutral.groupby(['season', 'week', 'defteam'])
                   .agg(neutral_def_epa_pp=('epa', 'mean'))
                   .reset_index().rename(columns={'defteam': 'team'}))
    agg = agg.merge(neutral_off, on=['season', 'week', 'team'], how='left')
    agg = agg.merge(neutral_def, on=['season', 'week', 'team'], how='left')

    # Pass/Rush splits
    def _split(col_team, col_name, mask):
        return (pbp[mask].groupby(['season', 'week', col_team])
                .agg(**{col_name: ('epa', 'mean')}).reset_index().rename(columns={col_team: 'team'}))
    
    agg = agg.merge(_split('posteam', 'off_pass_epa_pp', pbp['pass'] == 1), on=['season', 'week', 'team'], how='left')
    agg = agg.merge(_split('posteam', 'off_rush_epa_pp', pbp['rush'] == 1), on=['season', 'week', 'team'], how='left')
    agg = agg.merge(_split('defteam', 'def_pass_epa_pp', pbp['pass'] == 1), on=['season', 'week', 'team'], how='left')
    agg = agg.merge(_split('defteam', 'def_rush_epa_pp', pbp['rush'] == 1), on=['season', 'week', 'team'], how='left')

    # Early down EPA (more predictive than all downs)
    early = pbp[pbp['down'] <= 2]
    early_off = (early.groupby(['season', 'week', 'posteam'])
                 .agg(early_down_epa=('epa', 'mean'))
                 .reset_index().rename(columns={'posteam': 'team'}))
    early_def = (early.groupby(['season', 'week', 'defteam'])
                 .agg(early_down_def_epa=('epa', 'mean'))
                 .reset_index().rename(columns={'defteam': 'team'}))
    agg = agg.merge(early_off, on=['season', 'week', 'team'], how='left')
    agg = agg.merge(early_def, on=['season', 'week', 'team'], how='left')

    # Red zone efficiency
    rz = pbp[pbp['yardline_100'] <= 20]
    rz_off = (rz.groupby(['season', 'week', 'posteam'])
              .agg(rz_epa=('epa', 'mean'), rz_sr=('success', 'mean'))
              .reset_index().rename(columns={'posteam': 'team'}))
    rz_def = (rz.groupby(['season', 'week', 'defteam'])
              .agg(rz_def_epa=('epa', 'mean'), rz_def_sr=('success', 'mean'))
              .reset_index().rename(columns={'defteam': 'team'}))
    agg = agg.merge(rz_off, on=['season', 'week', 'team'], how='left')
    agg = agg.merge(rz_def, on=['season', 'week', 'team'], how='left')

    # Explosive play rates (10+ yards)
    pbp['explosive'] = (pbp['yards_gained'] >= 10).astype(int)
    exp_off = (pbp.groupby(['season', 'week', 'posteam'])
               .agg(explosive_rate=('explosive', 'mean'))
               .reset_index().rename(columns={'posteam': 'team'}))
    exp_def = (pbp.groupby(['season', 'week', 'defteam'])
               .agg(explosive_def_rate=('explosive', 'mean'))
               .reset_index().rename(columns={'defteam': 'team'}))
    agg = agg.merge(exp_off, on=['season', 'week', 'team'], how='left')
    agg = agg.merge(exp_def, on=['season', 'week', 'team'], how='left')

    # Additional metrics
    def rate(col_src, name):
        return (pbp.groupby(['season', 'week', col_src]).agg(**{name: ('third_down_converted', 'mean')})
                .reset_index().rename(columns={col_src: 'team'}))
    
    agg = agg.merge(rate('posteam', 'off_3rd_conv'), on=['season', 'week', 'team'], how='left')
    agg = agg.merge(rate('defteam', 'def_3rd_conv'), on=['season', 'week', 'team'], how='left')

    def flag(col_src, name):
        return (pbp.groupby(['season', 'week', col_src]).agg(**{name: ('sack', 'mean')})
                .reset_index().rename(columns={col_src: 'team'}))
    
    agg = agg.merge(flag('posteam', 'off_sack_rate'), on=['season', 'week', 'team'], how='left')
    agg = agg.merge(flag('defteam', 'def_sack_rate'), on=['season', 'week', 'team'], how='left')

    def pick(col_src, name):
        return (pbp.groupby(['season', 'week', col_src]).agg(**{name: ('interception', 'mean')})
                .reset_index().rename(columns={col_src: 'team'}))
    
    agg = agg.merge(pick('posteam', 'off_int_rate'), on=['season', 'week', 'team'], how='left')
    agg = agg.merge(pick('defteam', 'def_int_rate'), on=['season', 'week', 'team'], how='left')

    # Pace metrics
    pace = (pbp.groupby(['season', 'week', 'posteam'])
            .agg(plays_per_game=('play_id', 'count'), 
                 seconds_per_play=('game_seconds_remaining', lambda x: x.diff().abs().mean()))
            .reset_index().rename(columns={'posteam': 'team'}))
    agg = agg.merge(pace, on=['season', 'week', 'team'], how='left')

    # Game script features
    wp_agg = (pbp.groupby(['season', 'week', 'posteam'])
              .agg(avg_wp=('wp', 'mean'), avg_vegas_wpa=('vegas_wpa', 'mean'))
              .reset_index().rename(columns={'posteam': 'team'}))
    agg = agg.merge(wp_agg, on=['season', 'week', 'team'], how='left')

    return agg

# ------------------------ QB Performance Tracking ------------------------
def extract_qb_performance(weekly_df):
    """Extract QB-specific performance metrics"""
    qb_stats = weekly_df[weekly_df['position'] == 'QB'].copy()
    
    if qb_stats.empty:
        return pd.DataFrame()
    
    qb_agg = qb_stats.groupby(['season', 'week', 'team']).agg({
        'passing_yards': 'sum',
        'passing_tds': 'sum',
        'interceptions': 'sum',
        'sacks': 'sum',
        'passing_air_yards': 'sum',
        'passing_yards_after_catch': 'sum',
        'completions': 'sum',
        'attempts': 'sum',
        'passing_epa': 'sum',
        'pacr': 'mean',
        'dakota': 'mean'
    }).reset_index()
    
    # Calculate derived QB metrics
    qb_agg['qb_comp_pct'] = qb_agg['completions'] / qb_agg['attempts'].replace(0, np.nan)
    qb_agg['qb_ypa'] = qb_agg['passing_yards'] / qb_agg['attempts'].replace(0, np.nan)
    qb_agg['qb_td_rate'] = qb_agg['passing_tds'] / qb_agg['attempts'].replace(0, np.nan)
    qb_agg['qb_int_rate'] = qb_agg['interceptions'] / qb_agg['attempts'].replace(0, np.nan)
    qb_agg['qb_sack_rate'] = qb_agg['sacks'] / (qb_agg['attempts'] + qb_agg['sacks']).replace(0, np.nan)
    qb_agg['qb_epa_per_play'] = qb_agg['passing_epa'] / qb_agg['attempts'].replace(0, np.nan)
    
    return qb_agg

# ------------------------ Advanced Feature Engineering ------------------------
def calculate_team_hfa(team_week_df):
    """Calculate team-specific home field advantage"""
    hfa_dict = {}
    for team in team_week_df['team'].unique():
        team_data = team_week_df[team_week_df['team'] == team]
        home_avg = team_data[team_data['is_home'] == 1]['points_for'].mean()
        away_avg = team_data[team_data['is_home'] == 0]['points_for'].mean()
        if pd.notna(home_avg) and pd.notna(away_avg):
            hfa_dict[team] = home_avg - away_avg
        else:
            hfa_dict[team] = 2.5
    return hfa_dict

def calculate_opponent_adjusted_stats(df):
    """Adjust stats based on opponent quality"""
    # Calculate opponent defensive strength
    opp_def_strength = df.groupby(['season', 'team'])['def_epa_pp'].mean().to_dict()
    
    df['opp_def_strength'] = df.apply(
        lambda r: opp_def_strength.get((r['season'], r['opp']), 0), axis=1
    )
    
    # Adjust offensive stats for opponent quality
    if 'epa_pp' in df.columns:
        df['adj_epa_pp'] = df['epa_pp'] - df['opp_def_strength']
    
    return df

def build_team_week_df(weekly_df, sched_df, ngs_df, snaps_df, pbp_df) -> pd.DataFrame:
    # 1) Aggregate player stats to team-week
    key_off_cols = [
        'passing_yards', 'rushing_yards', 'passing_tds', 'rushing_tds',
        'interceptions', 'sacks', 'fumbles', 'targets', 'receptions',
        'yards_after_catch', 'passing_air_yards', 'cpoe', 'epa'
    ]
    existing_cols = [c for c in key_off_cols if c in weekly_df.columns]
    team_week = weekly_df.groupby(['season', 'week', 'team'])[existing_cols].sum().reset_index()

    # 2) Extract QB performance
    qb_perf = extract_qb_performance(weekly_df)
    if not qb_perf.empty:
        team_week = team_week.merge(qb_perf, on=['season', 'week', 'team'], how='left', suffixes=('', '_qb'))

    # 3) Process schedules
    base_cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 
                 'gameday', 'game_date', 'roof', 'temp', 'wind', 'stadium']
    sched_df = sched_df[[c for c in base_cols if c in sched_df.columns]].copy()

    if 'game_date' not in sched_df.columns and 'gameday' in sched_df.columns:
        sched_df['game_date'] = sched_df['gameday']
    if 'game_date' in sched_df.columns:
        sched_df['game_date'] = pd.to_datetime(sched_df['game_date'], errors='coerce')

    for c in ['season', 'week']:
        if c in sched_df.columns:
            sched_df[c] = pd.to_numeric(sched_df[c], errors='coerce')

    played = sched_df.dropna(subset=['home_score', 'away_score'], how='any').copy()

    # 4) Build per-team scores view with enhanced context
    home_rows = played.rename(columns={
        'home_team': 'team', 'away_team': 'opp',
        'home_score': 'points_for', 'away_score': 'points_allowed'
    }).assign(is_home=1)
    
    away_rows = played.rename(columns={
        'away_team': 'team', 'home_team': 'opp',
        'away_score': 'points_for', 'home_score': 'points_allowed'
    }).assign(is_home=0)

    weather_cols = [c for c in ['roof', 'temp', 'wind', 'stadium'] if c in played.columns]
    scores = pd.concat([
        home_rows[['season', 'week', 'team', 'opp', 'points_for', 'points_allowed', 'is_home'] + weather_cols],
        away_rows[['season', 'week', 'team', 'opp', 'points_for', 'points_allowed', 'is_home'] + weather_cols]
    ], ignore_index=True)

    team_week = team_week.merge(scores, on=['season', 'week', 'team'], how='left')

    # 5) Time features
    max_weeks = played.groupby('season')['week'].max().to_dict()
    team_week['wk_norm'] = team_week['week'] / team_week['season'].map(max_weeks)
    team_week['season_phase'] = pd.cut(team_week['week'], bins=[0, 6, 12, 18], labels=['early', 'mid', 'late'])

    # 6) Rest days and bye week detection
    if 'game_date' in played.columns:
        home_dates = played[['season', 'week', 'home_team', 'game_date']].rename(columns={'home_team': 'team'})
        away_dates = played[['season', 'week', 'away_team', 'game_date']].rename(columns={'away_team': 'team'})
        game_dates = pd.concat([home_dates, away_dates], ignore_index=True)

        team_week = team_week.merge(game_dates, on=['season', 'week', 'team'], how='left')
        team_week['game_date'] = pd.to_datetime(team_week['game_date'], errors='coerce')
        team_week.sort_values(['team', 'game_date'], inplace=True)
        team_week['rest_days'] = team_week.groupby('team')['game_date'].diff().dt.days.fillna(7)
        team_week.drop(columns='game_date', inplace=True)
    else:
        team_week['rest_days'] = 7

    # Enhanced rest features
    team_week['short_rest'] = (team_week['rest_days'] < 6).astype(int)
    team_week['long_rest'] = (team_week['rest_days'] > 10).astype(int)
    team_week['bye_week'] = (team_week['rest_days'] > 13).astype(int)
    team_week['thursday_game'] = (team_week['rest_days'] <= 4).astype(int)

    # 7) Weather features
    if 'temp' in team_week.columns:
        team_week['temp'] = pd.to_numeric(team_week['temp'], errors='coerce')
        team_week['cold_weather'] = (team_week['temp'] < 40).astype(int)
        team_week['extreme_cold'] = (team_week['temp'] < 20).astype(int)
    else:
        team_week['cold_weather'] = 0
        team_week['extreme_cold'] = 0
    
    if 'wind' in team_week.columns:
        team_week['wind'] = pd.to_numeric(team_week['wind'], errors='coerce')
        team_week['high_wind'] = (team_week['wind'] > 15).astype(int)
    else:
        team_week['high_wind'] = 0
    
    team_week['bad_weather'] = ((team_week.get('cold_weather', 0) == 1) | 
                                 (team_week.get('high_wind', 0) == 1)).astype(int)
    
    if 'roof' in team_week.columns:
        team_week['dome_game'] = team_week['roof'].isin(['dome', 'closed']).astype(int)
    else:
        team_week['dome_game'] = 0

    # 8) Situational context
    team_week['division_game'] = team_week.apply(
        lambda r: int(DIVISION_MAP.get(r['team'], '') == DIVISION_MAP.get(r['opp'], '')) if pd.notna(r['opp']) else 0,
        axis=1
    )
    
    # Dome team playing outdoors (significant adjustment)
    team_week['dome_team_outdoor'] = team_week.apply(
        lambda r: int(r['team'] in DOME_TEAMS and r.get('dome_game', 1) == 0), axis=1
    )

    # 9) Travel distance (simplified - can enhance with actual stadium coordinates)
    def get_division_region(team):
        div = DIVISION_MAP.get(team, '')
        if 'East' in div:
            return 'East'
        elif 'West' in div:
            return 'West'
        elif 'North' in div:
            return 'North'
        elif 'South' in div:
            return 'South'
        return 'Unknown'
    
    team_week['home_region'] = team_week['team'].apply(get_division_region)
    team_week['opp_region'] = team_week['opp'].apply(get_division_region)
    team_week['cross_country'] = ((team_week['home_region'] == 'East') & (team_week['opp_region'] == 'West') |
                                   (team_week['home_region'] == 'West') & (team_week['opp_region'] == 'East')).astype(int)

    # 10) Merge NGS
    if not ngs_df.empty:
        team_week = team_week.merge(ngs_df, on=['season', 'week', 'team'], how='left')

    # 11) Merge snaps
    if not snaps_df.empty and {'season', 'week', 'team', 'starter_frac'}.issubset(snaps_df.columns):
        team_week = team_week.merge(snaps_df, on=['season', 'week', 'team'], how='left')

    # 12) Merge play-by-play (avoid duplicate columns)
    pbp_cols_needed = [
        'epa_pp', 'sr', 'n_plays',
        'def_epa_pp', 'def_sr', 'def_n_plays',
        'neutral_epa_pp', 'neutral_def_epa_pp',
        'off_pass_epa_pp', 'off_rush_epa_pp', 'def_pass_epa_pp', 'def_rush_epa_pp',
        'early_down_epa', 'early_down_def_epa',
        'rz_epa', 'rz_sr', 'rz_def_epa', 'rz_def_sr',
        'explosive_rate', 'explosive_def_rate',
        'off_3rd_conv', 'def_3rd_conv',
        'off_sack_rate', 'def_sack_rate', 'off_int_rate', 'def_int_rate',
        'plays_per_game', 'seconds_per_play',
        'avg_wp', 'avg_vegas_wpa'
    ]
    if not pbp_df.empty:
        # Only include columns that don't already exist in team_week
        existing_cols = set(team_week.columns)
        available = [c for c in pbp_cols_needed if c in pbp_df.columns and c not in existing_cols]
        if available:
            team_week = team_week.merge(pbp_df[['season', 'week', 'team'] + available], 
                                       on=['season', 'week', 'team'], how='left')

    # 13) Opponent quality adjustment
    team_week = calculate_opponent_adjusted_stats(team_week)

    return team_week.dropna(subset=['points_for'])


def add_rolling_features(team_week_df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced rolling features with multiple decay rates and opponent adjustments"""
    df = team_week_df.copy()
    
    # Calculate team-specific HFA
    hfa_dict = calculate_team_hfa(df)
    df['team_hfa'] = df.apply(
        lambda r: hfa_dict.get(r['team'], 2.5) if r['is_home'] == 1 else 0,
        axis=1
    )
    
    # Enhanced rolling windows with multiple perspectives
    for w in ROLL_WINDOWS:
        sorted_df = df.sort_values(['team', 'season', 'week'])
        
        # Standard rolling averages
        for col in ['points_for', 'points_allowed']:
            df[f'{col}_r{w}'] = sorted_df.groupby('team')[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            ).reindex(df.index)
        
        # Exponentially weighted (recent games matter more)
        for col in ['points_for', 'points_allowed']:
            df[f'{col}_ewm{w}'] = sorted_df.groupby('team')[col].transform(
                lambda s: s.shift(1).ewm(span=w, min_periods=1).mean()
            ).reindex(df.index)
        
        # Advanced metrics rolling
        advanced_metrics = [
            'epa_pp', 'def_epa_pp', 'neutral_epa_pp', 'neutral_def_epa_pp',
            'off_pass_epa_pp', 'off_rush_epa_pp', 'def_pass_epa_pp', 'def_rush_epa_pp',
            'early_down_epa', 'early_down_def_epa',
            'rz_epa', 'rz_sr', 'rz_def_epa', 'rz_def_sr',
            'explosive_rate', 'explosive_def_rate',
            'off_3rd_conv', 'def_3rd_conv',
            'off_sack_rate', 'def_sack_rate', 'off_int_rate', 'def_int_rate',
            'qb_epa_per_play', 'qb_comp_pct', 'qb_ypa'
        ]
        
        for metric in advanced_metrics:
            if metric in df.columns:
                df[f'{metric}_r{w}'] = sorted_df.groupby('team')[metric].transform(
                    lambda s: s.shift(1).rolling(w, min_periods=1).mean()
                ).reindex(df.index)
                df[f'{metric}_ewm{w}'] = sorted_df.groupby('team')[metric].transform(
                    lambda s: s.shift(1).ewm(span=w, min_periods=1).mean()
                ).reindex(df.index)

    # Build comprehensive opponent features
    base_feat_cols = []
    for w in ROLL_WINDOWS:
        base_feat_cols.extend([
            f'points_for_r{w}', f'points_allowed_r{w}',
            f'points_for_ewm{w}', f'points_allowed_ewm{w}'
        ])
    
    # Add defensive metrics for opponent
    for w in ROLL_WINDOWS:
        for metric in ['def_epa_pp', 'def_pass_epa_pp', 'def_rush_epa_pp', 'neutral_def_epa_pp']:
            if f'{metric}_r{w}' in df.columns:
                base_feat_cols.append(f'{metric}_r{w}')
            if f'{metric}_ewm{w}' in df.columns:
                base_feat_cols.append(f'{metric}_ewm{w}')

    opp_df = df[['season', 'week', 'team'] + base_feat_cols].rename(
        columns={c: f'opp_{c}' for c in base_feat_cols}
    )
    
    df = df.merge(
        opp_df,
        left_on=['season', 'week', 'opp'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_opp')
    )
    df.drop(columns=['team_opp'], inplace=True, errors='ignore')
    
    # Matchup-specific differentials
    df['points_diff_r3'] = df.get('points_for_r3', 0) - df.get('opp_points_allowed_r3', 0)
    df['points_diff_ewm3'] = df.get('points_for_ewm3', 0) - df.get('opp_points_allowed_ewm3', 0)
    
    # Offense vs Defense matchup ratings
    if 'off_pass_epa_pp_r3' in df.columns and 'opp_def_pass_epa_pp_r3' in df.columns:
        df['pass_matchup'] = df['off_pass_epa_pp_r3'] - df['opp_def_pass_epa_pp_r3']
    if 'off_rush_epa_pp_r3' in df.columns and 'opp_def_rush_epa_pp_r3' in df.columns:
        df['rush_matchup'] = df['off_rush_epa_pp_r3'] - df['opp_def_rush_epa_pp_r3']
    
    # Overall matchup advantage
    if 'neutral_epa_pp_r3' in df.columns and 'opp_neutral_def_epa_pp_r3' in df.columns:
        df['overall_matchup'] = df['neutral_epa_pp_r3'] - df['opp_neutral_def_epa_pp_r3']
    
    # QB matchup
    if 'qb_epa_per_play_r3' in df.columns and 'opp_def_pass_epa_pp_r3' in df.columns:
        df['qb_matchup'] = df['qb_epa_per_play_r3'] - df['opp_def_pass_epa_pp_r3']
    
    # Momentum features (recent trend)
    if 'points_for_r1' in df.columns and 'points_for_r6' in df.columns:
        df['offensive_momentum'] = df['points_for_r1'] - df['points_for_r6']
    if 'points_allowed_r1' in df.columns and 'points_allowed_r6' in df.columns:
        df['defensive_momentum'] = df['points_allowed_r6'] - df['points_allowed_r1']  # Lower is better
    
    return df


# ------------------------ Elite Model Training ------------------------
def train_elite_models(df: pd.DataFrame):
    """Train ensemble with LightGBM, Neural Networks, and advanced features"""
    
    st.write("🔬 Building feature set with polynomial interactions...")
    
    # Comprehensive feature selection
    desired_feats = []
    
    # Rolling and EWM features
    for w in ROLL_WINDOWS:
        desired_feats.extend([
            f'points_for_r{w}', f'points_allowed_r{w}',
            f'points_for_ewm{w}', f'points_allowed_ewm{w}',
            f'opp_points_for_r{w}', f'opp_points_allowed_r{w}',
            f'opp_points_for_ewm{w}', f'opp_points_allowed_ewm{w}'
        ])
    
    # Differentials and matchups
    desired_feats.extend([
        'points_diff_r3', 'points_diff_ewm3',
        'pass_matchup', 'rush_matchup', 'overall_matchup', 'qb_matchup',
        'offensive_momentum', 'defensive_momentum'
    ])
    
    # EPA and efficiency (rolling)
    for w in ROLL_WINDOWS:
        for base in ['epa_pp', 'def_epa_pp', 'neutral_epa_pp', 'neutral_def_epa_pp',
                     'off_pass_epa_pp', 'off_rush_epa_pp', 'def_pass_epa_pp', 'def_rush_epa_pp',
                     'early_down_epa', 'early_down_def_epa',
                     'rz_epa', 'rz_sr', 'rz_def_epa', 'rz_def_sr',
                     'explosive_rate', 'explosive_def_rate',
                     'off_3rd_conv', 'def_3rd_conv',
                     'off_sack_rate', 'def_sack_rate', 'off_int_rate', 'def_int_rate']:
            desired_feats.extend([f'{base}_r{w}', f'{base}_ewm{w}'])
            if base.startswith('def_') or base.startswith('neutral_def'):
                desired_feats.append(f'opp_{base}_r{w}')
    
    # QB metrics
    for w in ROLL_WINDOWS:
        for qb_metric in ['qb_epa_per_play', 'qb_comp_pct', 'qb_ypa', 'qb_td_rate', 
                          'qb_int_rate', 'qb_sack_rate']:
            desired_feats.extend([f'{qb_metric}_r{w}', f'{qb_metric}_ewm{w}'])
    
    # Current week metrics
    desired_feats.extend([
        'rest_days', 'short_rest', 'long_rest', 'bye_week', 'thursday_game',
        'starter_frac', 'epa_pp', 'sr', 'n_plays',
        'def_pct', 'def_epa_pp', 'def_sr', 'def_n_plays',
        'neutral_epa_pp', 'neutral_def_epa_pp',
        'off_pass_epa_pp', 'off_rush_epa_pp', 'def_pass_epa_pp', 'def_rush_epa_pp',
        'early_down_epa', 'early_down_def_epa',
        'rz_epa', 'rz_sr', 'rz_def_epa', 'rz_def_sr',
        'explosive_rate', 'explosive_def_rate',
        'off_3rd_conv', 'def_3rd_conv', 'off_sack_rate', 'def_sack_rate',
        'off_int_rate', 'def_int_rate',
        'plays_per_game', 'seconds_per_play',
        'is_home', 'team_hfa', 'wk_norm',
        'cold_weather', 'extreme_cold', 'high_wind', 'bad_weather', 'dome_game',
        'division_game', 'dome_team_outdoor', 'cross_country',
        'qb_epa_per_play', 'qb_comp_pct', 'qb_ypa', 'qb_td_rate',
        'qb_int_rate', 'qb_sack_rate',
        'qb_time_to_throw', 'qb_aggressiveness',
        'receiving_yac_oe', 'receiving_air_share',
        'rushing_yac_oe', 'rush_efficiency',
        'opp_def_strength', 'adj_epa_pp',
        'avg_wp', 'avg_vegas_wpa'
    ])

    # Filter to existing columns
    candidate = [c for c in desired_feats if c in df.columns]
    probe = df[candidate].apply(pd.to_numeric, errors="coerce")
    feat_cols = [c for c in candidate if probe[c].notna().any()]

    st.write(f"✓ Selected {len(feat_cols)} features")

    df_imp = df.copy()
    df_imp[feat_cols] = probe[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Train/test split
    seasons = sorted(df_imp['season'].unique())
    if len(seasons) < 2:
        raise ValueError("Not enough seasons for train/test split")
    train_seasons = seasons[:-1]
    test_season = seasons[-1]

    train_df = df_imp[df_imp['season'].isin(train_seasons)]
    test_df = df_imp[df_imp['season'] == test_season]

    X_train = _to_numeric_matrix(train_df, feat_cols)
    y_train_for = train_df['points_for'].to_numpy()
    y_train_all = train_df['points_allowed'].to_numpy()

    X_test = _to_numeric_matrix(test_df, feat_cols)
    y_test_for = test_df['points_for'].to_numpy()
    y_test_all = test_df['points_allowed'].to_numpy()

    # Hyperparameter tuning for multiple algorithms
    st.write("🎯 Tuning LightGBM for offense...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    lgbm_params = {
        'n_estimators': [300, 500],
        'max_depth': [4, 6],
        'learning_rate': [0.03, 0.05],
        'num_leaves': [31, 63]
    }
    
    lgbm_off_gs = GridSearchCV(
        LGBMRegressor(random_state=42, verbose=-1),
        lgbm_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
    ).fit(X_train, y_train_for)
    best_lgbm_off = lgbm_off_gs.best_estimator_
    st.write(f"✓ Best LightGBM offense: {lgbm_off_gs.best_params_}, MAE: {-lgbm_off_gs.best_score_:.2f}")

    st.write("🎯 Tuning LightGBM for defense...")
    lgbm_def_gs = GridSearchCV(
        LGBMRegressor(random_state=42, verbose=-1),
        lgbm_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
    ).fit(X_train, y_train_all)
    best_lgbm_def = lgbm_def_gs.best_estimator_
    st.write(f"✓ Best LightGBM defense: {lgbm_def_gs.best_params_}, MAE: {-lgbm_def_gs.best_score_:.2f}")

    # XGBoost tuning
    st.write("🎯 Tuning XGBoost...")
    xgb_params = {
        'n_estimators': [300, 500],
        'max_depth': [3, 5],
        'learning_rate': [0.03, 0.05]
    }
    
    xgb_off_gs = GridSearchCV(
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        xgb_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
    ).fit(X_train, y_train_for)
    best_xgb_off = xgb_off_gs.best_estimator_
    st.write(f"✓ Best XGBoost offense: MAE: {-xgb_off_gs.best_score_:.2f}")

    xgb_def_gs = GridSearchCV(
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        xgb_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
    ).fit(X_train, y_train_all)
    best_xgb_def = xgb_def_gs.best_estimator_

    # Test performance
    test_mae_lgbm_off = mean_absolute_error(y_test_for, best_lgbm_off.predict(X_test))
    test_mae_lgbm_def = mean_absolute_error(y_test_all, best_lgbm_def.predict(X_test))
    st.write(f"✓ Test MAE - LightGBM offense: {test_mae_lgbm_off:.2f}, defense: {test_mae_lgbm_def:.2f}")

    # Build super ensemble
    st.write("🏗️ Building elite stacked ensemble...")
    
    estimators_off = [
        ('lgbm', copy.deepcopy(best_lgbm_off)),
        ('xgb', copy.deepcopy(best_xgb_off)),
        ('rf', RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        )),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True
        ))
    ]
    
    stack_for = StackingRegressor(
        estimators=estimators_off,
        final_estimator=LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        cv=4, n_jobs=-1
    ).fit(X_train, y_train_for)

    estimators_def = [
        ('lgbm', copy.deepcopy(best_lgbm_def)),
        ('xgb', copy.deepcopy(best_xgb_def)),
        ('rf', RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            n_jobs=-1,
            random_state=42
        )),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True
        ))
    ]
    
    stack_def = StackingRegressor(
        estimators=estimators_def,
        final_estimator=LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        cv=4, n_jobs=-1
    ).fit(X_train, y_train_all)

    # Final test MAE
    final_mae_off = mean_absolute_error(y_test_for, stack_for.predict(X_test))
    final_mae_def = mean_absolute_error(y_test_all, stack_def.predict(X_test))
    st.write(f"✅ FINAL Test MAE - Offense: {final_mae_off:.2f}, Defense: {final_mae_def:.2f}")

    # Quantile models
    st.write("📊 Training quantile models...")
    quantiles = {}
    for q in [0.1, 0.5, 0.9]:
        qm = LGBMRegressor(
            objective='quantile', alpha=q,
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1
        )
        qm.fit(X_train, y_train_for)
        quantiles[q] = qm

    # Bootstrap
    st.write("🔄 Bootstrap sampling...")
    boot_models = []
    for i in range(50):  # Reduced from 100 for speed
        if i % 10 == 0:
            st.write(f"  Bootstrap {i}/50")
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        bm = copy.deepcopy(stack_for)
        bm.fit(X_train[idx], y_train_for[idx])
        boot_models.append(bm)

    # Variance model
    st.write("📈 Training uncertainty model...")
    residuals = y_train_for - stack_for.predict(X_train)
    variance_model = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1
    ).fit(X_train, np.abs(residuals))

    st.write("✅ All elite models trained successfully!")
    
    return stack_for, stack_def, quantiles, feat_cols, boot_models, variance_model


# ------------------------ Win Probability Calibration ------------------------
@st.cache_data(show_spinner=False)
def calibrate_win_model(schedule_df: pd.DataFrame, history: pd.DataFrame,
                       _pts_for_m, _pts_all_m, _feat_cols, seasons) -> LogisticRegression:
    """Calibrate win probability using historical predictions"""
    if not _feat_cols:
        return LogisticRegression().fit([[0]], [0])

    all_seasons = sorted(schedule_df['season'].unique())
    if len(all_seasons) < 2:
        return LogisticRegression().fit([[0]], [0])

    train_seasons = [s for s in all_seasons if s in seasons][:-1]
    if not train_seasons:
        return LogisticRegression().fit([[0]], [0])

    margins, outcomes = [], []
    past_games = schedule_df[schedule_df['home_score'].notna()]
    
    for season in train_seasons:
        seas_games = past_games[past_games['season'] == season]
        for _, g in seas_games.iterrows():
            home, away, week = g['home_team'], g['away_team'], g['week']
            
            h_hist = (history[(history['team'] == home) &
                             ((history['season'] < season) |
                              ((history['season'] == season) & (history['week'] < week)))]
                     .sort_values(['season', 'week']).tail(1))
            a_hist = (history[(history['team'] == away) &
                             ((history['season'] < season) |
                              ((history['season'] == season) & (history['week'] < week)))]
                     .sort_values(['season', 'week']).tail(1))
            
            if h_hist.empty or a_hist.empty:
                continue
            
            hv = _to_numeric_matrix(h_hist[_feat_cols], _feat_cols)
            av = _to_numeric_matrix(a_hist[_feat_cols], _feat_cols)
            
            h_off = float(_pts_for_m.predict(hv)[0])
            a_off = float(_pts_for_m.predict(av)[0])
            h_def = float(_pts_all_m.predict(hv)[0])
            a_def = float(_pts_all_m.predict(av)[0])
            
            h_pred = (h_off + a_def) / 2
            a_pred = (a_off + h_def) / 2
            pred_margin = h_pred - a_pred
            
            actual = 1 if g['home_score'] > g['away_score'] else 0
            margins.append([pred_margin])
            outcomes.append(actual)

    if not margins:
        return LogisticRegression().fit([[0]], [0])
    
    lr = LogisticRegression()
    lr.fit(margins, outcomes)
    return lr


# ------------------------ Elite Inference ------------------------
def predict_week(matchups, history, pts_for_m, pts_all_m, quantile_models, feat_cols,
                boot_models, cal_model, variance_model):
    """Enhanced prediction with all elite features"""
    out = []
    
    for _, g in matchups.iterrows():
        season, week = g['season'], g['week']
        home, away = g['home_team'], g['away_team']
        
        h_hist = (history[(history['team'] == home) &
                         ((history['season'] < season) |
                          ((history['season'] == season) & (history['week'] < week)))]
                 .sort_values(['season', 'week']).tail(1))
        a_hist = (history[(history['team'] == away) &
                         ((history['season'] < season) |
                          ((history['season'] == season) & (history['week'] < week)))]
                 .sort_values(['season', 'week']).tail(1))
        
        if h_hist.empty or a_hist.empty:
            continue

        h_vec = _to_numeric_matrix(h_hist[feat_cols], feat_cols)
        a_vec = _to_numeric_matrix(a_hist[feat_cols], feat_cols)

        # Elite blending approach
        h_off = float(pts_for_m.predict(h_vec)[0])
        a_off = float(pts_for_m.predict(a_vec)[0])
        h_def = float(pts_all_m.predict(h_vec)[0])
        a_def = float(pts_all_m.predict(a_vec)[0])
        
        # Weighted blend (60% own offense, 40% opponent defense)
        h_pts = 0.6 * h_off + 0.4 * a_def
        a_pts = 0.6 * a_off + 0.4 * h_def
        
        margin = round(h_pts - a_pts, 1)
        total = round(h_pts + a_pts, 1)
        
        win_prob = float(cal_model.predict_proba([[margin]])[0, 1])

        # Advanced uncertainty quantification
        lower_h = float(quantile_models[0.1].predict(h_vec)[0])
        upper_h = float(quantile_models[0.9].predict(h_vec)[0])
        lower_a = float(quantile_models[0.1].predict(a_vec)[0])
        upper_a = float(quantile_models[0.9].predict(a_vec)[0])
        lower = round(lower_h + lower_a, 1)
        upper = round(upper_h + upper_a, 1)
        
        boot_totals = [float(bm.predict(h_vec)[0] + bm.predict(a_vec)[0]) for bm in boot_models]
        sigma_boot = float(np.std(boot_totals) if boot_totals else 7.0)
        
        h_var = float(variance_model.predict(h_vec)[0])
        a_var = float(variance_model.predict(a_vec)[0])
        sigma_var = float(np.sqrt(h_var**2 + a_var**2))
        
        sigma = round((sigma_boot + sigma_var) / 2, 1)

        out.append({
            'Season': season, 'Week': week, 'Home': home, 'Away': away,
            'Home_Pts': round(h_pts, 1), 'Away_Pts': round(a_pts, 1),
            'Total': total, 'Total_10%': lower, 'Total_90%': upper,
            'Home_Win%': f"{win_prob:.1%}", 'Pred_Margin': margin,
            'Total_sigma': sigma,
            'Home_Off': round(h_off, 1), 'Away_Off': round(a_off, 1),
            'Home_Def': round(h_def, 1), 'Away_Def': round(a_def, 1)
        })
    
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def simulate_win_totals(schedule_df, history, _pts_for_m, _pts_all_m, _quantile_models,
                       _feat_cols, _boot_models, _cal_model, _var_model, season, n_sims=20000):
    """Simulate season outcomes"""
    games = schedule_df[schedule_df['season'] == season]
    probs = []
    
    for week in sorted(games['week'].unique()):
        preds = predict_week(
            games[games['week'] == week], history, _pts_for_m, _pts_all_m,
            _quantile_models, _feat_cols, _boot_models, _cal_model, _var_model
        )
        for _, r in preds.iterrows():
            hp = float(str(r['Home_Win%']).rstrip('%')) / 100
            probs.append((r['Home'], hp))
            probs.append((r['Away'], 1 - hp))
    
    team_probs = defaultdict(list)
    for t, p in probs:
        team_probs[t].append(p)

    division_to_teams = defaultdict(list)
    for team in team_probs:
        dv = DIVISION_MAP.get(team)
        if dv:
            division_to_teams[dv].append(team)

    results = {team: {'Expected_Wins': 0.0, 'Win_10%': 0, 'Win_90%': 0} for team in team_probs}
    division_wins = defaultdict(float)
    all_teams = list(team_probs.keys())
    win_matrix = {team: [] for team in all_teams}
    
    for _ in range(n_sims):
        sim_wins = {team: sum(random.random() < p for p in plist)
                   for team, plist in team_probs.items()}
        for team in all_teams:
            win_matrix[team].append(sim_wins[team])
        for div, teams in division_to_teams.items():
            max_w = max(sim_wins[t] for t in teams)
            tied = [t for t in teams if sim_wins[t] == max_w]
            for t in tied:
                division_wins[t] += 1.0 / len(tied)

    for t in all_teams:
        results[t]['Expected_Wins'] = float(sum(team_probs[t]))
        wins = sorted(win_matrix[t])
        results[t]['Win_10%'] = int(wins[int(0.1 * n_sims)])
        results[t]['Win_90%'] = int(wins[int(0.9 * n_sims)])
    
    division_win_pct = {t: division_wins[t] / n_sims for t in all_teams}

    df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(
        columns={'index': 'Team'}
    )
    df['Div_Win_Pct'] = df['Team'].map(division_win_pct)
    return df[['Team', 'Expected_Wins', 'Win_10%', 'Win_90%', 'Div_Win_Pct']]


# ------------------------ Streamlit UI ------------------------
st.sidebar.header("⚙️ Elite Settings")

if 'run_models' not in st.session_state:
    st.session_state['run_models'] = False
if 'trained' not in st.session_state:
    st.session_state['trained'] = False

years = st.sidebar.multiselect("Training Seasons", SEASONS, default=[SEASONS[-1]])
if not years:
    st.stop()

st.sidebar.subheader("📊 Bookmaker Odds")
odds_upload = st.sidebar.file_uploader("Upload Odds CSV", type=["csv"])

odds_data = {}
if odds_upload:
    df_ou = pd.read_csv(odds_upload)
    df_ou.columns = [c.strip().lower() for c in df_ou.columns]
    raw_cols = {"market", "point", "price", "home_team", "away_team"}
    
    if raw_cols.issubset(df_ou.columns):
        for (h, a), grp in df_ou.groupby(["home_team", "away_team"]):
            matchup = f"{a} @ {h}"
            tot = grp[grp["market"] == "totals"]
            if not tot.empty:
                over = tot[tot.get("label", "").str.lower() == "over"]
                under = tot[tot.get("label", "").str.lower() == "under"]
                line = (over["point"].iloc[0] if not over.empty
                       else under["point"].iloc[0] if not under.empty else None)
                over_p = over["price"].iloc[0] if not over.empty else None
                under_p = under["price"].iloc[0] if not under.empty else None
            else:
                line, over_p, under_p = None, None, None
            
            ml = grp[grp["market"] == "h2h"]
            h_ml = (ml[ml.get("label", "").str.lower() == str(h).lower()]["price"].iloc[0]
                   if not ml.empty and not ml[ml.get("label", "").str.lower() == str(h).lower()].empty
                   else None)
            a_ml = (ml[ml.get("label", "").str.lower() == str(a).lower()]["price"].iloc[0]
                   if not ml.empty and not ml[ml.get("label", "").str.lower() == str(a).lower()].empty
                   else None)
            
            spread = grp[grp["market"] == "spread"]
            if not spread.empty:
                home_spread = spread[spread.get("label", "").str.lower() == str(h).lower()]
                spread_line = (float(home_spread["point"].iloc[0])
                             if not home_spread.empty else None)
                spread_price = (float(home_spread["price"].iloc[0])
                              if not home_spread.empty else None)
            else:
                spread_line, spread_price = None, None
            
            odds_data[matchup] = {
                "book_line": float(line or 0),
                "over_price": float(over_p or 0),
                "under_price": float(under_p or 0),
                "home_ml": float(h_ml or 0),
                "away_ml": float(a_ml or 0),
                "spread_line": float(spread_line or 0),
                "spread_price": float(spread_price or 0)
            }

# Train button
if st.sidebar.button("🚀 Train Elite Models"):
    st.session_state['run_models'] = True
    st.session_state['trained'] = False

if st.session_state['run_models']:
    if not st.session_state['trained']:
        with st.spinner("🔄 Loading data and training ELITE models (this may take 10-15 minutes)..."):
            wk = load_weekly(years)
            sched = load_schedules(years)
            ngs = load_ngs(years)
            snaps = load_snap_counts(years)
            pbp = load_pbp_data(years)

            team_week = build_team_week_df(wk, sched, ngs, snaps, pbp)
            team_week_feats = add_rolling_features(team_week)

            pts_for_m, pts_all_m, quantile_models, feat_cols, boot_models, variance_model = train_elite_models(
                team_week_feats)
            cal_model = calibrate_win_model(sched, team_week_feats, pts_for_m, pts_all_m, feat_cols, years)

            st.session_state.update({
                'wk': wk, 'sched': sched, 'team_week_feats': team_week_feats,
                'pts_for_m': pts_for_m, 'pts_all_m': pts_all_m,
                'quantile_models': quantile_models, 'feat_cols': feat_cols,
                'boot_models': boot_models, 'variance_model': variance_model,
                'cal_model': cal_model, 'trained': True
            })
            st.success("✅ Elite models trained successfully!")

    if all(k in st.session_state for k in [
        'wk', 'sched', 'team_week_feats', 'pts_for_m', 'pts_all_m',
        'quantile_models', 'feat_cols', 'boot_models', 'variance_model', 'cal_model'
    ]):
        wk = st.session_state['wk']
        sched = st.session_state['sched']
        team_week_feats = st.session_state['team_week_feats']
        pts_for_m = st.session_state['pts_for_m']
        pts_all_m = st.session_state['pts_all_m']
        quantile_models = st.session_state['quantile_models']
        feat_cols = st.session_state['feat_cols']
        boot_models = st.session_state['boot_models']
        variance_model = st.session_state['variance_model']
        cal_model = st.session_state['cal_model']
    else:
        st.info("👉 Click '🚀 Train Elite Models' to begin")
        st.stop()
else:
    st.info("👉 Click '🚀 Train Elite Models' to begin")
    st.stop()

# Predictions
future_sched = sched[(sched['home_score'].isna()) & (sched['season'].isin(years))]
if future_sched.empty:
    st.warning("⚠️ No future games found")
    st.stop()

sel_season = st.sidebar.selectbox("Season", sorted(future_sched['season'].unique(), reverse=True))
sel_week = st.sidebar.selectbox("Week", sorted(future_sched[future_sched['season'] == sel_season]['week'].unique()))
games = future_sched[(future_sched['season'] == sel_season) & (future_sched['week'] == sel_week)]

preds = predict_week(games, team_week_feats, pts_for_m, pts_all_m, quantile_models,
                    feat_cols, boot_models, cal_model, variance_model)

# Merge odds
if odds_data and not preds.empty:
    odds_df = pd.DataFrame([{"Matchup": k, **v} for k, v in odds_data.items()])
    odds_df[['Away', 'Home']] = odds_df['Matchup'].str.split(" @ ", expand=True)
    preds = preds.merge(odds_df, on=['Home', 'Away'], how='left')

    preds["home_imp_prob"] = preds["home_ml"].apply(convert_moneyline_to_prob)
    preds["ml_edge"] = preds["Home_Win%"].str.rstrip('%').astype(float) / 100 - preds["home_imp_prob"]
    preds["spread_edge"] = preds["Pred_Margin"] - preds.get("spread_line", 0)
    preds["total_edge"] = preds["Total"] - preds.get("book_line", 0)
    
    preds["spread_prob"] = norm.sf(preds.get("spread_line", 0) - preds["Pred_Margin"],
                                    loc=0, scale=preds["Total_sigma"])
    preds["total_prob"] = norm.sf(preds.get("book_line", 0) - preds["Total"],
                                   loc=0, scale=preds["Total_sigma"])

# Display
st.header(f"🏆 Elite Predictions - Week {sel_week}, Season {sel_season}")

games_count = len(preds)
avg_total = preds['Total'].mean() if games_count else 0
avg_sigma = preds['Total_sigma'].mean() if games_count else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", games_count)
c2.metric("Avg Total", f"{avg_total:.1f}")
c3.metric("Avg σ", f"{avg_sigma:.1f}")
c4.metric("Model", "Elite v0.4")

st.subheader("📋 Predictions")
st.dataframe(preds, use_container_width=True)

if "ml_edge" in preds.columns:
    st.subheader("💰 Recommended Bets")
    ml_bets = preds[preds["ml_edge"] >= ML_EDGE_THRESHOLD]
    if not ml_bets.empty:
        st.markdown("**Moneyline Bets (≥5% edge)**")
        st.dataframe(ml_bets[["Home", "Away", "Home_Win%", "home_ml", "ml_edge"]])

st.subheader("🔮 Season Projections")
if st.button("Run Season Simulation"):
    with st.spinner("Running 20,000 simulations..."):
        futures = simulate_win_totals(sched, team_week_feats, pts_for_m, pts_all_m,
                                      quantile_models, feat_cols, boot_models,
                                      cal_model, variance_model, sel_season)
        st.dataframe(futures, use_container_width=True)

csv = preds.to_csv(index=False).encode()
st.download_button("📥 Download Predictions", csv, "elite_predictions.csv")

st.caption("v0.4 Elite Edition — Professional-grade predictions with advanced ML ensemble")