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
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from scipy.stats import norm

# ------------------------ Odds helper and thresholds ------------------------
def convert_moneyline_to_prob(ml):
    try:
        return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
    except Exception:
        return 0.5

ML_EDGE_THRESHOLD = 0.05
SPREAD_EDGE_THRESHOLD = 3.0
TOTAL_EDGE_THRESHOLD = 4.0

# NFL key numbers for spreads and totals
SPREAD_KEY_NUMBERS = [
    3, 7, 6, 10, 4, 14, 1, 2, 17, 5, 8, 13, 11, 21, 20, 18, 24, 16, 9, 12,
    15, 28, 27, 19, 23, 31, 25, 22, 26, 35, 34, 30, 38, 29, 32, 37, 33, 0,
    41, 40, 36, 45, 42, 39, 43, 44, 49, 46, 48, 52, 58, 55, 59, 54, 51
]
TOTAL_KEY_NUMBERS = [
    41, 46, 37, 42, 44, 55, 51, 45, 43, 40, 47, 48, 33, 39, 38, 30, 36, 34,
    27, 25, 23, 17, 50, 31, 54, 29, 26, 35, 32, 24, 22, 20, 21, 19, 16, 15,
    18, 14, 12, 13, 11, 9, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0
]

# 2025 Strength of Schedule (Opp. Win Pct)
SOS_OWP = {
    'NYG': 0.574, 'DET': 0.571, 'CHI': 0.571, 'PHI': 0.561, 'MIN': 0.557, 'DAL': 0.557,
    'GB' : 0.557, 'WAS': 0.550, 'BAL': 0.533, 'PIT': 0.526, 'KC' : 0.522, 'LAC': 0.522,
    'CLE': 0.519, 'CIN': 0.509, 'DEN': 0.505, 'LV' : 0.502, 'LAR': 0.491, 'TB' : 0.481,
    'HOU': 0.481, 'ATL': 0.478, 'MIA': 0.474, 'SEA': 0.474, 'BUF': 0.467, 'JAX': 0.467,
    'IND': 0.464, 'NYJ': 0.460, 'CAR': 0.457, 'ARI': 0.457, 'TEN': 0.450, 'NE' : 0.429,
    'NO' : 0.419, 'SF' : 0.415
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

# Historical Home Field Advantage by team (example values - calculate from data)
TEAM_HFA = {team: 2.5 for team in SOS_OWP.keys()}  # Default 2.5, will be calculated

# ------------------------ Streamlit setup ------------------------
st.set_page_config(page_title="FoxEdge NFL Predictor v0.3", layout="wide", page_icon="🏈")

st.markdown("""
**FoxEdge — NFL Matchup Predictor (v0.3 Enhanced)**

Major Improvements:
- Fixed defense model logic with proper offense/defense blending
- Added opponent-adjusted features for matchup-specific predictions
- Exponential recency weighting for rolling windows
- Team-specific home field advantage
- Division game indicators
- Short rest penalties
- Weather impact modeling (dome/bad weather)
- Improved uncertainty quantification
- Separate pass/rush efficiency modeling
""")

# ------------------------ Config ------------------------
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

# ------------------------ Data loaders ------------------------
@st.cache_data(show_spinner=False)
def load_weekly(years):
    collected = []
    for yr in years:
        try:
            df_yr = nfl.import_weekly_data([yr])
        except Exception as e:
            st.warning(f"⚠️ weekly parquet {yr} not found — skipping. ({e})")
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
def load_ngs(years):
    groups = ['receiving', 'rushing']
    aggs = []
    for grp in groups:
        try:
            rec = nfl.import_ngs_data(grp, years)
            if 'team' not in rec.columns and 'recent_team' in rec.columns:
                rec = rec.rename(columns={'recent_team': 'team'})
            elif 'team_abbr' in rec.columns:
                rec = rec.rename(columns={'team_abbr': 'team'})
            if grp == 'receiving':
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
                if not yac_col:
                    continue
                rec = rec.rename(columns={yac_col[0]: 'receiving_yac_oe', **req})
                agg = rec.groupby(['season','week','team']).agg({
                    'receiving_yac_oe':'mean',
                    'receiving_air_share':'mean',
                    'receiving_yac_above':'mean',
                    'receiving_intended_air_yards':'mean',
                    'receiving_targets':'sum',
                    'receiving_receptions':'sum',
                    'receiving_catch_pct':'mean',
                    'receiving_avg_sep':'mean',
                    'receiving_avg_cushion':'mean'
                }).reset_index()
            else:
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
                if not yac_cols:
                    continue
                rec = rec.rename(columns={yac_cols[0]: 'rushing_yac_oe', **reqr})
                agg = rec.groupby(['season','week','team']).agg({
                    'rushing_yac_oe':'mean',
                    'rush_efficiency':'mean',
                    'rush_ttl':'mean',
                    'rush_ye_per_att':'mean',
                    'rushing_attempts':'sum',
                    'rushing_exp_yards':'mean',
                    'rushing_pct_over_exp':'mean',
                    'rush_vs_box_pct':'mean'
                }).reset_index()
        except Exception:
            continue
        aggs.append(agg)
    merged = None
    for a in aggs:
        merged = a if merged is None else merged.merge(a, on=['season','week','team'], how='outer')
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
        required = {'season','week','team','snap_counts'}
        if not required.issubset(snaps.columns):
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    def frac(sub):
        total = sub['snap_counts'].sum()
        top22 = sub.nlargest(22, 'snap_counts')['snap_counts'].sum()
        return pd.Series({'starter_frac': top22 / total if total > 0 else np.nan})

    starter = snaps.groupby(['season','week','team']).apply(frac).reset_index()
    def_pct = snaps.groupby(['season','week','team'])['defense_pct'].mean().reset_index().rename(columns={'defense_pct':'def_pct'})
    return starter.merge(def_pct, on=['season','week','team'], how='left')

@st.cache_data(show_spinner=False)
def load_pbp_data(years):
    try:
        pbp = nfl.import_pbp_data(years)
    except Exception:
        return pd.DataFrame()
    pbp['success'] = pbp['epa'] > 0
    agg = (pbp.groupby(['season','week','posteam'])
              .agg(epa_pp=('epa','mean'), sr=('success','mean'), n_plays=('play_id','count'))
              .reset_index().rename(columns={'posteam':'team'}))
    def_agg = (pbp.groupby(['season','week','defteam'])
                 .agg(def_epa_pp=('epa','mean'), def_sr=('success','mean'), def_n_plays=('play_id','count'))
                 .reset_index().rename(columns={'defteam':'team'}))
    agg = agg.merge(def_agg, on=['season','week','team'], how='left')

    # extra splits
    def _split(col_team, col_name, mask):
        return (pbp[mask].groupby(['season','week',col_team])
                .agg(**{col_name: ('epa','mean')}).reset_index().rename(columns={col_team:'team'}))
    agg = agg.merge(_split('posteam','off_pass_epa_pp', pbp['pass']==1), on=['season','week','team'], how='left')
    agg = agg.merge(_split('posteam','off_rush_epa_pp', pbp['rush']==1), on=['season','week','team'], how='left')
    agg = agg.merge(_split('defteam','def_pass_epa_pp', pbp['pass']==1), on=['season','week','team'], how='left')
    agg = agg.merge(_split('defteam','def_rush_epa_pp', pbp['rush']==1), on=['season','week','team'], how='left')

    def rate(col_src, name):
        return (pbp.groupby(['season','week',col_src]).agg(**{name: ('third_down_converted','mean')})
                .reset_index().rename(columns={col_src:'team'}))
    agg = agg.merge(rate('posteam','off_3rd_conv'), on=['season','week','team'], how='left')
    agg = agg.merge(rate('defteam','def_3rd_conv'), on=['season','week','team'], how='left')

    def flag(col_src, name):
        return (pbp.groupby(['season','week',col_src]).agg(**{name: ('sack','mean')})
                .reset_index().rename(columns={col_src:'team'}))
    agg = agg.merge(flag('posteam','off_sack_rate'), on=['season','week','team'], how='left')
    agg = agg.merge(flag('defteam','def_sack_rate'), on=['season','week','team'], how='left')

    def pick(col_src, name):
        return (pbp.groupby(['season','week',col_src]).agg(**{name: ('interception','mean')})
                .reset_index().rename(columns={col_src:'team'}))
    agg = agg.merge(pick('posteam','off_int_rate'), on=['season','week','team'], how='left')
    agg = agg.merge(pick('defteam','def_int_rate'), on=['season','week','team'], how='left')

    g2g = (pbp[pbp['goal_to_go']==1].groupby(['season','week','posteam']).agg(g2g_conv=('success','mean'))
           .reset_index().rename(columns={'posteam':'team'}))
    td_long = (pbp[(pbp['down']==3)&(pbp['ydstogo']>7)].groupby(['season','week','posteam'])
               .agg(td_conv_long=('success','mean')).reset_index().rename(columns={'posteam':'team'}))
    td_short = (pbp[(pbp['down']==3)&(pbp['ydstogo']<=7)].groupby(['season','week','posteam'])
                .agg(td_conv_short=('success','mean')).reset_index().rename(columns={'posteam':'team'}))
    nh = (pbp.groupby(['season','week','posteam']).agg(no_huddle_rate=('no_huddle','mean'))
          .reset_index().rename(columns={'posteam':'team'}))
    wp_agg = (pbp.groupby(['season','week','posteam']).agg(avg_wp=('wp','mean'), avg_vegas_wpa=('vegas_wpa','mean'))
              .reset_index().rename(columns={'posteam':'team'}))
    for extra in [g2g, td_long, td_short, nh, wp_agg]:
        agg = agg.merge(extra, on=['season','week','team'], how='left')
    return agg

# ------------------------ Feature engineering ------------------------
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
            hfa_dict[team] = 2.5  # Default
    return hfa_dict

def build_team_week_df(weekly_df, sched_df, ngs_df, snaps_df, pbp_df) -> pd.DataFrame:
    # 1) Aggregate player -> team-week
    key_off_cols = [
        'passing_yards','rushing_yards','passing_tds','rushing_tds',
        'interceptions','sacks','fumbles','targets','receptions',
        'yards_after_catch','passing_air_yards','cpoe','epa'
    ]
    existing_cols = [c for c in key_off_cols if c in weekly_df.columns]
    team_week = weekly_df.groupby(['season','week','team'])[existing_cols].sum().reset_index()

    # 2) Schedules: normalize columns and parse dates
    base_cols = ['season','week','home_team','away_team','home_score','away_score','gameday','game_date','roof','temp','wind']
    sched_df = sched_df[[c for c in base_cols if c in sched_df.columns]].copy()

    if 'game_date' not in sched_df.columns and 'gameday' in sched_df.columns:
        sched_df['game_date'] = sched_df['gameday']
    if 'game_date' in sched_df.columns:
        sched_df['game_date'] = pd.to_datetime(sched_df['game_date'], errors='coerce')

    for c in ['season','week']:
        if c in sched_df.columns:
            sched_df[c] = pd.to_numeric(sched_df[c], errors='coerce')

    # Only games that have been played
    played = sched_df.dropna(subset=['home_score', 'away_score'], how='any').copy()

    # 3) Build per-team scores view
    home_rows = played.rename(columns={
        'home_team':'team','away_team':'opp',
        'home_score':'points_for','away_score':'points_allowed'
    }).assign(is_home=1)
    away_rows = played.rename(columns={
        'away_team':'team','home_team':'opp',
        'away_score':'points_for','home_score':'points_allowed'
    }).assign(is_home=0)

    # Carry over weather/roof columns
    weather_cols = [c for c in ['roof','temp','wind'] if c in played.columns]
    scores = pd.concat([
        home_rows[['season','week','team','opp','points_for','points_allowed','is_home'] + weather_cols],
        away_rows[['season','week','team','opp','points_for','points_allowed','is_home'] + weather_cols]
    ], ignore_index=True)

    team_week = team_week.merge(scores, on=['season','week','team'], how='left')

    # 4) Seasonality normalization
    max_weeks = played.groupby('season')['week'].max().to_dict()
    team_week['wk_norm'] = team_week['week'] / team_week['season'].map(max_weeks)

    # 5) Rest days
    if 'game_date' in played.columns:
        home_dates = played[['season','week','home_team','game_date']].rename(columns={'home_team':'team'})
        away_dates = played[['season','week','away_team','game_date']].rename(columns={'away_team':'team'})
        game_dates = pd.concat([home_dates, away_dates], ignore_index=True)

        team_week = team_week.merge(game_dates, on=['season','week','team'], how='left')
        team_week['game_date'] = pd.to_datetime(team_week['game_date'], errors='coerce')
        team_week.sort_values(['team','game_date'], inplace=True)
        team_week['rest_days'] = (
            team_week.groupby('team')['game_date'].diff().dt.days.fillna(7)
        )
        team_week.drop(columns='game_date', inplace=True)
    else:
        team_week['rest_days'] = 7

    # 6) Enhanced rest features
    team_week['short_rest'] = (team_week['rest_days'] < 6).astype(int)
    team_week['long_rest'] = (team_week['rest_days'] > 10).astype(int)

    # 7) Weather features
    if 'temp' in team_week.columns:
        team_week['temp'] = pd.to_numeric(team_week['temp'], errors='coerce')
        team_week['bad_weather'] = ((team_week['temp'] < 40) | (team_week.get('wind', 0) > 15)).astype(int)
    else:
        team_week['bad_weather'] = 0
    
    if 'roof' in team_week.columns:
        team_week['dome_game'] = team_week['roof'].isin(['dome', 'closed']).astype(int)
    else:
        team_week['dome_game'] = 0

    # 8) Division game indicator
    team_week['division_game'] = team_week.apply(
        lambda r: int(DIVISION_MAP.get(r['team'], '') == DIVISION_MAP.get(r['opp'], '')) if pd.notna(r['opp']) else 0,
        axis=1
    )

    # 9) Merge NGS
    if not ngs_df.empty:
        team_week = team_week.merge(ngs_df, on=['season','week','team'], how='left')

    # 10) Merge snaps
    if not snaps_df.empty and {'season','week','team','starter_frac'}.issubset(snaps_df.columns):
        team_week = team_week.merge(snaps_df, on=['season','week','team'], how='left')

    # 11) Merge play-by-play efficiency
    pbp_cols_needed = [
        'season','week','team','epa_pp','sr','n_plays',
        'def_epa_pp','def_sr','def_n_plays',
        'off_pass_epa_pp','off_rush_epa_pp','def_pass_epa_pp','def_rush_epa_pp',
        'off_3rd_conv','def_3rd_conv',
        'off_sack_rate','def_sack_rate','off_int_rate','def_int_rate',
        'g2g_conv','td_conv_long','td_conv_short','no_huddle_rate','avg_wp','avg_vegas_wpa'
    ]
    if not pbp_df.empty and set(pbp_cols_needed).issubset(pbp_df.columns):
        team_week = team_week.merge(pbp_df, on=['season','week','team'], how='left')

    return team_week.dropna(subset=['points_for'])


def add_rolling_features(team_week_df: pd.DataFrame) -> pd.DataFrame:
    df = team_week_df.copy()
    
    # Calculate team-specific HFA
    hfa_dict = calculate_team_hfa(df)
    df['team_hfa'] = df.apply(
        lambda r: hfa_dict.get(r['team'], 2.5) if r['is_home'] == 1 else 0,
        axis=1
    )
    
    # Enhanced rolling windows with exponential weighting
    for w in ROLL_WINDOWS:
        sorted_df = df.sort_values(['team','season','week'])
        
        # Standard rolling averages
        df[f'pf_r{w}'] = sorted_df.groupby('team')['points_for'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        ).reindex(df.index)
        df[f'pa_r{w}'] = sorted_df.groupby('team')['points_allowed'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        ).reindex(df.index)
        
        # Exponentially weighted rolling averages (more recent = more weight)
        df[f'pf_ewm{w}'] = sorted_df.groupby('team')['points_for'].transform(
            lambda s: s.shift(1).ewm(span=w, min_periods=1).mean()
        ).reindex(df.index)
        df[f'pa_ewm{w}'] = sorted_df.groupby('team')['points_allowed'].transform(
            lambda s: s.shift(1).ewm(span=w, min_periods=1).mean()
        ).reindex(df.index)

        # Rolling for advanced metrics
        for base in ['def_epa_pp','def_sr','off_pass_epa_pp','off_rush_epa_pp','def_pass_epa_pp','def_rush_epa_pp',
                     'off_3rd_conv','def_3rd_conv','off_sack_rate','def_sack_rate','off_int_rate','def_int_rate']:
            if base in df.columns:
                df[f'{base}_r{w}'] = sorted_df.groupby('team')[base].transform(
                    lambda s: s.shift(1).rolling(w, min_periods=1).mean()
                ).reindex(df.index)
                df[f'{base}_ewm{w}'] = sorted_df.groupby('team')[base].transform(
                    lambda s: s.shift(1).ewm(span=w, min_periods=1).mean()
                ).reindex(df.index)

    if 'receiving_yac_oe' in df.columns:
        sorted_df = df.sort_values(['team','season','week'])
        df['rec_yac_oe_r3'] = sorted_df.groupby('team')['receiving_yac_oe'].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        ).reindex(df.index)

    # Build opponent features
    base_feat_cols = []
    for w in ROLL_WINDOWS:
        base_feat_cols.extend([f"pf_r{w}", f"pa_r{w}", f"pf_ewm{w}", f"pa_ewm{w}"])
    
    if 'rec_yac_oe_r3' in df.columns:
        base_feat_cols.append('rec_yac_oe_r3')

    # Add defensive metrics to opponent features
    for w in ROLL_WINDOWS:
        for metric in ['def_epa_pp', 'def_pass_epa_pp', 'def_rush_epa_pp']:
            if f'{metric}_r{w}' in df.columns:
                base_feat_cols.append(f'{metric}_r{w}')

    opp_df = df[['season','week','team'] + base_feat_cols].rename(
        columns={c: f"opp_{c}" for c in base_feat_cols}
    )
    df = df.merge(
        opp_df, 
        left_on=['season','week','opp'], 
        right_on=['season','week','team'], 
        how='left', 
        suffixes=('','_opp')
    )
    df.drop(columns=['team_opp'], inplace=True, errors='ignore')
    
    # Matchup-specific differentials
    df['pf_diff_r3'] = df.get('pf_r3', np.nan) - df.get('opp_pa_r3', np.nan)
    df['pa_diff_r3'] = df.get('pa_r3', np.nan) - df.get('opp_pf_r3', np.nan)
    df['pf_diff_ewm3'] = df.get('pf_ewm3', np.nan) - df.get('opp_pa_ewm3', np.nan)
    df['pa_diff_ewm3'] = df.get('pa_ewm3', np.nan) - df.get('opp_pf_ewm3', np.nan)
    
    # Offense vs Defense matchup ratings
    if 'off_pass_epa_pp_r3' in df.columns and 'opp_def_pass_epa_pp_r3' in df.columns:
        df['pass_matchup'] = df['off_pass_epa_pp_r3'] - df['opp_def_pass_epa_pp_r3']
    if 'off_rush_epa_pp_r3' in df.columns and 'opp_def_rush_epa_pp_r3' in df.columns:
        df['rush_matchup'] = df['off_rush_epa_pp_r3'] - df['opp_def_rush_epa_pp_r3']
    
    return df

# ------------------------ Model training ------------------------
def train_models(df: pd.DataFrame):
    """Train separate offense and defense models with enhanced features"""
    
    # Feature selection - include new enhanced features
    desired_feats = []
    
    # Rolling and EWM features
    for w in ROLL_WINDOWS:
        desired_feats.extend([
            f'pf_r{w}', f'pa_r{w}', f'pf_ewm{w}', f'pa_ewm{w}',
            f'opp_pf_r{w}', f'opp_pa_r{w}', f'opp_pf_ewm{w}', f'opp_pa_ewm{w}'
        ])
    
    # Differentials
    desired_feats.extend(['pf_diff_r3', 'pa_diff_r3', 'pf_diff_ewm3', 'pa_diff_ewm3'])
    
    # Matchup features
    desired_feats.extend(['pass_matchup', 'rush_matchup'])
    
    # EPA and efficiency metrics (rolling)
    for w in ROLL_WINDOWS:
        for base in ['def_epa_pp', 'def_sr', 'off_pass_epa_pp', 'off_rush_epa_pp',
                     'def_pass_epa_pp', 'def_rush_epa_pp', 'off_3rd_conv', 'def_3rd_conv',
                     'off_sack_rate', 'def_sack_rate', 'off_int_rate', 'def_int_rate']:
            desired_feats.extend([f'{base}_r{w}', f'{base}_ewm{w}'])
            desired_feats.append(f'opp_{base}_r{w}')
    
    # Current week metrics
    desired_feats.extend([
        'rest_days', 'short_rest', 'long_rest', 'starter_frac', 'epa_pp', 'sr', 'n_plays',
        'def_pct', 'def_epa_pp', 'def_sr', 'def_n_plays',
        'off_pass_epa_pp', 'off_rush_epa_pp', 'def_pass_epa_pp', 'def_rush_epa_pp',
        'off_3rd_conv', 'def_3rd_conv', 'off_sack_rate', 'def_sack_rate', 
        'off_int_rate', 'def_int_rate',
        'is_home', 'team_hfa', 'wk_norm', 'temp', 'wind',
        'bad_weather', 'dome_game', 'division_game',
        'receiving_yac_oe', 'receiving_air_share', 'receiving_yac_above', 'rec_yac_oe_r3',
        'rushing_yac_oe', 'rush_efficiency', 'rush_ttl', 'rush_ye_per_att',
        'receiving_intended_air_yards', 'receiving_targets', 'receiving_receptions',
        'receiving_catch_pct', 'receiving_avg_sep', 'receiving_avg_cushion',
        'rushing_attempts', 'rushing_exp_yards', 'rushing_pct_over_exp', 'rush_vs_box_pct',
        'g2g_conv', 'td_conv_long', 'td_conv_short', 'no_huddle_rate', 'avg_wp', 'avg_vegas_wpa'
    ])

    # Filter to existing columns
    candidate = [c for c in desired_feats if c in df.columns]
    probe = df[candidate].apply(pd.to_numeric, errors="coerce")
    feat_cols = [c for c in candidate if probe[c].notna().any()]

    df_imp = df.copy()
    df_imp[feat_cols] = probe[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

    # Tune offense model
    tscv_tune = TimeSeriesSplit(n_splits=5)
    xgb_param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.05, 0.07]
    }
    
    st.write("Tuning offense model...")
    xgb_for_gs = GridSearchCV(
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        xgb_param_grid, cv=tscv_tune, scoring='neg_mean_absolute_error', n_jobs=-1
    ).fit(X_train, y_train_for)
    best_for = xgb_for_gs.best_estimator_
    st.write(f"✓ Best XGB offense: {xgb_for_gs.best_params_}, MAE(cv): {-xgb_for_gs.best_score_:.2f}")

    st.write("Tuning defense model...")
    xgb_all_gs = GridSearchCV(
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        xgb_param_grid, cv=tscv_tune, scoring='neg_mean_absolute_error', n_jobs=-1
    ).fit(X_train, y_train_all)
    best_all = xgb_all_gs.best_estimator_
    st.write(f"✓ Best XGB defense: {xgb_all_gs.best_params_}, MAE(cv): {-xgb_all_gs.best_score_:.2f}")

    test_mae_for = mean_absolute_error(y_test_for, best_for.predict(X_test))
    test_mae_all = mean_absolute_error(y_test_all, best_all.predict(X_test))
    st.write(f"✓ Test MAE offense: {test_mae_for:.2f}, defense: {test_mae_all:.2f}")

    # Stacking ensembles
    st.write("Building stacked ensembles...")
    estimators_for = [
        ('xgb', copy.deepcopy(best_for)),
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            n_jobs=-1,
            random_state=42
        ))
    ]
    stack_for = StackingRegressor(
        estimators=estimators_for,
        final_estimator=XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        ),
        cv=4,
        n_jobs=-1
    ).fit(X_train, y_train_for)

    estimators_def = [
        ('xgb', copy.deepcopy(best_all)),
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            n_jobs=-1,
            random_state=42
        ))
    ]
    stack_allowed = StackingRegressor(
        estimators=estimators_def,
        final_estimator=XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        ),
        cv=4,
        n_jobs=-1
    ).fit(X_train, y_train_all)

    # Quantile models for uncertainty
    st.write("Training quantile models for uncertainty estimation...")
    quantiles = {}
    for q in [0.1, 0.5, 0.9]:
        qm = XGBRegressor(
            objective='reg:quantileerror', quantile_alpha=q,
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1
        )
        qm.fit(X_train, y_train_for)
        quantiles[q] = qm

    # Bootstrap for uncertainty
    st.write("Bootstrap sampling for prediction intervals...")
    boot_models = []
    n_boot = 100
    for i in range(n_boot):
        if i % 20 == 0:
            st.write(f"  Bootstrap iteration {i}/{n_boot}")
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        bm = copy.deepcopy(stack_for)
        bm.fit(X_train[idx], y_train_for[idx])
        boot_models.append(bm)

    # Variance model for heteroskedastic uncertainty
    st.write("Training variance model...")
    residuals = y_train_for - stack_for.predict(X_train)
    variance_model = XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, n_jobs=-1
    ).fit(X_train, np.abs(residuals))

    st.write("✓ All models trained successfully!")
    
    return stack_for, stack_allowed, quantiles, feat_cols, boot_models, variance_model

# ------------------------ Win prob calibration ------------------------
@st.cache_data(show_spinner=False)
def calibrate_win_model(
    schedule_df: pd.DataFrame,
    history: pd.DataFrame,
    _pts_for_m, _pts_all_m,
    _feat_cols, seasons
) -> LogisticRegression:
    """Calibrate win probability model using historical predictions"""
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
            
            # Get historical features for both teams
            h_hist = (history[(history['team']==home) & 
                             ((history['season']<season) | 
                              ((history['season']==season)&(history['week']<week)))]
                      .sort_values(['season','week']).tail(1))
            a_hist = (history[(history['team']==away) & 
                             ((history['season']<season) | 
                              ((history['season']==season)&(history['week']<week)))]
                      .sort_values(['season','week']).tail(1))
            
            if h_hist.empty or a_hist.empty:
                continue
            
            hv = _to_numeric_matrix(h_hist[_feat_cols], _feat_cols)
            av = _to_numeric_matrix(a_hist[_feat_cols], _feat_cols)
            
            # Predict using blended approach
            h_off = float(_pts_for_m.predict(hv)[0])
            a_off = float(_pts_for_m.predict(av)[0])
            h_def = float(_pts_all_m.predict(hv)[0])
            a_def = float(_pts_all_m.predict(av)[0])
            
            # Blend offense and opponent defense
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

# ------------------------ Inference ------------------------
def predict_week(matchups, history, pts_for_m, pts_all_m, quantile_models, feat_cols, 
                 boot_models, cal_model, variance_model):
    """Enhanced prediction with proper offense/defense blending"""
    out = []
    
    for _, g in matchups.iterrows():
        season, week = g['season'], g['week']
        home, away = g['home_team'], g['away_team']
        
        # Get most recent historical data for both teams
        h_hist = (history[(history['team']==home) & 
                         ((history['season']<season) | 
                          ((history['season']==season)&(history['week']<week)))]
                  .sort_values(['season','week']).tail(1))
        a_hist = (history[(history['team']==away) & 
                         ((history['season']<season) | 
                          ((history['season']==season)&(history['week']<week)))]
                  .sort_values(['season','week']).tail(1))
        
        if h_hist.empty or a_hist.empty:
            continue

        h_vec = _to_numeric_matrix(h_hist[feat_cols], feat_cols)
        a_vec = _to_numeric_matrix(a_hist[feat_cols], feat_cols)

        # CRITICAL FIX: Proper offense/defense blending
        # Predict what each team scores (offense) and allows (defense)
        h_off = float(pts_for_m.predict(h_vec)[0])   # Home offense
        a_off = float(pts_for_m.predict(a_vec)[0])   # Away offense
        h_def = float(pts_all_m.predict(h_vec)[0])   # Home defense (pts allowed)
        a_def = float(pts_all_m.predict(a_vec)[0])   # Away defense (pts allowed)
        
        # Blend: Home score = avg(home offense, away defense allows)
        # Away score = avg(away offense, home defense allows)
        h_pts = (h_off + a_def) / 2
        a_pts = (a_off + h_def) / 2
        
        margin = round(h_pts - a_pts, 1)
        total = round(h_pts + a_pts, 1)
        
        # Win probability from calibrated model
        win_prob = float(cal_model.predict_proba([[margin]])[0,1])

        # Uncertainty quantification
        # Method 1: Quantile regression
        lower_h = float(quantile_models[0.1].predict(h_vec)[0])
        upper_h = float(quantile_models[0.9].predict(h_vec)[0])
        lower_a = float(quantile_models[0.1].predict(a_vec)[0])
        upper_a = float(quantile_models[0.9].predict(a_vec)[0])
        lower = round(lower_h + lower_a, 1)
        upper = round(upper_h + upper_a, 1)
        
        # Method 2: Bootstrap distribution
        boot_totals = []
        for bm in boot_models:
            bh = float(bm.predict(h_vec)[0])
            ba = float(bm.predict(a_vec)[0])
            boot_totals.append(bh + ba)
        sigma_boot = float(np.std(boot_totals) if boot_totals else 7.0)
        
        # Method 3: Variance model
        h_var = float(variance_model.predict(h_vec)[0])
        a_var = float(variance_model.predict(a_vec)[0])
        sigma_var = float(np.sqrt(h_var**2 + a_var**2))
        
        # Combined sigma (average of methods)
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
    """Simulate season outcomes with enhanced prediction logic"""
    games = schedule_df[schedule_df['season']==season]
    probs = []
    
    for week in sorted(games['week'].unique()):
        preds = predict_week(
            games[games['week']==week], history, _pts_for_m, _pts_all_m,
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

def simulate_playoffs(schedule_df, history, pts_for_m, pts_all_m, quantile_models, 
                     feat_cols, boot_models, cal_model, var_model, season, n_sims=2000):
    """Simulate playoff outcomes"""
    def game_win_prob(team1, team2):
        h_hist = history[history["team"] == team1].sort_values(["season", "week"]).tail(1)
        a_hist = history[history["team"] == team2].sort_values(["season", "week"]).tail(1)
        if h_hist.empty or a_hist.empty:
            return 0.5
        hv = _to_numeric_matrix(h_hist[feat_cols], feat_cols)
        av = _to_numeric_matrix(a_hist[feat_cols], feat_cols)
        
        h_off = float(pts_for_m.predict(hv)[0])
        a_off = float(pts_for_m.predict(av)[0])
        h_def = float(pts_all_m.predict(hv)[0])
        a_def = float(pts_all_m.predict(av)[0])
        
        h_pts = (h_off + a_def) / 2
        a_pts = (a_off + h_def) / 2
        margin = h_pts - a_pts
        
        return float(cal_model.predict_proba([[margin]])[0, 1])

    full_df = simulate_win_totals(
        schedule_df, history, pts_for_m, pts_all_m, quantile_models,
        feat_cols, boot_models, cal_model, var_model, season, n_sims
    )
    
    teams = list(full_df["Team"].unique())
    division_teams = defaultdict(list)
    for t in teams:
        division_teams[DIVISION_MAP[t]].append(t)
    
    conf_teams = {
        "AFC": [t for t in teams if DIVISION_MAP[t].startswith("AFC")],
        "NFC": [t for t in teams if DIVISION_MAP[t].startswith("NFC")]
    }
    
    results = {team: defaultdict(int) for team in teams}

    for _ in range(n_sims):
        win_sample = {
            team: round(np.random.normal(
                full_df[full_df["Team"] == team]["Expected_Wins"].values[0], 1.5
            ), 1)
            for team in teams
        }
        
        div_winners = {}
        for div, tlist in division_teams.items():
            best = max(tlist, key=lambda t: win_sample[t])
            div_winners[div] = best

        conf_seeds = {}
        for conf, tlist in conf_teams.items():
            winners = [t for t in tlist if t in div_winners.values()]
            wildcards = [t for t in tlist if t not in winners]
            sorted_winners = sorted(winners, key=lambda t: -win_sample[t])
            sorted_wild = sorted(wildcards, key=lambda t: -win_sample[t])[:3]
            conf_seeds[conf] = sorted_winners + sorted_wild

        conf_finalists = {}
        for conf in ["AFC", "NFC"]:
            seeds = conf_seeds[conf]
            wc = seeds[1:]
            pairs = list(zip(wc[::2], wc[1::2]))
            winners = []
            for a, b in pairs:
                winners.append(a if game_win_prob(a, b) > 0.5 else b)
            
            winners.append(seeds[0])
            semi_pairs = list(zip(winners[::2], winners[1::2]))
            winners2 = []
            for a, b in semi_pairs:
                winners2.append(a if game_win_prob(a, b) > 0.5 else b)
                results[winners2[-1]]["round_2"] += 1
            
            champ = winners2[0]
            conf_finalists[conf] = champ
            results[champ]["sb_appear"] += 1

        afc_team = conf_finalists["AFC"]
        nfc_team = conf_finalists["NFC"]
        if game_win_prob(afc_team, nfc_team) > 0.5:
            results[afc_team]["sb_win"] += 1
        else:
            results[nfc_team]["sb_win"] += 1

    final = []
    for team, r in results.items():
        final.append({
            "Team": team,
            "Conf Champ %": round(r["round_2"] / n_sims * 100, 1),
            "SB Appear %": round(r["sb_appear"] / n_sims * 100, 1),
            "SB Win %": round(r["sb_win"] / n_sims * 100, 1)
        })
    return pd.DataFrame(final)

# ------------------------ Streamlit UI ------------------------
st.sidebar.header("⚙️ Settings")

if 'run_models' not in st.session_state:
    st.session_state['run_models'] = False
if 'trained' not in st.session_state:
    st.session_state['trained'] = False

years = st.sidebar.multiselect("Training Seasons", SEASONS, default=[SEASONS[-1]])
if not years:
    st.stop()

st.sidebar.subheader("📊 Bookmaker Odds")
odds_upload = st.sidebar.file_uploader(
    "Upload Odds CSV", type=["csv"],
    help=("Upload a CSV with columns: "
          "Matchup,Bookmaker Line,Over Price,Under Price,Home ML,Away ML "
          "or raw feed with 'market','point','price','home_team','away_team','label'.")
)

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
    else:
        template_cols = {"matchup", "bookmaker line", "over price", "under price", "home ml", "away ml"}
        if template_cols.issubset(df_ou.columns):
            for _, r in df_ou.iterrows():
                odds_data[r["matchup"]] = {
                    "book_line": float(r["bookmaker line"]),
                    "over_price": float(r["over price"]),
                    "under_price": float(r["under price"]),
                    "home_ml": float(r["home ml"]),
                    "away_ml": float(r["away ml"]),
                    "spread_line": float(r.get("spread line", 0)),
                    "spread_price": float(r.get("spread price", 0))
                }
        else:
            st.sidebar.error("Unrecognized odds schema. Please check CSV format.")

# Train/Update
if st.sidebar.button("🚀 Train / Update Models"):
    st.session_state['run_models'] = True
    st.session_state['trained'] = False

if st.session_state['run_models']:
    if not st.session_state['trained']:
        with st.spinner("🔄 Loading data and training enhanced models..."):
            wk = load_weekly(years)
            sched = load_schedules(years)
            ngs = load_ngs(years)
            snaps = load_snap_counts(years)
            pbp = load_pbp_data(years)

            team_week = build_team_week_df(wk, sched, ngs, snaps, pbp)
            team_week_feats = add_rolling_features(team_week)

            pts_for_m, pts_all_m, quantile_models, feat_cols, boot_models, variance_model = train_models(team_week_feats)
            cal_model = calibrate_win_model(sched, team_week_feats, pts_for_m, pts_all_m, feat_cols, years)

            st.session_state.update({
                'wk': wk, 'sched': sched, 'team_week_feats': team_week_feats,
                'pts_for_m': pts_for_m, 'pts_all_m': pts_all_m,
                'quantile_models': quantile_models, 'feat_cols': feat_cols,
                'boot_models': boot_models, 'variance_model': variance_model,
                'cal_model': cal_model, 'trained': True
            })
            st.success("✅ Models trained successfully!")

    # Retrieve from session state
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
        st.info("👉 Click '🚀 Train / Update Models' to load data and train models.")
        st.stop()
else:
    st.info("👉 Click '🚀 Train / Update Models' to load data and train models.")
    st.stop()

# --- Prepare predictions ---
future_sched = sched[(sched['home_score'].isna()) & (sched['season'].isin(years))]
if future_sched.empty:
    st.warning("⚠️ No future games found for selected seasons.")
    st.stop()

sel_season = st.sidebar.selectbox("Season to predict", sorted(future_sched['season'].unique(), reverse=True))
sel_week = st.sidebar.selectbox("Week", sorted(future_sched[future_sched['season'] == sel_season]['week'].unique()))
games = future_sched[(future_sched['season'] == sel_season) & (future_sched['week'] == sel_week)]

# Generate predictions
preds = predict_week(games, team_week_feats, pts_for_m, pts_all_m, quantile_models, 
                    feat_cols, boot_models, cal_model, variance_model)

# Round numeric columns
round_cols = ['Home_Pts', 'Away_Pts', 'Total', 'Total_10%', 'Total_90%', 'Total_sigma',
              'Home_Off', 'Away_Off', 'Home_Def', 'Away_Def']
for c in round_cols:
    if c in preds.columns:
        preds[c] = preds[c].round(1)

# Merge SOS
sos_df = pd.DataFrame(list(SOS_OWP.items()), columns=['Team', 'SOS'])
preds = preds.merge(sos_df.rename(columns={'Team': 'Home', 'SOS': 'home_SOS'}), on='Home', how='left')
preds = preds.merge(sos_df.rename(columns={'Team': 'Away', 'SOS': 'away_SOS'}), on='Away', how='left')

# Sidebar toggles
show_bands = st.sidebar.checkbox("Show confidence bands", value=True)
show_sigma = st.sidebar.checkbox("Show bootstrap σ", value=True)
show_components = st.sidebar.checkbox("Show Off/Def components", value=False)
min_win = st.sidebar.slider("Min Home Win %", 0.0, 100.0, 0.0)
preds = preds[preds['Home_Win%'].str.rstrip('%').astype(float) >= min_win]

# Merge odds and compute edges/EVs
if odds_data and not preds.empty:
    odds_df = pd.DataFrame([{"Matchup": k, **v} for k, v in odds_data.items()])
    odds_df[['Away', 'Home']] = odds_df['Matchup'].str.split(" @ ", expand=True)
    preds = preds.merge(odds_df, on=['Home', 'Away'], how='left')

    preds["home_imp_prob"] = preds["home_ml"].apply(convert_moneyline_to_prob)
    preds["away_imp_prob"] = preds["away_ml"].apply(convert_moneyline_to_prob)
    preds["ml_edge"] = preds["Home_Win%"].str.rstrip('%').astype(float) / 100 - preds["home_imp_prob"]
    preds["spread_edge"] = preds["Pred_Margin"] - preds.get("spread_line", 0)
    preds["total_edge"] = preds["Total"] - preds.get("book_line", 0)

    # Probability calculations using normal distribution
    preds["Total_sigma"] = preds["Total_sigma"].replace(0, 0.1)
    preds["spread_prob"] = norm.sf(
        preds.get("spread_line", 0) - preds["Pred_Margin"], 
        loc=0, 
        scale=preds["Total_sigma"]
    )
    preds["spread_ev"] = np.where(
        preds.get("spread_price", 0) < 0,
        preds["spread_prob"] * (100 / preds.get("spread_price", 0).abs()) - (1 - preds["spread_prob"]),
        preds["spread_prob"] * (preds.get("spread_price", 0).abs() / 100) - (1 - preds["spread_prob"])
    )
    preds["total_prob"] = norm.sf(
        preds.get("book_line", 0) - preds["Total"], 
        loc=0, 
        scale=preds["Total_sigma"]
    )
    preds["total_ev"] = np.where(
        preds.get("over_price", 0) < 0,
        preds["total_prob"] * (100 / preds.get("over_price", 0).abs()) - (1 - preds["total_prob"]),
        preds["total_prob"] * (preds.get("over_price", 0).abs() / 100) - (1 - preds["total_prob"])
    )

# ------------------------ Display KPIs ------------------------
st.header(f"📊 Week {sel_week} Predictions - Season {sel_season}")

games_count = len(preds)
avg_win = (preds['Home_Win%'].str.rstrip('%').astype(float) / 100).mean() if games_count else 0
avg_total = preds['Total'].mean() if games_count else 0
avg_sigma = preds['Total_sigma'].mean() if 'Total_sigma' in preds.columns and games_count else 0
avg_home_sos = preds['home_SOS'].mean() if 'home_SOS' in preds.columns else 0
avg_away_sos = preds['away_SOS'].mean() if 'away_SOS' in preds.columns else 0
retrain_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Games", games_count)
c2.metric("Avg Win %", f"{avg_win:.1%}")
c3.metric("Avg Total", f"{avg_total:.1f}")
c4.metric("Avg σ", f"{avg_sigma:.1f}")
c5.metric("Last Retrain", retrain_time)
c6.metric("Avg SOS (H/A)", f"{avg_home_sos:.3f}/{avg_away_sos:.3f}")

# ------------------------ Visualizations ------------------------
if not preds.empty:
    st.subheader("📈 Win% Distribution")
    win_pct_data = preds[['Home', 'Home_Win%']].copy()
    win_pct_data['Win%'] = win_pct_data['Home_Win%'].str.rstrip('%').astype(float) / 100
    win_pct_data = win_pct_data.set_index('Home')['Win%']
    st.bar_chart(win_pct_data)

    if show_bands and {'Total', 'Total_10%', 'Total_90%'}.issubset(preds.columns):
        st.subheader("📉 Predicted Total with Confidence Bands")
        total_data = preds[['Home', 'Total', 'Total_10%', 'Total_90%']].set_index('Home')
        st.line_chart(total_data)

    if show_sigma and 'Total_sigma' in preds.columns:
        st.subheader("🎯 Prediction Uncertainty (σ) by Game")
        sigma_data = preds[['Home', 'Total_sigma']].set_index('Home')['Total_sigma']
        st.bar_chart(sigma_data)

# ------------------------ Recommended Bets ------------------------
st.subheader("💰 Recommended Bets")

if "ml_edge" in preds.columns:
    ml_bets = preds[preds["ml_edge"] >= ML_EDGE_THRESHOLD].copy()
    st.markdown("**Moneyline Bets (edge ≥ 5%)**")
    if not ml_bets.empty:
        ml_display = ml_bets[["Home", "Away", "Home_Win%", "home_ml", "ml_edge"]]
        ml_display['ml_edge'] = (ml_display['ml_edge'] * 100).round(1).astype(str) + '%'
        st.dataframe(ml_display, use_container_width=True)
    else:
        st.write("No moneyline bets above threshold.")

if "spread_edge" in preds.columns:
    spread_bets = preds[
        (preds["spread_edge"].abs() >= SPREAD_EDGE_THRESHOLD) &
        (preds.get("spread_line", 0).round().isin(SPREAD_KEY_NUMBERS) |
         preds["Pred_Margin"].round().isin(SPREAD_KEY_NUMBERS))
    ].copy()
    st.markdown("**Spread Bets (edge ≥ 3 points, on key numbers)**")
    if not spread_bets.empty:
        spread_display = spread_bets[[
            "Home", "Away", "Pred_Margin", "spread_line", "spread_edge", 
            "spread_prob", "spread_price", "spread_ev"
        ]]
        spread_display['spread_prob'] = (spread_display['spread_prob'] * 100).round(1).astype(str) + '%'
        spread_display['spread_ev'] = (spread_display['spread_ev'] * 100).round(1).astype(str) + '%'
        st.dataframe(spread_display, use_container_width=True)
    else:
        st.write("No spread bets above threshold and on key numbers.")

if "total_edge" in preds.columns:
    total_bets = preds[
        (preds["total_edge"].abs() >= TOTAL_EDGE_THRESHOLD) &
        (preds.get("book_line", 0).round().isin(TOTAL_KEY_NUMBERS) |
         preds["Total"].round().isin(TOTAL_KEY_NUMBERS))
    ].copy()
    st.markdown("**Total Bets (edge ≥ 4 points, on key numbers)**")
    if not total_bets.empty:
        total_display = total_bets[[
            "Home", "Away", "Total", "book_line", "total_edge", 
            "total_prob", "over_price", "total_ev"
        ]]
        total_display['total_prob'] = (total_display['total_prob'] * 100).round(1).astype(str) + '%'
        total_display['total_ev'] = (total_display['total_ev'] * 100).round(1).astype(str) + '%'
        st.dataframe(total_display, use_container_width=True)
    else:
        st.write("No total bets above threshold and on key numbers.")

# ------------------------ Predictions Table ------------------------
st.subheader("📋 All Predictions")
display_cols = ['Season', 'Week', 'Home', 'Away', 'Home_Pts', 'Away_Pts', 'Total', 
                'Total_10%', 'Total_90%', 'Home_Win%', 'Pred_Margin', 'Total_sigma']
if show_components:
    display_cols.extend(['Home_Off', 'Away_Off', 'Home_Def', 'Away_Def'])
st.dataframe(preds[display_cols], use_container_width=True)

# ------------------------ Game Details ------------------------
st.subheader("🎮 Game Details")
for _, row in preds.iterrows():
    with st.expander(f"{row.Away} @ {row.Home} — Win% {row['Home_Win%']}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{row.Home} (Home)**")
            st.metric("Predicted Score", f"{row.Home_Pts:.1f}")
            if show_components:
                st.caption(f"Offense: {row.Home_Off:.1f} | Defense: {row.Home_Def:.1f}")
            st.caption(f"SOS: {row.home_SOS:.3f}")
        
        with col2:
            st.markdown(f"**{row.Away} (Away)**")
            st.metric("Predicted Score", f"{row.Away_Pts:.1f}")
            if show_components:
                st.caption(f"Offense: {row.Away_Off:.1f} | Defense: {row.Away_Def:.1f}")
            st.caption(f"SOS: {row.away_SOS:.3f}")
        
        st.markdown("---")
        st.markdown(f"**Game Total:** {row.Total:.1f}")
        if show_bands and 'Total_10%' in row and 'Total_90%' in row:
            st.markdown(f"• Expected range (80% CI): {row['Total_10%']:.1f} to {row['Total_90%']:.1f}")
        if show_sigma and 'Total_sigma' in row:
            st.markdown(f"• Prediction uncertainty (±1σ): {row.Total_sigma:.1f}")
        
        # Historical trend
        hist = team_week_feats[team_week_feats['team'] == row.Home].sort_values(['season', 'week']).tail(6)
        if not hist.empty:
            show_cols = [c for c in ['pf_r3', 'pa_r3', 'pf_ewm3', 'pa_ewm3'] if c in hist.columns]
            if show_cols:
                st.markdown("**Recent Performance Trend**")
                st.line_chart(hist.set_index('week')[show_cols])

# ------------------------ Futures ------------------------
st.subheader("🔮 Futures Projections")
if 'futures_df' not in st.session_state:
    st.session_state['futures_df'] = None

if st.button("🔄 Simulate Season Win Totals"):
    with st.spinner("Running 20,000 simulations..."):
        st.session_state['futures_df'] = simulate_win_totals(
            sched, team_week_feats, pts_for_m, pts_all_m,
            quantile_models, feat_cols, boot_models, cal_model, 
            variance_model, sel_season
        )

if st.session_state['futures_df'] is not None:
    st.write("**Season Win Total Projections**")
    futures_display = st.session_state['futures_df'].copy()
    futures_display['Div_Win_Pct'] = (futures_display['Div_Win_Pct'] * 100).round(1).astype(str) + '%'
    st.dataframe(futures_display, use_container_width=True)
    
    st.markdown("Upload a CSV with columns `Team,Vegas_OU` to compute edges.")
    vuploader = st.file_uploader("Win Totals CSV", type=["csv"])
    if vuploader:
        vs = pd.read_csv(vuploader)
        merged = st.session_state['futures_df'].merge(vs, on='Team', how='left')
        merged['Edge'] = merged['Expected_Wins'] - merged['Vegas_OU']
        st.write("**Projected Wins vs Vegas O/U**")
        edge_display = merged[['Team', 'Expected_Wins', 'Vegas_OU', 'Edge']].copy()
        edge_display['Expected_Wins'] = edge_display['Expected_Wins'].round(1)
        edge_display['Edge'] = edge_display['Edge'].round(1)
        st.dataframe(edge_display, use_container_width=True)

# ------------------------ Playoffs ------------------------
if st.button("🏆 Simulate Playoff Outcomes"):
    with st.spinner("Simulating playoff brackets..."):
        playoff_df = simulate_playoffs(
            sched, team_week_feats, pts_for_m, pts_all_m,
            quantile_models, feat_cols, boot_models, cal_model, 
            variance_model, sel_season
        )
        st.write("**Playoff Simulation Results (2,000 simulations)**")
        st.dataframe(playoff_df.sort_values("SB Win %", ascending=False), use_container_width=True)
        
        top_sb_team = playoff_df.sort_values("SB Win %", ascending=False).iloc[0]
        st.metric("🏆 Most Likely Super Bowl Winner", 
                 f"{top_sb_team['Team']} ({top_sb_team['SB Win %']}%)")

# ------------------------ Export ------------------------
csv = preds.to_csv(index=False).encode()
st.download_button("📥 Download Predictions CSV", csv, "predictions_enhanced.csv", "text/csv")

# ------------------------ Validation ------------------------
if st.sidebar.checkbox("🔬 Hold-Out Validation"):
    st.sidebar.write("Running hold-out test for first 2 weeks of current season")
    
    years_all = list(range(2018, datetime.now().year + 1))
    tw = load_weekly(years_all)
    sch2 = load_schedules(years_all)
    ng2 = load_ngs(years_all)
    sn2 = load_snap_counts(years_all)
    pb2 = load_pbp_data(years_all)
    df = add_rolling_features(build_team_week_df(tw, sch2, ng2, sn2, pb2))

    NEXT = datetime.now().year
    train = df[df['season'] < NEXT]
    hold = df[(df['season'] == NEXT) & (df['week'] <= 2)]
    
    if hold.empty:
        st.write(f"No data for season {NEXT} weeks 1–2. Skipping validation.")
    else:
        X_tr = _to_numeric_matrix(train[feat_cols], feat_cols)
        y_tr_for = train['points_for'].to_numpy()
        y_tr_all = train['points_allowed'].to_numpy()
        X_te = _to_numeric_matrix(hold[feat_cols], feat_cols)
        y_te_for = hold['points_for'].to_numpy()
        y_te_all = hold['points_allowed'].to_numpy()
        
        m_for = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ).fit(X_tr, y_tr_for)
        
        m_all = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ).fit(X_tr, y_tr_all)
        
        mae_for = mean_absolute_error(y_te_for, m_for.predict(X_te))
        mae_all = mean_absolute_error(y_te_all, m_all.predict(X_te))
        
        st.write(f"**Hold-out MAE offense:** {mae_for:.2f}")
        st.write(f"**Hold-out MAE defense:** {mae_all:.2f}")
        
        res_for = y_te_for - m_for.predict(X_te)
        res_all = y_te_all - m_all.predict(X_te)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(res_for, bins=20, edgecolor='black', alpha=0.7)
        ax1.set_title('Offense Residuals (points_for)')
        ax1.set_xlabel('Actual − Predicted')
        ax1.axvline(0, color='red', linestyle='--', linewidth=1)
        
        ax2.hist(res_all, bins=20, edgecolor='black', alpha=0.7)
        ax2.set_title('Defense Residuals (points_allowed)')
        ax2.set_xlabel('Actual − Predicted')
        ax2.axvline(0, color='red', linestyle='--', linewidth=1)
        
        st.pyplot(fig)

st.caption("v0.3 Enhanced — Improved offense/defense modeling with advanced features. Validate before staking.")