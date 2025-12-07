import copy
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import random
from collections import defaultdict

from sklearn.linear_model import LogisticRegression

# --- Odds helper and thresholds ---
def convert_moneyline_to_prob(ml):
    try:
        return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
    except:
        return 0.5

ML_EDGE_THRESHOLD = 0.05         # 5% moneyline edge
SPREAD_EDGE_THRESHOLD = 3.0      # 3-point spread edge
TOTAL_EDGE_THRESHOLD = 4.0       # 4-point total edge

# NFL key numbers for point spreads (final-score margins)
SPREAD_KEY_NUMBERS = [
    3, 7, 6, 10, 4, 14, 1, 2, 17, 5, 8, 13, 11, 21, 20, 18, 24, 16, 9, 12,
    15, 28, 27, 19, 23, 31, 25, 22, 26, 35, 34, 30, 38, 29, 32, 37, 33, 0,
    41, 40, 36, 45, 42, 39, 43, 44, 49, 46, 48, 52, 58, 55, 59, 54, 51
]

# NFL key numbers for totals (combined scores)
TOTAL_KEY_NUMBERS = [
    41, 46, 37, 42, 44, 55, 51, 45, 43, 40, 47, 48, 33, 39, 38, 30, 36, 34,
    27, 25, 23, 17, 50, 31, 54, 29, 26, 35, 32, 24, 22, 20, 21, 19, 16, 15,
    18, 14, 12, 13, 11, 9, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0
]

# --- 2025 Strength of Schedule (Opp. Win Pct) ---
SOS_OWP = {
    'NYG': 0.574, 'DET': 0.571, 'CHI': 0.571, 'PHI': 0.561, 'MIN': 0.557, 'DAL': 0.557,
    'GB' : 0.557, 'WAS': 0.550, 'BAL': 0.533, 'PIT': 0.526, 'KC' : 0.522, 'LAC': 0.522,
    'CLE': 0.519, 'CIN': 0.509, 'DEN': 0.505, 'LV' : 0.502, 'LAR': 0.491, 'TB' : 0.481,
    'HOU': 0.481, 'ATL': 0.478, 'MIA': 0.474, 'SEA': 0.474, 'BUF': 0.467, 'JAX': 0.467,
    'IND': 0.464, 'NYJ': 0.460, 'CAR': 0.457, 'ARI': 0.457, 'TEN': 0.450, 'NE' : 0.429,
    'NO' : 0.419, 'SF' : 0.415
}

# --- Division Mapping for NFL teams ---
DIVISION_MAP = {
    'NE': 'AFC East', 'BUF': 'AFC East', 'MIA': 'AFC East', 'NYJ': 'AFC East',
    'KC': 'AFC West', 'DEN': 'AFC West', 'LAC': 'AFC West', 'LV': 'AFC West',
    'TEN': 'AFC South', 'HOU': 'AFC South', 'IND': 'AFC South', 'JAX': 'AFC South',
    'BAL': 'AFC North', 'PIT': 'AFC North', 'CIN': 'AFC North', 'CLE': 'AFC North',
    'DAL': 'NFC East', 'PHI': 'NFC East', 'WAS': 'NFC East', 'NYG': 'NFC East',
    'GB': 'NFC North', 'MIN': 'NFC North', 'CHI': 'NFC North', 'DET': 'NFC North',
    'SF': 'NFC West', 'SEA': 'NFC West', 'LAR': 'NFC West',
    'LA': 'NFC West',  # alias for Los Angeles Rams
    'ARI': 'NFC West',
    'TB': 'NFC South', 'NO': 'NFC South', 'CAR': 'NFC South', 'ATL': 'NFC South'
}

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, brier_score_loss
from catboost import CatBoostRegressor

# Must be the first Streamlit command
st.set_page_config(page_title="FoxEdge NFL Predictor", layout="wide", page_icon="🏈")

"""
FoxEdge – NFL Matchup Predictor (v0.2)

Fixes vs v0.1
-------------
1. **Weekly data lacked a `team` column** → rename `recent_team` ⇒ `team` on load.
2. Added **team-score targets** by fusing `import_schedules` (home/away scores) – no more fantasy-points confusion.
3. Refactored feature-engineering to build team-week aggregates, then roll 1-/3-/6-week windows.
4. Defensive features now derived by matching opponent’s offensive aggregates – no need for missing `opponent` column.

NOTE:  Pulls 2018-present by default.  Expect first run ≈ 30–45 s.
"""

############################
# CONFIG
############################
ROLL_WINDOWS = [1, 3, 6]
# include next calendar year so new schedules (e.g., 2025) show up
SEASONS = list(range(2018, datetime.now().year + 1))

############################
# DATA LOADERS
############################
@st.cache_data(show_spinner=False)
def load_weekly(years):
    """Pull weekly player data only for seasons that actually exist in nflverse.

    nfl.import_weekly_data() 404s if you request a year whose parquet
    hasn't been published yet (e.g., the upcoming season).  This wrapper
    skips those seasons and warns the user instead of bombing the app.
    """
    collected = []
    for yr in years:
        try:
            df_yr = nfl.import_weekly_data([yr])
        except Exception as e:
            st.warning(f"⚠️  Weekly parquet for {yr} not found – skipping. ({e})")
            continue

        # Fix column name change
        if 'team' not in df_yr.columns and 'recent_team' in df_yr.columns:
            df_yr = df_yr.rename(columns={'recent_team': 'team'})
        collected.append(df_yr)

    if not collected:
        st.error("No weekly data loaded. Deselect unpublished seasons.")
        st.stop()

    return pd.concat(collected, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_schedules(years):
    return nfl.import_schedules(years)

@st.cache_data(show_spinner=False)
def load_ngs(years):
    """Pull NGS data for receiving and rushing; expanded aggregation per team-week."""
    groups = ['receiving', 'rushing']  # Skip passing as it lacks YAC-OE equivalent
    aggs = []
    for grp in groups:
        try:
            rec = nfl.import_ngs_data(grp, years)
            print(f"NGS {grp} columns: {rec.columns.tolist()}")  # Debug line
            # Normalize team column
            if 'team' not in rec.columns and 'recent_team' in rec.columns:
                rec = rec.rename(columns={'recent_team': 'team'})
            elif 'team_abbr' in rec.columns:
                rec = rec.rename(columns={'team_abbr': 'team'})
            else:
                st.warning(f"NGS group {grp} missing team identifier; skipping.")
                continue
            if grp == 'receiving':
                # Identify columns
                yac_cols = [c for c in rec.columns if 'yac' in c.lower() and ('oe' in c.lower() or 'expected' in c.lower())]
                share_col = 'percent_share_of_intended_air_yards'
                yac_above_col = 'avg_yac_above_expectation'
                # Additional columns for expanded features
                intended_air_yards_col = 'avg_intended_air_yards'
                targets_col = 'targets'
                receptions_col = 'receptions'
                catch_pct_col = 'catch_percentage'
                avg_sep_col = 'avg_separation'
                avg_cushion_col = 'avg_cushion'
                if (not yac_cols or
                    share_col not in rec.columns or
                    yac_above_col not in rec.columns or
                    intended_air_yards_col not in rec.columns or
                    targets_col not in rec.columns or
                    receptions_col not in rec.columns or
                    catch_pct_col not in rec.columns or
                    avg_sep_col not in rec.columns or
                    avg_cushion_col not in rec.columns):
                    st.warning(f"NGS receiving missing required columns; skipping.")
                    continue
                yac_col = yac_cols[0]
                rec = rec.rename(columns={
                    yac_col: 'receiving_yac_oe',
                    share_col: 'receiving_air_share',
                    yac_above_col: 'receiving_yac_above',
                    intended_air_yards_col: 'receiving_intended_air_yards',
                    targets_col: 'receiving_targets',
                    receptions_col: 'receiving_receptions',
                    catch_pct_col: 'receiving_catch_pct',
                    avg_sep_col: 'receiving_avg_sep',
                    avg_cushion_col: 'receiving_avg_cushion'
                })
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
            elif grp == 'rushing':
                yac_cols = [c for c in rec.columns if 'rush_yards_over_expected' in c.lower()]
                eff_col = 'efficiency'
                ttl_col = 'avg_time_to_los'
                ye_pa_col = 'rush_yards_over_expected_per_att'
                # Additional columns for expanded features
                rush_attempts_col = 'rush_attempts'
                expected_rush_yards_col = 'expected_rush_yards'
                pct_over_exp_col = 'rush_pct_over_expected'
                pct_8_box_col = 'percent_attempts_gte_eight_defenders'
                if (not yac_cols or
                    eff_col not in rec.columns or
                    ttl_col not in rec.columns or
                    ye_pa_col not in rec.columns or
                    rush_attempts_col not in rec.columns or
                    expected_rush_yards_col not in rec.columns or
                    pct_over_exp_col not in rec.columns or
                    pct_8_box_col not in rec.columns):
                    st.warning(f"NGS rushing missing required columns; skipping.")
                    continue
                yac_col = yac_cols[0]
                rec = rec.rename(columns={
                    yac_col: 'rushing_yac_oe',
                    eff_col: 'rush_efficiency',
                    ttl_col: 'rush_ttl',
                    ye_pa_col: 'rush_ye_per_att',
                    rush_attempts_col: 'rushing_attempts',
                    expected_rush_yards_col: 'rushing_exp_yards',
                    pct_over_exp_col: 'rushing_pct_over_exp',
                    pct_8_box_col: 'rush_vs_box_pct'
                })
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
            else:
                continue
        except Exception as e:
            st.warning(f"NGS fetch for {grp} failed – skipping ({e})")
            continue
        aggs.append(agg)
    # Merge on ['season','week','team']
    merged = None
    for agg in aggs:
        merged = agg if merged is None else merged.merge(agg, on=['season','week','team'], how='outer')
    return merged if merged is not None else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_snap_counts(years):
    """Pull snap counts and compute starter fraction per team-week for available years."""
    available_years = [y for y in years if y <= 2024]  # Exclude 2025
    if not available_years:
        st.warning("No valid years for snap counts; skipping.")
        return pd.DataFrame()
    try:
        snaps = nfl.import_snap_counts(available_years)
        print(f"Snap count years fetched: {snaps['season'].unique()}")  # Debug line
        print(f"Snap count columns: {snaps.columns.tolist()}")  # Debug line
        # Print out defensive snap columns for debugging
        if 'defense_snaps' in snaps.columns:
            print("defense_snaps column present")
        if 'defense_pct' in snaps.columns:
            print("defense_pct column present")
        # Combine offense and defense snaps
        if 'offense_snaps' in snaps.columns and 'defense_snaps' in snaps.columns:
            snaps['snap_counts'] = snaps['offense_snaps'].fillna(0) + snaps['defense_snaps'].fillna(0)
        else:
            st.warning("Snap count data missing offense_snaps or defense_snaps; skipping.")
            return pd.DataFrame()
        # Ensure required columns
        required_cols = {'season', 'week', 'team', 'snap_counts'}
        if not required_cols.issubset(snaps.columns):
            st.warning(f"Snap count data missing required columns: {required_cols - set(snaps.columns)}; skipping.")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Snap-count fetch failed – skipping starter_frac. ({e})")
        return pd.DataFrame()
    # Calculate fraction of snaps by top-22 players
    def frac(sub):
        total = sub['snap_counts'].sum()
        top22 = sub.nlargest(22, 'snap_counts')['snap_counts'].sum()
        return pd.Series({'starter_frac': top22 / total if total > 0 else np.nan})
    # Starter fraction
    starter = snaps.groupby(['season', 'week', 'team']).apply(frac).reset_index()
    # Defensive rotation: average defense_pct per team-week
    def_pct = snaps.groupby(['season', 'week', 'team'])['defense_pct'] \
                   .mean().reset_index().rename(columns={'defense_pct':'def_pct'})
    # Combine both
    agg = starter.merge(def_pct, on=['season', 'week', 'team'], how='left')
    print(f"Snap-count aggregated columns: {agg.columns.tolist()}")
    return agg

@st.cache_data(show_spinner=False)
def load_pbp_data(years):
    """Pull play-by-play EPA and success rate per team-week."""
    try:
        pbp = nfl.import_pbp_data(years)
        print(f"PBP raw columns: {pbp.columns.tolist()}")
    except Exception as e:
        st.warning(f"PBP fetch failed – skipping efficiency metrics. ({e})")
        return pd.DataFrame()
    pbp['success'] = pbp['epa'] > 0
    agg = (
        pbp.groupby(['season', 'week', 'posteam'])
           .agg(
               epa_pp=('epa', 'mean'),
               sr=('success', 'mean'),
               n_plays=('play_id', 'count')
           )
           .reset_index()
           .rename(columns={'posteam': 'team'})
    )
    print(f"PBP aggregated columns: {agg.columns.tolist()}")
    # Defensive play-by-play aggregates
    def_agg = (
        pbp.groupby(['season','week','defteam'])
           .agg(
              def_epa_pp=('epa','mean'),
              def_sr=('success','mean'),
              def_n_plays=('play_id','count')
           )
           .reset_index()
           .rename(columns={'defteam':'team'})
    )
    print(f"PBP defensive aggregated columns: {def_agg.columns.tolist()}")
    # Merge offense and defense aggregates
    agg = agg.merge(def_agg, on=['season','week','team'], how='left')
    print(f"PBP combined aggregated columns: {agg.columns.tolist()}")

    # --- Additional offense/defense aggregates ---
    # Offense pass vs rush EPA
    pass_agg = (
        pbp[pbp['pass'] == 1]
        .groupby(['season','week','posteam'])
        .agg(off_pass_epa_pp=('epa','mean'))
        .reset_index()
        .rename(columns={'posteam':'team'})
    )
    rush_agg = (
        pbp[pbp['rush'] == 1]
        .groupby(['season','week','posteam'])
        .agg(off_rush_epa_pp=('epa','mean'))
        .reset_index()
        .rename(columns={'posteam':'team'})
    )
    agg = agg.merge(pass_agg, on=['season','week','team'], how='left')
    agg = agg.merge(rush_agg, on=['season','week','team'], how='left')

    # Defense pass/rush EPA
    pass_def = (
        pbp[pbp['pass'] == 1]
        .groupby(['season','week','defteam'])
        .agg(def_pass_epa_pp=('epa','mean'))
        .reset_index()
        .rename(columns={'defteam':'team'})
    )
    rush_def = (
        pbp[pbp['rush'] == 1]
        .groupby(['season','week','defteam'])
        .agg(def_rush_epa_pp=('epa','mean'))
        .reset_index()
        .rename(columns={'defteam':'team'})
    )
    agg = agg.merge(pass_def, on=['season','week','team'], how='left')
    agg = agg.merge(rush_def, on=['season','week','team'], how='left')

    # Third-down conversion rate
    off_3rd = (
        pbp.groupby(['season','week','posteam'])
           .agg(off_3rd_conv=('third_down_converted','mean'))
           .reset_index()
           .rename(columns={'posteam':'team'})
    )
    def_3rd = (
        pbp.groupby(['season','week','defteam'])
           .agg(def_3rd_conv=('third_down_converted','mean'))
           .reset_index()
           .rename(columns={'defteam':'team'})
    )
    agg = agg.merge(off_3rd, on=['season','week','team'], how='left')
    agg = agg.merge(def_3rd, on=['season','week','team'], how='left')

    # Sack rate and interception rate
    off_sack = (
        pbp.groupby(['season','week','posteam'])
           .agg(off_sack_rate=('sack','mean'))
           .reset_index()
           .rename(columns={'posteam':'team'})
    )
    def_sack = (
        pbp.groupby(['season','week','defteam'])
           .agg(def_sack_rate=('sack','mean'))
           .reset_index()
           .rename(columns={'defteam':'team'})
    )
    off_int = (
        pbp.groupby(['season','week','posteam'])
           .agg(off_int_rate=('interception','mean'))
           .reset_index()
           .rename(columns={'posteam':'team'})
    )
    def_int = (
        pbp.groupby(['season','week','defteam'])
           .agg(def_int_rate=('interception','mean'))
           .reset_index()
           .rename(columns={'defteam':'team'})
    )
    agg = agg.merge(off_sack, on=['season','week','team'], how='left')
    agg = agg.merge(def_sack, on=['season','week','team'], how='left')
    agg = agg.merge(off_int, on=['season','week','team'], how='left')
    agg = agg.merge(def_int, on=['season','week','team'], how='left')

    # --- Add new aggregates ---
    # 1. Goal-to-go conversion
    g2g = (pbp[pbp['goal_to_go']==1]
           .groupby(['season','week','posteam'])
           .agg(g2g_conv=('success','mean'))
           .reset_index()
           .rename(columns={'posteam':'team'}))
    agg = agg.merge(g2g, on=['season','week','team'], how='left')

    # 2. Third-down long and short
    td_long = (pbp[(pbp['down']==3)&(pbp['ydstogo']>7)]
               .groupby(['season','week','posteam'])
               .agg(td_conv_long=('success','mean'))
               .reset_index().rename(columns={'posteam':'team'}))
    td_short = (pbp[(pbp['down']==3)&(pbp['ydstogo']<=7)]
                .groupby(['season','week','posteam'])
                .agg(td_conv_short=('success','mean'))
                .reset_index().rename(columns={'posteam':'team'}))
    agg = agg.merge(td_long, on=['season','week','team'], how='left')
    agg = agg.merge(td_short, on=['season','week','team'], how='left')

    # 3. No-huddle rate
    nh = (pbp.groupby(['season','week','posteam'])
          .agg(no_huddle_rate=('no_huddle','mean'))
          .reset_index().rename(columns={'posteam':'team'}))
    agg = agg.merge(nh, on=['season','week','team'], how='left')

    # 4. Mean wp and vegas_wpa
    wp_agg = (pbp.groupby(['season','week','posteam'])
              .agg(avg_wp=('wp','mean'), avg_vegas_wpa=('vegas_wpa','mean'))
              .reset_index().rename(columns={'posteam':'team'}))
    agg = agg.merge(wp_agg, on=['season','week','team'], how='left')

    return agg

############################
# FEATURE ENGINEERING
############################

def build_team_week_df(weekly_df: pd.DataFrame,
                       sched_df: pd.DataFrame,
                       ngs_df: pd.DataFrame,
                       snaps_df: pd.DataFrame,
                       pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-weeks → team-week then attach scores from schedules."""
    key_off_cols = [
        'passing_yards', 'rushing_yards', 'passing_tds', 'rushing_tds',
        'interceptions', 'sacks', 'fumbles', 'targets', 'receptions',
        'yards_after_catch', 'passing_air_yards', 'cpoe', 'epa'
    ]
    existing_cols = [c for c in key_off_cols if c in weekly_df.columns]

    # Sum stats over players → team-week snapshot
    team_week = (
        weekly_df.groupby(['season', 'week', 'team'])[existing_cols]
                 .sum()
                 .reset_index()
    )

    # Expand schedule into team perspective rows
    base_cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']
    sched_df = sched_df[base_cols].dropna(subset=['home_score'])  # filter out future games

    home_rows = sched_df.rename(columns={
        'home_team': 'team',
        'away_team': 'opp',
        'home_score': 'points_for',
        'away_score': 'points_allowed'
    }).assign(is_home=1)
    away_rows = sched_df.rename(columns={
        'away_team': 'team',
        'home_team': 'opp',
        'away_score': 'points_for',
        'home_score': 'points_allowed'
    }).assign(is_home=0)
    scores = pd.concat([
        home_rows[['season', 'week', 'team', 'opp', 'points_for', 'points_allowed', 'is_home']],
        away_rows[['season', 'week', 'team', 'opp', 'points_for', 'points_allowed', 'is_home']]
    ])

    team_week = team_week.merge(scores, on=['season', 'week', 'team'], how='left')

    # Home-field indicator was merged in as 'is_home'
    # Seasonality: normalized week-of-season
    max_weeks = sched_df.groupby('season')['week'].max().to_dict()
    team_week['wk_norm'] = team_week['week'] / team_week['season'].map(max_weeks)

    # Rest days – difference in calendar days since last game
    if 'date' in sched_df.columns:
        game_dates = (sched_df[['season', 'week', 'home_team', 'game_date']]
                      .rename(columns={'home_team': 'team'}))
        team_week = team_week.merge(game_dates, on=['season', 'week', 'team'])
        team_week = team_week.sort_values(['team', 'game_date'])
        team_week['rest_days'] = (team_week.groupby('team')['game_date']
                                               .diff()
                                               .dt.days
                                               .fillna(7))
        team_week.drop(columns='game_date', inplace=True)
    else:
        team_week['rest_days'] = 7  # fallback

    # --- Merge NGS explosive metric ---
    if not ngs_df.empty:
        team_week = team_week.merge(
            ngs_df, on=['season', 'week', 'team'], how='left'
        )

    # --- Merge snap counts starter_frac, if available ---
    if not snaps_df.empty and {'season','week','team','starter_frac'}.issubset(snaps_df.columns):
        team_week = team_week.merge(snaps_df, on=['season','week','team'], how='left')
    else:
        st.warning("No valid snap-count data; skipping starter_frac merge.")

    # --- Merge play-by-play efficiency, if available ---
    pbp_cols_needed = [
        'season','week','team','epa_pp','sr','n_plays',
        'def_epa_pp','def_sr','def_n_plays',
        'off_pass_epa_pp','off_rush_epa_pp','def_pass_epa_pp','def_rush_epa_pp',
        'off_3rd_conv','def_3rd_conv',
        'off_sack_rate','def_sack_rate','off_int_rate','def_int_rate'
    ]
    if not pbp_df.empty and set(pbp_cols_needed).issubset(pbp_df.columns):
        team_week = team_week.merge(pbp_df, on=['season','week','team'], how='left')
    else:
        st.warning("No valid PBP data; skipping efficiency merge.")

    return team_week.dropna(subset=['points_for'])


def add_rolling_features(team_week_df: pd.DataFrame) -> pd.DataFrame:
    """Attach rolling PF/PA means without nuking the 'team' column."""
    df = team_week_df.copy()

    # Build rolling features via groupby-transform, EXCLUDING current week (shift before rolling)
    for w in ROLL_WINDOWS:
        # Exclude current game: shift before rolling
        sorted_df = df.sort_values(['team', 'season', 'week'])
        df[f'pf_r{w}'] = (
            sorted_df
              .groupby('team')['points_for']
              .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
              .reindex(df.index)
        )
        df[f'pa_r{w}'] = (
            sorted_df
              .groupby('team')['points_allowed']
              .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
              .reindex(df.index)
        )
        if 'def_epa_pp' in df.columns:
            df[f'def_epa_r{w}'] = (
                sorted_df
                  .groupby('team')['def_epa_pp']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        if 'def_sr' in df.columns:
            df[f'def_sr_r{w}'] = (
                sorted_df
                  .groupby('team')['def_sr']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        # --- Rolling for new features ---
        # Offense pass/rush EPA
        if 'off_pass_epa_pp' in df.columns:
            df[f'off_pass_epa_r{w}'] = (
                sorted_df
                  .groupby('team')['off_pass_epa_pp']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        if 'off_rush_epa_pp' in df.columns:
            df[f'off_rush_epa_r{w}'] = (
                sorted_df
                  .groupby('team')['off_rush_epa_pp']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        if 'def_pass_epa_pp' in df.columns:
            df[f'def_pass_epa_r{w}'] = (
                sorted_df
                  .groupby('team')['def_pass_epa_pp']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        if 'def_rush_epa_pp' in df.columns:
            df[f'def_rush_epa_r{w}'] = (
                sorted_df
                  .groupby('team')['def_rush_epa_pp']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        # Third down conversion
        if 'off_3rd_conv' in df.columns:
            df[f'off_3rd_conv_r{w}'] = (
                sorted_df
                  .groupby('team')['off_3rd_conv']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        if 'def_3rd_conv' in df.columns:
            df[f'def_3rd_conv_r{w}'] = (
                sorted_df
                  .groupby('team')['def_3rd_conv']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        # Sack and interception rate
        if 'off_sack_rate' in df.columns:
            df[f'off_sack_rate_r{w}'] = (
                sorted_df
                  .groupby('team')['off_sack_rate']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        if 'def_sack_rate' in df.columns:
            df[f'def_sack_rate_r{w}'] = (
                sorted_df
                  .groupby('team')['def_sack_rate']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        if 'off_int_rate' in df.columns:
            df[f'off_int_rate_r{w}'] = (
                sorted_df
                  .groupby('team')['off_int_rate']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )
        if 'def_int_rate' in df.columns:
            df[f'def_int_rate_r{w}'] = (
                sorted_df
                  .groupby('team')['def_int_rate']
                  .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                  .reindex(df.index)
            )

    if 'rec_yac_oe' in df.columns:
        # Exclude current game: shift before rolling
        sorted_df = df.sort_values(['team', 'season', 'week'])
        df['rec_yac_oe_r3'] = (
            sorted_df
              .groupby('team')['rec_yac_oe']
              .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
              .reindex(df.index)
        )

    # --- Opponent form deltas ---
    # Prepare opponent rolling stats
    base_feat_cols = [f"pf_r{w}" for w in ROLL_WINDOWS] + [f"pa_r{w}" for w in ROLL_WINDOWS]
    if 'rec_yac_oe_r3' in df.columns:
        base_feat_cols.append('rec_yac_oe_r3')
    opp_df = df[['season', 'week', 'team'] + base_feat_cols].rename(
        columns={c: f"opp_{c}" for c in base_feat_cols}
    )
    # Merge opponent features
    df = df.merge(
        opp_df,
        left_on=['season', 'week', 'opp'],
        right_on=['season', 'week', 'team'],
        how='left',
        suffixes=('', '_opp')
    )
    df.drop(columns=['team_opp'], inplace=True)
    # Compute deltas: offense vs opponent defense, defense vs opponent offense
    df['pf_diff_r3'] = df['pf_r3'] - df['opp_pa_r3']
    df['pa_diff_r3'] = df['pa_r3'] - df['opp_pf_r3']

    return df

############################
# MODEL TRAINING
############################

def train_models(df):
    """Train offense/defense regressors, ensemble stacking, quantile models, and report CV."""
    # Define desired features
    desired_feats = [
        c for c in df.columns if c.startswith((
            'pf_r','pa_r','rec_yac_oe_r','pf_diff','pa_diff','def_epa_r','def_sr_r',
            'off_pass_epa_r','off_rush_epa_r','def_pass_epa_r','def_rush_epa_r',
            'off_3rd_conv_r','def_3rd_conv_r',
            'off_sack_rate_r','def_sack_rate_r','off_int_rate_r','def_int_rate_r'
        ))
    ] + [
        'rest_days','starter_frac','epa_pp','sr','n_plays',
        'def_pct','def_epa_pp','def_sr','def_n_plays',
        'off_pass_epa_pp','off_rush_epa_pp','def_pass_epa_pp','def_rush_epa_pp',
        'off_3rd_conv','def_3rd_conv',
        'off_sack_rate','def_sack_rate','off_int_rate','def_int_rate',
        'is_home','wk_norm'
    ] + [
        'receiving_yac_oe',
        'receiving_air_share',
        'receiving_yac_above',
        'rushing_yac_oe',
        'rush_efficiency',
        'rush_ttl',
        'rush_ye_per_att',
        'offense_pct',
        'def_pct',
        'weather',
        'surface',
        'roof',
        'temp',
        'wind'
    ] + [
        'receiving_intended_air_yards',
        'receiving_targets',
        'receiving_receptions',
        'receiving_catch_pct',
        'receiving_avg_sep',
        'receiving_avg_cushion',
        'rushing_attempts',
        'rushing_exp_yards',
        'rushing_pct_over_exp',
        'rush_vs_box_pct',
        'g2g_conv',
        'td_conv_long',
        'td_conv_short',
        'no_huddle_rate',
        'avg_wp',
        'avg_vegas_wpa'
    ]
    # Only keep features present in dataframe
    feat_cols = [c for c in desired_feats if c in df.columns]
    df_imp = df.copy().fillna(0)
    # Ensure new columns are filled with 0
    extra_cols = [
        'off_pass_epa_pp','off_rush_epa_pp','def_pass_epa_pp','def_rush_epa_pp',
        'off_3rd_conv','def_3rd_conv',
        'off_sack_rate','def_sack_rate','off_int_rate','def_int_rate'
    ]
    for col in extra_cols:
        if col in df_imp.columns:
            df_imp[col] = df_imp[col].fillna(0)
    # --- Time-based train/test split by season (last season as test) ---
    seasons = sorted(df_imp['season'].unique())
    if len(seasons) < 2:
        raise ValueError("Not enough seasons for train/test split")
    train_seasons = seasons[:-1]
    test_season = seasons[-1]
    train_df = df_imp[df_imp['season'].isin(train_seasons)]
    test_df = df_imp[df_imp['season'] == test_season]
    X_train, y_train_for, y_train_all = train_df[feat_cols], train_df['points_for'], train_df['points_allowed']
    X_test, y_test_for, y_test_all = test_df[feat_cols], test_df['points_for'], test_df['points_allowed']
    print(f"Features used in training: {feat_cols}")
    print(f"Sample y_train_for: {y_train_for[:5]}")
    print(f"Sample y_train_all: {y_train_all[:5]}")

    # Convert DataFrame inputs to numpy arrays to avoid pandas dtype issues with XGBoost
    X_train_np = X_train.values
    y_train_for_np = y_train_for.values
    y_train_all_np = y_train_all.values

    from sklearn.model_selection import GridSearchCV
    tscv_tune = TimeSeriesSplit(n_splits=5)
    # XGBoost hyperparameter grid
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    xgb_for_gs = GridSearchCV(
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42),
        xgb_param_grid, cv=tscv_tune, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    xgb_for_gs.fit(X_train_np, y_train_for_np)
    best_for = xgb_for_gs.best_estimator_
    st.write(f"Best XGB for offense: {xgb_for_gs.best_params_}, MAE: {-xgb_for_gs.best_score_:.2f}")
    # Repeat for defense
    xgb_all_gs = GridSearchCV(
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42),
        xgb_param_grid, cv=tscv_tune, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    xgb_all_gs.fit(X_train_np, y_train_all_np)
    best_all = xgb_all_gs.best_estimator_
    st.write(f"Best XGB for defense: {xgb_all_gs.best_params_}, MAE: {-xgb_all_gs.best_score_:.2f}")

    # Evaluate on hold-out test set
    from sklearn.metrics import mean_absolute_error
    test_mae_for = mean_absolute_error(y_test_for, best_for.predict(X_test.values))
    test_mae_all = mean_absolute_error(y_test_all, best_all.predict(X_test.values))
    st.write(f"Test MAE offense: {test_mae_for:.2f}, defense: {test_mae_all:.2f}")

    # Define base learners for stacking using tuned best_for/best_all
    estimators = [
        ('xgb', copy.deepcopy(best_for)),
        ('rf', RandomForestRegressor(n_estimators=150, max_depth=6)),
        ('cat', CatBoostRegressor(verbose=False, iterations=200, depth=5))
    ]
    # Offense stacking
    stack_for = StackingRegressor(estimators=estimators,
                                  final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3),
                                  cv=4, n_jobs=-1)
    stack_for.fit(X_train_np, y_train_for_np)
    # Defense stacking
    estimators_def = [
        ('xgb', copy.deepcopy(best_all)),
        ('rf', RandomForestRegressor(n_estimators=150, max_depth=6)),
        ('cat', CatBoostRegressor(verbose=False, iterations=200, depth=5))
    ]
    stack_allowed = StackingRegressor(estimators=estimators_def,
                                      final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3),
                                      cv=4, n_jobs=-1)
    stack_allowed.fit(X_train_np, y_train_all_np)

    # Quantile models for points_for (fit on train only)
    quantiles = {}
    for q in [0.1, 0.5, 0.9]:
        qm = XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, n_estimators=200, learning_rate=0.05, max_depth=4)
        qm.fit(X_train_np, y_train_for_np)
        quantiles[q] = qm

    # --- Bootstrap ensemble for points_for confidence (fit on train only) ---
    boot_models = []
    n_boot = 100
    for i in range(n_boot):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        bm = copy.deepcopy(stack_for)
        bm.fit(X_train_np[idx], y_train_for_np[idx])
        boot_models.append(bm)

    return stack_for, stack_allowed, quantiles, feat_cols, boot_models


# ---------------------- WIN PROB CALIBRATION ----------------------
@st.cache_data(show_spinner=False)
def calibrate_win_model(schedule_df: pd.DataFrame, history: pd.DataFrame,
                        _pts_for_m, _pts_all_m, _feat_cols, seasons) -> LogisticRegression:
    """
    Fit a logistic regression mapping predicted margin -> actual home win.
    Uses past seasons in `seasons`.
    """
    # --- Time-based calibration: split off last season for calibration, only use training seasons < test_season ---
    all_seasons = sorted(schedule_df['season'].unique())
    if len(all_seasons) < 2:
        return LogisticRegression().fit([[0]], [0])
    train_seasons = all_seasons[:-1]
    test_season = all_seasons[-1]
    margins = []
    outcomes = []
    # Loop through seasons and weeks (only training seasons)
    for season in train_seasons:
        # filter games with known scores
        past_games = schedule_df[(schedule_df['season'] == season) & schedule_df['home_score'].notna()]
        for _, g in past_games.iterrows():
            # get history snapshot before this game
            home, away = g['home_team'], g['away_team']
            week = g['week']
            h_hist = (history[(history['team']==home) &
                              ((history['season']<season) | ((history['season']==season)&(history['week']<week)))]
                      .sort_values(['season','week']).tail(1))
            a_hist = (history[(history['team']==away) &
                              ((history['season']<season) | ((history['season']==season)&(history['week']<week)))]
                      .sort_values(['season','week']).tail(1))
            if h_hist.empty or a_hist.empty:
                continue
            h_vec = h_hist[_feat_cols].fillna(0)
            a_vec = a_hist[_feat_cols].fillna(0)
            pred_margin = float(_pts_for_m.predict(h_vec.values)[0] - _pts_for_m.predict(a_vec.values)[0])
            actual = 1 if g['home_score'] > g['away_score'] else 0
            margins.append([pred_margin])
            outcomes.append(actual)
    if not margins:
        return LogisticRegression().fit([[0]], [0])
    lr = LogisticRegression()
    lr.fit(margins, outcomes)
    return lr


def predict_week(matchups, history, pts_for_m, pts_all_m, quantile_models, feat_cols, boot_models, cal_model):
    preds = []
    for _, g in matchups.iterrows():
        season, week = g['season'], g['week']
        home, away = g['home_team'], g['away_team']
        h_hist = (history[(history['team'] == home) &
                          ((history['season'] < season) |
                           ((history['season'] == season) & (history['week'] < week)))]
                  .sort_values(['season', 'week'])
                  .tail(1))
        a_hist = (history[(history['team'] == away) &
                          ((history['season'] < season) |
                           ((history['season'] == season) & (history['week'] < week)))]
                  .sort_values(['season', 'week'])
                  .tail(1))
        if h_hist.empty or a_hist.empty:
            continue

        h_vec = h_hist[feat_cols].fillna(0)
        a_vec = a_hist[feat_cols].fillna(0)

        h_pts = pts_for_m.predict(h_vec.values)[0]
        a_pts = pts_all_m.predict(a_vec.values)[0]
        print(f"Sample prediction - Home: {h_pts:.2f}, Away: {a_pts:.2f}")
        margin = round(h_pts - a_pts, 1)
        win_prob = cal_model.predict_proba([[margin]])[0,1]

        lower = quantile_models[0.1].predict(h_vec.values)[0] + quantile_models[0.1].predict(a_vec.values)[0]
        upper = quantile_models[0.9].predict(h_vec.values)[0] + quantile_models[0.9].predict(a_vec.values)[0]

        # Empirical sigma from bootstrap models
        boot_totals = [bm.predict(h_vec.values)[0] + bm.predict(a_vec.values)[0] for bm in boot_models]
        sigma = np.std(boot_totals)

        preds.append({
            'Season': season,
            'Week': week,
            'Home': home,
            'Away': away,
            'Home_Pts': round(h_pts, 1),
            'Away_Pts': round(a_pts, 1),
            'Total': round(h_pts + a_pts, 1),
            'Total_10%': round(lower,1),
            'Total_90%': round(upper,1),
            'Home_Win%': f"{win_prob:.1%}",
            'Pred_Margin': round(margin, 1),
            'Total_sigma': round(sigma,1),
        })
    return pd.DataFrame(preds)


@st.cache_data(show_spinner=False)
def simulate_win_totals(schedule_df, history, _pts_for_m, _pts_all_m, _quantile_models, _feat_cols, _boot_models, _cal_model, season, n_sims=20000):
    """Simulate season win totals for each team via Monte Carlo, including division win percentages."""
    # Collect per-game win probabilities for season
    games = schedule_df[schedule_df['season']==season]
    probs = []  # list of (team, win_prob) for each game
    for week in sorted(games['week'].unique()):
        wk_games = games[games['week']==week]
        preds = predict_week(wk_games, history, _pts_for_m, _pts_all_m, _quantile_models, _feat_cols, _boot_models, _cal_model)
        for _, row in preds.iterrows():
            home_prob = float(row['Home_Win%'].rstrip('%'))/100
            away_prob = 1 - home_prob
            probs.append((row['Home'], home_prob))
            probs.append((row['Away'], away_prob))
    # Organize by team
    team_probs = defaultdict(list)
    for team, p in probs:
        team_probs[team].append(p)

    # Build division to teams mapping
    division_to_teams = defaultdict(list)
    for team in team_probs:
        div = DIVISION_MAP.get(team)
        if div:
            division_to_teams[div].append(team)

    # Monte Carlo for wins and division titles
    results = {}
    division_wins = defaultdict(float)
    for team in team_probs:
        results[team] = {'Expected_Wins': 0.0, 'Win_10%': 0, 'Win_90%': 0}
    n_teams = len(team_probs)
    all_teams = list(team_probs.keys())
    win_matrix = {team: [] for team in all_teams}
    for sim in range(n_sims):
        sim_wins = {}
        for team, p_list in team_probs.items():
            w = sum(random.random() < p for p in p_list)
            sim_wins[team] = w
            win_matrix[team].append(w)
        # Division winners
        for div, teams in division_to_teams.items():
            div_team_wins = {tm: sim_wins[tm] for tm in teams}
            max_wins = max(div_team_wins.values())
            tied_teams = [tm for tm, w in div_team_wins.items() if w == max_wins]
            for tm in tied_teams:
                division_wins[tm] += 1.0 / len(tied_teams)
    # Aggregate results
    for team in all_teams:
        team_wins = win_matrix[team]
        results[team]['Expected_Wins'] = sum(team_probs[team])
        results[team]['Win_10%'] = int(sorted(team_wins)[int(0.1*n_sims)])
        results[team]['Win_90%'] = int(sorted(team_wins)[int(0.9*n_sims)])
    division_win_pct = {team: division_wins[team] / n_sims for team in all_teams}
    # Build DataFrame
    df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index':'Team'})
    df['Div_Win_Pct'] = df['Team'].map(division_win_pct)
    # Reorder columns
    df = df[['Team', 'Expected_Wins', 'Win_10%', 'Win_90%', 'Div_Win_Pct']]
    return df


# ---------------------- PLAYOFF SIMULATION ----------------------
def simulate_playoffs(schedule_df, history, pts_for_m, pts_all_m, quantile_models, feat_cols, boot_models, cal_model, season, n_sims=2000):
    """Simulate playoff bracket and championship outcomes."""
    # Use same game simulator
    def game_win_prob(team1, team2):
        h_hist = history[history["team"] == team1].sort_values(["season", "week"]).tail(1)
        a_hist = history[history["team"] == team2].sort_values(["season", "week"]).tail(1)
        if h_hist.empty or a_hist.empty:
            return 0.5
        h_vec = h_hist[feat_cols].fillna(0)
        a_vec = a_hist[feat_cols].fillna(0)
        margin = float(pts_for_m.predict(h_vec.values)[0] - pts_all_m.predict(a_vec.values)[0])
        return float(cal_model.predict_proba([[margin]])[0, 1])

    # Get full win total sim results
    full_df = simulate_win_totals(schedule_df, history, pts_for_m, pts_all_m, quantile_models, feat_cols, boot_models, cal_model, season, n_sims)
    teams = list(full_df["Team"].unique())
    division_teams = defaultdict(list)
    for t in teams:
        division_teams[DIVISION_MAP[t]].append(t)
    conf_teams = {
        "AFC": [t for t in teams if DIVISION_MAP[t].startswith("AFC")],
        "NFC": [t for t in teams if DIVISION_MAP[t].startswith("NFC")]
    }

    # Init counters
    results = {team: defaultdict(int) for team in teams}
    for sim in range(n_sims):
        # Sample 1 sim of win totals
        win_sample = {}
        for team in teams:
            sampled = np.random.normal(full_df[full_df["Team"] == team]["Expected_Wins"].values[0], 1.5)
            win_sample[team] = round(sampled, 1)

        # Determine division winners
        div_winners = {}
        for div, tlist in division_teams.items():
            best = max(tlist, key=lambda t: win_sample[t])
            div_winners[div] = best

        # Assign seeds per conference
        conf_seeds = {}
        for conf, tlist in conf_teams.items():
            winners = [t for t in tlist if t in div_winners.values()]
            wildcards = [t for t in tlist if t not in winners]
            sorted_winners = sorted(winners, key=lambda t: -win_sample[t])
            sorted_wild = sorted(wildcards, key=lambda t: -win_sample[t])[:3]
            conf_seeds[conf] = sorted_winners + sorted_wild

        # Simulate playoffs
        conf_finalists = {}
        for conf in ["AFC", "NFC"]:
            seeds = conf_seeds[conf]
            rounds = [seeds[1:]]  # wild card matchups
            for r in range(3):  # 3 rounds
                next_round = []
                top_seed = seeds[0]
                matchups = list(zip(rounds[-1][::2], rounds[-1][1::2]))
                if r == 0:
                    # Insert top seed in next round
                    next_round.append(top_seed)
                winners = []
                for a, b in matchups:
                    if game_win_prob(a, b) > 0.5:
                        winners.append(a)
                    else:
                        winners.append(b)
                next_round.extend(winners)
                rounds.append(next_round)
                for team in winners:
                    results[team][f"round_{r+1}"] += 1
            # Super Bowl finalist
            sb_team = rounds[-1][0]
            results[sb_team]["sb_appear"] += 1
            conf_finalists[conf] = sb_team

        # Super Bowl matchup
        afc_team = conf_finalists["AFC"]
        nfc_team = conf_finalists["NFC"]
        if game_win_prob(afc_team, nfc_team) > 0.5:
            results[afc_team]["sb_win"] += 1
        else:
            results[nfc_team]["sb_win"] += 1

    # --- Compute SB Win % column for each team ---
    for team in teams:
        sb_win_pct = round(results[team]["sb_win"] / n_sims * 100, 1)
        results[team]["SB Win %"] = sb_win_pct

    final = []
    for team, r in results.items():
        final.append({
            "Team": team,
            "Conf Champ %": round(r["round_2"] / n_sims * 100, 1),
            "SB Appear %": round(r["sb_appear"] / n_sims * 100, 1),
            "SB Win %": r["SB Win %"]
        })
    return pd.DataFrame(final)


# ---------------------- BRACKET GRAPHVIZ RENDERING ----------------------
def build_playoff_bracket_graph(conf_seeds):
    import graphviz

    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR')  # Left to Right bracket style

    # Add round labels
    dot.attr(label="   Wild Card Round → Divisional → Conference → Super Bowl", fontsize='20', labelloc='t')

    for conf, seeds in conf_seeds.items():
        prefix = conf[:3]  # AFC or NFC
        top_seed = seeds[0]
        wild_card = seeds[1:]

        # Wild Card round matchups
        for i in range(0, len(wild_card), 2):
            t1 = wild_card[i]
            t2 = wild_card[i+1]
            m_id = f"{prefix}_R1_{i}"
            dot.node(m_id, f"{t1} vs {t2}", shape="box")
            dot.edge(t1, m_id)
            dot.edge(t2, m_id)

        # Connect top seed with placeholder winner
        top_match = f"{prefix}_TOP"
        dot.node(top_match, f"{top_seed} (Bye)", shape="box")
        dot.edge(top_seed, top_match)

        # Semifinal matchups (placeholder logic)
        dot.node(f"{prefix}_R2_1", f"{prefix} Semifinal 1", shape="box")
        dot.node(f"{prefix}_R2_2", f"{prefix} Semifinal 2", shape="box")
        dot.edge(f"{prefix}_R1_0", f"{prefix}_R2_1")
        dot.edge(f"{prefix}_R1_2", f"{prefix}_R2_1")
        dot.edge(top_match, f"{prefix}_R2_2")
        dot.edge(f"{prefix}_R1_1", f"{prefix}_R2_2")

        # Conference final
        dot.node(f"{prefix}_FINAL", f"{conf} Championship", shape="box")
        dot.edge(f"{prefix}_R2_1", f"{prefix}_FINAL")
        dot.edge(f"{prefix}_R2_2", f"{prefix}_FINAL")

    # Super Bowl
    dot.node("SB", "🏆 Super Bowl", shape="ellipse")
    dot.edge("AFC_FINAL", "SB")
    dot.edge("NFC_FINAL", "SB")

    return dot

############################
# STREAMLIT INTERFACE
############################
st.sidebar.header("Settings")

# Persist whether models are trained
if 'run_models' not in st.session_state:
    st.session_state['run_models'] = False
if 'trained' not in st.session_state:
    st.session_state['trained'] = False

years = st.sidebar.multiselect("Training Seasons", SEASONS, default=[SEASONS[-1]])
if not years:
    st.stop()

# --- Odds upload/editor (same logic as MLB) ---
st.sidebar.subheader("Bookmaker Odds")
odds_upload = st.sidebar.file_uploader(
    "Upload Odds CSV", type=["csv"],
    help=("Upload a CSV with columns: "
          "Matchup,Bookmaker Line,Over Price,Under Price,Home ML,Away ML "
          "or raw feed with 'market','point','price','home_team','away_team'.")
)
odds_data = {}
if odds_upload:
    df_ou = pd.read_csv(odds_upload)
    df_ou.columns = [c.strip().lower() for c in df_ou.columns]
    raw_cols = {"market","point","price","home_team","away_team"}
    if raw_cols.issubset(df_ou.columns):
        for (h,a), grp in df_ou.groupby(["home_team","away_team"]):
            matchup = f"{a} @ {h}"
            # totals
            tot = grp[grp["market"]=="totals"]
            if not tot.empty:
                over = tot[tot["label"].str.lower()=="over"]
                under= tot[tot["label"].str.lower()=="under"]
                line = (over["point"].iloc[0] if not over.empty else
                        under["point"].iloc[0] if not under.empty else None)
                over_p = over["price"].iloc[0] if not over.empty else None
                under_p= under["price"].iloc[0] if not under.empty else None
            else:
                line, over_p, under_p = None, None, None
            # moneyline
            ml = grp[grp["market"]=="h2h"]
            h_ml = ml[ml["label"].str.lower()==h].price.iloc[0] if not ml.empty else None
            a_ml = ml[ml["label"].str.lower()==a].price.iloc[0] if not ml.empty else None
            # spreads
            spread = grp[grp["market"] == "spread"]
            if not spread.empty:
                home_spread = spread[spread["label"].str.lower() == h.lower()]
                away_spread = spread[spread["label"].str.lower() == a.lower()]
                spread_line = float(home_spread["point"].iloc[0]) if not home_spread.empty else None
                spread_price = float(home_spread["price"].iloc[0]) if not home_spread.empty else None
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
        template_cols = {"matchup","bookmaker line","over price","under price","home ml","away ml"}
        if template_cols.issubset(df_ou.columns):
            for _, r in df_ou.iterrows():
                spread_line = float(r.get("spread line", 0))
                spread_price = float(r.get("spread price", 0))
                odds_data[r["matchup"]] = {
                    "book_line": float(r["bookmaker line"]),
                    "over_price": float(r["over price"]),
                    "under_price": float(r["under price"]),
                    "home_ml": float(r["home ml"]),
                    "away_ml": float(r["away ml"]),
                    "spread_line": spread_line,
                    "spread_price": spread_price
                }
        else:
            st.sidebar.error("Unrecognized odds schema. Please check CSV format.")

# --- Model Training/Prediction Toggle Logic ---
if st.sidebar.button("🚀 Train / Update"):
    st.session_state['run_models'] = True
    st.session_state['trained'] = False

if st.session_state['run_models']:
    if not st.session_state['trained']:
        with st.spinner("Loading + crunching …"):
            # Data loading and training
            wk = load_weekly(years)
            sched = load_schedules(years)
            ngs = load_ngs(years)
            snaps = load_snap_counts(years)
            pbp = load_pbp_data(years)
            team_week = build_team_week_df(wk, sched, ngs, snaps, pbp)
            team_week_feats = add_rolling_features(team_week)
            pts_for_m, pts_all_m, quantile_models, feat_cols, boot_models = train_models(team_week_feats)
            cal_model = calibrate_win_model(sched, team_week_feats, pts_for_m, pts_all_m, feat_cols, years)
            # Save to session
            st.session_state['wk'] = wk
            st.session_state['sched'] = sched
            st.session_state['team_week_feats'] = team_week_feats
            st.session_state['pts_for_m'] = pts_for_m
            st.session_state['pts_all_m'] = pts_all_m
            st.session_state['quantile_models'] = quantile_models
            st.session_state['feat_cols'] = feat_cols
            st.session_state['boot_models'] = boot_models
            st.session_state['cal_model'] = cal_model
            st.session_state['trained'] = True
    # Pull from session for downstream use
    wk = st.session_state['wk']
    sched = st.session_state['sched']
    team_week_feats = st.session_state['team_week_feats']
    pts_for_m = st.session_state['pts_for_m']
    pts_all_m = st.session_state['pts_all_m']
    quantile_models = st.session_state['quantile_models']
    feat_cols = st.session_state['feat_cols']
    boot_models = st.session_state['boot_models']
    cal_model = st.session_state['cal_model']
else:
    st.info("Please click ‘🚀 Train / Update’ to load data and train models.")
    st.stop()

# --- Prepare predictions ---
future_sched = sched[(sched['home_score'].isna()) & (sched['season'].isin(years))]
sel_season = st.sidebar.selectbox("Season to predict", sorted(future_sched['season'].unique(), reverse=True))
sel_week = st.sidebar.selectbox("Week", sorted(future_sched['week'].unique()))
games = future_sched[(future_sched['season']==sel_season) & (future_sched['week']==sel_week)]
preds = predict_week(games, team_week_feats, pts_for_m, pts_all_m, quantile_models, feat_cols, boot_models, cal_model)
# Round numeric columns to 1 decimal place
round_cols = [col for col in ['Home_Pts','Away_Pts','Total','Total_10%','Total_90%','Total_sigma'] if col in preds.columns]
preds[round_cols] = preds[round_cols].round(1)

# --- Merge 2025 Strength of Schedule ---
sos_df = pd.DataFrame(list(SOS_OWP.items()), columns=['Team','SOS'])
# Home/away SOS
preds = preds.merge(sos_df.rename(columns={'Team':'Home','SOS':'home_SOS'}), on='Home', how='left')
preds = preds.merge(sos_df.rename(columns={'Team':'Away','SOS':'away_SOS'}), on='Away', how='left')

# --- Sidebar filters and toggles ---
show_bands = st.sidebar.checkbox("Show confidence bands", value=True)
show_sigma = st.sidebar.checkbox("Show bootstrap σ", value=True)
min_win = st.sidebar.slider("Min Home Win %", 0.0, 100.0, 0.0)
# Filter based on slider
preds = preds[preds['Home_Win%'].str.rstrip('%').astype(float) >= min_win]

# --- Merge odds into preds and compute edges ---
if odds_data:
    odds_df = pd.DataFrame([
        {"Matchup": k, **v} for k,v in odds_data.items()
    ])
    # split matchup string into Away and Home
    odds_df[["Away","Home"]] = odds_df["Matchup"].str.split(" @ ", expand=True)
    preds = preds.merge(odds_df, on=["Home","Away"], how="left")
    # implied probabilities
    preds["home_imp_prob"] = preds["home_ml"].apply(convert_moneyline_to_prob)
    preds["away_imp_prob"] = preds["away_ml"].apply(convert_moneyline_to_prob)
    preds["ml_edge"] = (
        preds["Home_Win%"].str.rstrip('%').astype(float)/100
        - preds["home_imp_prob"]
    )
    preds["spread_edge"] = preds["Pred_Margin"] - preds["spread_line"]
    preds["total_edge"] = preds["Total"] - preds["book_line"]

    # --- Spread and Total Win Probability & EV ---
    from scipy.stats import norm
    # Avoid divide-by-zero
    preds["Total_sigma"] = preds["Total_sigma"].replace(0, 0.1)
    # Spread win probability (prob model margin beats spread line)
    preds["spread_prob"] = norm.sf(preds["spread_line"] - preds["Pred_Margin"], loc=0, scale=preds["Total_sigma"])
    # Spread EV (for spread_price, payout if correct minus loss if incorrect, normalized for price sign)
    # If spread_price < 0: bet risk X to win 100, payout = 100/X; else: bet 100 to win X, payout = X/100
    preds["spread_ev"] = np.where(
        preds["spread_price"] < 0,
        preds["spread_prob"] * (100 / preds["spread_price"].abs()) - (1 - preds["spread_prob"]),
        preds["spread_prob"] * (preds["spread_price"].abs() / 100) - (1 - preds["spread_prob"])
    )
    # Total win probability (for over bets)
    preds["total_prob"] = norm.sf(preds["book_line"] - preds["Total"], loc=0, scale=preds["Total_sigma"])
    preds["total_ev"] = np.where(
        preds["over_price"] < 0,
        preds["total_prob"] * (100 / preds["over_price"].abs()) - (1 - preds["total_prob"]),
        preds["total_prob"] * (preds["over_price"].abs() / 100) - (1 - preds["total_prob"])
    )

# --- KPI Bar ---
games_count = len(preds)
avg_win = (preds['Home_Win%'].str.rstrip('%').astype(float)/100).mean() if games_count else 0
avg_total = preds['Total'].mean() if games_count else 0
avg_sigma = preds['Total_sigma'].mean() if 'Total_sigma' in preds.columns and games_count else 0
avg_home_sos = preds['home_SOS'].mean() if 'home_SOS' in preds.columns else 0
avg_away_sos = preds['away_SOS'].mean() if 'away_SOS' in preds.columns else 0
retrain_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
cols = st.columns(6)
c1, c2, c3, c4, c5, c6 = cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]
c1.metric("Games", games_count)
c2.metric("Avg Win %", f"{avg_win:.1%}")
c3.metric("Avg Total", f"{avg_total:.1f}")
c4.metric("Avg σ", f"{avg_sigma:.1f}")
c5.metric("Last Retrain", retrain_time)
c6.metric("Avg SOS (H/A)", f"{avg_home_sos:.3f}/{avg_away_sos:.3f}")

# --- Edge Scouting Charts ---
st.subheader("Win% Distribution")
st.bar_chart(preds['Home_Win%'].str.rstrip('%').astype(float)/100)
if show_bands and 'Total_10%' in preds.columns and 'Total_90%' in preds.columns:
    st.subheader("Predicted Total with Bands")
    st.line_chart(preds[['Total', 'Total_10%', 'Total_90%']])
if show_sigma and 'Total_sigma' in preds.columns:
    st.subheader("Total σ by Game")
    st.bar_chart(preds['Total_sigma'])

# --- Recommended Bets ---
st.subheader("Recommended Bets")

# Moneyline bets
ml_bets = preds[preds["ml_edge"] >= ML_EDGE_THRESHOLD] if "ml_edge" in preds.columns else pd.DataFrame()
st.markdown("**Moneyline Bets (edge ≥ 5%)**")
if not ml_bets.empty:
    st.dataframe(
        ml_bets[["Home","Away","Home_Win%","home_ml","ml_edge"]],
        use_container_width=True
    )
else:
    st.write("No moneyline bets above threshold.")

# Spread bets
spread_bets = pd.DataFrame()
if "spread_edge" in preds.columns:
    spread_bets = preds[
        (preds["spread_edge"] >= SPREAD_EDGE_THRESHOLD) &
        (
            preds["spread_line"].round().isin(SPREAD_KEY_NUMBERS) |
            preds["Pred_Margin"].round().isin(SPREAD_KEY_NUMBERS)
        )
    ]
st.markdown("**Spread Bets (edge ≥ 3 points, on key numbers)**")
if not spread_bets.empty:
    st.dataframe(
        spread_bets[["Home","Away","Pred_Margin","spread_line","spread_edge","spread_prob","spread_price","spread_ev"]],
        use_container_width=True
    )
else:
    st.write("No spread bets above threshold and on key numbers.")

# Total bets
total_bets = pd.DataFrame()
if "total_edge" in preds.columns:
    total_bets = preds[
        (preds["total_edge"] >= TOTAL_EDGE_THRESHOLD) &
        (
            preds["book_line"].round().isin(TOTAL_KEY_NUMBERS) |
            preds["Total"].round().isin(TOTAL_KEY_NUMBERS)
        )
    ]
st.markdown("**Total Bets (edge ≥ 4 points, on key numbers)**")
if not total_bets.empty:
    st.dataframe(
        total_bets[["Home","Away","Total","book_line","total_edge","total_prob","over_price","total_ev"]],
        use_container_width=True
    )
else:
    st.write("No total bets above threshold and on key numbers.")

# --- Predictions Table ---
st.subheader("Predictions Table")
st.dataframe(preds, use_container_width=True)

# --- Game Drill-Down ---
st.subheader("Game Details")
for _, row in preds.iterrows():
    with st.expander(f"{row.Away} @ {row.Home} — Win% {row['Home_Win%']}"):
        # Strength of Schedule display
        st.markdown(f"**Strength of Schedule:** {row.away_SOS:.3f} (Away) vs {row.home_SOS:.3f} (Home)")
        # Score breakdown with clear labels
        st.markdown(f"**{row.Away} (Away):** {row.Away_Pts:.1f}  \n**{row.Home} (Home):** {row.Home_Pts:.1f}")

        # Total and confidence (rounded and explained)
        total_line = f"**Model Total (most likely):** {row.Total:.1f}"
        if show_bands and 'Total_10%' in row and 'Total_90%' in row:
            total_line += f"  \n• Expected range (80%): {row['Total_10%']:.1f} to {row['Total_90%']:.1f}"
        if show_sigma and 'Total_sigma' in row:
            total_line += f"  \n• Uncertainty (±1σ): {row.Total_sigma:.1f}"
        st.markdown(total_line)
        st.caption("The model’s single total prediction may lie outside the 80% range; the range shows where the combined score is most likely to fall.")
        # Last 6 weeks form for home team
        hist = team_week_feats[team_week_feats['team']==row.Home].sort_values(['season','week']).tail(6)
        if not hist.empty:
            st.line_chart(hist.set_index('week')[['pf_r3','pa_r3']])
# --- Futures Projections ---
st.subheader("Futures Projections")
# Maintain simulation in session state
if 'futures_df' not in st.session_state:
    st.session_state['futures_df'] = None

if st.button("🔄 Simulate Season Win Totals"):
    with st.spinner("Simulating season (this may take a moment)…"):
        st.session_state['futures_df'] = simulate_win_totals(
            sched, team_week_feats, pts_for_m, pts_all_m,
            quantile_models, feat_cols, boot_models, cal_model, sel_season
        )

if st.session_state['futures_df'] is not None:
    st.write("Season Win Total Projections")
    st.dataframe(st.session_state['futures_df'], use_container_width=True)

    # Optional: allow user to upload Vegas win-total lines to compute edge
    st.markdown("Upload a CSV with columns `Team,Vegas_OU` to compute edges.")
    vuploader = st.file_uploader("Win Totals CSV", type=["csv"])
    if vuploader:
        vs = pd.read_csv(vuploader)
        merged = st.session_state['futures_df'].merge(vs, on='Team', how='left')
        merged['Edge'] = merged['Expected_Wins'] - merged['Vegas_OU']
        st.write("Projected Wins vs Vegas O/U")
        st.dataframe(merged[['Team','Expected_Wins','Vegas_OU','Edge']], use_container_width=True)
# --- Playoff Simulation Button ---
# --- Playoff Simulation Button ---
if st.button("🏆 Simulate Playoff Outcomes"):
    with st.spinner("Simulating playoff brackets …"):
        # Sort simulated win totals to get top 7 seeds per conference
        win_df = simulate_win_totals(
            sched, team_week_feats, pts_for_m, pts_all_m,
            quantile_models, feat_cols, boot_models, cal_model, sel_season, n_sims=1000
        )
        afc_teams = win_df[win_df["Team"].map(lambda t: DIVISION_MAP.get(t, "").startswith("AFC"))]
        nfc_teams = win_df[win_df["Team"].map(lambda t: DIVISION_MAP.get(t, "").startswith("NFC"))]
        afc_seeds = afc_teams.sort_values("Expected_Wins", ascending=False).head(7)["Team"].tolist()
        nfc_seeds = nfc_teams.sort_values("Expected_Wins", ascending=False).head(7)["Team"].tolist()
        conf_seeds_preview = {
            "AFC": afc_seeds,
            "NFC": nfc_seeds
        }
        playoff_df = simulate_playoffs(
            sched, team_week_feats, pts_for_m, pts_all_m,
            quantile_models, feat_cols, boot_models, cal_model, sel_season
        )
        st.write("Playoff Simulation Results")
        st.dataframe(playoff_df, use_container_width=True)

        # --- Highlight predicted SB winner ---
        top_sb_team = playoff_df.sort_values("SB Win %", ascending=False).iloc[0]
        st.metric("Predicted SB Winner", f"{top_sb_team['Team']} ({top_sb_team['SB Win %']}%)")

        # --- Bracket Visual ---
        conf_seeds_preview = {
            "AFC": ["BUF", "KC", "MIA", "CLE", "PIT", "IND", "BAL"],
            "NFC": ["PHI", "DET", "SEA", "TB", "DAL", "ATL", "SF"]
        }
        st.markdown("### 🧩 Bracket Preview")
        bracket_dot = build_playoff_bracket_graph(conf_seeds_preview)
        st.graphviz_chart(bracket_dot)
# --- Download CSV ---
csv = preds.to_csv(index=False).encode()
st.download_button("Download Predictions CSV", csv, "predictions.csv")

# --- Hold-Out Validation ---
if st.sidebar.checkbox("Hold-Out Validation"):
    st.sidebar.write("Running hold-out test for first 2 weeks of next season")
    years = list(range(2018, datetime.now().year + 1))
    # Prepare dataset
    tw   = load_weekly(years)
    sched = load_schedules(years)
    ngs   = load_ngs(years)
    snaps = load_snap_counts(years)
    pbp   = load_pbp_data(years)
    df = add_rolling_features(build_team_week_df(tw, sched, ngs, snaps, pbp))
    # Define features
    desired = [
        *[c for c in df.columns if c.startswith((
            'pf_r','pa_r','rec_yac_oe_r','pf_diff','pa_diff','def_epa_r','def_sr_r'
        ))],
        'rest_days','starter_frac','epa_pp','sr','n_plays',
        'def_pct','def_epa_pp','def_sr','def_n_plays',
        'is_home','wk_norm'
    ]
    feats = [c for c in desired if c in df.columns]
    # Split train vs hold-out
    NEXT = datetime.now().year
    train = df[df['season'] < NEXT]
    hold  = df[(df['season'] == NEXT) & (df['week'] <= 2)]
    if hold.empty:
        st.write(f"No data for season {NEXT} weeks 1–2. Skipping validation.")
    else:
        X_tr = train[feats].fillna(0)
        y_tr_for = train['points_for']
        y_tr_all = train['points_allowed']
        X_te = hold[feats].fillna(0)
        y_te_for = hold['points_for']
        y_te_all = hold['points_allowed']
        # Train & evaluate
        m_for = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8).fit(X_tr, y_tr_for)
        m_all = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8).fit(X_tr, y_tr_all)
        maef = mean_absolute_error(y_te_for, m_for.predict(X_te))
        maea = mean_absolute_error(y_te_all, m_all.predict(X_te))
        st.write(f"**Hold-out MAE offense:** {maef:.2f}")
        st.write(f"**Hold-out MAE defense:** {maea:.2f}")
        # Residual histogram
        res_all = y_te_all.values - m_all.predict(X_te)
        fig, ax = plt.subplots()
        ax.hist(res_all, bins=30, edgecolor='black')
        ax.set_title('Defense residuals (points_allowed)')
        ax.set_xlabel('Actual − Predicted')
        st.pyplot(fig)

st.caption("v0.2 – still a toy. Validate before staking.")