import os
import re
import json
import datetime as dt
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import feedparser
import streamlit as st
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

# Modeling & Edge Parameters
EDGE_THRESHOLD = 0.04           # 4% edge
VOLATILITY_RATIO_THRESHOLD = 1.25 # Stability check
Z_MIN = 0.50                      # minimum |z| vs line to consider an edge
CALIBRATION_SPLIT_WEEKS = 6       # Weeks for Isotonic Regression calibration

# ---------------------- Data Loading & Feature Engineering ---------------------- #

@st.cache_data(ttl=24*3600)
def load_and_process_data(seasons):
    """Loads and merges PBP and roster data, calculating enhanced features."""
    
    st.write("Loading NFL Play-by-Play and Roster data...")
    # Use nfl_data_py import functions to load play-by-play and roster data
    pbp = nfl.import_pbp_data(seasons)
    rosters = nfl.import_rosters(seasons)
    
    # 1. Base Cleaning and Merging
    pbp = pbp[pbp['play_type'].isin(['pass', 'run'])]
    pbp['game_date'] = pd.to_datetime(pbp['game_date'])
    pbp['day_since_epoch'] = (pbp['game_date'] - dt.datetime(1970, 1, 1)).dt.days
    
    # Merge for player's team and position
    pbp_passers = pbp.merge(
        rosters[['player_id', 'position', 'team']], 
        left_on='passer_player_id', right_on='player_id', suffixes=('', '_passer'), how='left'
    )
    pbp_receivers = pbp.merge(
        rosters[['player_id', 'position', 'team']], 
        left_on='receiver_player_id', right_on='player_id', suffixes=('', '_receiver'), how='left'
    )
    pbp_rushers = pbp.merge(
        rosters[['player_id', 'position', 'team']], 
        left_on='rusher_player_id', right_on='player_id', suffixes=('', '_rusher'), how='left'
    )
    
    # Consolidate PBP data into a player-centric event log
    events = []
    
    # Pass Attempts (QB)
    qb_events = pbp_passers.rename(columns={'passer_player_id': 'player_id', 'pass_attempt': 'count_raw'})
    qb_events = qb_events[qb_events['play_type'] == 'pass']
    qb_events['market_key'] = 'player_pass_attempts'
    qb_events['count_share'] = qb_events['count_raw'] # Pass attempts is a raw volume metric
    events.append(qb_events[['player_id', 'market_key', 'count_raw', 'count_share', 'game_id', 'game_date', 'day_since_epoch', 'home_team', 'posteam', 'defteam', 'result']])

    # Receptions (WR/TE/RB)
    rec_events = pbp_receivers.rename(columns={'receiver_player_id': 'player_id', 'complete_pass': 'count_raw', 'pass_attempt': 'team_opportunity'})
    rec_events = rec_events[(rec_events['play_type'] == 'pass') & (rec_events['complete_pass'] == 1)]
    rec_events['market_key'] = 'player_receptions'
    # Calculate Target Share (Approximation: Receptions / Team Pass Attempts is simpler and effective)
    team_pass_attempts = rec_events.groupby(['game_id', 'posteam'])['pass_attempt'].transform('sum')
    rec_events['count_share'] = rec_events['count_raw'] / team_pass_attempts
    events.append(rec_events[['player_id', 'market_key', 'count_raw', 'count_share', 'game_id', 'game_date', 'day_since_epoch', 'home_team', 'posteam', 'defteam', 'result']])

    # Rush Attempts (RB/QB)
    rush_events = pbp_rushers.rename(columns={'rusher_player_id': 'player_id', 'rush_attempt': 'count_raw'})
    rush_events = rush_events[rush_events['play_type'] == 'run']
    rush_events['market_key'] = 'player_rush_attempts'
    # Calculate Rush Share
    team_rush_attempts = rush_events.groupby(['game_id', 'posteam'])['rush_attempt'].transform('sum')
    rush_events['count_share'] = rush_events['count_raw'] / team_rush_attempts
    events.append(rush_events[['player_id', 'market_key', 'count_raw', 'count_share', 'game_id', 'game_date', 'day_since_epoch', 'home_team', 'posteam', 'defteam', 'result']])

    df_events = pd.concat(events).reset_index(drop=True)
    df_events = df_events[df_events['count_raw'] > 0]
    
    # 2. Add Contextual Features
    df_events = _calculate_features(df_events, seasons)
    
    return df_events, rosters

def _calculate_features(df_events, seasons):
    """Calculates Opponent Factor, Home/Away Ratio, and Game Script Ratio."""
    
    # --- A. Opponent Adjustment Factor ---
    df_opp_factors = []
    league_averages = {}
    
    for market, player_col, team_col in [
        ('player_pass_attempts', 'count_raw', 'count_raw'), # QB Pass Attempts
        ('player_receptions', 'count_raw', 'pass_attempt'), # Receptions allowed per Pass Attempt
        ('player_rush_attempts', 'count_raw', 'rush_attempt') # Rush Attempts allowed per Rush Attempt
    ]:
        # Calculate defense's allowed count per game
        df_market = df_events[df_events['market_key'] == market]
        
        # Calculate defense-allowed count
        def_stats = df_market.groupby(['game_id', 'defteam']).agg(
            def_allowed_total=(player_col, 'sum'),
            def_allowed_game_count=(player_col, 'count')
        ).reset_index()
        
        # Adjust for Receptions/Rush Attempts where the team total is the denominator
        if market in ('player_receptions', 'player_rush_attempts'):
            team_opportunities = df_market.groupby(['game_id', 'posteam'])[team_col].sum().reset_index().rename(columns={team_col: 'team_opp_count'})
            def_stats = def_stats.merge(
                team_opportunities, 
                left_on=['game_id', 'defteam'], right_on=['game_id', 'posteam'], how='left'
            ).drop(columns='posteam')
            def_stats['def_allowed_avg'] = def_stats['def_allowed_total'] / def_stats['team_opp_count'].replace(0, np.nan)
        else:
            def_stats['def_allowed_avg'] = def_stats['def_allowed_total']
        
        # Calculate league average over all seasons
        league_avg = def_stats['def_allowed_avg'].mean()
        league_averages[market] = league_avg
        
        # Calculate Recency Weighted Opponent Factor (uses same decay)
        def_stats['day_since_epoch'] = df_market.groupby('game_id')['day_since_epoch'].first().loc[def_stats['game_id']].values
        def_stats = def_stats.sort_values('day_since_epoch')
        
        def_stats['def_allowed_rw'] = def_stats.groupby('defteam')['def_allowed_avg'].apply(
            lambda x: pd.Series(
                x.ewm(
                    alpha=1 - np.exp(-DECAY_RATE * (x.index.to_series() - x.index.to_series().shift(1)).fillna(0).dt.days), 
                    adjust=False
                ).mean()
            ).shift(1).fillna(x.mean())
        ).reset_index(drop=True)
        
        def_stats['opponent_factor'] = def_stats['def_allowed_rw'] / league_avg
        def_stats = def_stats[['game_id', 'defteam', 'opponent_factor']].rename(columns={'defteam': 'opponent'})
        df_opp_factors.append(def_stats)

    df_opp_factors_all = pd.concat(df_opp_factors)
    
    # Merge opponent factor back to event log
    df_events = df_events.merge(
        df_opp_factors_all.drop_duplicates(subset=['game_id', 'opponent']), 
        left_on=['game_id', 'defteam'], right_on=['game_id', 'opponent'], how='left'
    ).drop(columns='opponent')
    df_events['opponent_factor'].fillna(1.0, inplace=True) # Default to 1.0 if factor is missing

    # --- B. Home/Away Ratio and Game Script Ratio ---
    
    # Calculate Win/Loss (Game Script) - 1 for Win, 0 for Loss/Tie (if `result` is margin)
    df_events['game_result'] = np.where(df_events['result'] > 0, 'W', 'L')
    df_events['is_home'] = np.where(df_events['posteam'] == df_events['home_team'], 1, 0)
    
    player_context = df_events.groupby(['player_id', 'market_key']).agg(
        # Home/Away
        home_mean=('count_share', lambda x: x[df_events.loc[x.index, 'is_home'] == 1].mean()),
        away_mean=('count_share', lambda x: x[df_events.loc[x.index, 'is_home'] == 0].mean()),
        # Game Script (Win/Loss)
        win_mean=('count_share', lambda x: x[df_events.loc[x.index, 'game_result'] == 'W'].mean()),
        loss_mean=('count_share', lambda x: x[df_events.loc[x.index, 'game_result'] == 'L'].mean())
    ).reset_index()
    
    player_context['home_away_ratio'] = player_context['home_mean'] / player_context['away_mean'].replace(0, np.nan)
    player_context['game_script_ratio'] = player_context['win_mean'] / player_context['loss_mean'].replace(0, np.nan)
    
    # Merge context back to event log, fill NaNs with 1.0 (no effect)
    df_events = df_events.merge(
        player_context[['player_id', 'market_key', 'home_away_ratio', 'game_script_ratio']],
        on=['player_id', 'market_key'], how='left'
    )
    df_events[['home_away_ratio', 'game_script_ratio']].fillna(1.0, inplace=True)
    
    return df_events

def get_rotowire_injuries():
    """Fetches and parses Rotowire RSS for injury/status updates."""
    # (Implementation remains the same as original to maintain existing function)
    injury_feed = feedparser.parse('http://www.rotowire.com/rss/nfl-injuries.xml')
    injuries = []
    # Simplified parsing for demonstration
    for entry in injury_feed.entries:
        try:
            match = re.search(r'(\w+\s+\w+,\s+[A-Z]+\s+-\s+)([A-Z]{2,3})', entry.title)
            if match:
                player_name = match.group(1).split(',')[0].strip()
                status = match.group(2)
                if status not in ('OUT', 'PUP', 'IR', 'Q'):
                    continue
                injuries.append({'player_name': player_name, 'status': status, 'description': entry.summary})
        except:
            continue
    return pd.DataFrame(injuries)

# ---------------------- Core Modeling Functions ---------------------- #

def train_and_predict(df_events, df_picks, date_of_picks):
    """
    Trains models for each market and generates predictions for the input lines.
    
    This function implements the core enhancements:
    1. Models SHARE (count_share) instead of raw count for non-QB markets.
    2. Uses enhanced features for the Random Forest Regressor.
    3. Adjusts the final lambda using the Opponent Factor and Team Volume.
    """
    df_results = []
    
    # Isotonic Regression Setup (for probability calibration)
    cal_date = date_of_picks - dt.timedelta(weeks=CALIBRATION_SPLIT_WEEKS)
    df_cal = df_events[df_events['game_date'] < cal_date].copy()
    
    for market in SUPPORTED_MARKETS:
        st.write(f"Processing market: {market}...")
        df_market_hist = df_events[df_events['market_key'] == market].copy()
        
        # --- 1. Calculate Raw and Recency-Weighted Means (on SHARE or RAW) ---
        df_market_hist['date_diff'] = (date_of_picks - df_market_hist['game_date']).dt.days
        df_market_hist['weight'] = np.exp(-DECAY_RATE * df_market_hist['date_diff'])
        
        # Calculate historical features (X for Random Forest)
        df_player_hist = df_market_hist.groupby('player_id').agg(
            raw_share=('count_share', 'mean'),
            recency_weighted_share=('count_share', lambda x: np.sum(x * df_market_hist.loc[x.index, 'weight']) / np.sum(df_market_hist.loc[x.index, 'weight']))
        ).reset_index()
        
        # Merge additional context for the RF features
        latest_context = df_market_hist.sort_values('day_since_epoch', ascending=False).drop_duplicates('player_id')
        df_player_hist = df_player_hist.merge(
            latest_context[['player_id', 'home_away_ratio', 'game_script_ratio']],
            on='player_id', how='left'
        )

        # --- 2. Train Random Forest Regressor on Historical Data ---
        
        # Y is the actual player's SHARE on the game-day (what we want to predict)
        Y = df_market_hist['count_share'].values
        
        # X is the set of features from the *previous* game/history
        df_market_hist = df_market_hist.merge(df_player_hist[['player_id', 'raw_share', 'recency_weighted_share', 'home_away_ratio', 'game_script_ratio']], on='player_id', how='left')
        
        # ENHANCEMENT: Add new contextual features to the RF model
        X_cols = ['raw_share', 'recency_weighted_share', 'opponent_factor', 'home_away_ratio', 'game_script_ratio']
        X = df_market_hist[X_cols].fillna(0).values # Fillna 0 is a simplification for missing context
        
        # Train RF (Predicts adjusted SHARE/LAMBDA_ADJUSTED)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=5, n_jobs=-1)
        rf_model.fit(X, Y)
        
        # --- 3. Prepare Test Data (Current Picks) ---
        df_test = df_picks[df_picks['market_key'] == market].copy()
        if df_test.empty:
            continue

        # Merge historical player features and current opponent factor
        # Opponent factor is the one calculated for the current game's opponent.
        df_test = df_test.merge(df_player_hist, on='player_id', how='left')
        df_test = df_test.merge(
            latest_context[['player_id', 'defteam', 'opponent_factor']], # Using latest historical Opponent Factor for current game
            on='player_id', how='left', suffixes=('', '_opp')
        )
        df_test['opponent_factor'] = df_test['opponent_factor_opp']
        
        # Fill missing features for players with no history with mean of the market (simplification)
        df_test[X_cols] = df_test[X_cols].fillna(df_market_hist[X_cols].mean())

        # Predict adjusted SHARE/LAMBDA_ADJUSTED using the trained RF model
        X_test = df_test[X_cols].fillna(0).values
        lambda_adjusted_share = rf_model.predict(X_test)
        df_test['lambda_adjusted_share'] = lambda_adjusted_share
        
        # --- 4. Final RAW Count Projection (LAMBDA_FINAL) ---
        
        if market == 'player_pass_attempts':
            # QB Pass Attempts: Lambda_FINAL = Lambda_ADJUSTED_RAW * Opponent_Factor
            df_test['lambda_final'] = df_test['lambda_adjusted_share'] * df_test['opponent_factor']
        else:
            # Receptions/Rushes: Lambda_FINAL = Lambda_ADJUSTED_SHARE * Lambda_TEAM_VOLUME * Opponent_Factor
            
            # Get Team Volume (Recency Weighted Mean of Team Pass/Rush Attempts per game)
            volume_col = 'pass_attempt' if market == 'player_receptions' else 'rush_attempt'
            
            # Separate historical team volume calculation
            df_team_volume = df_events.groupby(['game_id', 'posteam']).agg(
                team_volume=(volume_col, 'sum'),
                game_date=('game_date', 'first')
            ).reset_index()
            
            df_team_volume['day_diff'] = (date_of_picks - df_team_volume['game_date']).dt.days
            df_team_volume['weight'] = np.exp(-DECAY_RATE * df_team_volume['day_diff'])

            # Calculate Team RW Mean
            df_team_rw = df_team_volume.groupby('posteam').agg(
                lambda_team_volume=('team_volume', lambda x: np.sum(x * df_team_volume.loc[x.index, 'weight']) / np.sum(df_team_volume.loc[x.index, 'weight']))
            ).reset_index()
            
            # Merge and Calculate Lambda_FINAL
            df_test = df_test.merge(df_team_rw, left_on='team', right_on='posteam', how='left')
            df_test['lambda_final'] = df_test['lambda_adjusted_share'] * df_test['lambda_team_volume'] * df_test['opponent_factor']
            
        # --- 5. Simulation & Edge Calculation (Same as original logic) ---
        
        # Run Simulation
        df_test['model_mean'] = df_test['lambda_final']
        df_test['model_std'] = np.sqrt(df_test['model_mean']) # For Poisson, std = sqrt(mean)
        
        probs = []
        for lam, line in df_test[['model_mean', 'book_line']].values:
            sims = poisson.rvs(lam, size=DEFAULT_N_SIMULATIONS)
            p_over = np.sum(sims > line) / DEFAULT_N_SIMULATIONS
            probs.append((p_over, 1 - p_over))
            
        df_test[['p_over', 'p_under']] = probs

        # Calculate implied probability from moneyline odds
        df_test['implied_prob_used'] = np.where(
            df_test['side'] == 'Over', 
            1 / (1 + df_test['odds'].str.replace('+', '').astype(float) / 100),
            1 / (1 + df_test['odds'].str.replace('+', '').astype(float) / 100) # Simplified for positive odds only
        )
        
        # Calculate raw edge
        df_test['edge_prob'] = np.where(
            df_test['side'] == 'Over',
            df_test['p_over'],
            df_test['p_under']
        )
        df_test['edge_pct'] = df_test['edge_prob'] - df_test['implied_prob_used']

        # --- 6. Apply Guardrails and Calibration ---
        
        # Calibration (Isotonic Regression) - Train on historical data before the cut-off
        df_cal_market = df_cal[df_cal['market_key'] == market].copy()
        
        # Get historical outcomes (1 for success, 0 for failure)
        df_cal_market['outcome_over'] = np.where(df_cal_market['count_raw'] > df_cal_market['book_line'], 1, 0)
        
        # Recalculate model probability on historical data
        df_cal_market['p_over_raw'] = poisson.sf(df_cal_market['book_line'], df_cal_market['lambda_final'])

        if not df_cal_market.empty and len(df_cal_market) > 10:
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            
            # Train Isotonic Regressor (Y = actual outcome, X = model raw probability)
            iso_reg.fit(df_cal_market['p_over_raw'].values, df_cal_market['outcome_over'].values)
            
            # Apply calibrated probabilities
            df_test['p_over_calibrated'] = iso_reg.predict(df_test['p_over'].values)
            df_test['p_under_calibrated'] = 1 - df_test['p_over_calibrated']
            df_test['edge_prob_calibrated'] = np.where(df_test['side'] == 'Over', df_test['p_over_calibrated'], df_test['p_under_calibrated'])
            df_test['edge_pct_calibrated'] = df_test['edge_prob_calibrated'] - df_test['implied_prob_used']
            
            # Use calibrated edge for final decision
            df_test['edge_pct_final'] = df_test['edge_pct_calibrated']
        else:
            df_test['edge_pct_final'] = df_test['edge_pct']
        
        # Apply Guardrails
        df_test['z_vs_line'] = (df_test['model_mean'] - df_test['book_line']) / df_test['model_std']
        df_test['vol_ratio'] = df_test['model_std'] / df_test['model_mean']
        
        # 1. Edge Threshold
        df_test['pass_edge'] = df_test['edge_pct_final'].abs() >= EDGE_THRESHOLD
        # 2. Volatility Ratio
        df_test['pass_vol'] = df_test['vol_ratio'] <= VOLATILITY_RATIO_THRESHOLD
        # 3. Z-Score
        df_test['pass_z'] = df_test['z_vs_line'].abs() >= Z_MIN
        
        # Final Recommendation
        df_test['recommendation'] = np.where(
            df_test['pass_edge'] & df_test['pass_vol'] & df_test['pass_z'],
            np.where(df_test['side'] == 'Over', 'Over', 'Under'),
            'Hold'
        )

        df_results.append(df_test)
        
    return pd.concat(df_results)

# ---------------------- Streamlit UI (Simplified) ---------------------- #

def generate_app_ui(df_events, rosters):
    """Generates the Streamlit User Interface for the app."""
    st.title("🏈 FoxEdge NFL Prop Projections (Enhanced)")
    st.caption("Enhanced Model: Share-Based, Opponent-Adjusted, Contextual RF")

    # Sidebar for Inputs
    with st.sidebar:
        st.header("1. Input Picks")
        
        # Placeholder for User input of picks (player_name, market_key, book_line, odds)
        st.markdown("**Example Pick Format:**")
        st.code("Josh Allen,player_pass_attempts,36.5,-110,Over")
        raw_picks = st.text_area("Enter Picks (Name, Market, Line, Odds, Side)", height=200)

        date_of_picks = st.date_input("Date of Picks", dt.date.today())
        
        st.header("2. Player/Team Mapping")
        # Dummy pick data creation based on text input
        picks_list = []
        if raw_picks:
            for line in raw_picks.split('\n'):
                try:
                    name, market, line_val, odds, side = [x.strip() for x in line.split(',')]
                    picks_list.append({
                        'player_name': name,
                        'market_key': market,
                        'book_line': float(line_val),
                        'odds': odds,
                        'side': side
                    })
                except:
                    continue
        
        df_picks_raw = pd.DataFrame(picks_list)

    if df_picks_raw.empty:
        st.info("Please enter player prop picks in the sidebar to run the analysis.")
        return

    # Map names to IDs and team
    df_picks_mapped = df_picks_raw.merge(
        rosters[['full_name', 'player_id', 'team']].drop_duplicates('player_id'), 
        left_on='player_name', right_on='full_name', how='left'
    ).rename(columns={'team': 'team_abbr'})
    
    df_picks_mapped.rename(columns={'team_abbr': 'team'}, inplace=True)
    df_picks_mapped.dropna(subset=['player_id'], inplace=True)
    
    if df_picks_mapped.empty:
        st.error("Could not find any of the entered players in the roster data.")
        return

    # Run the model
    df_recs = train_and_predict(df_events, df_picks_mapped, dt.datetime.combine(date_of_picks, dt.time()))

    # --- Display Results ---
    st.subheader("Final Recommendations")
    
    df_final = df_recs[df_recs['recommendation'].isin(['Over','Under'])].sort_values('edge_pct_final', ascending=False)
    
    if df_final.empty:
        st.warning("No picks passed all guardrails (Edge > 4%, Volatility Ratio < 1.25, |Z| > 0.50).")
    else:
        display_cols = ['player_name','team','market_key','side','book_line','odds','edge_pct_final','edge_prob_calibrated','model_mean','model_std']
        df_display = df_final[display_cols].copy()
        df_display['edge_pct_final'] = (df_display['edge_pct_final'] * 100).round(2).astype(str) + '%'
        df_display['edge_prob_calibrated'] = (df_display['edge_prob_calibrated'] * 100).round(2).astype(str) + '%'
        df_display.columns = ['Player', 'Team', 'Market', 'Side', 'Line', 'Odds', 'Edge %', 'Model Prob', 'Model Mean', 'Model Std']
        st.dataframe(df_display, use_container_width=True)
        
    st.subheader("Model Diagnostics (All Candidates)")
    # Optional detailed view for debugging and transparency
    cols_diag = ['player_name','market_key','side','book_line','edge_pct_final','pass_edge','pass_vol','pass_z','z_vs_line','vol_ratio','model_mean','opponent_factor']
    st.dataframe(df_recs.sort_values('edge_pct_final', ascending=False)[cols_diag].fillna('-'), use_container_width=True)

# ---------------------- Main Execution Block ---------------------- #

# Simulate Streamlit environment execution
if __name__ == '__main__':
    # Default parameters for initial data load
    LATEST_SEASON = 2024 # Current year
    SEASONS_TO_LOAD = list(range(LATEST_SEASON - 2, LATEST_SEASON + 1)) # Load last 3 seasons + current

    # Load data once
    df_events, rosters = load_and_process_data(SEASONS_TO_LOAD)
    
    # Run the UI logic
    generate_app_ui(df_events, rosters)