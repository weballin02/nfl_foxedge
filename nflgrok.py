import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
import datetime

# Compute defensive stats and league averages
@st.cache_data
def compute_defensive_stats(weekly):
    teams = weekly['opponent_team'].unique()
    def_stats = {}
    
    # Calculate per-player averages against each defense
    for team in teams:
        def_df = weekly[weekly['opponent_team'] == team]
        if len(def_df) == 0:
            continue
            
        # Calculate what individual players average against this defense
        def_stats[team] = {
            'pass_yards_allowed': def_df['passing_yards'].mean(),
            'rush_yards_allowed': def_df['rushing_yards'].mean(),
            'rec_yards_allowed': def_df['receiving_yards'].mean(),
            'pass_td_allowed': def_df['passing_tds'].mean(),
            'rush_td_allowed': def_df['rushing_tds'].mean(),
            'rec_td_allowed': def_df['receiving_tds'].mean(),
            'pass_attempts_allowed': def_df['attempts'].mean() if 'attempts' in def_df.columns else 25.0,
            'carries_allowed': def_df['carries'].mean() if 'carries' in def_df.columns else def_df['rushing_attempts'].mean() if 'rushing_attempts' in def_df.columns else 12.0,
            'targets_allowed': def_df['targets'].mean() if 'targets' in def_df.columns else 6.0
        }
    
    # Compute league averages (what players typically average)
    if def_stats:
        league_avgs = {
            key: np.mean([stats[key] for stats in def_stats.values() if not np.isnan(stats[key])])
            for key in def_stats[list(def_stats.keys())[0]]
        }
    else:
        league_avgs = {
            'pass_yards_allowed': 180.0, 'rush_yards_allowed': 65.0, 'rec_yards_allowed': 75.0,
            'pass_td_allowed': 1.2, 'rush_td_allowed': 0.4, 'rec_td_allowed': 0.3,
            'pass_attempts_allowed': 25.0, 'carries_allowed': 12.0, 'targets_allowed': 6.0
        }
    return def_stats, league_avgs

# Get defensive rank adjustment
def get_defensive_rank(opponent, prop_type, def_stats, league_avgs):
    opp_stats = def_stats.get(opponent.upper(), league_avgs)
    if prop_type == "Passing Yards":
        avg_allowed = opp_stats['pass_yards_allowed']
        league_avg = league_avgs['pass_yards_allowed']
    elif prop_type == "Rushing Yards":
        avg_allowed = opp_stats['rush_yards_allowed']
        league_avg = league_avgs['rush_yards_allowed']
    elif prop_type == "Receiving Yards":
        avg_allowed = opp_stats['rec_yards_allowed']
        league_avg = league_avgs['rec_yards_allowed']
    elif prop_type == "Passing TD":
        avg_allowed = opp_stats['pass_td_allowed']
        league_avg = league_avgs['pass_td_allowed']
    elif prop_type == "Rushing TD":
        avg_allowed = opp_stats['rush_td_allowed']
        league_avg = league_avgs['rush_td_allowed']
    else:  # Receiving TD
        avg_allowed = opp_stats['rec_td_allowed']
        league_avg = league_avgs['rec_td_allowed']
    
    # More conservative adjustment - max 5% swing
    if league_avg != 0:
        raw_adjustment = (avg_allowed - league_avg) / league_avg
        # Cap the adjustment at ±0.05 (5%)
        adjustment = np.clip(raw_adjustment * 0.3, -0.05, 0.05)
    else:
        adjustment = 0
    return 1 + adjustment

# Project yards (improved approach with regularization and more data)
def project_yards(player_df, prop_type, adj_factor, num_recent=8):
    # Use more games for better stability
    player_df = player_df.tail(num_recent)
    
    if len(player_df) < 3:
        return None
        
    if prop_type == "Passing Yards":
        # Use attempts as primary predictor
        if 'attempts' not in player_df.columns or player_df['attempts'].isna().all():
            return None
        attempts = player_df['attempts'].values
        yards = player_df['passing_yards'].values
        
        # Calculate yards per attempt with recent weighting
        weights = np.linspace(0.5, 1.0, len(attempts))  # Recent games weighted more
        ypa = np.average(yards / attempts, weights=weights)
        
        # Use recent average attempts with some regression to mean
        recent_attempts = np.mean(attempts[-3:]) if len(attempts) >= 3 else np.mean(attempts)
        league_avg_attempts = 25.0
        projected_attempts = 0.7 * recent_attempts + 0.3 * league_avg_attempts
        
        proj = ypa * projected_attempts * adj_factor
        
    elif prop_type == "Rushing Yards":
        # Get carries column
        carries_col = 'carries' if 'carries' in player_df.columns else 'rushing_attempts'
        if carries_col not in player_df.columns or player_df[carries_col].isna().all():
            return None
            
        carries = player_df[carries_col].values
        yards = player_df['rushing_yards'].values
        
        # Calculate yards per carry with recent weighting
        weights = np.linspace(0.5, 1.0, len(carries))
        ypc = np.average(yards / carries, weights=weights)
        
        # Project carries based on recent usage
        recent_carries = np.mean(carries[-3:]) if len(carries) >= 3 else np.mean(carries)
        league_avg_carries = 12.0
        projected_carries = 0.7 * recent_carries + 0.3 * league_avg_carries
        
        proj = ypc * projected_carries * adj_factor
        
    else:  # Receiving Yards
        if 'targets' not in player_df.columns or player_df['targets'].isna().all():
            return None
            
        targets = player_df['targets'].values
        yards = player_df['receiving_yards'].values
        
        # Calculate yards per target with recent weighting
        weights = np.linspace(0.5, 1.0, len(targets))
        ypt = np.average(yards / targets, weights=weights)
        
        # Project targets based on recent usage
        recent_targets = np.mean(targets[-3:]) if len(targets) >= 3 else np.mean(targets)
        league_avg_targets = 6.0
        projected_targets = 0.7 * recent_targets + 0.3 * league_avg_targets
        
        proj = ypt * projected_targets * adj_factor
    
    return max(proj, 0)  # Ensure non-negative

# Project TDs (improved Bayesian approach)
def project_td(player_df, prop_type, adj_factor, num_recent=8):
    player_df = player_df.tail(num_recent)
    
    if prop_type == "Passing TD":
        tds = player_df['passing_tds'].dropna().values
        # More realistic priors for passing TDs
        mean_prior, var_prior = 1.0, 0.8
    elif prop_type == "Rushing TD":
        tds = player_df['rushing_tds'].dropna().values
        mean_prior, var_prior = 0.3, 0.4
    else:  # Receiving TD
        tds = player_df['receiving_tds'].dropna().values
        mean_prior, var_prior = 0.25, 0.3
    
    if len(tds) == 0:
        return None
    
    # Bayesian update with more conservative priors
    k = mean_prior**2 / var_prior
    b = mean_prior / var_prior
    shape = k + sum(tds)
    rate = b + len(tds)
    
    # Use weighted average of recent games
    weights = np.linspace(0.5, 1.0, len(tds))
    recent_avg = np.average(tds, weights=weights) if len(tds) > 1 else tds[0]
    
    # Blend Bayesian estimate with recent average
    bayesian_est = shape / rate
    blended_est = 0.6 * bayesian_est + 0.4 * recent_avg
    
    proj_mean = blended_est * adj_factor
    return max(proj_mean, 0)

# Add game context factors
def get_game_context_factor(player_df, is_home_game=True):
    """Get context factors that affect projections"""
    if len(player_df) < 3:
        return 1.0
    
    # Home field advantage (small boost for home players)
    home_factor = 1.02 if is_home_game else 0.98
    
    # Recent form factor (last 3 games vs previous games)
    if len(player_df) >= 6:
        recent_3 = player_df.tail(3)
        previous_3 = player_df.iloc[-6:-3]
        
        # Calculate average performance metric
        if 'passing_yards' in player_df.columns:
            recent_avg = recent_3['passing_yards'].mean()
            prev_avg = previous_3['passing_yards'].mean()
        elif 'rushing_yards' in player_df.columns:
            recent_avg = recent_3['rushing_yards'].mean()
            prev_avg = previous_3['rushing_yards'].mean()
        elif 'receiving_yards' in player_df.columns:
            recent_avg = recent_3['receiving_yards'].mean()
            prev_avg = previous_3['receiving_yards'].mean()
        else:
            return home_factor
        
        if prev_avg > 0:
            form_factor = min(1.1, max(0.9, recent_avg / prev_avg))
        else:
            form_factor = 1.0
    else:
        form_factor = 1.0
    
    return home_factor * form_factor

# Load data once
@st.cache_data
def load_data(year=2025):
    weekly = nfl.import_weekly_data([year-1, year])
    rosters = nfl.import_rosters([year])
    schedule = nfl.import_schedules([year])
    def_stats, league_avgs = compute_defensive_stats(weekly)
    return weekly, rosters, schedule, def_stats, league_avgs

weekly, rosters, schedule, def_stats, league_avgs = load_data()

# Filter upcoming games
schedule['gameday_dt'] = pd.to_datetime(schedule['gameday'])
current_time = datetime.datetime.now()
upcoming = schedule[schedule['gameday_dt'] > current_time].sort_values('gameday_dt')
upcoming_games = [f"{row['away_team']} @ {row['home_team']} on {row['gameday']}" for _, row in upcoming.iterrows()]

st.title("NFL Player Prop Predictor")

tab1, tab2, tab3 = st.tabs(["Single Player", "Game Matchup", "All Upcoming Games"])

with tab1:
    st.header("Single Player Projection")
    player_name = st.text_input("Enter Player Name (e.g., Patrick Mahomes)", key="single_player")
    prop_type = st.selectbox("Prop Type", ["Passing Yards", "Rushing Yards", "Receiving Yards", "Passing TD", "Rushing TD", "Receiving TD"], key="single_prop")
    opponent = st.text_input("Enter Opponent Abbrev (e.g., KC)", key="single_opp")
    num_recent_games = st.slider("Recent Games for Model", 3, 10, 5, key="single_recent")

    if st.button("Generate Projection", key="single_btn"):
        try:
            player_df = weekly[weekly['player_display_name'].str.contains(player_name, case=False, na=False)].sort_values('week')
            if player_df.empty:
                st.error("Player not found.")
            else:
                adj_factor = get_defensive_rank(opponent, prop_type, def_stats, league_avgs)
                context_factor = get_game_context_factor(player_df, is_home_game=True)  # Assume home for single player
                
                if "Yards" in prop_type:
                    proj = project_yards(player_df, prop_type, adj_factor * context_factor, num_recent_games)
                    if proj is not None:
                        # Add confidence interval
                        std_dev = proj * 0.15  # 15% coefficient of variation
                        st.write(f"Projected {prop_type}: {proj:.1f} (±{std_dev:.1f})")
                        st.write(f"Range: {proj-std_dev:.1f} - {proj+std_dev:.1f}")
                        st.write("Suggestion: Bet the under if line > projection + 0.5*std_dev")
                        st.write(f"Defensive adjustment: {adj_factor:.3f}, Context factor: {context_factor:.3f}")
                    else:
                        st.error("Insufficient data for yardage projection.")
                else:
                    proj = project_td(player_df, prop_type, adj_factor * context_factor, num_recent_games)
                    if proj is not None:
                        # TD projections are more volatile
                        std_dev = proj * 0.25  # 25% coefficient of variation
                        st.write(f"Projected {prop_type}: {proj:.2f} (±{std_dev:.2f})")
                        st.write(f"Range: {max(0, proj-std_dev):.2f} - {proj+std_dev:.2f}")
                        st.write("TD props are high-variance; use for anytime TD bets.")
                        st.write(f"Defensive adjustment: {adj_factor:.3f}, Context factor: {context_factor:.3f}")
                    else:
                        st.error("Insufficient data for TD projection.")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("Select Upcoming Game for Matchup Projections")
    selected_game = st.selectbox("Upcoming Games", upcoming_games, key="game_select")
    if selected_game:
        try:
            game_idx = upcoming_games.index(selected_game)
            game_row = upcoming.iloc[game_idx]
            away, home = game_row['away_team'], game_row['home_team']
            st.write(f"Projecting for key players in {away} @ {home}")

            # Get rosters for both teams (QB1, RB1-2, WR1-3, TE1-2)
            away_roster = rosters[(rosters['team'] == away) & rosters['depth_chart_position'].isin(['1', '2']) & rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])]
            home_roster = rosters[(rosters['team'] == home) & rosters['depth_chart_position'].isin(['1', '2']) & rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])]
            all_players = pd.concat([away_roster.assign(team=away, opp=home), home_roster.assign(team=home, opp=away)])

            projections = []
            for _, player in all_players.iterrows():
                player_df = weekly[weekly['player_id'] == player['player_id']].sort_values('week')
                if player_df.empty:
                    continue
                pos = player['position']
                opp = player['opp']
                props = {
                    'QB': ["Passing Yards", "Passing TD"],
                    'RB': ["Rushing Yards", "Rushing TD", "Receiving Yards"],
                    'WR': ["Receiving Yards", "Receiving TD"],
                    'TE': ["Receiving Yards", "Receiving TD"]
                }.get(pos, [])
                for prop in props:
                    adj_factor = get_defensive_rank(opp, prop, def_stats, league_avgs)
                    is_home = (player['team'] == home)
                    context_factor = get_game_context_factor(player_df, is_home_game=is_home)
                    
                    if "Yards" in prop:
                        proj = project_yards(player_df, prop, adj_factor * context_factor)
                        if proj is not None:
                            std_dev = proj * 0.15
                            projections.append({
                                'Player': player['player_name'],
                                'Team': player['team'],
                                'Prop': prop,
                                'Projection': f"{proj:.1f}",
                                'Range': f"{proj-std_dev:.1f}-{proj+std_dev:.1f}",
                                'Adj': f"{adj_factor:.2f}",
                                'Suggestion': 'Under if line > proj+0.5*std'
                            })
                    else:
                        proj = project_td(player_df, prop, adj_factor * context_factor)
                        if proj is not None:
                            std_dev = proj * 0.25
                            projections.append({
                                'Player': player['player_name'],
                                'Team': player['team'],
                                'Prop': prop,
                                'Projection': f"{proj:.2f}",
                                'Range': f"{max(0, proj-std_dev):.2f}-{proj+std_dev:.2f}",
                                'Adj': f"{adj_factor:.2f}",
                                'Suggestion': 'High variance'
                            })

            if projections:
                st.dataframe(pd.DataFrame(projections))
            else:
                st.write("No sufficient data for projections.")
        except Exception as e:
            st.error(f"Error processing game: {e}")

with tab3:
    st.header("Projections for All Upcoming Games")
    num_games = min(10, len(upcoming))  # Limit for performance
    if num_games == 0:
        st.write("No upcoming games found.")
    for i in range(num_games):
        game_row = upcoming.iloc[i]
        away, home, gameday = game_row['away_team'], game_row['home_team'], game_row['gameday']
        with st.expander(f"{away} @ {home} on {gameday}"):
            try:
                away_roster = rosters[(rosters['team'] == away) & rosters['depth_chart_position'].isin(['1', '2']) & rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])]
                home_roster = rosters[(rosters['team'] == home) & rosters['depth_chart_position'].isin(['1', '2']) & rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])]
                all_players = pd.concat([away_roster.assign(team=away, opp=home), home_roster.assign(team=home, opp=away)])

                projections = []
                for _, player in all_players.iterrows():
                    player_df = weekly[weekly['player_id'] == player['player_id']].sort_values('week')
                    if player_df.empty:
                        continue
                    pos = player['position']
                    opp = player['opp']
                    props = {
                        'QB': ["Passing Yards", "Passing TD"],
                        'RB': ["Rushing Yards", "Rushing TD", "Receiving Yards"],
                        'WR': ["Receiving Yards", "Receiving TD"],
                        'TE': ["Receiving Yards", "Receiving TD"]
                    }.get(pos, [])
                    for prop in props:
                        adj_factor = get_defensive_rank(opp, prop, def_stats, league_avgs)
                        is_home = (player['team'] == home)
                        context_factor = get_game_context_factor(player_df, is_home_game=is_home)
                        
                        if "Yards" in prop:
                            proj = project_yards(player_df, prop, adj_factor * context_factor)
                            if proj is not None:
                                std_dev = proj * 0.15
                                projections.append({
                                    'Player': player['player_name'],
                                    'Team': player['team'],
                                    'Prop': prop,
                                    'Projection': f"{proj:.1f}",
                                    'Range': f"{proj-std_dev:.1f}-{proj+std_dev:.1f}",
                                    'Adj': f"{adj_factor:.2f}",
                                    'Suggestion': 'Under if line > proj+0.5*std'
                                })
                        else:
                            proj = project_td(player_df, prop, adj_factor * context_factor)
                            if proj is not None:
                                std_dev = proj * 0.25
                                projections.append({
                                    'Player': player['player_name'],
                                    'Team': player['team'],
                                    'Prop': prop,
                                    'Projection': f"{proj:.2f}",
                                    'Range': f"{max(0, proj-std_dev):.2f}-{proj+std_dev:.2f}",
                                    'Adj': f"{adj_factor:.2f}",
                                    'Suggestion': 'High variance'
                                })

                if projections:
                    st.dataframe(pd.DataFrame(projections))
                else:
                    st.write("No sufficient data for projections.")
            except Exception as e:
                st.write(f"Error processing {away} @ {home}: {e}")