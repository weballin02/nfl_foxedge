import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
import nfl_data_py as nfl
import pmdarima as pm

# 1. Data Loading and Caching
@st.cache(ttl=86400, show_spinner=False)
def load_nfl_data(seasons):
    df_list = []
    for year in seasons:
        # load schedule, which includes home_score/away_score
        sched = nfl.import_schedules([year])
        sched = sched.reset_index()
        # normalize a reliable game_date column from common schedule fields
        date_col = next((c for c in ['game_date', 'gameday', 'game_datetime', 'game_time', 'start_time'] if c in sched.columns), None)
        if date_col is not None:
            sched['game_date'] = pd.to_datetime(sched[date_col])
        else:
            # fallback: try to parse whatever looks like a datetime in the index or raise a clear error
            raise KeyError("Could not determine a game date column in schedules. Expected one of ['game_date','gameday','game_datetime','game_time','start_time'].")
        # pivot for home team rows
        home = sched[['game_id','game_date','home_team','away_team','season','week','home_score','away_score']].rename(
            columns={
                'home_team': 'team_abbr',
                'home_score': 'points_for',
                'away_score': 'points_against'
            }
        )
        home['home_team'] = home['team_abbr']
        # pivot for away team rows
        away = sched[['game_id','game_date','home_team','away_team','season','week','home_score','away_score']].rename(
            columns={
                'away_team': 'team_abbr',
                'away_score': 'points_for',
                'home_score': 'points_against'
            }
        )
        away['away_team'] = away['team_abbr']
        # combine both
        df_list.append(pd.concat([home, away], ignore_index=True))
    data = pd.concat(df_list, ignore_index=True)
    data['game_date'] = pd.to_datetime(data['game_date'])
    data.sort_values('game_date', inplace=True)
    return data

# 2. Feature Engineering

def generate_rolling_features(df, windows=[1,3,5]):
    feats = []
    for w in windows:
        df[f'pts_for_{w}w'] = df.groupby('team_abbr')['points_for'].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        df[f'pts_against_{w}w'] = df.groupby('team_abbr')['points_against'].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
    return df

def latest_team_rollups_before(df_hist, cutoff_dates):
    """
    For a given mapping of team -> cutoff datetime, return the latest available rolling features row prior to that cutoff.
    cutoff_dates: pd.Series indexed by team_abbr with a per-team cutoff datetime (typically the upcoming game_date).
    """
    feat_cols = ['pts_for_1w','pts_for_3w','pts_for_5w','pts_against_1w','pts_against_3w','pts_against_5w']
    df_hist = df_hist.sort_values(['team_abbr','game_date'])
    latest_rows = []
    for team, cutoff in cutoff_dates.items():
        sub = df_hist[(df_hist['team_abbr']==team) & (df_hist['game_date'] < cutoff)]
        if not sub.empty:
            latest_rows.append(sub.iloc[-1][['team_abbr','game_date'] + feat_cols])
    if not latest_rows:
        return pd.DataFrame(columns=['team_abbr','game_date'] + feat_cols)
    return pd.DataFrame(latest_rows)


# 3. ARIMA Persistence Models

@st.cache(ttl=86400, show_spinner=False)
def train_arima_models(data):
    arima_models = {}
    for team in data['team_abbr'].unique():
        ts = (data.loc[data['team_abbr']==team]
                  .sort_values('game_date')
                  .set_index('game_date')['points_for']
                  .dropna())
        if ts.empty:
            # default league-ish baseline if no history
            arima_models[team] = float(data['points_for'].dropna().mean()) if data['points_for'].notna().any() else 21.0
            continue
        if len(ts) < 6:
            # too short for ARIMA; use team mean as a constant forecaster
            arima_models[team] = float(ts.mean())
            continue
        try:
            model = pm.auto_arima(ts, seasonal=False, error_action='ignore', suppress_warnings=True)
            arima_models[team] = model
        except Exception:
            # fall back to mean if auto_arima fails for this series
            arima_models[team] = float(ts.mean())
    return arima_models


def forecast_arima(arima_models, team):
    model = arima_models.get(team)
    if model is None:
        return np.nan
    # numeric fallback (mean)
    if isinstance(model, (int, float, np.floating)):
        return float(model)
    # otherwise assume a pmdarima model
    try:
        return float(model.predict(n_periods=1))
    except Exception:
        return np.nan

def build_upcoming_features(upcoming_sched, hist_df, arima_models):
    # Ensure upcoming has a usable game_date
    date_col = next((c for c in ['game_date', 'gameday', 'game_datetime', 'game_time', 'start_time'] if c in upcoming_sched.columns), None)
    if date_col is None:
        raise KeyError("Upcoming schedule is missing a date column.")
    upcoming = upcoming_sched.copy()
    upcoming['game_date'] = pd.to_datetime(upcoming[date_col])

    # Prepare per-team cutoff dates (the game's date for that team)
    home_cutoff = upcoming.set_index('home_team')['game_date']
    away_cutoff = upcoming.set_index('away_team')['game_date']

    # Pull latest historical rollups prior to each upcoming game
    feat_home = latest_team_rollups_before(hist_df, home_cutoff).rename(columns={
        'team_abbr':'home_team',
        'pts_for_1w':'home_pts1w','pts_for_3w':'home_pts3w','pts_for_5w':'home_pts5w',
        'pts_against_1w':'home_pa1w','pts_against_3w':'home_pa3w','pts_against_5w':'home_pa5w'
    })
    feat_away = latest_team_rollups_before(hist_df, away_cutoff).rename(columns={
        'team_abbr':'away_team',
        'pts_for_1w':'away_pts1w','pts_for_3w':'away_pts3w','pts_for_5w':'away_pts5w',
        'pts_against_1w':'away_pa1w','pts_against_3w':'away_pa3w','pts_against_5w':'away_pa5w'
    })

    # Merge features into upcoming frame
    upcoming = upcoming.merge(feat_home.drop(columns=['game_date']), on='home_team', how='left')
    upcoming = upcoming.merge(feat_away.drop(columns=['game_date']), on='away_team', how='left')

    # ARIMA forecasts
    upcoming['home_arima'] = upcoming['home_team'].apply(lambda t: forecast_arima(arima_models, t))
    upcoming['away_arima'] = upcoming['away_team'].apply(lambda t: forecast_arima(arima_models, t))

    return upcoming


# 4. Model Training

@st.cache(ttl=86400, show_spinner=False)
def train_ml_pipeline(df):
    df = df.copy()
    # compute team-level margin on each row (not strictly used after pairing, kept for possible diagnostics)
    df['margin'] = df['points_for'] - df['points_against']

    # rolling feature columns present on team rows
    roll_cols_for = [c for c in df.columns if c.startswith('pts_for_') and c.endswith('w')]
    roll_cols_against = [c for c in df.columns if c.startswith('pts_against_') and c.endswith('w')]
    roll_cols = roll_cols_for + roll_cols_against

    # Split into the true home-row and true away-row using original schedule columns
    df_home = df[df['team_abbr'] == df['home_team']][['game_id','team_abbr','game_date','points_for','points_against'] + roll_cols].rename(
        columns={'team_abbr':'home_team','points_for':'points_for_home','points_against':'points_against_home'}
    )
    df_away = df[df['team_abbr'] == df['away_team']][['game_id','team_abbr','game_date','points_for','points_against'] + roll_cols].rename(
        columns={'team_abbr':'away_team','points_for':'points_for_away','points_against':'points_against_away'}
    )

    # Rename rolling columns with side suffixes
    df_home = df_home.rename(columns={c: f"{c}_home" for c in roll_cols})
    df_away = df_away.rename(columns={c: f"{c}_away" for c in roll_cols})

    # Pair rows per game
    pairs = df_home.merge(df_away, on='game_id', how='inner', suffixes=('_home', '_away'))
    # If no pairs, bail early with a clear error
    if pairs.empty:
        raise ValueError("No paired home/away rows found for training. Expand seasons or verify schedule columns.")

    # Add ARIMA forecasts per team for extra signal
    arima_models = train_arima_models(df)
    pairs['home_arima'] = pairs['home_team'].apply(lambda t: forecast_arima(arima_models, t))
    pairs['away_arima'] = pairs['away_team'].apply(lambda t: forecast_arima(arima_models, t))

    # Feature set
    features = [
        'pts_for_1w_home','pts_for_3w_home','pts_for_5w_home',
        'pts_against_1w_home','pts_against_3w_home','pts_against_5w_home',
        'pts_for_1w_away','pts_for_3w_away','pts_for_5w_away',
        'pts_against_1w_away','pts_against_3w_away','pts_against_5w_away',
        'home_arima','away_arima'
    ]

    # Drop any rows missing required features
    pairs = pairs.dropna(subset=features)
    if pairs.empty:
        raise ValueError("Training set is empty after dropping rows with missing features. Ensure enough games are loaded (multiple seasons) or reduce window sizes.")

    # Targets and design matrix
    y_margin = (pairs['points_for_home'] - pairs['points_against_home']) - (pairs['points_for_away'] - pairs['points_against_away'])
    X = pairs[features]

    # Train LightGBM regression for margin
    reg = LGBMRegressor(n_estimators=300, learning_rate=0.05)
    reg.fit(X, y_margin)

    # Calibrated classifier on regressed margin
    margin_pred = reg.predict(X).reshape(-1, 1)
    y_win = (y_margin > 0).astype(int)
    tscv = TimeSeriesSplit(n_splits=5)
    clf = LogisticRegression(max_iter=300)
    clf_cal = CalibratedClassifierCV(estimator=clf, cv=tscv)
    clf_cal.fit(margin_pred, y_win)

    return reg, clf_cal, features

# 5. Streamlit App UI

def main():
    st.title("FoxEdge NFL Predictor")

    seasons = st.multiselect("Select seasons for training", [2019,2020,2021,2022,2023,2024,2025], default=[2019,2020,2021,2022,2023, 2024,2025])
    data = load_nfl_data(seasons)

    # Feature engineering
    data = generate_rolling_features(data)
    # Removed add_opponent_features from training data as it is no longer used in training

    # ARIMA models & forecasts
    arima_models = train_arima_models(data)
    data['home_arima'] = data.apply(lambda row: forecast_arima(arima_models, row['team_abbr']), axis=1)  # 'home_team' does not exist here, use 'team_abbr'
    data['away_arima'] = data.apply(lambda row: np.nan, axis=1)  # Placeholder to keep columns consistent; away_arima not used in training rows

    # Train ML models
    with st.spinner("Training ML models..."):
        reg_model, clf_model, train_feats = train_ml_pipeline(data)

    # Predict upcoming week
    st.header("Predict Upcoming Games")
    week = st.number_input("Enter week number", min_value=1, max_value=18, value=1)
    season = st.selectbox("Select season", seasons)

    # Load upcoming schedule for the chosen week/season
    sched = nfl.import_schedules([season])
    # Normalize date
    date_col = next((c for c in ['game_date','gameday','game_datetime','game_time','start_time'] if c in sched.columns), None)
    if date_col is None:
        st.error("Schedule is missing a usable date column.")
        return
    sched['game_date'] = pd.to_datetime(sched[date_col])

    upcoming_sched = sched[sched['week'] == week][['game_id','game_date','home_team','away_team','season','week']].copy()

    # Build upcoming features from historical data and ARIMA
    arima_models = train_arima_models(data)
    upcoming = build_upcoming_features(upcoming_sched, data, arima_models)

    # Prepare feature matrix for prediction
    feat_cols = [
        'home_pts1w','home_pts3w','home_pts5w','home_pa1w','home_pa3w','home_pa5w',
        'away_pts1w','away_pts3w','away_pts5w','away_pa1w','away_pa3w','away_pa5w',
        'home_arima','away_arima'
    ]
    # Train ML
    with st.spinner("Training ML models..."):
        reg_model, clf_model, train_feats = train_ml_pipeline(data)

    # Align columns (in case order differs)
    X_up = upcoming[feat_cols]
    upcoming['pred_margin'] = reg_model.predict(X_up)
    upcoming['win_prob'] = clf_model.predict_proba(upcoming['pred_margin'].values.reshape(-1,1))[:,1]

    display_cols = ['home_team','away_team','game_date','pred_margin','win_prob']
    st.dataframe(upcoming[display_cols].sort_values('game_date').reset_index(drop=True))


def compute_week_predictions(season: int, week: int, train_seasons=None) -> pd.DataFrame:
    """
    Programmatic entrypoint: returns a DataFrame with columns
    ['game_id','home_team','away_team','game_date','pred_margin','win_prob'] for the requested week.
    """
    if train_seasons is None:
        # default to the three seasons prior to the requested season when available
        train_seasons = [season-3, season-2, season-1]
        train_seasons = [y for y in train_seasons if y >= 2002]
    hist = load_nfl_data(train_seasons)
    sched = nfl.import_schedules([season])
    date_col = next((c for c in ['game_date','gameday','game_datetime','game_time','start_time'] if c in sched.columns), None)
    if date_col is None:
        raise KeyError("Schedule is missing a usable date column.")
    sched['game_date'] = pd.to_datetime(sched[date_col])
    upcoming_sched = sched[sched['week'] == week][['game_id','game_date','home_team','away_team','season','week']].copy()
    arima_models = train_arima_models(hist)
    upcoming = build_upcoming_features(upcoming_sched, hist, arima_models)
    reg_model, clf_model, train_feats = train_ml_pipeline(hist)
    feat_cols = [
        'home_pts1w','home_pts3w','home_pts5w','home_pa1w','home_pa3w','home_pa5w',
        'away_pts1w','away_pts3w','away_pts5w','away_pa1w','away_pa3w','away_pa5w',
        'home_arima','away_arima'
    ]
    X_up = upcoming[feat_cols]
    upcoming['pred_margin'] = reg_model.predict(X_up)
    upcoming['win_prob'] = clf_model.predict_proba(upcoming['pred_margin'].values.reshape(-1,1))[:,1]
    return upcoming[['game_id','home_team','away_team','game_date','pred_margin','win_prob']].sort_values('game_date').reset_index(drop=True)

if __name__ == "__main__":
    main()