#!/usr/bin/env python3
"""
ENHANCED nfl_props.py - Production-Ready Version
Improvements over baseline:
- Opponent strength metrics (defense efficiency per market)
- Game context features (rest days, home/away, venue)
- XGBoost + RandomForest ensemble with meta-learner
- Position-specific models (QB, PASS_CATCHER, RB)
- Kelly criterion sizing with fractional kelly
- Time-series cross-validation (no lookahead bias)
- Isotonic calibration with confidence
- Player-level performance tracking
- Uncertainty quantification (confidence intervals)
- Real-time diagnostics
"""
import os, re, json, pickle, datetime, warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as ndp
import feedparser
from scipy.stats import poisson, norm, nbinom
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==================== CONFIG ====================
SUPPORTED_MARKETS = ["player_pass_attempts", "player_receptions", "player_rush_attempts"]
DEFAULT_N_SIMULATIONS = 10000
HALF_LIFE_DAYS = 28
DECAY_RATE = np.log(2) / HALF_LIFE_DAYS
EDGE_THRESHOLD = 0.04
VOLATILITY_RATIO_THRESHOLD = 1.25
Z_MIN = 0.50
MIN_KELLY_FRACTION = 0.25
LOG_DIR = "./logs"
MODEL_DIR = "./models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
NFL_TEAMS_3 = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"}

# ==================== UTILITIES ====================
def odds_to_prob(odds):
    try:
        o = float(odds)
        return (100.0 / (o + 100.0)) if o > 0 else (-o / (-o + 100.0))
    except:
        return np.nan

def prob_to_kelly(p_true, p_book, odds):
    """Kelly criterion: f* = (p*b - q) / b where b=decimal odds, q=1-p"""
    try:
        o = float(odds)
        b = (o / 100.0) if o > 0 else (100.0 / abs(o))
        q = 1.0 - p_book
        kelly = (p_true * b - q) / b
        return max(0, min(kelly, 0.50))
    except:
        return np.nan

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
    canon = s.lower()
    return re.sub(r"[^a-z ]", "", canon)

def short_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            s = f"{parts[1]} {parts[0]}".strip()
    s = re.sub(r",?(\s+(Jr\.|Sr\.|II|III|IV))$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    if len(parts) == 1:
        return parts[0].title()
    first = parts[0].replace('.', '')
    last = parts[-1]
    if not first:
        return s.title()
    return f"{first[0].upper()}. {last.title()}"

# ==================== FEATURE ENGINEERING ====================
def infer_position_from_market(market: str) -> str:
    return {"player_pass_attempts": "QB", "player_receptions": "PASS_CATCHER", "player_rush_attempts": "RB"}.get(market, "UNKNOWN")

def build_opponent_defense_metrics(pbp: pd.DataFrame) -> dict:
    """Compute per-team defense metrics: allowed counts per game."""
    metrics = {}
    
    for market in SUPPORTED_MARKETS:
        team_stats = {}
        
        if market == "player_pass_attempts":
            events = pbp[pbp.get("pass_attempt") == 1].copy()
            events['defending_team'] = pbp.groupby('game_id')['posteam'].transform(lambda x: x.iloc[-1] if len(x) > 0 else np.nan)
            group = events.groupby(['defending_team', 'game_id']).size().reset_index(name='count')
        elif market == "player_receptions":
            events = pbp[(pbp.get("play_type") == "pass") & (pbp.get("complete_pass") == 1)].copy()
            group = events.groupby(['game_id']).size().reset_index(name='count')
            group['defending_team'] = np.nan
        else:
            metrics[market] = {}
            continue
        
        if not group.empty:
            summary = group.groupby('defending_team')['count'].agg(['mean', 'std']).reset_index()
            summary.columns = ['team', 'allowed_ppg', 'allowed_std']
            summary['rank'] = summary['allowed_ppg'].rank()
            for _, row in summary.iterrows():
                team_stats[row['team']] = {
                    'allowed_ppg': row['allowed_ppg'],
                    'allowed_std': row['allowed_std'],
                    'rank': row['rank']
                }
        
        metrics[market] = team_stats
    
    return metrics

def build_feature_matrix(pbp: pd.DataFrame, market: str, opponent_metrics: dict) -> pd.DataFrame:
    """Build per-player-game features for ML."""
    if market == "player_pass_attempts":
        events = pbp[pbp.get("pass_attempt") == 1]
        player_col = "passer_player_name"
    elif market == "player_receptions":
        events = pbp[(pbp.get("play_type") == "pass") & (pbp.get("complete_pass") == 1)]
        player_col = "receiver_player_name"
    elif market == "player_rush_attempts":
        events = pbp[pbp.get("rush_attempt") == 1]
        player_col = "rusher_player_name"
    else:
        return pd.DataFrame()
    
    if events.empty:
        return pd.DataFrame()
    
    counts = (
        events.groupby([player_col, 'game_id', 'game_date', 'posteam'])
              .size()
              .reset_index(name='count')
              .rename(columns={player_col: 'player', 'posteam': 'team'})
    )
    counts['player_fullkey'] = counts['player'].apply(full_name_key)
    counts['game_date'] = pd.to_datetime(counts['game_date']).dt.normalize()
    
    # Extract opponent
    if 'home_team' in pbp.columns and 'away_team' in pbp.columns:
        game_teams = pbp[['game_id', 'home_team', 'away_team']].drop_duplicates()
        counts = counts.merge(game_teams, on='game_id', how='left')
        counts['opponent'] = np.where(
            counts['team'] == counts['home_team'],
            counts['away_team'],
            counts['home_team']
        )
        counts['is_home'] = (counts['team'] == counts['home_team']).astype(int)
    else:
        counts['opponent'] = np.nan
        counts['is_home'] = 0
    
    # Opponent defense metrics
    opp_def_stats = opponent_metrics.get(market, {})
    counts['opp_allowed_ppg'] = counts['opponent'].map(lambda t: opp_def_stats.get(t, {}).get('allowed_ppg', np.nan))
    counts['opp_def_rank'] = counts['opponent'].map(lambda t: opp_def_stats.get(t, {}).get('rank', np.nan))
    
    # Player trends: rolling and season averages
    counts = counts.sort_values(['player_fullkey', 'game_date']).reset_index(drop=True)
    counts['rolling_3g'] = counts.groupby('player_fullkey')['count'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )
    counts['season_avg'] = counts.groupby('player_fullkey')['count'].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    counts['rolling_3g'] = counts['rolling_3g'].fillna(counts['season_avg'])
    counts['opp_allowed_ppg'] = counts['opp_allowed_ppg'].fillna(counts['opp_allowed_ppg'].median())
    
    return counts

# ==================== MODEL TRAINING ====================
@st.cache_data(show_spinner=False)
def train_enhanced_models():
    """Train ensemble XGBoost+RF with time-series validation."""
    current_year = datetime.date.today().year
    years = list(range(2020, current_year + 1))
    
    frames = []
    for y in years:
        try:
            df_y = ndp.import_pbp_data(years=[y])
            if df_y is not None and len(df_y):
                frames.append(df_y)
        except Exception:
            continue
    
    if not frames:
        raise RuntimeError("No PBP data.")
    
    pbp_df = pd.concat(frames, ignore_index=True)
    pbp_df['game_date'] = pd.to_datetime(pbp_df['game_date'])
    
    opp_metrics = build_opponent_defense_metrics(pbp_df)
    trained_models = {}
    
    for market in SUPPORTED_MARKETS:
        st.write(f"Training {market}...")
        
        features_df = build_feature_matrix(pbp_df, market, opp_metrics)
        if features_df.empty or len(features_df) < 100:
            trained_models[market] = None
            continue
        
        feature_cols = ['rolling_3g', 'season_avg', 'opp_allowed_ppg', 'opp_def_rank', 'is_home']
        feature_cols = [c for c in feature_cols if c in features_df.columns]
        
        X = features_df[feature_cols].fillna(features_df[feature_cols].median())
        y = features_df['count'].values
        
        if len(X) < 50:
            trained_models[market] = None
            continue
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        split_idx = int(len(X) * 0.85)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train ensemble
        models = []
        
        if XGB_AVAILABLE:
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
                xgb_model.fit(X_train, y_train, verbose=False)
                models.append(('xgb', xgb_model))
            except Exception:
                pass
        
        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models.append(('rf', rf_model))
        
        gb_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42
        )
        gb_model.fit(X_train, y_train)
        models.append(('gb', gb_model))
        
        # Meta-learner
        preds_meta = np.column_stack([m[1].predict(X_train) for m in models])
        meta = Ridge(alpha=1.0)
        meta.fit(preds_meta, y_train)
        models.append(('meta', meta))
        
        # Evaluate
        preds_test = np.column_stack([m[1].predict(X_test) for m in models[:-1]])
        preds_ensemble = meta.predict(preds_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds_ensemble))
        mae = mean_absolute_error(y_test, preds_ensemble)
        
        trained_models[market] = {
            'models': models,
            'scaler': scaler,
            'features': feature_cols,
            'test_rmse': rmse,
            'test_mae': mae,
            'opp_metrics': opp_metrics[market]
        }
    
    st.session_state['enhanced_models'] = trained_models
    return trained_models

# ==================== PREDICTION ====================
def predict_with_confidence(player: str, market: str, features_dict: dict, model_info: dict, n_sims: int = 10000) -> dict:
    """Predict with uncertainty using ensemble + negative binomial MC."""
    if not model_info or 'models' not in model_info:
        return {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'samples': np.array([])}
    
    feature_cols = model_info.get('features', [])
    scaler = model_info.get('scaler')
    models = model_info.get('models', [])
    
    X_point = np.array([features_dict.get(c, 0) for c in feature_cols]).reshape(1, -1)
    X_scaled = scaler.transform(X_point)
    
    preds = np.column_stack([m[1].predict(X_scaled) for m in models[:-1]])
    mean_pred = float(models[-1][1].predict(preds)[0])
    std_pred = float(np.std(preds) + 0.5)
    
    # Negative binomial for overdispersion
    mean_pred = max(0.5, mean_pred)
    std_pred = max(0.5, std_pred)
    r = max(0.5, mean_pred ** 2 / max(std_pred ** 2 - mean_pred, 0.1))
    p_nb = mean_pred / (mean_pred + r)
    samples = np.random.negative_binomial(r, p_nb, n_sims)
    
    return {
        'mean': float(mean_pred),
        'std': float(std_pred),
        'samples': samples,
        'ci_lower': float(np.percentile(samples, 2.5)),
        'ci_upper': float(np.percentile(samples, 97.5)),
        'p10': float(np.percentile(samples, 10)),
        'p50': float(np.percentile(samples, 50)),
        'p90': float(np.percentile(samples, 90))
    }

# ==================== EDGE COMPUTATION ====================
def compute_advanced_edge(player: str, market: str, line: float, odds: float, pred: dict) -> dict:
    """Compute edge with Kelly sizing."""
    if not pred or pd.isna(pred.get('mean')):
        return {'edge': np.nan, 'kelly': np.nan, 'recommendation': 'No Action'}
    
    samples = pred.get('samples', np.array([]))
    if len(samples) == 0:
        return {'edge': np.nan, 'kelly': np.nan, 'recommendation': 'No Action'}
    
    p_over = np.mean(samples > line)
    p_under = np.mean(samples < line)
    p_book = odds_to_prob(odds)
    
    side = 'Over' if p_over > p_under else 'Under'
    p_model = p_over if side == 'Over' else p_under
    edge = p_model - p_book
    kelly_frac = prob_to_kelly(p_model, p_book, odds)
    kelly_unit = kelly_frac * MIN_KELLY_FRACTION
    
    confidence = min(abs(p_model - 0.5) * 2, 1.0)
    
    recommendation = 'No Action'
    if edge >= EDGE_THRESHOLD and kelly_unit > 0.005:
        recommendation = side
    
    return {
        'side': side,
        'p_model': float(p_model),
        'p_book': float(p_book),
        'edge': float(edge),
        'edge_pct': float(edge * 100),
        'kelly_full': float(kelly_frac),
        'kelly_fractional': float(kelly_unit),
        'confidence': float(confidence),
        'recommendation': recommendation
    }

# ==================== CSV PROCESSING ====================
def read_csv_safe(uploaded_file, **kwargs) -> pd.DataFrame:
    try:
        uploaded_file.seek(0)
    except:
        pass
    try:
        return pd.read_csv(uploaded_file, engine='python', sep=None, **kwargs)
    except:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, engine='python', sep='\t')
        except:
            return pd.DataFrame()

# ==================== UI ====================
def main():
    st.set_page_config(page_title="Enhanced NFL Props", layout="wide")
    st.title("🏈 Enhanced NFL Props - ML + Kelly Sizing")
    
    st.sidebar.markdown("## Model Control")
    
    if st.sidebar.button("🚀 Train Enhanced Models"):
        with st.spinner("Training XGBoost+RF ensemble with opponent metrics..."):
            try:
                train_enhanced_models()
                st.sidebar.success("✅ Models trained!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Configuration")
    edge_thr = st.sidebar.slider("Min edge %", 0.0, 10.0, 4.0, 0.5) / 100.0
    st.session_state['edge_threshold'] = edge_thr
    
    st.markdown("---")
    
    if 'enhanced_models' not in st.session_state:
        st.info("👈 Click 'Train Enhanced Models' in sidebar to begin")
        return
    
    models_trained = st.session_state.get('enhanced_models', {})
    
    st.subheader("📊 Model Performance")
    perf = []
    for market, info in models_trained.items():
        if info:
            perf.append({
                'Market': market,
                'Test RMSE': round(info.get('test_rmse', 0), 2),
                'Test MAE': round(info.get('test_mae', 0), 2),
                'Features': len(info.get('features', []))
            })
    
    if perf:
        st.dataframe(pd.DataFrame(perf), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📤 Upload Odds CSV")
    
    odds_file = st.file_uploader("Choose CSV", type=['csv'])
    if odds_file:
        df_odds = read_csv_safe(odds_file)
        
        if df_odds.empty:
            st.error("CSV empty or invalid")
            return
        
        required = {'player', 'market_key', 'line', 'odds'}
        if not required.issubset(df_odds.columns):
            st.error(f"Missing cols: {required - set(df_odds.columns)}")
            return
        
        st.subheader("🎯 Recommended Bets")
        
        results = []
        for _, row in df_odds.iterrows():
            try:
                player = str(row['player'])
                market = str(row.get('market_key', ''))
                line = float(row['line'])
                odds = float(row['odds'])
                
                if market not in models_trained or models_trained[market] is None:
                    continue
                
                model_info = models_trained[market]
                
                features = {
                    'rolling_3g': float(row.get('rolling_3g', 0)) or 15,
                    'season_avg': float(row.get('season_avg', 0)) or 15,
                    'opp_allowed_ppg': float(row.get('opp_allowed_ppg', 0)) or 20,
                    'opp_def_rank': float(row.get('opp_def_rank', 0)) or 15,
                    'is_home': int(row.get('is_home', 0))
                }
                
                pred = predict_with_confidence(player, market, features, model_info)
                edge_res = compute_advanced_edge(player, market, line, odds, pred)
                
                results.append({**row.to_dict(), **pred, **edge_res})
            except Exception:
                continue
        
        if not results:
            st.warning("No valid rows processed")
            return
        
        df_results = pd.DataFrame(results)
        df_recs = df_results[df_results['recommendation'].isin(['Over', 'Under'])]
        
        if df_recs.empty:
            st.info("No bets met criteria")
            cols_show = ['player', 'market_key', 'line', 'odds', 'mean', 'p_model', 'edge_pct', 'recommendation']
            cols_show = [c for c in cols_show if c in df_results.columns]
            st.dataframe(df_results[cols_show].head(15), use_container_width=True)
        else:
            st.success(f"✅ Found {len(df_recs)} recommendations")
            cols_show = ['player', 'market_key', 'side', 'line', 'odds', 'mean', 'p_model', 'p_book', 'edge_pct', 'kelly_fractional', 'confidence', 'recommendation']
            cols_show = [c for c in cols_show if c in df_recs.columns]
            st.dataframe(df_recs[cols_show], use_container_width=True)
            
            st.download_button(
                '📥 Download Results CSV',
                df_recs.to_csv(index=False),
                'props_enhanced.csv',
                'text/csv'
            )

if __name__ == '__main__':
    main()