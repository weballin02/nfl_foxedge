#!/usr/bin/env python3

import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as ndp

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression

# Configuration Constants
SUPPORTED_MARKETS = [
    "player_pass_attempts",
    "player_receptions",
    "player_rush_attempts"
]

DEFAULT_N_SIMULATIONS = 10000
HALF_LIFE_DAYS = 28
DECAY_RATE = np.log(2) / HALF_LIFE_DAYS

EDGE_THRESHOLD_BASE = 0.04
VOLATILITY_RATIO_THRESHOLD = 1.25
Z_MIN = 0.5

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

def odds_to_prob(odds):
    try:
        o = float(odds)
        if o > 0:
            return 100.0 / (o + 100.0)
        else:
            return -o / (-o + 100.0)
    except:
        return np.nan

@st.cache_data(show_spinner=True)
def train_models_with_bayes_and_momentum(pbp_df):
    current_ts = pd.Timestamp(datetime.date.today())
    hist_means = {}
    raw_means = {}
    ensemble_models = {}

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
            raw_means[market] = {}
            ensemble_models[market] = None
            continue

        counts = (
            events.groupby([player_col, 'game_id', 'game_date'])
            .size().reset_index(name='count')
        )

        # Convert game_date to datetime to fix TypeError
        counts['game_date'] = pd.to_datetime(counts['game_date'], errors='coerce')

        counts['days_diff'] = (current_ts - counts['game_date']).dt.days

        raw_lambda = counts.groupby(player_col)['count'].mean()

        counts['weight'] = np.exp(-DECAY_RATE * counts['days_diff'])
        counts['wprod'] = counts['count'] * counts['weight']
        agg = counts.groupby(player_col)[['wprod', 'weight']].sum()
        weighted_lambda = (agg['wprod'] / agg['weight'])

        counts_per_player = counts.groupby(player_col).size()
        global_lambda = counts['count'].mean()
        alpha = 5  # shrinkage for Empirical Bayes

        shrunk_lambda = (weighted_lambda * counts_per_player + alpha * global_lambda) / (counts_per_player + alpha)

        recent_counts = counts[counts['days_diff'] <= 21]
        recent_lambda = recent_counts.groupby(player_col)['count'].mean()
        # Blend recency momentum
        final_lambda = 0.7 * shrunk_lambda + 0.3 * recent_lambda.reindex(shrunk_lambda.index).fillna(shrunk_lambda)

        hist_means[market] = final_lambda.to_dict()
        raw_means[market] = raw_lambda.to_dict()

        data = []
        labels = []
        for p in final_lambda.index:
            x = [
                raw_lambda.get(p, global_lambda),
                weighted_lambda.get(p, global_lambda),
                final_lambda.get(p, global_lambda),
                counts_per_player.get(p, 0)
            ]
            y = final_lambda.get(p, global_lambda)
            data.append(x)
            labels.append(y)

        if len(data) >= 10:
            X = np.array(data)
            y = np.array(labels)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            ensemble_models[market] = rf
        else:
            ensemble_models[market] = None

    return hist_means, raw_means, ensemble_models

def simulate_negative_binomial(lam, n_sims=DEFAULT_N_SIMULATIONS, overdispersion=1.2):
    size = overdispersion
    prob = size / (size + lam)
    return np.random.negative_binomial(size, prob, n_sims)

def compute_edges(df, hist_means, raw_means, ensemble_models, calibrators,
                  edge_threshold=EDGE_THRESHOLD_BASE,
                  vol_ratio_threshold=VOLATILITY_RATIO_THRESHOLD,
                  z_min=Z_MIN):

    results = []
    for _, row in df.iterrows():
        try:
            market = row['market_key']
            player = row['player']
            lam_base = hist_means.get(market, {}).get(player, np.nan)

            if np.isnan(lam_base) or lam_base <= 0:
                raise ValueError("Invalid base rate")

            ensemble = ensemble_models.get(market)
            if ensemble:
                r = raw_means.get(market, {}).get(player, lam_base)
                lam_adj = ensemble.predict([[r, lam_base, lam_base, 1]])[0]
            else:
                lam_adj = lam_base

            if lam_adj > 10:
                sim = np.random.normal(lam_adj, np.sqrt(lam_adj), DEFAULT_N_SIMULATIONS)
                sim = np.clip(sim, 0, None)
            else:
                sim = simulate_negative_binomial(lam_adj)

            mu = sim.mean()
            sigma = sim.std() + 1e-9

            line_used = row.get('line', np.nan)
            if pd.isna(line_used):
                raise ValueError("No line")

            p_over = np.mean(sim > line_used)
            p_under = np.mean(sim < line_used)

            cal = calibrators.get(market)
            if cal:
                if isinstance(cal, LogisticRegression):
                    p_over = cal.predict_proba([[p_over]])[0,1]
                    p_under = cal.predict_proba([[p_under]])[0,1]
                elif isinstance(cal, IsotonicRegression):
                    p_over = cal.predict(np.array([p_over]))[0]
                    p_under = cal.predict(np.array([p_under]))[0]

            z_vs_line = (mu - line_used) / sigma

            side_csv = row.get('side')
            if isinstance(side_csv, str) and side_csv in ('Over', 'Under'):
                side = side_csv
                p_side = p_over if side == 'Over' else p_under
                implied_prob_used = row.get('implied_prob', np.nan)
                edge_prob = p_side - implied_prob_used
                edge_pct = edge_prob * 100

                vol_ok = (sigma / max(mu, 1e-9)) <= vol_ratio_threshold
                z_ok = abs(z_vs_line) >= z_min
                edge_ok = edge_prob >= edge_threshold * (sigma / mu)
                rec = side if (edge_ok and vol_ok and z_ok) else 'No Action'
            else:
                side = 'Unknown'
                p_side = np.nan
                edge_prob = np.nan
                edge_pct = np.nan
                rec = 'No Action'

            results.append({
                **row.to_dict(),
                'lambda': lam_adj,
                'model_mean': mu,
                'model_std': sigma,
                'line': line_used,
                'delta_vs_line': mu - line_used,
                'z_vs_line': z_vs_line,
                'edge_prob': edge_prob,
                'edge_pct': edge_pct,
                'recommendation': rec,
                'side': side,
                'p_over': p_over,
                'p_under': p_under,
                'p_side': p_side,
                'pass_edge': edge_ok if 'edge_ok' in locals() else False,
                'pass_vol': vol_ok if 'vol_ok' in locals() else False,
                'pass_z': z_ok if 'z_ok' in locals() else False,
                'implied_prob_used': implied_prob_used
            })
        except Exception:
            results.append({
                **row.to_dict(),
                'lambda': np.nan,
                'model_mean': np.nan,
                'model_std': np.nan,
                'line': np.nan,
                'delta_vs_line': np.nan,
                'z_vs_line': np.nan,
                'edge_prob': np.nan,
                'edge_pct': np.nan,
                'recommendation': 'No Action',
                'side': np.nan,
                'p_over': np.nan,
                'p_under': np.nan,
                'p_side': np.nan,
                'pass_edge': False,
                'pass_vol': False,
                'pass_z': False,
                'implied_prob_used': np.nan
            })
    return pd.DataFrame(results)

@st.cache_data(show_spinner=True)
def train_logistic_calibrators(odds_hist, counts_frames):
    calibrators = {}
    for market in SUPPORTED_MARKETS:
        df_counts = counts_frames.get(market, pd.DataFrame())
        if df_counts.empty or odds_hist.empty:
            calibrators[market] = None
            continue
        merged = odds_hist[odds_hist['market_key'] == market].merge(
            df_counts,
            on=['player_fullkey', 'game_id'],
            how='inner'
        )
        if merged.empty:
            calibrators[market] = None
            continue
        merged['actual_over'] = merged.apply(lambda r: int(r['count'] > r['line']), axis=1)
        preds = merged['implied_prob'].fillna(0).values.reshape(-1, 1)
        labels = merged['actual_over'].values
        if len(labels) < 50:
            calibrators[market] = None
            continue
        lr = LogisticRegression()
        lr.fit(preds, labels)
        calibrators[market] = lr
    return calibrators

def main():
    st.title("Enhanced NFL Player Prop Projections with Date Fix")

    with st.spinner("Loading play-by-play data..."):
        try:
            current_year = datetime.date.today().year
            pbp = pd.concat([ndp.import_pbp_data([y]) for y in range(2020, current_year+1)], ignore_index=True)
            if pbp.empty:
                st.error("Play-by-play data is empty.")
                return
        except Exception as e:
            st.error(f"Failed to load play-by-play  {e}")
            return

    hist_means, raw_means, ensemble_models = train_models_with_bayes_and_momentum(pbp)

    st.write("Models trained successfully.")

    uploaded_file = st.file_uploader("Upload Odds CSV", type=["csv"])
    if uploaded_file:
        # Implement loading and processing odds CSV logic as needed
        try:
            odds_df = pd.read_csv(uploaded_file)
            odds_df['event_date'] = pd.to_datetime(odds_df.get('event_date', pd.NaT), errors='coerce')  # Safe conversion
            # Assuming normalized columns are present or preprocessed similarly
            calibrators = {}  # Load or train calibrators here if relevant
            results_df = compute_edges(odds_df, hist_means, raw_means, ensemble_models, calibrators)
            st.dataframe(results_df)
        except Exception as e:
            st.error(f"Error processing odds file: {e}")

if __name__ == "__main__":
    main()
