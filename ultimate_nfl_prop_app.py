
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate NFL Prop Pricer | Single-file Streamlit app

Design goals (consolidated from your scripts):
- Two engines you can toggle:
  1) Market-Fit (odds-only): de-vig two-way, fit parametric distribution per player+market+line (Normal for counts/yds, Poisson for TDs).
  2) Historical EB (optional): recency-weighted counts per player from PBP, EB shrink to market mean, optional isotonic calibration.
- Guardrails: caps on TD markets, Kelly cap, z-score and volatility checks.
- Line unit rescaling and robust CSV helpers.
- Unified odds conversion and Kelly utilities.
- Canonical keys for merge/join.

Requires: streamlit, pandas, numpy, scipy, sklearn, nfl_data_py, feedparser
Run:
  streamlit run ultimate_nfl_prop_app.py
"""

from __future__ import annotations
import os, io, re, math, json, logging, datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import norm, poisson
from scipy.optimize import minimize

# Optional: historical context
try:
    import nfl_data_py as ndp
    NFL_DATA_OK = True
except Exception:
    NFL_DATA_OK = False

from sklearn.isotonic import IsotonicRegression

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ultimate_nfl_prop")

# -------------------- Config --------------------
SUPPORTED_MARKETS = [
    "player_pass_attempts",
    "player_receptions",
    "player_rush_attempts",
    "player_pass_completions",
    "player_pass_yds",
    "player_reception_yds",
    "player_rush_yds",
    "player_pass_tds"
]
DEFAULT_N_SIMULATIONS = 10000

HALF_LIFE_DAYS = 28
DECAY_RATE = np.log(2) / HALF_LIFE_DAYS

# EV thresholds per market
DEFAULT_PROP_EV_MIN = {
    "player_pass_attempts": 0.010,
    "player_pass_completions": 0.010,
    "player_pass_tds": 0.012,
    "player_pass_yds": 0.010,
    "player_reception_yds": 0.010,
    "player_receptions": 0.010,
    "player_rush_attempts": 0.010,
    "player_rush_yds": 0.010,
}

# Kelly and limits
KELLY_CAP = 0.015
KELLY_FRACTION = 0.40

# Guardrails
EDGE_THRESHOLD_DEFAULT = 0.05
VOLATILITY_RATIO_THRESHOLD_DEFAULT = 1.25
Z_MIN_DEFAULT = 0.50

# Caps for risky markets
TD_OVER_MAX_P = 0.80   # never accept fair_prob(Ov 0.5) > 0.80 pre-calibration
TD_OVER_HARD_MAX = 0.85

# -------------------- Odds utilities --------------------
def american_to_prob(odds: int | float) -> float:
    try:
        odds = float(odds)
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)
    except Exception:
        return np.nan

def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds >= 100:
        return 1 + odds / 100.0
    if odds <= -100:
        return 1 + 100.0 / abs(odds)
    return 1.0

def prob_to_american(p: float) -> int:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    if p > 0.5:
        return -int(round(p/(1-p)*100))
    return int(round((1-p)/p*100))

def remove_vig_two_way(p_over_raw: float, p_under_raw: float) -> Tuple[float,float]:
    s = p_over_raw + p_under_raw
    if s <= 0:
        return 0.5, 0.5
    return p_over_raw/s, p_under_raw/s

def kelly_fraction_decimal(p: float, price_decimal: float, frac: float = KELLY_FRACTION) -> float:
    b = price_decimal - 1.0
    if b <= 0:
        return 0.0
    f_full = (p * (b + 1) - 1) / b
    out = max(0.0, f_full * frac)
    return 0.0 if not np.isfinite(out) else float(out)

# -------------------- CSV helpers --------------------
def read_csv_safe(uploaded_file, **kwargs) -> pd.DataFrame:
    defaults = dict(engine="python", sep=None)
    defaults.update(kwargs or {})
    try:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return pd.read_csv(uploaded_file, **defaults)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except pd.errors.ParserError:
        try:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            return pd.read_csv(uploaded_file, engine="python", sep="\t")
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

REQUIRED_COLS = ["game_id","commence_time","in_play","bookmaker","last_update",
                 "home_team","away_team","market","label","description","price","point"]

def load_props_csv(raw_file) -> pd.DataFrame:
    df = read_csv_safe(raw_file)
    if df is None or df.empty:
        return pd.DataFrame()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.warning(f"CSV missing columns: {missing}")
        return pd.DataFrame()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    df["description"] = df["description"].astype(str).str.strip()
    df["price"] = df["price"].apply(lambda x: int(str(x).replace('−','-')))
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    # Only two-way O/U rows
    df = df[df["label"].isin(["over","under"])].copy()
    return df

# -------------------- Priors --------------------
def prior_std_normal(market: str, line: float) -> float:
    m = str(market).strip().lower()
    line = float(line) if pd.notna(line) else 1.0
    if m in ("player_pass_yds","player_reception_yds","player_rush_yds"):
        return max(10.0, 0.40 * (abs(line)+1.0)**0.6)
    if m in ("player_pass_attempts","player_pass_completions","player_rush_attempts","player_receptions"):
        return max(1.5, 0.65 * (abs(line)+1.0)**0.5)
    return max(1.0, 0.5 * (abs(line)+1.0)**0.5)

def prior_lambda_poisson(line: float) -> float:
    line = float(line) if pd.notna(line) else 0.5
    return max(0.2, 0.90 * max(0.5, line))

# -------------------- Market-fit parametric model --------------------
def fit_normal_from_points(points: List[Tuple[float,float]]) -> Tuple[float,float,str]:
    xs = np.array([x for x,_ in points], dtype=float)
    ys = np.array([y for _,y in points], dtype=float)
    x0_mu = float(np.median(xs))
    x0_std = float(prior_std_normal("normal", max(1.0, abs(x0_mu))))
    def loss(theta):
        mu, sigma = theta
        sigma = max(1e-3, sigma)
        preds = 1 - norm.cdf(xs, loc=mu, scale=sigma)
        return np.nanmean((preds - ys)**2)
    res = minimize(loss, x0=np.array([x0_mu, x0_std]), method="Nelder-Mead")
    mu = float(res.x[0]); sigma = float(abs(res.x[1]) if np.isfinite(res.x[1]) else x0_std)
    return mu, max(1e-3, sigma), ("normal")

def fit_poisson_from_points(points: List[Tuple[float,float]]) -> Tuple[float, str]:
    # Solve for lambda that best matches tail probs vs half-line threshold approximation
    xs = np.array([x for x,_ in points], dtype=float)
    ys = np.array([y for _,y in points], dtype=float)
    x0 = float(max(0.2, np.median(xs)))
    def loss(lmbda):
        lmbda = float(max(0.05, lmbda))
        preds = 1 - poisson.cdf(np.floor(xs), lmbda)
        return np.nanmean((preds - ys)**2)
    res = minimize(lambda l: loss(l[0]), x0=np.array([x0]), method="Nelder-Mead")
    lam = float(max(0.05, res.x[0]))
    return lam, ("poisson")

def price_group(df_group: pd.DataFrame, market: str, ev_min: float) -> pd.DataFrame:
    # Build per-line over probability from de-vigged pairs per book
    # points: (line, q_over_fair)
    rows = []
    # group by line with both sides present
    pairs = {}
    for _, r in df_group.iterrows():
        side = r["label"]
        line = float(r["point"])
        imp = american_to_prob(r["price"])
        key = (line, r["bookmaker"])
        pairs.setdefault(key, {})[side] = imp
    points = []
    for (line, _), d in pairs.items():
        if "over" in d and "under" in d:
            p_over, p_under = remove_vig_two_way(d["over"], d["under"])
            points.append((line, p_over))
    # Fit a distribution
    model_params = {}
    fit_note = "market-fit"
    if market == "player_pass_tds":
        if points:
            lam, model_name = fit_poisson_from_points(points)
        else:
            lam = prior_lambda_poisson(df_group["point"].median())
            model_name = "poisson_prior"
        model_params = {"lambda": lam}
    else:
        if points:
            mu, sigma, model_name = fit_normal_from_points(points)
        else:
            mu = float(np.nanmedian(df_group["point"])) if df_group["point"].notna().any() else 1.0
            sigma = prior_std_normal(market, mu)
            model_name = "normal_prior"
        model_params = {"mu": mu, "sigma": sigma}
    # Price every row
    for _, r in df_group.iterrows():
        side = r["label"]
        line = float(r["point"])
        price = int(r["price"])
        if market == "player_pass_tds":
            lam = model_params["lambda"]
            # cap insane priors
            if line <= 0.5:
                # conservative cap
                p_over = 1 - poisson.cdf(np.floor(line), lam)
                p_over = min(p_over, TD_OVER_MAX_P)
            else:
                p_over = 1 - poisson.cdf(np.floor(line), lam)
            p_under = 1 - p_over
        else:
            mu = model_params["mu"]; sigma = model_params["sigma"]
            z = (line - mu) / max(1e-9, sigma)
            p_over = 1 - norm.cdf(line, loc=mu, scale=sigma)
            p_under = norm.cdf(line, loc=mu, scale=sigma)
        p_side = p_over if side == "over" else p_under
        # guardrails for TD overrates
        if market == "player_pass_tds" and line <= 0.5 and side == "over":
            p_side = min(p_side, TD_OVER_HARD_MAX)
        edge = float(p_side - american_to_prob(price))
        dec = american_to_decimal(price)
        ev = float(p_side*(dec-1.0) - (1 - p_side))
        kelly = min(KELLY_CAP, kelly_fraction_decimal(p_side, dec))
        fair_price = prob_to_american(p_side)
        rec = "BET" if (ev > 0 and edge >= ev_min) else "PASS"
        rows.append({
            "market": market,
            "player": r["description"],
            "selection": side.capitalize(),
            "line": line,
            "book": r["bookmaker"],
            "book_price": price,
            "fair_prob": round(float(p_side), 4),
            "fair_price": int(fair_price),
            "edge": round(float(edge), 4),
            "ev": round(float(ev), 4),
            "kelly": round(float(kelly), 4),
            "rec": rec,
            "notes": f"{fit_note} {model_params}"
        })
    return pd.DataFrame(rows)

def market_fit_pipeline(df_props: pd.DataFrame) -> pd.DataFrame:
    out_frames = []
    for (market, player), sub in df_props.groupby(["market","description"]):
        if market not in SUPPORTED_MARKETS:
            continue
        ev_min = DEFAULT_PROP_EV_MIN.get(market, 0.01)
        priced = price_group(sub, market, ev_min=ev_min)
        out_frames.append(priced)
    if not out_frames:
        return pd.DataFrame()
    out = pd.concat(out_frames, ignore_index=True)
    out["rec"] = pd.Categorical(out["rec"], categories=["BET","PASS"], ordered=True)
    out = out.sort_values(["rec","ev"], ascending=[True, False]).reset_index(drop=True)
    return out

# -------------------- Historical EB (optional) --------------------
@st.cache_data(show_spinner=False)
def load_pbp_years(start_year: int, end_year: int) -> pd.DataFrame:
    if not NFL_DATA_OK:
        return pd.DataFrame()
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
        return pd.DataFrame(columns=['player','game_id','game_date','count'])
    if market == 'player_pass_attempts':
        events = pbp[pbp.get('pass_attempt') == 1]; col = 'passer_player_name'
    elif market == 'player_receptions':
        events = pbp[(pbp.get('play_type') == 'pass') & (pbp.get('complete_pass') == 1)]; col = 'receiver_player_name'
    elif market == 'player_rush_attempts':
        events = pbp[pbp.get('rush_attempt') == 1]; col = 'rusher_player_name'
    else:
        return pd.DataFrame(columns=['player','game_id','game_date','count'])
    counts = events.groupby([col, 'game_id', 'game_date']).size().reset_index(name='count')
    counts = counts.rename(columns={col:'player'})
    return counts

def eb_shrink_lambda(player_mu: float, market_mu: float, n_games: int, tau: float = 6.0) -> float:
    w = n_games / float(n_games + tau)
    return w*player_mu + (1.0 - w)*market_mu

@st.cache_data(show_spinner=False)
def train_hist_means(start_year: int, end_year: int) -> Dict[str, Dict[str, float]]:
    pbp = load_pbp_years(start_year, end_year)
    if pbp.empty:
        return {}
    hist_means = {}
    for market in ["player_pass_attempts","player_receptions","player_rush_attempts"]:
        counts = _counts_from_pbp(pbp, market)
        if counts.empty:
            hist_means[market] = {}
            continue
        now = pd.Timestamp(datetime.date.today())
        counts['days_diff'] = (now - counts['game_date']).dt.days
        counts['weight'] = np.exp(-DECAY_RATE * counts['days_diff'])
        counts['wprod'] = counts['count'] * counts['weight']
        agg = counts.groupby('player')[['wprod','weight']].sum()
        weighted = (agg['wprod'] / agg['weight']).to_dict()
        hist_means[market] = weighted
    return hist_means

def apply_hist_prior(df_priced: pd.DataFrame, hist_means: Dict[str, Dict[str, float]], blend: float = 0.3) -> pd.DataFrame:
    """Blend market-fit fair_prob with direction implied by historical lambda vs line."""
    if not hist_means:
        return df_priced
    out = df_priced.copy()
    adj_probs = []
    for _, r in out.iterrows():
        m = r['market']; p = r['player']; line = r['line']
        base = r['fair_prob']
        if m in hist_means and p in hist_means[m] and pd.notna(line):
            lam = hist_means[m][p]
            # approx P(over) under Normal(mu=lam, sigma from prior)
            sigma = prior_std_normal(m, lam)
            p_over = 1 - norm.cdf(line, loc=lam, scale=sigma)
            p_side_hist = p_over if r['selection'].lower() == 'over' else 1 - p_over
            adj = (1-blend)*base + blend*p_side_hist
            adj_probs.append(min(max(adj, 0.01), 0.99))
        else:
            adj_probs.append(base)
    out['fair_prob_hist'] = np.round(adj_probs, 4)
    # Recompute pricing columns from adjusted prob
    rows = []
    for _, r in out.iterrows():
        p_side = r.get('fair_prob_hist', r['fair_prob'])
        price = r['book_price']
        dec = american_to_decimal(price)
        edge = p_side - american_to_prob(price)
        ev = p_side*(dec-1.0) - (1 - p_side)
        kelly = min(KELLY_CAP, kelly_fraction_decimal(p_side, dec))
        fair_price = prob_to_american(p_side)
        rec = "BET" if (ev > 0 and edge >= DEFAULT_PROP_EV_MIN.get(r['market'], 0.01)) else "PASS"
        rows.append({**r.to_dict(), "fair_prob": round(p_side,4), "fair_price": fair_price, "edge": round(edge,4), "ev": round(ev,4), "kelly": round(kelly,4), "rec": rec})
    return pd.DataFrame(rows)

# -------------------- UI --------------------
st.set_page_config(page_title="Ultimate NFL Prop Pricer", page_icon="🏈", layout="wide")
st.title("🏈 Ultimate NFL Prop Pricer")

with st.sidebar:
    st.markdown("### Engine")
    engine = st.radio("Choose engine", ["Market-Fit (odds only)", "Hybrid (odds + historical EB)"])
    st.markdown("### Guardrails")
    edge_thr = st.slider("Min edge (prob)", 0.00, 0.15, EDGE_THRESHOLD_DEFAULT, 0.005)
    vol_thr  = st.slider("Max volatility ratio (σ/μ)", 0.50, 2.00, VOLATILITY_RATIO_THRESHOLD_DEFAULT, 0.05)
    zmin_thr = st.slider("Min |z| vs line", 0.00, 3.00, Z_MIN_DEFAULT, 0.05)

    st.markdown("---")
    if engine == "Hybrid (odds + historical EB)":
        year_start = st.number_input("Historical start year", 2015, datetime.date.today().year, 2020)
        year_end   = st.number_input("Historical end year", 2015, datetime.date.today().year, datetime.date.today().year)
        blend = st.slider("Blend weight (hist -> market)", 0.0, 0.8, 0.30, 0.05)
        if not NFL_DATA_OK:
            st.info("nfl_data_py not available. Hybrid mode will act like Market-Fit.")

st.markdown("---")

file = st.file_uploader("Upload props odds CSV (Prop Detail or OddsAPI schema)", type=["csv"])
if not file:
    st.info("Upload an odds CSV to get started.")
    st.stop()

df_props = load_props_csv(file)
if df_props.empty:
    st.error("Could not parse the CSV. Check required columns and label/market normalization.")
    st.stop()

with st.spinner("Pricing props..."):
    priced = market_fit_pipeline(df_props)

if priced.empty:
    st.warning("No priced props produced. Check if markets are supported.")
    st.stop()

# Optional historical adjustment
if engine == "Hybrid (odds + historical EB)":
    hist = train_hist_means(int(year_start), int(year_end)) if NFL_DATA_OK else {}
    priced = apply_hist_prior(priced, hist, blend=blend)

# Present
cols_show = ["market","player","selection","line","book","book_price","fair_prob","fair_price","edge","ev","kelly","rec","notes"]
st.dataframe(priced[cols_show], use_container_width=True, hide_index=True)

# Downloads
st.download_button("Download CSV", data=priced.to_csv(index=False), file_name="priced_props.csv", mime="text/csv")

# Quick KPIs
k_bets = int((priced["rec"] == "BET").sum())
avg_edge = float(priced.loc[priced["rec"] == "BET", "edge"].mean()) if k_bets else 0.0
st.markdown(f"**Bets flagged:** {k_bets} &nbsp;&nbsp;|&nbsp;&nbsp; **Avg edge:** {avg_edge:.3f}")
