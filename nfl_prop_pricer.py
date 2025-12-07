
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Props Fair Pricer (Market-Fit Edition)
Works with ONLY an odds CSV. No external projections required.

How it produces "projections":
- For each (player, market) group, it gathers all offered lines/prices (across books and alts).
- Converts prices to de-vigged implied probabilities per line (if both sides exist for a line).
- Fits a parametric distribution to those points:
  * Normal(mean, std) for yards/attempts/receptions/attempts-like
  * Poisson(lambda) for TDs
- Uses the fitted distribution as the model to compute fair probs for each row,
  then EV, Kelly, and recommendation.

If there is only ONE side/line observed for a player+market (underdetermined),
it uses conservative league priors for std (Normal) or a mild λ prior (Poisson) centered at the quoted line,
then shrinks hard toward market to avoid fake edges.

Input CSV schema (same as before):
game_id,commence_time,in_play,bookmaker,last_update,home_team,away_team,market,label,description,price,point
"""

from __future__ import annotations

import sys, math, argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from scipy.optimize import minimize

KELLY_CAP = 0.015
KELLY_FRACTION = 0.4

# How much to shrink fitted probs back to market priors when both sides exist
SHRINK_TO_PRIOR = 0.15

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

REQUIRED_COLS = ["game_id","commence_time","in_play","bookmaker","last_update",
                 "home_team","away_team","market","label","description","price","point"]

MARKET_MODEL = {
    "player_pass_attempts": "normal",
    "player_pass_completions": "normal",
    "player_pass_tds": "poisson",
    "player_pass_yds": "normal",
    "player_reception_yds": "normal",
    "player_receptions": "normal",
    "player_rush_attempts": "normal",
    "player_rush_yds": "normal",
}

# Conservative league priors for std when underdetermined
def prior_std_normal(market: str, line: float) -> float:
    m = market.strip().lower()
    if m in ("player_pass_yds","player_reception_yds","player_rush_yds"):
        return max(10.0, 0.40 * (abs(line)+1.0)**0.6)
    if m in ("player_pass_attempts","player_pass_completions","player_rush_attempts","player_receptions"):
        return max(1.5, 0.65 * (abs(line)+1.0)**0.5)
    return max(1.0, 0.5 * (abs(line)+1.0)**0.5)

def prior_lambda_poisson(line: float) -> float:
    # Center around the line with a mild bias toward UNDER at half-lines
    return max(0.2, 0.90 * max(0.5, line))

# ----------------- odds utils -----------------

def american_to_prob(odds: int | float) -> float:
    odds = float(odds)
    if odds > 0: return 100.0/(odds+100.0)
    return abs(odds)/(abs(odds)+100.0)

def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds >= 100: return 1 + odds/100.0
    if odds <= -100: return 1 + 100.0/abs(odds)
    return 1.0

def prob_to_american(p: float) -> int:
    p = min(max(float(p), 1e-6), 1-1e-6)
    return -int(round(p/(1-p)*100)) if p>0.5 else int(round((1-p)/p*100))

def remove_vig_two_way(p_over_raw: float, p_under_raw: float) -> Tuple[float,float]:
    s = p_over_raw + p_under_raw
    if s <= 0: return 0.5, 0.5
    return p_over_raw/s, p_under_raw/s

def kelly_fraction_decimal(p: float, price_decimal: float, frac: float = KELLY_FRACTION) -> float:
    b = price_decimal - 1.0
    if b <= 0: return 0.0
    f_full = (p * (b + 1) - 1) / b
    return max(0.0, f_full * frac)

# ----------------- CSV -----------------

def load_props_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Props CSV missing columns: {missing}")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    df["description"] = df["description"].astype(str).str.strip()
    df["price"] = df["price"].apply(lambda x: int(str(x).replace("−","-")))
    df["point"] = df["point"].astype(float)
    # Normalize selection to over/under and filter unknown
    df = df[df["label"].isin(["over","under"])].copy()
    return df

# ----------------- fit distributions from market -----------------

def fit_normal_from_points(points: List[Tuple[float,float]]) -> Tuple[float,float,str]:
    """
    points: list of (line, q_over) pairs with q_over in (0,1)
    Fit mean and std to minimize squared error of tail probs.
    """
    # initial guess: mean near median line; std from prior on typical line
    xs = np.array([x for x,_ in points], dtype=float)
    ys = np.array([y for _,y in points], dtype=float)
    x0_mu = float(np.median(xs))
    x0_std = float(prior_std_normal("normal", max(1.0, abs(x0_mu))))
    def loss(theta):
        mu, log_std = theta
        s = math.exp(log_std)
        pred = 1.0 - norm.cdf(xs, loc=mu, scale=s)
        return np.mean((pred - ys)**2)
    res = minimize(loss, x0=np.array([x0_mu, math.log(x0_std)]), method="L-BFGS-B")
    if not res.success:
        mu, std = x0_mu, x0_std
        note = "normal prior-fallback"
    else:
        mu, std = float(res.x[0]), float(math.exp(res.x[1]))
        note = "normal fit"
    std = max(0.5, std)
    return mu, std, note

def fit_poisson_from_points(points: List[Tuple[float,float]]) -> Tuple[float,str]:
    """
    points: (line, q_over). Fit lambda by minimizing squared error of tail probs.
    """
    xs = np.array([x for x,_ in points], dtype=float)
    ys = np.array([y for _,y in points], dtype=float)
    x0 = float(np.median(xs))
    lam0 = prior_lambda_poisson(x0)
    def loss(log_lam):
        lam = math.exp(log_lam)
        preds = 1.0 - poisson.cdf(np.floor(xs).astype(int), lam)
        return np.mean((preds - ys)**2)
    res = minimize(loss, x0=np.array([math.log(lam0)]), method="L-BFGS-B")
    if not res.success:
        lam = lam0
        note = "poisson prior-fallback"
    else:
        lam = float(math.exp(res.x[0]))
        note = "poisson fit"
    lam = max(0.1, lam)
    return lam, note

# Build over/under priors at each line when both sides exist
def line_priors(line_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    over = line_df[line_df["label"]=="over"]
    under = line_df[line_df["label"]=="under"]
    if over.empty or under.empty: return None, None
    q_over_raw = american_to_prob(int(over.iloc[0]["price"]))
    q_under_raw = american_to_prob(int(under.iloc[0]["price"]))
    return remove_vig_two_way(q_over_raw, q_under_raw)

# ----------------- main pricing -----------------

def price_group(group: pd.DataFrame, market: str, ev_min: float) -> pd.DataFrame:
    # Collect (line, q_over_prior) points where both sides exist for fitting
    points = []
    line_to_priors: Dict[float, Tuple[Optional[float],Optional[float]]] = {}
    for line, ldf in group.groupby("point"):
        q_over, q_under = line_priors(ldf)
        line_to_priors[float(line)] = (q_over, q_under)
        if q_over is not None:
            points.append((float(line), float(q_over)))

    model = MARKET_MODEL.get(market, "normal")
    fit_note = ""
    if len(points) >= 2 or (model=="poisson" and len(points) >= 1):
        if model == "poisson":
            lam, fit_note = fit_poisson_from_points(points)
            def p_over_fn(x): return 1.0 - poisson.cdf(int(math.floor(x)), lam)
            model_params = {"lambda": lam}
        else:
            mu, std, fit_note = fit_normal_from_points(points)
            def p_over_fn(x): return 1.0 - norm.cdf(x, loc=mu, scale=std)
            model_params = {"mu": mu, "std": std}
    else:
        # underdetermined: set center ~ line median, use conservative priors
        med_line = float(np.median(group["point"].values))
        if model == "poisson":
            lam = prior_lambda_poisson(med_line)
            def p_over_fn(x): return 1.0 - poisson.cdf(int(math.floor(x)), lam)
            model_params = {"lambda": lam}
            fit_note = "poisson prior"
        else:
            std = prior_std_normal(market, med_line)
            mu = med_line  # center at line; this deliberately avoids fake edges
            def p_over_fn(x): return 1.0 - norm.cdf(x, loc=mu, scale=std)
            model_params = {"mu": mu, "std": std}
            fit_note = "normal prior"

    rows = []
    for idx, r in group.iterrows():
        line = float(r["point"])
        sel = str(r["label"])
        price = int(r["price"])
        # model prob
        p_over_model = float(np.clip(p_over_fn(line), 1e-4, 1-1e-4))
        p_side_model = p_over_model if sel=="over" else (1.0 - p_over_model)
        # shrink toward market prior if we have it at this exact line
        q_over, q_under = line_to_priors.get(line, (None, None))
        if q_over is not None and q_under is not None:
            prior = q_over if sel=="over" else q_under
            p_side = (1 - SHRINK_TO_PRIOR)*p_side_model + SHRINK_TO_PRIOR*prior
            edge = (p_over_model - q_over) if sel=="over" else ((1 - p_over_model) - q_under)
        else:
            p_side = p_side_model
            edge = p_side_model - 0.5  # weak proxy when no priors

        dec = american_to_decimal(price)
        ev = float(p_side*(dec-1.0) - (1 - p_side))
        kelly = min(KELLY_CAP, kelly_fraction_decimal(p_side, dec))
        fair_price = prob_to_american(p_side)
        rec = "BET" if (ev > 0 and edge >= ev_min) else "PASS"

        note_params = ", ".join([f"{k}={v:.2f}" for k,v in model_params.items()])
        note_prior = f"; prior_line" if q_over is not None else ""
        rows.append({
            "game_id": r["game_id"],
            "commence_time": r["commence_time"],
            "bookmaker": r["bookmaker"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "market": r["market"],
            "player": r["description"],
            "selection": r["label"].capitalize(),
            "line": line,
            "book_price": price,
            "fair_prob": round(float(p_side), 4),
            "fair_price": int(fair_price),
            "edge": round(float(edge), 4),
            "ev": round(float(ev), 4),
            "kelly": round(float(kelly), 4),
            "rec": rec,
            "notes": f"{fit_note} [{note_params}]{note_prior}"
        })
    return pd.DataFrame(rows)

def props_auto_pipeline(props_csv: str) -> pd.DataFrame:
    df = load_props_csv(props_csv)
    if df.empty:
        return pd.DataFrame()

    out_frames = []
    for (market, player), sub in df.groupby(["market","description"]):
        ev_min = DEFAULT_PROP_EV_MIN.get(market, 0.01)
        priced = price_group(sub, market, ev_min=ev_min)
        out_frames.append(priced)

    if not out_frames:
        return pd.DataFrame()

    out = pd.concat(out_frames, ignore_index=True)
    out["rec"] = pd.Categorical(out["rec"], categories=["BET","PASS"], ordered=True)
    out = out.sort_values(["rec","ev"], ascending=[True, False]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="Props fair pricer that fits projections from the market")
    ap.add_argument("--props-csv", required=True, help="CSV of prop odds (schema in docstring)")
    ap.add_argument("--save", default="", help="Optional output CSV path")
    args = ap.parse_args()

    out = props_auto_pipeline(args.props_csv)
    if out.empty:
        print("No priced props. Check CSV contents.")
        sys.exit(0)

    pd.set_option("display.max_columns", None); pd.set_option("display.width", 220)
    print(out.to_string(index=False))

    if args.save:
        out.to_csv(args.save, index=False)
        print(f"\nSaved: {args.save}")

if __name__ == "__main__":
    main()
