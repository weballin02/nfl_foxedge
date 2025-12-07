#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL Props Fair Pricer (Enhanced & Corrected)
Works with ONLY an odds CSV. No external projections required.

How it produces "projections":
- For each (player, market) group, gathers all offered lines/prices (across books and alts).
- Converts prices to de-vigged implied probabilities per line (if both sides exist for a line).
- Fits a parametric distribution to those points:
  * Normal(mean, std) for yards/attempts/receptions-like
  * Poisson(lambda) for TDs
- Uses the fitted distribution as the model to compute fair probs for each row,
  then EV, Kelly, and recommendation.

If only ONE side/line observed for a player+market (underdetermined),
uses conservative league priors for std (Normal) or mild lambda prior (Poisson) centered at the quoted line,
then shrinks hard toward market to avoid fake edges.

Input CSV schema:
game_id,commence_time,in_play,bookmaker,last_update,home_team,away_team,market,label,description,price,point

ENHANCEMENTS:
- Comprehensive logging for debugging
- Input validation with clear error messages
- Division-by-zero protection in utility functions
- Robust parameter bounds checking
- More conservative priors for underdetermined cases
- Better error handling in optimization routines
- NaN/inf protection in all calculations
"""

from __future__ import annotations

import sys
import math
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from scipy.optimize import minimize

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Configuration --------------------
KELLY_CAP = 0.015
KELLY_FRACTION = 0.4
SHRINK_TO_PRIOR = 0.15  # How much to shrink fitted probs back to market priors

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

REQUIRED_COLS = [
    "game_id", "commence_time", "in_play", "bookmaker", "last_update",
    "home_team", "away_team", "market", "label", "description", "price", "point"
]

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

# -------------------- Prior Functions --------------------

def prior_std_normal(market: str, line: float) -> float:
    """Conservative league priors for std when underdetermined."""
    try:
        m = market.strip().lower()
        line = float(line)
        
        if m in ("player_pass_yds", "player_reception_yds", "player_rush_yds"):
            return max(10.0, 0.40 * (abs(line) + 1.0) ** 0.6)
        
        if m in ("player_pass_attempts", "player_pass_completions", "player_rush_attempts", "player_receptions"):
            return max(1.5, 0.65 * (abs(line) + 1.0) ** 0.5)
        
        return max(1.0, 0.5 * (abs(line) + 1.0) ** 0.5)
    
    except Exception as e:
        logger.warning(f"Error in prior_std_normal({market}, {line}): {e}. Using conservative default.")
        return 2.0

def prior_lambda_poisson(line: float) -> float:
    """Center Poisson lambda around the line with mild bias toward UNDER at half-lines."""
    try:
        line = float(line)
        return max(0.2, 0.90 * max(0.5, line))
    except Exception as e:
        logger.warning(f"Error in prior_lambda_poisson({line}): {e}. Using default 1.0")
        return 1.0

# -------------------- Odds Utilities --------------------

def american_to_prob(odds: int | float) -> float:
    """Convert American odds to implied probability."""
    try:
        odds = float(odds)
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)
    except Exception as e:
        logger.error(f"Error in american_to_prob({odds}): {e}")
        return 0.5

def american_to_decimal(odds: int | float) -> float:
    """Convert American odds to decimal."""
    try:
        odds = float(odds)
        if odds >= 100:
            return 1 + odds / 100.0
        if odds <= -100:
            return 1 + 100.0 / abs(odds)
        return 1.0
    except Exception as e:
        logger.error(f"Error in american_to_decimal({odds}): {e}")
        return 1.0

def prob_to_american(p: float) -> int:
    """Convert probability to American odds."""
    try:
        p = min(max(float(p), 1e-6), 1 - 1e-6)
        if p > 0.5:
            return -int(round(p / (1 - p) * 100))
        return int(round((1 - p) / p * 100))
    except Exception as e:
        logger.error(f"Error in prob_to_american({p}): {e}")
        return 0

def remove_vig_two_way(p_over_raw: float, p_under_raw: float) -> Tuple[float, float]:
    """Remove vig from two-way market (over/under)."""
    try:
        s = float(p_over_raw) + float(p_under_raw)
        if s <= 1e-9:
            logger.warning(f"Two-way market has zero vig ({p_over_raw}, {p_under_raw}). Returning 50/50.")
            return 0.5, 0.5
        return float(p_over_raw) / s, float(p_under_raw) / s
    except Exception as e:
        logger.error(f"Error in remove_vig_two_way({p_over_raw}, {p_under_raw}): {e}")
        return 0.5, 0.5

def kelly_fraction_decimal(p: float, price_decimal: float, frac: float = KELLY_FRACTION) -> float:
    """Calculate Kelly fraction given probability and decimal odds."""
    try:
        p = float(p)
        price_decimal = float(price_decimal)
        b = price_decimal - 1.0
        
        if b <= 1e-9 or p <= 1e-9:
            return 0.0
        
        f_full = (p * (b + 1) - 1) / b
        result = max(0.0, f_full * frac)
        
        if not np.isfinite(result):
            return 0.0
        
        return float(result)
    except Exception as e:
        logger.error(f"Error in kelly_fraction_decimal({p}, {price_decimal}): {e}")
        return 0.0

# -------------------- CSV Loading --------------------

def load_props_csv(path: str) -> pd.DataFrame:
    """Load and validate props CSV."""
    logger.info(f"Loading props CSV from {path}")
    
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parse error: {e}")
        raise
    
    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.error(f"Props CSV missing columns: {missing}")
        raise ValueError(f"Props CSV missing columns: {missing}")
    
    # Clean and normalize
    try:
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["market"] = df["market"].astype(str).str.strip().str.lower()
        df["description"] = df["description"].astype(str).str.strip()
        df["price"] = df["price"].apply(lambda x: int(str(x).replace("−", "-")))
        df["point"] = df["point"].astype(float)
    except Exception as e:
        logger.error(f"Error normalizing CSV columns: {e}")
        raise
    
    # Filter to over/under only
    initial_rows = len(df)
    df = df[df["label"].isin(["over", "under"])].copy()
    filtered_rows = len(df)
    logger.info(f"Filtered to {filtered_rows}/{initial_rows} over/under rows")
    
    # Validate no NaN in critical columns
    critical_cols = ["price", "point", "description", "market"]
    for col in critical_cols:
        nans = df[col].isna().sum()
        if nans > 0:
            logger.warning(f"Column '{col}' has {nans} NaN values. Removing them.")
            df = df.dropna(subset=[col])
    
    return df

# -------------------- Distribution Fitting --------------------

def fit_normal_from_points(points: List[Tuple[float, float]]) -> Tuple[float, float, str]:
    """
    Fit Normal(mean, std) to (line, q_over) points.
    Minimizes squared error of tail probabilities.
    """
    if not points:
        logger.warning("fit_normal_from_points: No points provided. Using fallback.")
        return 0.0, 1.0, "normal_empty"
    
    try:
        xs = np.array([x for x, _ in points], dtype=float)
        ys = np.array([y for _, y in points], dtype=float)
        
        x0_mu = float(np.median(xs))
        x0_std = float(prior_std_normal("normal", max(1.0, abs(x0_mu))))
        
        def loss(theta):
            mu, log_std = theta
            s = math.exp(log_std)
            if s < 0.1:
                return 1e6
            pred = 1.0 - norm.cdf(xs, loc=mu, scale=s)
            pred = np.clip(pred, 1e-6, 1 - 1e-6)
            err = np.mean((pred - ys) ** 2)
            if not np.isfinite(err):
                return 1e6
            return err
        
        res = minimize(
            loss,
            x0=np.array([x0_mu, math.log(x0_std)]),
            method="L-BFGS-B",
            bounds=[(-1000, 1000), (math.log(0.1), math.log(100))]
        )
        
        if res.success and np.isfinite(res.fun):
            mu = float(res.x[0])
            std = float(math.exp(res.x[1]))
            logger.debug(f"Normal fit success: mu={mu:.2f}, std={std:.2f}")
            return mu, max(0.5, std), "normal_fit"
        else:
            logger.debug(f"Normal fit failed (loss={res.fun}). Using prior.")
            return x0_mu, x0_std, "normal_prior_fallback"
    
    except Exception as e:
        logger.error(f"Error in fit_normal_from_points: {e}")
        return float(np.median([x for x, _ in points])), 1.0, "normal_error"

def fit_poisson_from_points(points: List[Tuple[float, float]]) -> Tuple[float, str]:
    """
    Fit Poisson(lambda) to (line, q_over) points.
    Minimizes squared error of tail probabilities.
    """
    if not points:
        logger.warning("fit_poisson_from_points: No points provided. Using fallback.")
        return 1.0, "poisson_empty"
    
    try:
        xs = np.array([x for x, _ in points], dtype=float)
        ys = np.array([y for _, y in points], dtype=float)
        
        x0 = float(np.median(xs))
        lam0 = prior_lambda_poisson(x0)
        
        def loss(log_lam):
            lam = math.exp(log_lam)
            if lam < 0.05:
                return 1e6
            preds = 1.0 - poisson.cdf(np.floor(xs).astype(int), lam)
            preds = np.clip(preds, 1e-6, 1 - 1e-6)
            err = np.mean((preds - ys) ** 2)
            if not np.isfinite(err):
                return 1e6
            return err
        
        res = minimize(
            loss,
            x0=np.array([math.log(lam0)]),
            method="L-BFGS-B",
            bounds=[(math.log(0.05), math.log(50))]
        )
        
        if res.success and np.isfinite(res.fun):
            lam = float(math.exp(res.x[0]))
            logger.debug(f"Poisson fit success: lambda={lam:.2f}")
            return max(0.1, lam), "poisson_fit"
        else:
            logger.debug(f"Poisson fit failed (loss={res.fun}). Using prior.")
            return lam0, "poisson_prior_fallback"
    
    except Exception as e:
        logger.error(f"Error in fit_poisson_from_points: {e}")
        return 1.0, "poisson_error"

# -------------------- Line Priors --------------------

def line_priors(line_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Compute over/under de-vigged priors at a line (if both sides exist)."""
    try:
        over = line_df[line_df["label"] == "over"]
        under = line_df[line_df["label"] == "under"]
        
        if over.empty or under.empty:
            return None, None
        
        q_over_raw = american_to_prob(int(over.iloc[0]["price"]))
        q_under_raw = american_to_prob(int(under.iloc[0]["price"]))
        
        return remove_vig_two_way(q_over_raw, q_under_raw)
    
    except Exception as e:
        logger.error(f"Error in line_priors: {e}")
        return None, None

# -------------------- Main Pricing --------------------

def price_group(group: pd.DataFrame, market: str, ev_min: float) -> pd.DataFrame:
    """
    Price all rows in a (player, market) group using fitted/prior distributions.
    """
    logger.debug(f"Pricing group: {market}, {len(group)} rows")
    
    # Collect (line, q_over_prior) points where both sides exist for fitting
    points = []
    line_to_priors: Dict[float, Tuple[Optional[float], Optional[float]]] = {}
    
    for line, ldf in group.groupby("point"):
        q_over, q_under = line_priors(ldf)
        line_to_priors[float(line)] = (q_over, q_under)
        if q_over is not None:
            points.append((float(line), float(q_over)))
    
    model = MARKET_MODEL.get(market, "normal")
    fit_note = ""
    model_params = {}
    
    # Fit distribution if we have enough points
    if len(points) >= 2 or (model == "poisson" and len(points) >= 1):
        if model == "poisson":
            lam, fit_note = fit_poisson_from_points(points)
            
            def p_over_fn(x):
                try:
                    return 1.0 - poisson.cdf(int(math.floor(x)), lam)
                except:
                    return 0.5
            
            model_params = {"lambda": lam}
        else:
            mu, std, fit_note = fit_normal_from_points(points)
            
            def p_over_fn(x):
                try:
                    return 1.0 - norm.cdf(x, loc=mu, scale=std)
                except:
                    return 0.5
            
            model_params = {"mu": mu, "std": std}
    else:
        # Underdetermined: use conservative priors
        logger.debug(f"Underdetermined fit ({len(points)} points). Using priors.")
        med_line = float(np.median(group["point"].values))
        
        if model == "poisson":
            lam = prior_lambda_poisson(med_line)
            
            def p_over_fn(x):
                try:
                    return 1.0 - poisson.cdf(int(math.floor(x)), lam)
                except:
                    return 0.5
            
            model_params = {"lambda": lam}
            fit_note = "poisson_prior"
        else:
            std = prior_std_normal(market, med_line)
            mu = med_line  # Center at line to avoid fake edges
            
            def p_over_fn(x):
                try:
                    return 1.0 - norm.cdf(x, loc=mu, scale=std)
                except:
                    return 0.5
            
            model_params = {"mu": mu, "std": std}
            fit_note = "normal_prior"
    
    # Price each row
    rows = []
    for idx, r in group.iterrows():
        try:
            line = float(r["point"])
            sel = str(r["label"]).lower()
            price = int(r["price"])
            
            # Get model prob at this line
            p_over_model = float(np.clip(p_over_fn(line), 1e-4, 1 - 1e-4))
            p_side_model = p_over_model if sel == "over" else (1.0 - p_over_model)
            
            # Shrink toward market prior if available at this exact line
            q_over, q_under = line_to_priors.get(line, (None, None))
            if q_over is not None and q_under is not None:
                prior = q_over if sel == "over" else q_under
                p_side = (1 - SHRINK_TO_PRIOR) * p_side_model + SHRINK_TO_PRIOR * prior
                edge = (p_over_model - q_over) if sel == "over" else ((1 - p_over_model) - q_under)
            else:
                p_side = p_side_model
                edge = p_side_model - 0.5  # Weak proxy when no priors
            
            # Ensure probability is valid
            p_side = np.clip(float(p_side), 1e-6, 1 - 1e-6)
            
            dec = american_to_decimal(price)
            if dec <= 1.0:
                logger.warning(f"Invalid decimal odds: {dec} from price {price}")
                dec = 1.1
            
            ev = float(p_side * (dec - 1.0) - (1 - p_side))
            kelly = min(KELLY_CAP, kelly_fraction_decimal(p_side, dec))
            fair_price = prob_to_american(p_side)
            
            rec = "BET" if (ev > 0 and edge >= ev_min) else "PASS"
            
            note_params = ", ".join([f"{k}={v:.2f}" for k, v in model_params.items()])
            note_prior = "; prior_line" if q_over is not None else ""
            
            rows.append({
                "game_id": r["game_id"],
                "commence_time": r["commence_time"],
                "bookmaker": r["bookmaker"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "market": r["market"],
                "player": r["description"],
                "selection": r["label"].capitalize(),
                "line": round(line, 1),
                "book_price": price,
                "fair_prob": round(float(p_side), 4),
                "fair_price": int(fair_price),
                "edge": round(float(edge), 4),
                "ev": round(float(ev), 4),
                "kelly": round(float(kelly), 4),
                "rec": rec,
                "notes": f"{fit_note} [{note_params}]{note_prior}"
            })
        
        except Exception as e:
            logger.error(f"Error pricing row {idx}: {e}")
            continue
    
    return pd.DataFrame(rows)

# -------------------- Pipeline --------------------

def props_auto_pipeline(props_csv: str) -> pd.DataFrame:
    """Main pipeline: load CSV, price all props, return ranked DataFrame."""
    try:
        df = load_props_csv(props_csv)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()
    
    if df.empty:
        logger.warning("Loaded CSV is empty or has no over/under rows.")
        return pd.DataFrame()
    
    logger.info(f"Processing {df['market'].nunique()} markets, {df['description'].nunique()} unique players")
    
    out_frames = []
    for (market, player), sub in df.groupby(["market", "description"]):
        try:
            ev_min = DEFAULT_PROP_EV_MIN.get(market, 0.01)
            priced = price_group(sub, market, ev_min=ev_min)
            if not priced.empty:
                out_frames.append(priced)
                logger.debug(f"Priced {market}/{player}: {len(priced)} rows")
        except Exception as e:
            logger.error(f"Error pricing {market}/{player}: {e}")
            continue
    
    if not out_frames:
        logger.warning("No props were successfully priced.")
        return pd.DataFrame()
    
    out = pd.concat(out_frames, ignore_index=True)
    out["rec"] = pd.Categorical(out["rec"], categories=["BET", "PASS"], ordered=True)
    out = out.sort_values(["rec", "ev"], ascending=[True, False]).reset_index(drop=True)
    
    logger.info(f"Final output: {len(out)} priced props")
    logger.info(f"BET recommendations: {len(out[out['rec'] == 'BET'])}")
    
    return out

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(
        description="NFL Props fair pricer that fits projections from the market",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 nfl_props_pricer.py --props-csv props.csv
  python3 nfl_props_pricer.py --props-csv props.csv --save output.csv
        """
    )
    ap.add_argument("--props-csv", required=True, help="CSV of prop odds")
    ap.add_argument("--save", default="", help="Optional output CSV path")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity (default: INFO)")
    
    args = ap.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting NFL props fair pricer")
    out = props_auto_pipeline(args.props_csv)
    
    if out.empty:
        logger.warning("No priced props to display. Check input CSV and logs.")
        sys.exit(0)
    
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n" + out.to_string(index=False) + "\n")
    
    logger.info(f"Display: {len(out)} rows")
    
    if args.save:
        try:
            out.to_csv(args.save, index=False)
            logger.info(f"Saved to {args.save}")
            print(f"✓ Saved to: {args.save}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            print(f"✗ Error saving to {args.save}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()