#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
foxedge_nfl_prop_pricer.py

Purpose:
    Automated NFL player prop mispricing scanner.
    - Takes ONLY current lines + odds (multi-book, multi-alt if possible).
    - Fits a market-implied distribution for each (player, market) from those odds.
    - Flags where a specific book is off-consensus.
    - Computes EV, capped Kelly stake, and unit sizing off bankroll.
    - Outputs:
        * bets.csv (ranked betting card with stake)
        * audit_all_candidates.csv (full internals for receipts/content)

THIS IS NOT A "FANTASY PROJECTIONS" MODEL.
This is a market-disagreement model. If you try to sell it as
"my algorithm thinks Player X throws 274.3 yards," you're lying to yourself.
You are selling: "Book A is hanging a stale/soft line while consensus is here.
We tax the gap."

Input CSV columns (one row per book offer / alt line / side):
    game_id,commence_time,in_play,bookmaker,last_update,
    home_team,away_team,market,label,description,price,point

Where:
    - market is e.g. player_pass_yds, player_rush_yds, player_receptions, etc.
    - label is 'over' or 'under'
    - description is player name or player+stat description
    - price is American odds (-115, +120, etc.)
    - point is the stat line (e.g. 63.5 yards)

Example CLI:
    python3 foxedge_nfl_prop_pricer.py \
        --odds_path props_odds.csv \
        --bankroll 5000 \
        --ev_threshold 0.02 \
        --kelly_cap 0.02 \
        --min_prob 0.45 \
        --side_bias auto \
        --book_preference "FD,DK,PB" \
        --bets_out bets.csv \
        --audit_out audit_all_candidates.csv \
        --log-level INFO

Author: FoxEdge
No vibe betting. No fan fiction. Just mispricing.
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIG / CONSTANTS
# -----------------------------------------------------------------------------

# Markets modeled as approximately continuous ~ Normal
CONTINUOUS_MARKETS = {
    "player_pass_attempts",
    "player_pass_completions",
    "player_pass_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_attempts",
    "player_rush_yds",
}

# Markets modeled as discrete / rare-event ~ Poisson
POISSON_MARKETS = {
    "player_pass_tds",
    "player_tds_over",
}

SUPPORTED_MARKETS = CONTINUOUS_MARKETS | POISSON_MARKETS

REQUIRED_COLS = [
    "game_id", "commence_time", "in_play", "bookmaker", "last_update",
    "home_team", "away_team", "market", "label", "description", "price", "point"
]

# Default CLI thresholds
DEFAULT_BANKROLL = 5000.0
DEFAULT_EV_THRESHOLD = 0.02       # require >=2% EV
DEFAULT_KELLY_CAP = 0.02          # max Kelly fraction per bet
DEFAULT_MIN_PROB = 0.45           # don't tail fringe 20% hits just because EV says yes
DEFAULT_SIDE_BIAS = "auto"        # "auto" = pick EV side, else "over"/"under"

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logger = logging.getLogger("foxedge_nfl_prop_pricer")

def setup_logging(level_str: str = "INFO") -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(level)
    logger.info(f"Logging initialized at {level_str.upper()}")


# -----------------------------------------------------------------------------
# ODDS / MATH UTILITIES
# -----------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    """
    Convert American odds to decimal odds.
    +120 -> 2.20
    -150 -> 1.666...
    """
    american = float(american)
    if american == 0:
        raise ValueError("American odds cannot be 0.")
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def decimal_to_prob(decimal: float) -> float:
    """
    Decimal odds -> implied prob with vig.
    2.00 -> 0.5
    1.666.. -> 0.6
    """
    decimal = float(decimal)
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be > 1.")
    return 1.0 / decimal


def no_vig_two_way(over_american: float, under_american: float) -> Tuple[float, float]:
    """
    Remove vig from an Over/Under two-way market using proportional normalization.
    Return (p_over_fair, p_under_fair) such that they sum to ~1.
    We assume lines are symmetric (same stat line).
    """
    d_over = american_to_decimal(over_american)
    d_under = american_to_decimal(under_american)

    p_over_raw = decimal_to_prob(d_over)
    p_under_raw = decimal_to_prob(d_under)

    s = p_over_raw + p_under_raw
    if s <= 0:
        raise ValueError("Invalid odds; probabilities sum to zero or negative.")

    return p_over_raw / s, p_under_raw / s


def kelly_fraction(p: float, dec_odds: float, cap: float) -> float:
    """
    Kelly fraction for decimal odds with a hard cap.
    p        = win probability for chosen side (consensus)
    dec_odds = decimal odds at the book we're about to bet
    cap      = max Kelly fraction we allow (safety brake)

    Returns fraction of bankroll to stake (0..cap). Not dollars.
    """
    if not (0.0 < p < 1.0):
        return 0.0

    b = dec_odds - 1.0
    if b <= 0:
        return 0.0

    q = 1.0 - p
    k_full = (b * p - q) / b  # standard Kelly
    if k_full <= 0:
        return 0.0

    return min(k_full, cap)


def prob_to_american(p: float) -> int:
    """
    Convert a probability (0<p<1) to an equivalent fair American price.
    p=0.5 -> -100
    p=0.6 -> -150-ish
    p=0.4 -> +150-ish
    """
    # clamp for sanity
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    if p > 0.5:
        # favorite (negative odds)
        return -int(round((p / (1.0 - p)) * 100.0))
    else:
        # dog (positive odds)
        return int(round(((1.0 - p) / p) * 100.0))


# -----------------------------------------------------------------------------
# DISTRIBUTION HELPERS
# -----------------------------------------------------------------------------

def normal_survival(x: float, mu: float, sigma: float) -> float:
    """
    Survival function P(X > x) for Normal(mu, sigma).
    We use error function approximation instead of scipy to keep deps tight.
    """
    if sigma <= 0:
        return 0.0

    z = (x - mu) / sigma
    # 0.5 * (1 + erf(z/sqrt(2))) is CDF, so 1 - that is survival.
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    sf = 1.0 - cdf
    # clamp [0,1]
    if sf < 0.0:
        sf = 0.0
    elif sf > 1.0:
        sf = 1.0
    return sf


def poisson_cdf(k: int, lam: float) -> float:
    """
    Poisson CDF P(K <= k) for mean lam using direct summation.
    That's fine for TD props where lam ~ [0, ~3].
    """
    if lam <= 0:
        # degenerate: P(K<=k)=1 for k>=0, else 0
        return 1.0 if k >= 0 else 0.0

    k = int(math.floor(k))
    if k < 0:
        return 0.0

    ks = np.arange(0, k + 1)
    # e^-lam * lam^k / k!
    terms = np.exp(-lam) * (lam ** ks) / np.array([math.factorial(int(t)) for t in ks])
    return float(np.sum(terms))


def poisson_survival_at_line(line: float, lam: float) -> float:
    """
    For "Over x.5 TDs":
        Over(0.5) basically means score >=1 TD.
    So threshold = floor(line)+1.
    P(K >= threshold) = 1 - P(K <= threshold-1)
    """
    threshold = math.floor(line + 1e-9) + 1
    return max(0.0, min(1.0, 1.0 - poisson_cdf(threshold - 1, lam)))


# -----------------------------------------------------------------------------
# DATA LOADING / CLEANING
# -----------------------------------------------------------------------------

def load_prop_odds(path: str) -> pd.DataFrame:
    """
    Load the raw prop odds CSV, sanitize it, enforce required schema,
    drop unusable junk.

    We:
    - Check required columns
    - Normalize unicode minus in American odds
    - Force label to lowercase in {'over','under'}
    - Cast price to int
    - Cast point to float
    - Filter to supported markets only
    - Drop rows missing critical values
    """

    logger.info(f"Loading prop odds from {path}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parse error: {e}")
        raise

    # Validate required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.error(f"Input CSV missing columns: {missing}")
        raise ValueError(f"Input CSV missing columns: {missing}")

    # Normalize / clean columns
    df = df.copy()

    # lowercase/strip certain fields
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["description"] = df["description"].astype(str).str.strip()
    df["bookmaker"] = df["bookmaker"].astype(str).str.strip()

    # normalize unicode minus in price, then cast to int
    def _clean_price(x):
        x_str = str(x).replace("−", "-").strip()
        try:
            return int(float(x_str))
        except Exception:
            return np.nan

    df["price"] = df["price"].apply(_clean_price)

    # cast point to float
    df["point"] = pd.to_numeric(df["point"], errors="coerce")

    # Drop anything not Over/Under (just in case)
    df = df[df["label"].isin(["over", "under"])]

    # Keep only supported stat markets
    df = df[df["market"].isin(SUPPORTED_MARKETS)]

    # Critical columns must not be NaN
    crit = ["price", "point", "description", "market", "bookmaker"]
    before = len(df)
    df = df.dropna(subset=crit)
    after = len(df)
    dropped = before - after
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows due to NaN in critical columns")

    logger.info(f"Loaded {len(df)} usable rows after cleaning")

    return df


def pivot_two_way(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild two-way markets so that each row in the result represents:
        (bookmaker, player, market, line)
    with BOTH Over and Under prices present.

    Steps:
    - Split Over and Under
    - Inner-join on shared keys so we only keep lines with both sides posted
    - Compute each book's no-vig fair probabilities for that exact line
    - Compute decimal odds for each side
    """

    keys = [
        "game_id", "commence_time", "in_play", "bookmaker", "last_update",
        "home_team", "away_team", "market", "description", "point"
    ]

    over_df = (
        df[df["label"] == "over"][keys + ["price"]]
        .rename(columns={"price": "over_odds"})
    )
    under_df = (
        df[df["label"] == "under"][keys + ["price"]]
        .rename(columns={"price": "under_odds"})
    )

    merged = pd.merge(
        over_df,
        under_df,
        on=keys,
        how="inner"
    )

    # Deduplicate in case the feed duplicates identical rows
    merged = merged.drop_duplicates(
        keys + ["over_odds", "under_odds"]
    ).copy()

    # Cast to clean numeric types
    merged["point"] = merged["point"].astype(float)
    merged["over_odds"] = merged["over_odds"].astype(int)
    merged["under_odds"] = merged["under_odds"].astype(int)

    # Derive no-vig probs
    def _novig(row):
        try:
            return no_vig_two_way(row["over_odds"], row["under_odds"])
        except Exception as e:
            logger.debug(f"no_vig_two_way failed for row: {e}")
            return (np.nan, np.nan)

    merged[["p_over_novig", "p_under_novig"]] = merged.apply(
        _novig, axis=1, result_type="expand"
    )

    # kill rows where no-vig failed
    before_nv = len(merged)
    merged = merged.dropna(subset=["p_over_novig", "p_under_novig"])
    after_nv = len(merged)
    if after_nv < before_nv:
        logger.warning(f"Dropped {before_nv - after_nv} rows missing valid no-vig probs")

    # Decimal odds for later EV calc
    merged["dec_over"] = merged["over_odds"].apply(lambda o: american_to_decimal(o))
    merged["dec_under"] = merged["under_odds"].apply(lambda o: american_to_decimal(o))

    logger.info(f"Built {len(merged)} two-way offers with matched Over/Under")

    return merged


# -----------------------------------------------------------------------------
# CONSENSUS FITTING (MARKET-DERIVED DISTRIBUTION)
# -----------------------------------------------------------------------------

def fit_normal_from_points(lines: np.ndarray, p_over: np.ndarray) -> Tuple[float, float]:
    """
    Fit Normal(mu, sigma) to the (line -> no-vig Over prob) mapping
    using coarse-to-fine grid search to minimize MSE.

    We DO NOT invent priors here. If the market doesn't give us spread,
    you're not getting a fake sigma and calling it "edge".
    """

    # Start around the ~50% point to guess mu
    idx_mid = int(np.argmin(np.abs(p_over - 0.5)))
    mu0 = float(lines[idx_mid])

    # Rough sigma guess from spread in lines vs probs.
    # Try to infer how fast prob changes with line changes.
    try:
        hi_lines = lines[p_over <= 0.16]
        lo_lines = lines[p_over >= 0.84]
        if len(hi_lines) and len(lo_lines):
            approx_sigma = max(
                3e-3,
                abs(np.median(hi_lines) - np.median(lo_lines)) / 2.0
            )
        else:
            # fallback: IQR/1.35 style
            q75, q25 = np.percentile(lines, [75, 25])
            approx_sigma = max(3e-3, abs(q75 - q25) / 1.35)
    except Exception:
        approx_sigma = max(3e-3, (float(np.max(lines)) - float(np.min(lines))) / 4.0)

    def mse(mu: float, sigma: float) -> float:
        preds = np.array([normal_survival(x, mu, max(1e-6, sigma)) for x in lines])
        return float(np.mean((preds - p_over) ** 2))

    # coarse grid around mu0, approx_sigma
    mu_grid = np.linspace(mu0 - 40, mu0 + 40, 41)
    sig_grid = np.geomspace(
        max(0.5, approx_sigma / 3.0),
        approx_sigma * 3.0,
        25
    )

    best_mu, best_sig, best_err = mu0, approx_sigma, 1e9
    for mu in mu_grid:
        for sig in sig_grid:
            e = mse(mu, sig)
            if e < best_err:
                best_mu, best_sig, best_err = mu, sig, e

    # refine around best
    for _ in range(3):
        mu_grid = np.linspace(best_mu - 6, best_mu + 6, 25)
        sig_grid = np.geomspace(
            max(0.25, best_sig / 3.0),
            best_sig * 3.0,
            25
        )
        best_local_mu, best_local_sig, best_local_err = best_mu, best_sig, best_err
        for mu in mu_grid:
            for sig in sig_grid:
                e = mse(mu, sig)
                if e < best_local_err:
                    best_local_mu, best_local_sig, best_local_err = mu, sig, e
        best_mu, best_sig, best_err = best_local_mu, best_local_sig, best_local_err

    return float(best_mu), float(max(1e-3, best_sig))


def fit_poisson_from_points(lines: np.ndarray, p_over: np.ndarray) -> float:
    """
    Fit Poisson(lambda) so that P(K > line) best matches observed no-vig Over probs.
    We'll brute-force via coarse-to-fine geometric search.
    """

    idx_mid = int(np.argmin(np.abs(p_over - 0.5)))
    line_mid = float(lines[idx_mid])
    # rough starting lambda around ceil(line_mid)
    lam0 = max(0.05, math.floor(line_mid + 1e-9) + 0.5)

    def mse(lam: float) -> float:
        preds = np.array([
            poisson_survival_at_line(x, max(1e-4, lam))
            for x in lines
        ])
        return float(np.mean((preds - p_over) ** 2))

    lam_grid = np.geomspace(
        max(0.05, lam0 / 5.0),
        max(0.8, lam0 * 5.0),
        60
    )
    best_lam, best_err = lam0, 1e9
    for lam in lam_grid:
        e = mse(lam)
        if e < best_err:
            best_lam, best_err = lam, e

    # refine
    for _ in range(3):
        lam_grid = np.geomspace(
            max(0.03, best_lam / 3.0),
            best_lam * 3.0,
            60
        )
        local_lam, local_err = best_lam, best_err
        for lam in lam_grid:
            e = mse(lam)
            if e < local_err:
                local_lam, local_err = lam, e
        best_lam, best_err = local_lam, local_err

    return float(max(0.03, best_lam))


def fit_consensus_params(group: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Fit a consensus distribution for one (player, market) using all available
    (line -> no-vig Over prob) points across all books and alt lines.

    We REQUIRE actual variation. If the market hasn't given you enough signal
    (all same line, all same prob), we REFUSE to hallucinate. That protects
    bankroll from garbage edges on thin markets.

    Returns:
        {
            "dist": "normal" or "poisson",
            "mu": <float> or NaN,
            "sigma": <float> or NaN,
            "lam": <float> or NaN,
            "fit_note": "normal_consensus"/"poisson_consensus"
        }
    or None if we can't responsibly fit.
    """

    lines = group["point"].to_numpy(dtype=float)
    probs = group["p_over_novig"].to_numpy(dtype=float)

    # Basic sanity
    if len(lines) < 2:
        # You don't get a curve off one data point. That's how touts blow up.
        return None

    # We need meaningful spread in either line OR in implied probability.
    line_spread = float(np.max(lines) - np.min(lines))
    prob_spread = float(np.max(probs) - np.min(probs))

    # If the book cluster is basically saying the same thing at the same number,
    # we have no slope to learn. So no.
    if line_spread < 0.1 or prob_spread < 0.02:
        return None

    # clamp probs to sane [1e-4, 1-1e-4] so fitting doesn't explode
    probs = np.clip(probs, 1e-4, 1.0 - 1e-4)

    market = str(group["market"].iloc[0]).lower()

    if market in CONTINUOUS_MARKETS:
        mu, sigma = fit_normal_from_points(lines, probs)
        return {
            "dist": "normal",
            "mu": float(mu),
            "sigma": float(sigma),
            "lam": float("nan"),
            "fit_note": "normal_consensus"
        }

    elif market in POISSON_MARKETS:
        lam = fit_poisson_from_points(lines, probs)
        return {
            "dist": "poisson",
            "mu": float("nan"),
            "sigma": float("nan"),
            "lam": float(lam),
            "fit_note": "poisson_consensus"
        }

    else:
        # unsupported market category
        return None


def consensus_prob_over(params: Dict[str, float], line: float) -> float:
    """
    Given fitted consensus params and a specific stat line,
    return P(Over(line)) implied by that consensus curve.
    """
    if params["dist"] == "normal":
        return normal_survival(line, params["mu"], params["sigma"])
    else:
        return poisson_survival_at_line(line, params["lam"])


# -----------------------------------------------------------------------------
# PRICING / EV
# -----------------------------------------------------------------------------

def price_all(offers: pd.DataFrame) -> pd.DataFrame:
    """
    For every (player, market), fit the consensus distribution and use it
    to evaluate each book's Over/Under offer at that player's various lines.

    Adds:
        consensus_p_over
        consensus_p_under
        ev_over, ev_under
        edge_bp_over, edge_bp_under (how far book's own no-vig is from consensus)
        fair_over_american, fair_under_american (consensus -> fair price)
        consensus_mu, consensus_sigma, consensus_lam, fit_note

    Returns a giant candidate frame (not yet filtered).
    """

    if offers.empty:
        logger.warning("No two-way offers to price.")
        return offers.copy()

    groups = offers.groupby(["description", "market"], sort=False)
    chunks: List[pd.DataFrame] = []

    for (player, market), g in groups:
        params = fit_consensus_params(g)
        if params is None:
            # Not enough structure to trust. Skip this player+market entirely.
            continue

        gg = g.copy()

        # consensus probability of OVER/UNDER at each line
        gg["consensus_p_over"] = gg["point"].apply(
            lambda x: consensus_prob_over(params, float(x))
        )
        gg["consensus_p_under"] = 1.0 - gg["consensus_p_over"]

        # EV vs posted odds:
        # EV = p * dec - 1.0 (profit per $1 staked if we "win" with prob p)
        gg["ev_over"] = gg["consensus_p_over"] * gg["dec_over"] - 1.0
        gg["ev_under"] = gg["consensus_p_under"] * gg["dec_under"] - 1.0

        # Edge vs book's own no-vig fair probs (in basis points for readability)
        gg["edge_bp_over"] = (gg["consensus_p_over"] - gg["p_over_novig"]) * 10000.0
        gg["edge_bp_under"] = (gg["consensus_p_under"] - gg["p_under_novig"]) * 10000.0

        # Consensus fair American prices for Over and Under
        gg["fair_over_american"] = gg["consensus_p_over"].apply(prob_to_american)
        gg["fair_under_american"] = gg["consensus_p_under"].apply(prob_to_american)

        # Attach model params for audit
        if params["dist"] == "normal":
            gg["consensus_mu"] = params["mu"]
            gg["consensus_sigma"] = params["sigma"]
            gg["consensus_lam"] = np.nan
        else:
            gg["consensus_mu"] = np.nan
            gg["consensus_sigma"] = np.nan
            gg["consensus_lam"] = params["lam"]

        gg["fit_note"] = params.get("fit_note", "")

        chunks.append(gg)

    if not chunks:
        logger.warning("No consensus curves could be fit. Market is too thin or feed is junk.")
        cols = list(offers.columns) + [
            "consensus_p_over","consensus_p_under","ev_over","ev_under",
            "edge_bp_over","edge_bp_under",
            "fair_over_american","fair_under_american",
            "consensus_mu","consensus_sigma","consensus_lam","fit_note"
        ]
        return pd.DataFrame(columns=cols)

    priced = pd.concat(chunks, axis=0, ignore_index=True)
    logger.info(f"Consensus-priced {len(priced)} book offers across all players/markets")
    return priced


# -----------------------------------------------------------------------------
# PICK SELECTION / STAKING
# -----------------------------------------------------------------------------

def select_bets(
    priced: pd.DataFrame,
    bankroll: float,
    kelly_cap: float,
    ev_threshold: float,
    min_prob: float,
    side_bias: str,
    book_preference: Optional[List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each book/player/line offer:
        - Choose a side (Over or Under) based on EV unless side_bias forces it.
        - Throw out low-EV or low-winprob garbage.
        - Compute Kelly fraction (capped), stake $, and units.

    Returns:
        bets (filtered, ranked, final card)
        audit (full priced frame with all candidates)
    """

    df = priced.copy()
    if df.empty:
        logger.warning("No priced data. Nothing to select.")
        return df, priced

    def choose_side(row: pd.Series) -> pd.Series:
        """
        Decide whether we're betting Over or Under for this specific row.
        Also attach the metrics for that chosen side so downstream is clean.
        """
        if side_bias == "over":
            side = "OVER"
            ev = row["ev_over"]
            p = row["consensus_p_over"]
            dec = row["dec_over"]
            edge = row["edge_bp_over"]
            odds = row["over_odds"]
            fair_amer = row["fair_over_american"]
        elif side_bias == "under":
            side = "UNDER"
            ev = row["ev_under"]
            p = row["consensus_p_under"]
            dec = row["dec_under"]
            edge = row["edge_bp_under"]
            odds = row["under_odds"]
            fair_amer = row["fair_under_american"]
        else:
            # auto: take whichever side has higher EV
            if row["ev_over"] >= row["ev_under"]:
                side = "OVER"
                ev = row["ev_over"]
                p = row["consensus_p_over"]
                dec = row["dec_over"]
                edge = row["edge_bp_over"]
                odds = row["over_odds"]
                fair_amer = row["fair_over_american"]
            else:
                side = "UNDER"
                ev = row["ev_under"]
                p = row["consensus_p_under"]
                dec = row["dec_under"]
                edge = row["edge_bp_under"]
                odds = row["under_odds"]
                fair_amer = row["fair_under_american"]

        return pd.Series(
            [side, ev, p, dec, edge, odds, fair_amer],
            index=[
                "pick_side",
                "pick_ev",
                "pick_p",            # consensus win prob for chosen side
                "pick_dec",          # decimal odds for chosen side
                "pick_edge_bp",      # consensus - book no-vig, in basis points
                "pick_odds",         # book's American odds for that side
                "pick_fair_american" # consensus fair American for that side
            ]
        )

    picks = df.apply(choose_side, axis=1)
    df = pd.concat([df, picks], axis=1)

    # Threshold filters
    before_filter = len(df)
    df = df[(df["pick_ev"] >= ev_threshold) & (df["pick_p"] >= min_prob)]
    after_filter = len(df)
    logger.info(f"Filtered by EV >= {ev_threshold:.3f} and prob >= {min_prob:.3f}: "
                f"{before_filter} -> {after_filter} candidates")

    if df.empty:
        logger.warning("No candidates cleared EV/probability filters.")
        return df, priced

    # Kelly sizing w/ cap
    df["pick_kelly_frac"] = df.apply(
        lambda r: kelly_fraction(r["pick_p"], r["pick_dec"], cap=kelly_cap),
        axis=1
    )

    before_kelly = len(df)
    df = df[df["pick_kelly_frac"] > 0]
    after_kelly = len(df)
    logger.info(f"Dropped {before_kelly - after_kelly} zero/negative Kelly candidates")

    if df.empty:
        logger.warning("No candidates have +EV Kelly stake after cap. Done.")
        return df, priced

    # Stake sizing
    df["stake"] = (df["pick_kelly_frac"] * bankroll).round(2)

    # Define 1 betting unit as 1% of bankroll by default (you can change that downstream if you want)
    base_unit = max(1.0, round(0.01 * bankroll, 2))
    df["units"] = (df["stake"] / base_unit).round(2)

    # Book preference tiebreak
    if book_preference:
        pref_map = {b.strip(): i for i, b in enumerate(book_preference)}
        df["book_rank"] = df["bookmaker"].map(lambda b: pref_map.get(b, 999))
    else:
        df["book_rank"] = 999

    # Sort best -> worst:
    #   - higher EV first
    #   - then bigger consensus-vs-book edge_bp
    #   - then preferred book
    df = df.sort_values(
        by=["pick_ev", "pick_edge_bp", "book_rank"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # Final bet card columns (clean, presentation-ready)
    # This is what goes to bets.csv
    final_cols = [
        "bookmaker",
        "game_id", "commence_time", "in_play", "last_update",
        "home_team", "away_team",
        "market", "description", "point",
        "pick_side", "pick_odds", "pick_fair_american",
        "pick_p", "pick_ev",
        "stake", "units",
        "over_odds", "under_odds",
        "dec_over", "dec_under",
        "p_over_novig", "p_under_novig",
        "consensus_p_over", "consensus_p_under",
        "edge_bp_over", "edge_bp_under",
        "consensus_mu", "consensus_sigma", "consensus_lam",
        "fit_note"
    ]

    bets = df[final_cols].copy()

    # full audit dump includes EVERYTHING from 'priced' + pick_* and internals
    audit = df.copy()

    return bets, audit


# -----------------------------------------------------------------------------
# CLI / MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FoxEdge NFL Prop Market Mispricing Scanner (consensus vs book)."
    )
    parser.add_argument(
        "--odds_path",
        type=str,
        default="props_odds.csv",
        help="Path to input odds CSV (must have Over/Under rows for the same line)."
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=DEFAULT_BANKROLL,
        help="Bankroll in dollars (used for Kelly sizing)."
    )
    parser.add_argument(
        "--ev_threshold",
        type=float,
        default=DEFAULT_EV_THRESHOLD,
        help="Minimum EV to keep a pick (e.g. 0.02 = 2%)."
    )
    parser.add_argument(
        "--kelly_cap",
        type=float,
        default=DEFAULT_KELLY_CAP,
        help="Max Kelly fraction per pick (safety brake)."
    )
    parser.add_argument(
        "--min_prob",
        type=float,
        default=DEFAULT_MIN_PROB,
        help="Minimum consensus win probability for chosen side. "
             "Stops you from chasing +450 lotto tickets just because math likes them."
    )
    parser.add_argument(
        "--side_bias",
        type=str,
        default=DEFAULT_SIDE_BIAS,
        choices=["auto", "over", "under"],
        help="Force 'over'/'under' or let the script auto-select side with higher EV."
    )
    parser.add_argument(
        "--book_preference",
        type=str,
        default="",
        help="Comma-separated preferred books for tiebreak ordering, e.g. 'FD,DK,PB'."
    )
    parser.add_argument(
        "--bets_out",
        type=str,
        default="bets.csv",
        help="Output CSV path for final bet card."
    )
    parser.add_argument(
        "--audit_out",
        type=str,
        default="audit_all_candidates.csv",
        help="Output CSV path for full audit dump (all candidates, internals)."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity."
    )

    args = parser.parse_args()

    # init logging
    setup_logging(args.log_level)

    # Step 1: Load & clean raw lines/odds
    raw = load_prop_odds(args.odds_path)

    # Step 2: Merge Over/Under per line per book -> build offers
    offers = pivot_two_way(raw)
    if offers.empty:
        logger.error("No two-way markets found. "
                     "Your feed probably didn't include both Over and Under for the same line/book.")
        # Write empty outputs so downstream automation doesn't break
        pd.DataFrame().to_csv(args.audit_out, index=False)
        pd.DataFrame().to_csv(args.bets_out, index=False)
        return

    # Step 3: Price all offers vs market consensus
    priced = price_all(offers)
    if priced.empty:
        logger.error("No consensus curves could be fit. "
                     "You likely don't have enough alt lines / multiple books per player+market.")
        pd.DataFrame().to_csv(args.audit_out, index=False)
        pd.DataFrame().to_csv(args.bets_out, index=False)
        return

    # Step 4: Pick sides, filter, size bets
    book_pref_list = [b.strip() for b in args.book_preference.split(",")] if args.book_preference else None
    bets, audit = select_bets(
        priced=priced,
        bankroll=args.bankroll,
        kelly_cap=args.kelly_cap,
        ev_threshold=args.ev_threshold,
        min_prob=args.min_prob,
        side_bias=args.side_bias,
        book_preference=book_pref_list
    )

    # Step 5: Dump CSVs
    audit.to_csv(args.audit_out, index=False)
    bets.to_csv(args.bets_out, index=False)

    # Step 6: Console summary
    print(f"Candidates with consensus pricing: {len(priced):,}")
    print(f"Bets selected after filters: {len(bets):,}")

    if not bets.empty:
        total_stake = float(bets["stake"].sum())
        pct_bankroll = 100.0 * total_stake / max(1.0, args.bankroll)

        print(f"Total stake: ${total_stake:,.2f} ({pct_bankroll:.2f}% of bankroll)")

        # Top 10 preview for fast eyeballing / screenshots
        preview_cols = [
            "bookmaker",
            "description",
            "market",
            "point",
            "pick_side",
            "pick_odds",
            "pick_fair_american",
            "pick_p",
            "pick_ev",
            "stake",
            "units"
        ]
        top10 = bets[preview_cols].head(10)
        print("\nTop 10:")
        print(top10.to_string(index=False))
    else:
        print("No bets passed EV/prob/Kelly filters. Good. That means we didn't force action just to feel alive.")


if __name__ == "__main__":
    main()
