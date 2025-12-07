#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NFL Props Market Pricer (schema: prop_odds.csv)

This script prices NFL player props using ONLY current lines and odds
from a CSV named `prop_odds.csv` with the following columns:

  game_id, commence_time, in_play, bookmaker, last_update,
  home_team, away_team, market, label, description, price, point

Where:
  - market: one of
      player_pass_attempts,
      player_pass_completions,
      player_pass_tds,
      player_pass_yds,
      player_reception_yds,
      player_receptions,
      player_rush_attempts,
      player_rush_yds,
      player_tds_over
  - label: 'over' or 'under' (case-insensitive)
  - description: player name (e.g., "Travis Kelce")
  - price: American odds (e.g., -115, +105)
  - point: the line (float)

Pipeline
--------
1) Read prop_odds.csv, sanitize, and rebuild two-way markets by pivoting
   Over/Under rows into single offers per (bookmaker, player, market, line).
2) Convert odds to decimal, remove vig to get each book's fair two-way probs.
3) Fit a consensus distribution per (player, market) from all available
   lines and no-vig Over probabilities:
     - Normal for continuous markets (yards, attempts, receptions)
     - Poisson for touchdown counts
4) Use the fitted curve to compute a consensus fair Over prob at each line.
5) Compute EV vs each book's posted price, plus capped Kelly sizing.
6) Output:
     - bets.csv                 (filtered, ranked card)
     - audit_all_candidates.csv (everything with full internals)
     - console summary

CLI
---
python nfl_props_market_pricer.py \
  --bankroll 10000 \
  --ev_threshold 0.02 \
  --kelly_cap 0.02 \
  --min_prob 0.45 \
  --side_bias auto \
  --book_preference "FD,DK,PB" \
  --bets_out propswk8.csv \
  --audit_out audit_all_candidates.csv

Notes
-----
- You MUST have multiple books and/or alt lines per (player, market) to fit
  anything reliable. One line is a coin flip with lipstick.
- This prices *misalignment* vs the market-implied consensus. If you want
  fundamental projections, that’s a different pipeline.

Author: someone who’s tired of hand-wavy prop “models.”
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# ----------------------------- Config -------------------------------------- #

# Markets treated as approximately continuous (Normal)
CONTINUOUS_MARKETS = {
    "player_pass_attempts",
    "player_pass_completions",
    "player_pass_yds",
    "player_reception_yds",
    "player_receptions",
    "player_rush_attempts",
    "player_rush_yds",
}

# Markets treated as count/rare (Poisson)
POISSON_MARKETS = {
    "player_pass_tds",
    "player_tds_over",
}

SUPPORTED_MARKETS = CONTINUOUS_MARKETS | POISSON_MARKETS


# ----------------------------- Odds Math ----------------------------------- #

def american_to_decimal(american: float) -> float:
    if american == 0:
        raise ValueError("American odds cannot be 0.")
    return 1 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def decimal_to_prob(decimal: float) -> float:
    if decimal <= 1:
        raise ValueError("Decimal odds must be > 1.")
    return 1.0 / decimal


def no_vig_two_way(over_american: float, under_american: float) -> Tuple[float, float]:
    """
    Remove vig from a two-way market using proportional normalization.
    Returns (p_over_fair, p_under_fair); sums to 1.
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
    """
    if not (0 < p < 1):
        return 0.0
    b = dec_odds - 1.0
    q = 1.0 - p
    k = (b * p - q) / b if b > 0 else 0.0
    if k <= 0:
        return 0.0
    return min(k, cap)


# ------------------------- Distribution Utilities -------------------------- #

def normal_survival(x: float, mu: float, sigma: float) -> float:
    """
    Survival function P(X > x) for Normal(mu, sigma).
    """
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def poisson_cdf(k: int, lam: float) -> float:
    """
    CDF P(K <= k) for Poisson(lam) using direct sum.
    Works fine for the small lam typically seen in TD props.
    """
    if lam <= 0:
        return 1.0 if k >= 0 else 0.0
    ks = np.arange(0, max(0, k) + 1)
    terms = np.exp(-lam) * (lam ** ks) / np.array([math.factorial(int(t)) for t in ks])
    return float(terms.sum())


def poisson_survival_at_line(line: float, lam: float) -> float:
    """
    For Over(line) such as 0.5, 1.5, 2.5, etc., need K >= ceil(line).
    """
    threshold = math.floor(line + 1e-9) + 1
    return max(0.0, min(1.0, 1.0 - poisson_cdf(threshold - 1, lam)))


# --------------------- Fit consensus from market points -------------------- #

def fit_normal_from_points(lines: np.ndarray, p_over: np.ndarray) -> Tuple[float, float]:
    """
    Least-squares fit for Normal(mu, sigma), coarse-to-fine grid search.
    """
    idx_mid = np.argmin(np.abs(p_over - 0.5))
    mu0 = float(lines[idx_mid])

    try:
        hi_lines = lines[p_over <= 0.16]
        lo_lines = lines[p_over >= 0.84]
        if len(hi_lines) and len(lo_lines):
            approx_sigma = max(3e-3, abs(np.median(hi_lines) - np.median(lo_lines)) / 2.0)
        else:
            approx_sigma = max(3e-3, float(np.subtract(*np.percentile(lines, [75, 25]))) / 1.35)
    except Exception:
        approx_sigma = max(3e-3, (np.max(lines) - np.min(lines)) / 4.0)

    def sse(mu: float, sigma: float) -> float:
        preds = np.array([normal_survival(x, mu, max(1e-6, sigma)) for x in lines])
        return float(np.mean((preds - p_over) ** 2))

    mu_grid = np.linspace(mu0 - 40, mu0 + 40, 41)
    sig_grid = np.geomspace(max(0.5, approx_sigma / 3), approx_sigma * 3, 25)

    best = (mu0, approx_sigma, 1e9)
    for mu in mu_grid:
        for sig in sig_grid:
            e = sse(mu, sig)
            if e < best[2]:
                best = (mu, sig, e)

    mu_c, sig_c, _ = best
    for _ in range(3):
        mu_grid = np.linspace(mu_c - 6, mu_c + 6, 25)
        sig_grid = np.geomspace(max(0.25, sig_c / 3), sig_c * 3, 25)
        best = (mu_c, sig_c, 1e9)
        for mu in mu_grid:
            for sig in sig_grid:
                e = sse(mu, sig)
                if e < best[2]:
                    best = (mu, sig, e)
        mu_c, sig_c, _ = best

    return float(mu_c), float(max(1e-3, sig_c))


def fit_poisson_from_points(lines: np.ndarray, p_over: np.ndarray) -> float:
    """
    Least-squares fit for Poisson(lam) using survival at each line.
    """
    idx_mid = np.argmin(np.abs(p_over - 0.5))
    line_mid = float(lines[idx_mid])
    lam0 = max(0.05, math.floor(line_mid + 1e-9) + 0.5)

    def sse(lam: float) -> float:
        preds = np.array([poisson_survival_at_line(x, max(1e-4, lam)) for x in lines])
        return float(np.mean((preds - p_over) ** 2))

    lam_grid = np.geomspace(max(0.05, lam0 / 5.0), max(0.8, lam0 * 5.0), 60)
    best = (lam0, 1e9)
    for lam in lam_grid:
        e = sse(lam)
        if e < best[1]:
            best = (lam, e)

    lam_c, _ = best
    for _ in range(3):
        lam_grid = np.geomspace(max(0.03, lam_c / 3.0), lam_c * 3.0, 60)
        best = (lam_c, 1e9)
        for lam in lam_grid:
            e = sse(lam)
            if e < best[1]:
                best = (lam, e)
        lam_c, _ = best

    return float(max(0.03, lam_c))


# ----------------------------- IO & Prep ----------------------------------- #

def load_prop_odds(path: str = "prop_odds.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "game_id", "commence_time", "in_play", "bookmaker", "last_update",
        "home_team", "away_team", "market", "label", "description", "price", "point"
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"prop_odds.csv missing columns: {missing}")

    # Sanitize
    df = df.copy()
    df["market"] = df["market"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["description"] = df["description"].astype(str).str.strip()
    df["bookmaker"] = df["bookmaker"].astype(str).str.strip()
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")

    # Keep supported markets only
    df = df[df["market"].isin(SUPPORTED_MARKETS)].dropna(subset=["point", "price"])
    # Only rows that are Over/Under
    df = df[df["label"].isin(["over", "under"])]

    return df


def pivot_two_way(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn Over/Under rows into single offers per book/line.
    Grouping keys include the line, bookmaker, player, market, and game context.
    """
    keys = [
        "game_id", "commence_time", "in_play", "bookmaker", "last_update",
        "home_team", "away_team", "market", "description", "point"
    ]
    # Build separate over/under subsets
    over = df[df["label"] == "over"][keys + ["price"]].rename(columns={"price": "over_odds"})
    under = df[df["label"] == "under"][keys + ["price"]].rename(columns={"price": "under_odds"})

    # Merge to require both sides present
    merged = pd.merge(over, under, on=keys, how="inner")

    # Drop duplicates if any
    merged = merged.drop_duplicates(keys + ["over_odds", "under_odds"])

    # Convert to numeric types
    merged["point"] = merged["point"].astype(float)
    merged["over_odds"] = merged["over_odds"].astype(int)
    merged["under_odds"] = merged["under_odds"].astype(int)

    # Derive decimals and no-vig
    fair = merged.apply(lambda r: no_vig_two_way(r["over_odds"], r["under_odds"]), axis=1, result_type="expand")
    merged[["p_over_novig", "p_under_novig"]] = fair
    merged["dec_over"] = merged["over_odds"].apply(american_to_decimal)
    merged["dec_under"] = merged["under_odds"].apply(american_to_decimal)
    return merged


# ------------------------------- Pricing ----------------------------------- #

def fit_consensus_params(group: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Fit a consensus distribution for one (player, market) using all available lines
    and no-vig Over probabilities across books and alt lines.
    """
    lines = group["point"].to_numpy(dtype=float)
    probs = group["p_over_novig"].to_numpy(dtype=float)

    # Need variation
    if (np.max(lines) - np.min(lines) < 0.1) or (np.max(probs) - np.min(probs) < 0.02):
        return None

    probs = np.clip(probs, 1e-4, 1 - 1e-4)
    market = str(group["market"].iloc[0])

    if market in CONTINUOUS_MARKETS:
        mu, sigma = fit_normal_from_points(lines, probs)
        return {"dist": "normal", "mu": float(mu), "sigma": float(sigma)}
    elif market in POISSON_MARKETS:
        lam = fit_poisson_from_points(lines, probs)
        return {"dist": "poisson", "lam": float(lam)}
    return None


def consensus_prob_over(params: Dict[str, float], line: float) -> float:
    if params["dist"] == "normal":
        return normal_survival(line, params["mu"], params["sigma"])
    return poisson_survival_at_line(line, params["lam"])


def price_all(offers: pd.DataFrame) -> pd.DataFrame:
    """
    For each (player, market), fit consensus params, then compute:
      - consensus_p_over/under at each offer line
      - EV for Over/Under vs each book price
      - edge vs each book's no-vig fair
    """
    if offers.empty:
        return offers

    groups = offers.groupby(["description", "market"], sort=False)
    chunks: List[pd.DataFrame] = []

    for (player, market), g in groups:
        params = fit_consensus_params(g)
        if params is None:
            continue

        gg = g.copy()
        gg["consensus_p_over"] = gg["point"].apply(lambda x: consensus_prob_over(params, float(x)))
        gg["consensus_p_under"] = 1.0 - gg["consensus_p_over"]

        # EV vs posted odds
        gg["ev_over"] = gg["consensus_p_over"] * gg["dec_over"] - 1.0
        gg["ev_under"] = gg["consensus_p_under"] * gg["dec_under"] - 1.0

        # Edge vs each book's own no-vig fair
        gg["edge_bp_over"] = (gg["consensus_p_over"] - gg["p_over_novig"]) * 10000.0
        gg["edge_bp_under"] = (gg["consensus_p_under"] - gg["p_under_novig"]) * 10000.0

        if params["dist"] == "normal":
            gg["consensus_mu"] = params["mu"]
            gg["consensus_sigma"] = params["sigma"]
            gg["consensus_lam"] = np.nan
        else:
            gg["consensus_mu"] = np.nan
            gg["consensus_sigma"] = np.nan
            gg["consensus_lam"] = params["lam"]

        chunks.append(gg)

    if not chunks:
        return pd.DataFrame(columns=list(offers.columns) + [
            "consensus_p_over","consensus_p_under","ev_over","ev_under",
            "edge_bp_over","edge_bp_under","consensus_mu","consensus_sigma","consensus_lam"
        ])

    return pd.concat(chunks, axis=0, ignore_index=True)


def select_bets(priced: pd.DataFrame,
                bankroll: float,
                kelly_cap: float,
                ev_threshold: float,
                min_prob: float,
                side_bias: str,
                book_preference: Optional[List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pick a side per offer by EV (or forced bias), filter by thresholds,
    apply capped Kelly staking, and rank.
    """
    df = priced.copy()
    if df.empty:
        return df, priced

    def choose_side(r):
        if side_bias == "over":
            side = "OVER"; ev = r["ev_over"]; p = r["consensus_p_over"]; dec = r["dec_over"]; edge = r["edge_bp_over"]; odds = r["over_odds"]
        elif side_bias == "under":
            side = "UNDER"; ev = r["ev_under"]; p = r["consensus_p_under"]; dec = r["dec_under"]; edge = r["edge_bp_under"]; odds = r["under_odds"]
        else:
            if r["ev_over"] >= r["ev_under"]:
                side = "OVER"; ev = r["ev_over"]; p = r["consensus_p_over"]; dec = r["dec_over"]; edge = r["edge_bp_over"]; odds = r["over_odds"]
            else:
                side = "UNDER"; ev = r["ev_under"]; p = r["consensus_p_under"]; dec = r["dec_under"]; edge = r["edge_bp_under"]; odds = r["under_odds"]
        return pd.Series([side, ev, p, dec, edge, odds],
                         index=["pick_side","pick_ev","pick_p","pick_dec","pick_edge_bp","pick_odds"])

    picks = df.apply(choose_side, axis=1)
    df = pd.concat([df, picks], axis=1)

    # Thresholds
    df = df[(df["pick_ev"] >= ev_threshold) & (df["pick_p"] >= min_prob)]
    if df.empty:
        return df, priced

    # Kelly with cap
    df["pick_kelly_frac"] = df.apply(lambda r: kelly_fraction(r["pick_p"], r["pick_dec"], cap=kelly_cap), axis=1)
    df = df[df["pick_kelly_frac"] > 0]
    if df.empty:
        return df, priced

    df["stake"] = (df["pick_kelly_frac"] * bankroll).round(2)
    base_unit = max(1.0, round(0.01 * bankroll, 2))
    df["units"] = (df["stake"] / base_unit).round(2)

    # Book tiebreaks
    if book_preference:
        pref_map = {b.strip(): i for i, b in enumerate(book_preference)}
        df["book_rank"] = df["bookmaker"].map(lambda b: pref_map.get(b, 999))
    else:
        df["book_rank"] = 999

    df = df.sort_values(by=["pick_ev","pick_edge_bp","book_rank"], ascending=[False, False, True])

    # Final bet card columns
    keep = [
        "bookmaker","game_id","commence_time","home_team","away_team",
        "market","description","point",
        "pick_side","pick_odds","pick_p","pick_ev","stake","units",
        "over_odds","under_odds","dec_over","dec_under",
        "p_over_novig","p_under_novig","consensus_p_over","consensus_p_under",
        "edge_bp_over","edge_bp_under","consensus_mu","consensus_sigma","consensus_lam",
        "last_update","in_play"
    ]
    bets = df[keep].reset_index(drop=True)
    return bets, priced


# --------------------------------- Main ------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="NFL props pricing from prop_odds.csv (lines+odds only).")
    ap.add_argument("--odds_path", type=str, default="prop_odds.csv", help="Path to prop_odds.csv")
    ap.add_argument("--bankroll", type=float, default=5000.0, help="Bankroll in dollars.")
    ap.add_argument("--ev_threshold", type=float, default=0.02, help="Minimum EV to include (e.g., 0.02 = 2%).")
    ap.add_argument("--kelly_cap", type=float, default=0.02, help="Max Kelly fraction per bet.")
    ap.add_argument("--min_prob", type=float, default=0.45, help="Minimum consensus probability for chosen side.")
    ap.add_argument("--side_bias", type=str, default="auto", choices=["auto","over","under"], help="Force side or auto by EV.")
    ap.add_argument("--book_preference", type=str, default="", help="Comma-separated preferred books (for tie-breaks).")
    ap.add_argument("--bets_out", type=str, default="bets.csv", help="Output bet card path.")
    ap.add_argument("--audit_out", type=str, default="audit_all_candidates.csv", help="Output audit dump path.")
    args = ap.parse_args()

    raw = load_prop_odds(args.odds_path)
    offers = pivot_two_way(raw)

    if offers.empty:
        print("No two-way markets found. You likely lack matching Over/Under rows per line-book. Fix your feed.")
        return

    priced = price_all(offers)
    if priced.empty:
        print("Unable to fit consensus curves. You need multiple books and/or alt lines per player+market.")
        return

    book_pref = [b.strip() for b in args.book_preference.split(",")] if args.book_preference else None
    bets, audit = select_bets(
        priced=priced,
        bankroll=args.bankroll,
        kelly_cap=args.kelly_cap,
        ev_threshold=args.ev_threshold,
        min_prob=args.min_prob,
        side_bias=args.side_bias,
        book_preference=book_pref
    )

    audit.to_csv(args.audit_out, index=False)
    bets.to_csv(args.bets_out, index=False)

    print(f"Candidates with fitted consensus: {len(priced):,}")
    print(f"Bets selected: {len(bets):,}")
    if not bets.empty:
        total_stake = bets['stake'].sum()
        print(f"Total stake: ${total_stake:,.2f} ({100*total_stake/max(1,args.bankroll):.2f}% of bankroll)")
        print("\nTop 10:")
        print(bets[[
            "bookmaker","description","market","point","pick_side","pick_odds",
            "pick_p","pick_ev","stake","units"
        ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
