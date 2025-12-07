#!/usr/bin/env python3
"""
FoxEdge Prop Recommender — Daily Run Script
Reads a CSV of priced props (book & fair) and emits guardrailed, consensus-driven best bets.

USAGE (examples):
  # Strict multi-book
  python foxedge_prop_recommender.py \
      --input priced_props.csv \
      --out_csv recommendations.csv \
      --out_json recommendations.json \
      --min_books 2 --strong_edge_single 0.12 --max_edge_prop 0.18 \
      --max_abs_price 160 --min_ev 0.03 --kelly_cap 0.02 \
      --max_per_team_per_market 2 --top_per_market 5

  # Single-book rescue mode
  python foxedge_prop_recommender.py \
      --input priced_props.csv \
      --out_csv recommendationsa.csv \
      --out_json recommendationsa.json \
      --min_books 1 --strong_edge_single 0.06 --max_edge_prop 0.18 \
      --max_abs_price 160 --min_ev 0.03 --kelly_cap 0.02 \
      --max_per_team_per_market 2 --top_per_market 5 \
      --allow_single_book_ev --single_book_ev_floor 0.06

INPUT SCHEMA (required; extra columns ignored):
  game_id, commence_time, bookmaker, home_team, away_team,
  market, player, selection, line, book_price, fair_prob, fair_price

OUTPUTS:
  - CSV of filtered recommendations with audit fields
  - JSON array of plays (same fields) for programmatic use
Prints a per-market summary and blocker breakdown.
"""
import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------- Odds Utilities ----------------------
def american_to_prob(odds: float) -> float:
    """Implied probability from American odds (includes vig)."""
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_multiplier(odds: float) -> float:
    """Net payout multiplier b in Kelly formula."""
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def kelly_fraction(p: float, odds: float, kelly_cap: float = 0.02) -> float:
    """Kelly fraction with cap. Returns 0 if edge is negative/nonexistent."""
    if pd.isna(p) or pd.isna(odds):
        return 0.0
    b = american_to_multiplier(odds)
    f = (p * b - (1 - p)) / b
    if not np.isfinite(f) or f <= 0:
        return 0.0
    return float(min(f, kelly_cap))


def expected_value_per_dollar(p: float, odds: float) -> float:
    """EV per $1 staked using American odds."""
    if pd.isna(p) or pd.isna(odds):
        return np.nan
    b = american_to_multiplier(odds)
    return p * b - (1 - p)


# ---------------------- Core Pipeline ----------------------
KEY_COLS = ["game_id", "market", "player", "selection", "line"]

def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ---- Column normalization ----
    col_map = {}
    if "book_price" not in df.columns and "price" in df.columns:
        col_map["price"] = "book_price"
    if "line" not in df.columns and "point" in df.columns:
        col_map["point"] = "line"
    if "player" not in df.columns and "description" in df.columns:
        col_map["description"] = "player"
    if "selection" not in df.columns and "label" in df.columns:
        col_map["label"] = "selection"
    if col_map:
        df = df.rename(columns=col_map)

    # Required minimal columns after rename
    required = ["market", "player", "selection", "line", "book_price", "fair_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    # ---- Cleaning & types ----
    df["market"] = df["market"].astype(str).str.strip()
    df["selection"] = df["selection"].astype(str).str.strip().str.title()
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

    # Clean odds and line
    df["book_price"] = (
        df["book_price"].astype(str)
        .str.replace("−", "-", regex=False)
        .str.replace("\u2212", "-", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    df["book_price"] = pd.to_numeric(df["book_price"], errors="coerce").astype("Int64")

    df["line"] = df["line"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["line"] = pd.to_numeric(df["line"], errors="coerce")

    # Drop unusable
    df = df.dropna(subset=["book_price", "line", "fair_prob"]).copy()
    df["book_price"] = df["book_price"].astype(int)

    # Only Over/Under
    df = df[df["selection"].isin(["Over", "Under"])].copy()

    # Implied prob
    df["book_prob"] = df["book_price"].apply(american_to_prob)
    return df


def build_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across books for each unique prop key. Use bettor-best price for EV."""
    df = df.copy()
    df["edge_raw"] = df["fair_prob"] - df["book_prob"]
    df["ev_raw"] = df.apply(lambda r: expected_value_per_dollar(r["fair_prob"], r["book_price"]), axis=1)

    # Per-key: n_books, mean market prob, robust fair prob
    agg = (
        df.groupby(KEY_COLS, as_index=False)
          .agg(
              n_books=("bookmaker", "nunique"),
              mean_book_prob=("book_prob", "mean"),
              fair_prob=("fair_prob", "median"),
          )
    )

    # Best book for bettor (max price for the chosen side)
    idx_best = df.groupby(KEY_COLS)["book_price"].idxmax()
    best_tbl = (
        df.loc[idx_best, KEY_COLS + ["bookmaker", "book_price", "book_prob"]]
          .rename(columns={
              "bookmaker": "best_bookmaker",
              "book_price": "best_book_price",
              "book_prob": "best_book_prob",
          })
    )

    cons = agg.merge(best_tbl, on=KEY_COLS, how="left")

    # EV and consensus edge
    cons["ev"] = cons.apply(lambda r: expected_value_per_dollar(r["fair_prob"], r["best_book_price"]), axis=1)
    cons["edge"] = cons["fair_prob"] - cons["mean_book_prob"]

    return cons


def apply_guardrails(consensus: pd.DataFrame,
                     raw_df: pd.DataFrame,
                     min_books: int = 2,
                     strong_edge_single: float = 0.12,
                     max_edge_prop: float = 0.18,
                     max_abs_price: int = 160,
                     min_ev: float = 0.03,
                     kelly_cap: float = 0.02,
                     max_per_team_per_market: int = 2,
                     allow_single_book_ev: bool = True,
                     single_book_ev_floor: float = 0.06) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (cons_all_with_blockers, BETs_only). BETs has a stable schema for downstream.
    """
    expected_cols = KEY_COLS + [
        "home_team", "away_team", "n_books", "best_bookmaker", "best_book_price",
        "fair_prob", "mean_book_prob", "edge", "edge_capped", "ev",
        "kelly", "play"
    ]

    # Defensive empties
    if consensus is None or consensus.empty or raw_df is None or raw_df.empty:
        empty_all = (consensus if isinstance(consensus, pd.DataFrame) else pd.DataFrame()).copy()
        if "pass_reason" not in empty_all:
            empty_all["pass_reason"] = ""
        return empty_all, pd.DataFrame(columns=expected_cols)

    cons = consensus.copy()
    cons["pass_reason"] = ""

    # Require best price to be meaningfully better than market (avoid stale ties)
    # For dogs (plus odds): best_book_prob < mean_book_prob; for favs (minus odds): > mean
    cons["prob_improve"] = np.where(
        cons["best_book_price"] >= 0,
        cons["mean_book_prob"] - cons["best_book_prob"],   # lower better for dogs
        cons["best_book_prob"] - cons["mean_book_prob"]    # higher better for favs
    )
    # Dynamic improvement floor by consensus depth
    improve_floor = np.where(
        cons["n_books"] >= 5, 0.003,           # deep market
        np.where(cons["n_books"] >= 3, 0.005,  # medium consensus
                 0.010)                        # 1–2 books
    )
    cons["price_edge_ok"] = cons["prob_improve"] >= improve_floor
    cons.loc[~cons["price_edge_ok"], "pass_reason"] += "|price_edge"    

    # Book-count rule with smarter single-book EV floors by market/line
    # Default single-book EV floor
    sb_floor = np.full(len(cons), single_book_ev_floor, dtype=float)
    # Receptions 0.5 unders often spoof; require more EV
    mask_rec_05 = (cons["n_books"] == 1) & (cons["market"] == "player_receptions") & (cons["line"] == 0.5)
    sb_floor[mask_rec_05] = np.maximum(sb_floor[mask_rec_05], 0.08)
    # QB-ish rush attempts/yds with low lines: require even more EV
    mask_rush_low = (cons["n_books"] == 1) & (cons["market"].isin(["player_rush_attempts", "player_rush_yds"])) & (cons["line"] <= 2.5)
    sb_floor[mask_rush_low] = np.maximum(sb_floor[mask_rush_low], 0.10)

    # Single-book EV floors by market/line
    sb_floor = np.full(len(cons), single_book_ev_floor, dtype=float)

    mask_rec_05 = (cons["n_books"] == 1) & (cons["market"] == "player_receptions") & (cons["line"]     == 0.5)
    sb_floor[mask_rec_05] = np.maximum(sb_floor[mask_rec_05], 0.08)

    mask_rush_low = (cons["n_books"] == 1) & (cons["market"].isin(["player_rush_attempts",     "player_rush_yds"])) & (cons["line"] <= 2.5)
    sb_floor[mask_rush_low] = np.maximum(sb_floor[mask_rush_low], 0.10)

    # Multi-book EV floor by market (tighten noisy markets without killing volume)
    mb_floor = np.where(cons["market"] == "player_rush_attempts", 0.05, min_ev)

    # Book rule with floors
    mask_books = (cons["n_books"] >= min_books) | ((cons["n_books"] == 1) & (cons["edge"] >=     strong_edge_single))
    if allow_single_book_ev:
        mask_books |= ((cons["n_books"] == 1) & (cons["ev"].values >= sb_floor))

    # Apply multi-book EV floor
    mask_ev_ok = (cons["n_books"] >= 2) & (cons["ev"] >= mb_floor)

    # Final: must pass book rule AND either be single-book allowed or pass multi-book EV floor
    mask_keep = ( (cons["n_books"] == 1) & allow_single_book_ev & ((cons["ev"].values >= sb_floor) | (cons["edge"] >= strong_edge_single)) ) | \
                ( (cons["n_books"] >= 2) & mask_ev_ok ) | \
                ( (cons["n_books"] >= min_books) & (cons["edge"] >= 0) )
    cons.loc[~mask_keep, "pass_reason"] += "|books"

    # Price guard: worst absolute book price across raws per key
    price_span = (
        raw_df.groupby(KEY_COLS)
              .agg(worst_abs_price=("book_price", lambda s: float(np.max(np.where(s < 0, -s, s)))))
              .reset_index()
    )
    cons = cons.merge(price_span, on=KEY_COLS, how="left")
    cons["price_ok"] = cons["worst_abs_price"] <= max_abs_price
    cons.loc[~cons["price_ok"], "pass_reason"] += "|price"

    # EV floor
    cons["ev_ok"] = cons["ev"] >= min_ev
    cons.loc[~cons["ev_ok"], "pass_reason"] += "|ev"

    # Edge cap unless lots of agreement
    edge_cap = np.where(cons["n_books"] >= 3, 0.25, max_edge_prop)
    cons["edge_capped"] = np.minimum(cons["edge"], edge_cap)

    # Kelly
    # Scale Kelly cap by consensus depth: 1 book → 50% cap, 5+ books → 100% cap
    scale = np.clip((cons["n_books"] - 1) / 4.0, 0.5, 1.0)
    row_cap = kelly_cap * scale
    cons["kelly"] = cons.apply(
        lambda r: kelly_fraction(r["fair_prob"], r["best_book_price"],     kelly_cap=float(row_cap.loc[r.name])),
        axis=1
    )

    # Initial recommendation
    cons["rec"] = np.where(
        (cons["pass_reason"] == "") & (cons["edge_capped"] > 0) & (cons["kelly"] > 0),
        "BET",
        "PASS"
    )

    # Join teams for exposure checks
    first_rows = raw_df.sort_values("bookmaker").drop_duplicates(subset=KEY_COLS)
    team_cols = ["game_id", "home_team", "away_team", "market", "player", "selection", "line"]
    cons = cons.merge(first_rows[team_cols], on=KEY_COLS, how="left")

    # Exposure: keep at most N per team per market by EV rank
    def exposure_filter(group: pd.DataFrame) -> pd.DataFrame:
        if group is None or group.empty:
            return group if isinstance(group, pd.DataFrame) else pd.DataFrame(columns=expected_cols)
        kept_idx = []
        home_counts, away_counts = {}, {}
        for idx, row in group.sort_values("ev", ascending=False).iterrows():
            h = row.get("home_team", None)
            a = row.get("away_team", None)
            m = row.get("market", None)
            if h is not None and home_counts.get((m, h), 0) >= max_per_team_per_market:
                continue
            if a is not None and away_counts.get((m, a), 0) >= max_per_team_per_market:
                continue
            kept_idx.append(idx)
            if h is not None:
                home_counts[(m, h)] = home_counts.get((m, h), 0) + 1
            if a is not None:
                away_counts[(m, a)] = away_counts.get((m, a), 0) + 1
        return group.loc[kept_idx]

    cons_bets = cons[cons["rec"] == "BET"]
    cons_bets = exposure_filter(cons_bets)

    # Archetype throttle: limit QB rush Over spam to 1 per team per market
    if cons_bets is not None and not cons_bets.empty:
        def _is_qb(row) -> bool:
            name = str(row.get("player", "")).lower()
            # crude name list; replace with position data if available
            qb_tokens = ["mahomes","prescott","lawrence","stafford","allen","jackson","burrow","herbert",
                         "tua","hurts","rodgers","stroud","love","fields","murray","geno","goff","cousins",
                         "carr","bryce","mayfield","pickett","garoppolo","wilson","young","howell"]
            return any(t in name for t in qb_tokens)

        mask_qb_rush_over = (
            cons_bets["market"].isin(["player_rush_attempts", "player_rush_yds"])
            & cons_bets["selection"].eq("Over")
            & cons_bets.apply(_is_qb, axis=1)
        )
        if mask_qb_rush_over.any():
            tmp = cons_bets[mask_qb_rush_over].copy()
            keep_idx = []
            seen = set()
            for idx, r in tmp.sort_values("ev", ascending=False).iterrows():
                key = (r.get("home_team"), r.get("away_team"), r.get("market"))
                if key in seen:
                    continue
                seen.add(key)
                keep_idx.append(idx)
            cons_bets = pd.concat([cons_bets[~mask_qb_rush_over], cons_bets.loc[keep_idx]], ignore_index=False)

    # Reality brake: single-book, Kelly capped, tiny price improvement, modest EV -> PASS
    if cons_bets is not None and not cons_bets.empty and "prob_improve" in cons_bets:
        cons_bets["rec"] = np.where(
            (cons_bets["n_books"] == 1)
            & (cons_bets["kelly"] >= kelly_cap)
            & (cons_bets["prob_improve"] < 0.010)
            & (cons_bets["ev"] < 0.10),
            "PASS",
            "BET"
        )
        cons_bets = cons_bets[cons_bets["rec"] == "BET"]

    # If nothing survives, return both tables accordingly
    if cons_bets is None or cons_bets.empty:
        return cons, pd.DataFrame(columns=expected_cols)

    # Pretty 'play' string with best book
    cons_bets = cons_bets.copy()
    cons_bets["play"] = [
        (
            f"{r['away_team']} @ {r['home_team']} | {r['market']} | "
            f"{r['player']} {r['selection']} {r['line']} | "
            f"BestPrice: {int(r['best_book_price']) if pd.notna(r['best_book_price']) else 'NA'}"
            f" @ {r.get('best_bookmaker','?')} | "
            f"FairProb: {r['fair_prob']:.3f} | Edge: {r['edge_capped']:.1%} | "
            f"EV: {r['ev']:.2f} | Kelly: {r['kelly']:.3f} | Books: {int(r['n_books'])}"
        )
        for _, r in cons_bets.iterrows()
    ]

    # Ensure all expected columns exist
    for c in expected_cols:
        if c not in cons_bets.columns:
            cons_bets[c] = pd.Series(dtype="object")

    # Sort for display sensibly
    cons_bets = cons_bets.sort_values(["market", "ev", "edge_capped"], ascending=[True, False, False])

    return cons, cons_bets[expected_cols]


# ---------------------- Reporting ----------------------
def _summarize_blockers(cons_all: pd.DataFrame) -> dict:
    counts = {}
    if cons_all is None or cons_all.empty or "pass_reason" not in cons_all:
        return counts
    series = cons_all.loc[cons_all["pass_reason"] != "", "pass_reason"].astype(str)
    for reasons in series:
        for tag in reasons.split("|"):
            if tag:
                counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def print_summary_with_blockers(cons_all: pd.DataFrame,
                                cons_bets: pd.DataFrame,
                                top_per_market: int = 5) -> None:
    if cons_bets is None or cons_bets.empty:
        print("No BETs after guardrails.")
    else:
        print("=== FoxEdge Guardrailed Props — Top by Market ===")
        for market, grp in cons_bets.groupby("market"):
            print(f"\n[{market}]")
            for _, row in grp.head(top_per_market).iterrows():
                print(" - " + row["play"])

    print("\n=== Blockers (Global) ===")
    global_counts = _summarize_blockers(cons_all)
    if not global_counts:
        print(" none")
    else:
        print(" " + ", ".join(f"{k}:{v}" for k, v in global_counts.items()))
    # Per-market blockers
    if cons_all is not None and not cons_all.empty and "market" in cons_all:
        print("\n=== Blockers (Per-Market) ===")
        for mkt, sub in cons_all.groupby("market"):
            counts = _summarize_blockers(sub)
            if counts:
                top_line = ", ".join(f"{k}:{v}" for k, v in list(counts.items())[:5])
                print(f" {mkt}: {top_line}")

def print_top_n_global(cons_bets: pd.DataFrame, top_n: int = 10) -> None:
    if cons_bets is None or cons_bets.empty:
        print(f"\n=== Top {top_n} Props Across All Games ===\n No BETs available.")
        return

    print(f"\n=== Top {top_n} Props Across All Games ===")
    top_df = cons_bets.sort_values(["ev", "edge_capped", "kelly"], ascending=False).head(top_n)
    for _, row in top_df.iterrows():
        print(" - " + row["play"])

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="FoxEdge Prop Recommender — Daily Guardrailed Picks")
    ap.add_argument("--input", required=True, help="Path to input priced props CSV")
    ap.add_argument("--out_csv", required=True, help="Path to write recommendations CSV")
    ap.add_argument("--out_json", required=False, help="Path to write recommendations JSON")
    ap.add_argument("--min_books", type=int, default=2)
    ap.add_argument("--strong_edge_single", type=float, default=0.12)
    ap.add_argument("--max_edge_prop", type=float, default=0.18)
    ap.add_argument("--max_abs_price", type=int, default=160)
    ap.add_argument("--min_ev", type=float, default=0.03)
    ap.add_argument("--kelly_cap", type=float, default=0.02)
    ap.add_argument("--max_per_team_per_market", type=int, default=2)
    ap.add_argument("--top_per_market", type=int, default=5)
    ap.add_argument("--allow_single_book_ev", action="store_true",
                    help="Permit single-book props if EV ≥ single_book_ev_floor")
    ap.add_argument("--single_book_ev_floor", type=float, default=0.06,
                    help="EV floor for single-book props if allowed")
    ap.add_argument("--top_n_global", type=int, default=10,
                    help="Number of top bets to print across all games and markets")
    args = ap.parse_args()

    # Load and process
    df = load_input(args.input)
    if df.empty:
        print("Input is empty or missing required columns.", file=sys.stderr)
        sys.exit(2)

    consensus = build_consensus(df)
    cons_all, cons_bets = apply_guardrails(
        consensus, df,
        min_books=args.min_books,
        strong_edge_single=args.strong_edge_single,
        max_edge_prop=args.max_edge_prop,
        max_abs_price=args.max_abs_price,
        min_ev=args.min_ev,
        kelly_cap=args.kelly_cap,
        max_per_team_per_market=args.max_per_team_per_market,
        allow_single_book_ev=args.allow_single_book_ev,
        single_book_ev_floor=args.single_book_ev_floor
    )

    cols_out = KEY_COLS + [
        "home_team", "away_team", "n_books", "best_bookmaker", "best_book_price",
        "fair_prob", "mean_book_prob", "edge", "edge_capped", "ev", "kelly", "play"
    ]

    # Save CSV/JSON with stable schema even if empty
    if cons_bets is None or cons_bets.empty:
        pd.DataFrame(columns=cols_out).to_csv(args.out_csv, index=False)
        if args.out_json:
            pd.DataFrame(columns=cols_out).to_json(args.out_json, orient="records")
        print("No BETs after guardrails. Market tight or your input is thin.")
        return

    cons_bets[cols_out].to_csv(args.out_csv, index=False)
    if args.out_json:
        cons_bets[cols_out].to_json(args.out_json, orient="records")

    # Print summary + blockers
    print_summary_with_blockers(cons_all, cons_bets, top_per_market=args.top_per_market)
    print_top_n_global(cons_bets, top_n=args.top_n_global)


if __name__ == "__main__":
    main()
