#!/usr/bin/env python3
# nfl_fair_pricer.py
# FoxEdge NFL Fair Pricer v1.3 (merged)
#
# Purpose:
# - Scrape DK Network NFL splits (handle %, bet %, odds, lines)
# - Normalize into structured markets: moneyline, spread, total
# - Infer home/away, align lines to home team context
# - Convert odds -> implied prob -> no-vig prob
# - Apply handle/bet tilt as "sharp money pressure" (logit shift)
# - Model push probability for key numbers (spread & total)
# - Produce fair price, edge %, EV, and capped Kelly-style strength
# - Surface ranked betting candidates in Streamlit
#
# Key upgrades in this merged file:
# - Uses enhanced scraper with fallback + validation
# - Adds data health block so you SEE when scrape is trash
# - Adds CSV override for push probability tables
# - Renames "kelly" to "strength_score" for safety/comms
# - Adds "edge_source" tag so you know WHY it's popping
#
# Assumptions / warnings:
# - This is a pricing radar, not a fire control system.
# - Lines come from the same book you're "evaluating."
# - Handle/bet data can be bait. Treat it as signal-ish, not gospel.
# - Push tables are either heuristics or your CSV overrides. If those are wrong,
#   your spread/total fair odds are fantasy cosplay.
#
# You are responsible for sanity checking before staking or publishing.

import math
import time
import datetime
import re
from typing import Dict, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import streamlit as st

############################
# -------- SETTINGS -------
############################

APP_VERSION = "FoxEdge NFL Fair Pricer v1.3"
DEFAULT_EVENT_GROUP = "88808"  # NFL regular/weekly slate group used in both of your originals

USER_AGENT_HEADER = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT = 10

# Hardcoded push probability tables from baseline script
# These are heuristics. You should override with CSV in the sidebar for production.
EMP_SPREAD_PUSH_DEFAULT = {
    # key_number : push probability estimate on that exact number
    -10.0: 0.02,
    -9.5: 0.00,
    -9.0: 0.02,
    -8.5: 0.00,
    -8.0: 0.02,
    -7.5: 0.00,
    -7.0: 0.06,
    -6.5: 0.00,
    -6.0: 0.03,
    -5.5: 0.00,
    -5.0: 0.02,
    -4.5: 0.00,
    -4.0: 0.02,
    -3.5: 0.00,
    -3.0: 0.08,
    -2.5: 0.00,
    -2.0: 0.03,
    -1.5: 0.00,
    -1.0: 0.02,
    -0.5: 0.00,
    0.0:  0.07,
    0.5:  0.00,
    1.0:  0.02,
    1.5:  0.00,
    2.0:  0.03,
    2.5:  0.00,
    3.0:  0.08,
    3.5:  0.00,
    4.0:  0.02,
    4.5:  0.00,
    5.0:  0.02,
    5.5:  0.00,
    6.0:  0.03,
    6.5:  0.00,
    7.0:  0.06,
    7.5:  0.00,
    8.0:  0.02,
    8.5:  0.00,
    9.0:  0.02,
    9.5:  0.00,
    10.0: 0.02,
}

# Totals push probabilities broken into environment buckets.
# Baseline version scoped "low/mid/high" environments. We'll keep that API.
EMP_TOTAL_PUSH_DEFAULT = {
    # total_number : {bucket_name: push_prob}
    37.0: {"low": 0.05, "mid": 0.02, "high": 0.01},
    38.0: {"low": 0.05, "mid": 0.02, "high": 0.01},
    39.0: {"low": 0.05, "mid": 0.02, "high": 0.01},
    40.0: {"low": 0.06, "mid": 0.03, "high": 0.02},
    41.0: {"low": 0.06, "mid": 0.03, "high": 0.02},
    42.0: {"low": 0.06, "mid": 0.03, "high": 0.02},
    43.0: {"low": 0.05, "mid": 0.03, "high": 0.02},
    44.0: {"low": 0.05, "mid": 0.03, "high": 0.02},
    45.0: {"low": 0.05, "mid": 0.03, "high": 0.02},
    46.0: {"low": 0.03, "mid": 0.03, "high": 0.02},
    47.0: {"low": 0.03, "mid": 0.03, "high": 0.02},
    48.0: {"low": 0.03, "mid": 0.03, "high": 0.02},
    49.0: {"low": 0.02, "mid": 0.03, "high": 0.02},
    50.0: {"low": 0.02, "mid": 0.03, "high": 0.02},
    51.0: {"low": 0.02, "mid": 0.03, "high": 0.02},
    52.0: {"low": 0.02, "mid": 0.03, "high": 0.03},
    53.0: {"low": 0.02, "mid": 0.03, "high": 0.03},
    54.0: {"low": 0.02, "mid": 0.03, "high": 0.03},
    55.0: {"low": 0.01, "mid": 0.02, "high": 0.03},
    56.0: {"low": 0.01, "mid": 0.02, "high": 0.03},
    57.0: {"low": 0.01, "mid": 0.02, "high": 0.03},
}

LOW_TOTAL_CUTOFF = 41.0
HIGH_TOTAL_CUTOFF = 49.5

# sanity clamps for data; if market data violates these, we flag/drop it
MAX_ABS_SPREAD_OK = 30.0
MIN_TOTAL_OK = 20.0
MAX_TOTAL_OK = 75.0
MIN_ODDS_OK = -2000
MAX_ODDS_OK = 2000

#############################################
# -------------- UTILITIES -----------------
#############################################


def american_to_decimal(odds: float) -> Optional[float]:
    """Convert American odds to decimal odds."""
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    if o > 0:
        return 1.0 + (o / 100.0)
    else:
        return 1.0 + (100.0 / abs(o))


def american_to_prob(odds: float) -> Optional[float]:
    """Convert American odds to implied probability (includes vig)."""
    dec = american_to_decimal(odds)
    if dec is None or dec <= 1e-9:
        return None
    return 1.0 / dec


def prob_to_american(p: float) -> Optional[float]:
    """
    Convert no-vig probability p (win excluding push) into American odds.
    Assumes 0 < p < 1.
    """
    if p is None:
        return None
    if p <= 0 or p >= 1:
        return None
    # decimal odds
    dec = 1.0 / p
    if dec >= 2.0:
        # plus money
        return round((dec - 1.0) * 100.0, 0)
    else:
        # minus money
        return round(-100.0 / (dec - 1.0), 0)


def logit(x: float) -> float:
    """logit transform with safety."""
    x = min(max(x, 1e-9), 1 - 1e-9)
    return math.log(x / (1 - x))


def logistic(z: float) -> float:
    """inverse logit."""
    return 1.0 / (1.0 + math.exp(-z))


def remove_vig_two_way(p_a: float, p_b: float) -> Tuple[float, float]:
    """
    Normalize two implied probabilities so they sum to 1 (standard de-vig).
    Returns (p_a_novig, p_b_novig).
    """
    if p_a is None or p_b is None:
        return (None, None)
    s = p_a + p_b
    if s <= 1e-12:
        return (None, None)
    return p_a / s, p_b / s


def kelly_fraction(p_win: float, p_push: float, odds: float) -> Optional[float]:
    """
    Kelly fraction for markets with win/push/lose structure (spread/total).
    odds is American odds for the side we're considering.
    Assumes risk = 1u.
    EV math:
        outcome win: profit = payout
        outcome push: profit = 0
        outcome lose: profit = -1
    We solve standard Kelly on net EV over risk.
    """
    # Convert offered odds -> decimal -> payout multiple on win
    dec = american_to_decimal(odds)
    if dec is None:
        return None
    b = dec - 1.0  # net profit per 1 risked
    if b <= 0:
        return None

    # effective loss prob is (1 - p_win - p_push)
    p_loss = 1.0 - p_win - p_push
    if p_loss < 0 or p_loss > 1:
        return None

    # Kelly: f* = (b*p_win - p_loss) / b
    numer = b * p_win - p_loss
    denom = b
    if abs(denom) < 1e-12:
        return None
    f = numer / denom
    return f


def kelly_fraction_moneyline(p_win: float, odds: float) -> Optional[float]:
    """
    Simpler Kelly for moneyline (no push unless it's a tie market, which NFL ML isn't).
    odds is American odds for the side in question.
    """
    dec = american_to_decimal(odds)
    if dec is None:
        return None
    b = dec - 1.0
    if b <= 0:
        return None
    p_loss = 1.0 - p_win
    numer = b * p_win - p_loss
    denom = b
    if abs(denom) < 1e-12:
        return None
    return numer / denom


def safe_float(x):
    try:
        return float(str(x).replace("%","").strip())
    except:
        return None


def parse_percentage(txt: str) -> Optional[float]:
    """
    Takes something like '57%' or '57.3%' and returns 0.573.
    """
    if txt is None:
        return None
    m = re.findall(r"[\d.]+", str(txt))
    if not m:
        return None
    try:
        val = float(m[0]) / 100.0
        return val
    except:
        return None


def odds_from_text(raw: str) -> Optional[int]:
    """
    Extract American odds from text like '-110' '+105'.
    """
    if raw is None:
        return None
    m = re.findall(r"[-+]\d+", raw)
    if not m:
        return None
    try:
        v = int(m[0])
    except:
        return None
    # sanity check
    if v < MIN_ODDS_OK or v > MAX_ODDS_OK:
        return None
    return v


def line_from_text(raw: str) -> Optional[float]:
    """
    Extract numeric line (spread or total). Ex: '-2.5', '45.5'
    """
    if raw is None:
        return None
    m = re.findall(r"[-+]?\d+(\.\d+)?", raw)
    if not m:
        return None
    try:
        v = float(m[0] if isinstance(m[0], str) else m[0][0])
    except:
        # fallback if match is tuple-like
        try:
            v = float(m[0][0])
        except:
            return None
    return v


def total_bucket(total_line: float) -> str:
    """
    Classify total environment into 'low', 'mid', 'high'.
    Used to pull push probability for totals.
    """
    if total_line is None:
        return "mid"
    if total_line <= LOW_TOTAL_CUTOFF:
        return "low"
    if total_line >= HIGH_TOTAL_CUTOFF:
        return "high"
    return "mid"


# ---------------------- ODDS UPLOAD HELPERS ----------------------
import io

def parse_uploaded_odds(df_ou: pd.DataFrame) -> pd.DataFrame:
    """
    Ingest an uploaded odds CSV in either:

      1) Raw schema (book feed):
         - market, point, price, home_team, away_team, [label]

      2) Template schema (hand-built):
         - Matchup, Bookmaker Line, Over Price, Under Price,
           Home ML, Away ML, [Spread Line, Spread Price]

    Returns a DataFrame with one row per matchup:
      ['matchup','book_line','over_price','under_price',
       'home_ml','away_ml','spread_line','spread_price']
    """
    if df_ou is None or df_ou.empty:
        return pd.DataFrame(
            columns=[
                "matchup", "book_line", "over_price", "under_price",
                "home_ml", "away_ml", "spread_line", "spread_price",
            ]
        )

    df_ou = df_ou.copy()
    df_ou.columns = [c.strip().lower() for c in df_ou.columns]

    # Case 1: raw feed with market/point/price/home_team/away_team
    raw_cols = {"market", "point", "price", "home_team", "away_team"}
    if raw_cols.issubset(df_ou.columns):
        df_ou["market"] = df_ou["market"].astype(str).str.lower()
        if "label" in df_ou.columns:
            df_ou["label"] = df_ou["label"].astype(str)
        else:
            df_ou["label"] = ""

        out_rows = []
        for (h, a), grp in df_ou.groupby(["home_team", "away_team"]):
            matchup = f"{a} @ {h}"

            # Totals (over/under)
            tot = grp[grp["market"].isin(["totals", "total"])]
            line = over_p = under_p = None
            if not tot.empty:
                over = tot[tot["label"].str.lower().eq("over")]
                under = tot[tot["label"].str.lower().eq("under")]
                if not over.empty:
                    line = float(over["point"].iloc[0])
                    over_p = float(over["price"].iloc[0])
                if not under.empty:
                    if line is None:
                        line = float(under["point"].iloc[0])
                    under_p = float(under["price"].iloc[0])

            # Moneyline
            ml = grp[grp["market"] == "moneyline"]
            h_ml = a_ml = None
            if not ml.empty:
                labels = ml["label"].str.upper()
                home_ml_rows = ml[labels.isin([str(h).upper()])]
                away_ml_rows = ml[labels.isin([str(a).upper()])]
                if not home_ml_rows.empty:
                    h_ml = float(home_ml_rows["price"].iloc[0])
                if not away_ml_rows.empty:
                    a_ml = float(away_ml_rows["price"].iloc[0])
                if h_ml is None and len(ml) >= 1:
                    h_ml = float(ml["price"].iloc[0])
                if a_ml is None and len(ml) >= 2:
                    a_ml = float(ml["price"].iloc[1])

            # Spread: home team line & price
            sp = grp[grp["market"] == "spread"]
            spread_line = spread_price = None
            if not sp.empty:
                home_sp = sp[sp["label"].str.upper().eq(str(h).upper())]
                if not home_sp.empty:
                    spread_line = float(home_sp["point"].iloc[0])
                    spread_price = float(home_sp["price"].iloc[0])
                else:
                    spread_line = float(sp["point"].iloc[0])
                    spread_price = float(sp["price"].iloc[0])

            out_rows.append(
                {
                    "matchup": matchup,
                    "book_line": float(line or 0),
                    "over_price": float(over_p or 0),
                    "under_price": float(under_p or 0),
                    "home_ml": float(h_ml or 0),
                    "away_ml": float(a_ml or 0),
                    "spread_line": float(spread_line or 0),
                    "spread_price": float(spread_price or 0),
                }
            )

        return pd.DataFrame(out_rows)

    # Case 2: template schema
    template_cols = {
        "matchup",
        "bookmaker line",
        "over price",
        "under price",
        "home ml",
        "away ml",
    }
    if template_cols.issubset(df_ou.columns):
        rows = []
        for _, r in df_ou.iterrows():
            rows.append(
                {
                    "matchup": str(r["matchup"]).strip(),
                    "book_line": float(r["bookmaker line"]),
                    "over_price": float(r["over price"]),
                    "under_price": float(r["under price"]),
                    "home_ml": float(r["home ml"]),
                    "away_ml": float(r["away ml"]),
                    "spread_line": float(r.get("spread line", 0) or 0),
                    "spread_price": float(r.get("spread price", 0) or 0),
                }
            )
        return pd.DataFrame(rows)

    raise ValueError(
        "Unsupported odds CSV schema. Expected either raw book feed "
        "({market, point, price, home_team, away_team[,label]}) or template "
        "({Matchup, Bookmaker Line, Over Price, Under Price, Home ML, Away ML, ...})."
    )


def build_raw_rows_from_parsed_odds(parsed_odds: pd.DataFrame) -> List[Dict]:
    """Convert parsed per-matchup odds into raw_rows compatible with normalize_market_rows."""
    rows: List[Dict] = []
    if parsed_odds is None or parsed_odds.empty:
        return rows

    ts_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for _, r in parsed_odds.iterrows():
        matchup = str(r["matchup"]).strip()
        parts = matchup.split("@")
        if len(parts) == 2:
            away = parts[0].strip().title()
            home = parts[1].strip().title()
            matchup_display = f"{away} at {home}"
        else:
            away = None
            home = None
            matchup_display = matchup

        book_line = float(r.get("book_line", 0) or 0)
        over_price = float(r.get("over_price", 0) or 0)
        under_price = float(r.get("under_price", 0) or 0)
        home_ml = float(r.get("home_ml", 0) or 0)
        away_ml = float(r.get("away_ml", 0) or 0)
        spread_line = float(r.get("spread_line", 0) or 0)
        spread_price = float(r.get("spread_price", 0) or 0)

        # Totals
        if book_line != 0 and (over_price != 0 or under_price != 0):
            if over_price != 0:
                rows.append(
                    {
                        "timestamp_utc": ts_now,
                        "matchup": matchup_display,
                        "market_type": "Total",
                        "side_label": "Over",
                        "line_raw": str(book_line),
                        "odds_raw": str(int(over_price)),
                        "handle_pct": None,
                        "bets_pct": None,
                    }
                )
            if under_price != 0:
                rows.append(
                    {
                        "timestamp_utc": ts_now,
                        "matchup": matchup_display,
                        "market_type": "Total",
                        "side_label": "Under",
                        "line_raw": str(book_line),
                        "odds_raw": str(int(under_price)),
                        "handle_pct": None,
                        "bets_pct": None,
                    }
                )

        # Moneyline
        if home and home_ml != 0:
            rows.append(
                {
                    "timestamp_utc": ts_now,
                    "matchup": matchup_display,
                    "market_type": "Moneyline",
                    "side_label": home,
                    "line_raw": "",
                    "odds_raw": str(int(home_ml)),
                    "handle_pct": None,
                    "bets_pct": None,
                }
            )
        if away and away_ml != 0:
            rows.append(
                {
                    "timestamp_utc": ts_now,
                    "matchup": matchup_display,
                    "market_type": "Moneyline",
                    "side_label": away,
                    "line_raw": "",
                    "odds_raw": str(int(away_ml)),
                    "handle_pct": None,
                    "bets_pct": None,
                }
            )

        # Spread (home side only)
        if home and spread_line != 0 and spread_price != 0:
            rows.append(
                {
                    "timestamp_utc": ts_now,
                    "matchup": matchup_display,
                    "market_type": "Spread",
                    "side_label": home,
                    "line_raw": str(spread_line),
                    "odds_raw": str(int(spread_price)),
                    "handle_pct": None,
                    "bets_pct": None,
                }
            )

    return rows


#############################################
# -------- PUSH PROBABILITY HANDLING -------
#############################################

def load_push_overrides(uploaded_csv: Optional[bytes]):
    """
    Optional CSV override from sidebar.
    Two possible schemas you support:
    1) spread_key, push_prob
    2) total_key, bucket, push_prob
    We'll ingest into dicts that mirror EMP_SPREAD_PUSH_DEFAULT and EMP_TOTAL_PUSH_DEFAULT.
    Anything invalid gets ignored. We don't crash.
    """
    spread_override = {}
    total_override = {}

    if uploaded_csv is None:
        return spread_override, total_override

    try:
        df = pd.read_csv(uploaded_csv)
    except Exception:
        return spread_override, total_override

    # try spread-style override first
    # columns: spread_key, push_prob
    if {"spread_key","push_prob"}.issubset(df.columns):
        for _, row in df.iterrows():
            key = row["spread_key"]
            ppush = row["push_prob"]
            try:
                key_f = float(key)
                ppush_f = float(ppush)
                if 0 <= ppush_f <= 1:
                    spread_override[key_f] = ppush_f
            except:
                pass

    # try total-style override
    # columns: total_key, bucket, push_prob
    if {"total_key","bucket","push_prob"}.issubset(df.columns):
        for _, row in df.iterrows():
            key = row["total_key"]
            bucket = str(row["bucket"]).strip().lower()
            ppush = row["push_prob"]
            try:
                key_f = float(key)
                ppush_f = float(ppush)
                if 0 <= ppush_f <= 1 and bucket in ["low","mid","high"]:
                    if key_f not in total_override:
                        total_override[key_f] = {}
                    total_override[key_f][bucket] = ppush_f
            except:
                pass

    return spread_override, total_override


def get_spread_push_prob(spread_line_home: float,
                          spread_push_table: Dict[float,float]) -> float:
    """
    Return estimated push probability for the HOME spread number.
    """
    if spread_line_home is None:
        return 0.0
    # snap to nearest 0.5 or integer step we track
    # we take closest key
    if not spread_push_table:
        return 0.0
    diffs = [(abs(spread_line_home - k), k) for k in spread_push_table.keys()]
    diffs.sort(key=lambda x: x[0])
    nearest = diffs[0][1]
    return float(spread_push_table.get(nearest, 0.0))


def get_total_push_prob(total_line: float,
                        total_push_table: Dict[float,Dict[str,float]]) -> float:
    """
    Return estimated push probability for the total number
    based on environment bucket.
    """
    if total_line is None:
        return 0.0
    bucket = total_bucket(total_line)
    if not total_push_table:
        return 0.0
    diffs = [(abs(total_line - k), k) for k in total_push_table.keys()]
    diffs.sort(key=lambda x: x[0])
    nearest = diffs[0][1]
    bucket_map = total_push_table.get(nearest, {})
    return float(bucket_map.get(bucket, 0.0))


#############################################
# ----------- SCRAPE & PARSE ---------------
#############################################

def fetch_html(url: str) -> Optional[str]:
    """
    Request a page with basic headers.
    We don't actually embed Playwright here because that requires runtime env.
    Your enhanced script tried Playwright first and fell back to requests.
    We're standardizing on requests here + validation.
    If this becomes a JS-rendered page that needs browser automation,
    you re-inject Playwright later in this function.
    """
    try:
        resp = requests.get(url, headers=USER_AGENT_HEADER, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type","").lower():
            # not obviously HTML? bail.
            return None
        return resp.text
    except Exception:
        return None


def scrape_all_pages(event_group: str) -> List[Dict]:
    """
    Pull all NFL matchups and markets for the event_group.
    Your originals paginated via "tb_page=" links.
    We'll try page 1,2,3,... until we get no new games or hit sanity cap.
    We also attempt the 'range' variants from baseline (today/tomorrow/n7days)
    so you still get slates if day is empty.
    """
    data_rows = []
    seen_rows = set()

    # try multiple range contexts like baseline
    ranges = ["", "?range=today", "?range=tomorrow", "?range=n7days"]
    # defensive stop so we don't infinite loop
    MAX_PAGES = 10

    for rng in ranges:
        for page in range(1, MAX_PAGES + 1):
            if page == 1:
                url = (
                    f"https://dknetwork.draftkings.com/sportsbook/nfl/game-lines"
                    f"/{event_group}{rng}"
                )
            else:
                # replicate the '?tb_page=' style from baseline
                sep = "&" if "?" in rng else "?"
                url = (
                    f"https://dknetwork.draftkings.com/sportsbook/nfl/game-lines"
                    f"/{event_group}{rng}{sep}tb_page={page}"
                )

            html = fetch_html(url)
            if not html:
                break

            chunk = parse_game_lines_html(html)
            # dedupe by tuple signature
            added_this_page = 0
            for row in chunk:
                sig = (
                    row.get("matchup"),
                    row.get("market_type"),
                    row.get("side_label"),
                    row.get("line_raw"),
                    row.get("odds_raw"),
                    row.get("handle_pct"),
                    row.get("bets_pct"),
                )
                if sig not in seen_rows:
                    seen_rows.add(sig)
                    data_rows.append(row)
                    added_this_page += 1

            # if we didn't get anything new from this page, bail on pagination.
            if added_this_page == 0:
                break

    return data_rows


def parse_game_lines_html(html: str) -> List[Dict]:
    """
    Take the DK Network HTML for one page and pull out:
      matchup, market_type (Moneyline/Spread/Total),
      side_label (team or Over/Under),
      line_raw (spread number, total number, etc),
      odds_raw (e.g. -110),
      handle_pct (e.g. '57%'),
      bets_pct   (e.g. '43%'),
      timestamp
    This merges the approach from both scripts:
    - we parse row blocks that look like tables of markets
    - we aggressively try/except around every field so bad rows don't nuke the run
    """
    soup = BeautifulSoup(html, "html.parser")

    out_rows = []
    # Your originals looped over game containers, then each market row.
    # We replicate that general approach: find possible blocks, then read
    # Moneyline / Spread / Total rows within.

    # Heuristic selectors:
    game_blocks = soup.find_all(lambda tag: tag.name in ["div","section"] and "game-card" in " ".join(tag.get("class",[])).lower())
    # fallback if class changes
    if not game_blocks:
        game_blocks = soup.find_all(lambda tag: tag.name in ["div","section"] and "game" in " ".join(tag.get("class",[])).lower() and "card" in " ".join(tag.get("class",[])).lower())

    ts_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for g in game_blocks:
        # try to extract matchup like "Eagles at Cowboys"
        matchup_txt = None
        header = g.find(lambda t: t.name in ["h2","h3","div","span"] and "at" in t.get_text(strip=True).lower())
        if header:
            matchup_txt = header.get_text(" ", strip=True)
        else:
            # hail mary: grab first h2/h3
            hh = g.find(["h2","h3"])
            if hh:
                matchup_txt = hh.get_text(" ", strip=True)

        # find tables/rows for Market types inside this block
        market_tables = g.find_all(lambda t: t.name in ["table","div"] and ("moneyline" in t.get_text(" ", strip=True).lower() or "spread" in t.get_text(" ", strip=True).lower() or "total" in t.get_text(" ", strip=True).lower()))

        for mt in market_tables:
            # attempt row-level parsing
            # each row should correspond to a side of a market (Team A ML, Team B ML, Over, Under, etc)
            rows = mt.find_all("tr")
            if not rows:
                # sometimes it's div-based "rows"
                rows = mt.find_all(lambda t: t.name == "div" and "row" in " ".join(t.get("class",[])).lower())

            for r in rows:
                txtrow = r.get_text(" ", strip=True).lower()
                # identify market type
                mtype = None
                if "moneyline" in txtrow:
                    mtype = "Moneyline"
                elif "spread" in txtrow:
                    mtype = "Spread"
                elif "total" in txtrow or "over" in txtrow and "under" in txtrow:
                    mtype = "Total"

                side_label = None
                line_raw = None
                odds_raw = None
                handle_txt = None
                bets_txt = None

                # try to break td-like cells
                cells = r.find_all(["td","div","span"])
                cell_texts = [c.get_text(" ", strip=True) for c in cells]

                # crude heuristics per script behavior
                # we look for patterns like:
                # TeamName   -2.5   -110   62% handle   58% bets
                # Over 45.5  -105   71% handle   66% bets
                # Under 45.5 -115   29% handle   34% bets
                # etc.

                if len(cell_texts) >= 2:
                    side_label = cell_texts[0]

                # find first odds-looking token
                for tkn in cell_texts:
                    if re.search(r"^[+-]\d{2,4}$", tkn.strip()):
                        odds_raw = tkn.strip()
                        break

                # find first thing that looks like a numeric line (-2.5, 45.5, etc)
                for tkn in cell_texts:
                    if re.search(r"[-+]?\d+(\.\d+)?", tkn.strip()):
                        line_raw = tkn.strip()
                        break

                # pull %handle / %bets
                for tkn in cell_texts:
                    if "%" in tkn.lower() and "handle" in tkn.lower():
                        handle_txt = tkn
                    if "%" in tkn.lower() and "bet" in tkn.lower():
                        bets_txt = tkn

                # sanity guard: if we didn't get an odds token, skip
                # this protects us from parsing header rows
                if odds_raw is None:
                    continue

                out_rows.append(
                    {
                        "timestamp_utc": ts_now,
                        "matchup": matchup_txt,
                        "market_type": mtype,
                        "side_label": side_label,
                        "line_raw": line_raw,
                        "odds_raw": odds_raw,
                        "handle_pct": handle_txt,
                        "bets_pct": bets_txt,
                    }
                )

    return out_rows


#############################################
# ---------- NORMALIZATION LOGIC -----------
#############################################

def split_matchup(matchup: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to split 'Eagles at Cowboys' -> ('Eagles','Cowboys').
    Fallback to ('Team1','Team2') if 'at' not found.
    """
    if not matchup:
        return (None, None)
    low = matchup.strip()
    # try 'A at B'
    parts = re.split(r"\s+at\s+", low, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # fallback 'A vs B'
    parts = re.split(r"\s+vs\.?\s+", low, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # fallback super dumb split on space dash space
    parts = re.split(r"\s+-\s+", low)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # last fallback, just split first word vs rest
    toks = low.split()
    if len(toks) > 1:
        return toks[0].strip(), " ".join(toks[1:]).strip()
    return (low, None)


def label_is_home(label: str, home_team: str, away_team: str) -> Optional[bool]:
    """
    Determine if this row describes HOME side or AWAY side.
    We match substrings because DK sometimes uses just city or nickname.
    Returns True (home), False (away), or None (couldn't tell).
    """
    if label is None:
        return None
    l = label.lower()
    home = (home_team or "").lower()
    away = (away_team or "").lower()

    # strongest: full contains
    if home and home in l:
        return True
    if away and away in l:
        return False

    # fallback: first token compare
    if home_team:
        htok = home_team.split()[0].lower()
        if htok in l:
            return True
    if away_team:
        atok = away_team.split()[0].lower()
        if atok in l:
            return False

    # can't tell
    return None


def normalize_market_rows(raw_rows: List[Dict]) -> pd.DataFrame:
    """
    Take the scraped dict rows and turn them into a structured DataFrame with:
    matchup, home_team, away_team, market_type,
    side_role ('HOME','AWAY','OVER','UNDER'),
    line (float), odds (int),
    handle (0-1), bets (0-1),
    timestamp_utc
    Rows that can't be interpreted cleanly are dropped.
    """
    recs = []
    for r in raw_rows:
        matchup = r.get("matchup")
        mtype = r.get("market_type")
        side_label = r.get("side_label")

        away_team, home_team = split_matchup(matchup)

        # classify side
        side_role = None
        if mtype == "Total":
            if side_label and side_label.lower().startswith("over"):
                side_role = "OVER"
            elif side_label and side_label.lower().startswith("under"):
                side_role = "UNDER"
        elif mtype in ["Spread", "Moneyline"]:
            is_home_flag = label_is_home(side_label, home_team, away_team)
            if is_home_flag is True:
                side_role = "HOME"
            elif is_home_flag is False:
                side_role = "AWAY"

        odds_val = odds_from_text(r.get("odds_raw"))
        line_val = line_from_text(r.get("line_raw"))

        handle_val = parse_percentage(r.get("handle_pct"))
        bets_val = parse_percentage(r.get("bets_pct"))

        # sanity gates
        if odds_val is None:
            continue
        if odds_val < MIN_ODDS_OK or odds_val > MAX_ODDS_OK:
            continue

        if mtype == "Spread":
            # reject insane spreads
            if (line_val is None) or (abs(line_val) > MAX_ABS_SPREAD_OK):
                continue
        if mtype == "Total":
            if (line_val is None) or (line_val < MIN_TOTAL_OK) or (line_val > MAX_TOTAL_OK):
                continue

        recs.append(
            {
                "timestamp_utc": r.get("timestamp_utc"),
                "matchup": matchup,
                "home_team": home_team,
                "away_team": away_team,
                "market_type": mtype,
                "side_role": side_role,
                "line": line_val,
                "odds": odds_val,
                "handle": handle_val,
                "bets": bets_val,
            }
        )

    df = pd.DataFrame.from_records(recs)

    # ensure expected columns exist even if scrape returned nothing usable
    expected_cols = [
        "timestamp_utc",
        "matchup",
        "home_team",
        "away_team",
        "market_type",
        "side_role",
        "line",
        "odds",
        "handle",
        "bets",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype="float64" if col in ["line","odds","handle","bets"] else "object")

    # if we still have no rows at all, just return the empty frame with expected columns
    if df.empty:
        return df[expected_cols]

    # drop rows missing basic identifiers, but only if those columns actually exist
    if {"market_type", "matchup"}.issubset(df.columns):
        df = df.dropna(subset=["market_type", "matchup"])

    # dedupe
    df = df.drop_duplicates()

    return df


#############################################
# -------- PROBABILITY / PRICING -----------
#############################################

def apply_splits_tilt(p_novig_a: float,
                      p_novig_b: float,
                      handle_a: Optional[float],
                      bets_a: Optional[float],
                      handle_b: Optional[float],
                      bets_b: Optional[float],
                      splits_weight: float) -> Tuple[Optional[float], Optional[float], str]:
    """
    Adjust the no-vig pair (p_a, p_b) using handle vs bets skew as "sharp pressure".
    We work in logit space like the enhanced script, but keep logic from baseline.

    Return (p_adj_a, p_adj_b, edge_tag)
    edge_tag helps explain where the edge came from.

    splits_weight is user slider, e.g. 0.0 to 2.0 type range.
    """
    if p_novig_a is None or p_novig_b is None:
        return (p_novig_a, p_novig_b, "vig_only")

    # If we have at least one side with handle/bets data, we form a tilt signal.
    tilt_a = None
    if handle_a is not None and bets_a is not None:
        tilt_a = handle_a - bets_a  # positive => bigger money per ticket
    tilt_b = None
    if handle_b is not None and bets_b is not None:
        tilt_b = handle_b - bets_b

    # If neither tilt is usable, just return baseline
    if tilt_a is None and tilt_b is None:
        return (p_novig_a, p_novig_b, "vig_only")

    # define base logits
    z_a = logit(p_novig_a)
    z_b = logit(p_novig_b)

    # We want total to still sum to ~1 after adjustment, so we'll tweak both sides
    # and then renormalize.
    # If side A is getting heavy handle vs bets, we bump z_a upward by splits_weight*tilt_a.
    # For side B same idea.
    if tilt_a is None:
        tilt_a = 0.0
    if tilt_b is None:
        tilt_b = 0.0

    z_a_adj = z_a + splits_weight * tilt_a
    z_b_adj = z_b + splits_weight * tilt_b

    p_a_adj_raw = logistic(z_a_adj)
    p_b_adj_raw = logistic(z_b_adj)

    s = p_a_adj_raw + p_b_adj_raw
    if s <= 1e-12:
        return (p_novig_a, p_novig_b, "vig_only")

    p_a_final = p_a_adj_raw / s
    p_b_final = p_b_adj_raw / s

    # figure out our edge source label:
    edge_tag = "vig_only"
    if splits_weight != 0.0 and (abs(tilt_a) > 0.01 or abs(tilt_b) > 0.01):
        edge_tag = "handle_pressure"

    return (p_a_final, p_b_final, edge_tag)


def fair_price_spread_total(p_win: float,
                            p_push: float,
                            offered_odds: float) -> Dict[str, Optional[float]]:
    """
    Given adjusted p_win (vs that side), push prob p_push, and book odds,
    compute:
      - fair_decimal (excluding push vig)
      - fair_american
      - ev
      - kelly_fraction_raw
    """
    if p_win is None:
        return {
            "p_win": None,
            "p_push": None,
            "p_loss": None,
            "fair_decimal": None,
            "fair_american": None,
            "edge_pct": None,
            "ev": None,
            "kelly_raw": None,
        }

    p_loss = max(0.0, 1.0 - p_win - p_push)

    # fair decimal odds for this side if you were booking it:
    # we want implied prob of "win" relative to (win or lose), ignoring push
    denom = p_win + p_loss
    if denom <= 1e-12:
        fair_decimal = None
        fair_american = None
    else:
        fair_p = p_win / denom
        fair_decimal = 1.0 / fair_p if fair_p > 1e-12 else None
        fair_american = prob_to_american(fair_p)

    # EV of CURRENT offered odds:
    dec_offered = american_to_decimal(offered_odds)
    if dec_offered is None:
        ev_val = None
    else:
        profit_if_win = dec_offered - 1.0  # net
        ev_val = p_win * profit_if_win - p_loss * 1.0  # push yields 0

    # "edge_pct": how favorable is offered versus fair in terms of implied prob delta
    edge_pct = None
    if fair_decimal and dec_offered:
        # compare offered implied p_win vs fair_p
        offered_p_win = 1.0 / dec_offered
        denom2 = offered_p_win + (1.0 - offered_p_win)  # just 1 but keep form
        if denom2 > 0 and fair_p is not None:
            # difference in win prob vs fair
            edge_pct = (fair_p - offered_p_win) * 100.0

    kelly_raw = kelly_fraction(p_win, p_push, offered_odds)

    return {
        "p_win": p_win,
        "p_push": p_push,
        "p_loss": p_loss,
        "fair_decimal": fair_decimal,
        "fair_american": fair_american,
        "edge_pct": edge_pct,
        "ev": ev_val,
        "kelly_raw": kelly_raw,
    }


def fair_price_moneyline(p_win: float,
                         offered_odds: float) -> Dict[str, Optional[float]]:
    """
    Moneyline: no push.
    """
    if p_win is None:
        return {
            "p_win": None,
            "fair_decimal": None,
            "fair_american": None,
            "edge_pct": None,
            "ev": None,
            "kelly_raw": None,
        }

    p_loss = 1.0 - p_win
    dec_offered = american_to_decimal(offered_odds)

    # fair decimal
    if p_win <= 1e-12:
        fair_decimal = None
        fair_american = None
    else:
        fair_decimal = 1.0 / p_win
        fair_american = prob_to_american(p_win)

    # EV
    if dec_offered is None:
        ev_val = None
    else:
        profit_if_win = dec_offered - 1.0
        ev_val = p_win * profit_if_win - p_loss * 1.0

    # edge %
    edge_pct = None
    if dec_offered:
        offered_p_win = 1.0 / dec_offered
        edge_pct = (p_win - offered_p_win) * 100.0

    kelly_raw = kelly_fraction_moneyline(p_win, offered_odds)

    return {
        "p_win": p_win,
        "fair_decimal": fair_decimal,
        "fair_american": fair_american,
        "edge_pct": edge_pct,
        "ev": ev_val,
        "kelly_raw": kelly_raw,
    }


def build_market_evals(df: pd.DataFrame,
                       splits_weight: float,
                       spread_push_table: Dict[float,float],
                       total_push_table: Dict[float,Dict[str,float]]) -> pd.DataFrame:
    """
    This is where we stitch both sides of each market back together,
    compute adjusted win probs, push probs, fair prices, EV, Kelly, etc.

    Output columns (per side of each bettable thing):
      matchup, market_type, side_role, line, odds,
      p_win, p_push, p_loss,
      fair_american, edge_pct, ev,
      strength_score (capped Kelly),
      edge_source
    """
    out = []

    # group by matchup + market_type + line "key" to pair sides
    # Moneyline: line is irrelevant, we just group by matchup + market_type
    # Spread: must align by spread number from home perspective
    # Total: must align by total number
    # We'll create "pair_key" to unify that.

    # First create a working copy with computed pair_key
    work = df.copy()

    # For spreads, we need to normalize so line == home_team spread.
    # If side_role == HOME: line_home = line
    # If side_role == AWAY: line_home = -line
    # For totals, pair_key is just that total number.
    # For moneyline, pair_key can just be 'ML'
    def compute_pair_key(row):
        mtype = row["market_type"]
        if mtype == "Spread":
            if row["side_role"] == "HOME":
                return f"SPREAD@{row['matchup']}@{row['line']:.1f}"
            elif row["side_role"] == "AWAY":
                if row["line"] is None:
                    return None
                return f"SPREAD@{row['matchup']}@{(-row['line']):.1f}"
            else:
                return None
        elif mtype == "Total":
            if row["line"] is None:
                return None
            return f"TOTAL@{row['matchup']}@{row['line']:.1f}"
        elif mtype == "Moneyline":
            return f"ML@{row['matchup']}"
        else:
            return None

    work["pair_key"] = work.apply(compute_pair_key, axis=1)

    # Also compute normalized home_line / total_line for push prob later
    def compute_home_line(row):
        if row["market_type"] != "Spread":
            return None
        if row["side_role"] == "HOME":
            return row["line"]
        elif row["side_role"] == "AWAY":
            if row["line"] is None:
                return None
            return -row["line"]
        return None

    work["home_spread_line"] = work.apply(compute_home_line, axis=1)

    # totals line is same for Over/Under
    work["total_line_val"] = work.apply(
        lambda r: r["line"] if r["market_type"] == "Total" else None,
        axis=1
    )

    # group into pairs
    for pair_key, sub in work.groupby("pair_key"):
        if pair_key is None:
            continue
        # sanity check we actually have 2 sides usually
        # For ML: expect HOME + AWAY
        # For Spread: expect HOME + AWAY
        # For Total: expect OVER + UNDER
        if len(sub) < 2:
            # incomplete data -> skip evaluating. no hallucinated edge on partial
            continue

        # pick consistent matchup / home_team / away_team
        matchup_val = sub["matchup"].iloc[0]
        home_team = sub["home_team"].iloc[0]
        away_team = sub["away_team"].iloc[0]
        mtype = sub["market_type"].iloc[0]

        # extract sides
        sides = []
        for _, row in sub.iterrows():
            sides.append({
                "side_role": row["side_role"],
                "line": row["line"],
                "odds": row["odds"],
                "handle": row["handle"],
                "bets": row["bets"],
                "home_spread_line": row["home_spread_line"],
                "total_line_val": row["total_line_val"],
            })

        # match "HOME" vs "AWAY", "OVER" vs "UNDER"
        # This assumes 2 sides only.
        if len(sides) != 2:
            continue

        a = sides[0]
        b = sides[1]

        # implied prob from odds
        pA = american_to_prob(a["odds"])
        pB = american_to_prob(b["odds"])
        # de-vig
        pA_novig, pB_novig = remove_vig_two_way(pA, pB)

        # apply handle/bets tilt
        pA_adj, pB_adj, edge_tag = apply_splits_tilt(
            pA_novig,
            pB_novig,
            a["handle"], a["bets"],
            b["handle"], b["bets"],
            splits_weight=splits_weight
        )

        # Now produce market-specific fair pricing
        if mtype == "Moneyline":
            # no push
            resA = fair_price_moneyline(pA_adj, a["odds"])
            resB = fair_price_moneyline(pB_adj, b["odds"])

            # annotate & append
            for rec, sdata in [(resA,a),(resB,b)]:
                out.append({
                    "matchup": matchup_val,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_type": mtype,
                    "side_role": sdata["side_role"],
                    "line": None,
                    "offered_odds": sdata["odds"],
                    "p_win": rec["p_win"],
                    "p_push": 0.0,
                    "p_loss": (1.0 - (rec["p_win"] or 0.0)),
                    "fair_american": rec["fair_american"],
                    "edge_pct": rec["edge_pct"],
                    "ev": rec["ev"],
                    "kelly_raw": rec["kelly_raw"],
                    "edge_source": edge_tag if edge_tag != "vig_only" else "vig_or_flow",
                })

        elif mtype == "Spread":
            # compute push prob from normalized home spread line
            # we treat p_win for "HOME" as probability that HOME covers
            # and for "AWAY" as probability that AWAY covers
            # push prob is shared for that pair_key (same number)
            home_line = a["home_spread_line"] if a["home_spread_line"] is not None else b["home_spread_line"]
            p_push_est = get_spread_push_prob(home_line, spread_push_table)

            resA = fair_price_spread_total(pA_adj, p_push_est, a["odds"])
            resB = fair_price_spread_total(pB_adj, p_push_est, b["odds"])

            for rec, sdata in [(resA,a),(resB,b)]:
                out.append({
                    "matchup": matchup_val,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_type": mtype,
                    "side_role": sdata["side_role"],
                    "line": sdata["line"],
                    "offered_odds": sdata["odds"],
                    "p_win": rec["p_win"],
                    "p_push": rec["p_push"],
                    "p_loss": rec["p_loss"],
                    "fair_american": rec["fair_american"],
                    "edge_pct": rec["edge_pct"],
                    "ev": rec["ev"],
                    "kelly_raw": rec["kelly_raw"],
                    "edge_source": "push_key_number" if p_push_est > 0.0 and edge_tag == "vig_only" else edge_tag,
                })

        elif mtype == "Total":
            total_line = a["total_line_val"] if a["total_line_val"] is not None else b["total_line_val"]
            p_push_est = get_total_push_prob(total_line, total_push_table)

            resA = fair_price_spread_total(pA_adj, p_push_est, a["odds"])
            resB = fair_price_spread_total(pB_adj, p_push_est, b["odds"])

            for rec, sdata in [(resA,a),(resB,b)]:
                out.append({
                    "matchup": matchup_val,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_type": mtype,
                    "side_role": sdata["side_role"],  # OVER or UNDER
                    "line": sdata["line"],            # total number
                    "offered_odds": sdata["odds"],
                    "p_win": rec["p_win"],
                    "p_push": rec["p_push"],
                    "p_loss": rec["p_loss"],
                    "fair_american": rec["fair_american"],
                    "edge_pct": rec["edge_pct"],
                    "ev": rec["ev"],
                    "kelly_raw": rec["kelly_raw"],
                    "edge_source": "push_key_number" if p_push_est > 0.0 and edge_tag == "vig_only" else edge_tag,
                })

        else:
            # ignore other markets
            pass

    final_df = pd.DataFrame(out)

    return final_df


#############################################
# ----------------- UI ---------------------
#############################################

def main():
    st.set_page_config(page_title=APP_VERSION, layout="wide")

    # Sidebar controls
    st.sidebar.title("Controls")

    st.sidebar.markdown("**Data scrape target (event group)**")
    event_group = st.sidebar.text_input(
        "DraftKings event group id",
        value=DEFAULT_EVENT_GROUP,
        help="This identifies the slate/board on DK Network. 88808 was used in both originals for NFL."
    )

    st.sidebar.markdown("**Odds Source**")
    use_dk = st.sidebar.checkbox("Use DK Network scrape", value=True)
    odds_upload = st.sidebar.file_uploader(
        "Or upload odds CSV",
        type=["csv"],
        help=(
            "Same schemas as NFL predictor: raw book feed with "
            "market, point, price, home_team, away_team[,label] or template "
            "with: Matchup, Bookmaker Line, Over Price, Under Price, Home ML, Away ML[, Spread Line, Spread Price]."
        ),
    )

    st.sidebar.markdown("**Handle vs Bets Weighting**")
    splits_weight = st.sidebar.slider(
        "Splits Weight (handle minus bets -> 'sharp pressure')",
        min_value=0.0,
        max_value=2.0,
        value=0.75,
        step=0.05,
        help="0.0 = trust no handle/bet skew. Higher = weight heavy-money side as more 'true'."
    )

    st.sidebar.markdown("**Kelly Cap (Strength)***")
    kelly_cap = st.sidebar.slider(
        "Cap for Kelly-derived strength score",
        min_value=0.0,
        max_value=0.5,
        value=0.20,
        step=0.01,
        help="Caps the raw Kelly fraction. We publish this as a strength score, not bet sizing gospel."
    )

    st.sidebar.markdown("**Edge Thresholds (for surfacing 'BET')**")
    min_edge_pct_spread_total = st.sidebar.slider(
        "Min edge % (Spread/Total)",
        min_value=-5.0,
        max_value=15.0,
        value=2.0,
        step=0.5,
        help="Required edge_pct (%) to label SPREAD/TOTAL as BET instead of PASS."
    )
    min_edge_pct_ml = st.sidebar.slider(
        "Min edge % (Moneyline)",
        min_value=-5.0,
        max_value=15.0,
        value=2.0,
        step=0.5,
        help="Required edge_pct (%) to label ML as BET instead of PASS."
    )

    st.sidebar.markdown("**Optional Push Probability Override CSV**")
    st.sidebar.write("You can upload key-number push rates. Columns supported:")
    st.sidebar.write("spread_key,push_prob  OR  total_key,bucket,push_prob")
    uploaded_push_csv = st.sidebar.file_uploader(
        "Push Probability CSV (optional)",
        type=["csv"]
    )

    # scrape data OR consume uploaded odds
    raw_rows: List[Dict] = []
    if use_dk or odds_upload is None:
        raw_rows = scrape_all_pages(event_group)
        df_norm = normalize_market_rows(raw_rows)
    else:
        try:
            raw_csv = pd.read_csv(odds_upload)
            parsed_odds = parse_uploaded_odds(raw_csv)
            raw_rows = build_raw_rows_from_parsed_odds(parsed_odds)
            df_norm = normalize_market_rows(raw_rows)
        except Exception as e:
            st.error(f"Failed to parse uploaded odds CSV: {e}")
            df_norm = normalize_market_rows([])

    # overrides for push probs
    spread_override, total_override = load_push_overrides(uploaded_push_csv)

    spread_push_table = EMP_SPREAD_PUSH_DEFAULT.copy()
    spread_push_table.update(spread_override)

    total_push_table = {}
    # merge nested dicts
    for k,v in EMP_TOTAL_PUSH_DEFAULT.items():
        total_push_table[k] = v.copy()
    for k,v in total_override.items():
        if k not in total_push_table:
            total_push_table[k] = {}
        total_push_table[k].update(v)

    # evaluate
    priced_df = build_market_evals(
        df_norm,
        splits_weight=splits_weight,
        spread_push_table=spread_push_table,
        total_push_table=total_push_table
    )
    # ensure kelly_raw column exists even if no markets were priced (empty scrape / partial pairs)
    if "kelly_raw" not in priced_df.columns:
        priced_df["kelly_raw"] = pd.Series(dtype="float64")

    # Add strength_score (capped Kelly)
    priced_df["strength_score"] = priced_df["kelly_raw"].clip(lower=0.0).clip(upper=kelly_cap)

    # Decide BET / PASS
    def bet_flag(row):
        # defensively handle duplicate columns, missing cols, etc.
        edge_val = row.get("edge_pct")
        stg_val = row.get("strength_score")
        mkt_val = row.get("market_type")

        if pd.isna(edge_val):
            return "PASS"

        thr = min_edge_pct_ml if mkt_val == "Moneyline" else min_edge_pct_spread_total
        if edge_val >= thr and (stg_val is not None and stg_val > 0):
            return "BET"
        return "PASS"

    rec_series = priced_df.apply(bet_flag, axis=1)
    # If pandas returns a DataFrame because of duplicate column names upstream,
    # collapse to the first column so we always assign a 1D Series.
    if isinstance(rec_series, pd.DataFrame):
        rec_series = rec_series.iloc[:, 0]

    priced_df["rec"] = rec_series

    # ensure all display columns exist even if no markets priced / partial scrape
    expected_display_cols = [
        "rec",
        "matchup",
        "market_type",
        "side_role",
        "line",
        "offered_odds",
        "p_win",
        "p_push",
        "p_loss",
        "fair_american",
        "edge_pct",
        "ev",
        "strength_score",
        "edge_source",
    ]
    numeric_like = {
        "line",
        "offered_odds",
        "p_win",
        "p_push",
        "p_loss",
        "fair_american",
        "edge_pct",
        "ev",
        "strength_score",
    }
    for col in expected_display_cols:
        if col not in priced_df.columns:
            # numeric-style cols become empty float series, others become empty object series
            if col in numeric_like:
                priced_df[col] = pd.Series(dtype="float64")
            else:
                priced_df[col] = pd.Series(dtype="object")

    # also ensure matchup / market_type exist so the UI filters don't KeyError
    if "matchup" not in priced_df.columns:
        priced_df["matchup"] = pd.Series(dtype="object")
    if "market_type" not in priced_df.columns:
        priced_df["market_type"] = pd.Series(dtype="object")

    # data health / debug box
    st.header(APP_VERSION)
    colA, colB, colC, colD = st.columns(4)

    total_rows_scraped = len(raw_rows)
    total_rows_after_norm = len(df_norm)
    total_pairs_priced = len(priced_df)
    utc_timestamps = sorted(set([r.get("timestamp_utc") for r in raw_rows if r.get("timestamp_utc")]))
    scrape_ts_display = utc_timestamps[-1] if utc_timestamps else "n/a"

    with colA:
        st.metric("Scrape timestamp (UTC)", scrape_ts_display)
    with colB:
        st.metric("Rows scraped", total_rows_scraped)
    with colC:
        st.metric("Valid rows after parsing", total_rows_after_norm)
    with colD:
        st.metric("Priced betting sides", total_pairs_priced)

    # summary KPIs for bets only
    bets_only = priced_df[priced_df["rec"] == "BET"].copy()
    avg_edge = bets_only["edge_pct"].mean() if len(bets_only) else None
    avg_strength = bets_only["strength_score"].mean() if len(bets_only) else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actionable BET count", len(bets_only))
    with col2:
        st.metric("Avg Edge % (BETs)", f"{avg_edge:.2f}%" if avg_edge is not None else "n/a")
    with col3:
        st.metric("Avg Strength Score (BETs)", f"{avg_strength:.3f}" if avg_strength is not None else "n/a")

    # Filters for display
    st.subheader("Filters")
    all_matchups = ["(all)"] + sorted(priced_df["matchup"].dropna().unique().tolist())
    picked_matchup = st.selectbox("Matchup filter", all_matchups, index=0)

    all_markets = ["(all)"] + sorted(priced_df["market_type"].dropna().unique().tolist())
    picked_market = st.selectbox("Market filter", all_markets, index=0)

    view_df = priced_df.copy()
    if picked_matchup != "(all)":
        view_df = view_df[view_df["matchup"] == picked_matchup]
    if picked_market != "(all)":
        view_df = view_df[view_df["market_type"] == picked_market]

    # clean display columns
    disp_cols = [
        "rec",
        "matchup",
        "market_type",
        "side_role",
        "line",
        "offered_odds",
        "p_win",
        "p_push",
        "p_loss",
        "fair_american",
        "edge_pct",
        "ev",
        "strength_score",
        "edge_source",
    ]

    st.subheader("Priced Markets")
    st.dataframe(
        view_df[disp_cols].sort_values(
            by=["rec","edge_pct","strength_score"],
            ascending=[True, False, False]
        ).reset_index(drop=True)
    )

    # matchup cards like baseline did (group by matchup then show per-market summary)
    st.subheader("Matchup Cards")
    for mu, submu in priced_df.groupby("matchup"):
        st.markdown(f"### {mu}")
        subshow = submu[disp_cols].sort_values(
            by=["rec","edge_pct","strength_score"],
            ascending=[True, False, False]
        )
        st.table(subshow.reset_index(drop=True))

    # CSV export
    st.subheader("Export")
    csv_bytes = priced_df[disp_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download priced markets CSV",
        data=csv_bytes,
        file_name=f"nfl_fair_pricer_{int(time.time())}.csv",
        mime="text/csv"
    )

    st.caption(
        "strength_score = capped Kelly fraction. "
        "This is NOT bankroll sizing advice. "
        "edge_source explains where the edge mostly comes from "
        "(handle flow vs pure vig math vs key-number push protection)."
    )


if __name__ == "__main__":
    main()
