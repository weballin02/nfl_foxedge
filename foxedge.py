#!/usr/bin/env python3
"""
FOXEDGE NFL DKNetwork Edgeboard – Matchup-Aware Version

- Uses nfl_data_py play-by-play for last N seasons
- Builds team offense/defense metrics (EPA, success, explosive, pass rate, plays)
- Adjusts for opponent strength (SoS-adjusted EPA)
- Builds game-level matchup features
- Fits linear models for:
    - home margin
    - game total
- Fetches DKNetwork betting splits (NFL event group 88808)
- Maps splits to home/away, spread, total, moneyline
- Computes model vs market edges, EV, and Kelly stakes
- Renders spread & total edges in Streamlit

No CSV uploads. One file.
"""

import datetime as dt
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# DATA SOURCE: nfl_data_py
# -------------------------------------------------------------------

try:
    import nfl_data_py as nfl
except ImportError:
    nfl = None

# -------------------------------------------------------------------
# DKNETWORK SPLITS FETCHER (same skeleton as before, fixed matchup parsing)
# -------------------------------------------------------------------

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None


def fetch_dk_splits(
    event_group: int = 88808, date_range: str = "today"
) -> List[Dict[str, Any]]:
    """
    Fetch DKNetwork betting splits/handles for a given event group and date range.

    Returns a list of dicts with keys:
      matchup, game_time, market, side, odds, %handle, %bets, update_time
    """
    from urllib.parse import urlencode

    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
    full_url = f"{base}?{urlencode(params)}"

    def _clean(text: str) -> str:
        return re.sub(
            r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", text or "", flags=re.I
        ).strip()

    def _clean_odds(odds_str: str) -> int:
        try:
            return int(str(odds_str).replace("−", "-"))
        except Exception:
            try:
                return int(re.sub(r"[^-+\d]", "", str(odds_str)))
            except Exception:
                return 0

    html = ""

    # Try Playwright (if available)
    if sync_playwright is not None:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=UA)
                page.goto(full_url, wait_until="domcontentloaded", timeout=30000)
                try:
                    page.wait_for_selector("div.tb-se, .tb-war", timeout=10000)
                except Exception:
                    try:
                        page.wait_for_load_state("networkidle", timeout=8000)
                    except Exception:
                        pass
                html = page.content()
                browser.close()
        except Exception:
            html = ""

    # Fallback: plain GET
    if not html:
        try:
            resp = requests.get(full_url, headers={"User-Agent": UA}, timeout=20)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            return []

    soup = BeautifulSoup(html, "lxml")
    games = soup.select("div.tb-se, .tb-war")

    records: List[Dict[str, Any]] = []
    now_epoch = int(time.time())

    for game in games:
        title_node = game.select_one("div.tb-se-title h5, .tb-hwar h3, h3")
        title = _clean(title_node.get_text(strip=True)) if title_node else ""
        time_node = game.select_one("div.tb-se-title span, .tb-gmTime, time")
        game_time = _clean(time_node.get_text(strip=True)) if time_node else ""

        sections = game.select(".tb-market-wrap > div, .tb-war-mkt, .tb-mkt-wrap")
        for section in sections:
            mkt_node = section.select_one(
                ".tb-se-head > div, .tb-mkt-headline, .tb-mkth, h4"
            )
            market_name = _clean(mkt_node.get_text(strip=True)) if mkt_node else ""
            if market_name not in ("Moneyline", "Total", "Totals", "Spread"):
                continue

            for row in section.select(".tb-sodd, .tb-sodd-out, .tb-odd-row, tr"):
                side_node = row.select_one(
                    ".tb-slipline, .tb-line, .tb-side, td:first-child"
                )
                side_raw = _clean(
                    side_node.get_text(strip=True)
                ) if side_node else ""

                odds_node = row.select_one(
                    "a.tb-odd-s, .tb-odd-s, a[aria-label*='odds'], .tb-odds, td a"
                )
                raw_odds = _clean(
                    odds_node.get_text(strip=True)
                ) if odds_node else ""
                odds = _clean_odds(raw_odds)

                pct_texts = []
                for el in row.select(
                    ".tb-pct, .tb-sh, .tb-sb, .tb-pct-h, .tb-pct-b, span, td, div"
                ):
                    txt = el.get_text(" ", strip=True)
                    if "%" in txt:
                        pct_texts.append(txt)
                if not pct_texts:
                    flat_nodes = row.find_all(
                        string=lambda t: isinstance(t, str) and "%" in t
                    )
                    pct_texts = [t.strip() for t in flat_nodes]

                nums = []
                for t in pct_texts:
                    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
                    if m:
                        nums.append(m.group(1))
                    if len(nums) >= 2:
                        break
                if len(nums) < 2:
                    flat = " ".join(pct_texts)
                    nums = re.findall(r"(\d+(?:\.\d+)?)\s*%", flat)[:2]

                handle_pct, bets_pct = (nums + ["", ""])[:2]

                if not side_raw and not handle_pct and not bets_pct:
                    continue

                records.append(
                    {
                        "matchup": title,
                        "game_time": game_time,
                        "market": market_name,
                        "side": side_raw,
                        "odds": odds,
                        "%handle": float(handle_pct or 0),
                        "%bets": float(bets_pct or 0),
                        "update_time": now_epoch,
                    }
                )

    return records


def _is_moneyline(name: str) -> bool:
    n = (name or "").lower()
    return "moneyline" in n or n.endswith(" ml") or n == "ml"


def _is_total(name: str) -> bool:
    n = (name or "").lower()
    return "total" in n or "o/u" in n or "over/under" in n


# ---- TEAM NORMALIZATION ----

TEAM_ALIASES = {
    # NFC East
    "dal": "DAL",
    "cowboys": "DAL",
    "dallas": "DAL",
    "phi": "PHI",
    "eagles": "PHI",
    "philadelphia": "PHI",
    "nyg": "NYG",
    "giants": "NYG",
    "new york giants": "NYG",
    "was": "WAS",
    "wsh": "WAS",
    "commanders": "WAS",
    "washington": "WAS",
    # NFC North
    "gb": "GB",
    "packers": "GB",
    "green bay": "GB",
    "min": "MIN",
    "vikings": "MIN",
    "minnesota": "MIN",
    "det": "DET",
    "lions": "DET",
    "detroit": "DET",
    "chi": "CHI",
    "bears": "CHI",
    "chicago": "CHI",
    # NFC South
    "tb": "TB",
    "buccaneers": "TB",
    "bucs": "TB",
    "tampa bay": "TB",
    "atl": "ATL",
    "falcons": "ATL",
    "atlanta": "ATL",
    "car": "CAR",
    "panthers": "CAR",
    "carolina": "CAR",
    "no": "NO",
    "nor": "NO",
    "saints": "NO",
    "new orleans": "NO",
    # NFC West
    "sf": "SF",
    "49ers": "SF",
    "niners": "SF",
    "san francisco": "SF",
    "sea": "SEA",
    "seahawks": "SEA",
    "seattle": "SEA",
    "ari": "ARI",
    "cardinals": "ARI",
    "arizona": "ARI",
    "lar": "LAR",
    "la rams": "LAR",
    "rams": "LAR",
    # AFC East
    "ne": "NE",
    "patriots": "NE",
    "new england": "NE",
    "nyj": "NYJ",
    "jets": "NYJ",
    "new york jets": "NYJ",
    "mia": "MIA",
    "dolphins": "MIA",
    "miami": "MIA",
    "buf": "BUF",
    "bills": "BUF",
    "buffalo": "BUF",
    # AFC North
    "pit": "PIT",
    "steelers": "PIT",
    "pittsburgh": "PIT",
    "bal": "BAL",
    "ravens": "BAL",
    "baltimore": "BAL",
    "cle": "CLE",
    "browns": "CLE",
    "cleveland": "CLE",
    "cin": "CIN",
    "bengals": "CIN",
    "cincinnati": "CIN",
    # AFC South
    "jax": "JAX",
    "jac": "JAX",
    "jaguars": "JAX",
    "jacksonville": "JAX",
    "ind": "IND",
    "colts": "IND",
    "indianapolis": "IND",
    "hou": "HOU",
    "texans": "HOU",
    "houston": "HOU",
    "ten": "TEN",
    "titans": "TEN",
    "tennessee": "TEN",
    # AFC West
    "kc": "KC",
    "chiefs": "KC",
    "kansas city": "KC",
    "lv": "LV",
    "rai": "LV",
    "raiders": "LV",
    "las vegas": "LV",
    "lac": "LAC",
    "chargers": "LAC",
    "los angeles chargers": "LAC",
    "den": "DEN",
    "broncos": "DEN",
    "denver": "DEN",
}

ABBR_TO_NAME = {
    "DAL": "Cowboys",
    "PHI": "Eagles",
    "NYG": "Giants",
    "WAS": "Commanders",
    "GB": "Packers",
    "MIN": "Vikings",
    "DET": "Lions",
    "CHI": "Bears",
    "TB": "Buccaneers",
    "ATL": "Falcons",
    "CAR": "Panthers",
    "NO": "Saints",
    "SF": "49ers",
    "SEA": "Seahawks",
    "ARI": "Cardinals",
    "LAR": "Rams",
    "NE": "Patriots",
    "NYJ": "Jets",
    "MIA": "Dolphins",
    "BUF": "Bills",
    "PIT": "Steelers",
    "BAL": "Ravens",
    "CLE": "Browns",
    "CIN": "Bengals",
    "JAX": "Jaguars",
    "IND": "Colts",
    "HOU": "Texans",
    "TEN": "Titans",
    "KC": "Chiefs",
    "LV": "Raiders",
    "LAC": "Chargers",
    "DEN": "Broncos",
}


def _norm_team_to_abbr(s: str) -> Optional[str]:
    if not s:
        return None
    k = re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()
    if k in TEAM_ALIASES:
        return TEAM_ALIASES[k]
    k2 = k.replace(" ", "")
    if k2 in TEAM_ALIASES:
        return TEAM_ALIASES[k2]
    for token in k.split():
        if token in TEAM_ALIASES:
            return TEAM_ALIASES[token]
    return None


def _parse_matchup_teams(matchup: str) -> Tuple[Optional[str], Optional[str]]:
    """
    DK NFL format is typically "GB Packers @DET Lions" (no space after '@').
    """
    if not matchup:
        return None, None
    s = str(matchup).strip()

    if "@" in s:
        parts = [p.strip() for p in s.split("@", 1)]
        if len(parts) == 2:
            return parts[0], parts[1]

    for sep in [" vs ", " at ", " v ", " VS ", " Vs ", " AT "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep, 1)]
            if len(parts) == 2:
                return parts[0], parts[1]

    if "|" in s:
        parts = [p.strip() for p in s.split("|", 1)]
        if len(parts) == 2:
            return parts[0], parts[1]

    return None, None


def build_odds_from_splits(splits: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert DKNetwork splits rows into a long-format odds table.
    """
    rows = []
    for r in splits:
        matchup = r.get("matchup", "")
        game_time = r.get("game_time", "")
        market_name = r.get("market", "")
        side = r.get("side", "")
        odds = r.get("odds", 0)

        away_name, home_name = _parse_matchup_teams(matchup)
        away_abbr = _norm_team_to_abbr(away_name or "")
        home_abbr = _norm_team_to_abbr(home_name or "")
        if not away_abbr or not home_abbr:
            continue

        side_l = side.lower()
        away_long = ABBR_TO_NAME.get(away_abbr, "").lower()
        home_long = ABBR_TO_NAME.get(home_abbr, "").lower()

        label = None
        point = None

        # Moneyline
        if _is_moneyline(market_name):
            if away_name and away_name.lower() in side_l:
                label = "away"
            elif home_name and home_name.lower() in side_l:
                label = "home"
            elif away_long and away_long in side_l:
                label = "away"
            elif home_long and home_long in side_l:
                label = "home"
            else:
                label = "home" if odds < 0 else "away"

        # Spread
        elif str(market_name).lower() == "spread":
            m = re.search(r"([-+]?\d+(?:\.\d+)?)", side)
            if m:
                point = float(m.group(1))
            if away_name and away_name.lower() in side_l:
                label = "away"
            elif home_name and home_name.lower() in side_l:
                label = "home"
            elif away_long and away_long in side_l:
                label = "away"
            elif home_long and home_long in side_l:
                label = "home"

        # Totals
        elif _is_total(market_name):
            m = re.search(r"(\d+(?:\.\d+)?)", side)
            if m:
                point = float(m.group(1))
            if side_l.startswith("over"):
                label = "over"
            elif side_l.startswith("under"):
                label = "under"

        if not label or odds == 0:
            continue

        rows.append(
            {
                "home_team": home_abbr,
                "away_team": away_abbr,
                "market": market_name,
                "label": label,
                "price": odds,
                "point": point,
                "matchup": matchup,
                "game_time": game_time,
                "handle_pct": float(r.get("%handle", 0.0)),
                "bets_pct": float(r.get("%bets", 0.0)),
            }
        )

    return pd.DataFrame(rows)


def load_splits_with_dates(
    event_group: int = 88808, date_ranges: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if date_ranges is None:
        date_ranges = ["today", "tomorrow", "next7days"]

    last_splits_df = pd.DataFrame()
    last_odds_long = pd.DataFrame()
    last_used = ""

    for dr in date_ranges:
        rows = fetch_dk_splits(event_group=event_group, date_range=dr)
        splits_df = pd.DataFrame(rows)
        odds_long = build_odds_from_splits(rows) if rows else pd.DataFrame()

        if not splits_df.empty and not odds_long.empty:
            return splits_df, odds_long, dr

        last_splits_df = splits_df
        last_odds_long = odds_long
        last_used = dr

    return last_splits_df, last_odds_long, last_used


# -------------------------------------------------------------------
# MODELING: MATCHUP-AWARE TEAM METRICS & LINEAR PREDICTIONS
# -------------------------------------------------------------------

def american_to_prob(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def edge_vs_juice(
    win_prob: float, american_odds: int, stake: float = 1.0
) -> Tuple[float, float]:
    win_prob = max(0.0, min(1.0, win_prob))
    q = 1.0 - win_prob
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / -american_odds if american_odds < 0 else 1.0

    ev = win_prob * b * stake - q * stake
    if b <= 0:
        return ev, 0.0
    kelly = max(0.0, (b * win_prob - q) / b)
    return ev, kelly


@st.cache_data(show_spinner=False)
def load_history_and_pbp(seasons_back: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if nfl is None:
        st.error("nfl_data_py is not installed. `pip install nfl_data_py`")
        st.stop()

    this_year = dt.datetime.now().year
    seasons = list(range(this_year - seasons_back + 1, this_year + 1))

    sched = nfl.import_schedules(seasons)
    # REG season only, with scores
    sched = sched[(sched["game_type"] == "REG") & sched["home_score"].notna()].copy()

    # Pull play-by-play data for same seasons
    pbp = nfl.import_pbp_data(seasons, downcast=True)

    return sched, pbp


def _ensure_columns(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.copy()
    if "success" not in pbp.columns and "epa" in pbp.columns:
        pbp["success"] = (pbp["epa"] > 0).astype(float)
    if "explosive" not in pbp.columns:
        if "yards_gained" in pbp.columns:
            pbp["explosive"] = (pbp["yards_gained"] >= 20).astype(float)
        elif "epa" in pbp.columns:
            pbp["explosive"] = (pbp["epa"] >= 1.0).astype(float)
        else:
            pbp["explosive"] = 0.0
    if "pass" not in pbp.columns:
        if "play_type" in pbp.columns:
            pbp["pass"] = (pbp["play_type"] == "pass").astype(float)
        else:
            pbp["pass"] = 0.0
    if "rush" not in pbp.columns:
        if "play_type" in pbp.columns:
            pbp["rush"] = (pbp["play_type"] == "run").astype(float)
        else:
            pbp["rush"] = 0.0
    return pbp


@st.cache_data(show_spinner=False)
def build_team_metrics_and_models(
    seasons_back: int = 4,
) -> Dict[str, Any]:
    sched, pbp = load_history_and_pbp(seasons_back=seasons_back)
    pbp = _ensure_columns(pbp)

    # Filter to offensive plays
    mask_play = (
        (pbp["pass"] == 1)
        | (pbp["rush"] == 1)
    )
    if "play_type" in pbp.columns:
        mask_play |= pbp["play_type"].isin(["pass", "run"])
    pbp_play = pbp[mask_play].copy()

    # Offense metrics: by team-season (posteam)
    off_grp = pbp_play.groupby(["season", "posteam"], dropna=True)
    off = off_grp.agg(
        plays=("play_id", "count") if "play_id" in pbp_play.columns else ("epa", "count"),
        epa_per_play=("epa", "mean"),
        success_rate=("success", "mean"),
        explosive_rate=("explosive", "mean"),
        pass_rate=("pass", "mean"),
        games=("game_id", "nunique"),
    ).reset_index()
    off["plays_per_game"] = off["plays"] / off["games"]
    off.rename(columns={"posteam": "team"}, inplace=True)

    # Defense metrics: by team-season (defteam)
    def_grp = pbp_play.groupby(["season", "defteam"], dropna=True)
    deff = def_grp.agg(
        plays=("epa", "count"),
        epa_per_play=("epa", "mean"),
        success_rate=("success", "mean"),
        explosive_rate=("explosive", "mean"),
        games=("game_id", "nunique"),
    ).reset_index()
    deff["plays_per_game"] = deff["plays"] / deff["games"]
    deff.rename(columns={"defteam": "team"}, inplace=True)
    deff.rename(
        columns={
            "epa_per_play": "def_epa_per_play",
            "success_rate": "def_success_rate",
            "explosive_rate": "def_explosive_rate",
            "plays_per_game": "def_plays_per_game",
        },
        inplace=True,
    )

    team_metrics = pd.merge(
        off,
        deff[
            ["season", "team", "def_epa_per_play", "def_success_rate", "def_explosive_rate"]
        ],
        on=["season", "team"],
        how="left",
    )

    # Strength of schedule via schedule: opponents' offense/defense epa
    sched_cols = [
        "season",
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    sched = sched[sched_cols].copy()

    # Map team-season -> offensive & defensive EPA
    key = ["season", "team"]
    off_map = off.set_index(key)["epa_per_play"].to_dict()
    def_map = deff.set_index(key)["def_epa_per_play"].to_dict()

    sos_rows = []
    for _, row in sched.iterrows():
        season = row["season"]
        ht = row["home_team"]
        at = row["away_team"]

        # For each team in the game, record opponent's offense & defense strength
        for team, opp in [(ht, at), (at, ht)]:
            sos_rows.append(
                {
                    "season": season,
                    "team": team,
                    "opp_off_epa": off_map.get((season, opp), np.nan),
                    "opp_def_epa": def_map.get((season, opp), np.nan),
                }
            )

    sos_df = pd.DataFrame(sos_rows)
    sos_agg = (
        sos_df.groupby(["season", "team"])
        .agg(
            opp_off_epa_mean=("opp_off_epa", "mean"),
            opp_def_epa_mean=("opp_def_epa", "mean"),
        )
        .reset_index()
    )

    team_metrics = pd.merge(
        team_metrics, sos_agg, on=["season", "team"], how="left"
    )

    # Adjusted metrics: remove opponent strength
    team_metrics["adj_off_epa"] = (
        team_metrics["epa_per_play"] - team_metrics["opp_def_epa_mean"]
    )
    team_metrics["adj_def_epa"] = (
        team_metrics["def_epa_per_play"] - team_metrics["opp_off_epa_mean"]
    )

    # League total average per game
    sched["total_points"] = sched["home_score"] + sched["away_score"]
    sched["margin"] = sched["home_score"] - sched["away_score"]
    league_total_mean = sched["total_points"].mean()

    # Build game-level dataset for linear regression
    tm = team_metrics.set_index(["season", "team"])
    rows = []
    for _, g in sched.iterrows():
        season = g["season"]
        ht = g["home_team"]
        at = g["away_team"]

        if (season, ht) not in tm.index or (season, at) not in tm.index:
            continue

        h = tm.loc[(season, ht)]
        a = tm.loc[(season, at)]

        feat = {
            "season": season,
            "home_team": ht,
            "away_team": at,
            # basic adjusted EPA features
            "h_adj_off_epa": h["adj_off_epa"],
            "h_adj_def_epa": h["adj_def_epa"],
            "a_adj_off_epa": a["adj_off_epa"],
            "a_adj_def_epa": a["adj_def_epa"],
            # success & explosive rates
            "h_success": h["success_rate"],
            "h_def_success": h["def_success_rate"],
            "a_success": a["success_rate"],
            "a_def_success": a["def_success_rate"],
            "h_explosive": h["explosive_rate"],
            "a_explosive": a["explosive_rate"],
            # pass rates & volume as rough pace proxy
            "h_pass_rate": h["pass_rate"],
            "a_pass_rate": a["pass_rate"],
            "h_plays_pg": h["plays_per_game"],
            "a_plays_pg": a["plays_per_game"],
            # outcomes
            "total_points": g["total_points"],
            "margin": g["margin"],
        }
        rows.append(feat)

    games_df = pd.DataFrame(rows).dropna()

    feature_cols = [
        "h_adj_off_epa",
        "h_adj_def_epa",
        "a_adj_off_epa",
        "a_adj_def_epa",
        "h_success",
        "h_def_success",
        "a_success",
        "a_def_success",
        "h_explosive",
        "a_explosive",
        "h_pass_rate",
        "a_pass_rate",
        "h_plays_pg",
        "a_plays_pg",
    ]

    X = games_df[feature_cols].values.astype(float)
    X_design = np.column_stack([np.ones(len(X)), X])

    y_total = games_df["total_points"].values.astype(float)
    y_margin = games_df["margin"].values.astype(float)

    # Linear regression via least squares
    beta_total, *_ = np.linalg.lstsq(X_design, y_total, rcond=None)
    beta_margin, *_ = np.linalg.lstsq(X_design, y_margin, rcond=None)

    # Store team metrics in dict
    team_dict: Dict[Tuple[int, str], Dict[str, float]] = {}
    for (season, team), row in tm.iterrows():
        team_dict[(int(season), str(team))] = {
            "adj_off_epa": float(row["adj_off_epa"]),
            "adj_def_epa": float(row["adj_def_epa"]),
            "success_rate": float(row["success_rate"]),
            "def_success_rate": float(row["def_success_rate"]),
            "explosive_rate": float(row["explosive_rate"]),
            "pass_rate": float(row["pass_rate"]),
            "plays_per_game": float(row["plays_per_game"]),
        }

    model = {
        "feature_cols": feature_cols,
        "beta_total": beta_total,
        "beta_margin": beta_margin,
        "team_metrics": team_dict,
        "league_total_mean": float(league_total_mean),
    }

    return model


def predict_scores_for_matchup(
    model: Dict[str, Any], season: int, home_team: str, away_team: str
) -> Tuple[float, float]:
    """
    Predict home and away scores using linear models trained on matchup features.
    """
    tm = model["team_metrics"]
    key_h = (season, home_team)
    key_a = (season, away_team)

    # Fallback: if no season-specific metrics, try any season (take most recent)
    if key_h not in tm or key_a not in tm:
        candidates_h = sorted(
            [k for k in tm.keys() if k[1] == home_team], key=lambda x: x[0], reverse=True
        )
        candidates_a = sorted(
            [k for k in tm.keys() if k[1] == away_team], key=lambda x: x[0], reverse=True
        )
        if not candidates_h or not candidates_a:
            # fallback to league total split evenly
            total = model["league_total_mean"]
            return total / 2.0, total / 2.0
        key_h = candidates_h[0]
        key_a = candidates_a[0]

    h = tm[key_h]
    a = tm[key_a]

    X_vec = np.array(
        [
            h["adj_off_epa"],
            h["adj_def_epa"],
            a["adj_off_epa"],
            a["adj_def_epa"],
            h["success_rate"],
            h["def_success_rate"],
            a["success_rate"],
            a["def_success_rate"],
            h["explosive_rate"],
            a["explosive_rate"],
            h["pass_rate"],
            a["pass_rate"],
            h["plays_per_game"],
            a["plays_per_game"],
        ],
        dtype=float,
    )

    X_design = np.concatenate([[1.0], X_vec])

    beta_total = model["beta_total"]
    beta_margin = model["beta_margin"]

    pred_total = float(X_design @ beta_total)
    pred_margin = float(X_design @ beta_margin)

    home_pts = (pred_total + pred_margin) / 2.0
    away_pts = (pred_total - pred_margin) / 2.0

    return home_pts, away_pts


# -------------------------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="FOXEDGE NFL | Matchup Edgeboard",
        layout="wide",
    )

    st.title("🦊 FOXEDGE NFL | Matchup-Aware DK Edgeboard")

    with st.sidebar:
        st.header("Config")

        seasons_back = st.slider(
            "Seasons of history",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
        )
        spread_edge_thresh = st.slider(
            "Min spread edge (pts)", 0.5, 8.0, 2.0, 0.5
        )
        total_edge_thresh = st.slider(
            "Min total edge (pts)", 1.0, 12.0, 3.0, 0.5
        )
        sigma_spread = st.slider(
            "Spread volatility (σ)", 4.0, 10.0, 6.0, 0.5
        )
        sigma_total = st.slider(
            "Total volatility (σ)", 6.0, 16.0, 8.0, 1.0
        )
        kelly_cap = st.slider(
            "Kelly cap (fraction of bankroll)",
            0.01,
            0.25,
            0.07,
            0.01,
        )

        st.markdown("---")
        st.caption("Data: nfl_data_py (pbp & schedule) + DKNetwork betting splits.")

    # Build matchup-aware model
    with st.spinner("Loading NFL pbp & schedules; building matchup model..."):
        model = build_team_metrics_and_models(seasons_back=seasons_back)

    # DK splits
    with st.spinner("Fetching DKNetwork NFL betting splits (today/tomorrow/next7days)..."):
        splits_df, odds_long, dr_used = load_splits_with_dates(event_group=88808)

    if splits_df.empty or odds_long.empty:
        st.error(
            "Failed to parse odds from DKNetwork splits for today, tomorrow, or next 7 days.\n"
            "Either DK has no NFL board up or they changed markup again."
        )
        if not splits_df.empty:
            st.write("Raw splits (for debugging):")
            st.dataframe(splits_df.head(100), use_container_width=True)
        return

    st.caption(f"Using DKNetwork board for date range: **{dr_used}**")

    this_year = dt.datetime.now().year

    games = []
    grouped = odds_long.groupby(["home_team", "away_team"])
    for (home, away), df in grouped:
        pred_home, pred_away = predict_scores_for_matchup(model, this_year, home, away)
        model_spread = pred_home - pred_away  # home margin
        model_total = pred_home + pred_away

        # Spread from DK
        spread_rows = df[df["market"].str.lower() == "spread"].copy()
        home_spread = None
        away_spread = None
        if not spread_rows.empty:
            for _, r in spread_rows.iterrows():
                lbl = r["label"]
                pt = r["point"]
                if pt is None:
                    continue
                if lbl == "home":
                    home_spread = float(pt)
                elif lbl == "away":
                    away_spread = float(pt)
        if home_spread is None and away_spread is not None:
            home_spread = -float(away_spread)

        # Total from DK
        tot_rows = df[df["market"].str.lower().isin(["total", "totals"])].copy()
        total_line = None
        if not tot_rows.empty:
            over_rows = tot_rows[tot_rows["label"] == "over"]
            use_row = over_rows.iloc[0] if not over_rows.empty else tot_rows.iloc[0]
            if use_row["point"] is not None:
                total_line = float(use_row["point"])

        # Moneyline
        ml_rows = df[df["market"].apply(lambda x: _is_moneyline(str(x)))].copy()
        home_ml = None
        away_ml = None
        if not ml_rows.empty:
            for _, r in ml_rows.iterrows():
                if r["label"] == "home":
                    home_ml = int(r["price"])
                elif r["label"] == "away":
                    away_ml = int(r["price"])

        # Split handles
        handle_home = handle_away = None
        if not ml_rows.empty:
            for _, r in ml_rows.iterrows():
                if r["label"] == "home":
                    handle_home = float(r["handle_pct"])
                elif r["label"] == "away":
                    handle_away = float(r["handle_pct"])

        handle_over = handle_under = None
        if not tot_rows.empty:
            for _, r in tot_rows.iterrows():
                if r["label"] == "over":
                    handle_over = float(r["handle_pct"])
                elif r["label"] == "under":
                    handle_under = float(r["handle_pct"])

        # Spread edge & recommendation
        spread_edge_pts = None
        rec_side = None
        ev_spread = None
        kelly_spread = None

        if home_spread is not None:
            market_margin = -home_spread
            spread_edge_pts = model_spread - market_margin

            home_cover_prob = 1.0 / (1.0 + np.exp(-spread_edge_pts / sigma_spread))
            away_cover_prob = 1.0 - home_cover_prob

            home_ev, home_k = edge_vs_juice(home_cover_prob, -110)
            away_ev, away_k = edge_vs_juice(away_cover_prob, -110)

            if home_ev > away_ev:
                rec_side = f"{home} {home_spread:+g}"
                ev_spread = home_ev
                kelly_spread = min(home_k, kelly_cap)
            else:
                away_line = -home_spread
                rec_side = f"{away} {away_line:+g}"
                ev_spread = away_ev
                kelly_spread = min(away_k, kelly_cap)

        # Total edge & recommendation
        total_edge_pts = None
        rec_total = None
        ev_total = None
        kelly_total = None

        if total_line is not None:
            total_edge_pts = model_total - total_line
            over_prob = 1.0 / (1.0 + np.exp(-total_edge_pts / sigma_total))
            under_prob = 1.0 - over_prob

            over_ev, over_k = edge_vs_juice(over_prob, -110)
            under_ev, under_k = edge_vs_juice(under_prob, -110)

            if over_ev > under_ev:
                rec_total = f"Over {total_line:g}"
                ev_total = over_ev
                kelly_total = min(over_k, kelly_cap)
            else:
                rec_total = f"Under {total_line:g}"
                ev_total = under_ev
                kelly_total = min(under_k, kelly_cap)

        games.append(
            {
                "home_team": home,
                "away_team": away,
                "matchup": f"{away} @ {home}",
                "model_home": round(pred_home, 1),
                "model_away": round(pred_away, 1),
                "model_spread_home": round(model_spread, 2),
                "model_total": round(model_total, 1),
                "market_home_spread": home_spread,
                "market_total": total_line,
                "spread_edge_pts": spread_edge_pts,
                "total_edge_pts": total_edge_pts,
                "rec_spread": rec_side,
                "rec_total": rec_total,
                "ev_spread": ev_spread,
                "kelly_spread": kelly_spread,
                "ev_total": ev_total,
                "kelly_total": kelly_total,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "handle_home_ml": handle_home,
                "handle_away_ml": handle_away,
                "handle_over": handle_over,
                "handle_under": handle_under,
            }
        )

    edge_df = pd.DataFrame(games)

    spread_mask = edge_df["spread_edge_pts"].abs() >= spread_edge_thresh
    total_mask = edge_df["total_edge_pts"].abs() >= total_edge_thresh

    spread_picks = edge_df[spread_mask].copy()
    total_picks = edge_df[total_mask].copy()

    spread_picks = spread_picks.sort_values(
        "spread_edge_pts", key=lambda s: s.abs(), ascending=False
    )
    total_picks = total_picks.sort_values(
        "total_edge_pts", key=lambda s: s.abs(), ascending=False
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spread Edges")
        if spread_picks.empty:
            st.write("No spreads pass your edge threshold.")
        else:
            show_cols = [
                "matchup",
                "model_home",
                "model_away",
                "model_spread_home",
                "market_home_spread",
                "spread_edge_pts",
                "rec_spread",
                "ev_spread",
                "kelly_spread",
                "handle_home_ml",
                "handle_away_ml",
            ]
            st.dataframe(
                spread_picks[show_cols].style.format(
                    {
                        "model_home": "{:.1f}",
                        "model_away": "{:.1f}",
                        "model_spread_home": "{:+.2f}",
                        "market_home_spread": "{:+.1f}",
                        "spread_edge_pts": "{:+.2f}",
                        "ev_spread": "{:+.3f}",
                        "kelly_spread": "{:.3f}",
                        "handle_home_ml": "{:.1f}",
                        "handle_away_ml": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

    with col2:
        st.subheader("Total Edges")
        if total_picks.empty:
            st.write("No totals pass your edge threshold.")
        else:
            show_cols = [
                "matchup",
                "model_home",
                "model_away",
                "model_total",
                "market_total",
                "total_edge_pts",
                "rec_total",
                "ev_total",
                "kelly_total",
                "handle_over",
                "handle_under",
            ]
            st.dataframe(
                total_picks[show_cols].style.format(
                    {
                        "model_home": "{:.1f}",
                        "model_away": "{:.1f}",
                        "model_total": "{:.1f}",
                        "market_total": "{:.1f}",
                        "total_edge_pts": "{:+.2f}",
                        "ev_total": "{:+.3f}",
                        "kelly_total": "{:.3f}",
                        "handle_over": "{:.1f}",
                        "handle_under": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

    st.markdown("---")
    with st.expander("Raw DK splits & parsed odds", expanded=False):
        st.caption("Raw DK splits:")
        st.dataframe(splits_df.head(200), use_container_width=True, height=300)
        st.caption("Parsed odds table:")
        st.dataframe(odds_long.head(200), use_container_width=True, height=300)


if __name__ == "__main__":
    main()
