#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Playoffs Model — Automated DK + pybaseball (no CSVs)

What it does:
- Fetches MLB events and odds from DraftKings public API (moneyline + totals; tries multiple regions).
- Uses pybaseball to fetch probable pitchers (if available) and form features from Statcast.
- Applies playoff-specific weighting (shorter starter leash, bullpen leverage), weather/park hooks, optional ump bias.
- Simulates with Poisson; prices markets; computes EV and Kelly (capped).
- Prints a predictions table and, optionally, ranked bet recommendations.

Requirements:
  pip install pybaseball pandas numpy requests beautifulsoup4

Examples:
  python mlb_playoffs_auto.py --date 2025-10-01 --show-recs
  python mlb_playoffs_auto.py --show-recs --max-games 6
"""

import argparse
import datetime as dt
import math
import re
import logging
import sys
import warnings
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Config ----------
CFG = {
    "kelly_cap": 0.02,            # 2% bankroll cap
    "min_ev": 0.01,               # surface recs at EV >= 1% (tight playoff markets)
    "sim_n": 25000,
    "recent_days": 30,
    "rest_decay": 0.15,           # penalty per day under 4 SP rest days
    "bullpen_weight_oct": 0.62,
    "starter_weight_oct": 0.38,
    "cold_weather_run_scale": 0.993,  # per 5F below 70
    "wind_out_hr_boost": 1.03,
    "wind_in_hr_suppress": 0.97,
    "umpire_runs_unit": 0.15,
    "seed": 17
}

# ---------- Team name sanity layer ----------
TEAM_SYNONYMS = {
    "L.A. Dodgers": "Los Angeles Dodgers",
    "LA Dodgers": "Los Angeles Dodgers",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "L.A. Angels": "Los Angeles Angels",
    "LA Angels": "Los Angeles Angels",
    "Los Angeles Angels": "Los Angeles Angels",
    "San Francisco Giants": "San Francisco Giants",
    "Oakland Athletics": "Oakland Athletics",
    "Seattle Mariners": "Seattle Mariners",
    "San Diego Padres": "San Diego Padres",
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "Colorado Rockies": "Colorado Rockies",
    "Chicago Cubs": "Chicago Cubs",
    "Chicago White Sox": "Chicago White Sox",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "Cincinnati Reds": "Cincinnati Reds",
    "Atlanta Braves": "Atlanta Braves",
    "Miami Marlins": "Miami Marlins",
    "New York Mets": "New York Mets",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Washington Nationals": "Washington Nationals",
    "Boston Red Sox": "Boston Red Sox",
    "New York Yankees": "New York Yankees",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "Baltimore Orioles": "Baltimore Orioles",
    "Kansas City Royals": "Kansas City Royals",
    "Minnesota Twins": "Minnesota Twins",
    "Detroit Tigers": "Detroit Tigers",
    "Cleveland Guardians": "Cleveland Guardians",
    "Houston Astros": "Houston Astros",
    "Texas Rangers": "Texas Rangers",
    # Nickname-only fallbacks used occasionally by DKNetwork
    "Dodgers": "Los Angeles Dodgers",
    "Angels": "Los Angeles Angels",
    "Giants": "San Francisco Giants",
    "Athletics": "Oakland Athletics",
    "Mariners": "Seattle Mariners",
    "Padres": "San Diego Padres",
    "Diamondbacks": "Arizona Diamondbacks",
    "Rockies": "Colorado Rockies",
    "Cubs": "Chicago Cubs",
    "White Sox": "Chicago White Sox",
    "Brewers": "Milwaukee Brewers",
    "Cardinals": "St. Louis Cardinals",
    "Pirates": "Pittsburgh Pirates",
    "Reds": "Cincinnati Reds",
    "Braves": "Atlanta Braves",
    "Marlins": "Miami Marlins",
    "Mets": "New York Mets",
    "Phillies": "Philadelphia Phillies",
    "Nationals": "Washington Nationals",
    "Red Sox": "Boston Red Sox",
    "Yankees": "New York Yankees",
    "Rays": "Tampa Bay Rays",
    "Blue Jays": "Toronto Blue Jays",
    "Orioles": "Baltimore Orioles",
    "Royals": "Kansas City Royals",
    "Twins": "Minnesota Twins",
    "Tigers": "Detroit Tigers",
    "Guardians": "Cleveland Guardians",
    "Astros": "Houston Astros",
    "Rangers": "Texas Rangers",
}

ABBR = {
    "Los Angeles Dodgers": "LAD", "Los Angeles Angels": "LAA", "San Francisco Giants": "SFG", "Oakland Athletics": "OAK",
    "Seattle Mariners": "SEA", "San Diego Padres": "SDP", "Arizona Diamondbacks": "ARI", "Colorado Rockies": "COL",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Milwaukee Brewers": "MIL", "St. Louis Cardinals": "STL",
    "Pittsburgh Pirates": "PIT", "Cincinnati Reds": "CIN", "Atlanta Braves": "ATL", "Miami Marlins": "MIA",
    "New York Mets": "NYM", "Philadelphia Phillies": "PHI", "Washington Nationals": "WSN", "Boston Red Sox": "BOS",
    "New York Yankees": "NYY", "Tampa Bay Rays": "TBR", "Toronto Blue Jays": "TOR", "Baltimore Orioles": "BAL",
    "Kansas City Royals": "KCR", "Minnesota Twins": "MIN", "Detroit Tigers": "DET", "Cleveland Guardians": "CLE",
    "Houston Astros": "HOU", "Texas Rangers": "TEX"
}

# ---------- pybaseball imports with soft failure ----------
try:
    from pybaseball import playerid_lookup, statcast, team_pitching, park_factors
    try:
        from pybaseball import probables  # may not exist on older versions
        HAS_PROBABLES = True
    except Exception:
        HAS_PROBABLES = False
except Exception:
    playerid_lookup = None
    statcast = None
    team_pitching = None
    park_factors = None
    HAS_PROBABLES = False

# ---------- Odds: DraftKings event group for MLB ----------
DK_EVENTGROUP_URLS = [
    "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/84240?format=json",
    "https://sportsbook-us-ny.draftkings.com/sites/US-NY-SB/api/v5/eventgroups/84240?format=json",
    "https://sportsbook-ca-on.draftkings.com/sites/CA-ON-SB/api/v5/eventgroups/84240?format=json",
    "https://sportsbook.draftkings.com/api/v5/eventgroups/84240?format=json",
    "https://sportsbook.draftkings.com/api/sportscontent/v1/eventgroup/84240?format=json",
]

# DKNetwork splits page (used for odds + %bets/%handle pairing)
DKNETWORK_SPLITS_BASE = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
EG_MLB = 84240

# ---------- Utility ----------
def american_to_decimal(american: float) -> float:
    if pd.isna(american):
        return np.nan
    return 1 + (american / 100.0 if american >= 100 else 100.0 / abs(american))

def decimal_to_american(decimal_odds: float) -> float:
    return (decimal_odds - 1) * 100.0 if decimal_odds >= 2.0 else -100.0 / (decimal_odds - 1)

def kelly_fraction(p: float, dec_odds: float) -> float:
    if dec_odds <= 1.0 or not (0 < p < 1):
        return 0.0
    b = dec_odds - 1.0
    q = 1.0 - p
    frac = (b*p - q) / b
    return max(0.0, frac)

def conservative_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def today_str() -> str:
    # timezone-aware UTC date, avoids deprecation warnings
    return dt.datetime.now(dt.timezone.utc).date().isoformat()

# wall-clock stamp; timezone isn’t critical for modeling
def now_pt():
    return dt.datetime.now()

def parse_kv_blob(blob: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in str(blob).split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v  # string like wind dir
    return out

# ---------- DKNetwork splits utils ----------
def safe_int(x, default=None):
    try:
        return int(str(x).replace("−", "-"))
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def parse_total_side(side_text: str):
    s = str(side_text or "").strip()
    import re
    m = re.match(r"(?i)^(over|under)\s+([0-9]+(?:\.[0-9])?)", s)
    if not m:
        return None, None
    return m.group(1).title(), float(m.group(2))

#
# ---------- DraftKings scraping ----------

def fetch_dk_eventgroup(override_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Hardened DraftKings eventgroup fetch with retries and override URL."""
    import time
    ses = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://sportsbook.draftkings.com/",
        "Origin": "https://sportsbook.draftkings.com",
        "Connection": "keep-alive",
    }

    urls = [override_url] if override_url else DK_EVENTGROUP_URLS
    for url in urls:
        if not url:
            continue
        delay = 0.75
        for _ in range(3):
            try:
                r = ses.get(url, headers=headers, timeout=15)
                if r.ok:
                    js = r.json()
                    if js and "eventGroup" in js:
                        return js
            except requests.exceptions.SSLError:
                # hop to next base URL on TLS flake
                break
            except requests.exceptions.RequestException:
                pass
            time.sleep(delay)
            delay *= 1.8
    return None


def fetch_dk_splits(event_group: int, date_range: str = "today", timeout=20) -> pd.DataFrame:
    """
    Scrape DKNetwork betting splits pages (handles pagination).
    Returns tidy DataFrame with columns:
      matchup, game_time, market, side, odds, %handle, %bets, update_time
    """
    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
    first_url = base + "?" + "&".join(f"{k}={v}" for k, v in params.items())

    def get_html(url):
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text

    def discover_pages(html):
        soup = BeautifulSoup(html, "html.parser")
        urls = {first_url}
        pag = soup.select_one("div.tb_pagination")
        if pag:
            for a in pag.find_all("a", href=True):
                if "tb_page=" in a["href"]:
                    urls.add(a["href"])
        # sort by tb_page= if present
        def page_num(u):
            m = re.search(r"tb_page=(\d+)", u)
            return int(m.group(1)) if m else 1
        return [u for u in sorted(urls, key=page_num)]

    def parse_page(html):
        soup = BeautifulSoup(html, "html.parser")
        games = soup.select("div.tb-se")
        out = []
        now = now_pt()
        for g in games:
            title_node = g.select_one("div.tb-se-title h5")
            if not title_node:
                continue
            matchup = title_node.get_text(strip=True)
            # time text not reliably parseable; keep raw
            tnode = g.select_one("div.tb-se-title span")
            gtime = (tnode.get_text(strip=True) if tnode else "").replace("\xa0", " ")
            # market sections
            for section in g.select(".tb-market-wrap > div"):
                head = section.select_one(".tb-se-head > div")
                if not head:
                    continue
                market_name = head.get_text(strip=True)
                # We support Moneyline, Spread, Total; ML/Total are most reliable
                if market_name not in ("Moneyline", "Spread", "Total"):
                    continue
                for row in section.select(".tb-sodd"):
                    side_el = row.select_one(".tb-slipline")
                    odds_el = row.select_one("a.tb-odd-s")
                    if not side_el or not odds_el:
                        continue
                    side_raw = side_el.get_text(strip=True)
                    odds_raw = odds_el.get_text(strip=True)
                    odds_val = safe_int(odds_raw, default=None)
                    # find first two percent texts in the row (%handle then %bets)
                    pct_texts = [s.strip().replace("%", "") for s in row.find_all(string=lambda t: "%" in t)]
                    pct_handle, pct_bets = (pct_texts + ["", ""])[:2]
                    out.append({
                        "matchup": matchup,
                        "game_time": gtime,
                        "market": market_name,
                        "side": side_raw,
                        "odds": odds_val,
                        "%handle": pd.to_numeric(pct_handle, errors="coerce"),
                        "%bets": pd.to_numeric(pct_bets, errors="coerce"),
                        "update_time": now
                    })
        return out

    try:
        first_html = get_html(first_url)
    except Exception as e:
        logging.error(f"fetch_dk_splits first page failed: {e}")
        return pd.DataFrame(columns=["matchup","game_time","market","side","odds","%handle","%bets","update_time"])

    pages = discover_pages(first_html)
    records = []
    for url in pages:
        html = first_html if url == first_url else get_html(url)
        records.extend(parse_page(html))

    df = pd.DataFrame.from_records(records)
    if df.empty and date_range == "today":
        # Try tomorrow (DK sometimes shifts content boundary)
        return fetch_dk_splits(event_group, "tomorrow", timeout=timeout)

    # clean/normalize
    for c in ["%handle", "%bets"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df.dropna(subset=["matchup", "market", "side"])
    # normalize keys for grouping
    df["market_norm"] = df["market"].str.strip().str.lower()
    df["side_norm"] = df["side"].str.strip().str.lower()
    return df.reset_index(drop=True)


def games_from_splits(df: pd.DataFrame, target_date: str) -> List[Dict[str, Any]]:
    """Build list of game dicts {date, home, away, home_ml, away_ml, total, over_ml, under_ml}
    from DKNetwork splits. We infer home/away from matchup text: 'A @ B' or 'A at B' -> A away, B home;
    'A vs B' -> treat second as home. If names don't map to ABBR, skip.
    """
    import re
    out: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return out

    for matchup, g in df.groupby("matchup"):
        m = str(matchup)
        if "@" in m:
            parts = [p.strip() for p in m.split("@", 1)]
        elif re.search(r"\bat\b", m, flags=re.IGNORECASE):
            parts = [p.strip() for p in re.split(r"\bat\b", m, maxsplit=1, flags=re.IGNORECASE)]
        elif "vs" in m.lower():
            parts = [p.strip() for p in re.split(r"(?i)vs\.?", m, maxsplit=1)]
        else:
            parts = [p.strip() for p in re.split(r"\s+-\s+", m, maxsplit=1)]
        if len(parts) != 2:
            continue
        away_name_raw, home_name_raw = parts[0], parts[1]
        away = TEAM_SYNONYMS.get(away_name_raw, away_name_raw)
        home = TEAM_SYNONYMS.get(home_name_raw, home_name_raw)
        if away not in ABBR or home not in ABBR:
            continue

        sub_ml = g[g["market_norm"] == "moneyline"]
        home_ml = away_ml = math.nan
        home_tokens = {home.lower(), ABBR.get(home, "").lower(), home.split()[-1].lower()}
        away_tokens = {away.lower(), ABBR.get(away, "").lower(), away.split()[-1].lower()}
        for _, r in sub_ml.iterrows():
            side_raw = str(r["side"]).strip()
            side = side_raw.lower()
            # direct team name/abbr/mascot checks
            if side in home_tokens or any(tok and tok in side for tok in home_tokens) or side == "home":
                home_ml = r["odds"]
            elif side in away_tokens or any(tok and tok in side for tok in away_tokens) or side == "away":
                away_ml = r["odds"]
        # Fallback: if one side didn’t map, try assigning by order of appearance
        if (pd.isna(home_ml) or pd.isna(away_ml)) and len(sub_ml) == 2:
            first_odds = sub_ml.iloc[0]["odds"]
            second_odds = sub_ml.iloc[1]["odds"]
            # If we couldn’t identify which is which, assume first listed corresponds to away, second to home for "A @ B"
            if pd.isna(away_ml):
                away_ml = first_odds
            if pd.isna(home_ml):
                home_ml = second_odds

        sub_tot = g[g["market_norm"] == "total"]
        total_line = over_ml = under_ml = math.nan
        if not sub_tot.empty:
            totals = []
            for _, r in sub_tot.iterrows():
                ou, tot = parse_total_side(str(r["side"]))
                if ou and tot is not None:
                    totals.append((ou, float(tot), r["odds"]))
            if totals:
                vals = {}
                for ou, t, price in totals:
                    vals[t] = vals.get(t, 0) + 1
                common = max(vals.items(), key=lambda x: x[1])[0]
                for ou, t, price in totals:
                    if abs(t - common) < 1e-6:
                        if ou == "Over" and not pd.isna(price):
                            over_ml = price
                            total_line = t
                        if ou == "Under" and not pd.isna(price):
                            under_ml = price
                            total_line = t
        out.append({
            "event_id": None,
            "date": target_date,
            "home": home, "away": away,
            "home_ml": float(home_ml) if not pd.isna(home_ml) else math.nan,
            "away_ml": float(away_ml) if not pd.isna(away_ml) else math.nan,
            "total": float(total_line) if not pd.isna(total_line) else math.nan,
            "over_ml": float(over_ml) if not pd.isna(over_ml) else math.nan,
            "under_ml": float(under_ml) if not pd.isna(under_ml) else math.nan,
        })
    return out

def extract_ml_totals_from_dk(js: Dict[str, Any], target_date: str) -> List[Dict[str, Any]]:
    out = []
    eg = js.get("eventGroup", {})
    events = eg.get("events", []) or []
    categories = eg.get("offerCategories", []) or []
    moneyline_market: Dict[str, List[Dict[str, Any]]] = {}
    total_market: Dict[str, List[Dict[str, Any]]] = {}

    for cat in categories:
        for subcat in cat.get("offerSubcategoryDescriptors", []):
            for desc in subcat.get("offerSubcategory", {}).get("offerDescriptors", []):
                key = (desc.get("label") or "").lower()
                for offer in desc.get("offers", []):
                    for market in offer:
                        event_id = str(market.get("eventId"))
                        outcomes = market.get("outcomes", []) or []
                        if "moneyline" in key and "inning" not in key:
                            moneyline_market.setdefault(event_id, outcomes)
                        if "total" in key and ("runs" in key or "game total" in key) and "inning" not in key:
                            total_market.setdefault(event_id, []).extend(outcomes)

    for ev in events:
        try:
            eid = str(ev.get("eventId"))
            start = (ev.get("startDate") or "")[:10]
            if start != target_date:
                continue
            home = TEAM_SYNONYMS.get(ev.get("teamName2", ""), ev.get("teamName2", ""))
            away = TEAM_SYNONYMS.get(ev.get("teamName1", ""), ev.get("teamName1", ""))
            if home not in ABBR or away not in ABBR:
                continue
            # Moneyline
            h_ml = a_ml = np.nan
            for oc in moneyline_market.get(eid, []):
                label = (oc.get("label") or "").strip().lower()
                price = oc.get("oddsAmerican")
                if price is None:
                    continue
                if "home" in label or home.lower() in label:
                    h_ml = float(price)
                elif "away" in label or away.lower() in label:
                    a_ml = float(price)
            # Totals: pick modal line to avoid alt totals noise
            total_line = over_ml = under_ml = np.nan
            outs = total_market.get(eid, [])
            if outs:
                lines = [float(o["line"]) for o in outs if o.get("line") is not None]
                if lines:
                    vals, counts = np.unique(np.round(lines, 1), return_counts=True)
                    common_line = float(vals[np.argmax(counts)])
                    for oc in outs:
                        if oc.get("line") is None:
                            continue
                        if abs(float(oc["line"]) - common_line) < 1e-6:
                            lbl = (oc.get("label") or "").lower()
                            price = oc.get("oddsAmerican")
                            if price is None:
                                continue
                            if "over" in lbl:
                                total_line = common_line
                                over_ml = float(price)
                            elif "under" in lbl:
                                total_line = common_line
                                under_ml = float(price)
            out.append({
                "event_id": eid,
                "date": start,
                "home": home, "away": away,
                "home_ml": h_ml, "away_ml": a_ml,
                "total": total_line, "over_ml": over_ml, "under_ml": under_ml
            })
        except Exception:
            continue
    return out

# ---------- pybaseball features ----------
def last_n_team_offense(team_abbr: str, end_date: str, days: int = 30) -> Dict[str, float]:
    neutral = {"woba": 0.315, "iso": 0.160, "bb_k": 0.40, "barrel_rate": 0.065}
    if statcast is None:
        return neutral
    try:
        end = pd.to_datetime(end_date)
        start = end - pd.Timedelta(days=days)
        df = statcast(start_dt=start.strftime("%Y-%m-%d"), end_dt=end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return neutral
        df = df[df["bat_team"] == team_abbr]
        if df.empty:
            return neutral
        df["is_bb"] = (df["events"] == "walk").astype(int)
        df["is_k"] = (df["events"] == "strikeout").astype(int)
        df["is_hr"] = (df["events"] == "home_run").astype(int)
        df["is_bip"] = df["description"].notnull().astype(int)
        weights = {"single": 0.90,"double":1.25,"triple":1.60,"home_run":2.00,"walk":0.70,"hit_by_pitch":0.72}
        df["woba_contrib"] = 0.0
        for ev, w in weights.items():
            df.loc[df["events"] == ev, "woba_contrib"] = w
        pa = (df["is_bb"] + df["is_k"] + df["is_bip"]).sum()
        if pa <= 0:
            return neutral
        woba = df["woba_contrib"].sum() / pa
        iso = df["is_hr"].sum() / max(1, df["is_bip"].sum())
        bb_k = df["is_bb"].sum() / max(1, df["is_k"].sum())
        barrel_rate = df.get("barrel", pd.Series([0]*len(df))).mean() if "barrel" in df.columns else 0.065
        return {"woba": float(woba), "iso": float(iso), "bb_k": float(bb_k), "barrel_rate": float(barrel_rate)}
    except Exception:
        return neutral

def pitcher_recent_form(name: str, end_date: str, days: int = 30) -> Dict[str, Any]:
    neutral = {"k_bb": 0.13, "xwoba": 0.315, "gb": 0.42, "velo_trend": 0.0, "throws": "R", "last_start": None}
    if playerid_lookup is None or statcast is None or not name or name.strip() == "":
        return neutral
    try:
        parts = name.strip().split()
        pl = playerid_lookup(last=parts[-1], first=" ".join(parts[:-1]))
        if pl is None or pl.empty:
            return neutral
        pid = int(pl.iloc[0]["key_mlbam"])
        end = pd.to_datetime(end_date)
        start = end - pd.Timedelta(days=days)
        df = statcast(start_dt=start.strftime("%Y-%m-%d"), end_dt=end.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return neutral
        pdf = df[df["pitcher"] == pid].copy()
        if pdf.empty:
            return neutral
        pdf["is_bb"] = (pdf["events"] == "walk").astype(int)
        pdf["is_k"] = (pdf["events"] == "strikeout").astype(int)
        xw = pdf["estimated_woba_using_speedangle"].mean() if "estimated_woba_using_speedangle" in pdf.columns else 0.315
        k_rate = pdf["is_k"].sum() / max(1, len(pdf))
        bb_rate = pdf["is_bb"].sum() / max(1, len(pdf))
        gb = pdf.get("gb", pd.Series([0.42]*len(pdf))).mean()
        if "release_speed" in pdf.columns and "game_date" in pdf.columns:
            tmp = pdf.dropna(subset=["release_speed"]).groupby("game_date")["release_speed"].mean()
            velo_trend = float(tmp.iloc[-5:].mean() - tmp.iloc[:5].mean()) if len(tmp) >= 10 else 0.0
        else:
            velo_trend = 0.0
        throws = str(pl.iloc[0].get("throws", "R"))[:1] or "R"
        last_date = str(pd.to_datetime(pdf["game_date"]).max().date()) if "game_date" in pdf.columns else None
        return {"k_bb": float(k_rate - bb_rate), "xwoba": float(xw), "gb": float(gb), "velo_trend": float(velo_trend), "throws": throws, "last_start": last_date}
    except Exception:
        return neutral

def team_bullpen_quality(team_abbr: str, season: int) -> Dict[str, float]:
    neutral = {"xFIP": 4.20, "WAR": 0.0}
    try:
        if team_pitching is None:
            return neutral
        df = team_pitching(season, ind=1)
        if df is None or df.empty:
            return neutral
        name = [k for k, v in ABBR.items() if v == team_abbr]
        tname = name[0] if name else team_abbr
        row = df[df["Team"].str.contains(tname.split()[-1], case=False, regex=False)]
        if row.empty:
            return neutral
        xFIP = float(row.iloc[0].get("xFIP", 4.20))
        WAR = float(row.iloc[0].get("WAR", 0.0))
        return {"xFIP": xFIP, "WAR": WAR}
    except Exception:
        return neutral

def load_park_factors_df() -> Optional[pd.DataFrame]:
    if park_factors is None:
        return None
    try:
        return park_factors()
    except Exception:
        return None

# ---------- Environment & conversions ----------
def rest_days(last_start: Optional[str], game_date: str) -> int:
    if not last_start:
        return 4
    try:
        return int((pd.to_datetime(game_date) - pd.to_datetime(last_start)).days)
    except Exception:
        return 4

def env_run_multiplier(temp_f: Optional[float], wind_mph: Optional[float], wind_dir: Optional[str], park_row: Optional[pd.Series]) -> float:
    m = 1.0
    if temp_f is not None:
        try:
            m *= CFG["cold_weather_run_scale"] ** max(0, (70 - float(temp_f)) / 5.0)
        except Exception:
            pass
    if wind_mph is not None and wind_dir:
        try:
            w = float(wind_mph)
            d = str(wind_dir).lower()
            if w >= 10 and any(k in d for k in ["out","rf","cf","lf"]):
                m *= CFG["wind_out_hr_boost"]
            elif w >= 10 and "in" in d:
                m *= CFG["wind_in_hr_suppress"]
        except Exception:
            pass
    if park_row is not None and not park_row.empty:
        pf = park_row.get("Basic (5yr)", np.nan)
        if not pd.isna(pf):
            try:
                m *= float(pf) / 100.0
            except Exception:
                pass
    return float(m)

def umpire_total_shift(ump_sz: Optional[float]) -> float:
    if ump_sz is None:
        return 0.0
    try:
        return float(ump_sz) * CFG["umpire_runs_unit"]
    except Exception:
        return 0.0

def blend_runs_mu(sp_form: Dict[str, Any], opp_off: Dict[str, float], pen_q: Dict[str, float], rest_d: int) -> float:
    base = 4.40 * (opp_off["woba"] / 0.315) * (1.0 + 0.5 * (opp_off["iso"] - 0.160))
    sp_penalty = 1.0 + (CFG["rest_decay"] * (4 - rest_d)) if rest_d < 4 else 1.0
    sp_adj = (1.0 - 0.8 * (sp_form["k_bb"] - 0.12)) * (sp_form["xwoba"] / 0.315) * (1.0 - 0.15 * (sp_form["gb"] - 0.42))
    pen_adj = (pen_q["xFIP"] / 4.20)
    mu = CFG["starter_weight_oct"] * sp_adj * sp_penalty + CFG["bullpen_weight_oct"] * pen_adj
    return float(base * mu)

def poisson_price(mu_home: float, mu_away: float, total_shift: float, env_mult: float, n: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    mu_h = max(0.2, mu_home * env_mult + 0.5 * total_shift)
    mu_a = max(0.2, mu_away * env_mult + 0.5 * total_shift)
    rs_h = rng.poisson(mu_h, size=n)
    rs_a = rng.poisson(mu_a, size=n)
    home_win = (rs_h > rs_a).mean() + 0.5*(rs_h == rs_a).mean()
    tot = rs_h + rs_a
    return {
        "home_win_prob": float(home_win),
        "mu_home": float(mu_h),
        "mu_away": float(mu_a),
        "mean_total": float(tot.mean()),
        "sd_total": float(tot.std())
    }

# ---------- Probable pitchers ----------
def get_probables_for_date(date_str: str) -> Dict[Tuple[str, str], Tuple[str, str]]:
    prob_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    if not HAS_PROBABLES:
        return prob_map
    try:
        df = probables(date_str, date_str)
        if df is None or df.empty:
            return prob_map
        def canon(x: str) -> str:
            return TEAM_SYNONYMS.get(x, x)
        rows = []
        for _, r in df.iterrows():
            team = canon(str(r.get("team", "")))
            opp = canon(str(r.get("opponent", "")))
            pit = str(r.get("pitcher", "")).strip()
            if team and opp:
                rows.append((team, opp, pit))
        for team, opp, pit in rows:
            key1 = (team, opp)
            key2 = (opp, team)
            if key1 in prob_map:
                a, h = prob_map[key1]
                if not a:
                    prob_map[key1] = (pit, h)
            else:
                prob_map[key1] = (pit, "")
            if key2 in prob_map:
                a, h = prob_map[key2]
                if not h:
                    prob_map[key2] = (a, pit)
            else:
                prob_map[key2] = ("", pit)
        return prob_map
    except Exception:
        return prob_map

# ---------- Park factors lookup ----------
def park_row_for_team(team_full: str, pf_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if pf_df is None or "Team" not in pf_df.columns:
        return None
    try:
        token = team_full.split()[-1]
        df = pf_df[pf_df["Team"].str.contains(token, case=False, na=False)]
        if not df.empty:
            return df.iloc[0]
    except Exception:
        pass
    return None

# ---------- Game modeling ----------
def model_one(game: Dict[str, Any], date: str) -> Dict[str, Any]:
    home = TEAM_SYNONYMS.get(game["home"], game["home"])
    away = TEAM_SYNONYMS.get(game["away"], game["away"])
    home_abbr = ABBR.get(home)
    away_abbr = ABBR.get(away)
    if home_abbr is None or away_abbr is None:
        raise ValueError(f"Team mapping failed for {away} @ {home}")

    # Probable pitchers
    asp, hsp = "", ""
    # fast try via daily probables map
    prob_map = get_probables_for_date(date)
    if (away, home) in prob_map:
        asp, hsp = prob_map[(away, home)]

    # Offense form
    home_off = last_n_team_offense(home_abbr, date, days=CFG["recent_days"])
    away_off = last_n_team_offense(away_abbr, date, days=CFG["recent_days"])

    # Pitcher form
    hsp_form = pitcher_recent_form(hsp or "", date, days=CFG["recent_days"])
    asp_form = pitcher_recent_form(asp or "", date, days=CFG["recent_days"])
    h_rest = rest_days(hsp_form.get("last_start"), date)
    a_rest = rest_days(asp_form.get("last_start"), date)

    # Bullpen proxies
    season = pd.to_datetime(date).year
    hpen = team_bullpen_quality(home_abbr, season)
    apen = team_bullpen_quality(away_abbr, season)

    # Env: park factors only (hooks exist for wx/ump if you wire sources)
    pf_df = load_park_factors_df()
    p_row = park_row_for_team(home, pf_df)
    env_mult = env_run_multiplier(temp_f=None, wind_mph=None, wind_dir=None, park_row=p_row)
    ump_shift = umpire_total_shift(None)

    # Expected runs
    mu_away = blend_runs_mu(asp_form, home_off, apen, a_rest)
    mu_home = blend_runs_mu(hsp_form, away_off, hpen, h_rest)

    sim = poisson_price(mu_home, mu_away, ump_shift, env_mult, n=CFG["sim_n"], seed=abs(hash(home+away+date)) % (2**32-1))

    out = {
        "date": date,
        "away": away, "home": home,
        "away_sp": asp, "home_sp": hsp,
        "mu_home": sim["mu_home"], "mu_away": sim["mu_away"],
        "mean_total": sim["mean_total"], "sd_total": sim["sd_total"],
        "home_win_prob": sim["home_win_prob"],
        "home_ml": game.get("home_ml", np.nan), "away_ml": game.get("away_ml", np.nan),
        "total": game.get("total", np.nan), "over_ml": game.get("over_ml", np.nan), "under_ml": game.get("under_ml", np.nan)
    }

    # Moneyline pricing
    if not pd.isna(out["home_ml"]) and not pd.isna(out["away_ml"]):
        ph = conservative_clip(sim["home_win_prob"], 0.01, 0.99)
        pa = 1 - ph
        home_dec, away_dec = american_to_decimal(float(out["home_ml"])), american_to_decimal(float(out["away_ml"]))
        out["home_fair_odds"] = float(decimal_to_american(1.0 / ph))
        out["away_fair_odds"] = float(decimal_to_american(1.0 / pa))
        out["home_ev"] = ph*(home_dec-1) - (1-ph)
        out["away_ev"] = pa*(away_dec-1) - (1-pa)
        out["home_kelly"] = min(CFG["kelly_cap"], max(0.0, kelly_fraction(ph, home_dec)))
        out["away_kelly"] = min(CFG["kelly_cap"], max(0.0, kelly_fraction(pa, away_dec)))

    # Totals pricing
    if not pd.isna(out["total"]) and not pd.isna(out["over_ml"]) and not pd.isna(out["under_ml"]):
        mu = sim["mean_total"]; sd = max(0.8, math.sqrt(mu))
        z = (float(out["total"]) + 0.5 - mu) / sd
        from math import erf, sqrt
        over_prob = 0.5*(1 - erf(z / sqrt(2)))
        under_prob = 1 - over_prob
        out["over_prob"] = float(over_prob)
        out["under_prob"] = float(under_prob)
        over_dec = american_to_decimal(float(out["over_ml"]))
        under_dec = american_to_decimal(float(out["under_ml"]))
        out["over_ev"] = over_prob*(over_dec-1) - (1-over_prob)
        out["under_ev"] = under_prob*(under_dec-1) - (1-under_prob)
        out["over_kelly"] = min(CFG["kelly_cap"], max(0.0, kelly_fraction(over_prob, over_dec)))
        out["under_kelly"] = min(CFG["kelly_cap"], max(0.0, kelly_fraction(under_prob, under_dec)))

    return out

# ---------- DK -> games ----------
def build_games_from_dk(date: str, dk_url: Optional[str] = None) -> List[Dict[str, Any]]:
    # 1) Try DKNetwork splits first (pairs sides correctly and often works when JSON API sulks)
    # Decide range: map target date vs today
    today_iso = dt.datetime.now(dt.timezone.utc).date().isoformat()
    if date == today_iso:
        splits = fetch_dk_splits(EG_MLB, "today")
    else:
        # simple heuristic: if target date is tomorrow vs UTC today, use "tomorrow"; else still try today
        try:
            d_target = pd.to_datetime(date).date()
            d_today = pd.to_datetime(today_iso).date()
            range_key = "tomorrow" if (d_target - d_today).days == 1 else "today"
        except Exception:
            range_key = "today"
        splits = fetch_dk_splits(EG_MLB, range_key)
    games = games_from_splits(splits, date)
    if games:
        return games

    # 2) Fallback to DraftKings eventgroup JSON
    js = fetch_dk_eventgroup(override_url=dk_url)
    if not js:
        raise RuntimeError("Failed to fetch DraftKings event group.")
    events = extract_ml_totals_from_dk(js, target_date=date)
    if not events:
        raise RuntimeError(f"No MLB events found in DK feed for {date}.")
    return events

# ---------- Cleanup and recs ----------
def modelled_cleanup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    keys = [
        "home_ml","away_ml","home_fair_odds","away_fair_odds","home_ev","away_ev","home_kelly","away_kelly",
        "total","over_ml","under_ml","over_prob","under_prob","over_ev","under_ev","over_kelly","under_kelly"
    ]
    for r in rows:
        rr = dict(r)
        for k in keys:
            if k not in rr:
                rr[k] = np.nan
        out.append(rr)
    return out

def build_recs(df: pd.DataFrame) -> pd.DataFrame:
    picks: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        # ML
        if not pd.isna(r.get("home_ev")) and r["home_ev"] >= CFG["min_ev"] and r.get("home_kelly", 0) > 0:
            picks.append({
                "date": r["date"], "market": "ML", "selection": f"HOME {r['home']}",
                "game": f"{r['away']} @ {r['home']}",
                "price": int(r["home_ml"]) if not pd.isna(r["home_ml"]) else None,
                "fair": int(round(r["home_fair_odds"])) if not pd.isna(r.get("home_fair_odds")) else None,
                "edge": round(float(r["home_ev"]), 4), "kelly": round(float(r["home_kelly"]), 4)
            })
        if not pd.isna(r.get("away_ev")) and r["away_ev"] >= CFG["min_ev"] and r.get("away_kelly", 0) > 0:
            picks.append({
                "date": r["date"], "market": "ML", "selection": f"AWAY {r['away']}",
                "game": f"{r['away']} @ {r['home']}",
                "price": int(r["away_ml"]) if not pd.isna(r["away_ml"]) else None,
                "fair": int(round(r["away_fair_odds"])) if not pd.isna(r.get("away_fair_odds")) else None,
                "edge": round(float(r["away_ev"]), 4), "kelly": round(float(r["away_kelly"]), 4)
            })
        # Totals
        if not pd.isna(r.get("total")):
            if not pd.isna(r.get("over_ev")) and r["over_ev"] >= CFG["min_ev"] and r.get("over_kelly", 0) > 0:
                picks.append({
                    "date": r["date"], "market": "TOTAL", "selection": f"OVER {r['total']}",
                    "game": f"{r['away']} @ {r['home']}",
                    "price": int(r["over_ml"]) if not pd.isna(r["over_ml"]) else None,
                    "fair": int(round(decimal_to_american(1.0/float(r["over_prob"])))) if not pd.isna(r.get("over_prob")) else None,
                    "edge": round(float(r["over_ev"]), 4), "kelly": round(float(r["over_kelly"]), 4)
                })
            if not pd.isna(r.get("under_ev")) and r["under_ev"] >= CFG["min_ev"] and r.get("under_kelly", 0) > 0:
                picks.append({
                    "date": r["date"], "market": "TOTAL", "selection": f"UNDER {r['total']}",
                    "game": f"{r['away']} @ {r['home']}",
                    "price": int(r["under_ml"]) if not pd.isna(r["under_ml"]) else None,
                    "fair": int(round(decimal_to_american(1.0/float(r["under_prob"])))) if not pd.isna(r.get("under_prob")) else None,
                    "edge": round(float(r["under_ev"]), 4), "kelly": round(float(r["under_kelly"]), 4)
                })
    recs = pd.DataFrame(picks)
    if not recs.empty:
        recs.sort_values(by=["edge","kelly"], ascending=[False, False], inplace=True)
        recs.reset_index(drop=True, inplace=True)
    return recs

# ---------- CLI / Main ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLB Playoffs Model — Automated DK + pybaseball (no CSVs)")
    p.add_argument("--date", default=today_str(), help="YYYY-MM-DD (default: today UTC)")
    p.add_argument("--max-games", type=int, default=None, help="Limit number of games to process")
    p.add_argument("--show-recs", action="store_true", help="Show ranked bet recommendations")
    # Hooks for weather/ump if you wire a data source later:
    p.add_argument("--wx", action="append", default=[], help="Weather per game 'DET@BOS:temp=58,wind=12,dir=out'")
    p.add_argument("--ump", action="append", default=[], help="Ump per game 'DET@BOS:sz=0.4'")
    p.add_argument("--dk-url", default=None, help="Override DK eventgroup URL if your region blocks defaults")
    return p

def main():
    args = build_parser().parse_args()

    # optional env maps, not used unless you pass --wx/--ump
    env_map: Dict[str, Dict[str, Any]] = {}
    for s in args.wx:
        try:
            k, blob = s.split(":", 1)
            env_map.setdefault(k.strip().upper(), {}).update(parse_kv_blob(blob))
        except Exception:
            pass
    for s in args.ump:
        try:
            k, blob = s.split(":", 1)
            env_map.setdefault(k.strip().upper(), {}).update(parse_kv_blob(blob))
        except Exception:
            pass

    try:
        games = build_games_from_dk(args.date, dk_url=args.dk_url)
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        sys.exit(2)

    if args.max_games is not None:
        games = games[:args.max_games]

    modeled: List[Dict[str, Any]] = []
    for g in games:
        try:
            modeled.append(model_one(g, date=args.date))
        except Exception as e:
            print(f"[warn] failed modeling {g.get('away')} @ {g.get('home')}: {e}", file=sys.stderr)
            continue

    if not modeled:
        print("[fatal] No games modeled.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(modelled_cleanup(modeled))

    print("\n=== Predictions ===")
    cols = ["date","away","home","away_sp","home_sp","home_win_prob",
            "home_ml","away_ml","home_fair_odds","away_fair_odds","home_ev","away_ev",
            "total","over_ml","under_ml","over_prob","under_prob",
            "mu_home","mu_away","mean_total","sd_total","h_rest","a_rest"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context('display.max_columns', None, 'display.width', 160, 'display.precision', 4):
        print(df[cols].to_string(index=False))

    print("\n=== Bet Recommendations (ranked) ===")
    recs = build_recs(df)
    if recs.empty:
        print("No +EV edges ≥ min_ev at current prices.")
    else:
        with pd.option_context('display.max_columns', None, 'display.width', 160, 'display.precision', 4):
            print(recs.to_string(index=False))

if __name__ == "__main__":
    main()