# dk_parlays.py
# Pulls "most popular parlays" from DKNetwork and emits clean JSON.
# Usage:
#   python dk_parlays.py --top 15 --out dk_parlays.json
#   python dk_parlays.py --min-bet-count 100 --include MLB

import argparse
import json
import re
import sys
import time
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

try:
    import streamlit as st
    import pandas as pd
except Exception:
    st = None
    pd = None

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

DEFAULT_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-parlays/?cc_lg=88808&cc_sort=default"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# --- Player Props DKNetwork constants and helpers ---
PROPS_BASE = "https://dknetwork.draftkings.com/draftkings-sportsbook-player-props/"

def build_props_url(event_group: int = 88808, edate: str = "today") -> str:
    """
    Construct DKNetwork Player Props URL.
    tb_view=2 forces the props table view.
    tb_eg is the event group (84240 = MLB).
    tb_edate controls date scope: today | yesterday | last_7 | last_30
    """
    from urllib.parse import urlencode
    params = {"tb_view": "2", "tb_eg": str(event_group), "tb_edate": edate}
    return f"{PROPS_BASE}?{urlencode(params)}"

TEXT_ANCHORS_TYPE = {"Parlay", "SGPx", "Quick SGP", "Special"}
STOP_PREFIXES = (
    "Ends in:", "Stake:", "+ Add to Bet Slip", "WATCH DKN", "×", "Selection Count:"
)

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def parse_int_in_text(s: str, default: int = 0) -> int:
    m = re.search(r"[-+]?\d[\d,]*", s or "")
    if not m:
        return default
    return int(m.group(0).replace(",", ""))

def fetch_html(url: str, timeout: int = 20, retries: int = 2, backoff: float = 0.75) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff * (2 ** i))
    raise SystemExit(f"[error] fetch failed: {last_err}")

# --- Player Props fetcher ---
def fetch_props_html(event_group: int = 84240, edate: str = "today", use_playwright: bool = False) -> str:
    """
    Fetch the DKNetwork Player Props page HTML.
    If `use_playwright` is True and Playwright is available, render the page to catch JS-hydrated tables.
    Otherwise uses the existing `fetch_html` (requests) path.
    """
    url = build_props_url(event_group, edate)
    if use_playwright and sync_playwright is not None:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=UA)
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                try:
                    page.wait_for_selector("table, .tb-props, .tb-row, .tb-table", timeout=8000)
                except Exception:
                    try:
                        page.wait_for_load_state("networkidle", timeout=8000)
                    except Exception:
                        pass
                html = page.content()
                browser.close()
                if html and len(html) > 4096:
                    return html
        except Exception:
            pass
    return fetch_html(url)

def sniff_parlay_type(h3) -> Optional[str]:
    # Try previous siblings first (type tag often appears right above the title)
    prev = h3.previous_sibling
    for _ in range(6):
        if not prev:
            break
        txt = clean(getattr(prev, "get_text", lambda: "")())
        if txt in TEXT_ANCHORS_TYPE:
            return txt
        prev = prev.previous_sibling
    # Try next siblings if not found
    nxt = h3.next_sibling
    for _ in range(6):
        if not nxt:
            break
        txt = clean(getattr(nxt, "get_text", lambda: "")())
        if txt in TEXT_ANCHORS_TYPE:
            return txt
        nxt = nxt.next_sibling
    return None

# --- Player Props parsing helpers ---
ODDS_TOKEN_RE = re.compile(r"^[+-]\d{2,4}$")

def _clean_text(node):
    return clean(getattr(node, "get_text", lambda *a, **k: "")(" "))

def _odds_to_int(val: str) -> Optional[int]:
    if not val:
        return None
    try:
        s = str(val).strip().replace("−", "-")
        m = re.search(r"[-+]\d{2,4}", s)
        return int(m.group(0)) if m else None
    except Exception:
        return None

def parse_cards(html: str, verbose: bool = False) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    cards: List[Dict[str, Any]] = []

    # Candidate titles: h3 headings in the parlay list
    for h3 in soup.find_all("h3"):
        title = clean(h3.get_text())
        if not title or len(title) > 140 or len(title) < 2:
            continue

        # We only treat it as a parlay card if we can find an "Odds:" line nearby
        odds_val: Optional[int] = None
        ptype: Optional[str] = sniff_parlay_type(h3)

        # Try to extract Bet Count via concrete DOM: .dkcc-card-betcount > span
        # Walk up a few parents to find the container that holds dkcc-card-betcount.
        bet_count_val = 0
        ancestor = h3
        for _ in range(6):
            if not ancestor or not hasattr(ancestor, "select_one"):
                break
            bc_span = ancestor.select_one(".dkcc-card-betcount span")
            if bc_span:
                bet_count_val = parse_int_in_text(bc_span.get_text(), 0)
                break
            ancestor = getattr(ancestor, "parent", None)

        legs: List[str] = []
        capturing_legs = False

        # Walk forward through siblings until the next h3
        node = h3
        steps = 0
        while node and steps < 200:
            steps += 1
            node = node.next_sibling
            if not node:
                break

            # Stop if we hit the next card
            if getattr(node, "name", None) == "h3":
                break

            txt = clean(getattr(node, "get_text", lambda: "")())
            if not txt:
                continue

            # anchor fields
            if txt.startswith("Odds:"):
                odds_val = parse_int_in_text(txt)
                capturing_legs = True
                if verbose:
                    print(f"[dbg] {title}: odds={odds_val}", file=sys.stderr)
                continue

            if txt.startswith("Bet Count:"):
                if bet_count_val == 0:
                    bet_count_val = parse_int_in_text(txt, 0)
                if verbose:
                    print(f"[dbg] {title}: bet_count={bet_count_val}", file=sys.stderr)
                continue

            # legs live between Odds and first stop prefix
            if capturing_legs:
                if any(txt.startswith(p) for p in STOP_PREFIXES):
                    capturing_legs = False
                else:
                    # filter out obvious echoes
                    if not (txt.startswith("Odds:") or txt.startswith("Bet Count:")):
                        legs.append(txt)

        if odds_val is not None:
            cards.append({
                "title": title,
                "parlay_type": ptype or "Parlay",
                "odds": odds_val,
                "bet_count": bet_count_val,
                "legs": legs
            })

    # Deduplicate by title+odds in case the page echoes
    seen = set()
    unique_cards = []
    for c in cards:
        key = (c["title"], c["odds"])
        if key in seen:
            continue
        seen.add(key)
        unique_cards.append(c)

    # Rank: bet_count desc, then odds desc
    unique_cards.sort(key=lambda c: (c.get("bet_count", 0), c.get("odds", 0)), reverse=True)
    return unique_cards

def filter_cards(cards: List[Dict[str, Any]], include: Optional[str], min_bet_count: int) -> List[Dict[str, Any]]:
    out = []
    for c in cards:
        if c.get("bet_count", 0) < min_bet_count:
            continue
        if include:
            hay = " ".join([c.get("title","")] + c.get("legs", [])).lower()
            if include.lower() not in hay:
                continue
        out.append(c)
    return out

def fetch_dk_splits(event_group: int = 88808, date_range: str = "today", market: str = "All") -> List[Dict[str, Any]]:
    """
    Fetch DKNetwork betting splits/handles for a given event group and date range.
    Returns a list of dicts with keys:
      matchup, game_time, market, side, odds, %handle, %bets, update_time
    Notes:
      - No pytz; update_time is epoch seconds.
      - Falls back to requests if Playwright unavailable.
    """
    from urllib.parse import urlencode

    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
    full_url = f"{base}?{urlencode(params)}"

    def _clean(text: str) -> str:
        return re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", text or "", flags=re.I).strip()

    def _clean_odds(odds_str: str) -> int:
        try:
            return int(odds_str.replace("−", "-"))
        except Exception:
            try:
                return int(re.sub(r"[^-+\d]", "", odds_str))
            except Exception:
                return 0

    html = ""
    # Try Playwright first if available
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

    # Fallback to requests
    if not html:
        try:
            resp = requests.get(full_url, headers={"User-Agent": UA}, timeout=20)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            return []

    soup = BeautifulSoup(html, "lxml")  # use lxml for consistency
    # DK has two known structures: older `.tb-se` blocks, newer `.tb-war` blocks
    games = soup.select("div.tb-se, .tb-war")
    # If "today" is empty, try "yesterday" automatically
    if not games and date_range == "today":
        return fetch_dk_splits(event_group, "yesterday", market)

    records: List[Dict[str, Any]] = []
    now_epoch = int(time.time())

    for game in games:
        # Titles
        title_node = game.select_one("div.tb-se-title h5, .tb-hwar h3, h3")
        title = _clean(title_node.get_text(strip=True)) if title_node else ""
        time_node = game.select_one("div.tb-se-title span, .tb-gmTime, time")
        game_time = _clean(time_node.get_text(strip=True)) if time_node else ""

        # Market sections
        sections = game.select(".tb-market-wrap > div, .tb-war-mkt, .tb-mkt-wrap")
        for section in sections:
            mkt_node = section.select_one(".tb-se-head > div, .tb-mkt-headline, .tb-mkth, h4")
            market_name = _clean(mkt_node.get_text(strip=True)) if mkt_node else ""
            if market_name not in ("Moneyline", "Total", "Totals", "Spread"):
                continue

            # Rows
            for row in section.select(".tb-sodd, .tb-sodd-out, .tb-odd-row, tr"):
                side_node = row.select_one(".tb-slipline, .tb-line, .tb-side, td:first-child")
                side_raw = _clean(side_node.get_text(strip=True)) if side_node else ""

                odds_node = row.select_one("a.tb-odd-s, .tb-odd-s, a[aria-label*='odds'], .tb-odds, td a")
                raw_odds = _clean(odds_node.get_text(strip=True)) if odds_node else ""
                odds = _clean_odds(raw_odds)

                # Percentages: capture first two numbers with % across nested spans
                pct_texts = []
                for el in row.select(".tb-pct, .tb-sh, .tb-sb, .tb-pct-h, .tb-pct-b, span, td, div"):
                    txt = el.get_text(" ", strip=True)
                    if "%" in txt:
                        pct_texts.append(txt)
                if not pct_texts:
                    pct_texts = [t.strip() for t in row.find_all(string=lambda t: isinstance(t, str) and "%" in t)]
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

                # Skip if row is empty
                if not side_raw and not handle_pct and not bets_pct:
                    continue

                records.append({
                    "matchup": title,
                    "game_time": game_time,
                    "market": market_name,
                    "side": side_raw,
                    "odds": odds,
                    "%handle": float(handle_pct or 0),
                    "%bets": float(bets_pct or 0),
                    "update_time": now_epoch,
                })

    return records

def compute_hhi(series):
    s = [max(0.0, min(1.0, float(x))) for x in series if x is not None]
    return sum(v*v for v in s) if s else 0.0

def compute_parlay_surge_from_handles(fav_handles, over_handles, corr_thresholds=(0.70, 0.65)):
    # fav_handles / over_handles are lists of floats in [0,1]
    hhi_sides = compute_hhi(fav_handles)
    hhi_totals = compute_hhi(over_handles)
    n_corr = sum(1 for f,o in zip(fav_handles, over_handles) if f >= corr_thresholds[0] and o >= corr_thresholds[1])
    # bucket score
    score = 0
    if (hhi_sides >= 0.20) or (hhi_totals >= 0.20) or (n_corr >= 1):
        score = 1
    if (hhi_sides >= 0.22 and hhi_totals >= 0.22) or (n_corr >= 3):
        score = 2
    if (hhi_sides >= 0.25 and hhi_totals >= 0.25) or (n_corr >= 5):
        score = 3
    return {
        "parlay_surge_score": score,
        "hhi_sides": hhi_sides,
        "hhi_totals": hhi_totals,
        "n_corr": n_corr
    }

def _is_moneyline(name: str) -> bool:
    n = (name or "").lower()
    return "moneyline" in n or n.endswith(" ml") or n == "ml"

def _is_total(name: str) -> bool:
    n = (name or "").lower()
    return "total" in n or "o/u" in n or "over/under" in n

# ---- NFL team normalization ----
TEAM_ALIASES = {
    # NFC East
    "dal":"DAL","cowboys":"DAL","dallas":"DAL",
    "phi":"PHI","eagles":"PHI","philadelphia":"PHI",
    "nyg":"NYG","giants":"NYG","new york giants":"NYG",
    "was":"WAS","wsh":"WAS","commanders":"WAS","washington":"WAS",
    # NFC North
    "gb":"GB","packers":"GB","green bay":"GB",
    "min":"MIN","vikings":"MIN","minnesota":"MIN",
    "det":"DET","lions":"DET","detroit":"DET",
    "chi":"CHI","bears":"CHI","chicago":"CHI",
    # NFC South
    "tb":"TB","buccaneers":"TB","bucs":"TB","tampa bay":"TB",
    "atl":"ATL","falcons":"ATL","atlanta":"ATL",
    "car":"CAR","panthers":"CAR","carolina":"CAR",
    "no":"NO","nor":"NO","saints":"NO","new orleans":"NO",
    # NFC West
    "sf":"SF","49ers":"SF","niners":"SF","san francisco":"SF",
    "sea":"SEA","seahawks":"SEA","seattle":"SEA",
    "ari":"ARI","cardinals":"ARI","arizona":"ARI",
    "lar":"LAR","la rams":"LAR","rams":"LAR",
    # AFC East
    "ne":"NE","patriots":"NE","new england":"NE",
    "nyj":"NYJ","jets":"NYJ","new york jets":"NYJ",
    "mia":"MIA","dolphins":"MIA","miami":"MIA",
    "buf":"BUF","bills":"BUF","buffalo":"BUF",
    # AFC North
    "pit":"PIT","steelers":"PIT","pittsburgh":"PIT",
    "bal":"BAL","ravens":"BAL","baltimore":"BAL",
    "cle":"CLE","browns":"CLE","cleveland":"CLE",
    "cin":"CIN","bengals":"CIN","cincinnati":"CIN",
    # AFC South
    "jax":"JAX","jac":"JAX","jaguars":"JAX","jacksonville":"JAX",
    "ind":"IND","colts":"IND","indianapolis":"IND",
    "hou":"HOU","texans":"HOU","houston":"HOU",
    "ten":"TEN","titans":"TEN","tennessee":"TEN",
    # AFC West
    "kc":"KC","chiefs":"KC","kansas city":"KC",
    "lv":"LV","rai":"LV","raiders":"LV","las vegas":"LV",
    "lac":"LAC","chargers":"LAC","los angeles chargers":"LAC",
    "den":"DEN","broncos":"DEN","denver":"DEN",
}

ABBR_TO_NAME = {
    "DAL":"Cowboys","PHI":"Eagles","NYG":"Giants","WAS":"Commanders",
    "GB":"Packers","MIN":"Vikings","DET":"Lions","CHI":"Bears",
    "TB":"Buccaneers","ATL":"Falcons","CAR":"Panthers","NO":"Saints",
    "SF":"49ers","SEA":"Seahawks","ARI":"Cardinals","LAR":"Rams",
    "NE":"Patriots","NYJ":"Jets","MIA":"Dolphins","BUF":"Bills",
    "PIT":"Steelers","BAL":"Ravens","CLE":"Browns","CIN":"Bengals",
    "JAX":"Jaguars","IND":"Colts","HOU":"Texans","TEN":"Titans",
    "KC":"Chiefs","LV":"Raiders","LAC":"Chargers","DEN":"Broncos",
}

def _norm_team_to_abbr(s: str) -> Optional[str]:
    if not s: return None
    k = re.sub(r"[^a-z0-9 ]+","", s.lower()).strip()
    return TEAM_ALIASES.get(k) or TEAM_ALIASES.get(k.replace(" ", ""))

def derive_handles_from_splits(splits: List[Dict[str, Any]]):
    """Given list of split rows, compute two series:
    - fav_handles: per-game favorite %handle from a Moneyline market
    - over_handles: per-game Over %handle from a Totals market
    Returns (fav_handles, over_handles) as floats in [0,1].
    """
    fav_handles: List[float] = []
    over_handles: List[float] = []

    # group by matchup
    from collections import defaultdict
    by_match = defaultdict(list)
    for r in splits:
        by_match[r.get("matchup","")].append(r)

    for matchup, rows in by_match.items():
        # Favorite handle from Moneyline
        ml_rows = [r for r in rows if _is_moneyline(r.get("market",""))]
        fav_handle_pct = None
        if ml_rows:
            # choose side with most negative odds; fallback to largest handle if odds missing
            ml_rows_filtered = [r for r in ml_rows if r.get("odds") is not None]
            chosen = None
            if ml_rows_filtered:
                chosen = min(ml_rows_filtered, key=lambda r: r.get("odds", 10**9))
            else:
                chosen = max(ml_rows, key=lambda r: r.get("%handle", 0.0))
            fav_handle_pct = float(chosen.get("%handle", 0.0))
        # Over handle from Totals
        tot_rows = [r for r in rows if _is_total(r.get("market",""))]
        over_row = None
        for r in tot_rows:
            if str(r.get("side","")).lower().startswith("over"):
                over_row = r
                break
        over_handle_pct = float(over_row.get("%handle", 0.0)) if over_row else None

        if fav_handle_pct is not None:
            fav_handles.append(max(0.0, min(1.0, fav_handle_pct/100.0 if fav_handle_pct > 1 else fav_handle_pct)))
        if over_handle_pct is not None:
            over_handles.append(max(0.0, min(1.0, over_handle_pct/100.0 if over_handle_pct > 1 else over_handle_pct)))

    return fav_handles, over_handles

# ---- Native Refs & Pace overlay (self-contained) ----
from typing import Tuple

def _parse_matchup_teams(matchup: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a DK matchup string like 'Cowboys vs Eagles' or 'Bills at Jets' into (away, home) best-effort."""
    if not matchup:
        return None, None
    s = str(matchup)
    for sep in [" vs ", " at ", " @ ", " v ", " VS ", " Vs ", " AT "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep, 1)]
            if len(parts) == 2:
                return parts[0], parts[1]
    if "|" in s:
        parts = [p.strip() for p in s.split("|", 1)]
        if len(parts) == 2:
            return parts[0], parts[1]
    return None, None

def _z_norm(val: Any, mean: float, std: float) -> float:
    try:
        x = float(val)
    except Exception:
        return 0.0
    if std <= 1e-9:
        return 0.0
    return (x - mean) / std

def _compute_df_stats(df: "pd.DataFrame", cols: List[str]) -> Dict[str, Tuple[float,float]]:
    stats = {}
    for c in cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            stats[c] = (float(s.mean(skipna=True)), float(s.std(skipna=True, ddof=0)))
        except Exception:
            stats[c] = (0.0, 1.0)
    return stats

def enrich_splits_with_ref_pace_native(
    splits: List[Dict[str, Any]],
    ref_crews_df: "pd.DataFrame" = None,
    team_pace_df: "pd.DataFrame" = None,
    crew_assign_df: "pd.DataFrame" = None
) -> List[Dict[str, Any]]:
    """
    Outputs per row: crew_id, ref_name, vi, ug, dei
    VI  = z(-secs_per_play) + 0.7*z(explosive_pass_rate) + 0.5*z(over_rate_adj)
    UG  = z(+secs_per_play) - 0.6*z(over_rate_adj)     [higher = more Under gravity]
    DEI = z(dpi_rate) + 0.5*z(def_hold_rate) - 0.3*z(off_hold_rate)
    """
    if not splits:
        return splits
    rc = ref_crews_df if ref_crews_df is not None else pd.DataFrame()
    tp = team_pace_df if team_pace_df is not None else pd.DataFrame()
    ca = crew_assign_df if crew_assign_df is not None else pd.DataFrame()
    # refs columns
    if not rc.empty:
        rc = rc.rename(columns={
            "over_rate":"over_rate_adj",
            "over_bias":"over_rate_adj",
            "defensive_hold_rate":"def_hold_rate",
            "offensive_hold_rate":"off_hold_rate",
            "dpi":"dpi_rate",
            "crew":"crew_id",
            "name":"ref_name"
        })
        for col in ["over_rate_adj","dpi_rate","off_hold_rate","def_hold_rate"]:
            if col not in rc.columns:
                rc[col] = 0.0
    # team pace columns
    if not tp.empty:
        if "team" not in tp.columns and "abbr" in tp.columns:
            tp = tp.rename(columns={"abbr":"team"})
        for col in ["secs_per_play","explosive_pass_rate"]:
            if col not in tp.columns:
                tp[col] = 0.0
    # z-stats
    pace_cols = ["secs_per_play","explosive_pass_rate"]
    ref_cols  = ["over_rate_adj","dpi_rate","off_hold_rate","def_hold_rate"]
    pstats = _compute_df_stats(tp, pace_cols) if not tp.empty else {c:(0.0,1.0) for c in pace_cols}
    rstats = _compute_df_stats(rc, ref_cols)  if not rc.empty else {c:(0.0,1.0) for c in ref_cols}
    # indices
    assign_idx = {}
    if not ca.empty:
        for _, row in ca.iterrows():
            key = re.sub(r"[^a-z0-9]+", "", str(row.get("matchup",""))).lower()
            if key:
                assign_idx[key] = {"crew_id": row.get("crew_id"), "ref_name": row.get("ref_name")}
    tp_idx = {}
    if not tp.empty:
        for _, row in tp.iterrows():
            nm = str(row.get("team",""))
            if not nm: continue
            ab = _norm_team_to_abbr(nm) or nm
            tp_idx[str(ab).lower()] = row
            tp_idx[str(nm).lower()] = row
    rc_idx = {str(r.get("crew_id","")).strip().lower(): r for _, r in (rc if not rc.empty else pd.DataFrame()).iterrows()} if not rc.empty else {}
    # compute
    enriched = []
    for r in splits:
        row = dict(r)
        matchup = str(row.get("matchup",""))
        a, b = _parse_matchup_teams(matchup)
        a_abbr = _norm_team_to_abbr(a)
        b_abbr = _norm_team_to_abbr(b)
        sp_z = ep_z = 0.0
        if a_abbr and b_abbr and tp_idx:
            ta = tp_idx.get(a_abbr.lower()) or tp_idx.get(ABBR_TO_NAME.get(a_abbr, "").lower())
            tb = tp_idx.get(b_abbr.lower()) or tp_idx.get(ABBR_TO_NAME.get(b_abbr, "").lower())
            if ta is not None and tb is not None:
                sp = (pd.to_numeric(ta.get("secs_per_play"), errors="coerce") + pd.to_numeric(tb.get("secs_per_play"), errors="coerce")) / 2.0
                ep = (pd.to_numeric(ta.get("explosive_pass_rate"), errors="coerce") + pd.to_numeric(tb.get("explosive_pass_rate"), errors="coerce")) / 2.0
                sp_z = -_z_norm(sp, *pstats["secs_per_play"])            # faster pace => more variance
                ep_z =  _z_norm(ep, *pstats["explosive_pass_rate"])
        crew_id = None; ref_name = None
        key = re.sub(r"[^a-z0-9]+", "", matchup.lower())
        if assign_idx:
            meta = assign_idx.get(key)
            if meta:
                crew_id = meta.get("crew_id"); ref_name = meta.get("ref_name")
        over_z = dpi_z = off_h_z = def_h_z = 0.0
        if crew_id and rc_idx:
            rr = rc_idx.get(str(crew_id).lower())
            if rr is not None:
                over_z  = _z_norm(pd.to_numeric(rr.get("over_rate_adj"), errors="coerce"), *rstats["over_rate_adj"])
                dpi_z   = _z_norm(pd.to_numeric(rr.get("dpi_rate"), errors="coerce"), *rstats["dpi_rate"])
                off_h_z = _z_norm(pd.to_numeric(rr.get("off_hold_rate"), errors="coerce"), *rstats["off_hold_rate"])
                def_h_z = _z_norm(pd.to_numeric(rr.get("def_hold_rate"), errors="coerce"), *rstats["def_hold_rate"])
        vi  = (sp_z) + 0.7*(ep_z) + 0.5*(over_z)
        ug  = (-sp_z) - 0.6*(over_z)
        dei = (dpi_z) + 0.5*(def_h_z) - 0.3*(off_h_z)
        row.update({"crew_id": crew_id, "ref_name": ref_name, "vi": float(vi), "ug": float(ug), "dei": float(dei)})
        enriched.append(row)
    return enriched

def summarize_vi_ug(enriched_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    import math
    if not enriched_rows:
        return {"vi_mean": 0.0, "vi_med": 0.0, "ug_mean": 0.0, "ug_med": 0.0}
    def _med(xs):
        if not xs: return 0.0
        xs = sorted(xs); n = len(xs)
        return xs[n//2] if n % 2 == 1 else 0.5*(xs[n//2 - 1] + xs[n//2])
    vi_vals = [float(r.get("vi")) for r in enriched_rows if r.get("vi") is not None]
    ug_vals = [float(r.get("ug")) for r in enriched_rows if r.get("ug") is not None]
    return {
        "vi_mean": sum(vi_vals)/len(vi_vals) if vi_vals else 0.0,
        "vi_med": _med(vi_vals),
        "ug_mean": sum(ug_vals)/len(ug_vals) if ug_vals else 0.0,
        "ug_med": _med(ug_vals),
    }

# ---- NFL Referee analytics (library-driven, no CSVs) ----
try:
    import nfl_data_py as _nfl
except Exception:
    _nfl = None

MIN_GAMES_THRESHOLD = 48  # ~3 weeks of REG season games

def _pick_ref_data_year() -> int:
    import datetime as _dt
    yr = _dt.datetime.now().year
    if _nfl is None:
        return yr
    try:
        pbp_cur = _nfl.import_pbp_data([yr])
        reg_games = int(pbp_cur[pbp_cur.get('season_type') == 'REG']['game_id'].nunique()) if not pbp_cur.empty else 0
        if reg_games >= MIN_GAMES_THRESHOLD:
            return yr
    except Exception:
        pass
    return yr - 1

def load_referee_metrics(year: int | None = None) -> "pd.DataFrame":
    """Build per-ref metrics from nfl_data_py. Columns:
    name, games, penalties_pg, overall_diff_pct, over_bias_pts, dpi_pg, off_hold_pg, def_hold_pg, home_penalty_share
    """
    if _nfl is None or pd is None:
        return pd.DataFrame()
    y = int(year) if year is not None else _pick_ref_data_year()
    try:
        officials = _nfl.import_officials([y])
        pbp = _nfl.import_pbp_data([y])
        sched = _nfl.import_schedules([y])
    except Exception:
        return pd.DataFrame()
    if officials is None or pbp is None or sched is None or officials.empty or pbp.empty or sched.empty:
        return pd.DataFrame()

    # REG season only
    reg_ids = set(pbp.loc[pbp.get('season_type') == 'REG', 'game_id'].unique())
    officials = officials[officials['game_id'].isin(reg_ids)]
    pbp = pbp[pbp['game_id'].isin(reg_ids)]
    sched = sched[sched.get('game_type') == 'REG']

    # Referees only
    refs = officials[officials.get('off_pos') == 'R'][['game_id','name']].dropna()

    # Penalties dataset
    pen = pbp[pbp.get('penalty') == 1][['game_id','play_id','penalty_type','penalty_team','home_team','away_team']].copy()
    if pen.empty or refs.empty:
        return pd.DataFrame()

    # Home/Away tag
    def _who(row):
        if row['penalty_team'] == row['home_team']:
            return 'home'
        if row['penalty_team'] == row['away_team']:
            return 'away'
        return None
    pen['pen_against'] = pen.apply(_who, axis=1)

    # Join ref to penalties
    rp = pen.merge(refs, on='game_id', how='left')

    # League averages
    total_games = len(reg_ids) if reg_ids else 0
    lg_pen_per_game = len(pen) / total_games if total_games > 0 else 0.0

    # Per-type league avg per game
    lg_type_pg = pen.groupby('penalty_type')['play_id'].count().div(total_games).to_dict()

    # Over bias: actual total minus closing total_line
    tot_df = sched[['game_id','total','total_line']].dropna()
    tot_df['tot_diff'] = pd.to_numeric(tot_df['total'], errors='coerce') - pd.to_numeric(tot_df['total_line'], errors='coerce')
    ref_games = refs.groupby('name')['game_id'].nunique().rename('games')

    # Penalties per game per ref
    ref_pen = rp.groupby('name')['play_id'].count().rename('penalties')
    ref_ppg = (ref_pen / ref_games).rename('penalties_pg')

    # Type-specific per game
    type_counts = rp.groupby(['name','penalty_type'])['play_id'].count().rename('cnt').reset_index()
    def _pg(df, type_name):
        s = df[df['penalty_type'] == type_name].set_index('name')['cnt'] if not df.empty else pd.Series(dtype=float)
        return (s / ref_games).reindex(ref_games.index).fillna(0.0)
    dpi_pg = _pg(type_counts, 'Defensive Pass Interference').rename('dpi_pg')
    off_hold_pg = _pg(type_counts, 'Offensive Holding').rename('off_hold_pg')
    def_hold_pg = _pg(type_counts, 'Defensive Holding').rename('def_hold_pg')

    # Home bias share
    ha = rp.groupby(['name','pen_against'])['play_id'].count().unstack(fill_value=0)
    home_share = (ha.get('home', 0) / (ha.get('home', 0) + ha.get('away', 0)).replace(0, pd.NA)).fillna(0.0).rename('home_penalty_share')

    # Over bias per ref: mean tot_diff across their games
    ref_tot_diff = refs.merge(tot_df[['game_id','tot_diff']], on='game_id', how='left').groupby('name')['tot_diff'].mean().rename('over_bias_pts')

    df = pd.concat([ref_games, ref_ppg, dpi_pg, off_hold_pg, def_hold_pg, home_share, ref_tot_diff], axis=1).reset_index()
    df['overall_diff_pct'] = (df['penalties_pg'] - lg_pen_per_game) / (lg_pen_per_game if lg_pen_per_game else 1.0) * 100.0
    df = df.rename(columns={'name':'ref_name'})
    df = df.sort_values(['games','penalties_pg'], ascending=[False, False])
    return df

def build_ref_report_cards(df: "pd.DataFrame") -> list[str]:
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        ref = str(r.get('ref_name'))
        games = int(r.get('games') or 0)
        ppg = float(r.get('penalties_pg') or 0)
        diff = float(r.get('overall_diff_pct') or 0)
        over_bias = float(r.get('over_bias_pts') or 0)
        dpi = float(r.get('dpi_pg') or 0)
        offh = float(r.get('off_hold_pg') or 0)
        defh = float(r.get('def_hold_pg') or 0)
        home_share = float(r.get('home_penalty_share') or 0)
        favor = 'home' if home_share > 0.55 else ('away' if home_share < 0.45 else 'neutral')
        card = (
            f"# {ref}\n"
            f"Games: {games}\n"
            f"Penalties/Game: {ppg:.2f} ({diff:+.1f}% vs lg avg)\n"
            f"Over Bias (pts vs close): {over_bias:+.2f}\n"
            f"DPI/Game: {dpi:.2f} | Off Hold/Game: {offh:.2f} | Def Hold/Game: {defh:.2f}\n"
            f"Home-penalty share: {home_share:.1%} ({favor})\n"
        )
        out.append(card)
    return out

# ---- Internal-only VI/UG enrichment (no CSVs) ----
# Uses only DK splits data to produce soft variance/under-gravity overlays.
# Heuristics:
#  - Higher game totals -> higher VI (more scoring variance)
#  - Strong Under handle vs bets -> higher UG
#  - Spread magnitude adds variance (blowouts introduce garbage-time swings)

def _extract_game_total(rows_for_game: list[dict]) -> float | None:
    # Prefer the Over row's side text to parse a numeric total
    for r in rows_for_game:
        if _is_total(str(r.get("market",""))):
            if str(r.get("side",""))[:4].lower() == "over":
                tv = _parse_total_value(r.get("side",""))
                if tv is not None:
                    return float(tv)
    # fallback: any total row
    for r in rows_for_game:
        if _is_total(str(r.get("market",""))):
            tv = _parse_total_value(r.get("side",""))
            if tv is not None:
                return float(tv)
    return None


def enrich_splits_with_internal_heuristics(splits: list[dict]) -> list[dict]:
    if not splits:
        return []
    from collections import defaultdict
    import math
    by_match = defaultdict(list)
    for r in splits:
        by_match[str(r.get("matchup",""))].append(r)

    # Collect totals and spread abs for global stats
    totals, spreads = [], []
    for rows in by_match.values():
        gt = _extract_game_total(rows)
        if gt is not None:
            totals.append(gt)
        for r in rows:
            if "spread" in str(r.get("market","")).lower():
                sv = _parse_spread_value(r.get("side",""))
                if sv is not None:
                    spreads.append(abs(float(sv)))
    # fallback means/stds
    def _mean_std(xs: list[float]) -> tuple[float,float]:
        if not xs:
            return (44.0, 7.5)  # generic NFL priors
        m = sum(xs)/len(xs)
        v = sum((x-m)*(x-m) for x in xs)/max(1,len(xs)-1)
        return (m, math.sqrt(max(1e-9, v)))
    t_mu, t_sd = _mean_std(totals)
    s_mu, s_sd = _mean_std(spreads)

    out = []
    for matchup, rows in by_match.items():
        gt = _extract_game_total(rows)
        # compute handle vs bets delta on Under
        uh, ub = 0.0, 0.0
        for r in rows:
            if _is_total(str(r.get("market",""))) and str(r.get("side",""))[:5].lower() == "under":
                try:
                    uh = max(uh, float(r.get("%handle", r.get("handle_pct", 0.0)) or 0.0))
                    ub = max(ub, float(r.get("%bets", r.get("bets_pct", 0.0)) or 0.0))
                except Exception:
                    pass
        deltab = uh - ub
        # variance index: z(total) plus small z(abs spread)
        # parse one spread value for magnitude
        sp_abs = None
        for r in rows:
            if "spread" in str(r.get("market","")).lower():
                sv = _parse_spread_value(r.get("side",""))
                if sv is not None:
                    sp_abs = abs(float(sv)); break
        vi = 0.0
        if gt is not None and t_sd > 1e-9:
            vi += (gt - t_mu) / t_sd
        if sp_abs is not None and s_sd > 1e-9:
            vi += 0.35 * ((sp_abs - s_mu) / s_sd)
        # under gravity: inverse of total plus respect for Under handle>bets
        ug = 0.0
        if gt is not None and t_sd > 1e-9:
            ug += (t_mu - gt) / t_sd
        ug += 0.03 * deltab  # 3% per 1-pt handle minus bets
        # write one record per total Under row so it aligns with totals table
        wrote = False
        for r in rows:
            if _is_total(str(r.get("market",""))) and str(r.get("side",""))[:5].lower() in ("under","over"):
                rec = dict(r)
                rec.update({"vi": float(vi), "ug": float(ug), "dei": 0.0})
                out.append(rec); wrote = True
        if not wrote:
            # emit one generic row for the matchup
            out.append({"matchup": matchup, "vi": float(vi), "ug": float(ug), "dei": 0.0})
    return out

# ---- Weather & Status adjustments ----
WEATHER_DEFAULTS = {"wind_coeff_ug": 0.08, "wind_coeff_vi": -0.05}
STATUS_DEFAULTS = {"qb_out_penalty": 0.6, "ol_penalty_per_starter": 0.12}

def apply_weather_status_adjustments(enriched_rows: List[Dict[str, Any]], weather_df: "pd.DataFrame | None", status_df: "pd.DataFrame | None") -> List[Dict[str, Any]]:
    if not enriched_rows:
        return enriched_rows
    wdx, sdx = {}, {}
    if weather_df is not None and not weather_df.empty:
        for _, r in weather_df.iterrows():
            k = _norm_key(str(r.get("matchup","")))
            if k: wdx[k] = r
    if status_df is not None and not status_df.empty:
        for _, r in status_df.iterrows():
            ab = _norm_team_to_abbr(str(r.get("team","")))
            if ab: sdx[str(ab).lower()] = r
    out = []
    for r in enriched_rows:
        row = dict(r)
        key = _norm_key(str(row.get("matchup","")))
        vi = float(row.get("vi", 0.0) or 0.0)
        ug = float(row.get("ug", 0.0) or 0.0)
        if key in wdx:
            w = wdx[key]
            try:
                wind = float(w.get("wind_mph", 0) or 0)
                roof = str(w.get("roof",""))
                roof = roof.lower() if isinstance(roof, str) else ""
                if "closed" in roof or "dome" in roof: wind = 0.0
                ug += WEATHER_DEFAULTS["wind_coeff_ug"] * (wind / 10.0)
                vi += WEATHER_DEFAULTS["wind_coeff_vi"] * (wind / 10.0)
            except Exception:
                pass
        a, b = _parse_matchup_teams(str(row.get("matchup","")))
        for t in (a, b):
            ab = _norm_team_to_abbr(t)
            sd = sdx.get(str(ab).lower()) if ab else None
            if sd is None:
                continue
            try:
                qb_status = str(sd.get("qb_status",""))
                qb_out = 1 if qb_status.lower() in ("out","doubtful") else 0
                ol_out = int(sd.get("ol_starters_out", 0) or 0)
                ug += STATUS_DEFAULTS["qb_out_penalty"] * qb_out
                ug += STATUS_DEFAULTS["ol_penalty_per_starter"] * max(0, ol_out)
            except Exception:
                pass
        row.update({"vi": float(vi), "ug": float(ug)})
        out.append(row)
    return out

def maybe_enrich_with_ref_pace(splits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use internal heuristics overlay only (no CSVs). Falls back to raw splits."""
    if not splits:
        return splits
    try:
        enriched = enrich_splits_with_internal_heuristics(splits)
        return enriched
    except Exception:
        return splits

# ---------- Overlay CSV merger (fallback if nfl_refs is absent) ----------
def _norm_key(s: str) -> str:
    """Normalization for matchup keys during joins."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def enrich_with_overlay_csv(
    splits: List[Dict[str, Any]],
    overlay_rows: List[Dict[str, Any]],
    matchup_col: str = "matchup",
    fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Merge precomputed overlay stats (refs/pace composites) into splits by matchup.
    Expected overlay columns (any subset): matchup, crew_id, ref_name, vi, ug, dei
    Returns a new list of dicts containing original split fields plus overlay fields.
    """
    if not splits or not overlay_rows:
        return splits
    fields = fields or ["crew_id","ref_name","vi","ug","dei"]
    idx = {}
    for r in overlay_rows:
        key = _norm_key(str(r.get(matchup_col, "")))
        if not key:
            continue
        idx[key] = {k: r.get(k) for k in fields}
    out = []
    for r in splits:
        k = _norm_key(str(r.get("matchup", "")))
        merged = dict(r)
        if k in idx:
            merged.update(idx[k])
        out.append(merged)
    return out

def _df_like_to_rows(obj) -> List[Dict[str, Any]]:
    """Coerce DataFrame or CSV text into list-of-dicts rows."""
    try:
        import pandas as _pd
        if obj is None:
            return []
        if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict(orient="records")
        if isinstance(obj, str):
            from io import StringIO
            df = _pd.read_csv(StringIO(obj))
            return df.to_dict(orient="records")
    except Exception:
        pass
    return []

# ---- NFL Teaser Eligibility helpers (from splits) ----
SPREAD_NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)")

def _parse_spread_value(side_text: str) -> Optional[float]:
    """
    Extract a signed spread value like -7.5 or +2.5 from a DK 'Spread' side string.
    Returns float with sign if found, else None.
    """
    if not side_text:
        return None
    s = str(side_text).strip().replace("−", "-")
    # Prefer a signed number; if only unsigned is present, try to infer from presence of '+'/'-'
    m = re.search(r"[+-]\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    # fallback: unsigned number with implied sign from tokens
    m2 = SPREAD_NUM_RE.search(s)
    if m2:
        try:
            val = float(m2.group(1))
            # Heuristic: if " + " appears or a '+' token exists, treat as +val; if '-' present, treat as -val
            if "+" in s and "-" not in s:
                return +val
            if "-" in s and "+" not in s:
                return -val
            # Unknown sign: cannot infer robustly
            return None
        except Exception:
            return None
    return None

TOTAL_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")

def _parse_total_value(side_text: str) -> Optional[float]:
    """
    Extract total points number from 'Over/Under ##.#' side string.
    Returns float if found, else None.
    """
    if not side_text:
        return None
    s = str(side_text)
    m = TOTAL_NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def compute_teaser_candidates_from_splits(
    splits: List[Dict[str, Any]],
    fav_range: tuple = (-8.5, -7.5),
    dog_range: tuple = (1.5, 2.5),
    max_total: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Identify classic Wong-teaser style spots using only DK splits rows.
    - fav_range: favorites whose spread is between -8.5 and -7.5 (inclusive)
    - dog_range: underdogs between +1.5 and +2.5 (inclusive)
    - max_total: optional cap on the game total (e.g., 49.5). If provided, exclude games above this.
    Returns a list of dicts summarizing eligible legs.
    """
    if not splits:
        return []
    from collections import defaultdict
    by_match = defaultdict(list)
    for r in splits:
        by_match[r.get("matchup","")].append(r)

    out = []
    for matchup, rows in by_match.items():
        if not matchup:
            continue
        # Pull one representative total line for filtering (prefer the Over row)
        total_line = None
        for r in rows:
            if _is_total(str(r.get("market",""))):
                if str(r.get("side","")).lower().startswith("over"):
                    tv = _parse_total_value(r.get("side",""))
                    if tv is not None:
                        total_line = tv
                        break
        if max_total is not None and total_line is not None and total_line > max_total:
            continue

        # Scan spreads, collect candidate rows
        candidates = []
        for r in rows:
            if str(r.get("market","")).lower().find("spread") == -1:
                continue
            sv = _parse_spread_value(r.get("side",""))
            if sv is None:
                continue
            # capture handle/bets for color
            handle = float(r.get("%handle", r.get("handle_pct", 0.0)) or 0.0)
            bets = float(r.get("%bets", r.get("bets_pct", 0.0)) or 0.0)
            record = {
                "matchup": matchup,
                "side": r.get("side",""),
                "spread": sv,
                "odds": r.get("odds"),
                "handle_pct": handle,
                "bets_pct": bets,
                "game_total": total_line
            }
            # classify eligibility
            if fav_range[0] <= sv <= fav_range[1]:
                record["teaser_leg_type"] = "Favorite down"
                candidates.append(record)
            elif dog_range[0] <= sv <= dog_range[1]:
                record["teaser_leg_type"] = "Underdog up"
                candidates.append(record)

        # Deduplicate by spread bucket (pick the row with highest handle within each leg type)
        if candidates:
            from itertools import groupby
            candidates.sort(key=lambda d: (d["teaser_leg_type"], abs(d["spread"]), -float(d.get("handle_pct",0.0))))
            for lt, grp in groupby(candidates, key=lambda d: d["teaser_leg_type"]):
                best = max(list(grp), key=lambda d: (float(d.get("handle_pct",0.0)), -abs(d["spread"])))
                out.append(best)
    # Sort for readability
    out.sort(key=lambda d: (d["teaser_leg_type"], d["matchup"]))
    return out


# ---- NFL Teaser Probability & EV helpers ----
from math import erf, sqrt

def approx_cover_prob(spread: float, teaser_shift: float = 6.0, sigma: float = 13.0) -> float:
    """
    Approximate cover probability after applying a teaser shift to an NFL spread.
    Uses a normal model with sigma ≈ 13 points. Returns a value in [0,1].
    Positive spreads are dogs; negative spreads are favorites.
    """
    try:
        s = float(spread)
    except Exception:
        s = 0.0
    z = (-(s - teaser_shift)) / (sigma * sqrt(2))
    p = 0.5 * (1.0 + erf(z))
    return max(0.0, min(1.0, p))

def teaser_ev_two_leg(p1: float, p2: float, price: int = -120) -> float:
    """
    Expected value for a 2-leg teaser at a given book price (American odds).
    p1, p2 are per-leg win probabilities post-teaser. Stake = 1 unit.
    Returns EV in units.
    """
    try:
        p1 = float(p1); p2 = float(p2)
    except Exception:
        p1 = p2 = 0.5
    p = max(0.0, min(1.0, p1 * p2))
    payout = (100/(-price)) if price < 0 else (price/100)
    return p * payout - (1 - p)

def parse_player_props(html: str) -> List[Dict[str, Any]]:
    """
    Parse DKNetwork Player Props table into normalized rows.
    Returns list of dicts with keys:
      event, date, game_time, player, market, line, outcome, odds, team, matchup, update_time
    Handles both <table>-based and div-row layouts.
    """
    soup = BeautifulSoup(html, "lxml")
    out: List[Dict[str, Any]] = []
    now_epoch = int(time.time())

    # Pass 0: explicit DKNetwork props table under .pp-table-wrapper
    specific_rows = soup.select("div.pp-table-wrapper table tbody tr")
    if specific_rows:
        for tr in specific_rows:
            tds = tr.select("td")
            if len(tds) != 5:
                continue
            matchup = clean(tds[0].get_text(" ", strip=True))
            date_time = clean(tds[1].get_text(" ", strip=True))
            prop_text = clean(tds[2].get_text(" ", strip=True))
            line_text = clean(tds[3].get_text(" ", strip=True))
            # Odds live inside an <a>; capture visible text and href for traceability
            a = tds[4].select_one("a")
            odds_txt = clean(a.get_text(" ", strip=True) if a else tds[4].get_text(" ", strip=True))
            odds_val = _odds_to_int(odds_txt)
            if odds_val is None:
                continue
            href = a["href"] if a and a.has_attr("href") else None

            # Split "Player Market" text into (player, market)
            def split_player_market(s: str):
                markets = [
                    # MLB
                    "Hits + Runs + RBIs", "Strikeouts Thrown O/U", "Strikeouts Thrown",
                    "Home Runs", "Total Bases", "Stolen Bases O/U", "Stolen Bases",
                    "Walks Allowed", "RBIs", "Hits", "Runs",
                    # NFL (expanded)
                    "Anytime Touchdown Scorer", "Receiving Yards", "Receptions",
                    "Rushing Yards", "Passing Yards", "Passing Attempts", "Completions",
                    "Interceptions Thrown", "Rush Attempts", "Targets", "Passing TDs",
                    "Alt Receptions", "Alt Receiving Yards", "Alt Rushing Yards", "Alt Passing Yards",
                    "Longest Reception", "Longest Rush", "Longest Completion",
                    "Rush + Rec Yards", "Rushing + Receiving Yards"
                ]
                markets.sort(key=len, reverse=True)
                for mkt in markets:
                    if s.endswith(mkt):
                        player = s[: -len(mkt)].strip()
                        return player, mkt
                    if mkt in s:
                        parts = s.split(mkt, 1)
                        player = parts[0].strip()
                        return player, mkt
                toks = s.split()
                if len(toks) >= 3:
                    return " ".join(toks[:2]), " ".join(toks[2:])
                return s, None

            player, market = split_player_market(prop_text)

            out.append({
                "event": matchup or None,
                "date": date_time or None,
                "game_time": None,
                "player": player or None,
                "market": market or None,
                "line": line_text or None,
                "outcome": None,
                "odds": odds_val,
                "team": None,
                "matchup": matchup or None,
                "update_time": now_epoch,
                "dk_url": href
            })
        if out:
            return out

    # Try table-first layout
    for tbl in soup.select("table"):
        # Fast-path: headerless 5-col table (matchup, datetime, prop_text, line, odds-link)
        first_body_tr = tbl.select_one("tbody tr") or None
        if first_body_tr:
            tds = first_body_tr.select("td")
            if len(tds) == 5 and not tbl.select_one("thead"):
                # Parse every row as: [matchup, date_time, prop_text, line, odds]
                for tr in tbl.select("tbody tr"):
                    cells = [clean(td.get_text(" ", strip=True)) for td in tr.select("td")]
                    if len(cells) != 5:
                        continue
                    matchup = cells[0]
                    date_time = cells[1]
                    prop_text = cells[2]
                    line_text = cells[3]
                    # Odds are inside an <a>; grab visible odds token
                    odds_cell = tr.select_one("td:nth-child(5)")
                    odds_txt = clean(odds_cell.get_text(" ", strip=True)) if odds_cell else cells[4]
                    odds_val = _odds_to_int(odds_txt)
                    if odds_val is None:
                        continue

                    # Split prop_text into player and market using a keyword list
                    def split_player_market(s: str):
                        markets = [
                            # MLB
                            "Hits + Runs + RBIs", "Strikeouts Thrown O/U", "Strikeouts Thrown",
                            "Home Runs", "Total Bases", "Stolen Bases O/U", "Stolen Bases",
                            "Walks Allowed", "RBIs", "Hits", "Runs",
                            # NFL (expanded)
                            "Anytime Touchdown Scorer", "Receiving Yards", "Receptions",
                            "Rushing Yards", "Passing Yards", "Passing Attempts", "Completions",
                            "Interceptions Thrown", "Rush Attempts", "Targets", "Passing TDs",
                            "Alt Receptions", "Alt Receiving Yards", "Alt Rushing Yards", "Alt Passing Yards",
                            "Longest Reception", "Longest Rush", "Longest Completion",
                            "Rush + Rec Yards", "Rushing + Receiving Yards"
                        ]
                        # longest-first to avoid partial matches
                        markets.sort(key=len, reverse=True)
                        for mkt in markets:
                            if s.endswith(mkt):
                                player = s[: -len(mkt)].strip()
                                return player, mkt
                            # handle 'Player Market' without strict suffix (e.g., contains in middle)
                            if mkt in s:
                                parts = s.split(mkt, 1)
                                player = parts[0].strip()
                                return player, mkt
                        # fallback: first two words as name, rest as market
                        toks = s.split()
                        if len(toks) >= 3:
                            player = " ".join(toks[:2])
                            market = " ".join(toks[2:])
                            return player, market
                        return s, None

                    player, market = split_player_market(prop_text)

                    out.append({
                        "event": matchup or None,
                        "date": date_time or None,
                        "game_time": None,
                        "player": player or None,
                        "market": market or None,
                        "line": line_text or None,
                        "outcome": None,
                        "odds": odds_val,
                        "team": None,
                        "matchup": matchup or None,
                        "update_time": now_epoch,
                    })
                # Finished parsing this table; continue to next table
                continue

        headers = [clean(th.get_text()) for th in tbl.select("thead th")]
        if not headers:
            first_tr = tbl.select_one("tr")
            if first_tr:
                headers = [clean(th.get_text()) for th in first_tr.select("th, td")]
        norm = [h.lower() for h in headers]
        if not norm or not any("player" in h for h in norm):
            continue

        def idx(options):
            for i, h in enumerate(norm):
                if any(opt in h for opt in options):
                    return i
            return -1

        i_event   = idx(["event"])
        i_date    = idx(["date"])
        i_time    = idx(["time","game time","start"])
        i_player  = idx(["player"])
        i_market  = idx(["prop","market"])
        i_line    = idx(["line","target","threshold"])
        i_team    = idx(["team"])
        i_match   = idx(["matchup","game"])
        i_over    = idx(["over"])
        i_under   = idx(["under"])
        i_odds    = idx(["odds"])

        rows = tbl.select("tbody tr") or tbl.select("tr")[1:]
        for tr in rows:
            cells = [clean(td.get_text()) for td in tr.select("td")]
            if not cells:
                continue
            def v(i): return cells[i] if 0 <= i < len(cells) else ""
            player = v(i_player)
            if not player:
                continue
            base = {
                "event": v(i_event) or None,
                "date": v(i_date) or None,
                "game_time": v(i_time) or None,
                "player": player,
                "market": v(i_market) or None,
                "line": v(i_line) or None,
                "team": v(i_team) or None,
                "matchup": v(i_match) or None,
                "update_time": now_epoch,
            }
            if i_over != -1 or i_under != -1:
                over_odds = _odds_to_int(v(i_over)) if i_over != -1 else None
                under_odds = _odds_to_int(v(i_under)) if i_under != -1 else None
                if over_odds is not None:
                    out.append({**base, "outcome": "Over", "odds": over_odds})
                if under_odds is not None:
                    out.append({**base, "outcome": "Under", "odds": under_odds})
            else:
                oo = _odds_to_int(v(i_odds))
                if oo is not None:
                    # Try to infer Yes/No from market text
                    mtxt = (base["market"] or "").lower()
                    outcome = "Yes" if "yes" in mtxt else ("No" if "no" in mtxt else None)
                    out.append({**base, "outcome": outcome, "odds": oo})

    if out:
        return out

    # Div-row fallback
    rows = soup.select("div.tb-row, li, article, div[class*='row']")
    for row in rows:
        txt = _clean_text(row)
        if not txt:
            continue
        odds_tokens = [t for t in txt.split() if ODDS_TOKEN_RE.match(t)]
        if not odds_tokens:
            continue
        player = ""
        for sel in [".tb-player", ".player", "a[href*='/players/']", "strong", "h3", "h4"]:
            el = row.select_one(sel)
            if el:
                player = clean(el.get_text()); break
        if not player:
            m = re.search(r"[A-Z][a-z]+ [A-Z][a-zA-Z.\-']+", txt)
            player = m.group(0) if m else ""

        market = ""
        for sel in [".tb-market", ".market", ".prop-type", ".tb-prop"]:
            el = row.select_one(sel)
            if el:
                market = clean(el.get_text()); break
        if not market:
            for mkt in ["Total Bases", "Home Runs", "Hits", "Strikeouts", "RBIs", "Runs", "Stolen Bases"]:
                if mkt.lower() in txt.lower():
                    market = mkt; break

        line = ""
        for sel in [".tb-line", ".line", ".threshold", ".target"]:
            el = row.select_one(sel)
            if el:
                line = clean(el.get_text()); break

        matchup = ""
        for sel in [".tb-game", ".matchup", ".vs", ".tb-event", ".event"]:
            el = row.select_one(sel)
            if el:
                matchup = clean(el.get_text()); break

        over_node = row.find(string=re.compile(r"\bOver\b", re.I))
        under_node = row.find(string=re.compile(r"\bUnder\b", re.I))
        over_odds = _odds_to_int(_clean_text(over_node.parent)) if over_node and getattr(over_node, "parent", None) else None
        under_odds = _odds_to_int(_clean_text(under_node.parent)) if under_node and getattr(under_node, "parent", None) else None

        if over_odds is None and under_odds is None and odds_tokens:
            out.append({
                "event": None, "date": None, "game_time": None,
                "player": player or None, "market": market or None, "line": line or None,
                "outcome": None, "odds": _odds_to_int(odds_tokens[0]),
                "team": None, "matchup": matchup or None, "update_time": now_epoch,
            })
        else:
            if over_odds is not None:
                out.append({"event": None, "date": None, "game_time": None, "player": player or None, "market": market or None, "line": line or None, "outcome": "Over", "odds": over_odds, "team": None, "matchup": matchup or None, "update_time": now_epoch})
            if under_odds is not None:
                out.append({"event": None, "date": None, "game_time": None, "player": player or None, "market": market or None, "line": line or None, "outcome": "Under", "odds": under_odds, "team": None, "matchup": matchup or None, "update_time": now_epoch})
    return out

def compute_public_index_base_from_splits(splits: List[Dict[str, Any]], top_k: int = 3) -> float:
    """Average of the top-k most lopsided public %handle entries across ML/Spread/Total."""
    # keep only conventional markets
    rows = [r for r in splits if any(tok in (r.get("market","")).lower() for tok in ("moneyline","spread","total","o/u","over/under"))]
    pct_handles = sorted([float(r.get("%handle", 0.0)) for r in rows], reverse=True)[:max(1, top_k)]
    if not pct_handles:
        return 0.0
    return sum(pct_handles)/len(pct_handles)

# === NRFI Cult Factor helpers ===
def american_to_implied(odds_val: Any) -> Optional[float]:
    try:
        s = str(odds_val).strip()
        if not s:
            return None
        m = re.search(r"[-+]?\d+", s)
        if not m:
            return None
        o = int(m.group(0))
        if o == 0:
            return None
        if o > 0:
            return 100.0 / (o + 100.0)
        # negative
        return (-o) / ((-o) + 100.0)
    except Exception:
        return None

# --- Generic Early Under Factor for MLB NRFI and NFL 1H/1Q Unders ---
def compute_early_under_factor_from_odds_df(
    df,
    market_col: str,
    label_col: str,
    odds_col: str,
    game_col: str,
    market_match: Optional[List[str]] = None,
    under_label_prefix: str = "Under",
    threshold_prob: float = 0.56
):
    """
    Generic early-period binary market factor:
    - MLB: NRFI via market like 'totals_1st_1_innings' with label 'Under'
    - NFL: 1H/1Q Totals via market strings like 'Total 1st Half' or '1st Quarter Total' with label 'Under'
    Returns a dict compatible with the NRFI function keys, but named 'early_factor'.
    """
    if df is None or df.empty:
        return {"early_factor": 0, "breadth": 0.0, "n_games": 0, "n_strong": 0, "threshold": threshold_prob, "per_game": {}}

    work = df.copy()
    if market_col and market_col in work.columns and market_match:
        lc = work[market_col].astype(str).str.lower()
        mm = [m.lower() for m in market_match]
        work = work[lc.apply(lambda x: any(m in x for m in mm))].copy()

    if label_col not in work.columns or odds_col not in work.columns or game_col not in work.columns:
        return {"early_factor": 0, "breadth": 0.0, "n_games": 0, "n_strong": 0, "threshold": threshold_prob, "per_game": {}}

    work["_label"] = work[label_col].astype(str).str.strip().str.lower()
    work["_implied"] = work[odds_col].apply(american_to_implied)

    per_game = {}
    for g, grp in work.groupby(game_col):
        # collect implied probs where label begins with under_label_prefix
        probs = [p for p, l in zip(grp.get("_implied", []), grp.get("_label", [])) if l.startswith(under_label_prefix.lower()) and p is not None]
        if probs:
            probs_sorted = sorted(probs)
            n = len(probs_sorted)
            med = probs_sorted[n // 2] if n % 2 == 1 else 0.5 * (probs_sorted[n // 2 - 1] + probs_sorted[n // 2])
            per_game[g] = med

    n_games = len(per_game)
    if n_games == 0:
        return {"early_factor": 0, "breadth": 0.0, "n_games": 0, "n_strong": 0, "threshold": threshold_prob, "per_game": {}}

    n_strong = sum(1 for v in per_game.values() if v is not None and v >= threshold_prob)
    breadth = n_strong / float(n_games)

    if breadth < 0.15:
        factor = 0
    elif breadth < 0.30:
        factor = 1
    elif breadth < 0.50:
        factor = 2
    else:
        factor = 3

    return {
        "early_factor": factor,
        "breadth": breadth,
        "n_games": n_games,
        "n_strong": n_strong,
        "threshold": threshold_prob,
        "per_game": per_game
    }

def compute_nrfi_cult_factor_from_odds_df(df, market_col: str, label_col: str, odds_col: str, game_col: str, point_col: Optional[str] = None):
    """Compute NRFI Cult Factor from an odds dataframe.
    - Expect rows with market == 'totals_1st_1_innings'
    - label: 'Under' == NRFI, 'Over' == YRFI
    - odds: American odds for that outcome
    - game_col: column identifying the matchup/game
    Returns dict with factor (0-3), breadth, n_games, n_strong, threshold used, plus per-game implied probabilities.
    """
    if df is None or df.empty:
        return {"nrfi_factor": 0, "breadth": 0.0, "n_games": 0, "n_strong": 0, "threshold": 0.56, "per_game": {}}

    # Filter to first-inning totals market if present
    if market_col and market_col in df.columns:
        msk = df[market_col].astype(str).str.lower() == "totals_1st_1_innings"
        work = df[msk].copy()
        if work.empty:
            work = df.copy()
    else:
        work = df.copy()

    # Normalize label casing
    if label_col not in work.columns or odds_col not in work.columns or game_col not in work.columns:
        return {"nrfi_factor": 0, "breadth": 0.0, "n_games": 0, "n_strong": 0, "threshold": 0.56, "per_game": {}}

    work["_label"] = work[label_col].astype(str).str.strip().str.lower()
    work["_implied"] = work[odds_col].apply(american_to_implied)

    # Group by game, take median implied prob for NRFI (Under) across books
    per_game = {}
    for g, grp in work.groupby(game_col):
        nrfi_probs = [p for p,l in zip(grp.get("_implied", []), grp.get("_label", [])) if l.startswith("under") and p is not None]
        if nrfi_probs:
            # median
            nrfi_probs_sorted = sorted(nrfi_probs)
            n = len(nrfi_probs_sorted)
            med = nrfi_probs_sorted[n//2] if n % 2 == 1 else 0.5*(nrfi_probs_sorted[n//2-1] + nrfi_probs_sorted[n//2])
            per_game[g] = med

    n_games = len(per_game)
    if n_games == 0:
        return {"nrfi_factor": 0, "breadth": 0.0, "n_games": 0, "n_strong": 0, "threshold": 0.56, "per_game": {}}

    # Thresholds as discussed: treat widespread juiced NRFI as cult activity
    threshold = 0.56  # approx -127 fair price
    n_strong = sum(1 for v in per_game.values() if v is not None and v >= threshold)
    breadth = n_strong / float(n_games)

    # Map breadth to factor 0-3
    if breadth < 0.15:
        factor = 0
    elif breadth < 0.30:
        factor = 1
    elif breadth < 0.50:
        factor = 2
    else:
        factor = 3

    return {
        "nrfi_factor": factor,
        "breadth": breadth,
        "n_games": n_games,
        "n_strong": n_strong,
        "threshold": threshold,
        "per_game": per_game
    }


def render(state: dict | None = None):
    """Streamlit UI entrypoint suitable for both the router and standalone use.
    If `state` is provided (router page bucket), values persist there; otherwise
    it falls back to Streamlit's session_state for standalone runs.
    """
    if st is None or pd is None:
        return

    # NOTE: page config is handled by the router; keep it out here to avoid conflicts
    # try:
    #     st.set_page_config(page_title="Public Index Console", layout="wide")
    # except Exception:
    #     pass

    st.title("Public Index Console")
    st.caption("A minimal, procedural workflow. Three steps. No guessing.")

    # ---- Session State Defaults ----
    ss = state if state is not None else st.session_state
    ss.setdefault("parlays", None)
    ss.setdefault("splits", None)
    ss.setdefault("props", None)
    ss.setdefault("pib_last", None)
    ss.setdefault("surge_last", None)
    ss.setdefault("nrfi_last", None)
    ss.setdefault("public_index_preview", None)
    ss.setdefault("splits_enriched", None)
    ss.setdefault("viug_summ", None)
    ss.setdefault("overlay_rows", None)
    ss.setdefault("overlay_source_name", None)

    # Sidebar: status
    with st.sidebar:
        st.header("Status")
        st.write(f"Parlays: {'✅' if ss['parlays'] else '—'}")
        st.write(f"Splits: {'✅' if ss['splits'] else '—'}")
        st.write(f"Props: {'✅' if ss['props'] else '—'}")
        st.write("PI Base: " + (f"{ss['pib_last']:.2f}" if ss['pib_last'] is not None else "—"))
        st.write(f"Parlay Surge: {ss['surge_last']['parlay_surge_score'] if ss['surge_last'] else '—'}")
        st.write(f"Early Factor: {ss['nrfi_last']['nrfi_factor'] if ss['nrfi_last'] else (ss['early_last']['early_factor'] if ss.get('early_last') else '—')}")
        st.write("Public Index: " + (f"{ss['public_index_preview']:.2f}" if ss['public_index_preview'] is not None else "—"))

    # Tabs for the workflow steps
    step1, step2, step3, step4, content_tab, refs_tab, adv = st.tabs([
        "1) Fetch Parlays",
        "2) Fetch Splits & Surge",
        "3) Early Under Factor (NRFI/1H) & Public Index",
        "4) Player Props",
        "5) Content Builder",
        "Refs (live)",
        "Advanced"
    ])

    # ---------------- Refs (live) ----------------
    with refs_tab:
        st.subheader("Refs (live) — nfl_data_py")
        if _nfl is None:
            st.info("nfl_data_py not installed in this environment.")
        else:
            y_default = _pick_ref_data_year()
            colR1, colR2 = st.columns([1,3])
            with colR1:
                year_sel = st.number_input("Season year", min_value=2016, max_value=2100, value=int(y_default), step=1)
                if st.button("Build ref metrics", use_container_width=True):
                    try:
                        with st.spinner("Loading nfl_data_py and computing per-ref metrics…"):
                            df_refs = load_referee_metrics(int(year_sel))
                            if df_refs is None or df_refs.empty:
                                st.warning("No referee metrics available for that season.")
                            else:
                                st.success(f"Computed metrics for {len(df_refs)} referees")
                                st.dataframe(df_refs, use_container_width=True)
                                cards = build_ref_report_cards(df_refs)
                                if cards:
                                    st.download_button("Download Ref Report (MD)", data="\n\n".join(cards), file_name="referee_report_cards.md", mime="text/markdown")
                                st.download_button("Download Ref Metrics (CSV)", data=df_refs.to_csv(index=False), file_name="ref_metrics.csv", mime="text/csv")
                                # ---- Top 5 Ref Bias Tables ----
                                try:
                                    dfr = df_refs.copy()
                                    # Coerce numeric and fill NaNs
                                    for col in [
                                        "over_bias_pts", "penalties_pg", "home_penalty_share",
                                        "dpi_pg", "off_hold_pg", "def_hold_pg", "overall_diff_pct"
                                    ]:
                                        if col in dfr.columns:
                                            dfr[col] = pd.to_numeric(dfr[col], errors="coerce")
                                    dfr.fillna({
                                        "over_bias_pts": 0.0,
                                        "penalties_pg": 0.0,
                                        "home_penalty_share": 0.5,
                                        "dpi_pg": 0.0,
                                        "off_hold_pg": 0.0,
                                        "def_hold_pg": 0.0,
                                        "overall_diff_pct": 0.0,
                                    }, inplace=True)

                                    # Build tables
                                    cols_core = [c for c in ["ref_name","games","penalties_pg","over_bias_pts","home_penalty_share","dpi_pg","off_hold_pg","def_hold_pg"] if c in dfr.columns]

                                    top_over = dfr.sort_values("over_bias_pts", ascending=False).head(5)[cols_core]
                                    top_under = dfr.sort_values("over_bias_pts", ascending=True).head(5)[cols_core]
                                    top_home = dfr.sort_values("home_penalty_share", ascending=False).head(5)[cols_core]
                                    top_away = dfr.sort_values("home_penalty_share", ascending=True).head(5)[cols_core]
                                    top_flags = dfr.sort_values("penalties_pg", ascending=False).head(5)[cols_core]

                                    # Niceties: format percents
                                    def _fmt(df):
                                        df = df.copy()
                                        if "home_penalty_share" in df.columns:
                                            df["home_penalty_share"] = (df["home_penalty_share"].clip(0,1) * 100).round(1).astype(str) + "%"
                                        if "penalties_pg" in df.columns:
                                            df["penalties_pg"] = df["penalties_pg"].round(2)
                                        if "over_bias_pts" in df.columns:
                                            df["over_bias_pts"] = df["over_bias_pts"].map(lambda x: f"{x:+.2f}")
                                        for c in ["dpi_pg","off_hold_pg","def_hold_pg"]:
                                            if c in df.columns:
                                                df[c] = df[c].round(2)
                                        return df

                                    st.markdown("### Top 5 Ref Biases")
                                    colA, colB = st.columns(2)
                                    with colA:
                                        st.write("**Over bias (points vs closing total)**")
                                        st.dataframe(_fmt(top_over), use_container_width=True, height=220)
                                        st.write("**Home-heavy penalty share**")
                                        st.dataframe(_fmt(top_home), use_container_width=True, height=220)
                                    with colB:
                                        st.write("**Under bias (points vs closing total)**")
                                        st.dataframe(_fmt(top_under), use_container_width=True, height=220)
                                        st.write("**Away-heavy penalty share**")
                                        st.dataframe(_fmt(top_away), use_container_width=True, height=220)

                                    st.write("**Most flag-happy (penalties per game)**")
                                    st.dataframe(_fmt(top_flags), use_container_width=True, height=220)

                                except Exception as e:
                                    st.warning(f"Top-5 tables unavailable: {e}")
                    except Exception as e:
                        st.error(f"Error computing refs: {e}")
    # ---------------- Step 4: Player Props ----------------
    with step4:
        st.subheader("Step 4: Fetch Player Props (DKNetwork)")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            eg_props = st.number_input("Event Group (tb_eg)", min_value=1, max_value=999999, value=88808, step=1, help="84240 = MLB")
        with c2:
            edate_props = st.selectbox("Date (tb_edate)", options=["today","yesterday","last_7","last_30"], index=0)
        with c3:
            st.caption("Parses DKNetwork player props table into normalized rows with Over/Under odds.")
        use_pw = st.toggle("Use Playwright fallback", value=False, key="props_pw_toggle")
        if st.button("Fetch Player Props", use_container_width=True):
            try:
                with st.spinner("Fetching player props…"):
                    html = fetch_props_html(event_group=int(eg_props), edate=edate_props, use_playwright=use_pw)
                    props_rows = parse_player_props(html)
                    ss["props"] = props_rows
                st.success(f"Fetched {len(ss['props'])} props rows")
            except Exception as e:
                st.error(f"Error fetching props: {e}")

        if ss.get("props"):
            _df = pd.DataFrame(ss["props"])
            st.dataframe(_df.head(200), use_container_width=True)
            st.download_button("Download Props JSON", data=json.dumps(ss["props"], indent=2), file_name="dk_player_props.json", mime="application/json")
            st.download_button("Download Props CSV", data=_df.to_csv(index=False), file_name="dk_player_props.csv", mime="text/csv")

    # ---------------- Step 1: Parlays ----------------
    with step1:
        st.subheader("Step 1: Fetch most popular parlays")
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            url = st.text_input("DKNetwork Parlays URL", value=DEFAULT_URL)
        with col2:
            top_n = st.number_input("Top N", min_value=1, max_value=1000, value=10, step=1)
        with col3:
            min_bet_count = st.number_input("Min Bet Count", min_value=0, max_value=10**7, value=0, step=1)

        if st.button("Fetch DK Parlays", use_container_width=True):
            try:
                with st.spinner("Fetching parlays…"):
                    html = fetch_html(url)
                    cards = parse_cards(html, verbose=False)
                    cards = filter_cards(cards, None, int(min_bet_count))
                    if int(top_n) > 0:
                        cards = cards[: int(top_n)]
                    ss["parlays"] = cards
                st.success(f"Fetched {len(ss['parlays'])} parlays")
            except Exception as e:
                st.error(f"Error fetching parlays: {e}")

        if ss["parlays"]:
            df = pd.DataFrame(ss["parlays"])
            df_display = df.copy()
            df_display["legs"] = df_display["legs"].apply(lambda x: "\n".join(x) if isinstance(x, list) else "")
            st.dataframe(df_display[["title", "parlay_type", "odds", "bet_count", "legs"]], use_container_width=True)
            json_str = json.dumps({"top": ss["parlays"], "source": url, "as_of_epoch": int(time.time())}, indent=2, ensure_ascii=False)
            st.download_button("Download Parlays JSON", json_str, file_name="dk_parlays.json", mime="application/json")

    # ---------------- Step 2: Splits & Surge ----------------
    with step2:
        st.subheader("Step 2: Fetch betting splits and compute Parlay Surge")
        colA, colB, colC = st.columns([1,1,2])
        with colA:
            league = st.selectbox("League preset", ["NFL","MLB"], index=0)
            default_eg = 84240 if league == "MLB" else 88808
            eg = st.number_input("Event Group", min_value=1, max_value=999999, value=default_eg, step=1, help="Set DK tb_eg. Choose preset then override if needed.")
        with colB:
            edate = st.selectbox("Date Range", options=["today","yesterday","last_7","last_30"], index=0)
        with colC:
            st.caption("Loads the DK betting-splits page and extracts %handle/%bets by market/side.")

        disabled = sync_playwright is None
        fetch_lbl = "Fetch DK Splits (requires Playwright)" if disabled else "Fetch DK Splits"
        if st.button(fetch_lbl, disabled=disabled, use_container_width=True):
            try:
                with st.spinner("Fetching DK splits via Playwright…"):
                    rows = fetch_dk_splits(event_group=int(eg), date_range=edate)
                    ss["splits"] = rows
                    # compute immediately
                    fav_handles, over_handles = derive_handles_from_splits(rows)
                    surge = compute_parlay_surge_from_handles(fav_handles, over_handles)
                    pib = compute_public_index_base_from_splits(rows, top_k=3)
                    ss["pib_last"] = pib
                    ss["surge_last"] = surge
                    # internal overlay only (no CSVs)
                    try:
                        enriched = enrich_splits_with_internal_heuristics(rows)
                        ss["splits_enriched"] = enriched
                        ss["viug_summ"] = summarize_vi_ug(enriched) if enriched else None
                    except Exception:
                        ss["splits_enriched"] = None
                        ss["viug_summ"] = None
                st.success(f"Fetched {len(ss['splits'])} split rows • PI Base {pib:.2f} • Parlay Surge {surge['parlay_surge_score']}")
            except Exception as e:
                st.error(f"Error fetching splits: {e}")

        if ss["splits"]:
            with st.expander("Preview splits (first 50 rows)"):
                st.dataframe(pd.DataFrame(ss["splits"]).head(50), use_container_width=True)
            # Preview enriched splits if available
            if ss.get("splits_enriched"):
                with st.expander("Preview enriched splits (first 50 rows)"):
                    st.dataframe(pd.DataFrame(ss["splits_enriched"]).head(50), use_container_width=True)
            with st.container():
                if ss["pib_last"] is not None and ss["surge_last"] is not None:
                    surge = ss["surge_last"]
                    st.metric("Public Index Base (avg top-3 %handle)", f"{ss['pib_last']:.2f}")
                    st.metric("Parlay Surge Score", int(surge["parlay_surge_score"]))
                    st.caption(f"HHI Sides {surge['hhi_sides']:.4f} • HHI Totals {surge['hhi_totals']:.4f} • n_corr {surge['n_corr']}")
                # Show VI/UG summaries and overlay preview
                if ss.get("viug_summ"):
                    viug = ss["viug_summ"]
                    st.metric("Variance Index (median)", f"{viug['vi_med']:.2f}")
                    st.metric("Under Gravity (median)", f"{viug['ug_med']:.2f}")
                if ss.get("splits_enriched"):
                    with st.expander("Ref & Pace overlay preview"):
                        df_en = pd.DataFrame(ss["splits_enriched"]).copy()
                        if "%handle" in df_en.columns and "handle_pct" not in df_en.columns:
                            df_en.rename(columns={"%handle":"handle_pct"}, inplace=True)
                        if "%bets" in df_en.columns and "bets_pct" not in df_en.columns:
                            df_en.rename(columns={"%bets":"bets_pct"}, inplace=True)
                        cols_show = [c for c in ["matchup","market","side","odds","handle_pct","bets_pct","crew_id","ref_name","vi","ug","dei"] if c in df_en.columns]
                        st.dataframe(df_en[cols_show].head(50), use_container_width=True)


            st.divider()
            with st.expander("Teaser Eligibility Scanner (NFL) — from splits"):
                st.caption("Find classic Wong-style teaser legs directly from DK splits: favorites −8.5 to −7.5, dogs +1.5 to +2.5. Optional total cap filters out high-variance games.")
                colT1, colT2, colT3, colT4, colT5 = st.columns([1,1,1,1,2])
                with colT1:
                    fav_low = st.number_input("Fav low", value=-8.5, step=0.5, help="Lower bound for favorite spread")
                with colT2:
                    fav_high = st.number_input("Fav high", value=-7.5, step=0.5, help="Upper bound for favorite spread")
                with colT3:
                    dog_low = st.number_input("Dog low", value=1.5, step=0.5, help="Lower bound for underdog spread")
                with colT4:
                    dog_high = st.number_input("Dog high", value=2.5, step=0.5, help="Upper bound for underdog spread")
                with colT5:
                    use_total_cap = st.checkbox("Cap total", value=True)
                    max_total = st.number_input("Max total", value=49.5, step=0.5, disabled=not use_total_cap)
                if st.button("Scan Teaser Legs", use_container_width=True):
                    fav_range = (min(fav_low, fav_high), max(fav_low, fav_high))
                    dog_range = (min(dog_low, dog_high), max(dog_low, dog_high))
                    cap = float(max_total) if use_total_cap else None
                    legs = compute_teaser_candidates_from_splits(ss["splits"], fav_range=fav_range, dog_range=dog_range, max_total=cap)
                    if legs:
                        # Add approximate cover probabilities for a 6-point teaser
                        for rec in legs:
                            sp = float(rec.get("spread", 0.0) or 0.0)
                            rec["cover_prob"] = approx_cover_prob(spread=sp, teaser_shift=6.0, sigma=13.0)
                        df_teaser = pd.DataFrame(legs)
                        st.success(f"Found {len(df_teaser)} teaser-eligible legs")
                        cols = [c for c in ["matchup","teaser_leg_type","side","spread","game_total","handle_pct","bets_pct","cover_prob"] if c in df_teaser.columns]
                        st.dataframe(df_teaser[cols], use_container_width=True, height=340)

                        price = st.selectbox("Two-leg teaser price", options=[-130, -125, -120, -115], index=2)
                        from itertools import combinations
                        rows = []
                        items = df_teaser.to_dict(orient="records")
                        for a, b in combinations(items, 2):
                            ev = teaser_ev_two_leg(a.get("cover_prob", 0.5), b.get("cover_prob", 0.5), price=int(price))
                            rows.append({
                                "pair": f"{a['matchup']} + {b['matchup']}",
                                "type": f"{a.get('teaser_leg_type','')} + {b.get('teaser_leg_type','')}",
                                f"EV@{price}": round(ev, 4)
                            })
                        if rows:
                            df_pairs = pd.DataFrame(sorted(rows, key=lambda r: r[f"EV@{price}"], reverse=True)[:20])
                            st.write("**Best 2-leg EV combos (approx)**")
                            st.dataframe(df_pairs, use_container_width=True, height=280)
                        st.download_button("Download Teaser Legs CSV", df_teaser.to_csv(index=False), file_name="teaser_legs.csv", mime="text/csv")
                    else:
                        st.info("No teaser-eligible legs found under current filters.")

            st.download_button("Download Splits JSON", json.dumps(ss["splits"], indent=2), file_name="dk_splits.json", mime="application/json")

    # ---------------- Step 3: NRFI & Public Index ----------------
    with step3:
        st.subheader("Step 3: Early Under Factor — 1H/1Q Unders (NFL) or NRFI (MLB) — then Public Index")
        st.caption("Upload odds CSV. MLB: market='totals_1st_1_innings', label 'Under' for NRFI. NFL: use 1H/1Q Total markets where label is 'Under' (e.g., 'Total 1st Half').")
        odds_file = st.file_uploader("Upload NRFI/YRFI odds CSV", type=["csv"], key="odds_csv_uploader_main")

        cols = None
        if odds_file is not None:
            try:
                df_odds = pd.read_csv(odds_file)
                st.dataframe(df_odds.head(15), use_container_width=True)
                cols = list(df_odds.columns)
                game_col = st.selectbox("Game identifier column", options=cols, key="nrfi_game")
                market_col = st.selectbox("Market column", options=cols, index=(cols.index("market") if "market" in cols else 0), key="nrfi_market")
                label_col = st.selectbox("Label column", options=cols, index=(cols.index("label") if "label" in cols else 0), key="nrfi_label")
                odds_col = st.selectbox("Odds column (American)", options=cols, key="nrfi_odds")

                sport_mode = st.selectbox("Sport mode", options=["MLB (NRFI)", "NFL (1H/1Q Under)"], index=0, help="Select parsing intent for Early Under Factor")

                if st.button("Compute Early Under Factor", use_container_width=True):
                    if sport_mode.startswith("MLB"):
                        res = compute_nrfi_cult_factor_from_odds_df(df_odds, market_col, label_col, odds_col, game_col)
                        ss["nrfi_last"] = res
                        ss["early_last"] = {"early_factor": res.get("nrfi_factor", 0), "breadth": res.get("breadth", 0.0), "n_games": res.get("n_games", 0), "n_strong": res.get("n_strong", 0), "threshold": res.get("threshold", 0.56), "per_game": res.get("per_game", {})}
                        st.success(f"Early Factor {ss['early_last']['early_factor']} • Breadth {ss['early_last']['breadth']:.2%} • Strong {ss['early_last']['n_strong']}/{ss['early_last']['n_games']}")
                    else:
                        # NFL mode: match 1H/1Q totals markets
                        res = compute_early_under_factor_from_odds_df(
                            df_odds,
                            market_col=market_col,
                            label_col=label_col,
                            odds_col=odds_col,
                            game_col=game_col,
                            market_match=[
                                "1h total","total 1st half","1st half total","first half total",
                                "1q total","total 1st quarter","1st quarter total","first quarter total",
                                "half total","quarter total"
                            ],
                            under_label_prefix="Under",
                            threshold_prob=0.56
                        )
                        ss["early_last"] = res
                        # maintain legacy key for downstream math by mapping to 'nrfi_last'-like schema
                        ss["nrfi_last"] = {"nrfi_factor": res["early_factor"], "breadth": res["breadth"], "n_games": res["n_games"], "n_strong": res["n_strong"], "threshold": res["threshold"], "per_game": res["per_game"]}
                        st.success(f"Early Factor {res['early_factor']} • Breadth {res['breadth']:.2%} • Strong {res['n_strong']}/{res['n_games']}")

            except Exception as e:
                st.error(f"Error reading odds CSV: {e}")

        # If we have all three components, compute the Public Index preview
        if ss.get("pib_last") is not None and ss.get("surge_last") is not None and ss.get("nrfi_last") is not None:
            parlay_adj_map = [0, 5, 8, 12]
            nrfi_adj_map = [0, 4, 7, 10]
            parlay_adj = parlay_adj_map[min(3, max(0, int(ss['surge_last'].get('parlay_surge_score', 0))))]
            nrfi_adj = nrfi_adj_map[min(3, max(0, int(ss['nrfi_last'].get('nrfi_factor', ss['nrfi_last'].get('early_factor', 0)))))]
            pib_val = float(ss["pib_last"] or 0.0)
            # Optional overlay adj from refs/pace if available
            vi_med = ug_med = 0.0
            if ss.get("viug_summ"):
                vi_med = float(ss["viug_summ"].get("vi_med", 0.0) or 0.0)
                ug_med = float(ss["viug_summ"].get("ug_med", 0.0) or 0.0)
            overlay_adj = max(-6.0, min(6.0, (-4.0 * ug_med) + (2.0 * vi_med)))
            public_index = max(0.0, min(100.0, pib_val + parlay_adj + nrfi_adj + overlay_adj))
            ss["public_index_preview"] = public_index

            st.divider()
            st.subheader("Public Index")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("PI Base", f"{pib_val:.2f}")
            c2.metric("Parlay Adj", parlay_adj)
            c3.metric("NRFI/1H Adj", nrfi_adj)
            c4.metric("Refs/Pace Adj", f"{overlay_adj:+.2f}")
            c5.metric("Public Index", f"{public_index:.2f}")
            out = {
                "date_epoch": int(time.time()),
                "public_index": public_index,
                "index_base": pib_val,
                "parlay_surge_score": ss["surge_last"]["parlay_surge_score"],
                "parlay_adj": parlay_adj,
                "nrfi_factor": ss["nrfi_last"]["nrfi_factor"],
                "nrfi_adj": nrfi_adj,
                "overlay_adj": overlay_adj,
                "vi_med": vi_med,
                "ug_med": ug_med,
            }
            st.download_button("Download Public Index JSON", json.dumps(out, indent=2), file_name="public_index.json", mime="application/json")

    # ---------------- 5) Content Builder (Report & Post Generator) ----------------
    with content_tab:
        st.subheader("5) Content Builder — Report & Post Generator")
        st.caption("Auto-structures today’s outputs into a clean report plus copy blocks for Free and Paid posts.")

        # Guard rails
        if not ss.get("parlays") or not ss.get("splits") or ss.get("pib_last") is None:
            st.info("Run Steps 1–3 first. Need Parlays, Splits, and PI Base at minimum.")
        else:
            # ---------- Derive quick metrics ----------
            pib_val = float(ss.get("pib_last") or 0.0)
            surge = ss.get("surge_last") or {"parlay_surge_score": 0, "hhi_sides": 0.0, "hhi_totals": 0.0, "n_corr": 0}
            nrfi_last = ss.get("nrfi_last") or {"nrfi_factor": 0, "breadth": 0.0, "n_games": 0, "n_strong": 0}
            public_index_preview = ss.get("public_index_preview")

            vi_med = ug_med = 0.0
            if ss.get("viug_summ"):
                vi_med = float(ss["viug_summ"].get("vi_med", 0.0) or 0.0)
                ug_med = float(ss["viug_summ"].get("ug_med", 0.0) or 0.0)
            overlay_adj = max(-6.0, min(6.0, (-4.0 * ug_med) + (2.0 * vi_med)))
            live_pi = public_index_preview if public_index_preview is not None else max(0.0, min(100.0, pib_val + [0,5,8,12][min(3,int(surge.get('parlay_surge_score',0)))] + [0,4,7,10][min(3,int(nrfi_last.get('nrfi_factor',nrfi_last.get('early_factor',0))))] + overlay_adj))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PI Base", f"{pib_val:.2f}")
            c2.metric("Parlay Surge", int(surge.get("parlay_surge_score", 0)))
            c3.metric("Early Factor", int(nrfi_last.get("nrfi_factor", nrfi_last.get("early_factor", 0))))
            c4.metric("Public Index", f"{live_pi:.2f}")

            st.divider()
            # Ref & Pace Signals block (optional)
            if ss.get("splits_enriched"):
                st.divider()
                st.markdown("### Ref & Pace Signals")
                df_en = pd.DataFrame(ss["splits_enriched"]).copy()
                if "%handle" in df_en.columns and "handle_pct" not in df_en.columns:
                    df_en.rename(columns={"%handle":"handle_pct"}, inplace=True)
                if "%bets" in df_en.columns and "bets_pct" not in df_en.columns:
                    df_en.rename(columns={"%bets":"bets_pct"}, inplace=True)
                df_en["delta_handle_bets"] = (pd.to_numeric(df_en.get("handle_pct",0), errors="coerce").fillna(0.0) - pd.to_numeric(df_en.get("bets_pct",0), errors="coerce").fillna(0.0)).round(1)
                # Respected Unders with high UG
                if all(col in df_en.columns for col in ["market","side","ug"]):
                    ru_hug = df_en[(df_en["market"].astype(str).str.contains("Total", case=False)) &
                                   (df_en["side"].astype(str).str.contains("Under", case=False))].copy()
                    ru_hug = ru_hug.sort_values(["ug","handle_pct","delta_handle_bets"], ascending=[False, False, False]).head(8)
                    st.write("**Respected Unders with high Under Gravity (UG)**")
                    cols_show = [c for c in ["matchup","side","odds","handle_pct","bets_pct","ug","delta_handle_bets"] if c in ru_hug.columns]
                    st.dataframe(ru_hug[cols_show], use_container_width=True, height=220)
                # Public Overs in low UG
                if all(col in df_en.columns for col in ["market","side","ug"]):
                    po_lowug = df_en[(df_en["market"].astype(str).str.contains("Total", case=False)) &
                                     (df_en["side"].astype(str).str.contains("Over", case=False))].copy()
                    po_lowug = po_lowug.sort_values(["ug","bets_pct"], ascending=[True, False]).head(8)
                    st.write("**Public Overs in low UG (potential fades)**")
                    cols_show = [c for c in ["matchup","side","odds","handle_pct","bets_pct","ug"] if c in po_lowug.columns]
                    st.dataframe(po_lowug[cols_show], use_container_width=True, height=220)
                if ss.get("overlay_source_name"):
                    st.caption(f"Overlay source: {ss['overlay_source_name']}")

            # ---------- Build dataframes for splits ----------
            import pandas as _pd
            splits_df = _pd.DataFrame(ss["splits"]).copy()
            if not splits_df.empty:
                # Normalize for safety
                if "%handle" in splits_df.columns:
                    splits_df.rename(columns={"%handle":"handle_pct"}, inplace=True)
                if "%bets" in splits_df.columns:
                    splits_df.rename(columns={"%bets":"bets_pct"}, inplace=True)
                for col in ["handle_pct","bets_pct"]:
                    if col in splits_df.columns:
                        splits_df[col] = _pd.to_numeric(splits_df[col], errors="coerce").fillna(0.0)
                splits_df["delta_handle_bets"] = (splits_df.get("handle_pct",0) - splits_df.get("bets_pct",0)).round(1)

                # Public love = bets >> handle by 20+ points
                public_love = splits_df[(splits_df.get("bets_pct",0) - splits_df.get("handle_pct",0) >= 20)].copy()
                public_love["gap"] = (public_love["bets_pct"] - public_love["handle_pct"]).abs()
                public_love = public_love.sort_values(["gap","handle_pct"], ascending=[False, False]).head(10)

                # Sharp steam = handle >> bets by 20+ points
                sharp_steam = splits_df[(splits_df.get("handle_pct",0) - splits_df.get("bets_pct",0) >= 20)].copy()
                sharp_steam["gap"] = (sharp_steam["handle_pct"] - sharp_steam["bets_pct"]).abs()
                sharp_steam = sharp_steam.sort_values(["gap","handle_pct"], ascending=[False, False]).head(10)

                # Public piling into Overs
                public_overs = splits_df[(splits_df["market"].astype(str).str.contains("Total", case=False)) & (splits_df["side"].astype(str).str.contains("Over", case=False))].copy()
                public_overs = public_overs.sort_values(["bets_pct","handle_pct"], ascending=False).head(8)

                # Respected Unders (handle > bets by 15+)
                respected_unders = splits_df[(splits_df["market"].astype(str).str.contains("Total", case=False)) & (splits_df["side"].astype(str).str.contains("Under", case=False)) & ((splits_df["handle_pct"] - splits_df["bets_pct"]) >= 15)].copy()
                respected_unders = respected_unders.sort_values(["handle_pct","delta_handle_bets"], ascending=False).head(8)
            else:
                public_love = sharp_steam = public_overs = respected_unders = _pd.DataFrame()

            # ---------- Show concise tables ----------
            st.markdown("### Market Snapshot")
            colA, colB = st.columns(2)
            with colA:
                st.write("**Public love (bets ≫ handle)** — potential fades")
                st.dataframe(public_love[[c for c in ["matchup","market","side","odds","handle_pct","bets_pct","gap"] if c in public_love.columns]], use_container_width=True, height=260)
                st.write("**Public Overs** — crowd is leaning high totals")
                st.dataframe(public_overs[[c for c in ["matchup","side","odds","handle_pct","bets_pct"] if c in public_overs.columns]], use_container_width=True, height=260)
            with colB:
                st.write("**Sharp steam (handle ≫ bets)** — potential respect")
                st.dataframe(sharp_steam[[c for c in ["matchup","market","side","odds","handle_pct","bets_pct","gap"] if c in sharp_steam.columns]], use_container_width=True, height=260)
                st.write("**Respected Unders** — handle leading by 15+ pts")
                st.dataframe(respected_unders[[c for c in ["matchup","side","odds","handle_pct","bets_pct","delta_handle_bets"] if c in respected_unders.columns]], use_container_width=True, height=260)

            st.divider()

            # ---------- Parlay themes ----------
            st.markdown("### Parlay Crowd Themes")
            parlays = ss.get("parlays") or []
            legs_flat = " ".join([" ".join(p.get("legs", [])) for p in parlays]).lower()
            nrfi_count = legs_flat.count("no run 1st inning")
            yrfi_count = legs_flat.count("yes run 1st inning")
            hr_count = legs_flat.count("home run")
            tb_count = legs_flat.count("total bases")
            hits_count = legs_flat.count(" hits")

            td_any_count = legs_flat.count("anytime touchdown")
            rec_yards_count = legs_flat.count("receiving yards")
            rush_yards_count = legs_flat.count("rushing yards")
            pass_yards_count = legs_flat.count("passing yards")
            fh_under_count = legs_flat.count("1st half under") + legs_flat.count("first half under")
            fq_under_count = legs_flat.count("1st quarter under") + legs_flat.count("first quarter under")

            colP1, colP2, colP3, colP4, colP5, colP6 = st.columns(6)
            colP1.metric("NRFI legs", nrfi_count)
            colP2.metric("Anytime TD legs", td_any_count)
            colP3.metric("Receiving Yds legs", rec_yards_count)
            colP4.metric("Rushing Yds legs", rush_yards_count)
            colP5.metric("Passing Yds legs", pass_yards_count)
            colP6.metric("1H/1Q Under legs", fh_under_count + fq_under_count)

            if parlays:
                top_titles = [p.get("title","") for p in parlays[:5]]
                st.caption("Top Parlay Titles:")
                for t in top_titles:
                    st.write(f"• {t}")

            st.divider()

            # Optional: re-run teaser scan with defaults for the content pack
            try:
                teaser_legs = compute_teaser_candidates_from_splits(ss.get("splits") or [], fav_range=(-8.5,-7.5), dog_range=(1.5,2.5), max_total=49.5)
            except Exception:
                teaser_legs = []

            # ---------- Teaser Candidates (quick view) ----------
            st.markdown("### Teaser Candidates (quick view)")
            if teaser_legs:
                st.dataframe(pd.DataFrame(teaser_legs)[["matchup","teaser_leg_type","side","spread","game_total","handle_pct","bets_pct"]], use_container_width=True, height=200)
            else:
                st.caption("No teaser-eligible legs under default filters (fav −8.5..−7.5, dog +1.5..+2.5, total ≤ 49.5). Use Step 2 expander to tune.")

            # ---------- Auto-generated content blocks ----------
            st.markdown("### Generated Copy Blocks")
            # Build Free Post copy (macro smack + data, no picks)
            free_title = f"Early Under Kindergarten (NRFI/1H). Public Index: {int(public_index_preview or (pib_val + 0))}."
            free_body_lines = []
            if td_any_count or rec_yards_count or rush_yards_count or pass_yards_count:
                free_body_lines.append("Parlay bait is all touchdowns and yards today. Cute. That’s dopamine, not edge.")
            if nrfi_count > 0:
                free_body_lines.append("Top parlay theme today is NRFI. That’s not edge, that’s a support group.")
            if hr_count > 0:
                free_body_lines.append("HR ladders everywhere. When fireworks trend, mispriced Unders sneak by.")
            free_body_lines.append(f"Parlay Surge: {int(surge.get('parlay_surge_score',0))}  |  Early Factor: {int(nrfi_last.get('nrfi_factor', nrfi_last.get('early_factor', 0)))}  |  PI Base: {pib_val:.1f}")
            if ss.get("viug_summ"):
                free_body_lines.append(f"Refs/Pace overlay: VI_med {vi_med:.2f} | UG_med {ug_med:.2f} | Adj {overlay_adj:+.1f}")
            free_body_lines.append("If you want picks, pay for the edges. Free feed is for education and humiliation.")
            free_post = f"**{free_title}**\n\n" + "\n".join([f"• {ln}" for ln in free_body_lines]) + "\n\nCTA: Unlock the card."

            # Build Paid Post copy (actual angles)
            paid_title = "Today’s Card — YRFI & Unders Where The Market Shows Respect"
            paid_sections = []
            if not respected_unders.empty:
                sample_u = respected_unders.head(3)[[c for c in ["matchup","side","odds","handle_pct","bets_pct"] if c in respected_unders.columns]]
                paid_sections.append("**Respected Unders**\n" + sample_u.to_markdown(index=False))
            if not sharp_steam.empty:
                sample_s = sharp_steam.head(3)[[c for c in ["matchup","market","side","odds","handle_pct","bets_pct"] if c in sharp_steam.columns]]
                paid_sections.append("**Sharp Steam Alignments**\n" + sample_s.to_markdown(index=False))
            paid_post = f"**{paid_title}**\n\n" + "\n\n".join(paid_sections) if paid_sections else "**Card posts after lines are verified.**"

            st.write("**Free Post (copy-paste)**")
            st.text_area("Free Post", value=free_post, height=220)
            st.write("**Paid Post (copy-paste)**")
            st.text_area("Paid Post", value=paid_post, height=280)

            # ---------- Export pack ----------
            overlay_source = ss.get("overlay_source_name")
            overlay_sample = []
            try:
                if ss.get("overlay_rows"):
                    overlay_sample = (pd.DataFrame(ss["overlay_rows"]).head(25)).to_dict(orient="records")
            except Exception:
                overlay_sample = []
            export_payload = {
                "metrics": {
                    "pi_base": pib_val,
                    "parlay_surge": int(surge.get("parlay_surge_score", 0)),
                    "nrfi_factor": int(nrfi_last.get("nrfi_factor", 0)),
                    "public_index": public_index_preview,
                },
                "tables": {
                    "public_love": public_love.to_dict(orient="records") if not public_love.empty else [],
                    "sharp_steam": sharp_steam.to_dict(orient="records") if not sharp_steam.empty else [],
                    "public_overs": public_overs.to_dict(orient="records") if not public_overs.empty else [],
                    "respected_unders": respected_unders.to_dict(orient="records") if not respected_unders.empty else [],
                    "teaser_legs": teaser_legs,
                },
                "parlay_themes": {
                    "nrfi_legs": nrfi_count,
                    "yrfi_legs": yrfi_count,
                    "hr_legs": hr_count,
                    "tb_legs": tb_count,
                    "hits_legs": hits_count,
                    "top_titles": top_titles if parlays else []
                },
                "copy_blocks": {
                    "free_post": free_post,
                    "paid_post": paid_post
                },
                "overlays": {
                    "viug_summary": ss.get("viug_summ") or {},
                    "splits_enriched_preview": (pd.DataFrame(ss["splits_enriched"]).head(25).to_dict(orient="records") if ss.get("splits_enriched") else []),
                },
            }
            st.download_button("Download Content Pack (JSON)", data=json.dumps(export_payload, indent=2), file_name="content_pack.json", mime="application/json")

    # ---------------- Advanced (optional) ----------------
    with adv:
        st.subheader("Advanced: Upload custom splits JSON and compute Parlay Surge manually")
        st.caption("Use only if you are testing alternate sources. Otherwise, do Step 2.")
        uploaded_file = st.file_uploader("Upload Market Mood splits JSON file", type=["json"], key="splits_uploader_adv")
        if uploaded_file is not None:
            try:
                splits_json = json.load(uploaded_file)
                if isinstance(splits_json, list):
                    games_list = splits_json
                elif isinstance(splits_json, dict):
                    path = st.text_input("JSON path to games list (dot separated)", value="")
                    current = splits_json
                    if path.strip():
                        for k in path.strip().split("."):
                            if isinstance(current, dict):
                                current = current.get(k, [])
                            else:
                                current = []
                                break
                    games_list = current if isinstance(current, list) else []
                else:
                    games_list = []

                st.write(f"Detected {len(games_list)} entries")
                if games_list:
                    df_custom = pd.DataFrame(games_list)
                    st.dataframe(df_custom.head(10), use_container_width=True)
                    keys = list(df_custom.columns)
                    fav_handle_key = st.selectbox("Favorite Handle % key", options=keys, key="adv_fav_key")
                    over_handle_key = st.selectbox("Over Handle % key", options=keys, key="adv_over_key")

                    if st.button("Compute Parlay Surge (custom)", use_container_width=True):
                        def normalize_pct(vals):
                            out = []
                            for v in vals:
                                try:
                                    fv = float(v)
                                    if fv > 1.0:
                                        fv = fv / 100.0
                                    fv = max(0.0, min(1.0, fv))
                                    out.append(fv)
                                except Exception:
                                    out.append(0.0)
                            return out

                        fav_series = normalize_pct(df_custom[fav_handle_key].fillna(0).tolist())
                        over_series = normalize_pct(df_custom[over_handle_key].fillna(0).tolist())
                        surge = compute_parlay_surge_from_handles(fav_series, over_series)
                        st.success(f"Parlay Surge {surge['parlay_surge_score']} • HHI Sides {surge['hhi_sides']:.4f} • HHI Totals {surge['hhi_totals']:.4f} • n_corr {surge['n_corr']}")
            except Exception as e:
                st.error(f"Error loading JSON: {e}")

def main():
    ap = argparse.ArgumentParser(description="Scrape DKNetwork: Most popular parlays today")
    ap.add_argument("--url", default=DEFAULT_URL, help="DKNetwork parlay page URL")
    ap.add_argument("--top", type=int, default=10, help="Top N parlays to output")
    ap.add_argument("--out", default="", help="Write JSON to file (default stdout)")
    ap.add_argument("--include", default="", help="Substring filter (e.g., 'MLB', 'NRFI', team or player name)")
    ap.add_argument("--min-bet-count", type=int, default=0, help="Only include parlays with Bet Count >= this")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose stderr logs")
    ap.add_argument("--props", action="store_true", help="Fetch DKNetwork player props instead of parlays")
    ap.add_argument("--event-group", type=int, default=84240, help="DK tb_eg (e.g., 84240 = MLB)")
    ap.add_argument("--edate", type=str, default="today", choices=["today","yesterday","last_7","last_30"], help="DK tb_edate range")
    args = ap.parse_args()

    if args.props:
        html = fetch_props_html(event_group=int(args.event_group), edate=args.edate, use_playwright=False)
        props = parse_player_props(html)
        payload = {"source": build_props_url(args.event_group, args.edate), "as_of_epoch": int(time.time()), "props": props}
        data = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(data)
            print(f"[ok] wrote {args.out} ({len(props)} items)")
        else:
            print(data)
        return

    html = fetch_html(args.url)
    cards = parse_cards(html, verbose=args.verbose)
    cards = filter_cards(cards, args.include or None, args.min_bet_count)

    if args.top > 0:
        cards = cards[: args.top]

    payload = {
        "source": args.url,
        "as_of_epoch": int(time.time()),
        "top": cards
    }

    data = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"[ok] wrote {args.out} ({len(cards)} items)")
    else:
        print(data)

# Standalone entry: if executed directly, run the Streamlit UI when available,
# otherwise fall back to the CLI scraper mode.
if __name__ == "__main__":
    if "streamlit" in sys.modules and st is not None and pd is not None:
        render(None)  # uses st.session_state
    else:
        main()