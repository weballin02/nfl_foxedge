# app_nfl_pricer.py (ENHANCED & CORRECTED)
# Streamlit UI for NFL fair pricer (DK Network splits, event group 88808)
# Fixed critical issues: pandas API, error handling, validation, edge cases

from __future__ import annotations
import math, re, logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import io

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import pytz

from streamlit_autorefresh import st_autorefresh

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Page setup & CSS --------------------
st.set_page_config(
    page_title="NFL Fair Pricer | DK Splits",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Global palette */
:root {
  --bg:#0A0A0A; --panel:#121212; --ink:#E8ECEF; --muted:#A7B0B8;
  --teal:#00E6B8; --red:#FF1A3D; --gold:#FFD700; --accent:#8AE6FF; --green:#18C97E;
}

html, body, [class*="appview-container"] { background: var(--bg) !important; color: var(--ink) !important; }
section.main > div { padding-top: 0.5rem; }

/* Headline */
.hdr {
  font-family: "Bebas Neue", system-ui, -apple-system, Segoe UI, Roboto, Arial;
  letter-spacing: 0.5px; text-transform: uppercase; color: var(--ink);
  display: flex; align-items: center; gap: .6rem; margin: .25rem 0 1rem 0;
}
.hdr .pill { background: linear-gradient(90deg, var(--teal), var(--accent));
  color: #001a17; padding: .2rem .5rem; border-radius: 6px; font-weight: 800; }

/* KPI cards */
.kpi {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.3));
  border: 1px solid rgba(255,255,255,0.06);
  padding: 0.9rem 1rem; border-radius: 12px; height: 100%;
}
.kpi h3 { margin: 0; font-size: .9rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .6px; }
.kpi .val { font-size: 1.8rem; font-weight: 900; margin-top: .2rem; }
.kpi .sub { font-size: .9rem; color: var(--muted); }

/* Matchup card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.25));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px; padding: 1rem; margin-bottom: 0.75rem;
}
.card h4 { margin: 0 0 .45rem 0; font-size: 1.1rem; letter-spacing: .4px; }
.badge { display: inline-block; padding: .12rem .5rem; border-radius: 6px; font-weight: 700; letter-spacing: .4px; font-size: .8rem; margin-right: .35rem;}
.badge.ev { background: rgba(0,230,184,0.12); border: 1px solid rgba(0,230,184,0.35); color: var(--teal); }
.badge.neg { background: rgba(255,26,61,0.10); border: 1px solid rgba(255,26,61,0.25); color: var(--red); }
.badge.bet { background: rgba(24,201,126,0.16); border: 1px solid rgba(24,201,126,0.35); color: var(--green); }
.badge.pass { background: rgba(255,215,0,0.12); border: 1px solid rgba(255,215,0,0.35); color: var(--gold); }

/* Dataframe chrome */
.stDataFrame { border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; }

/* Sidebar polish */
.css-1d391kg, .stSidebar { background: var(--panel) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Math utils --------------------
def american_to_decimal(american: int) -> float:
    if american >= 100:
        return 1.0 + american / 100.0
    if american <= -100:
        return 1.0 + 100.0 / abs(american)
    raise ValueError(f"Bad american odds: {american}")

def american_to_prob(american: int) -> float:
    return 1.0 / american_to_decimal(american)

def prob_to_american(p: float) -> int:
    p = max(1e-9, min(1 - 1e-9, p))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

def remove_vig_two_way(p_a: float, p_b: float) -> Tuple[float, float]:
    s = p_a + p_b
    if s <= 0:
        return 0.5, 0.5
    return p_a / s, p_b / s

def kelly_fraction_decimal(p: float, dec: float) -> float:
    b = dec - 1.0
    if b <= 0:
        return 0.0
    frac = (p * (b + 1) - 1) / b
    return max(0.0, frac)

def logit(x: float) -> float:
    x = min(max(x, 1e-9), 1 - 1e-9)
    return math.log(x / (1 - x))

def logistic(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

# -------------------- Push probability model (NFL) --------------------

def is_integerish(x: Optional[float], tol: float = 1e-9) -> bool:
    if x is None or pd.isna(x):
        return False
    return abs(float(x) - round(float(x))) < tol

# Empirical push frequencies (bucketed)
EMP_SPREAD_PUSH = {
    3: {"low": 0.090, "mid": 0.080, "high": 0.070},
    7: {"low": 0.055, "mid": 0.050, "high": 0.045},
    10: {"low": 0.032, "mid": 0.030, "high": 0.028},
    6: {"low": 0.022, "mid": 0.020, "high": 0.018},
    4: {"low": 0.017, "mid": 0.015, "high": 0.014},
    14: {"low": 0.011, "mid": 0.010, "high": 0.009},
}

EMP_TOTAL_PUSH = {
    37: {"low": 0.030, "mid": 0.028, "high": 0.026},
    41: {"low": 0.032, "mid": 0.030, "high": 0.028},
    44: {"low": 0.031, "mid": 0.030, "high": 0.029},
    47: {"low": 0.029, "mid": 0.028, "high": 0.027},
    49: {"low": 0.028, "mid": 0.027, "high": 0.026},
}

EMP_BUCKET_THRESHOLDS = {"low": 42.0, "high": 49.0}

def _bucket_for_total(total_line: Optional[float]) -> str:
    if total_line is None or pd.isna(total_line):
        return "mid"
    t = float(total_line)
    if t < EMP_BUCKET_THRESHOLDS["low"]:
        return "low"
    if t > EMP_BUCKET_THRESHOLDS["high"]:
        return "high"
    return "mid"

@st.cache_data(show_spinner=False)
def load_empirical_overrides(file: Optional[bytes]) -> Tuple[dict, dict]:
    """
    Accept a CSV with columns: kind(spread/total), key(int), bucket(low/mid/high), push_prob.
    Returns (spread_dict, total_dict). Validates required columns and value ranges.
    """
    if not file:
        logger.info("No empirical override file; using defaults")
        return EMP_SPREAD_PUSH, EMP_TOTAL_PUSH
    
    try:
        df = pd.read_csv(io.BytesIO(file))
        
        # Validate required columns
        required_cols = {"kind", "key", "bucket", "push_prob"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.error(f"CSV missing columns: {missing_cols}")
            st.error(f"❌ CSV missing required columns: {missing_cols}")
            return EMP_SPREAD_PUSH, EMP_TOTAL_PUSH
        
        s = {}
        t = {}
        rows_processed = 0
        rows_invalid = 0
        
        for idx, r in df.iterrows():
            try:
                kind = str(r.get("kind", "")).strip().lower()
                key = int(r.get("key"))
                bucket = str(r.get("bucket", "mid")).strip().lower()
                val = float(r.get("push_prob"))
                
                # Validate push_prob is in [0, 1]
                if not (0 <= val <= 1):
                    logger.warning(f"Row {idx}: push_prob={val} out of range [0,1]. Skipping.")
                    rows_invalid += 1
                    continue
                
                # Validate kind and bucket
                if kind not in ("spread", "total"):
                    logger.warning(f"Row {idx}: kind='{kind}' not in (spread, total). Skipping.")
                    rows_invalid += 1
                    continue
                
                if bucket not in ("low", "mid", "high"):
                    logger.warning(f"Row {idx}: bucket='{bucket}' not in (low, mid, high). Skipping.")
                    rows_invalid += 1
                    continue
                
                if kind == "spread":
                    s.setdefault(key, {})[bucket] = val
                else:
                    t.setdefault(key, {})[bucket] = val
                
                rows_processed += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Row {idx}: Failed to parse ({e}). Skipping.")
                rows_invalid += 1
                continue
        
        logger.info(f"Loaded {rows_processed} empirical rows ({rows_invalid} invalid)")
        return (s or EMP_SPREAD_PUSH), (t or EMP_TOTAL_PUSH)
    
    except Exception as e:
        logger.error(f"Failed to load empirical file: {e}")
        st.error(f"❌ Error loading CSV: {e}")
        return EMP_SPREAD_PUSH, EMP_TOTAL_PUSH

def spread_push_prob(home_line: Optional[float], consensus_total: Optional[float], emp_spread: dict) -> float:
    """
    Return push probability for a spread. Half-point lines get minimal push prob.
    """
    if home_line is None:
        return 0.0
    
    # Half-point lines rarely push; use conservative default
    if not is_integerish(home_line):
        return 0.001
    
    key = int(abs(round(float(home_line))))
    bucket = _bucket_for_total(consensus_total)
    
    if key in emp_spread and bucket in emp_spread[key]:
        return float(emp_spread[key][bucket])
    
    # Fallback: generic estimate based on line size
    if key <= 3:
        return 0.05
    elif key <= 6:
        return 0.02
    else:
        return 0.01

def total_push_prob(total_line: Optional[float], emp_total: dict) -> float:
    """Return push probability for a total line."""
    if total_line is None:
        return 0.0
    
    # Half-point totals very rarely push
    if not is_integerish(total_line):
        return 0.001
    
    key = int(round(float(total_line)))
    bucket = _bucket_for_total(total_line)
    
    if key in emp_total and bucket in emp_total[key]:
        return float(emp_total[key][bucket])
    
    # Fallback: conservative generic estimate
    return 0.030

# Utilities to convert with push

def fair_decimal_from_winlose(p_win: float, p_push: float) -> float:
    """Return fair decimal odds given win/push probabilities (loss = 1 - p_win - p_push)."""
    p_win = float(p_win)
    p_push = float(p_push)
    p_lose = max(0.0, 1.0 - p_win - p_push)
    if p_win <= 0:
        return np.inf
    return 1.0 + (p_lose / p_win)

def american_from_decimal(dec: float) -> int:
    if not np.isfinite(dec) or dec <= 1.0:
        return 1000000  # effectively unbettable
    b = dec - 1.0
    if dec >= 2.0:
        return int(round(100 * b))
    else:
        return int(round(-100 / b))

def kelly_with_push(dec: float, p_win: float, p_push: float) -> float:
    """Kelly fraction for a bet with push probability."""
    b = dec - 1.0
    if b <= 0:
        return 0.0
    p_loss = max(0.0, 1.0 - p_win - p_push)
    frac = (b * p_win - p_loss) / b
    return float(max(0.0, frac))

def consensus_total_for_matchup(full_df: pd.DataFrame, matchup: str) -> Optional[float]:
    """Extract consensus total line for a matchup (median of TOTAL market)."""
    try:
        tot_lines = full_df[
            (full_df["matchup"] == matchup) & (full_df["market"] == "Total")
        ]["line"].dropna()
        if tot_lines.empty:
            return None
        return float(np.median(tot_lines.astype(float)))
    except Exception as e:
        logger.warning(f"Could not compute consensus total for {matchup}: {e}")
        return None

def clean_odds(s: str) -> int:
    s = (s or "").strip().replace("\u2212", "-")
    m = re.search(r"(-?\+?\d+)", s)
    if not m:
        raise ValueError(f"Cannot parse odds: {s}")
    val = int(m.group(1))
    if 0 < val < 100:
        val = -val
    return val

def parse_side_and_line(side_text: str) -> Tuple[str, Optional[float]]:
    s = (side_text or "").strip().lower()
    if "home" in s:
        return "HOME", None
    if "away" in s:
        return "AWAY", None
    m = re.search(r"(over|under)\s*([0-9]+(?:\.[0-9])?)", s)
    if m:
        return m.group(1).upper(), float(m.group(2))
    m2 = re.search(r"(home|away)\s*([+\-]?\d+(?:\.\d)?)", s)
    if m2:
        side = "HOME" if m2.group(1) == "home" else "AWAY"
        pts = float(m2.group(2))
        return side, pts
    return "", None

def infer_home_away_from_text(side_text: str, away_team: str, home_team: str) -> str:
    """Infer HOME/AWAY by matching side text against team names."""
    s = (side_text or "").lower()
    ah = (away_team or "").lower()
    hh = (home_team or "").lower()
    
    if hh and hh in s:
        return "HOME"
    if ah and ah in s:
        return "AWAY"
    
    def first_token(name: str) -> str:
        parts = re.split(r"[^a-z0-9]+", (name or "").lower())
        return parts[0] if parts else ""
    
    f_h = first_token(home_team)
    f_a = first_token(away_team)
    if f_h and f_h in s:
        return "HOME"
    if f_a and f_a in s:
        return "AWAY"
    return ""

# -------------------- Scraper --------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch_dk_splits(event_group: int = 88808, date_token: str = "today", _nonce: int | None = None) -> pd.DataFrame:
    """
    Fetch DK Network betting splits for NFL. Uses Playwright for first page (robust rendering),
    then requests for subsequent pages. Parses Moneyline, Spread, Total rows.
    Returns DataFrame or falls back to tomorrow/n7days if empty.
    Enhanced with error logging.
    """
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlencode, urlparse, parse_qs

    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_token, "tb_emt": "0"}
    first_url = f"{base}?{urlencode(params)}"

    def _clean(text: str) -> str:
        return re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", text or "", flags=re.I).strip()

    def _get_html(url: str) -> str:
        """Try Playwright for first page; fallback to requests."""
        if url == first_url:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    page.wait_for_selector("div.tb-se", timeout=10000)
                    html = page.content()
                    browser.close()
                    logger.info("Playwright render successful")
                    return html
            except Exception as e:
                logger.warning(f"Playwright failed: {type(e).__name__}: {e}. Falling back to requests.")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Language": "en-US,en;q=0.9"
            }
            resp = requests.get(url, headers=headers, timeout=25)
            resp.raise_for_status()
            logger.info(f"Requests fetch successful ({len(resp.text)} bytes)")
            return resp.text
        except requests.RequestException as e:
            logger.error(f"Requests failed for {url}: {e}")
            return ""

    def _discover_page_urls(html: str) -> list[str]:
        if not html:
            return [first_url]
        soup = BeautifulSoup(html, "html.parser")
        urls = {first_url}
        pag = soup.select_one("div.tb_pagination")
        if pag:
            for a in pag.find_all("a", href=True):
                href = a["href"]
                if "tb_page=" in href:
                    urls.add(href)
        
        def pnum(u: str) -> int:
            try:
                return int(parse_qs(urlparse(u).query).get("tb_page", ["1"])[0])
            except Exception:
                return 1
        
        return sorted(urls, key=pnum)

    def _parse_page(html: str) -> list[dict]:
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        games = soup.select("div.tb-se")
        out: list[dict] = []
        pac = pytz.timezone("America/Los_Angeles")
        now = datetime.now(pac)
        
        for g in games:
            title_el = g.select_one("div.tb-se-title h5")
            if not title_el:
                logger.debug("Skipping game with no title element")
                continue
            
            title = _clean(title_el.get_text(strip=True))
            time_el = g.select_one("div.tb-se-title span")
            game_time = _clean(time_el.get_text(strip=True)) if time_el else ""
            
            for section in g.select(".tb-market-wrap > div"):
                head = section.select_one(".tb-se-head > div")
                if not head:
                    logger.debug(f"Skipping section with no header in {title}")
                    continue
                
                market_name = _clean(head.get_text(strip=True))
                if market_name not in ("Moneyline", "Total", "Spread"):
                    continue
                
                for row in section.select(".tb-sodd"):
                    side_el = row.select_one(".tb-slipline")
                    odds_el = row.select_one("a.tb-odd-s")
                    
                    if not side_el or not odds_el:
                        logger.debug(f"Skipping row in {title}/{market_name} (missing side/odds)")
                        continue
                    
                    side_txt = _clean(side_el.get_text(strip=True))
                    oddstxt = _clean(odds_el.get_text(strip=True))
                    
                    # Parse odds
                    try:
                        odds_val = int(oddstxt.replace("−", "-"))
                    except ValueError:
                        try:
                            odds_val = int(float(oddstxt))
                        except ValueError:
                            logger.debug(f"Could not parse odds '{oddstxt}' in {title}/{market_name}")
                            continue
                    
                    # Extract percentages
                    pct_texts = [
                        s.strip().replace("%", "")
                        for s in row.find_all(string=lambda t: "%" in (t or ""))
                    ]
                    handle_pct, bets_pct = (pct_texts + ["", ""])[:2]
                    
                    try:
                        handle_pct_f = float(handle_pct or 0)
                        bets_pct_f = float(bets_pct or 0)
                    except ValueError:
                        handle_pct_f, bets_pct_f = 0.0, 0.0
                    
                    # Parse line if present
                    line_val = None
                    if market_name == "Spread":
                        m = re.search(r"([+\-]?\d+(?:\.5)?)", side_txt)
                        if m:
                            try:
                                line_val = float(m.group(1))
                            except ValueError:
                                pass
                    elif market_name == "Total":
                        m = re.search(r"(Over|Under)\s+([0-9]+(?:\.[0-9])?)", side_txt, re.I)
                        if m:
                            try:
                                line_val = float(m.group(2))
                            except ValueError:
                                pass
                    
                    out.append({
                        "matchup": title,
                        "game_time": game_time,
                        "market": market_name,
                        "side": side_txt,
                        "odds": odds_val,
                        "pct_handle": handle_pct_f,
                        "pct_bets": bets_pct_f,
                        "update_time": now,
                        "line": line_val,
                    })
        
        logger.info(f"Parsed {len(out)} rows from page")
        return out

    # Fetch first page and discover all pages
    first_html = _get_html(first_url)
    urls = _discover_page_urls(first_html)
    logger.info(f"Discovered {len(urls)} page(s)")

    # Parse all pages
    records: list[dict] = []
    for u in urls:
        html = first_html if u == first_url else _get_html(u)
        records.extend(_parse_page(html))

    # Fallback chain if empty
    if not records:
        logger.warning(f"No records for date_token={date_token}. Falling back...")
        if date_token == "today":
            return fetch_dk_splits(event_group, "tomorrow", _nonce=_nonce)
        if date_token == "tomorrow":
            return fetch_dk_splits(event_group, "n7days", _nonce=_nonce)
        logger.error("Exhausted fallback chain; returning empty DataFrame")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "matchup", "game_time", "market", "side", "odds",
                "pct_handle", "pct_bets", "update_time", "line"
            ]
        )
    
    logger.info(f"Returning DataFrame with {len(df)} rows")
    return df

# -------------------- Normalization & pricing --------------------
def split_matchup(title: str) -> Tuple[str, str]:
    t = title.replace("@", " at ")
    parts = re.split(r"\s+at\s+|\s+vs\.?\s+|\s+v\s+", t, flags=re.I)
    if len(parts) != 2:
        return title, ""
    return parts[0].strip(), parts[1].strip()

def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    away_home = df["matchup"].apply(split_matchup)
    df["away_team"] = away_home.apply(lambda x: x[0])
    df["home_team"] = away_home.apply(lambda x: x[1])
    
    df["key"] = (
        df["matchup"].astype(str) + "|" +
        df["market"].astype(str) + "|" +
        df["side"].astype(str) + "|" +
        df["odds"].astype(str)
    )
    df = df.drop_duplicates(subset="key").drop(columns=["key"])
    
    if "line" not in df.columns:
        df["line"] = None
    
    return df

@dataclass
class PriceRow:
    matchup: str
    game_time: str
    market: str
    selection: str
    line: Optional[float]
    dk_price: int
    fair_prob: float
    fair_price: int
    edge: float
    ev: float
    kelly: float
    notes: str

def adjust_with_splits(p: float, pct_handle: Optional[float], pct_bets: Optional[float], weight: float) -> float:
    if weight == 0 or pct_handle is None or pct_bets is None:
        return p
    diff = (pct_handle - pct_bets) / 100.0
    z = logit(p) + weight * diff
    return logistic(z)

def price_moneyline(group: pd.DataFrame, w: float, kcap: float) -> List[PriceRow]:
    out = []
    home_name = str(group.get("home_team").iloc[0]) if "home_team" in group.columns and not group.empty else ""
    away_name = str(group.get("away_team").iloc[0]) if "away_team" in group.columns and not group.empty else ""

    def classify(row_side: str) -> str:
        s = (row_side or "").lower()
        if "home" in s:
            return "HOME"
        if "away" in s:
            return "AWAY"
        return infer_home_away_from_text(row_side, away_name, home_name)

    sides: Dict[str, pd.Series] = {}
    for _, r in group.iterrows():
        lab = classify(str(r.get("side")))
        if lab in ("HOME", "AWAY"):
            sides[lab] = r
    
    if "HOME" not in sides or "AWAY" not in sides:
        logger.debug(f"Skipping moneyline group {group.get('matchup').iloc[0]}: incomplete sides")
        return out

    home, away = sides["HOME"], sides["AWAY"]
    p_h_raw = american_to_prob(int(home["odds"]))
    p_a_raw = american_to_prob(int(away["odds"]))
    p_h0, p_a0 = remove_vig_two_way(p_h_raw, p_a_raw)
    p_h = adjust_with_splits(p_h0, home.get("pct_handle"), home.get("pct_bets"), w)
    p_a = adjust_with_splits(p_a0, away.get("pct_handle"), away.get("pct_bets"), w)

    for sel, row, p0, p in (("HOME", home, p_h0, p_h), ("AWAY", away, p_a0, p_a)):
        dec = american_to_decimal(int(row["odds"]))
        ev = p * (dec - 1.0) - (1 - p)
        out.append(PriceRow(
            matchup=row["matchup"],
            game_time=row["game_time"],
            market="ML",
            selection=sel,
            line=None,
            dk_price=int(row["odds"]),
            fair_prob=p,
            fair_price=prob_to_american(p),
            edge=p - p0,
            ev=ev,
            kelly=min(kcap, kelly_fraction_decimal(p, dec)),
            notes=f"baseline={p0:.3f}; splits_w={w:.3f}"
        ))
    return out

def price_total(group: pd.DataFrame, w: float, kcap: float, emp_total: dict) -> List[PriceRow]:
    out: List[PriceRow] = []
    buckets: Dict[float, Dict[str, pd.Series]] = {}
    
    for _, r in group.iterrows():
        side, line = parse_side_and_line(str(r.get("side")))
        line = r.get("line", line)
        if side in ("OVER", "UNDER") and line is not None:
            line_key = round(float(line), 1)
            buckets.setdefault(line_key, {})[side] = r
    
    for line, d in buckets.items():
        if "OVER" not in d or "UNDER" not in d:
            logger.debug(f"Skipping total {line}: missing OVER or UNDER")
            continue
        
        over, under = d["OVER"], d["UNDER"]
        p_over_raw = american_to_prob(int(over["odds"]))
        p_under_raw = american_to_prob(int(under["odds"]))
        p_over0, p_under0 = remove_vig_two_way(p_over_raw, p_under_raw)
        p_over = adjust_with_splits(p_over0, over.get("pct_handle"), over.get("pct_bets"), w)
        p_under = adjust_with_splits(p_under0, under.get("pct_handle"), under.get("pct_bets"), w)
        
        q_push = total_push_prob(line, emp_total)
        
        for sel, row, p0, p_sel in (
            ("OVER", over, p_over0, p_over),
            ("UNDER", under, p_under0, p_under),
        ):
            p_win = max(0.0, p_sel * (1.0 - q_push))
            p_loss = max(0.0, (1.0 - p_sel) * (1.0 - q_push))
            dec_fair = fair_decimal_from_winlose(p_win, q_push)
            fair_price = american_from_decimal(dec_fair)
            dec_market = american_to_decimal(int(row["odds"]))
            ev = p_win * (dec_market - 1.0) - p_loss
            kelly = min(kcap, kelly_with_push(dec_market, p_win, q_push))
            
            out.append(PriceRow(
                matchup=row["matchup"],
                game_time=row["game_time"],
                market="TOTAL",
                selection=sel,
                line=float(line),
                dk_price=int(row["odds"]),
                fair_prob=p_sel,
                fair_price=fair_price,
                edge=p_sel - p0,
                ev=ev,
                kelly=kelly,
                notes=f"baseline={p0:.3f}; splits_w={w:.3f}; q_push={q_push:.3f}"
            ))
    
    return out

def price_spread(group: pd.DataFrame, w: float, kcap: float, full_df: pd.DataFrame, emp_spread: dict) -> List[PriceRow]:
    """
    Pair spread sides robustly and price with push probability.
    Handles team name inference and normalizes to HOME line convention.
    """
    out: List[PriceRow] = []
    buckets: Dict[float, Dict[str, pd.Series]] = {}

    home_name = str(group.get("home_team").iloc[0]) if "home_team" in group.columns and not group.empty else ""
    away_name = str(group.get("away_team").iloc[0]) if "away_team" in group.columns and not group.empty else ""
    matchup_name = str(group["matchup"].iloc[0]) if not group.empty else ""

    for _, r in group.iterrows():
        side, line = parse_side_and_line(str(r.get("side")))
        line = r.get("line", line)
        
        if side not in ("HOME", "AWAY"):
            side = infer_home_away_from_text(str(r.get("side")), away_name, home_name)
        
        if line is None or side not in ("HOME", "AWAY"):
            logger.debug(f"Skipping spread row: missing line or valid side ({side})")
            continue
        
        try:
            pts = float(line)
        except ValueError:
            logger.debug(f"Could not parse line '{line}'")
            continue
        
        # Normalize to HOME line (home favorite = negative)
        home_line = pts if side == "HOME" else -pts
        home_line = round(home_line, 1)
        
        buckets.setdefault(home_line, {})[side] = r

    cons_total = consensus_total_for_matchup(full_df, matchup_name)

    for home_line, d in buckets.items():
        if "HOME" not in d or "AWAY" not in d:
            logger.debug(f"Skipping spread {home_line}: missing HOME or AWAY")
            continue
        
        home, away = d["HOME"], d["AWAY"]
        p_h_raw = american_to_prob(int(home["odds"]))
        p_a_raw = american_to_prob(int(away["odds"]))
        p_h0, p_a0 = remove_vig_two_way(p_h_raw, p_a_raw)
        p_h = adjust_with_splits(p_h0, home.get("pct_handle"), home.get("pct_bets"), w)
        p_a = adjust_with_splits(p_a0, away.get("pct_handle"), away.get("pct_bets"), w)

        q_push = spread_push_prob(home_line, cons_total, emp_spread)

        for sel, row, p0, p, line_val in (
            ("HOME", home, p_h0, p_h, home_line),
            ("AWAY", away, p_a0, p_a, -home_line),
        ):
            p_win = max(0.0, p * (1.0 - q_push))
            p_loss = max(0.0, (1.0 - p) * (1.0 - q_push))
            dec_fair = fair_decimal_from_winlose(p_win, q_push)
            fair_price = american_from_decimal(dec_fair)
            dec_market = american_to_decimal(int(row["odds"]))
            ev = p_win * (dec_market - 1.0) - p_loss
            kelly = min(kcap, kelly_with_push(dec_market, p_win, q_push))
            
            out.append(PriceRow(
                matchup=row["matchup"],
                game_time=row["game_time"],
                market="SPREAD",
                selection=sel,
                line=float(0.0 if abs(line_val) < 1e-9 else line_val),
                dk_price=int(row["odds"]),
                fair_prob=p,
                fair_price=fair_price,
                edge=p - p0,
                ev=ev,
                kelly=kelly,
                notes=f"baseline={p0:.3f}; splits_w={w:.3f}; q_push={q_push:.3f}"
            ))
    
    return out

def choose_recs(rows: List[PriceRow], ev_side_min: float, ev_total_min: float) -> pd.DataFrame:
    """Filter and rank recommendations."""
    if not rows:
        return pd.DataFrame()
    
    recs = []
    for r in rows:
        ev_min = ev_side_min if r.market in ("ML", "SPREAD") else ev_total_min
        rec = "BET" if (r.ev > 0 and abs(r.edge) >= ev_min) else "PASS"
        
        recs.append({
            "matchup": r.matchup,
            "game_time": r.game_time,
            "market": r.market,
            "selection": r.selection,
            "line": r.line,
            "dk_price": r.dk_price,
            "fair_prob": round(r.fair_prob, 4),
            "fair_price": r.fair_price,
            "edge": round(r.edge, 4),
            "ev": round(r.ev, 4),
            "kelly": round(r.kelly, 4),
            "rec": rec,
            "notes": r.notes
        })
    
    df = pd.DataFrame(recs)
    cat = pd.Categorical(df["rec"], categories=["BET", "PASS"], ordered=True)
    df = (
        df.assign(_rec=cat)
        .sort_values(["_rec", "ev"], ascending=[True, False])
        .drop(columns=["_rec"])
        .reset_index(drop=True)
    )
    return df

# -------------------- Sidebar controls --------------------
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    
    date_token = st.selectbox(
        "Date",
        ["today", "tomorrow", "yesterday", "this week"],
        index=0,
        help="DK Network may ignore filters; parser reads visible games."
    )
    if date_token == "this week":
        date_token = "today"
    
    event_group = st.number_input("DK Event Group (NFL)", value=88808, step=1)
    if not (1 <= event_group <= 999999):
        st.error("❌ Event group must be 1–999999")
        st.stop()
    
    splits_weight = st.slider(
        "Splits tilt weight (logit)",
        min_value=-0.5,
        max_value=0.5,
        value=0.12,
        step=0.01,
        help="Positive: favor higher handle. Negative: contrarian."
    )
    
    ev_side_min = st.slider("Min edge for sides (prob delta)", 0.0, 0.03, 0.005, 0.001)
    ev_total_min = st.slider("Min edge for totals (prob delta)", 0.0, 0.03, 0.005, 0.001)
    kelly_cap = st.slider("Kelly cap", 0.0, 0.5, 0.20, 0.01)
    show_pass = st.toggle("Show PASS rows", value=False)
    
    st.divider()
    
    search_team = st.text_input("Filter by team (search)", value="")
    market_filter = st.multiselect(
        "Markets",
        ["ML", "SPREAD", "TOTAL"],
        default=["ML", "SPREAD", "TOTAL"]
    )
    
    st.divider()
    st.markdown("**Empirical push overrides (optional CSV)**")
    emp_file = st.file_uploader(
        "Upload push frequencies CSV",
        type=["csv"],
        help="Columns: kind(spread/total), key(int), bucket(low/mid/high), push_prob"
    )
    
    st.divider()
    auto_refresh = st.toggle("Auto refresh", value=True)
    refresh_sec = st.slider("Refresh interval (sec)", 15, 180, 60, 5)
    st.caption("Tip: Set weight negative for contrarian. Keep thresholds tight.")

# -------------------- Load empirical tables --------------------
emp_spread_tbl, emp_total_tbl = load_empirical_overrides(
    emp_file.read() if emp_file is not None else None
)

# -------------------- Auto-refresh & fetch --------------------
refresh_count = 0
if auto_refresh:
    refresh_count = st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")
    st.sidebar.markdown(f"🔄 Refresh Count: {refresh_count}")

try:
    with st.spinner("Scraping DK Network…"):
        raw = fetch_dk_splits(event_group, date_token, _nonce=refresh_count if auto_refresh else None)
except Exception as e:
    logger.error(f"Fetch failed: {e}")
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

# -------------------- Velocity calculation --------------------
now_ts = datetime.now(pytz.timezone("America/Los_Angeles"))
raw["ts"] = now_ts
raw["_key"] = (
    raw["matchup"].astype(str) + "|" +
    raw["market"].astype(str) + "|" +
    raw["side"].astype(str) + "|" +
    raw["line"].astype(str)
)

# Use session state with stable, unique key
snapshot_key = f"prev_snapshot_{event_group}"
prev = st.session_state.get(snapshot_key)

if prev is not None and not prev.empty:
    df_prev = prev.copy()
    df_prev = df_prev[["_key", "pct_handle", "pct_bets", "ts"]].rename(
        columns={
            "pct_handle": "pct_handle_prev",
            "pct_bets": "pct_bets_prev",
            "ts": "ts_prev"
        }
    )
    raw = raw.merge(df_prev, on="_key", how="left")
    
    # Calculate velocity (handle/bets per minute), handle zero-minute case
    raw["mins"] = (raw["ts"] - raw["ts_prev"]).dt.total_seconds() / 60.0
    raw["mins"] = raw["mins"].replace(0, np.nan)  # Avoid div by zero
    raw["handle_vpm"] = (raw["pct_handle"] - raw["pct_handle_prev"]) / raw["mins"]
    raw["bets_vpm"] = (raw["pct_bets"] - raw["pct_bets_prev"]) / raw["mins"]
else:
    raw["handle_vpm"] = np.nan
    raw["bets_vpm"] = np.nan

st.session_state[snapshot_key] = raw[["_key", "pct_handle", "pct_bets", "ts"]].copy()

# -------------------- Normalize & price --------------------
df = normalize_rows(raw)

for col in ("handle_vpm", "bets_vpm"):
    if col not in df.columns:
        df[col] = np.nan

rows: List[PriceRow] = []

for matchup, g in df[df["market"] == "Moneyline"].groupby("matchup"):
    rows.extend(price_moneyline(g, splits_weight, kelly_cap))

for matchup, g in df[df["market"] == "Total"].groupby("matchup"):
    rows.extend(price_total(g, splits_weight, kelly_cap, emp_total_tbl))

for matchup, g in df[df["market"] == "Spread"].groupby("matchup"):
    rows.extend(price_spread(g, splits_weight, kelly_cap, df, emp_spread_tbl))

rec_df = choose_recs(rows, ev_side_min, ev_total_min)

# -------------------- Join velocity data --------------------
if not rec_df.empty:
    jkey = ["matchup", "market", "selection", "line"]
    
    def side_label(row):
        s = str(row.get("side"))
        if row.get("market") == "Moneyline":
            lab = infer_home_away_from_text(s, row.get("away_team", ""), row.get("home_team", ""))
            return lab
        if row.get("market") == "Spread":
            lab = infer_home_away_from_text(s, row.get("away_team", ""), row.get("home_team", ""))
            return lab
        if row.get("market") == "Total":
            return "OVER" if "over" in s.lower() else ("UNDER" if "under" in s.lower() else s)
        return s
    
    tmp = df.copy()
    tmp["selection"] = tmp.apply(side_label, axis=1)
    tmp = tmp[["matchup", "market", "selection", "line", "handle_vpm", "bets_vpm"]]
    tmp = tmp.drop_duplicates(subset=jkey)  # Avoid duplicate row expansion on merge
    
    rec_df = rec_df.merge(tmp, on=jkey, how="left")

# -------------------- Filter results --------------------
if not show_pass and not rec_df.empty:
    rec_df = rec_df[rec_df["rec"] == "BET"]

if market_filter and not rec_df.empty:
    rec_df = rec_df[rec_df["market"].isin(market_filter)]

if search_team.strip() and not rec_df.empty:
    q = search_team.strip().lower()
    rec_df = rec_df[rec_df["matchup"].str.lower().str.contains(q)]

# -------------------- Display: Header & KPIs --------------------
st.markdown(
    '<div class="hdr"><span class="pill">NFL</span><h1 style="margin:0">Fair Pricer · DK Splits</h1></div>',
    unsafe_allow_html=True
)

col_k1, col_k2, col_k3, col_k4 = st.columns([1, 1, 1, 1])

with col_k1:
    val = len(rec_df[rec_df["rec"] == "BET"]) if not rec_df.empty else 0
    st.markdown(
        f'<div class="kpi"><h3>Actionable Bets</h3><div class="val">{val}</div><div class="sub">EV > 0 & edge ≥ min</div></div>',
        unsafe_allow_html=True
    )

with col_k2:
    avg_ev = rec_df["ev"].mean() if not rec_df.empty else 0.0
    st.markdown(
        f'<div class="kpi"><h3>Avg EV</h3><div class="val">{avg_ev:.3f}</div><div class="sub">Across visible rows</div></div>',
        unsafe_allow_html=True
    )

with col_k3:
    top = rec_df.sort_values("ev", ascending=False).head(1)
    top_ev = float(top["ev"].iloc[0]) if not top.empty else 0.0
    st.markdown(
        f'<div class="kpi"><h3>Top EV</h3><div class="val">{top_ev:.3f}</div><div class="sub">Highest value on board</div></div>',
        unsafe_allow_html=True
    )

with col_k4:
    ts = raw["update_time"].max() if not raw.empty else None
    ts_str = ts.strftime("%-m/%-d %I:%M %p %Z") if isinstance(ts, datetime) else "n/a"
    st.markdown(
        f'<div class="kpi"><h3>Data Timestamp</h3><div class="val">{ts_str}</div><div class="sub">Pacific time</div></div>',
        unsafe_allow_html=True
    )

st.divider()

# -------------------- Display: Main table & controls --------------------
lc, rc = st.columns([2, 1], vertical_alignment="top")

with lc:
    st.subheader("Recommended Bets (ranked)")
    if rec_df.empty:
        st.info("ℹ️ No actionable rows with current thresholds/filters.")
    else:
        view_cols = [
            "matchup", "game_time", "market", "selection", "line",
            "dk_price", "fair_price", "fair_prob", "edge", "ev", "kelly",
            "handle_vpm", "bets_vpm", "rec"
        ]
        grid = rec_df[view_cols].copy()
        
        # Style with corrected .map() instead of deprecated .applymap()
        def color_ev(val):
            if pd.isna(val):
                return ""
            if val >= 0.01:
                return "background-color: rgba(0,230,184,0.15); color:#00E6B8;"
            if val <= -0.005:
                return "background-color: rgba(255,26,61,0.12); color:#FF1A3D;"
            return ""
        
        def color_rec(val):
            if val == "BET":
                return "background-color: rgba(24,201,126,0.16); color:#18C97E; font-weight:700;"
            return "background-color: rgba(255,215,0,0.12); color:#FFD700;"
        
        styler = (
            grid.style
            .map(color_ev, subset=["edge", "ev"])
            .map(color_rec, subset=["rec"])
        )
        st.dataframe(styler, use_container_width=True, height=420)

with rc:
    st.subheader("Download")
    if not rec_df.empty:
        csv = rec_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export CSV",
            csv,
            file_name="nfl_fair_pricer_recs.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.subheader("Quick Notes")
    st.markdown("""
- **De-vig baseline** from DK prices; **splits tilt** nudges fair probs by handle vs tickets.
- **EV** = p·(dec−1) − (1−p). **Kelly** capped by sidebar.
- Set **weight < 0** to force contrarian bias.
- **Push prob** accounted for spreads & totals; half-points use conservative defaults.
""")

st.divider()

# -------------------- Display: Matchup cards --------------------
st.subheader("Matchup Detail")
if rec_df.empty:
    st.caption("Nothing to show. Loosen filters or lower thresholds.")
else:
    for matchup, sub in rec_df.groupby("matchup"):
        gtime = sub["game_time"].iloc[0] if "game_time" in sub else ""
        c1, c2, c3 = st.columns([3, 2, 2])
        
        with c1:
            st.markdown(
                f'<div class="card"><h4>{matchup}</h4><div class="badge">Kickoff: {gtime}</div></div>',
                unsafe_allow_html=True
            )
        
        with c2:
            side_rows = sub[sub["market"].isin(["ML", "SPREAD"])].sort_values("ev", ascending=False)
            if not side_rows.empty:
                r = side_rows.iloc[0]
                bclass = "bet" if r["rec"] == "BET" else "pass"
                line_str = "" if pd.isna(r["line"]) else f"{r['line']:+g}"
                handle_badge = "" if pd.isna(r.get("handle_vpm")) else f"<div class='badge'>Δ Handle {r['handle_vpm']:+.2f}%/min</div>"
                bets_badge = "" if pd.isna(r.get("bets_vpm")) else f"<div class='badge'>Δ Bets {r['bets_vpm']:+.2f}%/min</div>"
                
                st.markdown(
                    f"<div class='card'><h4>Best Side</h4>"
                    f"<div class='badge {bclass}'>{r['market']} {r['selection']} {line_str}</div>"
                    f"<div class='badge ev'>EV {r['ev']:.3f}</div> "
                    f"<div class='badge'>Kelly {r['kelly']:.3f}</div> "
                    f"<div class='badge'>DK {r['dk_price']:+}</div> "
                    f"<div class='badge'>Fair {r['fair_price']:+}</div>"
                    f"{handle_badge}{bets_badge}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown("<div class='card'><h4>Best Side</h4><div class='badge pass'>None</div></div>", unsafe_allow_html=True)
        
        with c3:
            tot_rows = sub[sub["market"] == "TOTAL"].sort_values("ev", ascending=False)
            if not tot_rows.empty:
                t = tot_rows.iloc[0]
                bclass = "bet" if t["rec"] == "BET" else "pass"
                line_str_t = "" if pd.isna(t["line"]) else f"{t['line']:.1f}"
                handle_badge_t = "" if pd.isna(t.get("handle_vpm")) else f"<div class='badge'>Δ Handle {t['handle_vpm']:+.2f}%/min</div>"
                bets_badge_t = "" if pd.isna(t.get("bets_vpm")) else f"<div class='badge'>Δ Bets {t['bets_vpm']:+.2f}%/min</div>"
                
                st.markdown(
                    f"<div class='card'><h4>Best Total</h4>"
                    f"<div class='badge {bclass}'>{t['selection']} {line_str_t}</div> "
                    f"<div class='badge ev'>EV {t['ev']:.3f}</div> "
                    f"<div class='badge'>Kelly {t['kelly']:.3f}</div> "
                    f"<div class='badge'>DK {t['dk_price']:+}</div> "
                    f"<div class='badge'>Fair {t['fair_price']:+}</div>"
                    f"{handle_badge_t}{bets_badge_t}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="card"><h4>Best Total</h4><div class="badge pass">None</div></div>',
                    unsafe_allow_html=True
                )
        
        with st.expander(f"All markets · {matchup}", expanded=False):
            st.dataframe(
                sub.sort_values(["rec", "market", "ev"], ascending=[True, True, False]).reset_index(drop=True),
                use_container_width=True,
                height=220
            )

# -------------------- Footer --------------------
st.markdown("---")
st.caption(
    "Source: DK Network betting splits. Event Group 88808 (NFL). "
    "Scraper handles pagination automatically. "
    "Enhanced with validation, error logging, and push probability accounting."
)