# nfl_odds_analysis.py
# Auto-generated unified app that merges "market_mood_nfl.py" and "do_not_bet_nfl.py"
# into a single Streamlit multi-tab application without altering core logic.
# Both original bodies are wrapped into functions and exposed via tabs.
# Imports are deduplicated; page config is set once here.

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime, timedelta
import pytz
import altair as alt
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import json  # for markdown snippet generation
import qrcode
import logging
from streamlit_autorefresh import st_autorefresh
from urllib.parse import urlencode, urlparse, parse_qs
from playwright.sync_api import sync_playwright
import re
import textwrap
import math
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
import datetime, pytz
import datetime
from scipy.stats import zscore
from xhtml2pdf import pisa
from scipy.stats import rankdata
import io
import random

import streamlit as st

st.set_page_config(page_title="NFL Odds Analysis", page_icon="🏈", layout="wide")

# ---------------- Wrapped Apps ----------------

def render_market_mood():


    # ---- Implied Probability Helpers ----
    def implied_prob_from_odds(odds):
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        elif odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 0.5

    def implied_prob_from_spread(odds):
        """Dummy: Use ML implied for now (placeholder for spread conversion)."""
        return implied_prob_from_odds(odds)

    def implied_prob_from_total(odds):
        """Dummy: Use ML implied for now (placeholder for total conversion)."""
        return implied_prob_from_odds(odds)

    # ---- Config & Constants ----
    HISTORY_FILE = "mood_history.csv"
    SPLITS_HISTORY_FILE = "splits_history.csv"
    GATE_THRESHOLD = 60  # Mood score threshold for gating hint

    # ---- Pro Signal Thresholds (tuned for top handicappers) ----
    EXPOSURE_THRESHOLD = 2000       # minimum liability in $ to flag exposure
    ODDLOT_THRESHOLD = 0.10         # 10% ticket/handle divergence
    CORR_THRESHOLD = 0.02           # 2% correlation breach between markets

    # ---- Reverse Line Movement & Steam Move Thresholds ----
    RLM_BET_THRESHOLD = 5.0        # %bets increase threshold for RLM
    STEAM_ODDS_THRESHOLD = 3       # absolute odds change for Steam Move

    

    # --- bet_logs absolute path config (shared with mlb/claude/grokpitch/grokhit) ----
    BETLOGS_ROOT = os.getenv("BETLOGS_ROOT", os.getcwd())
    BETLOGS_DB = os.path.join(BETLOGS_ROOT, "bet_logs.db")

    def _ensure_parent(path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass

    def clean_odds(odds_str):
        try:
            # support unicode minus
            return int(str(odds_str).replace("−", "-").strip())
        except Exception:
            try:
                return int(float(odds_str))
            except Exception:
                return None

    def _parse_total_from_side(side_text: str):
        """
        From strings like 'Over 8.5' or 'Under 9', return (side, total_float).
        Returns (None, None) if not a totals side.
        """
        try:
            s = str(side_text).strip()
            if s.lower().startswith("over ") or s.lower().startswith("under "):
                parts = s.split()
                if len(parts) >= 2:
                    return parts[0].title(), float(parts[1])
        except Exception:
            pass
        return None, None

    # --- Spread parsing helper
    def _parse_spread_from_side(side_text: str):
        """
        From strings like 'Eagles -3.5' or 'Cowboys +2', return float spread value.
        Returns None if not a spread side.
        """
        try:
            s = str(side_text).strip()
            parts = s.split()
            # typical DK splits format: '<TeamName> <+/-number>'
            if len(parts) >= 2:
                last = parts[-1].replace("−", "-")
                if last.startswith(("+", "-")):
                    return float(last)
        except Exception:
            pass
        return None

    # ---------- Logging helpers (schema consumed by proof_maker.py) ----------
    def log_blog_pick_to_db(pick_data: dict, db_path: str = BETLOGS_DB):
        _ensure_parent(db_path)
        try:
            with sqlite3.connect(db_path) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS blog_pick_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_date TEXT,
                        matchup TEXT,
                        bet_type TEXT,
                        confidence TEXT,
                        edge_pct REAL,
                        odds REAL,
                        predicted_total REAL,
                        predicted_winner TEXT,
                        predicted_margin REAL,
                        bookmaker_total REAL,
                        analysis TEXT
                    );
                """)
                cols = ["log_date","matchup","bet_type","confidence","edge_pct","odds",
                        "predicted_total","predicted_winner","predicted_margin","bookmaker_total","analysis"]
                row = [pick_data.get(k) for k in cols]
                con.execute(f"INSERT INTO blog_pick_logs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", row)
        except Exception as e:
            logging.error(f"[market_mood.log_blog_pick_to_db] {e}")

    def log_bet_card_to_db(card: dict, db_path: str = BETLOGS_DB):
        _ensure_parent(db_path)
        try:
            with sqlite3.connect(db_path) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS bet_card_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id TEXT,
                        home_team TEXT,
                        away_team TEXT,
                        game_time TEXT,
                        combined_runs REAL,
                        delta REAL,
                        total_bet_rec TEXT,
                        ml_bet_rec TEXT,
                        over_edge REAL,
                        under_edge REAL,
                        home_ml_edge REAL,
                        away_ml_edge REAL,
                        home_ensemble_pred REAL,
                        away_ensemble_pred REAL,
                        combined_ensemble_pred REAL,
                        ensemble_confidence TEXT,
                        statcast_home_exit REAL,
                        statcast_home_angle REAL,
                        statcast_away_exit REAL,
                        statcast_away_angle REAL,
                        pitcher_home_exit REAL,
                        pitcher_home_angle REAL,
                        pitcher_away_exit REAL,
                        pitcher_away_angle REAL,
                        weather_home_temp REAL,
                        weather_home_wind REAL,
                        weather_away_temp REAL,
                        weather_away_wind REAL,
                        home_pitcher TEXT,
                        away_pitcher TEXT,
                        bookmaker_line REAL,
                        over_price REAL,
                        under_price REAL,
                        home_ml_book REAL,
                        away_ml_book REAL,
                        log_date TEXT
                    );
                """)
                defaults = {
                    "game_id": None, "home_team": None, "away_team": None, "game_time": None,
                    "combined_runs": None, "delta": None, "total_bet_rec": None, "ml_bet_rec": None,
                    "over_edge": None, "under_edge": None, "home_ml_edge": None, "away_ml_edge": None,
                    "home_ensemble_pred": None, "away_ensemble_pred": None, "combined_ensemble_pred": None,
                    "ensemble_confidence": "splits:mood", "statcast_home_exit": None, "statcast_home_angle": None,
                    "statcast_away_exit": None, "statcast_away_angle": None, "pitcher_home_exit": None,
                    "pitcher_home_angle": None, "pitcher_away_exit": None, "pitcher_away_angle": None,
                    "weather_home_temp": None, "weather_home_wind": None, "weather_away_temp": None,
                    "weather_away_wind": None, "home_pitcher": None, "away_pitcher": None, "bookmaker_line": None,
                    "over_price": None, "under_price": None, "home_ml_book": None, "away_ml_book": None,
                    "log_date": datetime.utcnow().isoformat()
                }
                payload = defaults | {k: card.get(k) for k in defaults.keys()}
                cols = list(defaults.keys())
                row = [payload[k] for k in cols]
                con.execute(f"INSERT INTO bet_card_logs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", row)
        except Exception as e:
            logging.error(f"[market_mood.log_bet_card_to_db] {e}")

    def log_snapshot_row(game_id, matchup, market, side, book, snapshot_type, odds_val=None, total_val=None, spread_val=None, db_path: str = BETLOGS_DB):
        _ensure_parent(db_path)
        try:
            with sqlite3.connect(db_path) as con:
                # Defensive column add for back-compat
                try:
                    cols = pd.read_sql_query("PRAGMA table_info(line_snapshots)", con)
                    if "spread" not in cols["name"].tolist():
                        con.execute("ALTER TABLE line_snapshots ADD COLUMN spread REAL;")
                except Exception:
                    pass
                con.execute("""
                    CREATE TABLE IF NOT EXISTS line_snapshots (
                        game_id        TEXT,
                        matchup        TEXT,
                        market         TEXT,
                        side           TEXT,
                        book           TEXT,
                        snapshot_type  TEXT,   -- 'OPEN' | 'CLOSE' | 'YOUR_REC'
                        odds           INTEGER,
                        total          REAL,
                        spread         REAL,
                        timestamp_utc  TEXT,
                        PRIMARY KEY (game_id, market, side, book, snapshot_type)
                    );
                """)
                ts = datetime.utcnow().isoformat()
                con.execute("""
                    INSERT OR IGNORE INTO line_snapshots
                    (game_id, matchup, market, side, book, snapshot_type, odds, total, spread, timestamp_utc)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (str(game_id), str(matchup), str(market), str(side), str(book), str(snapshot_type),
                      None if odds_val is None else int(odds_val),
                      None if total_val is None else float(total_val),
                      None if spread_val is None else float(spread_val),
                      ts))
        except Exception as e:
            logging.error(f"[market_mood.log_snapshot_row] {e}")

    # ---- CLOSE/YOUR_REC snapshot and CLV helpers ----
    def log_close_snapshot(df):
        """
        Write CLOSE snapshots for each row in the current DraftKings splits DataFrame.
        Mirrors log_snapshot(...) but uses snapshot_type='CLOSE'.
        """
        for _, r in df.iterrows():
            matchup = r.get('matchup')
            market = r.get('market')
            side = r.get('side')
            odds = clean_odds(r.get('odds'))
            total_val = None; spread_val = None
            mkt = str(market).lower()
            if mkt.startswith("total"):
                ou, tot = _parse_total_from_side(side)
                total_val = tot
            elif mkt.startswith("spread"):
                spread_val = _parse_spread_from_side(side)
            date_iso = datetime.now(pytz.timezone("America/Los_Angeles")).date().isoformat()
            gid = f"{date_iso}_{str(matchup).replace(' ','_')}_{market}_{str(side).replace(' ','_')}"[:128]
            log_snapshot_row(gid, matchup, market, side, "DK", "CLOSE", odds_val=odds, total_val=total_val, spread_val=spread_val)

    def log_your_rec_snapshot(matchup, market, side, odds_val=None, total_val=None, spread_val=None):
        """
        Record a 'YOUR_REC' snapshot for a specific market side at the moment you decide to recommend it.
        """
        date_iso = datetime.now(pytz.timezone("America/Los_Angeles")).date().isoformat()
        gid = f"{date_iso}_{str(matchup).replace(' ','_')}_{market}_{str(side).replace(' ','_')}"[:128]
        log_snapshot_row(gid, matchup, market, side, "DK", "YOUR_REC", odds_val=odds_val, total_val=total_val, spread_val=spread_val)

    def compute_and_log_clv(db_path: str = BETLOGS_DB) -> pd.DataFrame:
        """
        Compute Closing Line Value (CLV) for every (game_id, market, side, book) that has both
        a YOUR_REC and a CLOSE snapshot. Logs results into a `clv_logs` table and returns a DataFrame.
        CLV metrics:
          - clv_prob_pp: change in implied probability (close - entry) in percentage points based on odds
          - clv_line_move: for totals only, close_total - entry_total with side-aware sign
          - clv_spread_move: for spreads, close_spread - entry_spread with sign correction
        """
        with sqlite3.connect(db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS clv_logs (
                    game_id TEXT,
                    matchup TEXT,
                    market TEXT,
                    side TEXT,
                    book TEXT,
                    entry_odds INTEGER,
                    close_odds INTEGER,
                    entry_total REAL,
                    close_total REAL,
                    entry_spread REAL,
                    close_spread REAL,
                    clv_prob_pp REAL,
                    clv_line_move REAL,
                    clv_spread_move REAL,
                    computed_utc TEXT,
                    PRIMARY KEY (game_id, market, side, book)
                );
            """)
            snaps = pd.read_sql_query("SELECT * FROM line_snapshots", con)
        if snaps.empty:
            return pd.DataFrame(columns=["game_id","matchup","market","side","book","entry_odds","close_odds","entry_total","close_total","entry_spread","close_spread","clv_prob_pp","clv_line_move","clv_spread_move","computed_utc"])

        snaps = snaps[snaps["book"] == "DK"].copy()
        keys = ["game_id","matchup","market","side","book"]
        entry = snaps[snaps["snapshot_type"] == "YOUR_REC"][keys + ["odds","total","spread"]].rename(columns={"odds":"entry_odds","total":"entry_total","spread":"entry_spread"})
        close  = snaps[snaps["snapshot_type"] == "CLOSE"][keys + ["odds","total","spread"]].rename(columns={"odds":"close_odds","total":"close_total","spread":"close_spread"})
        merged = entry.merge(close, on=keys, how="inner")

        def _impl_prob(odds_val):
            try:
                o = int(odds_val)
            except Exception:
                return None
            if o is None:
                return None
            if o > 0:
                return 100.0 / (o + 100.0)
            return (-o) / (100.0 - o)

        def _clv_prob_pp(row):
            eo = row.get("entry_odds")
            co = row.get("close_odds")
            if eo is None or co is None:
                return None
            pe = _impl_prob(eo)
            pc = _impl_prob(co)
            if pe is None or pc is None:
                return None
            return round(pc - pe, 3)

        def _clv_line_move(row):
            if str(row.get("market","")).lower().startswith("total"):
                et = row.get("entry_total")
                ct = row.get("close_total")
                if et is None or ct is None:
                    return None
                side = str(row.get("side","")).lower()
                move = ct - et
                if side.startswith("over"):
                    return round(et - ct, 3)   # positive = moved in our favor
                if side.startswith("under"):
                    return round(ct - et, 3)
                return round(move, 3)
            return None

        def _clv_spread_move(row):
            if str(row.get("market","")).lower().startswith("spread"):
                es = row.get("entry_spread")
                cs = row.get("close_spread")
                if es is None or cs is None:
                    return None
                sign = -1.0 if es < 0 else 1.0
                return round(sign * (cs - es), 3)
            return None

        if merged.empty:
            return pd.DataFrame(columns=["game_id","matchup","market","side","book","entry_odds","close_odds","entry_total","close_total","entry_spread","close_spread","clv_prob_pp","clv_line_move","clv_spread_move","computed_utc"])

        merged["clv_prob_pp"] = merged.apply(_clv_prob_pp, axis=1)
        merged["clv_line_move"] = merged.apply(_clv_line_move, axis=1)
        merged["clv_spread_move"] = merged.apply(_clv_spread_move, axis=1)
        merged["computed_utc"] = datetime.utcnow().isoformat()

        with sqlite3.connect(db_path) as con:
            rows = merged[["game_id","matchup","market","side","book","entry_odds","close_odds","entry_total","close_total","entry_spread","close_spread","clv_prob_pp","clv_line_move","clv_spread_move","computed_utc"]].to_records(index=False)
            con.executemany("""
                INSERT INTO clv_logs (game_id, matchup, market, side, book, entry_odds, close_odds, entry_total, close_total, entry_spread, close_spread, clv_prob_pp, clv_line_move, clv_spread_move, computed_utc)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(game_id, market, side, book) DO UPDATE SET
                    entry_odds=excluded.entry_odds,
                    close_odds=excluded.close_odds,
                    entry_total=excluded.entry_total,
                    close_total=excluded.close_total,
                    entry_spread=excluded.entry_spread,
                    close_spread=excluded.close_spread,
                    clv_prob_pp=excluded.clv_prob_pp,
                    clv_line_move=excluded.clv_line_move,
                    clv_spread_move=excluded.clv_spread_move,
                    computed_utc=excluded.computed_utc
            """, list(rows))
        return merged

    # ---- Custom FoxEdge Styling ----
    st.markdown(
        """<style>
        .css-1lcbmhc {background: linear-gradient(90deg, #0e0f11, #1f1f23);}
        .stMetric {border: 2px solid #ff3c78; border-radius: 8px; padding: 12px; background-color: #1a1a1d;}
        .stProgress > div > div > div > div {background: #ff3c78;}
        </style>""",
        unsafe_allow_html=True
    )
    st.markdown(
        """<style>
        .caption-box p {white-space: pre-wrap; word-break: break-word; max-width: 700px; margin: 0 auto;}
        .button-link {background-color: #ff3c78; color: #ffffff; padding: 10px 20px; border-radius:6px; text-decoration: none; font-weight: bold; display: inline-block; margin-top: 10px;}
        .chart-title {color: #ffffff; font-weight: bold; font-size: 18px;}
        .axis-label {color: #ffffff !important;}
        #MainMenu, footer {visibility: hidden;}
        </style>""",
        unsafe_allow_html=True
    )
    # ---- Global Expander Header Style ----
    st.markdown(
        """<style>
        /* Color expander headers uniformly to dark background */
        .streamlit-expanderHeader {
            background-color: #1f1f23 !important;
            border-radius: 4px;
            padding: 5px;
        }
        </style>""",
        unsafe_allow_html=True
    )

    # ---- Page Navigation ----
    page = st.sidebar.radio("Select View", ["Overview", "Game Details"])

    # Refresh interval selector
    refresh_interval = st.sidebar.slider(
        "Auto-refresh interval (sec)", min_value=10, max_value=300, value=60, step=5
    )

    # Pro-signal threshold sliders
    EXPOSURE_THRESHOLD = st.sidebar.slider(
        "Exposure Liability Threshold ($)", min_value=0, max_value=10000, value=EXPOSURE_THRESHOLD, step=100
    )
    ODDLOT_THRESHOLD = st.sidebar.slider(
        "Odd-Lot Divergence Threshold (%)", min_value=0.0, max_value=0.5, value=ODDLOT_THRESHOLD, step=0.01, format="%.2f"
    )
    CORR_THRESHOLD = st.sidebar.slider(
        "Correlation Breach Threshold (%)", min_value=0.0, max_value=0.1, value=CORR_THRESHOLD, step=0.005, format="%.3f"
    )
    RLM_BET_THRESHOLD = st.sidebar.slider(
        "RLM % Bets Δ Threshold", min_value=0.0, max_value=20.0, value=RLM_BET_THRESHOLD, step=0.5, format="%.1f"
    )
    STEAM_ODDS_THRESHOLD = st.sidebar.slider(
        "Steam Odds Δ Threshold", min_value=0, max_value=10, value=STEAM_ODDS_THRESHOLD, step=1
    )

    # ---- Auto-Refresh ----
    refresh_count = st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
    st.sidebar.markdown(f"🔄 Refresh Count: {refresh_count}")
    # Force re-fetch on each refresh
    st.session_state["loaded_snapshot"] = False

    # ---- Database Setup (lazy; helpers manage tables) ----
    st.caption(f"bet_logs target → {BETLOGS_DB}")

    # ---- Data Fetching ----
    def fetch_dk_splits(event_group: int = 88808, date_range: str = "today", market: str = "All") -> pd.DataFrame:
        """
        Fetch DKNetwork betting splits for the given event_group/date_range across
        ALL pagination pages. Keeps the original output schema.

        - Discovers pagination links in `div.tb_pagination`
        - Always includes page 1
        - Uses Playwright for the first page (if available), then requests for others
        - Parses Moneyline, Total, and Spread sections
        - If `today` is empty, automatically retries `tomorrow`
        - If `today` and `tomorrow` are empty, automatically retries `n7days` (DK's next-7-days window)
        """

        base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
        params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
        first_url = f"{base}?{urlencode(params)}"

        def clean(text: str) -> str:
            return re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", text or "", flags=re.I).strip()

        def _get_html(url: str) -> str:
            """Try Playwright for the first page; fallback to requests."""
            if url == first_url:
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page()
                        page.goto(url)
                        page.wait_for_selector("div.tb-se", timeout=10000)
                        html = page.content()
                        browser.close()
                        return html
                except Exception:
                    pass
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, headers=headers, timeout=20)
                resp.raise_for_status()
                return resp.text
            except Exception:
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
                    continue
                title = clean(title_el.get_text(strip=True))
                time_el = g.select_one("div.tb-se-title span")
                game_time = clean(time_el.get_text(strip=True)) if time_el else ""
                for section in g.select(".tb-market-wrap > div"):
                    head = section.select_one(".tb-se-head > div")
                    if not head:
                        continue
                    market_name = clean(head.get_text(strip=True))
                    if market_name not in ("Moneyline", "Total", "Spread"):
                        continue
                    for row in section.select(".tb-sodd"):
                        side_el = row.select_one(".tb-slipline")
                        odds_el = row.select_one("a.tb-odd-s")
                        if not side_el or not odds_el:
                            continue
                        side = clean(side_el.get_text(strip=True))
                        oddstxt = clean(odds_el.get_text(strip=True))
                        try:
                            odds_val = int(oddstxt.replace("−", "-"))
                        except Exception:
                            try:
                                odds_val = int(float(oddstxt))
                            except Exception:
                                odds_val = 0
                        pct_texts = [s.strip().replace("%", "") for s in row.find_all(string=lambda t: "%" in t)]
                        handle_pct, bets_pct = (pct_texts + ["", ""])[:2]
                        spread_val = None
                        if market_name == "Spread":
                            spread_val = _parse_spread_from_side(side)
                        out.append({
                            "matchup": title,
                            "game_time": game_time,
                            "market": market_name,
                            "side": side,
                            "odds": odds_val,
                            "spread": spread_val,
                            "%handle": float(handle_pct or 0),
                            "%bets": float(bets_pct or 0),
                            "update_time": now,
                        })
            return out

        # 1) Fetch first page and discover all page urls
        first_html = _get_html(first_url)
        all_urls = _discover_page_urls(first_html)

        # 2) Parse all pages
        records: list[dict] = []
        for url in all_urls:
            html = first_html if url == first_url else _get_html(url)
            records.extend(_parse_page(html))

        # 3) Fallbacks: today -> tomorrow -> next 7 days
        if not records:
            if date_range == "today":
                return fetch_dk_splits(event_group, "tomorrow", market)
            if date_range == "tomorrow":
                return fetch_dk_splits(event_group, "n7days", market)

        return pd.DataFrame(records, columns=[
            "matchup","game_time","market","side","odds","spread","%handle","%bets","update_time"
        ])

    def log_snapshot(df):
        """
        Write OPEN snapshots for each row in the DraftKings splits DataFrame.
        For 'Total' market, parse the numeric total; for 'Moneyline', store odds.
        """
        for _, r in df.iterrows():
            matchup = r.get('matchup')
            market = r.get('market')
            side = r.get('side')
            odds = clean_odds(r.get('odds'))
            total_val = None; spread_val = None
            mkt = str(market).lower()
            if mkt.startswith("total"):
                ou, tot = _parse_total_from_side(side)
                total_val = tot
            elif mkt.startswith("spread"):
                spread_val = _parse_spread_from_side(side)
            # deterministic ID per matchup/side/market on date
            date_iso = datetime.now(pytz.timezone("America/Los_Angeles")).date().isoformat()
            gid = f"{date_iso}_{str(matchup).replace(' ','_')}_{market}_{str(side).replace(' ','_')}"[:128]
            log_snapshot_row(gid, matchup, market, side, "DK", "OPEN", odds_val=odds, total_val=total_val, spread_val=spread_val)

    # ---- Mood Trend Chart ----
    def market_mood_ring(df):
        df["irrationality"] = abs(df["%bets"] - df["%handle"])
        mood_score = df["irrationality"].mean()
        hist = update_history(mood_score)
        sparkline = (
            alt.Chart(hist)
               .mark_line(point=True)
               .encode(x='date:T', y='score:Q')
               .properties(width=700, height=100)
        )
        st.subheader("Mood Trend (Last 7 days)")
        st.altair_chart(sparkline, use_container_width=True)

    # ---- Game Details Page ----
    def show_game_details(df):
        # Compute overall mood score for this dataset
        df["irrationality"] = abs(df["%bets"] - df["%handle"])
        mood_score = df["irrationality"].mean()
        st.header("Game Details")
        # Top outlier alerts
        game_scores = df.groupby("matchup")["irrationality"].mean()
        top_irr_matchup = game_scores.idxmax()
        top_irr_value = game_scores.max()
        fade_scores = df.groupby("matchup")["%bets"].max()
        top_fade_matchup = fade_scores.idxmax()
        top_fade_value = fade_scores.max()
        st.markdown("---")
        st.markdown(f"**🔥 Top Irrationality:** {top_irr_matchup} — {top_irr_value:.1f}%")
        st.markdown(f"**📉 Top Fade Candidate:** {top_fade_matchup} — {top_fade_value:.1f}%")
        st.markdown("---")
        tabs = st.tabs(["Irrationality", "Public Fade Ratio"])

        with tabs[0]:
            st.markdown("---")
            st.subheader("Game-by-Game Irrationality")
            for matchup in df["matchup"].unique():
                sub = df[df["matchup"] == matchup].copy()
                sub["irrationality"] = (sub["%bets"] - sub["%handle"]).abs()
                game_score = sub["irrationality"].mean()
                # Choose icon color based on irrationality level
                if game_score >= 20:
                    icon = "🔴"
                elif game_score >= 15:
                    icon = "🟠"
                elif game_score >= 10:
                    icon = "🟡"
                else:
                    icon = "🟢"
                with st.expander(f"{icon} {matchup}", expanded=False):
                    display_df = sub[["market", "side", "%bets", "%handle", "irrationality"]].rename(
                        columns={
                            "market": "Bet Type",
                            "side": "Side",
                            "%bets": "% Bets",
                            "%handle": "% Handle",
                            "irrationality": "Irrationality"
                        }
                    )
                    st.dataframe(display_df, use_container_width=True)
                    # Compute and display a game-level gauge
                    st.metric(label="Avg Irrationality", value=f"{game_score:.1f}%")
                    gauge_data = pd.DataFrame({"value": [game_score]})
                    gauge_chart = (
                        alt.Chart(gauge_data)
                           .mark_arc(innerRadius=40, outerRadius=60)
                           .encode(
                               theta=alt.Theta('value:Q', scale=alt.Scale(domain=[0,100])),
                               color=alt.Color(
                                   'value:Q',
                                   scale=alt.Scale(
                                       domain=[0,25,50,75,100],
                                       range=['#00ff00','#ffff00','#ffa500','#ff0000','#ff3c78']
                                   )
                               )
                           )
                           .properties(width=150, height=150)
                    )
                    st.altair_chart(gauge_chart, use_container_width=False)
                    # Donut chart for game-level irrationality (pink/dark)
                    rec_game = pd.DataFrame({
                        "value": [game_score, 100 - game_score],
                        "type": ["Irrational", "Rational"]
                    })
                    donut_game = (
                        alt.Chart(rec_game)
                           .mark_arc(innerRadius=20, outerRadius=40)
                           .encode(
                               theta=alt.Theta("value:Q"),
                               color=alt.Color("type:N", scale=alt.Scale(range=['#ff3c78','#001f3f']))
                           )
                           .properties(width=100, height=100)
                    )
                    st.altair_chart(donut_game, use_container_width=False)
                    bar_game = (
                        "<div style='background:#333;padding:2px;border-radius:4px;'>"
                        f"<div style='width:{game_score}%;background:#ff3c78;height:10px;border-radius:4px;'></div>"
                        "</div>"
                    )
                    st.markdown(bar_game, unsafe_allow_html=True)
                    # Narrative insight
                    if game_score > 10:
                        callout = "Sharp money detected—line movers in action."
                    elif game_score >= 6:
                        callout = "Moderate skew—consider fading the public."
                    elif game_score >= 3:
                        callout = "Mild noise—nothing dramatic."
                    else:
                        callout = "Balanced market—no clear edge."
                    st.markdown(f"**Insight:** {callout}")
            st.markdown("---")

        with tabs[1]:
            st.markdown("---")
            st.subheader("Game-by-Game Public Fade Ratio")
            for matchup in df["matchup"].unique():
                sub = df[df["matchup"] == matchup].copy()
                # use the maximum %bets to measure fade skew
                fade_score = sub["%bets"].max()
                # choose icon color based on fade skew
                if fade_score > 75:
                    icon = "🔴"
                elif fade_score >= 66:
                    icon = "🟠"
                elif fade_score >= 55:
                    icon = "🟡"
                else:
                    icon = "🟢"
                with st.expander(f"{icon} {matchup}", expanded=False):
                    # summary metric & gauge
                    st.metric(label="Max Public Bet %", value=f"{fade_score:.1f}%")
                    gauge_data = pd.DataFrame({"value":[fade_score]})
                    gauge_chart = (
                        alt.Chart(gauge_data)
                           .mark_arc(innerRadius=40, outerRadius=60)
                           .encode(
                               theta=alt.Theta('value:Q', scale=alt.Scale(domain=[0,100])),
                               color=alt.Color(
                                   'value:Q',
                                   scale=alt.Scale(
                                       domain=[0,25,50,75,100],
                                       range=['#00ff00','#ffff00','#ffa500','#ff0000','#ff3c78']
                                   )
                               )
                           )
                           .properties(width=150, height=150)
                    )
                    st.altair_chart(gauge_chart, use_container_width=False)
                    # pink/dark donut + bar
                    rec_fade = pd.DataFrame({
                        "value":[fade_score, 100-fade_score],
                        "type":["Public","Other"]
                    })
                    donut_fade = (
                        alt.Chart(rec_fade)
                           .mark_arc(innerRadius=20, outerRadius=40)
                           .encode(
                               theta=alt.Theta("value:Q"),
                               color=alt.Color("type:N", scale=alt.Scale(range=['#001f3f','#888888']))
                           )
                           .properties(width=100, height=100)
                    )
                    st.altair_chart(donut_fade, use_container_width=False)
                    bar_fade = (
                        "<div style='background:#333;padding:2px;border-radius:4px;'>"
                        f"<div style='width:{fade_score}%;background:#001f3f;height:10px;border-radius:4px;'></div>"
                        "</div>"
                    )
                    st.markdown(bar_fade, unsafe_allow_html=True)
                    # detailed table
                    display_df = sub[["market","side","%bets"]].rename(
                        columns={"market":"Bet Type","side":"Side","%bets":"% Bets"}
                    )
                    st.dataframe(display_df, use_container_width=True)
                    # Narrative insight
                    if fade_score > 70:
                        callout = "Extreme public lean—high fade potential."
                    elif fade_score >= 60:
                        callout = "Heavy public skew—prime fade candidate."
                    elif fade_score >= 55:
                        callout = "Mild public lean—monitor but low conviction."
                    else:
                        callout = "Balanced bets—no clear fade edge."
                    st.markdown(f"**Insight:** {callout}")
            st.markdown("---")

        # Mood-ball visualization
        try:
            ball = Image.open("mood_ball.png").convert("RGBA")
            gradient_stops = [
                (0,   '#00ff00'),
                (25,  '#ffff00'),
                (50,  '#ffa500'),
                (75,  '#ff0000'),
                (100, '#ff3c78'),
            ]
            def hex_to_rgb(h):
                h = h.lstrip('#')
                return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            cards_sorted = sorted(
                [{"matchup": r["matchup"], "irr": abs(r["%bets"] - r["%handle"])} for r in df.to_dict("records")],
                key=lambda x: x["irr"], reverse=True
            )[:4]
            quad_img = Image.new("RGBA", ball.size)
            width, height = ball.size
            inset = int(min(width, height) * 0.08)
            for i, card in enumerate(cards_sorted):
                angle0, angle1 = i*90, (i+1)*90
                stop_idx = min(int(card["irr"] // 25), len(gradient_stops)-1)
                rgb = hex_to_rgb(gradient_stops[stop_idx][1])
                mask_q = Image.new("L", ball.size, 0)
                md_q = ImageDraw.Draw(mask_q)
                md_q.pieslice([(inset, inset), (width-inset, height-inset)], start=angle0, end=angle1, fill=200)
                overlay = Image.new("RGBA", ball.size, rgb + (150,))
                quad_img = Image.composite(overlay, quad_img, mask_q)
            tinted_ball = Image.alpha_composite(quad_img, ball)
            st.image(tinted_ball, caption="Market Mood Ball (Heatmap)", use_container_width=True)
        except Exception as e:
            st.write(f"⚠️ Failed to load mood_ball.png: {e}")

        st.subheader("🎯 Market Mood Ring")
        st.metric(label="Market Irrationality Index", value=f"{mood_score:.2f}%")

        # Gauge chart
        gauge_data = pd.DataFrame({'value': [mood_score]})
        gauge_base = (
            alt.Chart(gauge_data)
               .mark_arc(innerRadius=60, outerRadius=120)
               .encode(
                   theta=alt.Theta('value:Q', scale=alt.Scale(domain=[0,100])),
                   color=alt.Color(
                       'value:Q',
                       scale=alt.Scale(
                           domain=[0,25,50,75,100],
                           range=['#00ff00','#ffff00','#ffa500','#ff0000','#ff3c78']
                       )
                   )
               )
               .properties(width=350, height=350)
        )
        gauge_text = (
            alt.Chart(gauge_data)
               .mark_text(size=60, color='#ffffff')
               .encode(text=alt.Text('value:Q', format='.1f'))
               .properties(width=350, height=350)
        )
        if mood_score > GATE_THRESHOLD:
            st.markdown(
                """
                <style>
                @keyframes pulse {0%{transform:scale(1);}50%{transform:scale(1.05);}100%{transform:scale(1);}}
                .pulse svg {animation:pulse 2s infinite;}
                </style>
                """, unsafe_allow_html=True
            )
            st.markdown('<div class="pulse">', unsafe_allow_html=True)
            st.altair_chart(gauge_base + gauge_text, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.altair_chart(gauge_base + gauge_text, use_container_width=True)

        # What-If Market Mood
        st.subheader("What-If Market Mood")
        shift = st.slider("Simulate public shift (±%)", -20, 20, 0)
        df_sim = df.copy()
        df_sim["%bets_sim"] = df_sim["%bets"] + shift
        df_sim["%bets_sim"] = df_sim["%bets_sim"].clip(0,100)
        df_sim["irr_sim"] = abs(df_sim["%bets_sim"] - df_sim["%handle"])
        score_sim = df_sim["irr_sim"].mean()
        sim_data = pd.DataFrame({"value":[score_sim]})
        sim_base = gauge_base.encode(theta=alt.Theta('value:Q', scale=alt.Scale(domain=[0,100])))
        sim_text = gauge_text.encode(text=alt.Text('value:Q', format='.1f'))
        st.metric(label="Simulated Irrationality", value=f"{score_sim:.2f}%")
        st.altair_chart(sim_base + sim_text, use_container_width=True)

        mood_chart = alt.Chart(df).mark_bar().encode(
            x="matchup", y="irrationality", color="irrationality"
        ).properties(width=700)
        st.altair_chart(mood_chart, use_container_width=True)

        ball_path = export_assets(tinted_ball)
        caption = generate_caption(mood_score, df.sort_values("%bets", ascending=False).to_dict('records'))
        caption = append_hashtags(caption, mood_score)

        st.subheader("Suggested Caption")
        st.markdown(
            f"""
            <div class="caption-box" style="background-color:#1a1a1d; border:2px solid #ff3c78; border-radius:8px; padding:12px;">
                <p style="color:#ffffff; font-size:18px;">{caption}</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="text-align:center;"><a href="{make_winible_link("caption_cta")}" class="button-link">Unlock the Full Playbook</a></div>',
            unsafe_allow_html=True
        )

        qr_link = make_winible_link('qr_cta')
        qr_img = qrcode.make(qr_link)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        qr_path = Path("exports") / f"qr_{ts}.png"
        qr_img.save(qr_path)
        st.image(qr_path, caption="Scan to Subscribe", use_container_width=False)

        st.subheader("Markdown Snippet")
        md_snippet = generate_markdown_snippet(ball_path, qr_path, caption)
        st.code(md_snippet)

        with st.sidebar.expander("Pro Tools", expanded=False):
            if mood_score > GATE_THRESHOLD:
                st.warning("🔒 Premium Unlocked: See which markets the public is blindly chasing — subscribers only!")
            st.write(f"🔗 Subscribe: {make_winible_link('mood_ring')}")
            subject, body = generate_email_snippet(mood_score, df.sort_values("%bets", ascending=False).to_dict('records'))
            st.subheader("Email/SMS Snippet")
            st.write(f"**Subject:** {subject}")
            st.write(body)
            info_path = generate_infographic(tinted_ball, mood_score, df.sort_values("%bets", ascending=False).to_dict('records'))
            st.image(info_path, caption="Fast Facts Infographic", use_container_width=True)

    # ---- PFR Dashboard ----
    def pfr_dashboard(df):
        public_heavy = df[df["%bets"] > 70].sort_values("%bets", ascending=False)
        st.subheader("📉 Public Fade Ratio (PFR) Dashboard")
        if public_heavy.empty:
            st.write("No heavy public bets found.")
        else:
            trap_chart = (
                alt.Chart(public_heavy)
                   .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                   .encode(
                       x=alt.X('side:N', sort='-y', title='Side'),
                       y=alt.Y('%bets:Q', title='Public Bet %'),
                       color=alt.Color('%bets:Q', scale=alt.Scale(domain=[0,100], range=['#ff3c78','#001f3f'])),
                       tooltip=[alt.Tooltip('side:N'), alt.Tooltip('%bets:Q', format='.1f')]
                   )
                   .properties(width=350, height=350)
            )
            labels = (
                alt.Chart(public_heavy)
                   .mark_text(align='center', baseline='bottom', dy=-5, size=20, color='#ffffff')
                   .encode(
                       x=alt.X('side:N', sort='-y'),
                       y=alt.Y('%bets:Q'),
                       text=alt.Text('%bets:Q', format='.0f')
                   )
                   .properties(width=350, height=350)
            )
            st.altair_chart(trap_chart + labels, use_container_width=True)

    # ---- Helper Functions ----
    def export_assets(tinted_ball, *args):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path("exports")
        outdir.mkdir(exist_ok=True)
        ball_path = outdir / f"mood_ball_{ts}.png"
        tinted_ball.save(ball_path)
        return ball_path

    def generate_caption(mood_score, top_fades):
        hook = "This market’s more irrational than a Coors night—"
        proof = f"{mood_score:.1f}% avg split vs handle"
        fade = f"Top fade: {top_fades[0]['matchup']} at {top_fades[0]['%bets']:.0f}% public"
        return f"{hook}{proof}. {fade}. 🔓 $5 to unlock the full playbook."

    def append_hashtags(base_caption, mood_score):
        tags = ["#MLB", "#SportsBetting", "#Edge"]
        if mood_score > 50:
            tags.append("#FadeThePublic")
        return f"{base_caption}\n\n" + " ".join(tags) + "\n⏰ Market moves fast—don’t get left in the dust."

    def make_winible_link(campaign):
        base = "https://winible.com/foxedgeai"
        return f"{base}?utm_source=streamlit&utm_campaign={campaign}"

    def generate_email_snippet(mood_score, top_fades):
        subject = f"Market Madness at {mood_score:.1f}%"
        body = (
            f"The public’s on the wrong side of {top_fades[0]['matchup']} with "
            f"{top_fades[0]['%bets']:.0f}% of bets. "
            f"Full breakdown unlocked here → {make_winible_link('email_snippet')}"
        )
        return subject, body

    def generate_infographic(tinted_ball, mood_score, top_fades):
        stats_img = tinted_ball.copy()
        draw = ImageDraw.Draw(stats_img)
        font = ImageFont.load_default()
        text = (
            f"Mood: {mood_score:.1f}%\n"
            f"Top fade: {top_fades[0]['matchup']} at {top_fades[0]['%bets']:.0f}% public"
        )
        draw.multiline_text((10, 10), text, fill=(255, 255, 255), font=font)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path("exports")
        outdir.mkdir(exist_ok=True)
        path = outdir / f"fast_facts_{ts}.png"
        stats_img.save(path)
        return path

    def generate_markdown_snippet(ball_path, qr_path, caption):
        return (
            f"![Market Mood Ball]({ball_path})\n\n"
            f"{caption}\n\n"
            f"[Unlock the Full Playbook]({make_winible_link('md_cta')})\n\n"
            f"![Scan to Subscribe]({qr_path})"
        )

    # ---- Pro Signal Helper Functions ----
    def compute_exposure(splits, odds_A, odds_B):
        """
        Estimate bookmaker liability exposure for a market side.
        Returns dict with liability_A, liability_B, and hot_side.
        """
        def liability(handle, odds):
            if odds > 0:
                return handle
            else:
                return handle * (100/abs(odds))
        liab_A = liability(splits['handle_A'], odds_A)
        liab_B = liability(splits['handle_B'], odds_B)
        hot = 'A' if liab_A > liab_B else 'B'
        return {'liability_A': liab_A, 'liability_B': liab_B, 'hot_side': hot}

    def oddlot_signal(splits, threshold=ODDLOT_THRESHOLD):
        """
        Compare ticket vs handle percentage divergence.
        Returns dict with market, side, divergence if above threshold.
        """
        ticket_pct_A = splits['tickets_A'] / (splits['tickets_A'] + splits['tickets_B'])
        handle_pct_A = splits['handle_A'] / (splits['handle_A'] + splits['handle_B'])
        div = abs(ticket_pct_A - handle_pct_A)
        if div >= threshold:
            side = 'A' if ticket_pct_A > handle_pct_A else 'B'
            return {'side': side, 'divergence': div}
        return None

    def corr_breach_signal(odds_ml, odds_sp, odds_to, threshold=CORR_THRESHOLD):
        """
        Flag breaches in implied probability across ML, spread, and total.
        Returns dict with deltas if any exceed threshold.
        """
        p_ml = implied_prob_from_odds(odds_ml)
        p_sp = implied_prob_from_spread(odds_sp)
        p_to = implied_prob_from_total(odds_to)
        d_ml_sp = abs(p_ml - p_sp)
        d_ml_to = abs(p_ml - p_to)
        if d_ml_sp >= threshold or d_ml_to >= threshold:
            return {'d_ml_sp': d_ml_sp, 'd_ml_to': d_ml_to, 'p_ml': p_ml, 'p_sp': p_sp, 'p_to': p_to}
        return None

    def update_history(mood_score):
        today = datetime.now().strftime("%Y-%m-%d")
        if Path(HISTORY_FILE).exists():
            hist = pd.read_csv(HISTORY_FILE)
        else:
            hist = pd.DataFrame(columns=["date","score"])
        new_row = pd.DataFrame([{"date": today, "score": mood_score}])
        hist = pd.concat([hist, new_row], ignore_index=True).tail(7)
        hist.to_csv(HISTORY_FILE, index=False)
        hist["date"] = pd.to_datetime(hist["date"])
        return hist

    def update_splits_history(df):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        df_h = df.copy(); df_h['snapshot_time'] = ts
        if Path(SPLITS_HISTORY_FILE).exists():
            hist = pd.read_csv(SPLITS_HISTORY_FILE)
        else:
            hist = pd.DataFrame()
        prev = hist[hist['snapshot_time'] == hist['snapshot_time'].iloc[-1]] if not hist.empty else pd.DataFrame(columns=df_h.columns)
        hist = pd.concat([hist, df_h], ignore_index=True)
        hist.to_csv(SPLITS_HISTORY_FILE, index=False)
        return prev

    # ---- App Layout ----
    if page == "Overview":
        with st.container():
            col1, col2 = st.columns([3,1], gap="large")
            with col1:
                st.title("⚾ MLB Market Mood & Public Fade Dashboard")
                st.markdown(
                    "<h2 style='color:#ffffff; font-size:24px; margin-bottom:5px;'>FoxEdge Market Sentiment Snapshot</h2>"
                    f"<p style='color:#bbbbbb; font-size:14px; margin-top:0;'>As of {datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%b %d, %Y • %I:%M %p PT')}</p>",
                    unsafe_allow_html=True
                )
            with col2:
                # Always fetch the latest splits & save snapshot
                splits_df = fetch_dk_splits()
                # Apply EWMA smoothing to reduce noise
                splits_df['bets_smooth'] = splits_df['%bets'].ewm(span=3, adjust=False).mean()
                splits_df['handle_smooth'] = splits_df['%handle'].ewm(span=3, adjust=False).mean()
                # Save daily snapshot (overwrite)
                snapshot_dir = Path("snapshots")
                snapshot_dir.mkdir(exist_ok=True)
                snapshot_path = snapshot_dir / f"splits_snapshot_{datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%Y%m%d')}.csv"
                splits_df.to_csv(snapshot_path, index=False)

                # ---- Daily flag rotation (PT) + automation controls ----
                pac = pytz.timezone("America/Los_Angeles")
                today_key = datetime.now(pac).strftime("%Y%m%d")
                if st.session_state.get("daily_key") != today_key:
                    # Midnight catch-up is not wired here because this app always writes a dated CSV above.
                    st.session_state["daily_key"] = today_key
                    st.session_state["open_sql_logged"] = False
                    st.session_state["close_sql_logged"] = False
                    st.session_state["clv_sql_logged"] = False

                with st.sidebar.expander("CLV Automation", expanded=False):
                    st.checkbox("Enable auto CLOSE + CLV", value=st.session_state.get("auto_clv_enabled", True), key="auto_clv_enabled")
                    st.slider("Auto CLOSE hour (PT)", min_value=20, max_value=23, value=int(st.session_state.get("auto_close_hour", 23)), step=1, key="auto_close_hour")
                    st.caption("After this local hour, the app logs CLOSE snapshots and computes CLV once per day.")

                with st.sidebar.expander("CLV Tools", expanded=False):
                    if st.button("📥 Log CLOSE snapshots now"):
                        log_close_snapshot(splits_df)
                        st.success("CLOSE snapshots recorded to SQLite (line_snapshots).")

                    if st.button("🧮 Compute & Save CLV"):
                        clv_df = compute_and_log_clv()
                        st.session_state["clv_df"] = clv_df
                        st.success(f"CLV computed for {0 if clv_df is None else len(clv_df)} picks and saved to clv_logs.")

                    st.markdown(f"**Status:** OPEN={'✅' if st.session_state.get('open_sql_logged') else '❌'} • CLOSE={'✅' if st.session_state.get('close_sql_logged') else '❌'} • CLV={'✅' if st.session_state.get('clv_sql_logged') else '❌'}")
                    st.caption(f"Daily key: {st.session_state.get('daily_key')}")

                # Load previous snapshot for RLM/Steam
                prev_data = st.session_state.get("prev_data")
                # Normalize legacy prev_data to ensure required columns exist
                if prev_data is not None:
                    required_cols = ["matchup","market","side","%bets","odds","spread"]
                    for _col in required_cols:
                        if _col not in prev_data.columns:
                            prev_data[_col] = None

                # Log to SQLite once per session
                if "open_sql_logged" not in st.session_state:
                    st.session_state["open_sql_logged"] = False
                if not st.session_state["open_sql_logged"]:
                    log_snapshot(splits_df)
                    st.session_state["open_sql_logged"] = True
                    st.caption(f"📼 OPEN snapshot logged to {BETLOGS_DB}")

                # ---- Automated CLOSE snapshot + CLV (once per day) ----
                try:
                    now_pt = datetime.now(pac)
                    if st.session_state.get("auto_clv_enabled", True):
                        if (now_pt.hour >= int(st.session_state.get("auto_close_hour", 23))) and not st.session_state.get("close_sql_logged", False):
                            log_close_snapshot(splits_df)
                            st.session_state["close_sql_logged"] = True
                            st.caption(f"📼 CLOSE snapshots logged automatically to {BETLOGS_DB}")
                        if st.session_state.get("close_sql_logged", False) and not st.session_state.get("clv_sql_logged", False):
                            clv_df_auto = compute_and_log_clv()
                            st.session_state["clv_df"] = clv_df_auto
                            st.session_state["clv_sql_logged"] = True
                            st.caption(f"🧮 CLV computed & saved automatically for {0 if clv_df_auto is None else len(clv_df_auto)} picks.")
                except Exception as e:
                    logging.error(f"[market_mood_ncaaf] auto CLOSE/CLV failed: {e}")

                # Compare to previous splits history
                prev_splits = update_splits_history(splits_df)
                if not prev_splits.empty:
                    merged = splits_df.merge(
                        prev_splits[['matchup','market','side','%bets','%handle']],
                        on=['matchup','market','side'], suffixes=('','_prev')
                    )
                    merged['bets_delta']   = merged['%bets'] - merged['%bets_prev']
                    merged['handle_delta'] = merged['%handle'] - merged['%handle_prev']
                    st.markdown("---")
                    st.subheader("📊 Split Movements Since Last Run")
                    with st.expander("Show Split Movements"):
                        st.dataframe(merged, use_container_width=True)
                    st.markdown("---")

                # ---- Pro Signals Section ----
                st.markdown("---")
                st.subheader("🔍 Pro Edge Signals")

                pro_signals = []
                # Iterate each game-side in splits_df
                for idx, row in splits_df.iterrows():
                    # assume rows are ordered with 'A' and 'B' side pairs
                    # fetch the matching opposite row
                    market = row['matchup']
                    side = row['side']
                    # build a small DataFrame to pair A/B, constrain by market type
                    pair = splits_df[(splits_df['matchup'] == market) & (splits_df['market'] == row['market'])]
                    if len(pair) < 2:
                        continue
                    # determine A and B by order
                    rA, rB = pair.iloc[0], pair.iloc[1]
                    # compute exposure
                    exp = compute_exposure(
                        {'handle_A': rA['handle_smooth'], 'handle_B': rB['handle_smooth']},
                        rA['odds'], rB['odds']
                    )
                    if max(exp['liability_A'], exp['liability_B']) > EXPOSURE_THRESHOLD:
                        pro_signals.append(f"{market}: Exposure hot side {exp['hot_side']} (liability {max(exp['liability_A'],exp['liability_B']):.0f})")
                    # odd-lot signal
                    ol = oddlot_signal(
                        {'tickets_A': rA['%bets'], 'tickets_B': rB['%bets'], 'handle_A': rA['handle_smooth'], 'handle_B': rB['handle_smooth']}
                    )
                    if ol:
                        pro_signals.append(f"{market}: Odd-lot divergence {ol['divergence']:.2%} on side {ol['side']}")
                    # correlation breach (requires odds spread & total if available)
                    # skip if no total/spread in splits_df
                    # dummy example: using same odds for sp and to
                    cb = corr_breach_signal(rA['odds'], rA['odds'], rA['odds'])
                    # Note: still using odds-based approx since spread/total prob converters are placeholders.
                    if cb:
                        pro_signals.append(f"{market}: Corr breach ML-SP {cb['d_ml_sp']:.2%}, ML-TO {cb['d_ml_to']:.2%}")
                    # Reverse Line Movement detection
                    if prev_data is not None:
                        try:
                            prev_row = prev_data[(prev_data["matchup"]==market) & (prev_data["market"]==row["market"]) & (prev_data["side"]==side)]
                            if not prev_row.empty:
                                prev_row = prev_row.iloc[0]
                                delta_bets = row["%bets"] - prev_row["%bets"]
                                delta_odds = row["odds"] - prev_row["odds"]
                                # RLM: public bets ↑ but odds move opposite
                                if delta_bets >= RLM_BET_THRESHOLD and (delta_odds * delta_bets) < 0:
                                    pro_signals.append(f"{market} {side}: ⚡ RLM (Δbets {delta_bets:.1f}%, Δodds {delta_odds})")
                                # Steam Move detection
                                if abs(delta_odds) >= STEAM_ODDS_THRESHOLD:
                                    pro_signals.append(f"{market} {side}: 💨 Steam move (Δodds {delta_odds})")
                        except Exception:
                            # Legacy or malformed prev_data; skip RLM/steam this cycle
                            pass
                if pro_signals:
                    for sig in pro_signals:
                        st.markdown(f"- {sig}")
                else:
                    st.success("✅ Market At Equilibrium: No pro edges detected.")

                # -- Log strongest public fade and irrationality as blog picks (for receipts)
                try:
                    if not splits_df.empty:
                        date_iso = datetime.now(pytz.timezone("America/Los_Angeles")).date().isoformat()
                        # Strongest public fade = max %bets
                        fade_idx = splits_df['%bets'].idxmax()
                        fade_row = splits_df.loc[fade_idx]
                        log_blog_pick_to_db({
                            "log_date": datetime.utcnow().isoformat(),
                            "matchup": str(fade_row.get('matchup')),
                            "bet_type": f"Public Fade • {fade_row.get('market')}",
                            "confidence": None,
                            "edge_pct": float(fade_row.get('%bets', 0.0)),
                            "odds": clean_odds(fade_row.get('odds')),
                            "predicted_total": None,
                            "predicted_winner": None,
                            "predicted_margin": None,
                            "bookmaker_total": None,
                            "analysis": f"Public {float(fade_row.get('%bets',0)):.1f}% on {fade_row.get('side')}; handle {float(fade_row.get('%handle',0)):.1f}%."
                        })
                        # Strongest irrationality = max abs(%bets - %handle)
                        irr = (splits_df["%bets"] - splits_df["%handle"]).abs()
                        irr_idx = irr.idxmax()
                        irr_row = splits_df.loc[irr_idx]
                        log_blog_pick_to_db({
                            "log_date": datetime.utcnow().isoformat(),
                            "matchup": str(irr_row.get('matchup')),
                            "bet_type": f"Market Mood • {irr_row.get('market')}",
                            "confidence": None,
                            "edge_pct": float(irr.iloc[irr_idx]) if irr_idx is not None else None,
                            "odds": clean_odds(irr_row.get('odds')),
                            "predicted_total": None,
                            "predicted_winner": None,
                            "predicted_margin": None,
                            "bookmaker_total": None,
                            "analysis": f"Irrationality {float(irr.iloc[irr_idx]) if irr_idx is not None else 0.0:.1f} between bets and handle on {irr_row.get('side')}."
                        })
                        # Also log YOUR_REC snapshots for CLV tracking
                        try:
                            fade_total_val = None
                            fade_spread_val = None
                            if str(fade_row.get('market','')).lower().startswith("total"):
                                ou, tval = _parse_total_from_side(fade_row.get('side'))
                                fade_total_val = tval
                            if str(fade_row.get('market','')).lower().startswith("spread"):
                                fade_spread_val = _parse_spread_from_side(fade_row.get('side'))
                            log_your_rec_snapshot(
                                matchup=str(fade_row.get('matchup')),
                                market=str(fade_row.get('market')),
                                side=str(fade_row.get('side')),
                                odds_val=clean_odds(fade_row.get('odds')),
                                total_val=fade_total_val,
                                spread_val=fade_spread_val
                            )
                        except Exception as _e:
                            logging.error(f"[market_mood_ncaaf] YOUR_REC fade log failed: {_e}")

                        try:
                            irr_total_val = None
                            irr_spread_val = None
                            if str(irr_row.get('market','')).lower().startswith("total"):
                                ou, tval = _parse_total_from_side(irr_row.get('side'))
                                irr_total_val = tval
                            if str(irr_row.get('market','')).lower().startswith("spread"):
                                irr_spread_val = _parse_spread_from_side(irr_row.get('side'))
                            log_your_rec_snapshot(
                                matchup=str(irr_row.get('matchup')),
                                market=str(irr_row.get('market')),
                                side=str(irr_row.get('side')),
                                odds_val=clean_odds(irr_row.get('odds')),
                                total_val=irr_total_val,
                                spread_val=irr_spread_val
                            )
                        except Exception as _e:
                            logging.error(f"[market_mood_ncaaf] YOUR_REC irr log failed: {_e}")
                except Exception as e:
                    logging.error(f"[market_mood] blog logging failed: {e}")

                # Store current snapshot for next run
                st.session_state["prev_data"] = splits_df[["matchup","market","side","%bets","odds","spread"]].copy()

    elif page == "Game Details":
        try:
            splits_df
        except NameError:
            splits_df = fetch_dk_splits()
        show_game_details(splits_df)

    st.markdown(
        """
        <p style='color:#dddddd; font-size:16px;'>
        <strong>Market Mood Ring</strong>: Measures the average difference between public bet tickets and handle percentages — a higher value means the market is 
        more irrational relative to model.
        <br>
        <strong>Public Fade Ratio (PFR)</strong>: Highlights sides where over 70% of bets are on one side; these represent the best fade opportunities.
        </p>
        """, unsafe_allow_html=True
    )

    with st.container():
        st.markdown("<div style='padding:20px; background-color:#0e0f11; border:2px solid #ff3c78;'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,1], gap='large')
        with col1:
            st.markdown("<div class='chart-title'>🎯 Market Mood Ring</div>", unsafe_allow_html=True)
            market_mood_ring(splits_df)
        with col2:
            st.markdown("<div class='chart-title'>📉 Public Fade Ratio</div>", unsafe_allow_html=True)
            pfr_dashboard(splits_df)
        # ---- CLV Results Table ----
        if "clv_df" in st.session_state and isinstance(st.session_state["clv_df"], pd.DataFrame) and not st.session_state["clv_df"].empty:
            st.subheader("📈 CLV Results (Your Picks vs Closing Line)")
            show_cols = ["matchup","market","side","entry_odds","close_odds","entry_total","close_total","entry_spread","close_spread","clv_prob_pp","clv_line_move","clv_spread_move"]
            try:
                st.dataframe(st.session_state["clv_df"][show_cols], use_container_width=True)
            except Exception:
                st.dataframe(st.session_state["clv_df"], use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<img src='https://your-domain.com/foxedge-logo.png' "
            "style='position:absolute; bottom:10px; right:10px; width:80px; opacity:0.6;'>",
            unsafe_allow_html=True
        )



def render_do_not_bet():


    # === Daily Summary helpers (global) ===
    try:
        aggrid_available = True
    except ModuleNotFoundError:
        aggrid_available = False
        st.warning("📦 Optional dependency *streamlit‑aggrid* not found. Falling back to basic tables. "
                   "Run `pip install streamlit‑aggrid` for interactive grids.")

    pacific_tz = pytz.timezone("America/Los_Angeles")

    # ==== League/Source configuration ====
    LEAGUE_NAME = "NFL"
    # DraftKings Event Group IDs: NFL=88808, NCAAF=87637, MLB=84240
    EVENT_GROUP = 88808

    # === bet_logs target (shared across apps) ===
    BETLOGS_ROOT = os.getenv("BETLOGS_ROOT", os.getcwd())
    BETLOGS_DB = os.path.join(BETLOGS_ROOT, "bet_logs.db")

    def _ensure_parent(path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass

    # === SQLite connection for odds snapshots (open & close) ===
    _ensure_parent(BETLOGS_DB)
    conn = sqlite3.connect(BETLOGS_DB, check_same_thread=False)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS line_snapshots (
                game_id        TEXT,
                matchup        TEXT,
                market         TEXT,
                side           TEXT,
                book           TEXT,
                snapshot_type  TEXT,   -- 'OPEN' | 'CLOSE' | 'YOUR_REC'
                odds           INTEGER,
                total          REAL,
                timestamp_utc  TEXT,
                PRIMARY KEY (game_id, market, side, book, snapshot_type)
            )
            """
        )

    def clean_odds(odds_str):
        """Extract the first signed integer (American odds) from any string."""
        match = re.search(r"[-+]?\d+", str(odds_str))
        return match.group(0) if match else ""

    # === Odds‑snapshot helpers =================================================
    def _log_snapshot(df: pd.DataFrame, market: str, snap_type: str, date_override: str | None = None) -> None:
        """
        Append an odds snapshot to the line_snapshots table.
        • df must contain columns: matchup, side, odds
        • game_id = _mk_id(date_iso, matchup, market, side)
        """
        pac = pytz.timezone("America/Los_Angeles")
        date_iso = date_override or datetime.datetime.now(pac).date().isoformat()
        utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat()

        rows = []
        for _, r in df.iterrows():
            matchup = str(r.get("matchup",""))
            side = str(r.get("side",""))
            if not matchup or not side:
                continue
            gid = _mk_id(date_iso, matchup, market, side)
            rows.append(
                (
                    gid,
                    matchup,
                    market,
                    side,
                    'DK',
                    snap_type,
                    int(clean_odds(r.get('odds'))) if r.get('odds') not in (None, "") else None,
                    None,
                    utc_now,
                )
            )
        if rows:
            with conn:
                conn.executemany(
                    """INSERT OR IGNORE INTO line_snapshots
                       (game_id, matchup, market, side, book, snapshot_type,
                        odds, total, timestamp_utc)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
            # also persist a dated CSV for midnight catch‑up
            try:
                Path("snapshots").mkdir(parents=True, exist_ok=True)
                df.to_csv(Path("snapshots") / f"splits_snapshot_{date_iso.replace('-','')}.csv", index=False)
            except Exception:
                pass

    # --- Helper functions for CLOSE/CLV logging ---
    def log_close_snapshot(df: pd.DataFrame, market: str, date_override: str | None = None) -> None:
        """Force‑write CLOSE snapshots for all rows in df for the given market."""
        _log_snapshot(df, market, "CLOSE", date_override=date_override)

    def compute_and_log_clv(db_path: str = BETLOGS_DB) -> pd.DataFrame:
        """
        Compute Closing Line Value (CLV) for every (game_id, market, side, book) that has both
        a YOUR_REC and a CLOSE snapshot. Logs results into a `clv_logs` table and returns a DataFrame.
        """
        with sqlite3.connect(db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS clv_logs (
                    game_id TEXT,
                    matchup TEXT,
                    market TEXT,
                    side TEXT,
                    book TEXT,
                    entry_odds INTEGER,
                    close_odds INTEGER,
                    entry_total REAL,
                    close_total REAL,
                    clv_prob_pp REAL,
                    clv_line_move REAL,
                    computed_utc TEXT,
                    PRIMARY KEY (game_id, market, side, book)
                );
            """)
            snaps = pd.read_sql_query("SELECT * FROM line_snapshots", con)
        if snaps.empty:
            return pd.DataFrame(columns=["game_id","matchup","market","side","book","entry_odds","close_odds","entry_total","close_total","clv_prob_pp","clv_line_move","computed_utc"])

        snaps = snaps[snaps["book"] == "DK"].copy()
        keys = ["game_id","matchup","market","side","book"]
        entry = snaps[snaps["snapshot_type"] == "YOUR_REC"][keys + ["odds","total"]].rename(columns={"odds":"entry_odds","total":"entry_total"})
        close = snaps[snaps["snapshot_type"] == "CLOSE"][keys + ["odds","total"]].rename(columns={"odds":"close_odds","total":"close_total"})
        merged = entry.merge(close, on=keys, how="inner")

        def _impl_prob(odds_val):
            try:
                o = int(odds_val)
            except Exception:
                return None
            if o > 0:
                return 100.0 / (o + 100.0)
            return (-o) / (100.0 - o)

        if merged.empty:
            return pd.DataFrame(columns=["game_id","matchup","market","side","book","entry_odds","close_odds","entry_total","close_total","clv_prob_pp","clv_line_move","computed_utc"])

        merged["clv_prob_pp"] = merged.apply(
            lambda r: None if (r["entry_odds"] is None or r["close_odds"] is None)
            else round((_impl_prob(r["close_odds"]) - _impl_prob(r["entry_odds"])) * 100.0, 3),
            axis=1
        )
        merged["clv_line_move"] = None
        merged["computed_utc"] = datetime.utcnow().isoformat()

        with sqlite3.connect(db_path) as con:
            rows = merged[["game_id","matchup","market","side","book","entry_odds","close_odds","entry_total","close_total","clv_prob_pp","clv_line_move","computed_utc"]].to_records(index=False)
            con.executemany("""
                INSERT INTO clv_logs (game_id, matchup, market, side, book, entry_odds, close_odds, entry_total, close_total, clv_prob_pp, clv_line_move, computed_utc)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(game_id, market, side, book) DO UPDATE SET
                    entry_odds=excluded.entry_odds,
                    close_odds=excluded.close_odds,
                    entry_total=excluded.entry_total,
                    close_total=excluded.close_total,
                    clv_prob_pp=excluded.clv_prob_pp,
                    clv_line_move=excluded.clv_line_move,
                    computed_utc=excluded.computed_utc
            """, list(rows))
        return merged

    def _maybe_log_close_snapshot(df: pd.DataFrame, market: str) -> None:
        """
        Write a 'CLOSE' snapshot ≤5 min before kickoff for rows in *df*.
        Assumes df['game_time'] is a string like '4:05 PM'.
        """
        now_pt = datetime.datetime.now(pacific_tz)
        closers = []

        for _, r in df.iterrows():
            try:
                # Parse time string; combine with today's date
                gt = datetime.datetime.strptime(r['game_time'], '%I:%M %p')
                game_dt = pacific_tz.localize(
                    datetime.datetime.combine(now_pt.date(), gt.time())
                )
                seconds_out = (game_dt - now_pt).total_seconds()
                if 0 <= seconds_out <= 300:        # within 5 minutes
                    closers.append(r)
            except Exception:
                continue

        if closers:
            _log_snapshot(pd.DataFrame(closers), market, "CLOSE")

    # === Proof logging helpers (shared schema expected by proof_maker) ======
    def log_blog_pick_to_db(pick_data: dict, db_path: str = BETLOGS_DB):
        _ensure_parent(db_path)
        try:
            with sqlite3.connect(db_path) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS blog_pick_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_date TEXT,
                        matchup TEXT,
                        bet_type TEXT,
                        confidence TEXT,
                        edge_pct REAL,
                        odds REAL,
                        predicted_total REAL,
                        predicted_winner TEXT,
                        predicted_margin REAL,
                        bookmaker_total REAL,
                        analysis TEXT
                    );
                """)
                cols = ["log_date","matchup","bet_type","confidence","edge_pct","odds",
                        "predicted_total","predicted_winner","predicted_margin","bookmaker_total","analysis"]
                row = [pick_data.get(k) for k in cols]
                con.execute(f"INSERT INTO blog_pick_logs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", row)
        except Exception as e:
            logging.error(f"[do_not_bet.log_blog_pick_to_db] {e}")

    def log_bet_card_to_db(card: dict, db_path: str = BETLOGS_DB):
        _ensure_parent(db_path)
        try:
            with sqlite3.connect(db_path) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS bet_card_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id TEXT,
                        home_team TEXT,
                        away_team TEXT,
                        game_time TEXT,
                        combined_runs REAL,
                        delta REAL,
                        total_bet_rec TEXT,
                        ml_bet_rec TEXT,
                        over_edge REAL,
                        under_edge REAL,
                        home_ml_edge REAL,
                        away_ml_edge REAL,
                        home_ensemble_pred REAL,
                        away_ensemble_pred REAL,
                        combined_ensemble_pred REAL,
                        ensemble_confidence TEXT,
                        statcast_home_exit REAL,
                        statcast_home_angle REAL,
                        statcast_away_exit REAL,
                        statcast_away_angle REAL,
                        pitcher_home_exit REAL,
                        pitcher_home_angle REAL,
                        pitcher_away_exit REAL,
                        pitcher_away_angle REAL,
                        weather_home_temp REAL,
                        weather_home_wind REAL,
                        weather_away_temp REAL,
                        weather_away_wind REAL,
                        home_pitcher TEXT,
                        away_pitcher TEXT,
                        bookmaker_line REAL,
                        over_price REAL,
                        under_price REAL,
                        home_ml_book REAL,
                        away_ml_book REAL,
                        log_date TEXT
                    );
                """)
                # minimally-populated row for splits-based recs
                defaults = {
                    "game_id": None, "home_team": None, "away_team": None, "game_time": None,
                    "combined_runs": None, "delta": None, "total_bet_rec": None, "ml_bet_rec": None,
                    "over_edge": None, "under_edge": None, "home_ml_edge": None, "away_ml_edge": None,
                    "home_ensemble_pred": None, "away_ensemble_pred": None, "combined_ensemble_pred": None,
                    "ensemble_confidence": "splits", "statcast_home_exit": None, "statcast_home_angle": None,
                    "statcast_away_exit": None, "statcast_away_angle": None, "pitcher_home_exit": None,
                    "pitcher_home_angle": None, "pitcher_away_exit": None, "pitcher_away_angle": None,
                    "weather_home_temp": None, "weather_home_wind": None, "weather_away_temp": None,
                    "weather_away_wind": None, "home_pitcher": None, "away_pitcher": None, "bookmaker_line": None,
                    "over_price": None, "under_price": None, "home_ml_book": None, "away_ml_book": None,
                    "log_date": datetime.utcnow().isoformat()
                }
                payload = defaults | {k: card.get(k) for k in defaults.keys()}
                cols = list(defaults.keys())
                row = [payload[k] for k in cols]
                con.execute(f"INSERT INTO bet_card_logs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", row)
        except Exception as e:
            logging.error(f"[do_not_bet.log_bet_card_to_db] {e}")

    def log_snapshot_your_rec(game_id, market, side, total_or_none, odds_or_none, db_path: str = BETLOGS_DB):
        """
        Convenience: write a YOUR_REC snapshot for either Totals (total value) or ML/Spread (odds).
        """
        try:
            with sqlite3.connect(db_path) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS line_snapshots (
                        game_id        TEXT,
                        matchup        TEXT,
                        market         TEXT,
                        side           TEXT,
                        book           TEXT,
                        snapshot_type  TEXT,
                        odds           INTEGER,
                        total          REAL,
                        timestamp_utc  TEXT,
                        PRIMARY KEY (game_id, market, side, book, snapshot_type)
                    );
                """)
                con.execute(
                    "INSERT OR IGNORE INTO line_snapshots (game_id, matchup, market, side, book, snapshot_type, odds, total, timestamp_utc) VALUES (?,?,?,?,?,?,?,?,?)",
                    (str(game_id), None, market, side, "DK", "YOUR_REC",
                     int(odds_or_none) if odds_or_none not in (None, "") else None,
                     float(total_or_none) if total_or_none not in (None, "") else None,
                     datetime.utcnow().isoformat())
                )
        except Exception as e:
            logging.error(f"[do_not_bet.log_snapshot_your_rec] {e}")

    # --- tiny parsers for totals/ID ---
    def _parse_total_from_side(side_text: str):
        """
        From strings like 'Over 8.5' or 'Under 9', return (side, total_float).
        Returns (None, None) if not a totals side.
        """
        try:
            s = str(side_text)
            if s.lower().startswith("over ") or s.lower().startswith("under "):
                parts = s.split()
                if len(parts) >= 2:
                    return parts[0].title(), float(parts[1])
        except Exception:
            pass
        return None, None

    def _mk_id(date_iso: str, matchup: str, market: str, side: str) -> str:
        return f"{date_iso}_{matchup.replace(' ','_')}_{market}_{side.replace(' ','_')}"[:128]

    # alias for legacy variable name used later in the script
    pac = pacific_tz

    def _build_snapshot(df: pd.DataFrame):
        """
        Identify the top public sucker bet, the biggest sharp smash,
        and the heaviest reverse‑line move using only in‑app fields.
        """
        if df.empty:
            return None, None, None

        tmp = df.copy()
        tmp["bets"]  = pd.to_numeric(tmp["%bets"],   errors="coerce")
        tmp["money"] = pd.to_numeric(tmp["%handle"], errors="coerce")
        tmp["diff"]  = tmp["bets"] - tmp["money"]

        top_sucker  = tmp.sort_values("diff", ascending=False).iloc[0]
        tmp["sharp_gap"] = tmp["money"] - tmp["bets"]
        sharp_smash = tmp.sort_values("sharp_gap", ascending=False).iloc[0]

        tmp["odds_move"] = pd.to_numeric(tmp["odds_move"], errors="coerce")
        rlm_candidates = tmp[
            ((tmp["bets"] > 50) & (tmp["odds_move"] < 0)) |
            ((tmp["bets"] < 50) & (tmp["odds_move"] > 0))
        ]
        biggest_rlm = None
        if not rlm_candidates.empty:
            biggest_rlm = rlm_candidates.reindex(
                rlm_candidates["odds_move"].abs().sort_values(ascending=False).index
            ).iloc[0]

        return top_sucker, sharp_smash, biggest_rlm



    def generate_daily_summary(recs_df: pd.DataFrame, full_df: pd.DataFrame) -> str:
        """
        Build the Public Enemy Lines markdown summary from app data only.
        """
        top_sucker, sharp_smash, biggest_rlm = _build_snapshot(full_df)
        date_str = datetime.now(pacific_tz).strftime('%B %d, %Y')

        lines = [
            f"## 🔥 {LEAGUE_NAME} Public Enemy Lines — {date_str}",
            "",
            "### 1️⃣ Slate Snapshot",
            "",
            "| Metric | Game / Side | Bets / Money |",
            "| --- | --- | --- |"
        ]

        if top_sucker is not None:
            lines.append(f"| Biggest Public Sucker Bet | {top_sucker['side']} | "
                         f"{int(top_sucker['bets'])}% / {int(top_sucker['money'])}% |")
        if sharp_smash is not None:
            lines.append(f"| Largest Sharp Smash | {sharp_smash['side']} | "
                         f"{int(sharp_smash['bets'])}% / {int(sharp_smash['money'])}% |")
        if biggest_rlm is not None:
            move_desc = f"{int(biggest_rlm['odds_move']):+d}c vs public"
            lines.append(f"| Heaviest RLM | {biggest_rlm['side']} | {move_desc} |")

        lines += ["", "### 2️⃣ Game Cards", ""]
        for _, r in recs_df.iterrows():
            rlm_flag = "✅" if r.get("rlm_flag") else "—"
            edge_pct = r.get("value_diff", 0) * 100
            lines += [
                f"#### ⚾ {r['matchup']} — {r['side']} {r['odds']}",
                "",
                "| Bets / Money | Edge % | Kelly % | RLM | Confidence |",
                "| --- | --- | --- | --- | --- |",
                f"| {r['%bets']}% / {r['%handle']}% | {edge_pct:.1f} | "
                f"{r['stake_pct']*100:.1f} | {rlm_flag} | {r['confidence']:.1f} |",
                ""
            ]

        lines += [
            "### 3️⃣ Recap & Bankroll Exposure",
            "",
            "| Pick | Odds | Kelly % | Units |",
            "| --- | --- | --- | --- |"
        ]

        total_units = 0.0
        for _, r in recs_df.iterrows():
            units = round(r["stake_pct"] * 100 / 2, 2)  # half‑Kelly metaphor
            total_units += units
            lines.append(f"| {r['side']} | {r['odds']} | {r['stake_pct']*100:.1f}% | {units:.2f} |")

        lines += ["", f"**Total Risk:** {total_units:.2f} U  "]
        return "\n".join(lines)

    # --- Full Report Generator ---
    def generate_full_report(current_df: pd.DataFrame, recs_df: pd.DataFrame) -> str:
        """
        Build the premium *Inner‑Circle Deck* for paying insiders.
        – Uses only current_df & recs_df (no historicals required)
        Sections:
          1. Exec Snapshot
          2. Buy List
          3. Trap of the Day
          4. Watch List (<5 bp edges)
          5. Strategy Note & Glossary
        """
        pac   = pytz.timezone("America/Los_Angeles")
        today = datetime.now(pac).strftime("%B %d, %Y")

        # --- Snapshot numbers -------------------------------------------------
        board = current_df["matchup"].nunique()
        num_traps   = (current_df["verdict"] == "🚨 Public Trap").sum()
        num_sharp   = (current_df["verdict"] == "💰 Sharp Side").sum()
        # --- Additional Signal Counts ---
        num_donotbet    = int((current_df["verdict"] == "⛔ Do Not Bet").sum())
        num_big_movers  = int(current_df["big_mover_flag"].sum()) if "big_mover_flag" in current_df.columns else 0
        num_divergences = int(current_df["divergence_flag"].sum()) if "divergence_flag" in current_df.columns else 0
        num_rlm         = int(current_df["rlm_flag"].sum()) if "rlm_flag" in current_df.columns else 0
        num_kelly       = int(current_df["kelly_trigger"].sum()) if "kelly_trigger" in current_df.columns else 0
        proj_roi    = recs_df.get("edge", recs_df.get("value_diff", pd.Series(0))).mean() if not recs_df.empty else 0
        risk_units  = recs_df["stake_pct"].sum()*100 if not recs_df.empty else 0

        # Public IQ meter = average % bets
        crowd_iq = int(pd.to_numeric(current_df["%bets"], errors="coerce").mean())

        # Identify worst trap (highest tickets %)
        trap_row = None
        if num_traps:
            trap_row = current_df[current_df["verdict"]=="🚨 Public Trap"].sort_values("%bets", ascending=False).iloc[0]
            trap_gap = int(float(trap_row["%bets"]) - float(trap_row["%handle"]))

        # Build Watch List (<5 bp edges or <0.05 abs(value_diff))
        watch_df = current_df.copy()
        watch_df["value_diff"] = watch_df.get("value_diff", 0)
        watch_list = watch_df[
            watch_df["value_diff"].abs().between(0.0001, 0.05)
        ].sort_values("value_diff", ascending=False).head(3)

        lines: list[str] = []
        # --- HEADER -----------------------------------------------------------
        lines.append(f"🔥 **Public Enemies Report — {LEAGUE_NAME} // {today}**")
        lines.append("*Confidential.*  *Sharp eyes only.*")
        lines.append("")

        # --- EXEC SNAPSHOT ----------------------------------------------------
        lines.append("### EXEC SNAPSHOT")
        # Build Crowd‑IQ gauge first
        filled = int(crowd_iq // 10)
        gauge  = "█" * filled + "░" * (10 - filled)

        # Color‑code ROI
        if proj_roi > 0:
            roi_color = "#4FD27F"
        elif proj_roi < 0:
            roi_color = "#FF5555"
        else:
            roi_color = "#FFFFFF"
        roi_html = f"<span style='color:{roi_color}'>{proj_roi:+.1%}</span>"

        # Clean bullet list snapshot
        lines += [
          f"- 🎮 **Slate:** {board} games",
          f"- 🤡 **Traps:** {num_traps}",
          f"- 💰 **Sharp sides:** {num_sharp}",
          f"- ⛔ **Do Not Bet:** {num_donotbet}",
          f"- 📈 **Big Movers:** {num_big_movers}",
          f"- 🔀 **Divergences:** {num_divergences}",
          f"- 🔄 **RLMs:** {num_rlm}",
          f"- ⚡ **Kelly Triggers:** {num_kelly}",
          f"- 📈 **Projected ROI:** {roi_html}",
          f"- 🎯 **Risk:** {risk_units:.2f} U",
          f"- 🧠 **Crowd IQ:** {crowd_iq} `{gauge}`",
          ""
        ]

        # --- Sharp Sides Detail ---
        if num_sharp:
            lines.append("### 💰 Sharp Sides Detail")
            for _, s in current_df[current_df["verdict"]=="💰 Sharp Side"].iterrows():
                lines.append(f"- {s['matchup']} — {s['side']} {s['odds']} ({s['%bets']}% bets / {s['%handle']}% handle)")
            lines.append("")

        # --- BUY LIST ---------------------------------------------------------
        if recs_df.empty:
            lines.append("### ✅ BUY LIST (none today — discipline)")
        else:
            lines.append("### ✅ BUY LIST")
        if recs_df.empty:
            pass
        else:
            # --- Bet Recommendations Snapshot ---
            for _, r in recs_df.sort_values("confidence", ascending=False).iterrows():
                units = round(r.stake_pct * 100 / 2, 2)
                edge  = (r.edge if "edge" in r else r.value_diff) * 100
                clv   = f"{int(r.odds_move):+d}c"
                lines.append(f"**{r.side} {r.odds}** {units:.2f} u")
                lines.append(f"Edge {edge:+.1f} bp | Move {clv} | Crowd {r['%bets']}% / Cash {r['%handle']}%")
                lines.append("")

        # --- TRAP OF THE DAY --------------------------------------------------
        lines.append("### ❌ TRAP OF THE DAY")
        if trap_row is None:
            lines.append("_No screaming public trap detected — stay alert._")
        else:
            lines.append(f"**{trap_row['side']} {trap_row['odds']}**")
            lines.append(f"{trap_row['%bets']} % tickets / {trap_row['%handle']} % money  → gap {trap_gap:+} pp")
            lines.append("Zero steam. Book begging for action. Fade or stay flat.")
        lines.append("")

        # --- Public Traps Detail ---
        if num_traps:
            lines.append("### 🚨 Public Traps Detail")
            for _, t in current_df[current_df["verdict"]=="🚨 Public Trap"].iterrows():
                lines.append(f"- {t['matchup']} — {t['side']} {t['odds']} ({t['%bets']}% bets / {t['%handle']}% handle)")
            lines.append("")

        # --- Do Not Bet Detail ---
        do_not_bets = current_df[current_df["verdict"]=="⛔ Do Not Bet"]
        if not do_not_bets.empty:
            lines.append("### ⛔ Do Not Bet Detail")
            for _, d in do_not_bets.iterrows():
                lines.append(f"- {d['matchup']} — {d['side']} {d['odds']} ({d['%bets']}% bets / {d['%handle']}% handle)")
            lines.append("")

        # --- Big Mover Detail ---
        if num_big_movers:
            lines.append("### 📈 Big Mover Detail")
            for _, m in current_df[current_df["big_mover_flag"]].iterrows():
                lines.append(f"- {m['matchup']} — {m['side']} moved {m['odds_move']}c at {m['odds']} ({m['%bets']}% bets / {m['%handle']}% handle)")
            lines.append("")

        # --- Divergence Detail ---
        if num_divergences:
            lines.append("### 🔀 Divergence Detail")
            for _, dv in current_df[current_df["divergence_flag"]].iterrows():
                lines.append(f"- {dv['matchup']} — {dv['side']} {dv['odds']} (Z={dv['value_diff_z']:.2f})")
            lines.append("")

        # --- RLM Detail ---
        if num_rlm:
            lines.append("### 🔄 RLM Detail")
            for _, r in current_df[current_df["rlm_flag"]].iterrows():
                lines.append(f"- {r['matchup']} — {r['side']} {r['odds']} (move {r['odds_move']}c)")
            lines.append("")

        # --- Kelly Trigger Detail ---
        if num_kelly:
            lines.append("### ⚡ Kelly Trigger Detail")
            for _, k in current_df[current_df["kelly_trigger"]].iterrows():
                lines.append(f"- {k['matchup']} — {k['side']} {k['odds']} (stake {k['stake_pct']*100:.1f}%)")
            lines.append("")

        # --- Avoid Tag Detail ---
        if current_df["avoid_tag"].astype(bool).any():
            lines.append("### ❌ Avoid Tag Detail")
            for _, a in current_df[current_df["avoid_tag"] != ""].iterrows():
                lines.append(f"- {a['matchup']} — {a['side']} {a['odds']} ({a['%bets']}% bets / {a['%handle']}% handle)")
            lines.append("")

        # --- Clown Tag Detail ---
        if current_df["clown_tag"].astype(bool).any():
            lines.append("### 🤡 Clown Tag Detail")
            for _, c in current_df[current_df["clown_tag"] != ""].iterrows():
                lines.append(f"- {c['matchup']} — {c['side']} {c['odds']} ({c['%bets']}% bets / {c['%handle']}% handle)")
            lines.append("")

        # --- Fade Public Detail ---
        if num_fades:
            lines.append("### 🌪️ Fade Public Detail")
            for _, f in current_df[current_df["contrarian_signal"] == "Fade Public"].iterrows():
                lines.append(f"- {f['matchup']} — {f['side']} {f['odds']} ({f['%bets']}% bets / {f['%handle']}% handle) ▶️ move {f['odds_move']}c")
            lines.append("")

        # --- Back Public Detail ---
        if num_backs:
            lines.append("### 🛬 Back Public Detail")
            for _, b in current_df[current_df["contrarian_signal"] == "Back Public"].iterrows():
                lines.append(f"- {b['matchup']} — {b['side']} {b['odds']} ({b['%bets']}% bets / {b['%handle']}% handle) ▶️ move {b['odds_move']}c")
            lines.append("")

        # --- WATCH LIST -------------------------------------------------------
        lines.append("### 👀 WATCH LIST (edges < 5 bp)")
        if watch_list.empty:
            lines.append("_Board is tight — no watch edges yet._")
        else:
            for _, w in watch_list.iterrows():
                edge_bp = w["value_diff"]*100
                lines.append(f"• {w['side']} {w['odds']} — provisional edge {edge_bp:+.1f} bp")
        lines.append("")

        # --- STRATEGY NOTE + GLOSSARY ----------------------------------------
        lines.append("**Discipline wins. Clip ≤ 5 % roll. Let the squares bankroll our edge.**")
        lines.append("")
        lines += [
            "*Edge = price edge in <span title='Basis-point = 0.01 %'>**basis-points**</span>*",
            "*Gap  = tickets % − money % in <span title='Percentage-point = 1 %'>**percentage-points**</span>*"
        ]
        lines.append("")

        # Add generation timestamp footer
        lines.append(f"<div style='text-align:right;font-size:12px;color:#777;'>Report generated {datetime.now(pac).strftime('%H:%M')} PT</div>")

        return "\n".join(lines)


    # --- Sharp-Chat Log Generator ---
    def generate_chat_report(current_df: pd.DataFrame, recs_df: pd.DataFrame) -> str:
        """
        Builds a punchy “INSIDER EDGE ALERT” ready for social share.
        • Bold hooks
        • Emojis for scannability
        • Clear CTA to drive clicks / subs
        """

        pac   = pytz.timezone("America/Los_Angeles")
        today = datetime.now(pac).strftime("%b %d")

        board = current_df["matchup"].nunique()
        traps = current_df[current_df["verdict"] == "🚨 Public Trap"]
        sharps= current_df[current_df["verdict"] == "💰 Sharp Side"]

        lines = []
        # — HEADLINE HOOK —
        lines.append(f"🚨 **INSIDER EDGE ALERT – {LEAGUE_NAME} // {today}** 🚨")
        lines.append("*Leaked from the sharp syndicate desk*")
        lines.append(" ")

        # — QUICK COUNT —
        lines.append(f"🗺️ Card: **{board} games**  |  🎯 Sharp sides: **{len(sharps)}**  |  🤡 Public traps: **{len(traps)}**")
        lines.append(" ")

        # — PUBLIC TRAP CALL‑OUT —
        if not traps.empty:
            worst = traps.sort_values('%bets', ascending=False).iloc[0]
            gap   = int(float(worst['%bets']) - float(worst['%handle']))
            lines.append(f"❌ **Square Trap of the Day:** {worst['side']} {worst['odds']}")
            lines.append(f"   {worst['%bets']} % tickets / {worst['%handle']} % money  → gap {gap:+} pp")
            lines.append("   No steam. Book begging for action. *We’re fading*.")
        else:
            lines.append("❌ No mega‑trap yet, but we’re hunting.")
        lines.append(" ")

        # — SHARP BUY SPOT —
        if recs_df.empty:
            lines.append("⏳ **No verified edges to fire – discipline stays paid.**")
        else:
            best = recs_df.sort_values("confidence", ascending=False).iloc[0]
            units = round(best['stake_pct']*100/2, 2)
            edge  = best.get('edge', best.get('value_diff',0))*100
            move  = best['odds_move']
            lines.append(f"💰 **Sharp Buy Loaded:** {best['side']} {best['odds']}")
            lines.append(f"   Edge {edge:+.1f} bp • Stake {units}u • Line move {move:+d}c")
            lines.append(f"   Crowd {best['%bets']} % vs Cash {best['%handle']} % — we’re on the smarter side.")
        lines.append(" ")

        # — FAST GLOSSARY —
        lines.append("_Tickets = % bets • Money = % handle • Edge in bp (0.01 % per bp)_")
        lines.append(" ")

        # — CTA —
        lines.append("👉 **Steal the full board & minute‑by‑minute steam alerts → [Join the Pro Feed]**")
        lines.append("_Slots close when lines move._")

        return "\n".join(lines)



    def generate_social_posts(df):
        """
        Generate social media posts (tweet, IG caption, TikTok script) from top recommendation.
        """
        top = df.sort_values("confidence", ascending=False).iloc[0]
        tweet = f"🤡 Clown Pick: {top['side']} ({top['odds']}) is getting {top['%bets']}% of bets. Odds haven’t moved. Books aren’t scared. #DoNotBetClub"
        ig = f"""❌ TRAP GAME: {top['matchup']}
    🔻 Side: {top['side']}
    💸 Public Bets: {top['%bets']}%
    📉 Line Move: {top['odds_move']} pts
    🧠 Trap Score: {top['trap_score']}/10
    🕵️ {top['trap_explanation']}

    Join the resistance:
    👉 @publicenemylines
    📥 Link in bio to get daily alerts"""
        tiktok = f"""🎯 Game: {top['matchup']}
    🚨 Bet: {top['side']} at {top['odds']}
    📊 Public: {top['%bets']}% of bets

    Looks like an easy win, right?
    Wrong.

    No line move.
    No sharp money.
    Trap Score: {top['trap_score']}/10

    Books want you to take this bait.

    🛑 Don’t be the sucker.
    Powered by Public Enemy Lines."""
        return {"tweet": tweet, "ig": ig, "tiktok": tiktok}
    # --- Audit Log (always run this block) ---
    #!/usr/bin/env python3

    

    # --- Print / PDF mode toggle ---------------------------------------------
    print_mode = st.sidebar.checkbox("🖨️ Print / PDF Mode", value=False)
    if print_mode:
        # Hide entire sidebar in CSS
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] {display:none !important;}
            </style>
            """,
            unsafe_allow_html=True
        )

    # Inject card-grid styling (optional neon border)
    st.markdown("""
    <style>
    div.card {
        background-color: #111 !important;
        border: 2px solid #39FF14 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Brand Theme Overrides & Hero Banner ---
    st.markdown("""
    <style>
    /* --- Brand Theme Overrides --- */
    body, .stApp {
        background-color: #0A0A0A !important;
        color: #F5F5F5 !important;
        font-family: 'Inter', sans-serif;
    }
    thead tr th {
        background-color: #0E0E0D !important;
        color: #F5F5F5 !important;
        font-size: 14px !important;
    }
    tbody tr td {
        font-size: 13px !important;
    }
    tbody tr:hover {
        background-color: #1A1A1A !important;
    }
    a {
        color: #00E6FF !important;
    }
    div.hero {
        width: 100%;
        padding: 16px 0;
        margin-bottom: 8px;
        text-align: center;
        font-size: 32px;
        font-weight: 900;
        background: linear-gradient(90deg,#FF0033 0%,#FF0033 40%,rgba(255,0,51,0) 100%);
    }
    div.hero .date{
        font-size:18px;
        font-weight:400;
        color:#F5F5F5;
    }
    </style>
    """, unsafe_allow_html=True)

    # ----- Responsive tweaks for mobile -----
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    /* ----- Responsive tweaks for mobile ----- */
    @media only screen and (max-width: 800px) {
        div.hero {font-size:24px;}
        /* Force card grid to 2 columns */
        .stApp > div > div > div > div:nth-child(1) > div {grid-template-columns: repeat(2, 1fr) !important;}
    }
    </style>
    """, unsafe_allow_html=True)

    def is_primetime_game(row):
        try:
            update = row.get("update_time", None)
            if not isinstance(update, datetime):
                return False
            matchup = row.get("matchup", "")
            if LEAGUE_NAME == "NCAAF":
                big_programs = [
                    "Alabama","Georgia","Ohio State","Michigan","Notre Dame","Texas","USC",
                    "LSU","Oklahoma","Penn State","Clemson","Oregon","Florida State","Tennessee"
                ]
                return any(team in matchup for team in big_programs)
            else:
                big_teams = [
                    "Yankees","Red Sox","Dodgers","Cubs","Giants","Cardinals",
                    "Astros","Braves","Mets","Phillies","White Sox"
                ]
                return any(t in matchup for t in big_teams)
        except Exception:
            return False

    # Auto-refresh counter with dynamic interval based on PT time
    pacific = pytz.timezone("America/Los_Angeles")
    current_pt = datetime.now(pacific)
    # ISO strings for today & tomorrow (Pacific time)
    date_today_iso    = current_pt.strftime("%Y-%m-%d")
    date_tomorrow_iso = (current_pt + timedelta(days=1)).strftime("%Y-%m-%d")

    # Decide the default interval first, then allow session‑state overrides
    default_interval = 600 if current_pt.hour < 6 else 60
    refresh_interval_seconds = st.session_state.get("refresh_interval_seconds", default_interval)

    refresh_count = st_autorefresh(interval=refresh_interval_seconds * 1000, key="auto_refresh")
    if not print_mode:
        st.sidebar.markdown(f"🔄 Refresh Count: {refresh_count}")

        # Countdown progress bar until next refresh
        refresh_interval = refresh_interval_seconds  # seconds
        if "last_count" not in st.session_state or st.session_state["last_count"] != refresh_count:
            st.session_state["last_count"] = refresh_count
            st.session_state["last_refresh_time"] = datetime.now(pacific)

        elapsed = (datetime.now(pacific) - st.session_state["last_refresh_time"]).seconds
        progress = int((elapsed / refresh_interval) * 100)
        progress = min(max(progress, 0), 100)
        bar = st.sidebar.progress(progress)
        remaining = refresh_interval - elapsed if (refresh_interval - elapsed) > 0 else 0
        st.sidebar.markdown(f"⏳ {remaining}s until next refresh")


    # Hero Banner Title (replaces st.title)


    st.markdown(f"<div class='hero'>PUBLIC ENEMY LINES – {LEAGUE_NAME} TRAP BOARD<br><span class='date'>{current_pt.strftime('%B %d, %Y')}</span></div>", unsafe_allow_html=True)

    # ---- Daily flag rotation (PT) + midnight catch‑up ----
    pac = pacific_tz
    today_key = datetime.now(pac).strftime("%Y%m%d")
    if st.session_state.get("daily_key") != today_key:
        prev_key = st.session_state.get("daily_key")
        prev_close_done = st.session_state.get("close_sql_logged", False)
        prev_clv_done = st.session_state.get("clv_sql_logged", False)
        if prev_key and (prev_key != today_key) and (not prev_close_done or not prev_clv_done):
            try:
                ypath = Path("snapshots") / f"splits_snapshot_{prev_key}.csv"
                if ypath.exists():
                    df_y = pd.read_csv(ypath)
                    date_iso_prev = f"{prev_key[:4]}-{prev_key[4:6]}-{prev_key[6:]}"
                    log_close_snapshot(df_y, market="Moneyline", date_override=date_iso_prev)
                    st.session_state["close_sql_logged"] = True
                    clv_df_mid = compute_and_log_clv()
                    st.session_state["clv_df"] = clv_df_mid
                    st.session_state["clv_sql_logged"] = True
                    st.caption(f"🧮 Midnight catch-up CLV computed for {len(clv_df_mid) if clv_df_mid is not None else 0} picks from {prev_key}.")
                else:
                    logging.warning(f"[do_not_bet_ncaaf] Missing snapshot CSV for {prev_key}, cannot catch up.")
            except Exception as e:
                logging.error(f"[do_not_bet_ncaaf] midnight catch-up failed: {e}")
        st.session_state["daily_key"] = today_key
        st.session_state["open_sql_logged"] = False
        st.session_state["close_sql_logged"] = False
        st.session_state["clv_sql_logged"] = False

    # ---- CLV Automation Controls ----
    with st.sidebar.expander("CLV Automation", expanded=False):
        auto_enabled = st.checkbox("Enable auto CLOSE + CLV", value=st.session_state.get("auto_clv_enabled", True), key="auto_clv_enabled")
        close_hour = st.slider("Auto CLOSE hour (PT)", min_value=20, max_value=23, value=int(st.session_state.get("auto_close_hour", 23)), step=1, key="auto_close_hour")
        st.caption("After this local hour, the app logs CLOSE snapshots and computes CLV once per day.")



    # ——— Helper: fetch splits directly ———
    def fetch_dk_splits(event_group: int, date_range: str, market: str) -> pd.DataFrame:
        """Fetch DraftKings Network betting splits for the selected market.

        Parameters
        ----------
        event_group : int
            DraftKings event group id (e.g., 87637 for NCAAF).
        date_range : str
            "today", "tomorrow", or "n7days" (auto‑fallback if empty).
        market : str
            Market filter ("Moneyline", "Spread", "Total" or "All").

        Returns
        -------
        pd.DataFrame
            Columns: matchup, game_time, market, side, odds, %handle, %bets, update_time, edate_scope
        """

        base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"

        def _build_url(edate: str, page: int | None = None) -> str:
            params = {"tb_eg": event_group, "tb_edate": edate, "tb_emt": "0"}
            if page is not None:
                params["tb_page"] = str(page)
            return f"{base}?{urlencode(params)}"

        first_url = _build_url(date_range)

        def clean(text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", text, flags=re.I)
            return text.strip()

        def _get_html(url: str) -> str:
            # Try Playwright for the very first page; requests for everything else
            if url == first_url:
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page()
                        page.goto(url)
                        page.wait_for_selector("div.tb-se", timeout=10000)
                        html = page.content()
                        browser.close()
                        return html
                except Exception:
                    pass
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, headers=headers, timeout=20)
                resp.raise_for_status()
                return resp.text
            except Exception:
                return ""

        def _discover_page_urls(first_html: str, edate: str) -> list[str]:
            if not first_html:
                return [first_url]
            soup = BeautifulSoup(first_html, "html.parser")
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
            pac = pytz.timezone("America/Los_Angeles")
            now = datetime.now(pac)
            out: list[dict] = []
            for game in games:
                title_el = game.select_one("div.tb-se-title h5")
                if not title_el:
                    continue
                title = clean(title_el.get_text(strip=True))
                time_el = game.select_one("div.tb-se-title span")
                game_time = clean(time_el.get_text(strip=True)) if time_el else ""
                for section in game.select(".tb-market-wrap > div"):
                    head = section.select_one(".tb-se-head > div")
                    if not head:
                        continue
                    market_name = clean(head.get_text(strip=True))
                    if market not in ("All", "") and market_name != market:
                        continue
                    # Only parse known markets; adjust if you want Spread too
                    if market_name not in ("Moneyline", "Total", "Spread"):
                        continue
                    for row in section.select(".tb-sodd"):
                        side_el = row.select_one(".tb-slipline")
                        odds_el = row.select_one("a.tb-odd-s")
                        if not side_el or not odds_el:
                            continue
                        side = clean(side_el.get_text(strip=True))
                        raw_odds = clean(odds_el.get_text(strip=True))
                        odds = clean_odds(raw_odds.replace("−", "-"))
                        pct_texts = [s.strip().replace("%", "") for s in row.find_all(string=lambda t: "%" in t)]
                        handle_pct, bets_pct = (pct_texts + ["", ""])[:2]
                        out.append({
                            "matchup": title,
                            "game_time": game_time,
                            "market": market_name,
                            "side": side,
                            "odds": odds,
                            "%handle": handle_pct,
                            "%bets": bets_pct,
                            "update_time": now,
                        })
            return out

        # 1) pull first page, discover the rest
        first_html = _get_html(first_url)
        page_urls = _discover_page_urls(first_html, date_range)

        # 2) parse all pages
        records: list[dict] = []
        for url in page_urls:
            html = first_html if url == first_url else _get_html(url)
            records.extend(_parse_page(html))

        # 3) fallbacks if empty
        if not records:
            if date_range == "today":
                df = fetch_dk_splits(event_group, "tomorrow", market)
                if not df.empty:
                    df["edate_scope"] = "tomorrow"
                return df
            if date_range == "tomorrow":
                df = fetch_dk_splits(event_group, "n7days", market)
                if not df.empty:
                    df["edate_scope"] = "n7days"
                return df

        # Build dataframe
        cols = ["matchup", "game_time", "market", "side", "odds", "%handle", "%bets", "update_time"]
        df = pd.DataFrame(records, columns=cols)
        if not df.empty and "edate_scope" not in df.columns:
            df["edate_scope"] = date_range
        return df

    # ——— Sidebar controls ———
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        market = st.selectbox("Market", ["All", "Moneyline", "Spread", "Total"], index=1)
        if st.button("🔄 Fetch Current Splits"):
            st.session_state.clear()
            st.rerun()

        prev_file = st.file_uploader("Upload Previous Splits CSV", type="csv")
        handle_thresh = st.slider("Min Handle % Move", -100, 100, 0)
        bets_thresh  = st.slider("Min Bets % Move",   -100, 100, 0)

        if st.button("🔁 Reset Open Odds Snapshot"):
            current_df.to_csv("latest_snapshot.csv", index=False)
            st.success("Open odds snapshot reset.")
            st.session_state["loaded_snapshot"] = True
            st.session_state["open_dfs"]["Moneyline"] = current_df.copy()

        # --- Cool-Down Mode ----------------------------------------------
        cool_mode = st.checkbox("😌 Cool-Down Mode (slow refresh & hide urgency cues)", value=False)
        if cool_mode:
            st.session_state["refresh_interval_seconds"] = 600  # 10-minute cycle
            st.markdown(
                "<style>div[id^='stickyCTA'], div[data-testid='stProgress']{display:none !important;}</style>",
                unsafe_allow_html=True
            )

    # ——— Fetch current splits on each refresh ———

    with st.spinner("Fetching current splits..."):
        current_df = fetch_dk_splits(EVENT_GROUP, "today", market)
        scope = (
            current_df["edate_scope"].iloc[0]
            if (not current_df.empty and "edate_scope" in current_df.columns)
            else "today"
        )
        st.session_state["using_tomorrow"] = (scope == "tomorrow")
        st.session_state["using_n7days"] = (scope == "n7days")

    # --- Handle empty after fallback -----------------------------------------
    if current_df.empty:
        st.warning(
            "No DraftKings splits available for today **or** tomorrow. "
            "Extending the refresh interval to 30 minutes."
        )
        st.session_state["refresh_interval_seconds"] = 1800  # 30‑minute back‑off
        st.stop()
    else:
        # Restore fast refresh when data exists
        st.session_state["refresh_interval_seconds"] = 60

    # Display last fetched timestamp
    if "update_time" in current_df.columns:
        last_time = current_df["update_time"].max()
        formatted = last_time.strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.markdown(f"**Last Updated:** {formatted} PT")
        st.markdown(f"**Last Updated:** {formatted} PT")

    # Store initial odds snapshots per market
    snapshot_path = "latest_snapshot.csv"
    if "open_dfs" not in st.session_state:
        if os.path.exists(snapshot_path):
            try:
                snapshot_df = pd.read_csv(snapshot_path)
                st.session_state["open_dfs"] = {"Moneyline": snapshot_df}
                st.session_state["loaded_snapshot"] = True
            except Exception:
                st.session_state["open_dfs"] = {}
                st.session_state["loaded_snapshot"] = False
        else:
            st.session_state["open_dfs"] = {}
            st.session_state["loaded_snapshot"] = False
    # Initialize open_df for this market on first load
    open_dfs = st.session_state["open_dfs"]
    if market not in open_dfs:
        open_dfs[market] = current_df.copy()
    open_df = open_dfs[market]
    # --- Safety net for sessions started before the 5‑09 patch ---------------
    required_cols = {"matchup", "market", "side"}
    if not required_cols.issubset(open_df.columns):
        st.warning("Resetting internal snapshot (columns changed after patch).")
        open_df = current_df.copy()
        open_dfs[market] = open_df

    # Save latest snapshot to disk
    if not st.session_state.get("loaded_snapshot", False):
        current_df.to_csv(snapshot_path, index=False)

    # ▶️  OPEN‑odds snapshot → SQLite  (once per session, pinned to PT date)
    pac = pacific_tz
    today_key = datetime.now(pac).strftime("%Y%m%d")
    if not st.session_state.get('open_sql_logged', False):
        date_iso_today = f"{today_key[:4]}-{today_key[4:6]}-{today_key[6:]}"
        _log_snapshot(current_df.copy(), market, "OPEN", date_override=date_iso_today)
        st.session_state['open_sql_logged'] = True
        st.caption(f"📼 OPEN snapshot logged to {BETLOGS_DB}")

    # ---- Automated CLOSE snapshot + CLV (once per day) ----
    try:
        now_pt = datetime.now(pac)
        if st.session_state.get("auto_clv_enabled", True):
            # 1) trigger CLOSE after hour
            if (now_pt.hour >= int(st.session_state.get("auto_close_hour", 23))) and not st.session_state.get("close_sql_logged", False):
                date_iso_today = f"{today_key[:4]}-{today_key[4:6]}-{today_key[6:]}"
                log_close_snapshot(current_df.copy(), market, date_override=date_iso_today)
                st.session_state["close_sql_logged"] = True
                st.caption(f"📼 CLOSE snapshots logged automatically to {BETLOGS_DB}")
            # 2) compute CLV once
            if st.session_state.get("close_sql_logged", False) and not st.session_state.get("clv_sql_logged", False):
                clv_df_auto = compute_and_log_clv()
                st.session_state["clv_df"] = clv_df_auto
                st.session_state["clv_sql_logged"] = True
                st.caption(f"🧮 CLV computed &amp; saved automatically for {0 if clv_df_auto is None else len(clv_df_auto)} picks.")
    except Exception as e:
        logging.error(f"[do_not_bet_ncaaf] auto CLOSE/CLV failed: {e}")

    # ——— Compute Odds Movement from Open ———
    # Convert odds strings to ints
    def to_int_odds(o):
        """
        Convert odds value to integer. Handles strings with '+' or '-' and numeric types.
        """
        try:
            # Ensure value is string to replace symbols, then convert to int
            return int(str(o).replace("+", "").replace("−", "-"))
        except Exception:
            return 0
    merged_odds = pd.merge(
        current_df, open_df,
        on=["matchup","market","side"],
        suffixes=("_cur","_open")
    )
    merged_odds["odds_move"] = merged_odds["odds_cur"].apply(to_int_odds) - merged_odds["odds_open"].apply(to_int_odds)
    # Bring odds_move into current_df for recommendations
    current_df = pd.merge(
        current_df, 
        merged_odds[["matchup","market","side","odds_move"]],
        on=["matchup","market","side"]
    )

    current_df["is_primetime"] = current_df.apply(is_primetime_game, axis=1)

    # ===== Trap/Edge Verdict Tagging Logic (Trap Summary block moved up here) =====
    def verdict(row):
        try:
            bets = float(row["%bets"])
            handle = float(row["%handle"])
            move = int(row["odds_move"])
        except:
            return ""

        if bets >= 75 and handle < bets and move == 0:
            return "🚨 Public Trap"
        elif handle - bets >= 10:
            return "💰 Sharp Side"
        elif 48 <= bets <= 52 and abs(move) <= 5:
            return "⛔ Do Not Bet"
        else:
            return ""



    # ===== Trap Score, Avoid Tag, Clown Pick, Trap Explanation Enhancements =====
    current_df["verdict"] = current_df.apply(verdict, axis=1)

    # Trap Score Columns
    current_df["bets_minus_handle"] = current_df["%bets"].astype(float) - current_df["%handle"].astype(float)
    current_df["value_diff_z"] = zscore(current_df["value_diff"].fillna(0)) if "value_diff" in current_df.columns else 0

    def trap_score(row):
        score = 0
        try:
            bets = float(row["%bets"])
            handle = float(row["%handle"])
            move = abs(int(row["odds_move"]))
            is_primetime = row.get("is_primetime", False)
            value_z = row.get("value_diff_z", 0)

            score += (bets - handle) / 10.0
            score += move / 10.0
            score += 0.5 if is_primetime else 0
            score += value_z

        except Exception:
            pass

        return round(score, 1)

    current_df["trap_score"] = current_df.apply(trap_score, axis=1)

    def clown_score(row):
        try:
            bets = float(row["%bets"])
            handle = float(row["%handle"])
            move = int(row["odds_move"])
            ratio = bets / (handle + 0.01)
            score = (bets - handle) / 10 + ratio
            score += 1 if move >= 0 else 0
            return round(min(score, 10), 1)
        except:
            return 0

    current_df["clown_score"] = current_df.apply(clown_score, axis=1)

    # Do Not Bet Alert Tag
    def avoid_tag(row):
        try:
            if row["verdict"] == "" and 70 <= float(row["%bets"]) <= 80 and row["odds_move"] == 0:
                return "❌ Avoid - Public Leaning, No Edge"
        except Exception:
            return ""
        return ""

    current_df["avoid_tag"] = current_df.apply(avoid_tag, axis=1)

    # Clown Pick Label
    def clown_pick(row):
        try:
            if float(row["%bets"]) >= 90 and row["odds_move"] >= 0:
                return "🤡 Clown Pick of the Day"
        except Exception:
            return ""
        return ""

    current_df["clown_tag"] = current_df.apply(clown_pick, axis=1)

    def generate_clown_tweet(df):
        if df.empty:
            return ""
        top = df.sort_values("trap_score", ascending=False).iloc[0]
        return f"🤡 Clown Pick: {top['side']} ({top['odds']}) is getting {top['%bets']}% of bets. Odds haven’t moved. Books aren’t scared. #DoNotBetClub"

    st.subheader("🤡 Clown Pick Tweet")
    tweet_text = generate_clown_tweet(current_df[current_df["clown_tag"] != ""])
    if tweet_text:
        st.code(tweet_text)
    else:
        st.markdown("_No clown pick today._")

    # Trap Explanation Text
    def trap_explanation(row):
        if row["verdict"] == "🚨 Public Trap":
            return f"⚠️ {row['%bets']}% of bets on {row['side']}, but odds haven’t moved. Books inviting action."
        return ""

    current_df["trap_explanation"] = current_df.apply(trap_explanation, axis=1)

    # ===== Trap/Sharp/Fade/Back summary metrics (moved here after current_df is defined) =====
    num_traps = (current_df["verdict"] == "🚨 Public Trap").sum()
    num_sharp = (current_df["verdict"] == "💰 Sharp Side").sum()
    # Additional signal counts
    num_donotbet    = int((current_df["verdict"] == "⛔ Do Not Bet").sum())
    num_big_movers  = int(current_df["big_mover_flag"].sum()) if "big_mover_flag" in current_df.columns else 0
    num_divergences = int(current_df["divergence_flag"].sum()) if "divergence_flag" in current_df.columns else 0
    num_rlm         = int(current_df["rlm_flag"].sum()) if "rlm_flag" in current_df.columns else 0
    num_kelly       = int(current_df["kelly_trigger"].sum()) if "kelly_trigger" in current_df.columns else 0
    # insights_df will be defined later; for now, create placeholder values
    num_fades = None
    num_backs = None


    # ▶️  Attempt CLOSE‑odds snapshot (runs every refresh; INSERT OR IGNORE prevents dupes)
    _maybe_log_close_snapshot(current_df.copy(), market)

    # --- Tabbed layout skeleton ----------------------------------------------
    if "recs" not in st.session_state:
        st.session_state["recs"] = pd.DataFrame()
    slate_tab, snapshot_tab, signals_tab, graphics_tab, deep_tab = st.tabs(
        ["🗓️ Today's Slate", "📊 Snapshot", "📈 Signals", "🖼 Graphics", "🔬 Deep Dive"]
    )

    # --- Today's Slate Tab ---
    with slate_tab:

        pac = pytz.timezone("America/Los_Angeles")
        using_tomorrow = st.session_state.get("using_tomorrow", False)
        using_n7 = st.session_state.get("using_n7days", False)
        if using_n7:
            display_date = f"{datetime.now(pac).strftime('%b %d')} → {(datetime.now(pac)+timedelta(days=7)).strftime('%b %d')}"
            header_label = "Next 7 Days Slate"
        elif using_tomorrow:
            display_date = (datetime.now(pac) + timedelta(days=1)).strftime('%B %d, %Y')
            header_label = "Tomorrow's Slate"
        else:
            display_date = datetime.now(pac).strftime('%B %d, %Y')
            header_label = "Today's Slate"
        st.subheader(f"🗓️ {header_label}")
        st.markdown(f"**Date:** {display_date}")
    
        # Total games on slate
        total_games = current_df["matchup"].nunique()
        st.markdown(f"**Total Games:** {total_games}")
    
        # Raw unfiltered odds data
        st.markdown("**Raw Odds Data:**")
        st.dataframe(current_df, use_container_width=True)

    # --- SNAPSHOT TAB ---
    with snapshot_tab:
        with st.sidebar.expander("CLV Tools", expanded=False):
            if st.button("📥 Log CLOSE snapshots now"):
                date_iso_today = f"{today_key[:4]}-{today_key[4:6]}-{today_key[6:]}"
                log_close_snapshot(current_df.copy(), market, date_override=date_iso_today)
                st.session_state["close_sql_logged"] = True
                st.success("CLOSE snapshots recorded to SQLite (line_snapshots).")
            if st.button("🧮 Compute & Save CLV"):
                clv_df = compute_and_log_clv()
                st.session_state["clv_df"] = clv_df
                st.session_state["clv_sql_logged"] = True
                st.success(f"CLV computed for {0 if clv_df is None else len(clv_df)} picks and saved to clv_logs.")
            st.markdown(f"**Status:** OPEN={'✅' if st.session_state.get('open_sql_logged') else '❌'} • CLOSE={'✅' if st.session_state.get('close_sql_logged') else '❌'} • CLV={'✅' if st.session_state.get('clv_sql_logged') else '❌'}")
            st.caption(f"Daily key: {st.session_state.get('daily_key')}")
        # Condensed snapshot table with emojis
        table = "| Metric | Count |\n|---|---:|\n"
        table += f"| 🚨 Traps | {num_traps} |\n"
        table += f"| 💰 Sharp Sides | {num_sharp} |\n"
        table += f"| ⛔ Do-Not-Bets | {num_donotbet} |\n"
        table += f"| 📈 Big Movers | {num_big_movers} |\n"
        table += f"| 🔀 Divergences | {num_divergences} |\n"
        table += f"| 🔄 RLMs | {num_rlm} |\n"
        table += f"| ⚡ Kelly Triggers | {num_kelly} |\n"
        st.markdown(table, unsafe_allow_html=True)

        # --- Full Markdown Report section -------------------------------------
        if "full_report_generated" not in st.session_state:
            st.session_state["full_report_generated"] = False

        if st.button("📄 Generate Full Trap Report", key="gen_full_report"):
            full_md = generate_full_report(current_df, st.session_state.get("recs", pd.DataFrame()))
            st.session_state["full_report"] = full_md
            st.session_state["full_report_generated"] = True

        if st.button("💬 Generate Sharp‑Chat Log", key="gen_chat_report"):
            chat_md = generate_chat_report(current_df, st.session_state.get("recs", pd.DataFrame()))
            st.session_state["chat_report"] = chat_md
            st.session_state["chat_generated"] = True

        if st.session_state.get("full_report_generated"):
            st.markdown("### 📰 Public Enemy Lines – Full Report")
            st.markdown(st.session_state["full_report"], unsafe_allow_html=True)
            st.download_button(
                "Download Full Report (Markdown)",
                data=st.session_state["full_report"],
                file_name=f"pel_full_report_{datetime.now(pacific_tz).strftime('%Y%m%d')}.md",
                mime="text/markdown",
                key="dl_full_report"
            )
            # PDF export logic for full report
            try:
                def md_to_pdf(md_content, output_file):
                    html = md2_markdown(md_content)
                    with open(output_file, "w+b") as result_file:
                        pisa.CreatePDF(html, dest=result_file)
                # Ensure exports dir exists
                os.makedirs("exports", exist_ok=True)
                date_str = datetime.now(pacific_tz).strftime("%Y%m%d")
                md_to_pdf(
                    st.session_state["full_report"],
                    f"exports/pel_full_report_{date_str}.pdf"
                )
                st.success(f"PDF version exported to exports/pel_full_report_{date_str}.pdf")
            except ImportError as e:
                st.warning("PDF export requires `markdown2` and `xhtml2pdf`. Please install these packages to enable PDF downloads.")
            except Exception as e:
                st.warning(f"PDF export failed: {e}")

        if st.session_state.get("chat_generated"):
            st.markdown("### 💬 Sharp‑Chat Log")
            st.markdown(st.session_state["chat_report"], unsafe_allow_html=True)
            st.download_button(
                "Download Chat Log (Markdown)",
                data=st.session_state["chat_report"],
                file_name=f"pel_chat_{datetime.now(pacific_tz).strftime('%Y%m%d')}.md",
                mime="text/markdown",
                key="dl_chat_report"
            )
            # PDF export logic for chat report
            try:
                def md_to_pdf(md_content, output_file):
                    html = md2_markdown(md_content)
                    with open(output_file, "w+b") as result_file:
                        pisa.CreatePDF(html, dest=result_file)
                os.makedirs("exports", exist_ok=True)
                date_str = datetime.now(pacific_tz).strftime("%Y%m%d")
                md_to_pdf(
                    st.session_state["chat_report"],
                    f"exports/pel_chat_{date_str}.pdf"
                )
                st.success(f"PDF version exported to exports/pel_chat_{date_str}.pdf")
            except ImportError as e:
                st.warning("PDF export requires `markdown2` and `xhtml2pdf`. Please install these packages to enable PDF downloads.")
            except Exception as e:
                st.warning(f"PDF export failed: {e}")
    # The Fades/Backs counts are shown after insights_df is defined below

    # ===== Trap Summary section (moved up) =====
    top_traps = current_df[current_df["verdict"] == "🚨 Public Trap"]
    if not top_traps.empty:
        st.markdown("### ⚠️ Trap Game Summary")

    # --- Combined Signal Definitions Expander ---
    with st.expander("ℹ️ Signal Definitions", expanded=False):
        st.markdown("""
    **🚨 Public Trap** – Majority of bets on one side with no (or opposite) line movement.  
    **💰 Sharp Side** – Handle % exceeds bet %, indicating professional money.  
    **Fade Public** – Public bet % far exceeds implied win probability.  
    **Back Public** – Rare spot where public % is low but climbing *with* line movement.
    """)
    # --- CTA Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;font-size:20px;'>🚀 Want these traps by noon PT every day? "
        "<a href='https://yourexamplelink.com' style='color:#00E6FF;'>Subscribe here</a> • "
        "<a href='#' style='color:#00E6FF;'>Share today’s report</a></div>",
        unsafe_allow_html=True
    )




    # Add Signals Tab Wrapper
    with signals_tab:
        st.subheader("Current Splits")

        signal_view = st.selectbox(
            "Filter Signals",
            [
              "All",
              "🚨 Public Trap",
              "💰 Sharp Side",
              "⛔ Do Not Bet",
              "📈 Big Mover",
              "🔀 Divergence",
              "🔄 RLM",
              "⚡ Kelly Trigger"
            ]
        )


        # ====== Professional NCAAF Betting Insights (moved up to ensure insights_df is defined before merge) ======
    def calc_implied_prob(odds_str):
        match = re.search(r"[-+]?\d+", str(odds_str))
        if not match:
            return 0
        odds = int(match.group(0))
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return -odds / (100.0 - odds)

    def square_signal(row):
        try:
            bets = float(row["%bets"])
            implied = float(row.get("implied_prob", 0))
            return round((bets / 100 - implied) * 100, 1)
        except:
            return 0

    current_df["square_signal"] = current_df.apply(square_signal, axis=1)

    insights_df = current_df.copy()
    insights_df["implied_prob"] = insights_df["odds"].apply(calc_implied_prob)
    # Convert %bets to fraction
    insights_df["pct_bets_frac"] = insights_df["%bets"].astype(float) / 100.0
    # Compute value differential: public % minus implied %
    # Ensure value_diff is numeric before using nlargest/nsmallest
    insights_df["value_diff"] = insights_df["pct_bets_frac"] - insights_df["implied_prob"]
    insights_df["value_diff"] = pd.to_numeric(insights_df["value_diff"], errors="coerce").fillna(0)

    # Top value opportunities where public is overbetting (value_diff positive)
    insights_df["value_diff"] = pd.to_numeric(insights_df["value_diff"], errors="coerce").fillna(0)
    top_over = insights_df.nlargest(3, "value_diff")
    # Top value opportunities where public is underbetting (value_diff negative)
    top_under = insights_df.nsmallest(3, "value_diff")

    st.markdown("#### Top Value Opportunities (Public Overbetting)")
    if aggrid_available:
        go_over = GridOptionsBuilder.from_dataframe(top_over)
        go_over.configure_default_column(resizable=True, filter=True, sortable=True)
        AgGrid(top_over, gridOptions=go_over.build(), theme="material", height=200, key="top_over_grid")
    else:
        st.table(top_over[["matchup", "market", "side", "odds", "%bets", "implied_prob", "value_diff"]])

    st.markdown("#### Top Value Opportunities (Public Underbetting)")
    if aggrid_available:
        go_under = GridOptionsBuilder.from_dataframe(top_under)
        go_under.configure_default_column(resizable=True, filter=True, sortable=True)
        AgGrid(top_under, gridOptions=go_under.build(), theme="material", height=200, key="top_under_grid")
    else:
        st.table(top_under[["matchup", "market", "side", "odds", "%bets", "implied_prob", "value_diff"]])

    # Commentary on insights
    st.markdown("#### Insight Commentary")
    for _, row in top_over.iterrows():
        st.markdown(f"- 🟢 **{row['side']}** vs **{row['matchup']}**: Public bets {row['%bets']}% vs implied {row['implied_prob']*100:.1f}% → overbet by {row['value_diff']*100:.1f}%.")
    for _, row in top_under.iterrows():
        st.markdown(f"- 🔴 **{row['side']}** vs **{row['matchup']}**: Public bets {row['%bets']}% vs implied {row['implied_prob']*100:.1f}% → underbet by {-(row['value_diff']*100):.1f}%.")

    # ——— Advanced NCAAF Betting Systems & Trends ———
    st.subheader("Advanced Betting Systems & Trends")

    # Public vs. Implied ratio (public % bets divided by implied probability)
    insights_df["bets_to_implied_ratio"] = insights_df["pct_bets_frac"] / insights_df["implied_prob"]

    # Contrarian signal: Fade Public if ratio >1.2, Back Public if ratio <0.8
    def contrarian_sig(ratio):
        if ratio > 1.2:
            return "Fade Public"
        elif ratio < 0.8:
            return "Back Public"
        else:
            return "No Clear Signal"

    insights_df["contrarian_signal"] = insights_df["bets_to_implied_ratio"].apply(contrarian_sig)

    # Now that insights_df is defined, we can count Fades/Backs and show the full summary
    num_fades = (insights_df["contrarian_signal"] == "Fade Public").sum()
    num_backs = (insights_df["contrarian_signal"] == "Back Public").sum()
    st.markdown(f"### 📊 Today: {num_traps} Traps • {num_sharp} Sharp Sides • {num_fades} Fades • {num_backs} Backs")

    # Ensure bets_to_implied_ratio is numeric before using nlargest/nsmallest
    insights_df["bets_to_implied_ratio"] = pd.to_numeric(insights_df["bets_to_implied_ratio"], errors="coerce").fillna(0)
    # Top 3 Contrarian Opportunities
    top_fade  = insights_df[insights_df["contrarian_signal"]=="Fade Public"].nlargest(3, "bets_to_implied_ratio")
    insights_df["bets_to_implied_ratio"] = pd.to_numeric(insights_df["bets_to_implied_ratio"], errors="coerce").fillna(0)
    top_back  = insights_df[insights_df["contrarian_signal"]=="Back Public"].nsmallest(3, "bets_to_implied_ratio")

    st.markdown("#### Top Fade Public Opportunities")
    st.table(top_fade[["matchup", "market", "side", "odds", "%bets", "implied_prob", "bets_to_implied_ratio"]])

    st.markdown("#### Top Back Public Opportunities")
    st.table(top_back[["matchup", "market", "side", "odds", "%bets", "implied_prob", "bets_to_implied_ratio"]])



    # Safety check: Ensure all required metrics exist in insights_df, fill with zeros if missing
    for metric in ["value_diff", "bets_to_implied_ratio", "win_prob_gap", "odds_move"]:
        if metric not in insights_df.columns:
            insights_df[metric] = 0

    # Convert to percentile ranks
    for metric in ["value_diff", "bets_to_implied_ratio", "win_prob_gap", "odds_move"]:
        colname = f"pctl_{metric}"
        insights_df[colname] = rankdata(insights_df[metric].fillna(0), method='average') / len(insights_df)

    # Weighted composite scoring (manual weights can be tuned)
    insights_df["confidence"] = (
        0.4 * insights_df["pctl_value_diff"] +
        0.3 * insights_df["pctl_bets_to_implied_ratio"] +
        0.2 * insights_df["pctl_win_prob_gap"] +
        0.1 * insights_df["pctl_odds_move"]
    ) * 5  # Scale to 0–5

    st.subheader("📈 Confidence Score Distribution")
    insights_df["confidence"] = pd.to_numeric(insights_df["confidence"], errors="coerce").fillna(0)
    conf_chart = alt.Chart(insights_df).mark_bar().encode(
        x=alt.X('confidence:Q', bin=alt.Bin(maxbins=20)),
        y='count()',
        tooltip=['confidence']
    ).properties(width=600, height=300)
    st.altair_chart(conf_chart, use_container_width=True)

    if "clv_df" in st.session_state and isinstance(st.session_state["clv_df"], pd.DataFrame) and not st.session_state["clv_df"].empty:
        st.subheader("📈 CLV Results (Your Picks vs Closing Line)")
        show_cols = ["matchup","market","side","entry_odds","close_odds","entry_total","close_total","clv_prob_pp","clv_line_move"]
        try:
            st.dataframe(st.session_state["clv_df"][show_cols], use_container_width=True)
        except Exception:
            st.dataframe(st.session_state["clv_df"], use_container_width=True)

    # Merge computed confidence from insights_df into current_df
    if "confidence" in insights_df.columns:
        current_df = current_df.merge(
            insights_df[["matchup", "market", "side", "confidence"]],
            on=["matchup", "market", "side"],
            how="left"
        )
        current_df["confidence"] = current_df["confidence"].fillna(0.0)
    else:
        current_df["confidence"] = 0.0

    # Ensure value_diff_z exists and has no nulls
    if "value_diff_z" in current_df.columns:
        current_df["value_diff_z"] = current_df["value_diff_z"].fillna(0.0)
    else:
        current_df["value_diff_z"] = 0.0

    # Merge Kelly stake_pct from insights_df
    if "stake_pct" in insights_df.columns:
        current_df = current_df.merge(
            insights_df[["matchup", "market", "side", "stake_pct"]],
            on=["matchup", "market", "side"],
            how="left"
        )
        current_df["stake_pct"] = current_df["stake_pct"].fillna(0.0)
    else:
        current_df["stake_pct"] = 0.0

    # ——— New edge signals ———
    # Big Mover: absolute odds change ≥ 5
    current_df["big_mover_flag"]   = current_df["odds_move"].abs() >= 5
    # Divergence: public vs implied Z-score ≥ 1.5
    current_df["divergence_flag"]  = current_df["value_diff_z"] >= 1.5
    # Kelly Trigger: suggested stake ≥ 3%
    current_df["kelly_trigger"]    = current_df["stake_pct"] >= 0.03
    # ——— Reverse Line Movement (RLM) flag ———
    current_df["rlm_flag"] = current_df.apply(
        lambda row: (
            (row.get("contrarian_signal") == "Fade Public" and row.get("odds_move", 0) <= -5)
         or (row.get("contrarian_signal") == "Back Public" and row.get("odds_move", 0) >= 5)
        ),
        axis=1
    )


    # Display each signal/game as a card in a 3-column grid
    records = current_df.to_dict('records')
    # Apply filter based on selected signal type
    # Apply filter based on selected signal type
    if signal_view == "All":
        pass
    elif signal_view in ["🚨 Public Trap","💰 Sharp Side","⛔ Do Not Bet"]:
        records = [r for r in records if r.get("verdict") == signal_view]
    elif signal_view == "📈 Big Mover":
        records = [r for r in records if r.get("big_mover_flag")]
    elif signal_view == "🔀 Divergence":
        records = [r for r in records if r.get("divergence_flag")]
    elif signal_view == "🔄 RLM":
        records = [r for r in records if r.get("rlm_flag")]
    elif signal_view == "⚡ Kelly Trigger":
        records = [r for r in records if r.get("kelly_trigger")]
    # Optionally, show info if no records after filter
    if not records:
        st.info(f"No signals to display for: {signal_view}")
    cols = st.columns(3)
    for i, row in enumerate(records):
        col = cols[i % 3]
        with col:
            # Render card content with HTML so CSS applies
            card_html = f"""<div class='card'>
    <h3>{row['matchup']}</h3>
    <p><strong>Side:</strong> {row['side']} ({row['odds']})</p>
    <p><strong>Trap Score:</strong> {row['trap_score']}/10 &nbsp; <strong>Confidence:</strong> {row.get('confidence', 0.0):.1f}/5</p>
    <p><strong>% Bets:</strong> {row['%bets']}% &nbsp; <strong>% Handle:</strong> {row['%handle']}% &nbsp; <strong>Move:</strong> {row['odds_move']}</p>
    </div>"""
            st.markdown(card_html, unsafe_allow_html=True)
            # Buttons remain functional beneath styled block
            if st.button(f"Generate Posts {i}", key=f"gen_{i}"):
                posts = generate_social_posts(pd.DataFrame([row]))
                st.code(posts['tweet'], language="text")
                st.code(posts['ig'], language="text")
                st.code(posts['tiktok'], language="text")
            st.download_button(
                "Download Data",
                data=pd.DataFrame([row]).to_csv(index=False),
                file_name=f"{row['matchup'].replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"download_{i}"
            )

    with deep_tab:
        # Collapse detailed analytics into expander
        with st.expander("🔽 More Analytics"):
            # ===== Export structured signal data to CSV and log by timestamp =====
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{LEAGUE_NAME.lower()}_signals_{datetime.now(pacific).strftime('%Y%m%d_%H%M')}.csv")
            current_df.to_csv(log_file, index=False)

            # ===== Streamlit checkbox filters to explore signals =====
            st.sidebar.markdown("### Verdict Filter")
            signal_filter = st.sidebar.selectbox("Show only...", ["All", "🚨 Public Trap", "💰 Sharp Side", "⛔ Do Not Bet"])
            filtered_df = current_df
            if signal_filter != "All":
                filtered_df = current_df[current_df["verdict"] == signal_filter]

            # ===== Sidebar CTA links for sharing and subscriptions =====
            st.sidebar.markdown("### 📬 Share & Subscribe")
            st.sidebar.markdown("[🔗 Share Today’s Report](#)")
            st.sidebar.markdown("[💌 Subscribe to Trap Alerts](#)")

            # ===== Download button for filtered trap/edge data =====
            st.download_button(
                "📥 Download Current Signal Data",
                data=filtered_df.to_csv(index=False),
                file_name="filtered_signals.csv",
                mime="text/csv"
            )


            # ——— Historical Team Trends Integration ———
            # Built-in win/loss & MOV stats since 2020
            win_stats = {
        "LA Dodgers": {"win_pct": 0.643, "MOV": 1.5, "run_line_diff": 0.2},
        "Atlanta":    {"win_pct": 0.583, "MOV": 0.9, "run_line_diff": 0.0},
        "Houston":    {"win_pct": 0.576, "MOV": 0.9, "run_line_diff": 0.1},
        "NY Yankees": {"win_pct": 0.562, "MOV": 0.7, "run_line_diff": -0.2},
        "Tampa Bay":  {"win_pct": 0.560, "MOV": 0.5, "run_line_diff": 0.2},
        "Milwaukee":  {"win_pct": 0.550, "MOV": 0.4, "run_line_diff": 0.0},
        "Philadelphia":{"win_pct":0.544, "MOV":0.4, "run_line_diff":-0.1},
        "San Diego":  {"win_pct":0.540, "MOV":0.5, "run_line_diff":-0.1},
        "Seattle":    {"win_pct":0.539, "MOV":0.2, "run_line_diff":0.2},
        "SF Giants":  {"win_pct":0.535, "MOV":0.3, "run_line_diff":0.1},
        "Cleveland":  {"win_pct":0.531, "MOV":0.1, "run_line_diff":0.1},
        "Toronto":    {"win_pct":0.526, "MOV":0.3, "run_line_diff":-0.1},
        "NY Mets":    {"win_pct":0.525, "MOV":0.3, "run_line_diff":-0.1},
        "St. Louis":  {"win_pct":0.516, "MOV":0.0, "run_line_diff":0.1},
        "Minnesota":  {"win_pct":0.500, "MOV":0.1, "run_line_diff":-0.3},
        "Boston":     {"win_pct":0.500, "MOV":0.0, "run_line_diff":-0.1},
        "Chi Cubs":   {"win_pct":0.491, "MOV":0.0, "run_line_diff":0.1},
        "Baltimore":  {"win_pct":0.487, "MOV":-0.3,"run_line_diff":0.1},
        "Detroit":    {"win_pct":0.478, "MOV":-0.3,"run_line_diff":0.4},
        "Cincinnati":{"win_pct":0.473, "MOV":-0.2,"run_line_diff":0.1},
        "Arizona":    {"win_pct":0.464, "MOV":-0.2,"run_line_diff":0.2},
        "Texas":      {"win_pct":0.457, "MOV":-0.3,"run_line_diff":0.0},
        "Kansas City":{"win_pct":0.442, "MOV":-0.5,"run_line_diff":0.2},
        "LA Angels":  {"win_pct":0.440, "MOV":-0.6,"run_line_diff":-0.3},
        "Miami":      {"win_pct":0.438, "MOV":-0.7,"run_line_diff":-0.1},
        "Sacramento": {"win_pct":0.430, "MOV":-0.8,"run_line_diff":-0.1},
        "Chi Sox":    {"win_pct":0.429, "MOV":-0.5,"run_line_diff":-0.3},
        "Pittsburgh": {"win_pct":0.410, "MOV":-1.0,"run_line_diff":-0.1},
        "Washington":{"win_pct":0.409, "MOV":-0.9,"run_line_diff":0.0},
        "Colorado":   {"win_pct":0.395, "MOV":-1.2,"run_line_diff":-0.1}
    }

            # Built-in over/under trends since 2020
            ou_stats = {
        "LA Dodgers":{"over_pct":0.522, "under_pct":0.478, "total_diff":0.5},
        "LA Angels": {"over_pct":0.520, "under_pct":0.480, "total_diff":0.5},
        "Seattle":   {"over_pct":0.519, "under_pct":0.481, "total_diff":0.5},
        "Arizona":   {"over_pct":0.513, "under_pct":0.487, "total_diff":0.8},
        "Baltimore":{"over_pct":0.513, "under_pct":0.487, "total_diff":0.5},
        "Texas":     {"over_pct":0.508, "under_pct":0.492, "total_diff":0.5},
        "Boston":    {"over_pct":0.506, "under_pct":0.495, "total_diff":0.7},
        "Minnesota": {"over_pct":0.505, "under_pct":0.495, "total_diff":0.5},
        "Philadelphia":{"over_pct":0.505,"under_pct":0.495,"total_diff":0.7},
        "Pittsburgh":{"over_pct":0.504,"under_pct":0.496,"total_diff":0.4},
        "Miami":     {"over_pct":0.499, "under_pct":0.501, "total_diff":0.6},
        "Tampa Bay":{"over_pct":0.497, "under_pct":0.503, "total_diff":0.4},
        "Atlanta":   {"over_pct":0.497, "under_pct":0.503, "total_diff":0.4},
        "Washington":{"over_pct":0.496,"under_pct":0.504,"total_diff":0.6},
        "San Diego":{"over_pct":0.495, "under_pct":0.505,"total_diff":0.4},
        "NY Mets":   {"over_pct":0.495, "under_pct":0.506, "total_diff":0.5},
        "SF Giants": {"over_pct":0.493, "under_pct":0.507, "total_diff":0.5},
        "Milwaukee": {"over_pct":0.492, "under_pct":0.508, "total_diff":0.3},
        "Sacramento":{"over_pct":0.491,"under_pct":0.509,"total_diff":0.5},
        "Chi Cubs":  {"over_pct":0.490, "under_pct":0.510, "total_diff":0.6},
        "Toronto":   {"over_pct":0.489, "under_pct":0.511, "total_diff":0.4},
        "Cincinnati":{"over_pct":0.486,"under_pct":0.514,"total_diff":0.4},
        "St. Louis":{"over_pct":0.486,"under_pct":0.514,"total_diff":0.4},
        "NY Yankees":{"over_pct":0.485,"under_pct":0.515,"total_diff":0.4},
        "Houston":   {"over_pct":0.476, "under_pct":0.524, "total_diff":0.3},
        "Cleveland":{"over_pct":0.476, "under_pct":0.524,"total_diff":0.2},
        "Detroit":   {"over_pct":0.471, "under_pct":0.529,"total_diff":0.2},
        "Chi Sox":   {"over_pct":0.464, "under_pct":0.536,"total_diff":0.3},
        "Colorado":  {"over_pct":0.464, "under_pct":0.536,"total_diff":0.1},
        "Kansas City":{"over_pct":0.463,"under_pct":0.537,"total_diff":0.2}
    }

            # Map historical stats into insights_df
            insights_df["win_pct"]        = insights_df["side"].map(lambda t: win_stats.get(t, {}).get("win_pct", 0))
            insights_df["MOV"]            = insights_df["side"].map(lambda t: win_stats.get(t, {}).get("MOV", 0))
            insights_df["run_line_diff"]  = insights_df["side"].map(lambda t: win_stats.get(t, {}).get("run_line_diff", 0))
            insights_df["over_pct"]       = insights_df["side"].map(lambda t: ou_stats.get(t, {}).get("over_pct", 0))
            insights_df["under_pct"]      = insights_df["side"].map(lambda t: ou_stats.get(t, {}).get("under_pct", 0))
            insights_df["total_diff"]     = insights_df["side"].map(lambda t: ou_stats.get(t, {}).get("total_diff", 0))

            # Compute refined metrics
            insights_df["win_prob_gap"] = insights_df["win_pct"] - insights_df["implied_prob"]
            insights_df["total_bias"]   = insights_df["over_pct"] - 0.5

            # Refined signals based on historical trends and market
            def refined_signal(row):
                if row["market"] == "Moneyline" and row["win_prob_gap"] > 0.05:
                    return "Strong Moneyline Value"
                if row["market"] == "Total" and row["total_bias"] > 0.05:
                    return "Lean Over"
                return "No Clear Edge"

            insights_df["refined_signal"] = insights_df.apply(refined_signal, axis=1)

            # Display refined opportunities
            st.markdown("#### Top Refined Betting Opportunities")
            top_refined = insights_df[insights_df["refined_signal"] != "No Clear Edge"]

            if top_refined.empty:
                # Fallback logic: lower thresholds
                fallback_refined = insights_df[
                    (insights_df["win_prob_gap"] > 0.025) |
                    (insights_df["total_bias"] > 0.025)
                ]
                fallback_refined["refined_signal"] = fallback_refined.apply(refined_signal, axis=1)
                top_refined = fallback_refined
            st.table(top_refined[[
                "matchup","market","side","odds","%bets","implied_prob",
                "win_pct","win_prob_gap","over_pct","total_bias","refined_signal"
            ]])

            #
            # --- Kelly Criterion stake sizing ---
            def american_to_decimal(odds_str):
                match = re.search(r"[-+]?\d+", str(odds_str))
                if not match:
                    return 0
                o = int(match.group(0))
                return 1 + o/100.0 if o > 0 else 1 + 100.0/abs(o)

            def kelly_fraction(p, b):
                q = 1 - p
                return max((b * p - q) / b, 0)

            # Calculate decimal odds, edge and Kelly fraction
            insights_df["decimal_odds"] = insights_df["odds"].apply(american_to_decimal)
            insights_df["edge"] = insights_df["implied_prob"] - insights_df["pct_bets_frac"]
            insights_df["kelly_f"] = insights_df.apply(
                lambda r: kelly_fraction(r["implied_prob"], r["decimal_odds"] - 1),
                axis=1
            )
            # Cap stake at 5% of bankroll
            insights_df["stake_pct"] = insights_df["kelly_f"].clip(0, 0.05)

            #
            # --- Reverse Line Movement (RLM) Confirmation ---
            def rlm_signal(row):
                if row["contrarian_signal"] == "Fade Public" and row["odds_move"] <= -5:
                    return "Fade Public"
                if row["contrarian_signal"] == "Back Public" and row["odds_move"] >= 5:
                    return "Back Public"
                return ""

            insights_df["rlm_signal"] = insights_df.apply(rlm_signal, axis=1)


            # ——— Bet Recommendations ———
            st.subheader("Bet Recommendations")
            # Build a recommendation column: prefer refined_signal, otherwise rlm_signal (if present), otherwise ""
            insights_df["bet_recommendation"] = insights_df.apply(
                lambda r: r["refined_signal"]
                          if r["refined_signal"] != "No Clear Edge"
                          else (r["rlm_signal"]
                                if r["rlm_signal"] != ""
                                else ""),
                axis=1
            )

            # Filter only actionable recommendations, with progressive fallback logic and stake floor
            recs = insights_df[
                (insights_df["bet_recommendation"] != "") &
                (insights_df["stake_pct"] > 0) &
                (insights_df["confidence"] >= 0)
            ]
            if recs.empty:
                recs = insights_df[
                    (insights_df["bet_recommendation"] != "") &
                    (insights_df["stake_pct"] > 0)
                ]
            if recs.empty:
                recs = insights_df[insights_df["bet_recommendation"] != ""]

            # Ensure a minimum 1% stake for any actionable recommendation
            recs.loc[recs["stake_pct"] <= 0, "stake_pct"] = 0.01
            st.session_state["recs"] = recs.copy()

            if len(recs) > 0:
                recs = recs.sort_values("confidence", ascending=False).head(5)
                # --- KPI Row -------------------------------------------------------
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("🚨 Traps", int(num_traps))
                kpi2.metric("💰 Sharp Sides", int(num_sharp))
                proj_roi = f"{(recs.get('edge', recs['value_diff']).mean()*100):.1f}%"
                kpi3.metric("Projected ROI", proj_roi)
                risk_units = recs["stake_pct"].sum()*100
                kpi4.metric("Units at Risk", f"{risk_units:.2f} U")
                if recs["confidence"].max() < 1:
                    st.markdown("_Note: These are low-confidence edges based on limited signals._")
                styled_recs = recs[[
                    "matchup", "market", "side", "odds", "%bets", "%handle",
                    "implied_prob", "win_prob_gap", "value_diff",
                    "refined_signal", "rlm_signal", "stake_pct", "confidence",
                    "trap_score", "clown_score", "square_signal",
                    "is_primetime", "avoid_tag", "clown_tag"
                ]].style.apply(
                    lambda s: ['background-color:#290000' if float(v) >= 70 else '' for v in s], subset=['%bets']
                )
                st.dataframe(styled_recs, use_container_width=True)

                # ---- Log actionable recs to bet_logs.db for proof & CLV ----
                try:
                    date_iso = datetime.now(pacific).date().isoformat()
                    logged = 0
                    for r in recs.itertuples(index=False):
                        matchup = str(getattr(r, "matchup", ""))
                        market_r = str(getattr(r, "market", ""))
                        side_r   = str(getattr(r, "side", ""))
                        odds_r   = getattr(r, "odds", None)
                        if not matchup or not side_r or not market_r:
                            continue
                        gid = _mk_id(date_iso, matchup, market_r, side_r)
                        # Determine bet card fields based on market
                        total_bet_rec = None
                        ml_bet_rec = None
                        total_val = None
                        odds_val = None
                        if market_r.lower().startswith("total"):
                            ou, tot = _parse_total_from_side(side_r)
                            total_bet_rec = f"{ou} {tot}" if ou and tot is not None else side_r
                            total_val = tot
                            odds_val = int(clean_odds(odds_r)) if odds_r not in (None,"") else None
                        elif market_r.lower().startswith("moneyline"):
                            ml_bet_rec = side_r
                            odds_val = int(clean_odds(odds_r)) if odds_r not in (None,"") else None
                        else:
                            # For Spread or other markets, treat like ML with odds only
                            ml_bet_rec = f"{side_r}"
                            odds_val = int(clean_odds(odds_r)) if odds_r not in (None,"") else None
                        # Write bet card
                        log_bet_card_to_db({
                            "game_id": gid,
                            "home_team": None,
                            "away_team": None,
                            "game_time": None,
                            "combined_runs": None,
                            "delta": float(getattr(r, "value_diff", 0.0)),
                            "total_bet_rec": total_bet_rec,
                            "ml_bet_rec": ml_bet_rec,
                            "bookmaker_line": float(total_val) if total_val is not None else None,
                            "home_ml_book": odds_val if (ml_bet_rec and odds_val is not None) else None,
                            "log_date": datetime.utcnow().isoformat()
                        })
                        # YOUR_REC snapshot for CLV
                        log_snapshot_your_rec(gid, market_r, side_r, total_val, odds_val)
                        # Blog-facing archive row
                        edge_bp = float(getattr(r, "value_diff", 0.0)) * 100.0
                        log_blog_pick_to_db({
                            "log_date": datetime.utcnow().isoformat(),
                            "matchup": matchup,
                            "bet_type": market_r,
                            "confidence": f"{float(getattr(r,'confidence',0.0)):.1f}/5",
                            "edge_pct": edge_bp,
                            "odds": odds_val,
                            "predicted_total": None,
                            "predicted_winner": None,
                            "predicted_margin": None,
                            "bookmaker_total": float(total_val) if total_val is not None else None,
                            "analysis": f"{side_r} {odds_val if odds_val is not None else ''} | edge {edge_bp:+.1f} bp | rlm {getattr(r,'rlm_signal','') or ''}"
                        })
                        logged += 1
                    st.success(f"Logged {logged} recommendations to {BETLOGS_DB}")
                except Exception as e:
                    st.warning(f"Logging recs failed: {e}")

                # --- ⏬ DAILY SUMMARY MARKDOWN GENERATION & DOWNLOAD ⏬ ---

                pac = pytz.timezone("America/Los_Angeles")

                # ---------- helper functions ----------
                def _build_snapshot(df: pd.DataFrame):
                    """Return (top_sucker, sharp_smash, biggest_rlm) rows for the slate snapshot."""
                    tmp = df.copy()
                    tmp["bets"]  = pd.to_numeric(tmp["%bets"],   errors="coerce")
                    tmp["money"] = pd.to_numeric(tmp["%handle"], errors="coerce")
                    tmp["diff"]  = tmp["bets"] - tmp["money"]

                    top_sucker   = tmp.sort_values("diff", ascending=False).iloc[0]
                    tmp["sharp_gap"] = tmp["money"] - tmp["bets"]
                    sharp_smash  = tmp.sort_values("sharp_gap", ascending=False).iloc[0]

                    tmp["odds_move"] = pd.to_numeric(tmp["odds_move"], errors="coerce")
                    rlm_candidates = tmp[
                        ((tmp["bets"] > 50) & (tmp["odds_move"] < 0)) |
                        ((tmp["bets"] < 50) & (tmp["odds_move"] > 0))
                    ]
                    biggest_rlm = None
                    if not rlm_candidates.empty:
                        biggest_rlm = rlm_candidates.reindex(
                            rlm_candidates["odds_move"].abs().sort_values(ascending=False).index
                        ).iloc[0]
                    return top_sucker, sharp_smash, biggest_rlm

                def generate_daily_summary(recs_df: pd.DataFrame, full_df: pd.DataFrame) -> str:
                    """Build the Public Enemy Lines markdown report (entirely from in‑app data)."""
                    top_sucker, sharp_smash, biggest_rlm = _build_snapshot(full_df)

                    lines = []
                    lines.append(f"## 🔥 NCAAF Public Enemy Lines — {datetime.now(pac).strftime('%B %d, %Y')}")
                    lines.append("")
                    lines.append("### 1️⃣ Slate Snapshot")
                    lines.append("")
                    lines.append("| Metric | Game / Side | Bets / Money |")
                    lines.append("| --- | --- | --- |")
                    lines.append(f"| Biggest Public Sucker Bet | {top_sucker['side']} | {int(top_sucker['bets'])}% / {int(top_sucker['money'])}% |")
                    lines.append(f"| Largest Sharp Smash | {sharp_smash['side']} | {int(sharp_smash['bets'])}% / {int(sharp_smash['money'])}% |")
                    if biggest_rlm is not None:
                        move_desc = f"{int(biggest_rlm['odds_move']):+d}c vs public"
                        lines.append(f"| Heaviest RLM | {biggest_rlm['side']} | {move_desc} |")

                    lines.append("")
                    lines.append("### 2️⃣ Game Cards")
                    lines.append("")
                    for _, r in recs_df.iterrows():
                        rlm_flag = "✅" if r.get("rlm_flag") else "—"
                        edge_pct = r.get("value_diff", 0) * 100
                        lines.append(f"#### ⚾ {r['matchup']} — {r['side']} {r['odds']}")
                        lines.append("")
                        lines.append("| Bets / Money | Edge % | Kelly % | RLM | Confidence |")
                        lines.append("| --- | --- | --- | --- | --- |")
                        lines.append(f"| {r['%bets']}% / {r['%handle']}% | {edge_pct:.1f} | {r['stake_pct']*100:.1f} | {rlm_flag} | {r['confidence']:.1f} |")
                        lines.append("")

                    lines.append("### 3️⃣ Recap & Bankroll Exposure")
                    lines.append("")
                    lines.append("| Pick | Odds | Kelly % | Units |")
                    lines.append("| --- | --- | --- | --- |")

                    total_units = 0.0
                    for _, r in recs_df.iterrows():
                        units = round(r["stake_pct"] * 100 / 2, 2)  # half‑Kelly → unit metaphor
                        total_units += units
                        lines.append(f"| {r['side']} | {r['odds']} | {r['stake_pct']*100:.1f}% | {units:.2f} |")

                    lines.append("")
                    lines.append(f"**Total Risk:** {total_units:.2f} U  \n")
                    return "\n".join(lines)

                daily_summary_md = generate_daily_summary(recs, current_df)

                st.markdown("### 📜 Daily Summary Report")
                st.markdown(daily_summary_md, unsafe_allow_html=True)

                st.download_button(
                    "Download Summary Markdown",
                    data=daily_summary_md,
                    file_name=f"pel_ncaaf_summary_{datetime.now(pac).strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
                # --- ⏫ END SUMMARY SECTION ⏫ ---
            else:
                st.markdown("_No betting recommendations meet even fallback criteria today._")
                # Still render a Daily Summary even when no actionable recs
                daily_summary_md = generate_daily_summary(recs.head(0), current_df)
                st.markdown("### 📜 Daily Summary Report")
                st.markdown(daily_summary_md, unsafe_allow_html=True)
                st.download_button(
                    "Download Summary Markdown",
                    data=daily_summary_md,
                    file_name=f"pel_ncaaf_summary_{datetime.now(pac).strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )

            if not recs.empty:
                st.download_button(
                    "📥 Download Trap Digest",
                    data=recs.to_csv(index=False),
                    file_name=f"trap_digest_{datetime.now(pacific).strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            # --- Audit Log (always run this block) ---
            log_file = "audit_log.csv"
            audit_cols = [
                "run_time", "matchup", "market", "side", "odds",
                "bet_recommendation", "stake_pct", "confidence"
            ]

            audit = recs.copy()
            audit["run_time"] = datetime.now(pacific)

            # Keep only audit_cols to prevent field mismatch
            audit = audit[audit_cols]

            audit.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)

            # ——— If previous uploaded, compute movement ———
            if prev_file is not None:
                prev_df = pd.read_csv(prev_file, parse_dates=["update_time"])
                # ensure numeric
                for col in ["%handle", "%bets"]:
                    current_df[col] = pd.to_numeric(current_df[col], errors="coerce")
                    prev_df[col]    = pd.to_numeric(prev_df[col], errors="coerce")
                merged = pd.merge(
                    current_df, prev_df,
                    on=["matchup", "market", "side"],
                    suffixes=("_cur", "_prev")
                )
                merged["handle_move"] = merged["%handle_cur"] - merged["%handle_prev"]
                merged["bets_move"]   = merged["%bets_cur"]   - merged["%bets_prev"]

                # apply thresholds
                filt = merged[
                    (merged["handle_move"].abs() >= handle_thresh) &
                    (merged["bets_move"].abs()   >= bets_thresh )
                ].sort_values("handle_move", ascending=False)

                st.subheader("Line Movement")
                st.dataframe(
                    filt[[
                        "matchup",
                        "game_time_cur",
                        "market",
                        "side",
                        "odds_cur",
                        "%handle_cur",
                        "%bets_cur",
                        "handle_move",
                        "bets_move",
                    ]]
                )
            else:
                st.info("Upload a previous splits CSV to see line movement.")

            st.subheader("🛑 Bet Interceptor")
            team_input = st.selectbox("Who are you about to bet on?", sorted(current_df["side"].unique()))
            selected = current_df[current_df["side"] == team_input]
            if selected.empty:
                st.error("No data found for that selection.")
            else:
                row = selected.iloc[0]

                if row["verdict"] == "🚨 Public Trap":
                    st.error(f"🚨 {team_input} is a trap. {row['%bets']}% of bets and the line hasn't moved. Books want your money.")
                elif row["clown_tag"]:
                    st.warning(f"{row['clown_tag']}")
                else:
                    st.success("✅ No trap indicators — proceed with caution.")

            # ===== Display audit_log.csv at the end of the script =====
            if os.path.exists("audit_log.csv"):
                st.subheader("📊 Audit Log (Recent Entries)")
                audit_df = pd.read_csv("audit_log.csv")
                st.dataframe(audit_df.tail(20))

            if os.path.exists("audit_log.csv"):
                st.subheader("📉 Trap Tracker Leaderboard")
                try:
                    audit_df = pd.read_csv("audit_log.csv", on_bad_lines="skip")
                    if "trap_score" in audit_df.columns:
                        audit_df["trap_score"] = pd.to_numeric(audit_df["trap_score"], errors="coerce")
                        top_traps = audit_df.sort_values("trap_score", ascending=False).dropna(subset=["trap_score"]).head(10)
                        st.table(top_traps[["run_time", "matchup", "side", "odds", "trap_score", "confidence"]])
                    else:
                        st.info("No trap score history available yet.")
                except Exception as e:
                    st.warning(f"Trap Tracker failed to load: {e}")


            # ===== Sucker Leaderboard =====
            st.subheader("🤕 Sucker Leaderboard")
            suckers = current_df.copy()
            suckers["bets"] = pd.to_numeric(suckers["%bets"], errors="coerce")
            suckers["handle"] = pd.to_numeric(suckers["%handle"], errors="coerce")
            suckers["sucker_index"] = suckers["bets"] - suckers["handle"]
            top_suckers = suckers.sort_values("sucker_index", ascending=False).head(5)
            st.table(top_suckers[["matchup", "side", "odds", "%bets", "%handle", "sucker_index"]])

            # ===== Sharp Signal Frequency Meter =====
            st.subheader("🧠 Sharp Signal Frequency Meter")
            total = len(current_df)
            sharp_count = (current_df["verdict"] == "💰 Sharp Side").sum()
            pct = (sharp_count / total * 100) if total else 0
            st.metric("Sharp Side Count", sharp_count, f"{pct:.1f}% of board")

            # ===== Trap Whisper Quotes =====
            trap_quotes = [
                "The sharpest bet is the one you *don’t* make.",
                "When the public agrees, the book smiles.",
                "Avoiding traps isn’t sexy — it’s smart.",
                "The crowd loves comfort. The book loves the crowd.",
                "Discipline beats excitement every time.",
                "Traps don’t scream. They whisper. You either hear it — or lose."
            ]
            st.markdown("### 🗣️ Trap Whisper")
            st.markdown(f"_{random.choice(trap_quotes)}_")


            # ===== Social Content Generation Section =====
            st.markdown("### 📣 Generate Social Content from Today's Traps")
            if not recs.empty:
                posts = generate_social_posts(recs)
                if st.button("Generate Tweet"):
                    st.code(posts["tweet"], language="text")
                if st.button("Generate Instagram Caption"):
                    st.code(posts["ig"], language="text")
                if st.button("Generate TikTok Script"):
                    st.code(posts["tiktok"], language="text")
            else:
                st.warning("No recommendations available to generate social content.")

            # ===== Longform Content Generation =====

            def generate_longform_content(df):
                if df.empty:
                    return "# No recommendations to generate longform report."
                top = df.sort_values("trap_score", ascending=False).iloc[0]

                trap_intros = [
                    "Books are baiting the public again — and they’re biting.",
                    "Another day, another overpriced favorite. Let’s take them down.",
                    "You’d think bettors would learn… but no, they’re back on the square special.",
                    "The public’s piling in. The books? Smirking.",
                    "It’s a trap. Again. And the herd’s all in.",
                    "Spotting traps is half the game. Avoiding them? That’s the other half.",
                    "This isn’t sharp. It’s square bait. Let’s torch it.",
                    "Odds haven’t moved. Bets are flying in. You already know what that means."
                ]

                interceptor_alerts = [
                    "Thinking of betting this one? Slam the brakes.",
                    "One look at the numbers and you should already be fading.",
                    "This isn’t value — it’s bait.",
                    "Your gut might say yes. The data screams RUN.",
                    "Feels easy. That’s the danger sign.",
                    "There’s no sharp money here. Just you, the trap, and the book laughing.",
                    "They want you to feel confident. That’s the trick.",
                    "Don’t fall for comfort. It’s engineered deception."
                ]

                play_section_openers = [
                    "We don’t bet often. But when we do, it’s calculated.",
                    "Here are the only sides we’d even consider today.",
                    "Only the sharpest bets make this cut.",
                    "These are vetted. Filtered. Ruthlessly.",
                    "Think of these as your sniper picks. Nothing else qualifies.",
                    "No emotion. No bias. Just edge.",
                    "You want discipline? Start here.",
                    "If it’s not in this list, it’s not worth your money."
                ]

                value_watch_openers = [
                    "Not placing these yet — but they’re screaming value.",
                    "Watch these lines. They’re about to move or be misread.",
                    "The price is off. Whether you bet now or later, know they’re misaligned.",
                    "These are on deck. One shift and we pounce.",
                    "Bets not fired yet — but crosshairs are locked.",
                ]

                fade_public_intros = [
                    "When the herd moves as one, we move the other way.",
                    "Mass conviction ≠ correctness. In fact, it’s often the opposite.",
                    "Public pressure creates value — if you have the balls to go against it.",
                    "If you’re with the crowd, you’re probably wrong. Here’s who we’re fading.",
                    "Being early and contrarian wins. These are your fade signals."
                ]

                final_thoughts = [
                    "Sharp betting isn’t about chasing action. It’s about avoiding mistakes.",
                    "Most people lose by following noise. You win by filtering it.",
                    "The smartest move is often the one you *don’t* make.",
                    "Books love the public for a reason. You’re not them.",
                    "We don’t bet for fun. We bet to win. Or we don’t bet at all."
                ]

                trap_explanation = top.get("trap_explanation", "Public is loading one side. Books haven’t flinched. That’s bait, not value.")

                markdown = f"""# 🔥 Public Enemy Lines: Today's Trap Report – [{datetime.now().strftime('%B %d, %Y')}]

    Most bettors today will walk straight into these traps.  
    We’re here to expose them before the books do.

    ---

    ## 🚨 Today’s Top Public Traps

    {random.choice(trap_intros)}

    ### ❌ {top['matchup']} — Side: {top['side']}
    - 🧠 Trap Score: {top['trap_score']}/10
    - 📊 Bets: {top['%bets']}% | Handle: {top['%handle']}%
    - 📉 Line: {top['odds']} → Odds Move: {top['odds_move']}
    - 📝 Why it's a trap: {trap_explanation}

    ---

    ## 🛑 Interceptor Warning

    {random.choice(interceptor_alerts)}

    - 📊 Public: {top['%bets']}%
    - 📉 Line hasn’t moved
    - 🚨 Verdict: {top['verdict'] or 'N/A'}

    Stay sharp.

    ---

    ## ✅ Sharps-Only: Approved Bets

    {random.choice(play_section_openers)}

    ### 🎯 {top['matchup']} — Side: {top['side']}
    - 💡 Signal: {top['bet_recommendation']}
    - 🧠 Confidence Score: {top['confidence']:.2f}
    - 📈 Stake: {top['stake_pct']*100:.1f}%
    - 📉 Line Move: {top['odds_move']}

    ---

    ## 💰 Mispriced Market Watch

    {random.choice(value_watch_openers)}

    ### 💸 {top['matchup']} — {top['side']}
    - 📊 Public: {top['%bets']}%
    - 🧮 Implied Win Prob: {top['implied_prob']*100:.1f}%
    - ⚖️ Value Edge: {top['value_diff']*100:.1f}%

    ---

    ## 🔁 Crowd Control: Public Fade Signals

    {random.choice(fade_public_intros)}

    ### 🚫 {top['matchup']} — {top['side']}
    - 📊 Public Ratio: {top.get('bets_to_implied_ratio', 1.2):.2f}
    - 🤖 Fade Triggered

    ---

    ## 🧠 Final Thought

    {random.choice(final_thoughts)}

    **This is Public Enemy Lines.**  
    You’re not here to bet with the herd.  
    You’re here to beat the book.

    ---

    ## 📬 Want This Daily?

    - 🧠 Smarter fades  
    - 🚫 No square picks  
    - 📈 Verified model edges  
    - 🔗 Shareable trap cards

    📥 [Subscribe Now]  
    📤 [Download Today’s Report]
    """
                return markdown

            longform = generate_longform_content(recs)
            if st.button("💥 Fire Full Content Drop"):
                st.code(longform, language="markdown")
                date_str = datetime.now().strftime('%Y%m%d')
                md_path = f"exports/trap_longform_{date_str}.md"
                os.makedirs("exports", exist_ok=True)
                with open(md_path, "w") as f:
                    f.write(longform)
                st.success(f"Longform trap report saved as Markdown: {md_path}")
                # Try PDF export
                try:
                    def md_to_pdf(md_content, output_file):
                        html = md2_markdown(md_content)
                        with open(output_file, "w+b") as result_file:
                            pisa.CreatePDF(html, dest=result_file)
                    pdf_path = f"exports/trap_longform_{date_str}.pdf"
                    md_to_pdf(longform, pdf_path)
                    st.success(f"PDF version exported to {pdf_path}")
                except ImportError as e:
                    st.warning("PDF export requires `markdown2` and `xhtml2pdf`. Please install these packages to enable PDF downloads.")
                except Exception as e:
                    st.warning(f"PDF export failed: {e}")
            else:
                st.warning("No recommendations available to generate social content.")
    # ===== Trap Card Image Generation Section =====

    # ===== Trap Card Image Generation Section =====

    with st.expander("🖼 Generate Trap Graphic"):
        if not recs.empty:
            top = recs.sort_values("trap_score", ascending=False).iloc[0]
            if st.button("📸 Generate Trap Card Image"):
                generate_trap_card_image(top, "trap_card.png")
                st.image("trap_card.png", caption="Trap Card", use_container_width=True)
                with open("trap_card.png", "rb") as f:
                    st.download_button("Download Trap Card", f, file_name="trap_card.png")
        else:
            st.warning("No trap data available for image generation.")

    # ===== Public Enemy of the Day Section =====
    st.subheader("🤡 Public Enemy of the Day")
    enemy = current_df[current_df["clown_tag"] != ""].sort_values("clown_score", ascending=False).head(1)
    if not enemy.empty:
        row = enemy.iloc[0]
        st.markdown(f"""
    ### 🤡 {row['matchup']} — {row['side']}
    - **Bets:** {row['%bets']}%  
    - **Handle:** {row['%handle']}%  
    - **Odds:** {row['odds']}  
    - **Line Move:** {row['odds_move']}  
    - **Clown Score:** {row['clown_score']}  
    - **Trap Score:** {row['trap_score']}  
    - **Confidence:** {row['confidence']:.2f}/5  
    - **Verdict:** {row['verdict'] or 'No Signal'}

    📉 {row['trap_explanation'] or 'Classic bait. Odds steady despite heavy betting.'}
    """)
    else:
        st.markdown("_No clear Public Enemy today._")


    # ===== Graphic Generation for Core Signals =====

    # --- Image Generation Functions for Graphic Cards ---
    def generate_public_enemy_card(row, output_path="public_enemy.png"):
        W, H = 1080, 1080
        background = Image.new("RGBA", (W, H), color=(20, 20, 20, 255))
        draw = ImageDraw.Draw(background)
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        title_font = ImageFont.truetype(font_path, 70)
        sub_font = ImageFont.truetype(font_path, 50)
        small_font = ImageFont.truetype(font_path, 35)
        # Title
        draw.text((60, 60), "🤡 Public Enemy of the Day", font=title_font, fill=(255, 80, 80))
        # Matchup and Side
        draw.text((60, 180), f"{row['matchup']}", font=title_font, fill=(255,255,255))
        draw.text((60, 270), f"Side: {row['side']} ({row['odds']})", font=sub_font, fill=(255,255,255))
        # Stats block
        y = 370
        icon_dir = "/Users/matthewfox/Desktop/FoxEdgeAI/pel_emojis"
        def paste_icon(name, x, y):
            try:
                icon = Image.open(os.path.join(icon_dir, name)).convert("RGBA").resize((40,40))
                background.paste(icon, (x, y), icon)
            except Exception:
                pass
        paste_icon("clown.png", 60, y)
        draw.text((120, y), f"Clown Score: {row.get('clown_score', 0)}", font=sub_font, fill=(255,255,255))
        y += 70
        paste_icon("stats.png", 60, y)
        draw.text((120, y), f"{row.get('%bets','')}% Bets vs {row.get('%handle','')}% Handle", font=sub_font, fill=(255,255,255))
        y += 70
        paste_icon("line_down.png", 60, y)
        draw.text((120, y), f"Line Move: {row.get('odds_move','')}", font=sub_font, fill=(255,255,255))
        y += 70
        paste_icon("magnifier.png", 60, y)
        draw.text((120, y), f"Trap Score: {row.get('trap_score', 0)}", font=sub_font, fill=(255,255,255))
        y += 70
        paste_icon("brain.png", 60, y)
        draw.text((120, y), f"Confidence: {row.get('confidence', 0):.2f}/5", font=sub_font, fill=(255,255,255))
        # Explanation
        explanation = row.get("trap_explanation", "")
        wrapper = textwrap.TextWrapper(width=32)
        lines = wrapper.wrap(text=explanation)
        y_text = 800
        for line in lines:
            draw.text((60, y_text), line, font=small_font, fill=(180,180,180))
            y_text += 45
        # Branding
        try:
            icon = Image.open(os.path.join(icon_dir, "box.png")).convert("RGBA").resize((40,40))
            background.paste(icon, (W//2-100, 1000), icon)
        except Exception:
            pass
        draw.text((W//2-30, 1000), "Public Enemy Lines", font=small_font, fill=(180,180,180))
        background = background.convert("RGBA")
        background.save(output_path)

    def generate_ssi_card(ssi_value, emoji, output_path="ssi_card.png"):
        W, H = 1080, 1080
        background = Image.new("RGBA", (W, H), color=(20, 20, 20, 255))
        draw = ImageDraw.Draw(background)
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        title_font = ImageFont.truetype(font_path, 70)
        sub_font = ImageFont.truetype(font_path, 50)
        small_font = ImageFont.truetype(font_path, 35)
        # Title
        draw.text((60, 60), "📊 Sucker Sentiment Index", font=title_font, fill=(255, 200, 80))
        # Value + Emoji
        draw.text((60, 200), f"{ssi_value:.1f}% {emoji}", font=title_font, fill=(255,255,255))
        draw.text((60, 300), "Average % of public bets on trap games today", font=sub_font, fill=(180,180,180))
        # Emoji icon
        icon_dir = "/Users/matthewfox/Desktop/FoxEdgeAI/pel_emojis"
        emoji_icon = {"🧼":"soap.png", "🥴":"woozy.png", "🤡":"clown.png"}
        icon_file = emoji_icon.get(emoji, "clown.png")
        try:
            icon = Image.open(os.path.join(icon_dir, icon_file)).convert("RGBA").resize((40,40))
            background.paste(icon, (60, 400), icon)
        except Exception:
            pass
        # Explanation
        draw.text((60, 500), "Higher = public more delusional", font=small_font, fill=(180,180,180))
        # Branding
        try:
            icon = Image.open(os.path.join(icon_dir, "box.png")).convert("RGBA").resize((40,40))
            background.paste(icon, (W//2-100, 1000), icon)
        except Exception:
            pass
        draw.text((W//2-30, 1000), "Public Enemy Lines", font=small_font, fill=(180,180,180))
        background = background.convert("RGBA")
        background.save(output_path)

    def generate_sucker_leaderboard_card(df, output_path="sucker_leaderboard.png"):
        W, H = 1080, 1080
        background = Image.new("RGBA", (W, H), color=(20, 20, 20, 255))
        draw = ImageDraw.Draw(background)
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        title_font = ImageFont.truetype(font_path, 70)
        sub_font = ImageFont.truetype(font_path, 50)
        small_font = ImageFont.truetype(font_path, 35)
        # Title
        draw.text((60, 60), "🤕 Sucker Leaderboard", font=title_font, fill=(255, 120, 120))
        # Table headers
        draw.text((60, 170), "Matchup", font=sub_font, fill=(255,255,255))
        draw.text((420, 170), "Side", font=sub_font, fill=(255,255,255))
        draw.text((600, 170), "%Bets", font=sub_font, fill=(255,255,255))
        draw.text((760, 170), "%Handle", font=sub_font, fill=(255,255,255))
        draw.text((940, 170), "SuckerIdx", font=sub_font, fill=(255,255,255))
        # Rows
        icon_dir = "/Users/matthewfox/Desktop/FoxEdgeAI/pel_emojis"
        y = 230
        for idx, row in df.iterrows():
            try:
                icon = Image.open(os.path.join(icon_dir, "clown.png")).convert("RGBA").resize((40,40))
                background.paste(icon, (20, y+10), icon)
            except Exception:
                pass
            draw.text((60, y), str(row['matchup'])[:18], font=small_font, fill=(255,255,255))
            draw.text((420, y), str(row['side']), font=small_font, fill=(255,255,255))
            draw.text((600, y), f"{row['%bets']}", font=small_font, fill=(255,255,255))
            draw.text((760, y), f"{row['%handle']}", font=small_font, fill=(255,255,255))
            draw.text((940, y), f"{row['sucker_index']:.1f}", font=small_font, fill=(255,255,255))
            y += 65
            if y > 900:
                break
        # Branding
        try:
            icon = Image.open(os.path.join(icon_dir, "box.png")).convert("RGBA").resize((40,40))
            background.paste(icon, (W//2-100, 1000), icon)
        except Exception:
            pass
        draw.text((W//2-30, 1000), "Public Enemy Lines", font=small_font, fill=(180,180,180))
        background = background.convert("RGBA")
        background.save(output_path)

    def generate_sharp_meter_card(count, total, output_path="sharp_meter.png"):
        W, H = 1080, 1080
        background = Image.new("RGBA", (W, H), color=(20, 20, 20, 255))
        draw = ImageDraw.Draw(background)
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        title_font = ImageFont.truetype(font_path, 70)
        sub_font = ImageFont.truetype(font_path, 50)
        small_font = ImageFont.truetype(font_path, 35)
        # Title
        draw.text((60, 60), "🧠 Sharp Signal Frequency", font=title_font, fill=(120, 220, 255))
        pct = 0 if total == 0 else (count / total) * 100
        draw.text((60, 200), f"{count} / {total} ({pct:.1f}%)", font=title_font, fill=(255,255,255))
        draw.text((60, 300), "Sharp Sides Detected Today", font=sub_font, fill=(180,180,180))
        # Meter bar
        bar_x, bar_y = 60, 400
        bar_w = int(900 * pct / 100)
        draw.rectangle([bar_x, bar_y, bar_x + 900, bar_y + 60], outline=(120,220,255), width=5)
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 60], fill=(120,220,255,160))
        # Branding
        icon_dir = "/Users/matthewfox/Desktop/FoxEdgeAI/pel_emojis"
        try:
            icon = Image.open(os.path.join(icon_dir, "brain.png")).convert("RGBA").resize((40,40))
            background.paste(icon, (bar_x + bar_w - 20, bar_y + 10), icon)
        except Exception:
            pass
        try:
            icon = Image.open(os.path.join(icon_dir, "box.png")).convert("RGBA").resize((40,40))
            background.paste(icon, (W//2-100, 1000), icon)
        except Exception:
            pass
        draw.text((W//2-30, 1000), "Public Enemy Lines", font=small_font, fill=(180,180,180))
        background = background.convert("RGBA")
        background.save(output_path)

    def generate_trap_whisper_card(quote, output_path="trap_whisper.png"):
        W, H = 1080, 1080
        background = Image.new("RGBA", (W, H), color=(20, 20, 20, 255))
        draw = ImageDraw.Draw(background)
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        title_font = ImageFont.truetype(font_path, 70)
        sub_font = ImageFont.truetype(font_path, 50)
        small_font = ImageFont.truetype(font_path, 35)
        # Title
        draw.text((60, 60), "🗣️ Trap Whisper", font=title_font, fill=(180, 180, 255))
        # Quote, wrapped
        wrapper = textwrap.TextWrapper(width=22)
        lines = wrapper.wrap(text=quote)
        y = 220
        for line in lines:
            draw.text((60, y), line, font=title_font, fill=(255,255,255))
            y += 90
        # Emoji icon
        icon_dir = "/Users/matthewfox/Desktop/FoxEdgeAI/pel_emojis"
        try:
            icon = Image.open(os.path.join(icon_dir, "whisper.png")).convert("RGBA").resize((40,40))
            background.paste(icon, (60, 900), icon)
        except Exception:
            pass
        # Branding
        try:
            icon = Image.open(os.path.join(icon_dir, "box.png")).convert("RGBA").resize((40,40))
            background.paste(icon, (W//2-100, 1000), icon)
        except Exception:
            pass
        draw.text((W//2-30, 1000), "Public Enemy Lines", font=small_font, fill=(180,180,180))
        background = background.convert("RGBA")
        background.save(output_path)

    # Wrap Graphics Section in Graphics Tab
    with graphics_tab:
        st.markdown("---")
        # --- Core Signal Graphics Grid ---
        st.header("🖼️ Core Signal Graphics")

        # Ensure graphics are generated before layout
        public_enemy_path, ssi_path, sharp_path, sucker_path, whisper_path = None, None, None, None, None

        # 1. Public Enemy Card
        enemy = current_df[current_df["clown_tag"] != ""].sort_values("clown_score", ascending=False).head(1)
        if not enemy.empty:
            row_enemy = enemy.iloc[0]
            public_enemy_path = "public_enemy.png"
            generate_public_enemy_card(row_enemy, public_enemy_path)

        # 2. Sucker Sentiment Index Card
        trap_games = current_df[current_df["verdict"] == "🚨 Public Trap"]
        if not trap_games.empty:
            ssi_val = trap_games["%bets"].astype(float).mean()
            ssi_emoji = "🧼" if ssi_val < 70 else "🥴" if ssi_val < 80 else "🤡"
            ssi_path = "ssi_card.png"
            generate_ssi_card(ssi_val, ssi_emoji, ssi_path)

        # 3. Sharp Signal Frequency Meter
        sharp_path = "sharp_meter.png"
        generate_sharp_meter_card(sharp_count, total, sharp_path)

        # 4. Sucker Leaderboard Card
        if not top_suckers.empty:
            sucker_path = "sucker_leaderboard.png"
            generate_sucker_leaderboard_card(top_suckers, sucker_path)

        # 5. Trap Whisper Quote
        whisper_path = "trap_whisper.png"
        generate_trap_whisper_card(random.choice(trap_quotes), whisper_path)

        # --- Display graphics in responsive grids ---
        # First row: Enemy, SSI, Sharp Meter
        row1_cols = st.columns(3)
        graphics_row1 = [public_enemy_path, ssi_path, sharp_path]
        captions_row1 = ["Public Enemy of the Day", "Sucker Sentiment Index", "Sharp Signal Frequency"]

        for col, img, cap in zip(row1_cols, graphics_row1, captions_row1):
            if img and os.path.exists(img):
                with col:
                    st.image(img, caption=cap, use_container_width=True)
                    with open(img, "rb") as f:
                        st.download_button(f"Download {cap}", f, file_name=os.path.basename(img))

        # Second row: Sucker Leaderboard & Trap Whisper
        row2_cols = st.columns(2)
        graphics_row2 = [(sucker_path, "Sucker Leaderboard"), (whisper_path, "Trap Whisper")]

        for col, (img, cap) in zip(row2_cols, graphics_row2):
            if img and os.path.exists(img):
                with col:
                    st.image(img, caption=cap, use_container_width=True)
                    with open(img, "rb") as f:
                        st.download_button(f"Download {cap}", f, file_name=os.path.basename(img))



# ---------------- Data Access Helpers ----------------
import os, sqlite3, pandas as pd, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def _resolve_db_path():
    # Priority: sidebar input -> ENV -> default
    default = os.environ.get("BETLOGS_DB", "winible.db")
    db_path = st.sidebar.text_input("DB path (SQLite)", value=default, help="Path to SQLite DB containing line_snapshots/clv_logs tables.")
    st.sidebar.caption(f"Using DB: {db_path}")
    return db_path

def _read_table(db_path, table):
    try:
        with sqlite3.connect(db_path) as con:
            df = pd.read_sql_query(f"SELECT * FROM {table}", con)
        return df
    except Exception as e:
        st.warning(f"Could not read table `{table}` from {db_path}: {e}")
        return pd.DataFrame()

def _compute_clv_from_snapshots(snaps: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty:
        return pd.DataFrame(columns=["game_id","matchup","market","side","book",
                                     "entry_odds","close_odds","entry_total","close_total",
                                     "entry_spread","close_spread",
                                     "clv_prob_pp","clv_line_move","clv_spread_move","computed_utc"])
    # Ensure required columns exist
    needed = {"game_id","market","side","book","snapshot_type","odds","total","spread","timestamp_utc"}
    missing = needed - set(snaps.columns)
    if missing:
        st.error(f"Missing columns in line_snapshots: {missing}")
        return pd.DataFrame()

    recs = snaps[snaps["snapshot_type"]=="YOUR_REC"].copy()
    close = snaps[snaps["snapshot_type"]=="CLOSE"].copy()

    keys = ["game_id","market","side","book"]
    # If matchup present, carry it
    carry_cols = ["matchup"] if "matchup" in snaps.columns else []
    recs_small = recs[keys + carry_cols + ["odds","total","spread"]].rename(columns={
        "odds":"entry_odds","total":"entry_total","spread":"entry_spread"
    })
    close_small = close[keys + ["odds","total","spread"]].rename(columns={
        "odds":"close_odds","total":"close_total","spread":"close_spread"
    })
    out = pd.merge(recs_small, close_small, on=keys, how="left")
    # Simple CLV metrics
    # Probability point proxy only for totals based on 10 cents ≈ ~2% move: not precise; your scripts also compute better proxies.
    out["clv_line_move"] = (out["entry_total"].astype(float) - out["close_total"].astype(float))
    out["clv_spread_move"] = (out["entry_spread"].astype(float) - out["close_spread"].astype(float))
    out["clv_prob_pp"] = np.where(out["entry_odds"].notna() & out["close_odds"].notna(),
                                  (out["close_odds"] - out["entry_odds"])/100.0, np.nan)
    out["computed_utc"] = datetime.utcnow().isoformat()
    # Order columns
    cols = ["game_id"] + carry_cols + ["market","side","book",
            "entry_odds","close_odds","entry_total","close_total","entry_spread","close_spread",
            "clv_prob_pp","clv_line_move","clv_spread_move","computed_utc"]
    return out[cols]

def _render_clv_charts(clv_df: pd.DataFrame):
    if clv_df.empty:
        st.info("No CLV rows to chart.")
        return
    # Histogram: total line move where applicable
    if clv_df["clv_line_move"].notna().any():
        fig = plt.figure()
        clv_df["clv_line_move"].dropna().hist(bins=30)
        plt.title("Distribution of Total Line Moves (Entry - Close)")
        plt.xlabel("Points vs Close (positive = beat close)")
        plt.ylabel("Count")
        st.pyplot(fig)
    # Histogram: spread moves
    if clv_df["clv_spread_move"].notna().any():
        fig = plt.figure()
        clv_df["clv_spread_move"].dropna().hist(bins=30)
        plt.title("Distribution of Spread Moves (Entry - Close)")
        plt.xlabel("Points vs Close (positive = beat close)")
        plt.ylabel("Count")
        st.pyplot(fig)

def _summary_cards(clv_df: pd.DataFrame):
    if clv_df.empty:
        return
    total_n = len(clv_df)
    beat_total = (clv_df["clv_line_move"] > 0).sum(skipna=True)
    beat_spread = (clv_df["clv_spread_move"] > 0).sum(skipna=True)
    avg_pp = clv_df["clv_prob_pp"].mean(skipna=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recs", f"{total_n}")
    c2.metric("Beat Close (Totals)", f"{beat_total}/{total_n}", f"{(beat_total/total_n*100):.1f}%")
    c3.metric("Beat Close (Spread)", f"{beat_spread}/{total_n}")
    c4.metric("Avg Prob. Shift (pp)", f"{avg_pp:.3f}" if pd.notna(avg_pp) else "—")

# ---------------- CLV-only Dashboard ----------------
def render_clv_only():
    st.header("CLV-Only Dashboard")
    db_path = _resolve_db_path()
    # Prefer clv_logs, compute from snapshots if empty
    clv = _read_table(db_path, "clv_logs")
    if clv.empty:
        snaps = _read_table(db_path, "line_snapshots")
        clv = _compute_clv_from_snapshots(snaps)
    else:
        # ensure expected columns
        expected = {"game_id","market","side","book","entry_odds","close_odds","entry_total","close_total","computed_utc"}
        missing = expected - set(clv.columns)
        if missing:
            st.warning(f"clv_logs present but missing columns {missing}. Attempting recompute from snapshots.")
            snaps = _read_table(db_path, "line_snapshots")
            clv = _compute_clv_from_snapshots(snaps)

    if clv.empty:
        st.info("No CLV data available.")
        return

    _summary_cards(clv)

    # Filters
    cols = st.columns(4)
    market = cols[0].selectbox("Market", ["All"] + sorted([m for m in clv["market"].dropna().unique().tolist()]))
    book = cols[1].selectbox("Book", ["All"] + sorted([b for b in clv["book"].dropna().unique().tolist()]))
    since = cols[2].text_input("Since (YYYY-MM-DD)", value="")
    # Filter apply
    filt = clv.copy()
    if market != "All":
        filt = filt[filt["market"]==market]
    if book != "All":
        filt = filt[filt["book"]==book]
    if since:
        try:
            filt = filt[pd.to_datetime(filt.get("computed_utc", datetime.utcnow())) >= pd.to_datetime(since)]
        except Exception:
            st.warning("Bad date filter ignored.")

    st.subheader("CLV Table")
    st.dataframe(filt.sort_values("computed_utc", ascending=False), use_container_width=True)

    st.subheader("Distributions")
    _render_clv_charts(filt)

# ---------------- Consolidated Proof Dashboard ----------------
def render_proof():
    st.header("Consolidated Proof")
    db_path = _resolve_db_path()

    # Mood recap: pull some aggregates from line_snapshots (OPEN vs CLOSE deltas)
    snaps = _read_table(db_path, "line_snapshots")
    if snaps.empty:
        st.info("No snapshots available.")
        return

    # High-level mood: average move from OPEN to CLOSE per market
    pivot = None
    try:
        open_ = snaps[snaps["snapshot_type"]=="OPEN"].copy()
        close_ = snaps[snaps["snapshot_type"]=="CLOSE"].copy()
        keys = ["game_id","market","side","book"]
        op = open_[keys + ["odds","total","spread"]].rename(columns={"odds":"open_odds","total":"open_total","spread":"open_spread"})
        cl = close_[keys + ["odds","total","spread"]].rename(columns={"odds":"close_odds","total":"close_total","spread":"close_spread"})
        pivot = pd.merge(op, cl, on=keys, how="inner")
        pivot["total_move"] = pivot["open_total"] - pivot["close_total"]
        pivot["spread_move"] = pivot["open_spread"] - pivot["close_spread"]
    except Exception as e:
        st.warning(f"Could not compute OPEN→CLOSE moves: {e}")

    c1, c2, c3 = st.columns(3)
    if pivot is not None and not pivot.empty:
        c1.metric("Avg Total Move", f"{pivot['total_move'].mean():.2f}")
        c2.metric("Avg Spread Move", f"{pivot['spread_move'].mean():.2f}")
        c3.metric("Count Markets", f"{len(pivot)}")
    else:
        c1.metric("Avg Total Move", "—")
        c2.metric("Avg Spread Move", "—")
        c3.metric("Count Markets", "0")

    # Your recs
    recs = snaps[snaps["snapshot_type"]=="YOUR_REC"].copy()
    st.subheader("Your Recommendations (latest)")
    if recs.empty:
        st.info("No YOUR_REC rows found.")
    else:
        st.dataframe(recs.sort_values("timestamp_utc", ascending=False).head(200), use_container_width=True)

    # CLV proof
    st.subheader("CLV Proof")
    clv = _read_table(db_path, "clv_logs")
    if clv.empty:
        clv = _compute_clv_from_snapshots(snaps)
    if clv.empty:
        st.info("No CLV data to show.")
        return
    _summary_cards(clv)
    st.dataframe(clv.sort_values("computed_utc", ascending=False).head(500), use_container_width=True)


# ---------------- Background Logger (Optional) ----------------
import threading, time
from datetime import datetime, timedelta

# Try to import logging helpers from Market Mood block if available in globals after first render
_background_thread = None

def _start_background_logger(interval_minutes: int = 15, close_hour_local: int = 20):
    """Periodically refresh OPEN snapshots and attempt a daily CLOSE pass.
    Requires `update_splits_history()` and optionally `log_close_snapshot(df)`+`get_current_board_df()` to exist.
    """
    global _background_thread
    if _background_thread and _background_thread.is_alive():
        return

    st.session_state.setdefault("_bg_stop", False)

    def _runner():
        while not st.session_state.get("_bg_stop", False):
            try:
                if "update_splits_history" in globals():
                    update_splits_history()
                else:
                    st.toast("update_splits_history() not found; OPEN logging skipped.", icon="⚠️")
            except Exception as e:
                st.toast(f"OPEN logging error: {e}", icon="⚠️")

            now = datetime.now()
            if now.hour >= close_hour_local:
                try:
                    if "log_close_snapshot" in globals() and "get_current_board_df" in globals():
                        df = get_current_board_df()
                        log_close_snapshot(df)
                except Exception as e:
                    st.toast(f"CLOSE logging error: {e}", icon="⚠️")

            time.sleep(max(60, int(interval_minutes*60)))

    _background_thread = threading.Thread(target=_runner, daemon=True)
    _background_thread.start()

def _stop_background_logger():
    st.session_state["_bg_stop"] = True

def render_background_logger_controls():
    st.sidebar.subheader("Background Logger")
    enabled = st.sidebar.toggle("Enable scheduled logging (OPEN/CLOSE)", value=False,
                                help="When on, the app will periodically log OPEN snapshots and attempt a nightly CLOSE pass.")
    interval = st.sidebar.number_input("Interval minutes", min_value=5, max_value=120, value=15, step=5)
    close_hour = st.sidebar.number_input("Close hour (local 0-23)", min_value=0, max_value=23, value=20, step=1)
    if enabled:
        _start_background_logger(interval_minutes=int(interval), close_hour_local=int(close_hour))
    else:
        _stop_background_logger()


def _derive_totals_column(df):
    """
    Ensure a numeric 'total' column exists.
    If missing, try to parse from 'side' like 'Over 8.5' / 'Under 9'.
    Returns df with a float 'total' where available.
    """
    import pandas as pd, re as _re
    if df is None or df.empty:
        return df
    if "total" not in df.columns:
        df = df.copy()
        def _parse_total(x):
            try:
                m = _re.search(r'(?:Over|Under)\s*([0-9]+(?:\.[0-9]+)?)', str(x), flags=_re.IGNORECASE)
                return float(m.group(1)) if m else None
            except Exception:
                return None
        if "side" in df.columns:
            df["total"] = df["side"].map(_parse_total)
    # Normalize to float
    try:
        df["total"] = df["total"].astype(float)
    except Exception:
        pass
    return df


def _to_float_with_fractions(text):
    """Convert strings like '8½' '9¼' '7¾' to float; fall back to float() if possible."""
    frac_map = {'½': 0.5, '¼': 0.25, '¾': 0.75}
    s = str(text)
    # Replace unicode minus
    s = s.replace('−','-')
    # If any known vulgar fraction exists, split base and add
    for sym, val in frac_map.items():
        if sym in s:
            # find preceding number
            m = re.search(r'(-?\d+)', s)
            if m:
                base = float(m.group(1))
                return base + val
    # Extract first float-like token
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

def _derive_totals_column(df):
    """
    Ensure a numeric 'total' column exists.
    Sources tried in order:
      1) existing df['total'] coerced to float (handles '8½' etc.)
      2) df['side'] patterns like 'Over 8.5', 'U 9', 'o8½', 'Under 47 (-110)'
      3) df['market'] patterns that include 'total' or 'O/U' figures
      4) df['desc'] or df['label'] if present
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    # Coerce existing 'total' if present
    if 'total' in df.columns:
        try:
            df['total'] = df['total'].apply(_to_float_with_fractions)
        except Exception:
            pass

    def extract_from_side(val):
        s = str(val)
        # Normalize
        s = s.replace('O/U', 'Over/Under').replace('o/u','Over/Under')
        # Common compact forms: 'O 8.5', 'U9', 'o8½', 'u 9.0'
        patterns = [
            r'(?:Over|Under)\s*([0-9]+(?:\.[0-9]+)?)',
            r'\b[OUou]\s*([0-9]+(?:\.[0-9]+)?)\b',
            r'\b[OUou]([0-9]+(?:\.[0-9]+)?)\b',
            r'(?:Over|Under)\s*([0-9]+[½¼¾])',
            r'\b[OUou]\s*([0-9]+[½¼¾])\b',
        ]
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                return _to_float_with_fractions(m.group(1))
        return None

    def extract_from_market(val):
        s = str(val)
        # e.g., 'total', 'Totals', 'O/U 47.5', 'Total 8½'
        patterns = [
            r'(?:Total|TOTALS?|O/U)\s*([0-9]+(?:\.[0-9]+)?)',
            r'(?:Total|TOTALS?|O/U)\s*([0-9]+[½¼¾])',
        ]
        for pat in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                return _to_float_with_fractions(m.group(1))
        return None

    # If total still missing or NaN, try to derive from side/market/desc/label
    need = ('total' not in df.columns) or df['total'].isna().all()
    if need:
        candidate = None
        if 'side' in df.columns:
            candidate = df['side'].apply(extract_from_side)
        if (candidate is None or candidate.isna().all()) and 'market' in df.columns:
            candidate = df['market'].apply(extract_from_market)
        if (candidate is None or candidate.isna().all()):
            for alt in ['desc','label','bet_type','title']:
                if alt in df.columns:
                    candidate = df[alt].apply(extract_from_side)
                    if not candidate.isna().all():
                        break
        if candidate is not None:
            df['total'] = candidate

    return df

# ---------------- Main UI ----------------
st.title("NFL Odds Analysis")

tabs = st.tabs(["Market Mood", "Do/Don't Bet Engine", "CLV Only", "Proof"])

with tabs[0]:
    try:
        render_market_mood()
    except Exception as e:
        st.error(f"Market Mood tab error: {e}")

with tabs[1]:
    try:
        render_do_not_bet()
    except Exception as e:
        st.error(f"Do/Don't Bet tab error: {e}")