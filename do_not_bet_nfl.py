import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz
from PIL import Image, ImageDraw, ImageFont
import textwrap
import math

import os
import sqlite3
import logging
from pathlib import Path
import streamlit as st

# === Daily Summary helpers (global) ===
from datetime import datetime
import pytz
import pandas as pd
import altair as alt
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
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

import re
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
    import datetime, pytz
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
    import datetime
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

# --- Spread parser & small utils ---
_spread_num_re = re.compile(r"([+-]?\d+(?:\.\d+)?)")

def _parse_spread_from_side(side_text: str):
    """
    From strings like 'Eagles -3' or 'Chiefs +2.5', return float spread.
    Returns None if not a spread side.
    """
    s = str(side_text or "")
    m = _spread_num_re.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _sign(x: float) -> int:
    try:
        return (x > 0) - (x < 0)
    except Exception:
        return 0

KEY_NUMBERS = {3.0, 7.0}

def _crossed_key_numbers(v1: float | None, v2: float | None) -> bool:
    """True if abs(spread) crossed 3 or 7 between v1 and v2."""
    try:
        if v1 is None or v2 is None:
            return False
        a1, a2 = abs(float(v1)), abs(float(v2))
        for k in KEY_NUMBERS:
            if (a1 < k <= a2) or (a2 < k <= a1):
                return True
        return False
    except Exception:
        return False

# Simple primetime detector using kickoff local hour
PRIMETIME_HOURS = {17, 18, 19, 20}
def is_primetime_simple(game_time_str: str) -> bool:
    try:
        gt = datetime.strptime(str(game_time_str), '%I:%M %p')
        return gt.hour in PRIMETIME_HOURS
    except Exception:
        return False

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
            f"#### 🏈 {r['matchup']} — {r['side']} {r['odds']}",
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
      f"- ⛔ **Gate blocks:** {int((current_df['dnb_decision']=='DO_NOT_BET').sum())}",
      f"- ⏳ **Gate waits:** {int((current_df['dnb_decision']=='WAIT').sum())}",
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
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh
import os

st.set_page_config(
    page_title="DK Line Movement",
    page_icon="🟥",
    layout="wide"
)

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
        return is_primetime_simple(row.get("game_time", ""))
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
                logging.warning(f"[do_not_bet_nfl] Missing snapshot CSV for {prev_key}, cannot catch up.")
        except Exception as e:
            logging.error(f"[do_not_bet_nfl] midnight catch-up failed: {e}")
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
    from urllib.parse import urlencode, urlparse, parse_qs
    from playwright.sync_api import sync_playwright
    import re

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
import os
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
    logging.error(f"[do_not_bet_nfl] auto CLOSE/CLV failed: {e}")

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

# --- Derive spread/total values from side text for current/open snapshots ---
cur_keys = ["matchup","market","side"]

open_spreads = open_df[cur_keys].copy()
open_spreads["spread_open"] = open_spreads.apply(
    lambda r: _parse_spread_from_side(r["side"]) if r["market"]=="Spread" else None, axis=1
)
open_spreads["total_open"]  = open_spreads.apply(
    lambda r: _parse_total_from_side(r["side"])[1] if r["market"]=="Total" else None, axis=1
)

cur_spreads = current_df[cur_keys].copy()
cur_spreads["spread_cur"] = cur_spreads.apply(
    lambda r: _parse_spread_from_side(r["side"]) if r["market"]=="Spread" else None, axis=1
)
cur_spreads["total_cur"]  = cur_spreads.apply(
    lambda r: _parse_total_from_side(r["side"])[1] if r["market"]=="Total" else None, axis=1
)

current_df = current_df.merge(open_spreads[[*cur_keys,"spread_open","total_open"]], on=cur_keys, how="left")
current_df = current_df.merge(cur_spreads[[*cur_keys,"spread_cur","total_cur"]], on=cur_keys, how="left")

# Deltas from open snapshot
current_df["spread_delta"] = current_df.apply(
    lambda r: (r["spread_cur"] - r["spread_open"]) if pd.notnull(r.get("spread_cur")) and pd.notnull(r.get("spread_open")) else 0.0,
    axis=1
)
current_df["total_delta"]  = current_df.apply(
    lambda r: (r["total_cur"]  - r["total_open"])  if pd.notnull(r.get("total_cur"))  and pd.notnull(r.get("total_open"))  else 0.0,
    axis=1
)

# Primetime flag from kickoff hour
current_df["is_primetime"] = current_df.apply(lambda r: is_primetime_simple(r.get("game_time","")), axis=1)

# --- Session odds history for head-fake detection ---
if "odds_hist" not in st.session_state:
    st.session_state["odds_hist"] = {}

def _mk_hist_key(r):
    return f"{r['matchup']}|{r['market']}|{r['side']}"

hist = st.session_state["odds_hist"]
current_df["headfake_flag"] = False
for _, r in current_df.iterrows():
    k = _mk_hist_key(r)
    prev_seq = hist.get(k, [])
    mv = int(r.get("odds_move", 0) or 0)
    new_seq = (prev_seq + [mv])[-3:]
    hist[k] = new_seq
    if len(new_seq) >= 2:
        if _sign(new_seq[-1]) != _sign(new_seq[-2]) and abs(new_seq[-1]) >= 5 and abs(new_seq[-2]) >= 5:
            current_df.loc[
                (current_df["matchup"]==r["matchup"]) &
                (current_df["market"]==r["market"]) &
                (current_df["side"]==r["side"]),
                "headfake_flag"
            ] = True

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
from scipy.stats import zscore

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

# === Microstructure & Do-Not-Bet gating ===
current_df["divergence_pp"] = pd.to_numeric(current_df["%handle"], errors="coerce") - pd.to_numeric(current_df["%bets"], errors="coerce")
current_df["stubborn01"] = ((current_df["odds_move"].abs() <= 1) | (current_df["spread_delta"].abs() <= 0.5)).astype(int)

# Strict RLM: moved against ticket side by >=5c AND handle-vs-bets gap >=20 pp
current_df["rlm_strict"] = current_df.apply(
    lambda r: ((float(r.get("%bets",0)) > 50 and int(r.get("odds_move",0)) <= -5) or
               (float(r.get("%bets",0)) < 50 and int(r.get("odds_move",0)) >= 5)) and
              (abs(float(r.get("%handle",0)) - float(r.get("%bets",0))) >= 20),
    axis=1
)

current_df["kn_cross"] = current_df.apply(
    lambda r: (r["market"]=="Spread") and _crossed_key_numbers(r.get("spread_open"), r.get("spread_cur")), axis=1
)
current_df["missed_value_spread"] = current_df.apply(
    lambda r: (abs(abs(float(r.get("spread_cur",0) or 0)) - abs(float(r.get("spread_open",0) or 0))) + (0.8 if _crossed_key_numbers(r.get("spread_open"), r.get("spread_cur")) else 0.0)) if r["market"]=="Spread" else 0.0,
    axis=1
)

def _dnb_score_and_decision(row):
    score = 0.0
    reasons = []
    # Hard block: missed key number move on spread
    if row["market"] == "Spread" and bool(row["kn_cross"]) and float(row["missed_value_spread"]) >= 1.0:
        reasons.append("Missed key number move")
        return 1.0, "DO_NOT_BET", reasons

    # Soft scoring
    score += 0.20 * float(row["stubborn01"]) * min(1.0, abs(float(row["divergence_pp"])) / 12.0)
    if bool(row.get("rlm_strict", False)):
        score += 0.25; reasons.append("Reverse line vs tickets")
    if bool(row.get("headfake_flag", False)):
        score += 0.15; reasons.append("Head-fake pattern")
    if bool(row.get("kn_cross", False)):
        score += 0.15; reasons.append("Crossed 3/7 from open")
    if bool(row.get("is_primetime", False)) and float(row.get("confidence",0)) < 3.5:
        score += 0.10; reasons.append("Primetime efficiency")

    if score >= 0.6:
        return score, "DO_NOT_BET", reasons
    if score >= 0.4:
        return score, "WAIT", reasons
    return score, "OK", reasons

_dnb = current_df.apply(_dnb_score_and_decision, axis=1, result_type="expand")
current_df["do_not_bet_score"] = _dnb[0]
current_df["dnb_decision"]    = _dnb[1]
current_df["dnb_reasons"]     = _dnb[2].apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")

# ===== Trap/Sharp/Fade/Back summary metrics (moved here after current_df is defined) =====
num_traps = (current_df["verdict"] == "🚨 Public Trap").sum()
num_sharp = (current_df["verdict"] == "💰 Sharp Side").sum()
# Additional signal counts
num_donotbet    = int((current_df["verdict"] == "⛔ Do Not Bet").sum())
num_big_movers  = int(current_df["big_mover_flag"].sum()) if "big_mover_flag" in current_df.columns else 0
num_divergences = int(current_df["divergence_flag"].sum()) if "divergence_flag" in current_df.columns else 0
num_rlm         = int(current_df["rlm_strict"].sum()) if "rlm_strict" in current_df.columns else 0
num_kelly       = int(current_df["kelly_trigger"].sum()) if "kelly_trigger" in current_df.columns else 0
num_dnb_blocks = int((current_df["dnb_decision"] == "DO_NOT_BET").sum()) if "dnb_decision" in current_df.columns else 0
num_waits      = int((current_df["dnb_decision"] == "WAIT").sum()) if "dnb_decision" in current_df.columns else 0
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
    from datetime import datetime, timedelta
    import pytz

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
    table += f"| ⛔ Do-Not-Bet (gate) | {num_dnb_blocks} |\n"
    table += f"| ⏳ Wait (gate) | {num_waits} |\n"
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
            from markdown2 import markdown as md2_markdown
            from xhtml2pdf import pisa
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
            from markdown2 import markdown as md2_markdown
            from xhtml2pdf import pisa
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
          "⚡ Kelly Trigger",
          "⛔ Gate: Do Not Bet",
          "⏳ Gate: Wait",
        ]
    )


    # ====== Professional NFL Betting Insights (moved up to ensure insights_df is defined before merge) ======
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

# ——— Advanced NFL Betting Systems & Trends ———
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


from scipy.stats import rankdata

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

import altair as alt
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
current_df["big_mover_flag"] = current_df["odds_move"].abs() >= 5
# Divergence: public vs implied Z-score ≥ 1.5
current_df["divergence_flag"]  = current_df["value_diff_z"] >= 1.5
# Kelly Trigger: suggested stake ≥ 3%
current_df["kelly_trigger"]    = current_df["stake_pct"] >= 0.03
# Merge contrarian_signal from insights into current_df for RLM logic
if "contrarian_signal" in insights_df.columns:
    current_df = current_df.merge(
        insights_df[["matchup", "market", "side", "contrarian_signal"]],
        on=["matchup", "market", "side"],
        how="left"
    )
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
elif signal_view == "⛔ Gate: Do Not Bet":
    records = [r for r in records if r.get("dnb_decision") == "DO_NOT_BET"]
elif signal_view == "⏳ Gate: Wait":
    records = [r for r in records if r.get("dnb_decision") == "WAIT"]
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
<p><strong>Divergence:</strong> {row.get('divergence_pp',0):+0.0f} pp &nbsp; <strong>RLM:</strong> {bool(row.get('rlm_strict', False))}</p>
<p><strong>Spread Δ:</strong> {row.get('spread_delta',0):+0.1f} &nbsp; <strong>Total Δ:</strong> {row.get('total_delta',0):+0.1f} &nbsp; <strong>KN Cross:</strong> {bool(row.get('kn_cross', False))}</p>
<p><strong>Gate:</strong> {row.get('dnb_decision','')} ({row.get('do_not_bet_score',0):.2f})<br/><em>{row.get('dnb_reasons','')}</em></p>
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
        export_cols = list(current_df.columns)
        current_df.to_csv(log_file, index=False, columns=export_cols)

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
    "Kansas City":   {"win_pct":0.782, "MOV":5.4, "run_line_diff":-0.2},
    "Buffalo":       {"win_pct":0.716, "MOV":8.8, "run_line_diff":3.5},
    "Philadelphia":  {"win_pct":0.634, "MOV":3.8, "run_line_diff":0.9},
    "Baltimore":     {"win_pct":0.626, "MOV":6.3, "run_line_diff":2.4},
    "Green Bay":     {"win_pct":0.622, "MOV":4.2, "run_line_diff":1.8},
    "Tampa Bay":     {"win_pct":0.606, "MOV":4.7, "run_line_diff":1.8},
    "Pittsburgh":    {"win_pct":0.575, "MOV":-0.4, "run_line_diff":0.0},
    "LA Rams":       {"win_pct":0.570, "MOV":1.4, "run_line_diff":0.3},
    "San Francisco": {"win_pct":0.570, "MOV":4.1, "run_line_diff":0.3},
    "Minnesota":     {"win_pct":0.570, "MOV":0.1, "run_line_diff":-0.6},
    "Dallas":        {"win_pct":0.562, "MOV":3.1, "run_line_diff":1.3},
    "Cincinnati":    {"win_pct":0.551, "MOV":1.3, "run_line_diff":1.1},
    "Seattle":       {"win_pct":0.547, "MOV":0.7, "run_line_diff":0.2},
    "Miami":         {"win_pct":0.547, "MOV":1.1, "run_line_diff":0.4},
    "Detroit":       {"win_pct":0.529, "MOV":0.3, "run_line_diff":1.1},
    "New Orleans":   {"win_pct":0.500, "MOV":2.0, "run_line_diff":1.3},
    "LA Chargers":   {"win_pct":0.494, "MOV":0.2, "run_line_diff":-1.0},
    "Indianapolis":  {"win_pct":0.488, "MOV":-0.4, "run_line_diff":-0.8},
    "Cleveland":     {"win_pct":0.471, "MOV":-2.5, "run_line_diff":-2.4},
    "Washington":    {"win_pct":0.460, "MOV":-2.6, "run_line_diff":-0.2},
    "Tennessee":     {"win_pct":0.454, "MOV":-1.9, "run_line_diff":-0.8},
    "Las Vegas":     {"win_pct":0.424, "MOV":-3.1, "run_line_diff":-0.9},
    "Denver":        {"win_pct":0.412, "MOV":-1.7, "run_line_diff":-0.3},
    "Arizona":       {"win_pct":0.412, "MOV":-1.3, "run_line_diff":0.2},
    "Atlanta":       {"win_pct":0.393, "MOV":-3.2, "run_line_diff":-2.2},
    "New England":   {"win_pct":0.388, "MOV":-1.6, "run_line_diff":-0.2},
    "Houston":       {"win_pct":0.379, "MOV":-3.9, "run_line_diff":-0.1},
    "NY Giants":     {"win_pct":0.341, "MOV":-6.4, "run_line_diff":-1.1},
    "Chicago":       {"win_pct":0.341, "MOV":-3.8, "run_line_diff":-0.2},
    "Jacksonville":  {"win_pct":0.314, "MOV":-5.2, "run_line_diff":-1.9},
    "NY Jets":       {"win_pct":0.298, "MOV":-6.9, "run_line_diff":-2.6},
    "Carolina":      {"win_pct":0.286, "MOV":-6.6, "run_line_diff":-2.2}
}

        # Built-in over/under trends since 2020
        ou_stats = {
    "Detroit":       {"over_pct":0.568, "under_pct":0.432, "total_diff":3.2},
    "Minnesota":     {"over_pct":0.565, "under_pct":0.435, "total_diff":1.5},
    "Buffalo":       {"over_pct":0.538, "under_pct":0.462, "total_diff":1.9},
    "Indianapolis":  {"over_pct":0.536, "under_pct":0.464, "total_diff":2.1},
    "Dallas":        {"over_pct":0.528, "under_pct":0.472, "total_diff":2.9},
    "Las Vegas":     {"over_pct":0.524, "under_pct":0.476, "total_diff":1.3},
    "Green Bay":     {"over_pct":0.517, "under_pct":0.483, "total_diff":1.1},
    "San Francisco": {"over_pct":0.516, "under_pct":0.484, "total_diff":0.7},
    "Cleveland":     {"over_pct":0.512, "under_pct":0.488, "total_diff":1.7},
    "Philadelphia":  {"over_pct":0.500, "under_pct":0.500, "total_diff":2.3},
    "Cincinnati":    {"over_pct":0.494, "under_pct":0.506, "total_diff":1.7},
    "Carolina":      {"over_pct":0.494, "under_pct":0.506, "total_diff":0.7},
    "Tampa Bay":     {"over_pct":0.489, "under_pct":0.511, "total_diff":0.9},
    "New England":   {"over_pct":0.488, "under_pct":0.512, "total_diff":-0.8},
    "Tennessee":     {"over_pct":0.488, "under_pct":0.512, "total_diff":0.5},
    "Chicago":       {"over_pct":0.482, "under_pct":0.518, "total_diff":0.1},
    "Arizona":       {"over_pct":0.482, "under_pct":0.519, "total_diff":-0.2},
    "Washington":    {"over_pct":0.477, "under_pct":0.523, "total_diff":2.0},
    "LA Chargers":   {"over_pct":0.477, "under_pct":0.523, "total_diff":1.0},
    "Baltimore":     {"over_pct":0.472, "under_pct":0.528, "total_diff":-0.2},
    "NY Jets":       {"over_pct":0.470, "under_pct":0.530, "total_diff":-0.5},
    "Kansas City":   {"over_pct":0.465, "under_pct":0.535, "total_diff":-1.4},
    "Miami":         {"over_pct":0.465, "under_pct":0.535, "total_diff":0.2},
    "Jacksonville":  {"over_pct":0.459, "under_pct":0.541, "total_diff":-0.2},
    "Seattle":       {"over_pct":0.459, "under_pct":0.541, "total_diff":0.4},
    "Pittsburgh":    {"over_pct":0.442, "under_pct":0.558, "total_diff":0.1},
    "Denver":        {"over_pct":0.441, "under_pct":0.560, "total_diff":-0.6},
    "LA Rams":       {"over_pct":0.435, "under_pct":0.565, "total_diff":-1.5},
    "New Orleans":   {"over_pct":0.430, "under_pct":0.570, "total_diff":-0.3},
    "Houston":       {"over_pct":0.425, "under_pct":0.575, "total_diff":-0.1},
    "Atlanta":       {"over_pct":0.417, "under_pct":0.583, "total_diff":0.1},
    "NY Giants":     {"over_pct":0.337, "under_pct":0.663, "total_diff":-2.2}
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
            from datetime import datetime
            import io

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
                lines.append(f"## 🔥 NFL Public Enemy Lines — {datetime.now(pac).strftime('%B %d, %Y')}")
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
                    lines.append(f"#### 🏈 {r['matchup']} — {r['side']} {r['odds']}")
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
                file_name=f"pel_nfl_summary_{datetime.now(pac).strftime('%Y%m%d')}.md",
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
                file_name=f"pel_nfl_summary_{datetime.now(pac).strftime('%Y%m%d')}.md",
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
        import random
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
        import random
        from datetime import datetime

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
                from markdown2 import markdown as md2_markdown
                from xhtml2pdf import pisa
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