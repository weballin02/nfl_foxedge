import datetime
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import nfl_data_py as nfl
import matplotlib.pyplot as plt

# Image rendering (fallback-safe)
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None
import math
import re
import html as _html
import urllib.request as _urlreq
from html.parser import HTMLParser as _HTMLParser

"""
NFL Referee Report Engine (Enhanced with EPA/WPA Impact Analysis)
- Uses nfl_data_py + pandas only
- Computes per-ref ATS/O/U records, penalty profiles, fairness indices, leaderboards
- NOW INCLUDES: EPA/WPA impact analysis, high-leverage penalties, situational context
- Supports multi-year blending with simple empirical-Bayes shrinkage
- Exports Markdown, CSV, and JSON ready for content

Limitations by design (no new data sources):
- No future crew assignments
- No odds history / line move tracking
- No tickets/handle splits
- No penalty accuracy/correctness data (requires external sources)
"""

# ----------------------
# Config
# ----------------------
CURRENT_YEAR = datetime.datetime.now().year
YEARS_BACK = 3                 # include up to N seasons total (if available)
USE_MULTIYEAR = True           # blend across seasons when available
MIN_GAMES_SEASON = 8           # minimum games in current season to report season-only stats
MIN_GAMES_TOTAL = 20           # minimum total games across blended seasons for leaderboards
SHRINK_K = 20                  # pseudo-counts for EB shrinkage on % stats
OUTDIR = Path("outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Spread buckets for conditioning
SPREAD_BUCKETS = [
    ("pk_to_-3", -3.0, 0.0),       # home favorite up to -3 (includes pick/heavy 0 to -3)
    ("-3.5_to_-6.5", -6.5, -3.0),
    ("-7_or_more", -100.0, -6.5),  # strong home fave
]

# Tiers and BBI weights
Z_TIER_THRESHOLDS = {
    "S": 1.0,
    "A": 0.5,
    "B": -0.5,
    "C": -1.0,
}
BBI_WEIGHTS = {
    "z_over_pct_adj": 0.4,
    "z_ats_pct_adj": 0.3,
    "z_pens_per_game": 0.2,
    "z_pen_yds_diff_avg": 0.1,
}
LOW_BUCKET_N = 5
PLOTS_DIR = OUTDIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Card rendering constants
CARD_W = 1080
CARD_H = 1350
BG_COLOR = (11, 11, 11)          # #0B0B0B
ACCENT_RED = (255, 0, 51)        # #FF0033
ACCENT_TEAL = (26, 255, 213)     # #1AFFD5
WHITE = (255, 255, 255)
MUTED = (170, 170, 170)

# -----------------------------------------
# Football Zebras assignment scraping (FBZ)
# -----------------------------------------
FBZ_TEAM_MAP = {
    "Cardinals":"ARI","Falcons":"ATL","Ravens":"BAL","Bills":"BUF","Panthers":"CAR","Bears":"CHI","Bengals":"CIN",
    "Browns":"CLE","Cowboys":"DAL","Broncos":"DEN","Lions":"DET","Packers":"GB","Texans":"HOU","Colts":"IND",
    "Jaguars":"JAX","Chiefs":"KC","Raiders":"LV","Chargers":"LAC","Rams":"LAR","Dolphins":"MIA","Vikings":"MIN",
    "Patriots":"NE","Saints":"NO","Giants":"NYG","Jets":"NYJ","Eagles":"PHI","Steelers":"PIT","49ers":"SF",
    "Seahawks":"SEA","Buccaneers":"TB","Titans":"TEN","Commanders":"WAS"
}

def _last_token(s: str) -> str:
    return str(s or "").strip().split()[-1] if s else ""

class _FBZArticleText(_HTMLParser):
    """Collect visible article text from the <article> tag, inserting line breaks at block-ish tags."""
    def __init__(self):
        super().__init__()
        self.in_article = False
        self.buf = []
    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        if self.in_article and tag in ("br","p","li","h2","h3"):
            self.buf.append("\n")
    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False
        if self.in_article and tag in ("p","li","h2","h3","div"):
            self.buf.append("\n")
    def handle_data(self, data):
        if self.in_article:
            t = data.replace("\xa0"," ").strip()
            if t:
                self.buf.append(t + " ")

def fbz_fetch_article_lines(url: str, timeout: int = 20) -> list[str]:
    """Fetch Football Zebras article HTML and return canonicalized text lines."""
    req = _urlreq.Request(url, headers={"User-Agent":"Mozilla/5.0", "Accept-Language":"en-US,en;q=0.9"})
    with _urlreq.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8","ignore")
    raw = _html.unescape(raw)
    p = _FBZArticleText(); p.feed(raw)
    text = "".join(p.buf)
    lines = [re.sub(r"\s+"," ", l).strip() for l in text.split("\n")]
    return [l for l in lines if l]

def fbz_parse_rows(lines: list[str], week: int, season: int) -> pd.DataFrame:
    """Parse canonicalized lines into a structured assignment table."""
    is_date = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,\s+[A-Za-z]+\.?\s+\d{1,2}$", re.I)
    is_match= re.compile(r"^([A-Za-z .]+?)\s+(at|vs\.)\s+([A-Za-z .]+?)$", re.I)
    is_time = re.compile(r"\b\d{1,2}(:\d{2})?\s*[ap]\.m\.\b", re.I)
    is_net  = re.compile(r"(Prime|Peacock|FOX|CBS|NFLN|ESPN(?:\s*ESPN\+)?|ABC|Amazon)", re.I)
    is_name = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")

    rows = []; curr_date = ""
    i = 0
    while i < len(lines):
        t = lines[i]
        if is_date.search(t):
            curr_date = t; i += 1; continue
        m = is_match.match(t)
        if not m:
            i += 1; continue

        away_raw, sep, home_raw = m.group(1), m.group(2).lower(), m.group(3)
        away = FBZ_TEAM_MAP.get(_last_token(away_raw), _last_token(away_raw))
        home = FBZ_TEAM_MAP.get(_last_token(home_raw), _last_token(home_raw))
        site = "neutral" if sep.startswith("vs") else "home"

        ref, tim, net = "", "", ""
        j = 1
        # greedy lookahead
        if i+j < len(lines) and (("referee" in lines[i+j].lower()) or is_name.match(lines[i+j])): ref = lines[i+j]; j += 1
        if i+j < len(lines) and is_time.search(lines[i+j]): tim = lines[i+j]; j += 1
        # network on same or next line
        if tim:
            after = is_time.sub("", tim).strip()
            if after and is_net.search(after): net = after
        if not net and i+j < len(lines) and is_net.search(lines[i+j]): net = lines[i+j]; j += 1

        ref = re.sub(r"\s*is the referee.*$", "", ref, flags=re.I).strip()
        rows.append({
            "season": season, "week": int(week), "game_date": curr_date, "kickoff_et": tim,
            "home_team": home, "away_team": away, "site_type": site,
            "network": net, "ref_name": ref
        })
        i += j
    df = pd.DataFrame(rows)
    if "ref_name" in df.columns:
        df["ref_name"] = df["ref_name"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def get_ref_assignments_from_fbz(week: int, season: int, url: str | None = None, min_rows: int = 8) -> pd.DataFrame | None:
    """
    One-shot fetch/parse of Football Zebras assignments for a given week.
    Does not poll; just returns a DataFrame or None on failure/insufficient rows.
    """
    try:
        if url is None:
            # allow env override by setting global FBZ_URL before calling main()
            url = globals().get("FBZ_URL", None)
            if url is None:
                return None
        lines = fbz_fetch_article_lines(url)
        df = fbz_parse_rows(lines, week=week, season=season)
        if df is None or df.empty or len(df) < min_rows:
            return None
        df = df[df["ref_name"].astype(str).str.len() >= 3].copy()
        return df if not df.empty else None
    except Exception:
        return None

def merge_assignments_into_ref_games(ref_games: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Merge FBZ assignments into our ref_games table by week/home/away.
    Overwrites ref_name for matching rows in the target week.
    """
    if assignments is None or assignments.empty:
        return ref_games
    needed = {"week", "home_team", "away_team", "ref_name"}
    if not needed.issubset(assignments.columns):
        return ref_games
    a = assignments.copy()
    a["home_team"] = a["home_team"].astype(str).str.upper()
    a["away_team"] = a["away_team"].astype(str).str.upper()
    g = ref_games.copy()
    g["home_team"] = g["home_team"].astype(str).str.upper()
    g["away_team"] = g["away_team"].astype(str).str.upper()
    key = ["week","home_team","away_team"]
    merged = g.merge(a[key + ["ref_name"]].rename(columns={"ref_name":"ref_name_fbz"}), on=key, how="left")
    mask = merged["ref_name_fbz"].notna() & (merged["ref_name_fbz"].astype(str).str.len() > 0)
    merged.loc[mask, "ref_name"] = merged.loc[mask, "ref_name_fbz"]
    merged.drop(columns=["ref_name_fbz"], inplace=True)
    return merged

def _load_font(size: int):
    """Try to load preferred fonts; fall back to default."""
    if ImageFont is None:
        return None
    for name in ["Impact.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def _measure_text(draw, text: str, font):
    """Return (width, height) for text across Pillow versions."""
    try:
        if hasattr(draw, "textbbox"):
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return (r - l, b - t)
        if hasattr(draw, "textsize"):
            return draw.textsize(text, font=font)
    except Exception:
        pass
    # Fallback heuristic if everything else fails
    return (len(text) * 8, 16)

# Optional: historical odds from Excel (open/close, min/max). If present, we enrich market metrics.
ODDS_CANDIDATES = [
    os.environ.get("FLAGFILES_ODDS_XLSX", "").strip() or None,
    str(OUTDIR / "NFL Historical Results and Odds.xlsx"),
    "NFL Historical Results and Odds.xlsx",
    os.path.join("data", "NFL Historical Results and Odds.xlsx"),
]
ODDS_CANDIDATES = [p for p in ODDS_CANDIDATES if p]

# Optional: 2025 crew roster CSV (Referee, U, DJ, LJ, FJ, SJ, BJ, RO, RA)
CREWS_CANDIDATES = [
    os.environ.get("FLAGFILES_CREWS_CSV", "").strip() or None,
    str(OUTDIR / "nfl_2025_officiating_crews.csv"),
    "nfl_2025_officiating_crews.csv",
    os.path.join("data", "nfl_2025_officiating_crews.csv"),
]
CREWS_CANDIDATES = [p for p in CREWS_CANDIDATES if p]

# Embedded 2025 NFL officiating crews (source: Football Zebras 2025 crews PDF)
CREWS_2025_EMBEDDED = [
    {"Referee": "Brad Allen", "U": "Marcus Woods", "DJ": "Sarah Thomas", "LJ": "Walter Flowers", "FJ": "Rick Patterson", "SJ": "Chad Hill", "BJ": "Tyree Walton", "RO": "Kevin Brown", "RA": "Randy Roseberry"},
    {"Referee": "Clete Blakeman", "U": "Scott Campbell", "DJ": "Andy Warner", "LJ": "Kent Payne", "FJ": "Karina Tovar", "SJ": "", "BJ": "", "RO": "Chad Adams", "RA": "Amber Cipriani"},
    {"Referee": "Carl Cheffers", "U": "Derek Anderson", "DJ": "Daniel Gallagher", "LJ": "Quentin Givens", "FJ": "Nate Jones", "SJ": "Eugene Hall", "BJ": "Martin Hankins", "RO": "Brian Matoren", "RA": "Daniel Bouldrick"},
    {"Referee": "Land Clark", "U": "Mark Pellis", "DJ": "Tom Stephan", "LJ": "Jeff Hutcheon", "FJ": "Jabir Walker", "SJ": "Dominique Pender", "BJ": "Brad Freeman", "RO": "Gerald Frye", "RA": "Kris Raper"},
    {"Referee": "Alan Eck", "U": "Tab Slaughter", "DJ": "David Oliver", "LJ": "Greg Bradley", "FJ": "John Jenkins", "SJ": "Dale Shaw", "BJ": "Grantis Bell", "RO": "Joe Wollan", "RA": "Larry Hanson"},
    {"Referee": "Adrian Hill", "U": "Roy Ellison", "DJ": "Derick Bowers", "LJ": "Julian Mapp", "FJ": "Tra Boger", "SJ": "Clay Reynard", "BJ": "Greg Steed", "RO": "Bob Hubbell", "RA": "Durwood Manley"},
    {"Referee": "Shawn Hochuli", "U": "Larry Smith", "DJ": "Jim Quirk", "LJ": "Tim Podraza", "FJ": "Jason Ledet", "SJ": "Jim Quirk", "BJ": "Russell", "RO": "Nicholson", "RA": "Adam Choate"},
    {"Referee": "John Hussey", "U": "Duane Heydt", "DJ": "Max Causey", "LJ": "Carl Johnson", "FJ": "Anthony Flemming", "SJ": "Allen Baynes", "BJ": "Matt Edwards", "RO": "Andrew Lambert", "RA": "Sebrina Brunson"},
    {"Referee": "Alex Kemp", "U": "Brandon Ellison", "DJ": "Mike Carr", "LJ": "Rusty Baynes", "FJ": "Sean Petty", "SJ": "Lo van Pham", "BJ": "Scott Helverson", "RO": "Tim England", "RA": "Julie Johnson"},
    {"Referee": "Clay Martin", "U": "Steve Woods", "DJ": "Jerod Phillips", "LJ": "Brian Perry", "FJ": "Dave Hawkshaw", "SJ": "Alonzo Ramsey", "BJ": "Greg Wilson", "RO": "Bryant Thompson", "RA": "Artenzia Young-Seigler"},
    {"Referee": "Alex Moore", "U": "Terry Killens", "DJ": "Dana McKenzie", "LJ": "Tom Eaton", "FJ": "Mearl Robinson", "SJ": "Anthony Jeffries", "BJ": "Terrence Miles", "RO": "Mike Cerimeli", "RA": "Desiree Abrams"},
    {"Referee": "Scott Novak", "U": "Mike Morton", "DJ": "Brian Sakowski", "LJ": "Mark Stewart", "FJ": "Terry Brown", "SJ": "Don Willard", "BJ": "Tony Josselyn", "RO": "Matt Sumstine", "RA": "Brian Davies"},
    {"Referee": "Brad Rogers", "U": "Bryan Neale", "DJ": "Patrick Turner", "LJ": "Kevin Codey", "FJ": "Joe Blubaugh", "SJ": "David Meslow", "BJ": "Greg Yette", "RO": "Denise Crudup", "RA": "Brian Smith"},
    {"Referee": "Shawn Smith", "U": "Tra Blake", "DJ": "Jay Bilbo", "LJ": "Jeff Seeman", "FJ": "Dyrol Prioleau", "SJ": "Boris Cheek", "BJ": "Dino Paganelli", "RO": "Mike Wimmer", "RA": "Larry Hill Jr."},
    {"Referee": "Ron Torbert", "U": "Barry Anderson", "DJ": "Frank LeBlanc", "LJ": "Brian Bolinger", "FJ": "Ryan Dickson", "SJ": "Keith Washington", "BJ": "Courtney Brown", "RO": "Kevin Stine", "RA": "Marty Abezetian"},
    {"Referee": "Bill Vinovich", "U": "Scott Walker", "DJ": "Dale Keller", "LJ": "Tripp Sutter", "FJ": "Aaron Santi", "SJ": "Jimmy Buchanan", "BJ": "Todd Prukop", "RO": "Chad Wakefield", "RA": "Jim Van Geffen"},
    {"Referee": "Craig Wrolstad", "U": "Brandon Cruse", "DJ": "Danny Short", "LJ": "Brett Bergman", "FJ": "Jeff Shears", "SJ": "Frank Steratore", "BJ": "Rich Martinez", "RO": "Gavin Anderson", "RA": "Ken Hall"},
]

# ----------------------
# Helpers
# ----------------------

def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return (
        s.replace("\u00a0", " ")  # non-breaking space
         .replace("  ", " ")
         .strip()
    )

def _sanitize_crews(df: pd.DataFrame) -> pd.DataFrame:
    """Light hygiene: blank obviously broken role entries, normalize spacing."""
    if df is None or df.empty:
        return df
    roles = ["U","DJ","LJ","FJ","SJ","BJ","RO","RA"]
    for c in roles:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _norm_name(x))
            df[c] = df[c].apply(lambda x: "—" if isinstance(x, str) and x != "" and (" " not in x) else x)
    df["crew_members"] = df.apply(lambda r: [{"role":role, "name": r[role]} for role in roles if r.get(role)], axis=1)
    df["crew_compact"] = df.apply(lambda r: ", ".join([f"{role}: {r[role]}" for role in roles if r.get(role)]), axis=1)
    return df

TEAM_NAME_TO_ABBR = {
    # AFC East
    "New England Patriots": "NE", "Miami Dolphins": "MIA", "Buffalo Bills": "BUF", "New York Jets": "NYJ",
    # AFC North
    "Pittsburgh Steelers": "PIT", "Baltimore Ravens": "BAL", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    # AFC South
    "Indianapolis Colts": "IND", "Tennessee Titans": "TEN", "Jacksonville Jaguars": "JAX", "Houston Texans": "HOU",
    # AFC West
    "Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Denver Broncos": "DEN",
    # NFC East
    "Dallas Cowboys": "DAL", "Philadelphia Eagles": "PHI", "New York Giants": "NYG", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS", "Washington Redskins": "WAS",
    # NFC North
    "Green Bay Packers": "GB", "Chicago Bears": "CHI", "Minnesota Vikings": "MIN", "Detroit Lions": "DET",
    # NFC South
    "New Orleans Saints": "NO", "Tampa Bay Buccaneers": "TB", "Carolina Panthers": "CAR", "Atlanta Falcons": "ATL",
    # NFC West
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Los Angeles Rams": "LA", "St. Louis Rams": "LA", "Arizona Cardinals": "ARI",
}

def _try_read_excel(path: str, sheet: str | None = None) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    except Exception:
        return None

def load_odds_history_from_excel() -> pd.DataFrame | None:
    """Attempt to load odds history from known locations. Returns normalized DataFrame or None."""
    for p in ODDS_CANDIDATES:
        if not p or not os.path.exists(p):
            continue
        df = _try_read_excel(p, sheet="Data")
        if df is None or df.empty:
            df = _try_read_excel(p, sheet=None)
        if df is None or df.empty:
            continue
        cols = {c.strip(): c for c in df.columns if isinstance(c, str)}
        date_col = next((c for c in cols if c.lower().startswith("date")), None)
        home_col = next((c for c in cols if "home" in c.lower() and "team" in c.lower()), None)
        away_col = next((c for c in cols if "away" in c.lower() and "team" in c.lower()), None)
        sp_open = next((c for c in cols if c.lower().startswith("home line open")), None)
        sp_close = next((c for c in cols if c.lower().startswith("home line close")), None)
        tot_open = next((c for c in cols if c.lower().startswith("total score open")), None)
        tot_close = next((c for c in cols if c.lower().startswith("total score close")), None)
        hs = next((c for c in cols if c.lower().startswith("home score")), None)
        as_ = next((c for c in cols if c.lower().startswith("away score")), None)
        playoff_col = next((c for c in cols if "playoff" in c.lower()), None)
        if not all([date_col, home_col, away_col, sp_open, sp_close, tot_open, tot_close]):
            continue
        keep = [date_col, home_col, away_col, sp_open, sp_close, tot_open, tot_close]
        if hs and as_: keep += [hs, as_]
        if playoff_col: keep += [playoff_col]
        df = df.loc[:, keep].copy()
        rename = {
            date_col: "game_date",
            home_col: "home_team_full",
            away_col: "away_team_full",
            sp_open: "spread_home_open",
            sp_close: "spread_home_close",
            tot_open: "total_open",
            tot_close: "total_close",
        }
        if hs: rename[hs] = "home_score_odds"
        if as_: rename[as_] = "away_score_odds"
        if playoff_col: rename[playoff_col] = "is_playoff"
        df = df.rename(columns=rename)

        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        for c in ["spread_home_open", "spread_home_close", "total_open", "total_close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["home_team"] = df["home_team_full"].map(TEAM_NAME_TO_ABBR).fillna(df["home_team_full"])
        df["away_team"] = df["away_team_full"].map(TEAM_NAME_TO_ABBR).fillna(df["away_team_full"])

        if "is_playoff" in df.columns:
            df = df[df["is_playoff"].astype(str).str.lower().isin(["false", "0", "no"]) | df["is_playoff"].isna()]

        df = df.dropna(subset=["spread_home_open", "spread_home_close", "total_open", "total_close"])
        df = df.sort_values(["game_date"]).drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")
        return df
    return None

def merge_odds_into_ref_games(ref_games: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    if odds is None or odds.empty:
        return ref_games
    g = ref_games.copy()
    date_col = None
    for cand in ["gameday", "game_date", "game_day"]:
        if cand in g.columns:
            date_col = cand
            break
    if date_col is None:
        merged = g.merge(
            odds[["home_team", "away_team", "spread_home_open", "spread_home_close", "total_open", "total_close", "game_date"]],
            on=["home_team", "away_team"], how="left", suffixes=("", "_odds")
        )
    else:
        g["game_date_norm"] = pd.to_datetime(g[date_col]).dt.normalize()
        odds["game_date_norm"] = pd.to_datetime(odds["game_date"]).dt.normalize()
        merged = g.merge(
            odds[["game_date_norm", "home_team", "away_team", "spread_home_open", "spread_home_close", "total_open", "total_close"]],
            left_on=["game_date_norm", "home_team", "away_team"],
            right_on=["game_date_norm", "home_team", "away_team"], how="left"
        )
        merged.drop(columns=["game_date_norm"], inplace=True)

    for c in ["spread_home_open", "spread_home_close", "total_open", "total_close"]:
        if c not in merged.columns:
            merged[c] = np.nan

    merged["steam_spread"] = merged["spread_home_close"] - merged["spread_home_open"]
    merged["steam_total"] = merged["total_close"] - merged["total_open"]

    merged["ats_error_open"] = (merged["home_margin"] + merged["spread_home_open"]).abs()
    merged["ats_error_close"] = (merged["home_margin"] + merged["spread_home_close"]).abs()
    merged["total_error_open"] = (merged["final_total"] - merged["total_open"]).abs()
    merged["total_error_close"] = (merged["final_total"] - merged["total_close"]).abs()

    merged["respect_spread"] = (merged["ats_error_close"] < merged["ats_error_open"]).astype(float)
    merged["respect_total"] = (merged["total_error_close"] < merged["total_error_open"]).astype(float)

    merged["dir_consistent_total"] = (np.sign(merged["steam_total"]).fillna(0) == merged["ou_result"]).astype(float)
    merged["dir_consistent_spread"] = (np.sign(merged["steam_spread"]).fillna(0) == merged["home_ats"]).astype(float)
    return merged

# ----------------------
# 2025 Crews loader & integration
# ----------------------

def load_2025_crews() -> pd.DataFrame | None:
    for path in CREWS_CANDIDATES:
        try:
            if path and os.path.exists(path):
                df = pd.read_csv(path)
                rename = {c: c.strip() for c in df.columns}
                df = df.rename(columns=rename)
                expected = ["Referee","U","DJ","LJ","FJ","SJ","BJ","RO","RA"]
                missing = [c for c in expected if c not in df.columns]
                if missing:
                    continue
                for c in expected:
                    df[c] = df[c].apply(_norm_name)
                roles = ["U","DJ","LJ","FJ","SJ","BJ","RO","RA"]
                df["crew_members"] = df.apply(lambda r: [{"role":role, "name": r[role]} for role in roles if r.get(role)], axis=1)
                df["crew_compact"] = df.apply(lambda r: ", ".join([f"{role}: {r[role]}" for role in roles if r.get(role)]), axis=1)
                df["ref_name_key"] = df["Referee"].apply(_norm_name).str.lower()
                df = _sanitize_crews(df)
                return df
        except Exception:
            continue
    try:
        if CREWS_2025_EMBEDDED:
            df = pd.DataFrame(CREWS_2025_EMBEDDED)
            expected = ["Referee","U","DJ","LJ","FJ","SJ","BJ","RO","RA"]
            for c in expected:
                if c not in df.columns:
                    df[c] = ""
                df[c] = df[c].apply(_norm_name)
            roles = ["U","DJ","LJ","FJ","SJ","BJ","RO","RA"]
            df["crew_members"] = df.apply(lambda r: [{"role":role, "name": r[role]} for role in roles if r.get(role)], axis=1)
            df["crew_compact"] = df.apply(lambda r: ", ".join([f"{role}: {r[role]}" for role in roles if r.get(role)]), axis=1)
            df["ref_name_key"] = df["Referee"].apply(_norm_name).str.lower()
            return df
    except Exception:
        return None
    return None

def attach_crews_to_summary(ref_summary: pd.DataFrame, crews: pd.DataFrame | None) -> pd.DataFrame:
    if crews is None or ref_summary is None or ref_summary.empty:
        return ref_summary
    df = ref_summary.copy()
    df["ref_name_key"] = df["ref_name"].apply(_norm_name).str.lower()
    keep = crews[["ref_name_key","Referee","crew_compact","crew_members"]].copy()
    merged = df.merge(keep, on="ref_name_key", how="left")
    merged.drop(columns=["ref_name_key"], inplace=True)
    return merged

def eb_shrink_rate(p_raw: float, n: int, p_league: float, k: int = SHRINK_K) -> float:
    """Empirical-Bayes shrinkage of a rate/percentage toward league mean."""
    if n <= 0:
        return p_league
    return (p_raw * n + p_league * k) / (n + k)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sd


def safe_div(a, b):
    return a / b if b else 0.0


# ----------------------
# Load data (multi-year if available)
# ----------------------

def load_years() -> list[int]:
    years = [CURRENT_YEAR - i for i in range(YEARS_BACK)] if USE_MULTIYEAR else [CURRENT_YEAR]
    valid_years = []
    for y in years:
        try:
            sch = nfl.import_schedules([y])
            if not sch.empty:
                valid_years.append(y)
        except Exception:
            continue
    if not valid_years:
        valid_years = [CURRENT_YEAR - 1]
    return sorted(valid_years)


def load_frames(years: list[int]):
    schedules = nfl.import_schedules(years)
    pbp = nfl.import_pbp_data(years)
    officials = nfl.import_officials(years)
    schedules = schedules[schedules["game_type"] == "REG"].copy()
    if "season_type" in pbp.columns:
        pbp = pbp[pbp["season_type"] == "REG"].copy()
    return schedules, pbp, officials


# ----------------------
# NEW: EPA/WPA Impact Analysis
# ----------------------

def compute_penalty_impact_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance penalty data with EPA/WPA impact and situational context.
    Returns a DataFrame with penalty-level impact metrics.
    """
    pen_cols = [
        "game_id", "play_id", "penalty", "penalty_type", "penalty_yards",
        "penalty_team", "home_team", "away_team", "posteam", "defteam",
        "down", "ydstogo", "yardline_100", "qtr", "game_seconds_remaining",
        "score_differential", "wp", "epa", "wpa",
        "first_down", "first_down_penalty", "penalty_1st_down"
    ]
    
    # Only keep columns that exist
    available_cols = [c for c in pen_cols if c in pbp.columns]
    pens = pbp[available_cols].copy()
    pens = pens[pens["penalty"] == 1]
    
    # Tag penalties against home/away
    pens["pen_against_side"] = np.where(
        pens["penalty_team"] == pens["home_team"], "home",
        np.where(pens["penalty_team"] == pens["away_team"], "away", "other")
    )
    
    # High-leverage situation flags
    if "qtr" in pens.columns:
        pens["is_4th_qtr"] = pens["qtr"] == 4
    else:
        pens["is_4th_qtr"] = False
        
    if "game_seconds_remaining" in pens.columns:
        pens["is_final_2min"] = pens["game_seconds_remaining"] <= 120
    else:
        pens["is_final_2min"] = False
        
    if "yardline_100" in pens.columns:
        pens["is_red_zone"] = pens["yardline_100"] <= 20
        pens["is_goal_to_go"] = pens["yardline_100"] <= 10
    else:
        pens["is_red_zone"] = False
        pens["is_goal_to_go"] = False
        
    if "score_differential" in pens.columns:
        pens["is_close_game"] = pens["score_differential"].abs() <= 8
        pens["is_one_score"] = pens["score_differential"].abs() <= 7
    else:
        pens["is_close_game"] = False
        pens["is_one_score"] = False
    
    # High-leverage flag (multiple conditions)
    pens["is_high_leverage"] = (
        pens["is_4th_qtr"] & pens["is_close_game"]
    ) | pens["is_final_2min"] | (pens["is_red_zone"] & pens["is_close_game"])
    
    # EPA/WPA absolute impact
    if "epa" in pens.columns:
        pens["epa_abs"] = pens["epa"].abs()
    else:
        pens["epa_abs"] = 0
        
    if "wpa" in pens.columns:
        pens["wpa_abs"] = pens["wpa"].abs()
    else:
        pens["wpa_abs"] = 0
    
    # Down/distance context
    if "down" in pens.columns and "ydstogo" in pens.columns:
        pens["down_distance"] = pens["down"].astype(str) + " & " + pens["ydstogo"].astype(str)
        pens["is_3rd_down"] = pens["down"] == 3
        pens["is_4th_down"] = pens["down"] == 4
        pens["is_passing_down"] = (pens["down"].isin([2, 3])) & (pens["ydstogo"] >= 7)
    else:
        pens["down_distance"] = "unknown"
        pens["is_3rd_down"] = False
        pens["is_4th_down"] = False
        pens["is_passing_down"] = False
    
    # First down by penalty detection
    pens["fd_by_pen"] = False
    for col in ["first_down_penalty", "penalty_1st_down"]:
        if col in pens.columns:
            pens["fd_by_pen"] = pens["fd_by_pen"] | pens[col].fillna(False).astype(bool)
    
    return pens


def aggregate_impact_by_game_ref(penalty_impacts: pd.DataFrame, officials: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate EPA/WPA and situational penalty metrics by game and referee.
    """
    # Get referee per game
    refs = officials[officials["off_pos"] == "R"][["game_id", "name", "season"]].rename(columns={"name": "ref_name"})
    
    # Merge penalties with referee info
    pen_ref = penalty_impacts.merge(refs, on="game_id", how="left")
    pen_ref = pen_ref[pen_ref["ref_name"].notna()]
    
    # Aggregate by game_id and ref_name
    impact_agg = pen_ref.groupby(["game_id", "ref_name"]).agg(
        # EPA/WPA totals
        total_epa_impact=("epa_abs", "sum"),
        total_wpa_impact=("wpa_abs", "sum"),
        avg_epa_per_penalty=("epa_abs", "mean"),
        avg_wpa_per_penalty=("wpa_abs", "mean"),
        
        # High-leverage penalties
        high_leverage_penalties=("is_high_leverage", "sum"),
        red_zone_penalties=("is_red_zone", "sum"),
        fourth_qtr_penalties=("is_4th_qtr", "sum"),
        final_2min_penalties=("is_final_2min", "sum"),
        close_game_penalties=("is_close_game", "sum"),
        
        # Down/distance context
        third_down_penalties=("is_3rd_down", "sum"),
        fourth_down_penalties=("is_4th_down", "sum"),
        passing_down_penalties=("is_passing_down", "sum"),
        first_downs_by_penalty=("fd_by_pen", "sum"),
    ).reset_index()
    
    return impact_agg


# ----------------------
# Build per-ref, per-game table (ENHANCED)
# ----------------------

PENALTY_WHITELIST = {
    "Defensive Pass Interference",
    "Defensive Holding",
    "Illegal Contact",
    "Roughing the Passer",
    "Unnecessary Roughness",
    "Offensive Holding",
    "False Start",
    "Delay of Game",
    "Illegal Substitution",
    "Illegal Shift",
    "Illegal Motion",
}

def build_ref_game_table(schedules: pd.DataFrame, pbp: pd.DataFrame, officials: pd.DataFrame) -> tuple:
    """Enhanced to include EPA/WPA impact analysis."""
    
    # Identify the Referee per game
    refs = officials[officials["off_pos"] == "R"][["game_id", "name", "season"]].rename(columns={"name": "ref_name"})

    # NEW: Compute penalty impact metrics from pbp
    penalty_impacts = compute_penalty_impact_metrics(pbp)
    
    # Get impact aggregations by game and ref
    impact_by_game = aggregate_impact_by_game_ref(penalty_impacts, officials)
    
    # Traditional penalty aggregations
    pens = penalty_impacts.copy()
    
    # Aggregate penalties at game level for home/away counts and yards
    agg_pen = pens.groupby(["game_id", "pen_against_side"]).agg(
        pen_count=("play_id", "count"),
        pen_yards=("penalty_yards", "sum"),
    ).reset_index()
    agg_piv = agg_pen.pivot(index="game_id", columns="pen_against_side", values=["pen_count", "pen_yards"]).fillna(0)
    agg_piv.columns = [f"{a}_{b}" for a, b in agg_piv.columns]
    agg_piv = agg_piv.reset_index()

    # Per-game penalty type counts (wide)
    type_counts = pens.groupby(["game_id", "penalty_type"]).size().rename("type_count").reset_index()

    # Merge schedule info
    sch_cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "home_score", "away_score", "spread_line", "total_line"
    ]
    sch = schedules[sch_cols].copy()

    ref_games = sch.merge(refs, on=["game_id", "season"], how="left")
    ref_games = ref_games.merge(agg_piv, on="game_id", how="left")
    
    # NEW: Merge impact metrics
    ref_games = ref_games.merge(impact_by_game, on=["game_id", "ref_name"], how="left")

    # Normalize spread sign
    sp = ref_games["spread_line"].dropna()
    if not sp.empty:
        neg_rate = (sp < 0).mean()
        pos_rate = (sp > 0).mean()
        sign_factor = 1
        if pos_rate > neg_rate:
            sign_factor = -1
        ref_games["spread_norm"] = sign_factor * ref_games["spread_line"]
    else:
        ref_games["spread_norm"] = ref_games["spread_line"]

    # Drop games without an identified referee
    ref_games = ref_games[ref_games["ref_name"].notna()].copy()
    ref_games["ref_name"] = ref_games["ref_name"].astype(str).str.strip()
    ref_games = ref_games[~ref_games["ref_name"].str.lower().isin({"nan", "none", "", "0"})].copy()

    # Outcomes
    ref_games["final_total"] = ref_games["home_score"].fillna(0) + ref_games["away_score"].fillna(0)
    ref_games["home_margin"] = ref_games["home_score"].fillna(0) - ref_games["away_score"].fillna(0)

    # ATS outcomes
    spread_mask = ref_games["spread_norm"].notna()
    ref_games["home_ats"] = np.nan
    ref_games.loc[spread_mask & (ref_games["home_margin"] + ref_games["spread_norm"] > 0), "home_ats"] = 1
    ref_games.loc[spread_mask & (ref_games["home_margin"] + ref_games["spread_norm"] == 0), "home_ats"] = 0
    ref_games.loc[spread_mask & (ref_games["home_margin"] + ref_games["spread_norm"] < 0), "home_ats"] = -1

    ref_games["home_is_fav"] = np.where(spread_mask, (ref_games["spread_norm"] < 0).astype(int), np.nan)

    # O/U
    total_mask = ref_games["total_line"].notna()
    ref_games["ou_result"] = np.nan
    ref_games.loc[total_mask & (ref_games["final_total"] > ref_games["total_line"]), "ou_result"] = 1
    ref_games.loc[total_mask & (ref_games["final_total"] == ref_games["total_line"]), "ou_result"] = 0
    ref_games.loc[total_mask & (ref_games["final_total"] < ref_games["total_line"]), "ou_result"] = -1

    # Penalty fairness metrics
    for col in ["pen_count_home", "pen_count_away", "pen_yards_home", "pen_yards_away"]:
        if col not in ref_games.columns:
            ref_games[col] = 0
    ref_games["pen_count_diff_away_minus_home"] = ref_games["pen_count_away"] - ref_games["pen_count_home"]
    ref_games["pen_yards_diff_away_minus_home"] = ref_games["pen_yards_away"] - ref_games["pen_yards_home"]

    # Flag Files angles at per-game level
    p3 = pens.copy()
    p3["is_3rd"] = p3.get("down", np.nan) == 3
    defensive_set = {"Defensive Pass Interference", "Defensive Holding", "Illegal Contact", "Roughing the Passer", "Unnecessary Roughness"}
    p3["is_def_flag"] = p3["penalty_type"].isin(defensive_set)

    bail = p3.groupby("game_id").apply(
        lambda df: pd.Series({
            "bailout_3rd_def_flags": int((df["is_3rd"] & df["is_def_flag"]).sum()),
            "bailout_3rd_fd":        int((df["is_3rd"] & df["is_def_flag"] & df["fd_by_pen"]).sum())
        })
    , include_groups=False).reset_index()
    ref_games = ref_games.merge(bail, on="game_id", how="left").fillna({"bailout_3rd_def_flags":0, "bailout_3rd_fd":0})

    # Free yards
    pass_def_flags = {"Defensive Pass Interference","Defensive Holding","Illegal Contact"}
    fy = pens[pens["penalty_type"].isin(pass_def_flags)].groupby("game_id")["penalty_yards"].sum().rename("free_pass_yards").reset_index()
    ref_games = ref_games.merge(fy, on="game_id", how="left").fillna({"free_pass_yards":0})

    # Drive-killers
    kill_types = {"Offensive Holding","False Start"}
    dk = pens[pens["penalty_type"].isin(kill_types)].copy()
    dk["early_down"] = dk.get("down", np.nan).isin([1,2])
    dk["drive_kill_flag"] = False
    dk.loc[dk["penalty_type"]=="Offensive Holding", "drive_kill_flag"] = dk["early_down"] & (dk["penalty_yards"].abs() >= 10)
    if "ydstogo" in dk.columns:
        fs_mask = (dk["penalty_type"]=="False Start") & (dk["down"]==2) & (pd.to_numeric(dk["ydstogo"], errors="coerce")<=4)
        dk.loc[fs_mask, "drive_kill_flag"] = True
    dk_agg = dk.groupby("game_id")["drive_kill_flag"].sum().rename("drive_killers").reset_index()
    ref_games = ref_games.merge(dk_agg, on="game_id", how="left").fillna({"drive_killers":0})

    # Tempo drags
    drag_types = {"Delay of Game","Illegal Substitution","Illegal Shift","Illegal Motion"}
    td = pens[pens["penalty_type"].isin(drag_types)].groupby("game_id").size().rename("tempo_drags").reset_index()
    ref_games = ref_games.merge(td, on="game_id", how="left").fillna({"tempo_drags":0})

    # First downs by penalty
    fdp = pens.copy()
    fdp_agg = fdp.groupby("game_id")["fd_by_pen"].sum().rename("first_downs_by_pen").reset_index()
    ref_games = ref_games.merge(fdp_agg, on="game_id", how="left").fillna({"first_downs_by_pen":0})

    # QB protection tilt
    gf_types = {"Roughing the Passer","Unnecessary Roughness"}
    gfw = pens[pens["penalty_type"].isin(gf_types)].groupby("game_id").size().rename("qb_protect_flags").reset_index()
    ref_games = ref_games.merge(gfw, on="game_id", how="left").fillna({"qb_protect_flags":0})

    return ref_games, type_counts, penalty_impacts


# ----------------------
# League baselines & top penalty types
# ----------------------

def league_baselines(ref_games: pd.DataFrame, type_counts: pd.DataFrame) -> dict:
    total_games = len(ref_games)
    home_ats_w = (ref_games["home_ats"] == 1).sum()
    ou_over_w = (ref_games["ou_result"] == 1).sum()
    p_home_ats = safe_div(home_ats_w, ref_games["home_ats"].notna().sum() - (ref_games["home_ats"] == 0).sum())
    p_ou_over = safe_div(ou_over_w, ref_games["ou_result"].notna().sum() - (ref_games["ou_result"] == 0).sum())

    league_pen_count = (ref_games["pen_count_home"] + ref_games["pen_count_away"]).fillna(0).mean()

    top_types = (
        type_counts.groupby("penalty_type")["type_count"].sum().sort_values(ascending=False).head(10).index.tolist()
    )
    top_types = list(dict.fromkeys(list(PENALTY_WHITELIST) + top_types))

    type_pg = (
        type_counts.groupby("penalty_type")["type_count"].sum() / ref_games["game_id"].nunique()
    )

    return {
        "total_games": total_games,
        "p_home_ats": p_home_ats,
        "p_ou_over": p_ou_over,
        "penalties_per_game": league_pen_count,
        "top_types": top_types,
        "type_per_game": type_pg.to_dict(),
    }


# ----------------------
# Aggregate per-ref stats (ENHANCED)
# ----------------------

def aggregate_refs(ref_games: pd.DataFrame, type_counts: pd.DataFrame, baselines: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enhanced to include EPA/WPA impact aggregations."""
    
    top_types = baselines["top_types"]
    type_pivot = (
        type_counts[type_counts["penalty_type"].isin(top_types)]
        .pivot_table(index="game_id", columns="penalty_type", values="type_count", aggfunc="sum", fill_value=0)
        .reset_index()
    )

    g = ref_games.merge(type_pivot, on="game_id", how="left").fillna(0)
    
    for c in ["steam_spread","steam_total","respect_spread","respect_total",
              "ats_error_open","ats_error_close","total_error_open","total_error_close",
              "dir_consistent_spread","dir_consistent_total"]:
        if c not in g.columns:
            g[c] = np.nan
    if "spread_norm" not in g.columns:
        g["spread_norm"] = g.get("spread_line", np.nan)

    # STAGE 1: Deduplicate per-game outcomes
    impact_cols = [
        'total_epa_impact', 'total_wpa_impact', 'avg_epa_per_penalty', 'avg_wpa_per_penalty',
        'high_leverage_penalties', 'red_zone_penalties', 'fourth_qtr_penalties',
        'final_2min_penalties', 'close_game_penalties', 'third_down_penalties',
        'fourth_down_penalties', 'passing_down_penalties', 'first_downs_by_penalty'
    ]
    
    agg_dict = {
        'season': 'first',
        'home_ats': 'first',
        'ou_result': 'first',
        'pen_count_home': 'first',
        'pen_count_away': 'first',
        'pen_yards_home': 'first',
        'pen_yards_away': 'first',
        'pen_count_diff_away_minus_home': 'first',
        'pen_yards_diff_away_minus_home': 'first',
        'bailout_3rd_fd': 'first',
        'free_pass_yards': 'first',
        'drive_killers': 'first',
        'tempo_drags': 'first',
        'first_downs_by_pen': 'first',
        'qb_protect_flags': 'first',
        'steam_spread': 'first',
        'steam_total': 'first',
        'respect_spread': 'first',
        'respect_total': 'first',
        'ats_error_open': 'first',
        'ats_error_close': 'first',
        'total_error_open': 'first',
        'total_error_close': 'first',
        'dir_consistent_spread': 'first',
        'dir_consistent_total': 'first',
        'spread_norm': 'first',
    }
    
    # Add impact columns if they exist
    for col in impact_cols:
        if col in g.columns:
            agg_dict[col] = 'first'
    
    g0 = g.groupby(['ref_name', 'game_id'], as_index=False).agg(agg_dict)
    
    # Add penalty type columns
    for t in top_types:
        if t in g.columns:
            type_agg = g.groupby(['ref_name', 'game_id'])[t].sum().reset_index()
            type_agg = type_agg.rename(columns={t: f"{t}_per_game"})
            g0 = g0.merge(type_agg, on=['ref_name', 'game_id'], how='left').fillna(0)

    # STAGE 2: Aggregate per-ref from deduplicated data
    base_agg = {
        "games": ("game_id", "nunique"),
        "seasons": ("season", lambda x: sorted(set(x))),
        "ats_w": ("home_ats", lambda s: (s == 1).sum()),
        "ats_l": ("home_ats", lambda s: (s == -1).sum()),
        "ats_p": ("home_ats", lambda s: (s == 0).sum()),
        "ou_over": ("ou_result", lambda s: (s == 1).sum()),
        "ou_under": ("ou_result", lambda s: (s == -1).sum()),
        "ou_push": ("ou_result", lambda s: (s == 0).sum()),
        "pen_home": ("pen_count_home", "sum"),
        "pen_away": ("pen_count_away", "sum"),
        "yards_home": ("pen_yards_home", "sum"),
        "yards_away": ("pen_yards_away", "sum"),
        "pen_cnt_diff_avg": ("pen_count_diff_away_minus_home", "mean"),
        "pen_yds_diff_avg": ("pen_yards_diff_away_minus_home", "mean"),
        "bailout_3rd_fd": ("bailout_3rd_fd", "sum"),
        "free_pass_yards": ("free_pass_yards", "sum"),
        "drive_killers": ("drive_killers", "sum"),
        "tempo_drags": ("tempo_drags", "sum"),
        "first_downs_by_pen": ("first_downs_by_pen", "sum"),
        "qb_protect_flags": ("qb_protect_flags", "sum"),
        "steam_spread_avg": ("steam_spread", "mean"),
        "steam_total_avg": ("steam_total", "mean"),
        "respect_spread_rate": ("respect_spread", "mean"),
        "respect_total_rate": ("respect_total", "mean"),
        "ats_error_open_avg": ("ats_error_open", "mean"),
        "ats_error_close_avg": ("ats_error_close", "mean"),
        "total_error_open_avg": ("total_error_open", "mean"),
        "total_error_close_avg": ("total_error_close", "mean"),
        "dir_consistent_spread_rate": ("dir_consistent_spread", "mean"),
        "dir_consistent_total_rate": ("dir_consistent_total", "mean"),
    }
    
    # NEW: Add impact metric aggregations
    for col in impact_cols:
        if col in g0.columns:
            if 'avg_' in col:
                base_agg[col] = (col, "mean")
            else:
                base_agg[col] = (col, "sum")
    
    grp = g0.groupby(["ref_name"]).agg(**base_agg).reset_index()
    grp["ref_name"] = grp["ref_name"].astype(str)

    # Derived rates
    grp["ats_decisions"] = grp["ats_w"] + grp["ats_l"]
    grp["ats_pct_raw"] = grp.apply(lambda r: safe_div(r["ats_w"], r["ats_decisions"]), axis=1)
    grp["ou_decisions"] = grp["ou_over"] + grp["ou_under"]
    grp["over_pct_raw"] = grp.apply(lambda r: safe_div(r["ou_over"], r["ou_decisions"]), axis=1)

    grp["pens_per_game"] = (grp["pen_home"] + grp["pen_away"]) / grp["games"].replace(0, np.nan)
    grp["penalty_yards_pg"] = (grp["yards_home"] + grp["yards_away"]) / grp["games"].replace(0, np.nan)

    # EB shrinkage
    p_home_ats = baselines["p_home_ats"]
    p_ou_over = baselines["p_ou_over"]
    grp["ats_pct_adj"] = grp.apply(lambda r: eb_shrink_rate(r["ats_pct_raw"], int(r["ats_decisions"]), p_home_ats), axis=1)
    grp["over_pct_adj"] = grp.apply(lambda r: eb_shrink_rate(r["over_pct_raw"], int(r["ou_decisions"]), p_ou_over), axis=1)

    grp["home_bias_ratio"] = (grp["pen_home"] / grp["pen_away"]).replace({np.inf: np.nan})

    # Penalty type per-game rates
    for t in baselines["top_types"]:
        col_name = f"{t}_per_game"
        if col_name in g0.columns:
            per_game = g0.groupby("ref_name")[col_name].sum() / g0.groupby("ref_name")["game_id"].nunique()
            grp[f"{t}_per_game"] = per_game.reindex(grp["ref_name"]).values
            league_pg = baselines.get("type_per_game", {}).get(t, np.nan)
            if not np.isnan(league_pg):
                grp[f"{t}_delta_vs_lg"] = grp[f"{t}_per_game"] - league_pg
        else:
            grp[f"{t}_per_game"] = 0.0

    # Convert angle sums to per-game
    denom = grp["games"].replace(0, np.nan)
    grp["bailout_3rd_fd_per_game"] = grp["bailout_3rd_fd"] / denom
    grp["free_pass_yards_pg"] = grp["free_pass_yards"] / denom
    grp["drive_killers_pg"] = grp["drive_killers"] / denom
    grp["tempo_drags_pg"] = grp["tempo_drags"] / denom
    grp["fd_by_pen_pg"] = grp["first_downs_by_pen"] / denom
    grp["qb_protect_pg"] = grp["qb_protect_flags"] / denom
    
    # NEW: Convert impact metrics to per-game
    for col in ['total_epa_impact', 'total_wpa_impact', 'high_leverage_penalties',
                'red_zone_penalties', 'fourth_qtr_penalties', 'final_2min_penalties',
                'close_game_penalties', 'third_down_penalties', 'fourth_down_penalties',
                'passing_down_penalties']:
        if col in grp.columns:
            grp[f"{col}_pg"] = grp[col] / denom

    # Z-scores
    eligible = grp[grp["games"] >= MIN_GAMES_TOTAL].copy()
    z_cols = ["pens_per_game", "ats_pct_adj", "over_pct_adj", "pen_cnt_diff_avg", "pen_yds_diff_avg"]
    
    # Add impact metric z-scores
    if "total_epa_impact_pg" in eligible.columns:
        z_cols.append("total_epa_impact_pg")
    if "high_leverage_penalties_pg" in eligible.columns:
        z_cols.append("high_leverage_penalties_pg")
    
    zmap = {}
    for c in z_cols:
        if c in eligible.columns:
            z = zscore(eligible[c].astype(float))
            zmap[c] = pd.Series(0.0, index=grp.index)
            zmap[c].loc[eligible.index] = z.values
            grp[f"z_{c}"] = zmap[c]

    # Spread bucket splits
    def bucketize(sl):
        if pd.isna(sl):
            return None
        for name, lo, hi in SPREAD_BUCKETS:
            if hi <= sl <= 0 or (lo <= sl < hi):
                if name == "-7_or_more" and sl <= -7:
                    return name
                if name == "-3.5_to_-6.5" and -6.5 <= sl < -3.0:
                    return name
                if name == "pk_to_-3" and -3.0 <= sl <= 0:
                    return name
        return None

    tmp = g0.copy()
    tmp["spread_bucket"] = tmp["spread_norm"].apply(bucketize)
    if "home_is_fav" not in tmp.columns:
        tmp["home_is_fav"] = np.where(tmp["spread_norm"].notna(), (tmp["spread_norm"] < 0).astype(int), np.nan)
    tmp = tmp[tmp["home_is_fav"] == 1]
    buck = (
        tmp
        .groupby(["ref_name", "spread_bucket"]).agg(
            games=("game_id", "nunique"),
            ats_w=("home_ats", lambda s: (s == 1).sum()),
            ats_l=("home_ats", lambda s: (s == -1).sum()),
        )
        .reset_index()
    )
    buck["ats_pct"] = buck.apply(lambda r: safe_div(r["ats_w"], (r["ats_w"] + r["ats_l"])) if (r["ats_w"] + r["ats_l"])>0 else 0, axis=1)

    bucket_json = {
        ref: df.drop(columns=["ref_name"]).to_dict(orient="records")
        for ref, df in buck.groupby("ref_name")
    }
    grp["fav_bucket_ats"] = grp["ref_name"].map(bucket_json).apply(lambda x: x if isinstance(x, list) else [])

    # Tiers
    def tier_from_z(z):
        if pd.isna(z):
            return "-"
        if z >= Z_TIER_THRESHOLDS["S"]:
            return "S"
        if z >= Z_TIER_THRESHOLDS["A"]:
            return "A"
        if z >= Z_TIER_THRESHOLDS["B"]:
            return "B"
        if z >= Z_TIER_THRESHOLDS["C"]:
            return "C"
        return "D"

    grp["tier_over"] = grp["z_over_pct_adj"].apply(tier_from_z)
    grp["tier_ats"] = grp["z_ats_pct_adj"].apply(tier_from_z)
    grp["tier_flags"] = grp["z_pens_per_game"].apply(tier_from_z)
    grp["tier_fairness"] = grp["z_pen_yds_diff_avg"].apply(tier_from_z)

    # BBI
    def compose_bbi(row):
        val = 0.0
        for k, w in BBI_WEIGHTS.items():
            val += w * float(row.get(k, 0.0))
        return val
    grp["bbi"] = grp.apply(compose_bbi, axis=1)

    return grp.sort_values("ref_name").reset_index(drop=True), eligible


# ----------------------
# NEW: High-Impact Penalty Leaderboards
# ----------------------

def make_impact_leaderboards(ref_summary: pd.DataFrame, penalty_impacts: pd.DataFrame, officials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create leaderboards for high-impact penalties."""
    lb = {}
    elig = ref_summary[ref_summary["games"] >= MIN_GAMES_TOTAL].copy()
    
    def add_low_n_note(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["note"] = ""
        out.loc[out["games"] < LOW_BUCKET_N, "note"] = "[LOW N]"
        return out
    
    # EPA impact leaders
    if "total_epa_impact_pg" in elig.columns:
        t = elig.sort_values("total_epa_impact_pg", ascending=False)[
            ["ref_name", "games", "total_epa_impact_pg", "avg_epa_per_penalty"]
        ].head(15)
        lb["highest_epa_impact"] = add_low_n_note(t)
    
    # High-leverage penalty leaders
    if "high_leverage_penalties_pg" in elig.columns:
        t = elig.sort_values("high_leverage_penalties_pg", ascending=False)[
            ["ref_name", "games", "high_leverage_penalties_pg"]
        ].head(15)
        lb["most_high_leverage"] = add_low_n_note(t)
    
    # Red zone penalty leaders
    if "red_zone_penalties_pg" in elig.columns:
        t = elig.sort_values("red_zone_penalties_pg", ascending=False)[
            ["ref_name", "games", "red_zone_penalties_pg"]
        ].head(15)
        lb["most_red_zone_flags"] = add_low_n_note(t)
    
    # Fourth quarter penalty leaders
    if "fourth_qtr_penalties_pg" in elig.columns:
        t = elig.sort_values("fourth_qtr_penalties_pg", ascending=False)[
            ["ref_name", "games", "fourth_qtr_penalties_pg"]
        ].head(15)
        lb["most_4th_qtr_flags"] = add_low_n_note(t)
    
    # Most impactful individual penalties (from penalty_impacts data)
    refs = officials[officials["off_pos"] == "R"][["game_id", "name"]].rename(columns={"name": "ref_name"})
    pen_with_ref = penalty_impacts.merge(refs, on="game_id", how="left")
    
    if "epa_abs" in pen_with_ref.columns:
        # Top 20 most impactful individual penalties
        top_impact = pen_with_ref.nlargest(20, "epa_abs")[[
            "ref_name", "game_id", "penalty_type", "penalty_yards",
            "down_distance", "epa_abs", "wpa_abs", "is_high_leverage"
        ]].copy()
        top_impact = top_impact.rename(columns={
            "epa_abs": "epa_impact",
            "wpa_abs": "wpa_impact"
        })
        lb["top_individual_penalties"] = top_impact
    
    return lb


# ----------------------
# Leaderboards (ENHANCED)
# ----------------------

def make_leaderboards(ref_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lb = {}
    elig = ref_summary[ref_summary["games"] >= MIN_GAMES_TOTAL].copy()

    def add_low_n_note(df: pd.DataFrame, decisions_col: str | None = None) -> pd.DataFrame:
        out = df.copy()
        out["note"] = ""
        crit = out["games"] < LOW_BUCKET_N
        if decisions_col and decisions_col in out.columns:
            crit = crit | (out[decisions_col] < LOW_BUCKET_N)
        out.loc[crit, "note"] = "[LOW N]"
        return out

    # Flaggiest
    t = elig.sort_values("pens_per_game", ascending=False)[
        ["ref_name", "games", "pens_per_game", "z_pens_per_game"]
    ].head(15)
    lb["flaggiest"] = add_low_n_note(t)

    # Over / Under
    t = elig.assign(ou_decisions=elig["ou_over"] + elig["ou_under"]) \
            .sort_values("over_pct_adj", ascending=False)[
        ["ref_name", "games", "ou_decisions", "over_pct_adj", "z_over_pct_adj"]
    ].head(15)
    lb["over_crews"] = add_low_n_note(t, decisions_col="ou_decisions")

    t = elig.assign(ou_decisions=elig["ou_over"] + elig["ou_under"]) \
            .sort_values("over_pct_adj")[
        ["ref_name", "games", "ou_decisions", "over_pct_adj", "z_over_pct_adj"]
    ].head(15)
    lb["under_crews"] = add_low_n_note(t, decisions_col="ou_decisions")

    # Fairness
    t = elig.sort_values("pen_cnt_diff_avg", ascending=False)[
        ["ref_name", "games", "pen_cnt_diff_avg", "pen_yds_diff_avg", "z_pen_cnt_diff_avg"]
    ].head(15)
    lb["home_fairness_against"] = add_low_n_note(t)

    t = elig.sort_values("pen_cnt_diff_avg")[
        ["ref_name", "games", "pen_cnt_diff_avg", "pen_yds_diff_avg", "z_pen_cnt_diff_avg"]
    ].head(15)
    lb["road_fairness_against"] = add_low_n_note(t)

    # ATS kings / fades
    t = elig.assign(ats_decisions=elig["ats_w"] + elig["ats_l"]) \
            .sort_values("ats_pct_adj", ascending=False)[
        ["ref_name", "games", "ats_decisions", "ats_pct_adj", "z_ats_pct_adj"]
    ].head(15)
    lb["ats_kings"] = add_low_n_note(t, decisions_col="ats_decisions")

    t = elig.assign(ats_decisions=elig["ats_w"] + elig["ats_l"]) \
            .sort_values("ats_pct_adj")[
        ["ref_name", "games", "ats_decisions", "ats_pct_adj", "z_ats_pct_adj"]
    ].head(15)
    lb["ats_fades"] = add_low_n_note(t, decisions_col="ats_decisions")

    # Market move / respect leaderboards
    if "steam_total_avg" in elig.columns:
        t = elig.sort_values("steam_total_avg", ascending=False)[["ref_name","games","steam_total_avg"]].head(15)
        lb["steam_total_up"] = add_low_n_note(t)
        t = elig.sort_values("steam_total_avg")[["ref_name","games","steam_total_avg"]].head(15)
        lb["steam_total_down"] = add_low_n_note(t)
    if "steam_spread_avg" in elig.columns:
        t = elig.sort_values("steam_spread_avg", ascending=False)[["ref_name","games","steam_spread_avg"]].head(15)
        lb["steam_spread_toward_home"] = add_low_n_note(t)
        t = elig.sort_values("steam_spread_avg")[["ref_name","games","steam_spread_avg"]].head(15)
        lb["steam_spread_toward_away"] = add_low_n_note(t)
    if "respect_total_rate" in elig.columns:
        t = elig.sort_values("respect_total_rate", ascending=False)[["ref_name","games","respect_total_rate"]].head(15)
        lb["market_respect_total"] = add_low_n_note(t)
    if "respect_spread_rate" in elig.columns:
        t = elig.sort_values("respect_spread_rate", ascending=False)[["ref_name","games","respect_spread_rate"]].head(15)
        lb["market_respect_spread"] = add_low_n_note(t)

    # Flag Files angle boards
    if "bailout_3rd_fd_per_game" in elig.columns:
        t = elig.sort_values("bailout_3rd_fd_per_game", ascending=False)[["ref_name","games","bailout_3rd_fd_per_game"]].head(15)
        lb["third_down_bailouts"] = add_low_n_note(t)
    if "free_pass_yards_pg" in elig.columns:
        t = elig.sort_values("free_pass_yards_pg", ascending=False)[["ref_name","games","free_pass_yards_pg"]].head(15)
        lb["free_yards_crews"] = add_low_n_note(t)
    if "qb_protect_pg" in elig.columns:
        t = elig.sort_values("qb_protect_pg", ascending=False)[["ref_name","games","qb_protect_pg"]].head(15)
        lb["qb_sanctuary"] = add_low_n_note(t)

    return lb


# ----------------------
# Extra analytics
# ----------------------

def build_global_sanity(ref_games: pd.DataFrame) -> pd.DataFrame:
    df = ref_games.copy()
    fav = df[df["home_is_fav"] == 1]
    dog = df[df["home_is_fav"] == 0]
    def pct(series):
        return safe_div((series == 1).sum(), ((series == 1) | (series == -1)).sum())
    rows = []
    rows.append({"metric": "Home favorites ATS%", "value": pct(fav["home_ats"])})
    rows.append({"metric": "Home dogs ATS%", "value": pct(dog["home_ats"])})
    rows.append({"metric": "League Over%", "value": safe_div((df["ou_result"] == 1).sum(), ((df["ou_result"] == 1) | (df["ou_result"] == -1)).sum())})
    return pd.DataFrame(rows)


def compute_prop_leaderboards(ref_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lbs = {}
    dpi_cols = [c for c in ref_summary.columns if c.startswith("Defensive Pass Interference_per_game")]
    hold_cols = [c for c in ref_summary.columns if c.startswith("Offensive Holding_per_game")]
    if dpi_cols:
        c = dpi_cols[0]
        lbs["dpi_crews"] = ref_summary.sort_values(c, ascending=False)[["ref_name", "games", c]].head(15)
    if hold_cols:
        c = hold_cols[0]
        lbs["holding_crews"] = ref_summary.sort_values(c, ascending=False)[["ref_name", "games", c]].head(15)
    return lbs


def generate_narratives(ref_summary: pd.DataFrame, baselines: dict) -> dict:
    lines_over, lines_home, lines_road = [], [], []
    base_over = baselines.get("p_ou_over", 0.5)
    for _, r in ref_summary.iterrows():
        if r.get("games", 0) < MIN_GAMES_TOTAL:
            continue
        ref = r["ref_name"]
        overp = float(r.get("over_pct_adj", np.nan))
        atsp = float(r.get("ats_pct_adj", np.nan))
        ppg = float(r.get("pens_per_game", np.nan))
        bbi = float(r.get("bbi", 0.0))
        if not np.isnan(overp):
            if overp >= base_over + 0.05:
                lines_over.append({"ref": ref, "hook": f"{ref} games run hot: Over {overp*100:.1f}% (n={int(r['games'])}).", "bbi": round(bbi, 2)})
            elif overp <= base_over - 0.05:
                lines_over.append({"ref": ref, "hook": f"{ref} buries totals: Over {overp*100:.1f}% (n={int(r['games'])}).", "bbi": round(bbi, 2)})
        if not np.isnan(atsp):
            if atsp >= 0.58:
                lines_home.append({"ref": ref, "hook": f"Home edge with {ref}: home ATS {atsp*100:.1f}%.", "bbi": round(bbi, 2)})
            elif atsp <= 0.42:
                lines_home.append({"ref": ref, "hook": f"Fade home sides with {ref}: home ATS {atsp*100:.1f}%.", "bbi": round(bbi, 2)})
        fd = float(r.get("pen_yds_diff_avg", 0.0))
        if fd >= 5:
            lines_road.append({"ref": ref, "hook": f"Road teams eat flags with {ref}: +{fd:.1f} yds per game.", "bbi": round(bbi, 2)})
        elif fd <= -5:
            lines_road.append({"ref": ref, "hook": f"Home teams eat flags with {ref}: {fd:.1f} yds per game.", "bbi": round(bbi, 2)})
    return {
        "hooks_over": lines_over,
        "hooks_home": lines_home,
        "hooks_road": lines_road,
    }


def export_public_csv(ref_summary: pd.DataFrame):
    cols = [
        "ref_name",
        "games",
        "pens_per_game",
        "ats_pct_adj",
        "over_pct_adj",
        "pen_cnt_diff_avg",
        "pen_yds_diff_avg",
        "home_bias_ratio",
        "bailout_3rd_fd_per_game",
        "free_pass_yards_pg",
        "drive_killers_pg",
        "tempo_drags_pg",
        "fd_by_pen_pg",
        "qb_protect_pg",
    ]
    
    # Add impact columns
    impact_cols = [
        "total_epa_impact_pg", "total_wpa_impact_pg", "avg_epa_per_penalty", "avg_wpa_per_penalty",
        "high_leverage_penalties_pg", "red_zone_penalties_pg", "fourth_qtr_penalties_pg",
        "final_2min_penalties_pg", "close_game_penalties_pg"
    ]
    for c in impact_cols:
        if c in ref_summary.columns:
            cols.append(c)
    
    market_cols = [
        "steam_spread_avg","steam_total_avg","respect_spread_rate","respect_total_rate",
        "ats_error_open_avg","ats_error_close_avg","total_error_open_avg","total_error_close_avg",
        "dir_consistent_spread_rate","dir_consistent_total_rate"
    ]
    for c in market_cols:
        if c in ref_summary.columns:
            cols.append(c)

    out = ref_summary.loc[:, [c for c in cols if c in ref_summary.columns]].copy()
    out.to_csv(OUTDIR / "ref_cards_public.csv", index=False)


def plot_ref_trends(ref_games: pd.DataFrame, baselines: dict, top_refs: list[str], ref_summary: pd.DataFrame | None = None, max_games: int = 10):
    for ref in top_refs:
        df = ref_games[ref_games["ref_name"] == ref].copy()
        if df.empty:
            continue
        df = df.sort_values(["season", "week"]).tail(max_games)
        league_avg_flags = float(baselines.get("penalties_per_game", 0.0))
        x_vals = [f"Wk {int(w)}" for w in df["week"].tolist()]
        df["flags_total"] = df["pen_count_home"].fillna(0) + df["pen_count_away"].fillna(0)

        plt.figure()
        xs = list(range(len(df)))
        plt.plot(xs, df["flags_total"].values, label="flags")
        if league_avg_flags > 0:
            plt.axhline(y=league_avg_flags, linestyle="--")
        over_mask = df["ou_result"] == 1
        under_mask = df["ou_result"] == -1
        plt.scatter([x for x,m in zip(xs, over_mask) if m], df.loc[over_mask, "flags_total"], marker='o', label='Over')
        plt.scatter([x for x,m in zip(xs, under_mask) if m], df.loc[under_mask, "flags_total"], marker='x', label='Under')
        cov_mask = df["home_ats"] == 1
        noc_mask = df["home_ats"] == -1
        plt.scatter([x for x,m in zip(xs, cov_mask) if m], df.loc[cov_mask, "flags_total"], marker='^', label='Home cover')
        plt.scatter([x for x,m in zip(xs, noc_mask) if m], df.loc[noc_mask, "flags_total"], marker='v', label='Home no-cover')

        plt.title(f"{ref} flags last {len(df)} games")
        plt.xlabel("week")
        plt.xticks(ticks=xs, labels=x_vals, rotation=0)
        plt.ylabel("flags")
        plt.text(0.99, 0.02, f"Flag Files | last {len(df)} | n={len(df)}", ha='right', va='bottom', transform=plt.gca().transAxes)
        if ref_summary is not None and "ref_name" in ref_summary.columns and "crew_compact" in ref_summary.columns:
            crew_str = ref_summary.loc[ref_summary["ref_name"] == ref, "crew_compact"].dropna()
            if not crew_str.empty and crew_str.iloc[0]:
                plt.gcf().text(0.01, 0.01, f"Crew: {crew_str.iloc[0]}", ha='left', va='bottom')
        plt.legend(loc="best")
        fn = PLOTS_DIR / f"{ref.replace(' ', '_')}_flags_last{len(df)}.png"
        plt.savefig(fn, bbox_inches="tight")
        plt.close()

# ----------------------
# Ref card image rendering
# ----------------------
def _draw_ring_gauge(draw: "ImageDraw.ImageDraw", cx: int, cy: int, r: int, pct: float, title: str, subtitle: str):
    """Draw a circular gauge showing pct (0..1)."""
    if draw is None:
        return
    pct = max(0.0, min(1.0, float(pct if not pd.isna(pct) else 0.0)))
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.arc(bbox, start=0, end=360, fill=MUTED, width=18)  # background ring
    sweep = 360 * pct
    draw.arc(bbox, start=-90, end=-90 + sweep, fill=ACCENT_TEAL, width=22)  # value arc
    f_big = _load_font(64); f_small = _load_font(28)
    txt = f"{int(round(pct * 100))}%"
    w, h = _measure_text(draw, txt, f_big) if f_big else (0, 0)
    draw.text((cx - w // 2, cy - h // 2), txt, fill=WHITE, font=f_big)
    tw, th = _measure_text(draw, title, f_small) if f_small else (0, 0)
    draw.text((cx - tw // 2, cy + r + 8), title, fill=WHITE, font=f_small)
    if subtitle:
        sw, sh = _measure_text(draw, subtitle, f_small) if f_small else (0, 0)
        draw.text((cx - sw // 2, cy + r + 8 + th + 2), subtitle, fill=MUTED, font=f_small)

def _wrap_lines(text: str, width: int, font) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines, cur = [], []
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10))) if Image else None
    for w in words:
        test = " ".join(cur + [w])
        tw = draw.textlength(test, font=font) if draw else len(test) * 7
        if tw <= width:
            cur.append(w)
        else:
            lines.append(" ".join(cur)); cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines

def find_assignment_for_ref(ref_name: str, ref_games: pd.DataFrame, current_week: int | None):
    """Return the most recent assignment for the ref, preferring current season and current week."""
    df = ref_games[ref_games["ref_name"] == ref_name].copy()
    if df.empty:
        return None
    # Constrain to current season first (avoid matching Week N from prior years)
    cur_season = globals().get("_current_season", None)
    if cur_season is None and "season" in df.columns and df["season"].notna().any():
        cur_season = int(pd.to_numeric(df["season"], errors="coerce").max())
    if cur_season is not None and "season" in df.columns:
        df = df[pd.to_numeric(df["season"], errors="coerce") == int(cur_season)]
        if df.empty:
            return None
    # Prefer an explicit current-week match within the current season
    if current_week is not None and "week" in df.columns:
        cur = df[pd.to_numeric(df["week"], errors="coerce") == int(current_week)]
        if not cur.empty:
            return cur.sort_values(["season", "week"]).iloc[-1]
    # Fallback: most recent game in current season
    return df.sort_values(["season", "week"]).iloc[-1]

def render_ref_card_image(ref_row: pd.Series, assignment_row: pd.Series | None, pen_df: pd.DataFrame, current_week: int | None = None):
    """Render one PNG card for a referee."""
    if Image is None:
        return None
    img = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    f_title = _load_font(120); f_label = _load_font(44)
    f_body = _load_font(36);  f_small = _load_font(28)

    # Header
    name = str(ref_row.get("ref_name", "Unknown")).strip()
    draw.text((60, 40), name, fill=WHITE, font=f_title)

    # Assignment
    left_y = 200
    show_matchup = False
    if assignment_row is not None and current_week is not None:
        try:
            asn_wk = assignment_row.get("week", None)
            asn_season = assignment_row.get("season", globals().get("_current_season"))
            cur_season = globals().get("_current_season")
            season_ok = (cur_season is None) or (asn_season is None) or (int(asn_season) == int(cur_season))
            if asn_wk is not None and not pd.isna(asn_wk) and int(asn_wk) == int(current_week) and season_ok:
                show_matchup = True
        except Exception:
            show_matchup = False

    if show_matchup:
        home = assignment_row.get("home_team", "")
        away = assignment_row.get("away_team", "")
        draw.text((60, left_y), f"{away} @ {home}", fill=WHITE, font=f_label)
        draw.text((60, left_y + 56), f"Week {int(current_week)}", fill=MUTED, font=f_body)

    # Gauges
    gx = 160
    _draw_ring_gauge(draw, gx, 470, 110, ref_row.get("ats_pct_adj", np.nan), "Home ATS", "")
    _draw_ring_gauge(draw, gx + 340, 470, 110, ref_row.get("over_pct_adj", np.nan), "Over Rate", "")

    # Metrics
    y0 = 660
    lines = [
        f"Flags/G: {float(ref_row.get('pens_per_game', float('nan'))):.2f}",
        f"Bias yds (away-home): {float(ref_row.get('pen_yds_diff_avg', 0)):.1f}",
    ]
    if "total_epa_impact_pg" in ref_row and not pd.isna(ref_row["total_epa_impact_pg"]):
        lines.append(f"EPA impact/G: {float(ref_row['total_epa_impact_pg']):.2f}")
    if "high_leverage_penalties_pg" in ref_row and not pd.isna(ref_row["high_leverage_penalties_pg"]):
        lines.append(f"High-leverage/G: {float(ref_row['high_leverage_penalties_pg']):.2f}")
    y = y0
    for t in lines:
        draw.text((60, y), t, fill=WHITE, font=f_body); y += 44

    # Impactful calls (for assigned game if available)
    calls_y = y0
    draw.text((600, calls_y - 10), "Impactful calls", fill=WHITE, font=f_label)
    calls = []
    if assignment_row is not None and "game_id" in assignment_row and not pd.isna(assignment_row["game_id"]):
        gid = assignment_row["game_id"]
        refs = pen_df.copy()
        if "game_id" in refs.columns:
            refs = refs[refs["game_id"] == gid]
        if "ref_name" in refs.columns:
            refs = refs[refs["ref_name"] == name]
        if "epa_abs" in refs.columns:
            refs = refs.sort_values("epa_abs", ascending=False).head(3)
            for _, rr in refs.iterrows():
                typ = str(rr.get("penalty_type", "Penalty"))
                epa = rr.get("epa_abs", np.nan)
                wpa = rr.get("wpa_abs", np.nan)
                dd = str(rr.get("down_distance", "")) if "down_distance" in refs.columns else ""
                calls.append(f"- {typ} ({dd}) | EPA {epa:.2f}" + (f", WPA {wpa:.2f}" if not pd.isna(wpa) else ""))
    if not calls:
        calls = ["- No high-impact calls recorded for this assignment."]
    yy = calls_y + 46
    for line in calls:
        for seg in _wrap_lines(line, 440, f_small):
            draw.text((600, yy), seg, fill=MUTED, font=f_small); yy += 32

    return img
def export_ref_card_images(
    ref_summary: pd.DataFrame,
    ref_games: pd.DataFrame,
    penalty_impacts: pd.DataFrame,
    current_week: int | None,
):
    """Generate a per-ref PNG card for the current week (or latest assignment)."""
    if Image is None:
        print("PIL not available; skipping ref card images.")
        return

    outdir = OUTDIR / "ref_cards_images"
    outdir.mkdir(parents=True, exist_ok=True)

    elig = ref_summary[ref_summary.get("games", 0) >= MIN_GAMES_TOTAL].copy()

    # Ensure penalties have ref_name
    pen_df = penalty_impacts.copy()
    if "ref_name" not in pen_df.columns:
        try:
            officials = nfl.import_officials(load_years())
            rmap = (
                officials[officials["off_pos"] == "R"][["game_id", "name"]]
                .rename(columns={"name": "ref_name"})
            )
            pen_df = pen_df.merge(rmap, on="game_id", how="left")
        except Exception:
            pen_df["ref_name"] = None

    for _, row in elig.iterrows():
        # Prefer direct FBZ assignment row for the current week if available
        assign = None
        if current_week is not None and "_fbz_assignments" in globals():
            fbz = globals()["_fbz_assignments"]
            if isinstance(fbz, pd.DataFrame) and not fbz.empty:
                cand = fbz[
                    (fbz["ref_name"].astype(str).str.strip() == str(row["ref_name"]).strip())
                    & (fbz["week"].astype(int) == int(current_week))
                ]
                if not cand.empty:
                    top = cand.iloc[0]
                    assign = pd.Series({
                        "week": int(current_week),
                        "season": globals().get("_current_season"),
                        "home_team": top.get("home_team", ""),
                        "away_team": top.get("away_team", ""),
                        "game_id": np.nan  # unknown pregame
                    })

        if assign is None:
            assign = find_assignment_for_ref(row["ref_name"], ref_games, current_week)

        img = render_ref_card_image(row, assign, pen_df, current_week=current_week)
        if img is None:
            continue

        if current_week is not None:
            wk = int(current_week)
        else:
            wk = (
                int(assign.get("week", 0))
                if assign is not None and not pd.isna(assign.get("week", np.nan))
                else 0
            )

        safe_name = str(row["ref_name"]).replace(" ", "_")
        fn = outdir / f"{safe_name}_Wk{wk if wk else 'NA'}.png"
        img.save(fn)

# ----------------------
# Content exports
# ----------------------

def export_csv_json(ref_summary: pd.DataFrame, leaderboards: dict):
    ref_csv = OUTDIR / "ref_cards.csv"
    lb_csv = OUTDIR / "ref_leaderboards.csv"
    copy_json = OUTDIR / "ref_copy.json"

    ref_summary.to_csv(ref_csv, index=False)

    all_lb = []
    for name, df in leaderboards.items():
        tmp = df.copy()
        tmp.insert(0, "leaderboard", name)
        all_lb.append(tmp)
    if all_lb:
        pd.concat(all_lb, ignore_index=True).to_csv(lb_csv, index=False)

    lines = []
    for _, r in ref_summary.sort_values("games", ascending=False).iterrows():
        if r["games"] < MIN_GAMES_TOTAL:
            continue
        ref = r["ref_name"]
        ats = round(100 * r["ats_pct_adj"], 1)
        overp = round(100 * r["over_pct_adj"], 1)
        ppg = round(r["pens_per_game"], 2)
        fairness = round(r["pen_cnt_diff_avg"], 2)
        
        line_dict = {
            "ref": ref,
            "headline": f"{ref}: home ATS {ats}% | Over {overp}% | {ppg} flags/gm | diff {fairness} pens (away−home)",
            "sample": int(r["games"]),
            "disclaimer": f"EB-shrunk toward league base; pushes removed; n={int(r['games'])}",
            "fav_bucket_ats": r.get("fav_bucket_ats", []),
        }
        
        # Add impact metrics if available
        if "total_epa_impact_pg" in r and not pd.isna(r["total_epa_impact_pg"]):
            line_dict["epa_impact_per_game"] = round(r["total_epa_impact_pg"], 2)
        if "high_leverage_penalties_pg" in r and not pd.isna(r["high_leverage_penalties_pg"]):
            line_dict["high_leverage_per_game"] = round(r["high_leverage_penalties_pg"], 2)
        
        lines.append(line_dict)

    crews_payload = None
    try:
        _crews_df = globals().get("_crews_df", None)
        if _crews_df is not None and not _crews_df.empty:
            crews_payload = {
                row["Referee"]: {
                    "members": row["crew_members"],
                    "compact": row["crew_compact"],
                }
                for _, row in _crews_df.iterrows()
            }
    except Exception:
        crews_payload = None

    with open(copy_json, "w") as f:
        payload = {"bias_bombs": lines}
        if crews_payload:
            payload["crews_2025"] = crews_payload
        json.dump(payload, f, indent=2)


def export_markdown(ref_summary: pd.DataFrame, baselines: dict, leaderboards: dict):
    md_path = OUTDIR / "referee_report_cards.md"

    lines = []
    lines.append(f"# NFL Referee Report Cards (Enhanced with EPA/WPA Impact Analysis)\n")
    lines.append(f"Blended={USE_MULTIYEAR}, years={YEARS_BACK}\n")
    lines.append(f"League baselines: Home ATS ~ {baselines['p_home_ats']:.3f}, Over ~ {baselines['p_ou_over']:.3f}, Avg flags/gm ~ {baselines['penalties_per_game']:.2f}.\n")
    lines.append("Method: ATS is evaluated from the home team perspective; pushes removed. EB shrinkage applied toward league averages.\n")
    lines.append("NEW: EPA/WPA impact analysis shows penalty influence on game outcomes.\n")

    sanity = build_global_sanity(globals().get("_ref_games_for_md", pd.DataFrame())) if "_ref_games_for_md" in globals() else None
    if sanity is not None and not sanity.empty:
        lines.append("\n## Global sanity checks\n")
        try:
            sanity_md = sanity.to_markdown(index=False)
            sanity_lines = sanity_md.split('\n')
            clean_lines = []
            for line in sanity_lines:
                if not (line.strip() and '|' in line and line.count('|') >= 2 and 
                       all(part.strip().replace('.', '').replace('-', '').isdigit() or part.strip() == '' 
                           for part in line.split('|')[1:-1] if part.strip())):
                    clean_lines.append(line)
            lines.append('\n'.join(clean_lines))
        except Exception:
            for _, row in sanity.iterrows():
                lines.append(f"- {row['metric']}: {row['value']:.3f}")
        lines.append("_Crew rosters compiled from public reporting; roles may rotate. Some entries may be pending confirmation._\n")
        lines.append("\n")

    try:
        _crews_df = globals().get("_crews_df", None)
        if _crews_df is not None and not _crews_df.empty:
            lines.append("\n## 2025 Crew Rosters\n")
            crew_tbl = _crews_df[["Referee","crew_compact"]].copy()
            lines.append(crew_tbl.to_markdown(index=False))
            lines.append("\n")
    except Exception:
        pass

    for name, df in leaderboards.items():
        lines.append(f"\n## Leaderboard: {name}\n")
        lines.append(df.to_markdown(index=False))
        lines.append("\n")

    for _, r in ref_summary.sort_values(["games", "ref_name"], ascending=[False, True]).iterrows():
        lines.append(f"\n---\n# Referee: {r['ref_name']}\n")
        if int(r['games']) < MIN_GAMES_TOTAL:
            lines.append("**LOW SAMPLE** — stats are noisy; EB-shrunk to league.")
        lines.append(f"Seasons: {', '.join(map(str, r['seasons']))} | Games: {int(r['games'])}\n")
        lines.append("Note: bucket stats exclude pushes/missing lines; EB applied to top-line rates.\n")
        lines.append(f"Home ATS: {int(r['ats_w'])}-{int(r['ats_l'])}-{int(r['ats_p'])} (adj {100*r['ats_pct_adj']:.1f}%)\n")
        lines.append(f"Totals O/U: {int(r['ou_over'])}-{int(r['ou_under'])}-{int(r['ou_push'])} (adj Over {100*r['over_pct_adj']:.1f}%)\n")
        lines.append(f"Flags/game: {r['pens_per_game']:.2f} | Fairness (away-home): cnt {r['pen_cnt_diff_avg']:.2f}, yds {r['pen_yds_diff_avg']:.2f}\n")
        
        # NEW: Impact metrics
        if "total_epa_impact_pg" in r and not pd.isna(r["total_epa_impact_pg"]):
            lines.append(f"EPA Impact/Game: {r['total_epa_impact_pg']:.2f} | Avg EPA/Penalty: {r.get('avg_epa_per_penalty', 0):.2f}\n")
        if "high_leverage_penalties_pg" in r and not pd.isna(r["high_leverage_penalties_pg"]):
            lines.append(f"High-Leverage Penalties/Game: {r['high_leverage_penalties_pg']:.2f}\n")
        if "red_zone_penalties_pg" in r and not pd.isna(r["red_zone_penalties_pg"]):
            lines.append(f"Red Zone Flags/Game: {r['red_zone_penalties_pg']:.2f} | 4th Qtr Flags/Game: {r.get('fourth_qtr_penalties_pg', 0):.2f}\n")
        
        try:
            lines.append(f"Angles: 3rd-down bailouts {r['bailout_3rd_fd_per_game']:.2f}/gm | Free pass yds {r['free_pass_yards_pg']:.1f}/gm | Drive-killers {r['drive_killers_pg']:.2f}/gm | Tempo drags {r['tempo_drags_pg']:.2f}/gm | FD by pen {r['fd_by_pen_pg']:.2f}/gm | QB protect {r['qb_protect_pg']:.2f}/gm\n")
        except Exception:
            pass
        
        crew_compact = r.get("crew_compact", None)
        if isinstance(crew_compact, str) and crew_compact.strip():
            lines.append(f"Crew: {crew_compact}\n")
        
        if r.get("fav_bucket_ats"):
            lines.append("Home favorite spread buckets (ATS):")
            for item in r["fav_bucket_ats"]:
                sb = item.get("spread_bucket")
                gm = int(item.get("games", 0))
                w = int(item.get("ats_w", 0))
                l = int(item.get("ats_l", 0))
                if w + l > 0:
                    pct = 100.0 * w / (w + l)
                else:
                    pct = 0.0
                low_tag = " [LOW N]" if gm < LOW_BUCKET_N else ""
                lines.append(f"- {sb}: {w}-{l} (n={gm}) ATS {pct:.1f}%{low_tag}")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


# ----------------------
# Content-pack generators
# ----------------------

def _fmt_pct(x):
    try:
        return f"{100*float(x):.1f}%"
    except Exception:
        return "-"

def select_top_refs_for_content(ref_summary: pd.DataFrame, k_teaser: int = 1, k_reminder: int = 3):
    elig = ref_summary[ref_summary["games"] >= MIN_GAMES_TOTAL].copy()
    if elig.empty:
        return [], []
    teaser = elig.sort_values(["bbi", "over_pct_adj"], ascending=[False, False]).head(k_teaser)
    pool = []
    pool.append(elig.sort_values("bbi", ascending=False).head(k_reminder * 2))
    pool.append(elig.sort_values("over_pct_adj", ascending=False).head(k_reminder * 2))
    pool.append(elig.sort_values("ats_pct_adj", ascending=False).head(k_reminder * 2))
    pool_df = pd.concat(pool).drop_duplicates(subset=["ref_name"])
    reminder = pool_df.head(k_reminder)
    return teaser, reminder

def export_flag_files_free_teaser(ref_summary: pd.DataFrame, baselines: dict):
    teaser, _ = select_top_refs_for_content(ref_summary, k_teaser=1, k_reminder=3)
    out = []
    wk = globals().get("_current_week")
    hdr = f"# Flag Files: Week {wk} — Free Bias Bomb\n" if wk else "# Flag Files: Week — Free Bias Bomb\n"
    out.append(hdr)
    if teaser.empty:
        out.append("No eligible referees yet. Collect more games.")
    else:
        r = teaser.iloc[0]
        n_ou = int(r.get("ou_decisions", 0))
        n_ats = int(r.get("ats_decisions", 0))
        out.append(f"**{r['ref_name']}**\n")
        out.append(f"• Flags/Gm: {r['pens_per_game']:.2f} (lg avg {baselines['penalties_per_game']:.2f})")
        out.append(f"\n• Over rate: {_fmt_pct(r['over_pct_adj'])} (n={n_ou})")
        out.append(f"\n• Home ATS: {_fmt_pct(r['ats_pct_adj'])} (n={n_ats})")
        
        # NEW: Add impact metrics
        if "total_epa_impact_pg" in r and not pd.isna(r["total_epa_impact_pg"]):
            out.append(f"\n• EPA Impact/Game: {r['total_epa_impact_pg']:.2f}")
        if "high_leverage_penalties_pg" in r and not pd.isna(r["high_leverage_penalties_pg"]):
            out.append(f"\n• High-Leverage Flags/Game: {r['high_leverage_penalties_pg']:.2f}")
        
        crew_compact = r.get("crew_compact", "")
        if isinstance(crew_compact, str) and crew_compact.strip():
            out.append(f"\n• Crew: {crew_compact}")
        out.append("\n\n**Translation:** If you're betting this week, account for this ref's tilt.\n")
        out.append("\nUnlock the full Flag Files slate for every crew and every bias before Sunday.")
    (OUTDIR / "flag_files_free_teaser.md").write_text("\n".join(out))

def export_flag_files_paid_full(ref_summary: pd.DataFrame, baselines: dict, leaderboards: dict):
    lines = []
    wk = globals().get("_current_week")
    title = f"# Flag Files: NFL — Week {wk} Full Referee Report Cards (Enhanced)\n" if wk else "# Flag Files: NFL — Full Referee Report Cards (Enhanced)\n"
    lines.append(title)
    lines.append(
        f"**League baselines** — Home ATS: {_fmt_pct(baselines['p_home_ats'])}, "
        f"Over: {_fmt_pct(baselines['p_ou_over'])}, Flags/Gm: {baselines['penalties_per_game']:.2f}\n"
    )
    lines.append("Method: ATS is from the HOME perspective; pushes removed. `spread_norm` is defined so negative = home favorite. EB shrinkage applied toward league averages.\n")
    lines.append("NEW: EPA/WPA impact analysis quantifies penalty influence on game outcomes.\n")
    lines.append("Note on buckets: entries marked [LOW N] indicate small samples (wide binomial 95% CI). Treat as directional.\n")

    try:
        gsan = build_global_sanity(globals().get("_ref_games_for_md", pd.DataFrame()))
        if gsan is not None and not gsan.empty:
            lines.append("\n**Window sanity:** " + "; ".join([f"{m} {_fmt_pct(v)}" for m,v in zip(gsan["metric"], gsan["value"])]) + "\n")
    except Exception:
        pass

    def _md_table(df, cols):
        if df is None or df.empty:
            return "_No data._\n"
        base_cols = [c for c in cols if c in df.columns]
        if "note" in df.columns:
            base_cols.append("note")
        show = df.loc[:, base_cols].copy()
        return show.to_markdown(index=False) + "\n"

    sections = [
        ("flaggiest", ["ref_name", "games", "pens_per_game"]),
        ("over_crews", ["ref_name", "games", "ou_decisions", "over_pct_adj"]),
        ("under_crews", ["ref_name", "games", "ou_decisions", "over_pct_adj"]),
        ("ats_kings", ["ref_name", "games", "ats_decisions", "ats_pct_adj"]),
        ("ats_fades", ["ref_name", "games", "ats_decisions", "ats_pct_adj"]),
        ("home_fairness_against", ["ref_name", "games", "pen_cnt_diff_avg", "pen_yds_diff_avg"]),
        ("road_fairness_against", ["ref_name", "games", "pen_cnt_diff_avg", "pen_yds_diff_avg"]),
        ("third_down_bailouts", ["ref_name", "games", "bailout_3rd_fd_per_game"]),
        ("free_yards_crews", ["ref_name", "games", "free_pass_yards_pg"]),
        ("qb_sanctuary", ["ref_name", "games", "qb_protect_pg"]),
        # NEW: Impact leaderboards
        ("highest_epa_impact", ["ref_name", "games", "total_epa_impact_pg", "avg_epa_per_penalty"]),
        ("most_high_leverage", ["ref_name", "games", "high_leverage_penalties_pg"]),
        ("most_red_zone_flags", ["ref_name", "games", "red_zone_penalties_pg"]),
        ("most_4th_qtr_flags", ["ref_name", "games", "fourth_qtr_penalties_pg"]),
    ]
    
    for name, cols in sections:
        if name in leaderboards:
            title = name.replace("_", " ").title()
            lines.append(f"\n## {title}\n")
            lines.append(_md_table(leaderboards[name], cols))

    # Market Moves & Respect (render once after the section loop)
    if any(k in leaderboards for k in ["steam_total_up", "steam_total_down", "market_respect_total", "market_respect_spread"]):
        lines.append("\n---\n## Market Moves & Respect\n")
        if "steam_total_up" in leaderboards:
            lines.append("**Totals steam up (avg close−open)**\n")
            lines.append(_md_table(leaderboards["steam_total_up"], ["ref_name", "games", "steam_total_avg"]))
        if "steam_total_down" in leaderboards:
            lines.append("**Totals steam down (avg close−open)**\n")
            lines.append(_md_table(leaderboards["steam_total_down"], ["ref_name", "games", "steam_total_avg"]))
        if "market_respect_total" in leaderboards:
            lines.append("**Market respect rate (totals)**\n")
            lines.append(_md_table(leaderboards["market_respect_total"], ["ref_name", "games", "respect_total_rate"]))
        if "market_respect_spread" in leaderboards:
            lines.append("**Market respect rate (spreads)**\n")
            lines.append(_md_table(leaderboards["market_respect_spread"], ["ref_name", "games", "respect_spread_rate"]))

    # NEW: Top individual penalty impacts section (also once)
    if "top_individual_penalties" in leaderboards:
        lines.append("\n---\n## Top 20 Most Impactful Individual Penalties\n")
        lines.append("These individual penalty calls had the highest EPA impact on game outcomes.\n")
        lines.append(_md_table(
            leaderboards["top_individual_penalties"],
            ["ref_name", "penalty_type", "penalty_yards", "down_distance", "epa_impact", "wpa_impact", "is_high_leverage"]
        ))

    lines.append("\n---\n## Per-Ref Cards (Concise)\n")

    dpi_set = set(leaderboards.get("dpi_crews", pd.DataFrame()).get("ref_name", []))
    hold_set = set(leaderboards.get("holding_crews", pd.DataFrame()).get("ref_name", []))

    for _, r in ref_summary.sort_values(["games", "ref_name"], ascending=[False, True]).iterrows():
        if int(r["games"]) < MIN_GAMES_TOTAL:
            continue

        badges = []
        try:
            if not pd.isna(r.get("rank_over", np.nan)) and int(r.get("rank_over")) <= 10:
                badges.append("Over-lean")
            if not pd.isna(r.get("rank_flags", np.nan)) and int(r.get("rank_flags")) <= 10:
                badges.append("Flaggiest")
            if not pd.isna(r.get("rank_ats", np.nan)) and int(r.get("rank_ats")) <= 10:
                badges.append("Home-edge")
        except Exception:
            pass
        if r["ref_name"] in dpi_set:
            badges.append("DPI heavy")
        if r["ref_name"] in hold_set:
            badges.append("Holding crew")
        badge_str = (" [" + ", ".join(badges) + "]") if badges else ""

        lines.append(f"\n**{r['ref_name']}**{badge_str} — Games: {int(r['games'])}")

        d_over = r["over_pct_adj"] - baselines["p_ou_over"]
        d_ats  = r["ats_pct_adj"]  - baselines["p_home_ats"]
        n_ou = int(r.get("ou_decisions", 0))
        n_ats = int(r.get("ats_decisions", 0))

        lines.append(
            "Flags/Gm "
            f"{r['pens_per_game']:.2f} | "
            f"Home ATS {_fmt_pct(r['ats_pct_adj'])} ({d_ats:+.1%} vs lg, n={n_ats}) | "
            f"Over {_fmt_pct(r['over_pct_adj'])} ({d_over:+.1%} vs lg, n={n_ou}) | "
            f"Bias yds (away-home) {r['pen_yds_diff_avg']:.1f}"
        )
        
        # NEW: Impact line
        impact_parts = []
        if "total_epa_impact_pg" in r and not pd.isna(r["total_epa_impact_pg"]):
            impact_parts.append(f"EPA impact {r['total_epa_impact_pg']:.2f}/gm")
        if "high_leverage_penalties_pg" in r and not pd.isna(r["high_leverage_penalties_pg"]):
            impact_parts.append(f"high-leverage {r['high_leverage_penalties_pg']:.2f}/gm")
        if "red_zone_penalties_pg" in r and not pd.isna(r["red_zone_penalties_pg"]):
            impact_parts.append(f"red zone {r['red_zone_penalties_pg']:.2f}/gm")
        if impact_parts:
            lines.append("Impact → " + " | ".join(impact_parts))

        try:
            lines.append(
                f"Angles → 3rd-down bailouts {r['bailout_3rd_fd_per_game']:.2f}/gm, free pass yds {r['free_pass_yards_pg']:.1f}/gm, drive-killers {r['drive_killers_pg']:.2f}/gm, tempo drags {r['tempo_drags_pg']:.2f}/gm, FD by pen {r['fd_by_pen_pg']:.2f}/gm, QB protect {r['qb_protect_pg']:.2f}/gm"
            )
        except Exception:
            pass

        buckets = r.get("fav_bucket_ats", [])
        if buckets:
            bdf = pd.DataFrame(buckets)
            bdf["n"]   = bdf.get("games", 0)
            bdf["dec"] = bdf.get("ats_w", 0) + bdf.get("ats_l", 0)
            bdf["pct"] = bdf.apply(lambda rr: safe_div(rr.get("ats_w", 0), rr.get("dec", 0)), axis=1)
            bdf = bdf.sort_values(["pct", "n"], ascending=[False, False])
            top = bdf.iloc[0]
            tag = " [LOW N]" if int(top.get("games", 0)) < LOW_BUCKET_N else ""
            lines.append(
                f"Best home-fave bucket: {top.get('spread_bucket')} "
                f"{int(top.get('ats_w',0))}-{int(top.get('ats_l',0))} "
                f"(n={int(top.get('games',0))}) ATS {_fmt_pct(top.get('pct',0))}{tag}"
            )

    (OUTDIR / "flag_files_paid_full.md").write_text("\n".join(lines))

def export_flag_files_sunday_reminder(ref_summary: pd.DataFrame):
    _, reminder = select_top_refs_for_content(ref_summary, k_teaser=1, k_reminder=3)
    lines = []
    wk = globals().get("_current_week")
    hdr = f"# Flag Files: Week {wk} — Sunday Whistle Watch\n" if wk else "# Flag Files: Sunday Whistle Watch\n"
    lines.append(hdr)
    if reminder.empty:
        lines.append("Collect more games before trusting ref tilts.")
    else:
        for _, r in reminder.iterrows():
            n_ou = int(r.get("ou_over", 0) + r.get("ou_under", 0))
            n_ats = int(r.get("ats_w", 0) + r.get("ats_l", 0))
            base_line = f"- {r['ref_name']}: Flags/Gm {r['pens_per_game']:.2f} | Home ATS {_fmt_pct(r['ats_pct_adj'])} (n={n_ats}) | Over {_fmt_pct(r['over_pct_adj'])} (n={n_ou})"
            
            # Add impact metric if available
            if "high_leverage_penalties_pg" in r and not pd.isna(r["high_leverage_penalties_pg"]):
                base_line += f" | High-leverage {r['high_leverage_penalties_pg']:.2f}/gm"
            
            lines.append(base_line)
    lines.append("\nFull slate cards are live in the paid drop.")
    (OUTDIR / "flag_files_sunday_reminder.md").write_text("\n".join(lines))

def export_flag_files_image_prompts(ref_summary: pd.DataFrame, baselines: dict):
    teaser, _ = select_top_refs_for_content(ref_summary, k_teaser=1, k_reminder=3)
    payload = {}
    if not teaser.empty:
        r = teaser.iloc[0]
        elements = [
            {"type": "headline_text", "text": f"REF BOMB: {r['ref_name']}", "font": "Impact", "size": 120, "color": "#FF0033", "position": {"x": 540, "y": 180}},
            {"type": "stat_block", "title": "Flags/Game", "value": f"{r['pens_per_game']:.2f}", "subtitle": f"lg {baselines['penalties_per_game']:.2f}", "position": {"x": 540, "y": 460}},
            {"type": "stat_block", "title": "Home ATS", "value": _fmt_pct(r['ats_pct_adj']), "subtitle": "EB adj", "position": {"x": 540, "y": 640}},
            {"type": "stat_block", "title": "Over Rate", "value": _fmt_pct(r['over_pct_adj']), "subtitle": "EB adj", "position": {"x": 540, "y": 820}},
        ]
        
        # Add impact metric if available
        if "total_epa_impact_pg" in r and not pd.isna(r["total_epa_impact_pg"]):
            elements.append(
                {"type": "stat_block", "title": "EPA Impact/Gm", "value": f"{r['total_epa_impact_pg']:.2f}", "subtitle": "expected points", "position": {"x": 540, "y": 1000}}
            )
        
        elements.append(
            {"type": "footer_text", "text": "Full Flag Files → paid drop", "font": "Inter", "size": 40, "color": "#FFFFFF", "position": {"x": 540, "y": 1200}}
        )
        
        payload["free_teaser"] = {
            "version": "FoxEdge_Image_Prompt_v2",
            "goal": "Create a free teaser image highlighting top BBI ref with EPA impact.",
            "canvas": {"width_px": 1080, "height_px": 1350, "dpi": 300},
            "brand": {"palette": {"bg_dark": "#0B0B0B", "neon_red": "#FF0033", "neon_teal": "#1AFFD5", "white": "#FFFFFF"}, "fonts": {"headline": "Impact", "mono": "Roboto Mono", "body": "Inter"}},
            "elements": elements
        }
    (OUTDIR / "flag_files_image_prompts.json").write_text(json.dumps(payload, indent=2))

def export_flag_files_pack(ref_summary: pd.DataFrame, baselines: dict, leaderboards: dict):
    export_flag_files_free_teaser(ref_summary, baselines)
    export_flag_files_paid_full(ref_summary, baselines, leaderboards)
    export_flag_files_sunday_reminder(ref_summary)
    export_flag_files_image_prompts(ref_summary, baselines)


# -----------------------------------------------------
# Ref Intelligence Digest: consolidated business summary
# -----------------------------------------------------
def export_ref_intelligence_digest(ref_summary: pd.DataFrame, baselines: dict, leaderboards: dict, hooks: dict | None = None, top_k: int = 10):
    """
    Produce a single-page markdown 'Ref Intelligence Digest' that condenses the
    most commercially useful outputs (baselines, leaderboards, top refs, hooks).
    """
    path = OUTDIR / "ref_intel_digest.md"
    lines = []

    # Header
    wk = globals().get("_current_week")
    hdr = f"# Ref Intelligence Digest — Week {wk}\n" if wk else "# Ref Intelligence Digest\n"
    lines.append(hdr)
    lines.append(f"Home ATS base: {_fmt_pct(baselines.get('p_home_ats', 0.5))} | "
                 f"Over base: {_fmt_pct(baselines.get('p_ou_over', 0.5))} | "
                 f"Avg flags/gm: {float(baselines.get('penalties_per_game', float('nan'))):.2f}\n")
    lines.append("_Rates EB-shrunk toward league; pushes removed. Negative spread_norm indicates home favorite._\n")

    # Helper to render a compact leaderboard from `leaderboards`
    def section_from_lb(lb_key: str, title: str, cols: list[str]):
        df = leaderboards.get(lb_key)
        if df is None or getattr(df, "empty", True):
            return
        keep = [c for c in cols if c in df.columns]
        show = df.loc[:, keep].head(top_k).copy()
        # Standardize column labels for compactness
        rename = {
            "ref_name": "Ref",
            "games": "G",
            "pens_per_game": "Flags/G",
            "over_pct_adj": "Over%",
            "ats_pct_adj": "Home ATS%",
            "steam_total_avg": "TotSteam",
            "steam_spread_avg": "SprSteam",
            "respect_total_rate": "TotRespect",
            "respect_spread_rate": "SprRespect",
            "total_epa_impact_pg": "EPA/G",
            "avg_epa_per_penalty": "Avg EPA/Pen",
            "high_leverage_penalties_pg": "HL/G",
            "red_zone_penalties_pg": "RZ/G",
            "fourth_qtr_penalties_pg": "Q4/G"
        }
        show = show.rename(columns={k: v for k, v in rename.items() if k in show.columns})
        try:
            lines.append(f"\n## {title}\n")
            lines.append(show.to_markdown(index=False))
        except Exception:
            lines.append(f"\n## {title}\n")
            for _, r in show.iterrows():
                parts = [str(r.get("Ref", r.get("ref_name", "?")))]
                if "Flags/G" in show.columns: parts.append(f"Flags/G {r['Flags/G']:.2f}")
                if "Over%" in show.columns: parts.append(f"Over {100*float(r['Over%']):.1f}%")
                if "Home ATS%" in show.columns: parts.append(f"Home ATS {100*float(r['Home ATS%']):.1f}%")
                lines.append("- " + " | ".join(parts))
        lines.append("\n")

    # Core leaderboards
    section_from_lb("flaggiest", "Flaggiest Crews", ["ref_name","games","pens_per_game"])
    section_from_lb("over_crews", "Over-Leaning Crews", ["ref_name","games","ou_decisions","over_pct_adj"])
    section_from_lb("under_crews", "Under-Leaning Crews", ["ref_name","games","ou_decisions","over_pct_adj"])
    section_from_lb("ats_kings", "Home ATS Kings", ["ref_name","games","ats_decisions","ats_pct_adj"])
    section_from_lb("ats_fades", "Home ATS Fades", ["ref_name","games","ats_decisions","ats_pct_adj"])
    section_from_lb("home_fairness_against", "Home-Fairness Against (penalty diffs)", ["ref_name","games","pen_cnt_diff_avg","pen_yds_diff_avg"])
    section_from_lb("road_fairness_against", "Road-Fairness Against (penalty diffs)", ["ref_name","games","pen_cnt_diff_avg","pen_yds_diff_avg"])

    # Impact leaderboards (if present)
    section_from_lb("highest_epa_impact", "Highest EPA Impact (per game)", ["ref_name","games","total_epa_impact_pg","avg_epa_per_penalty"])
    section_from_lb("most_high_leverage", "Most High-Leverage Flags (per game)", ["ref_name","games","high_leverage_penalties_pg"])
    section_from_lb("most_red_zone_flags", "Most Red-Zone Flags (per game)", ["ref_name","games","red_zone_penalties_pg"])
    section_from_lb("most_4th_qtr_flags", "Most 4th-Quarter Flags (per game)", ["ref_name","games","fourth_qtr_penalties_pg"])

    # Market views if present
    if any(k in leaderboards for k in ["steam_total_up","steam_total_down","market_respect_total","market_respect_spread"]):
        lines.append("\n## Market Moves & Respect\n")
        df = leaderboards.get("steam_total_up")
        if df is not None and not df.empty:
            lines.append("**Totals steam up (avg close−open)**")
            lines.append(df[["ref_name","games","steam_total_avg"]].head(top_k).rename(columns={"ref_name":"Ref","games":"G","steam_total_avg":"TotSteam"}).to_markdown(index=False))
        df = leaderboards.get("steam_total_down")
        if df is not None and not df.empty:
            lines.append("\n**Totals steam down (avg close−open)**")
            lines.append(df[["ref_name","games","steam_total_avg"]].head(top_k).rename(columns={"ref_name":"Ref","games":"G","steam_total_avg":"TotSteam"}).to_markdown(index=False))
        df = leaderboards.get("market_respect_total")
        if df is not None and not df.empty:
            lines.append("\n**Market respect (totals)**")
            lines.append(df[["ref_name","games","respect_total_rate"]].head(top_k).rename(columns={"ref_name":"Ref","games":"G","respect_total_rate":"TotRespect"}).to_markdown(index=False))
        df = leaderboards.get("market_respect_spread")
        if df is not None and not df.empty:
            lines.append("\n**Market respect (spreads)**")
            lines.append(df[["ref_name","games","respect_spread_rate"]].head(top_k).rename(columns={"ref_name":"Ref","games":"G","respect_spread_rate":"SprRespect"}).to_markdown(index=False))
        lines.append("\n")

    # Top refs one-liners (business-facing bullets)
    elig = ref_summary[ref_summary.get("games", 0) >= MIN_GAMES_TOTAL].copy()
    elig = elig.sort_values(["bbi","over_pct_adj","ats_pct_adj","pens_per_game"], ascending=[False, False, False, False]).head(top_k)
    if not elig.empty:
        lines.append("## Top Refs: Bias Bomb Bullets\n")
        for _, r in elig.iterrows():
            n_ou = int(r.get("ou_over", 0) + r.get("ou_under", 0))
            n_ats = int(r.get("ats_w", 0) + r.get("ats_l", 0))
            parts = [
                f"**{r['ref_name']}**",
                f"Flags/G {float(r.get('pens_per_game', 0)):.2f}",
                f"Home ATS {_fmt_pct(r.get('ats_pct_adj', 0))} (n={n_ats})",
                f"Over {_fmt_pct(r.get('over_pct_adj', 0))} (n={n_ou})",
                f"Bias yds (away-home) {float(r.get('pen_yds_diff_avg', 0)):.1f}"
            ]
            if not pd.isna(r.get("total_epa_impact_pg", np.nan)):
                parts.append(f"EPA/G {float(r['total_epa_impact_pg']):.2f}")
            if not pd.isna(r.get("high_leverage_penalties_pg", np.nan)):
                parts.append(f"HL/G {float(r['high_leverage_penalties_pg']):.2f}")
            lines.append("- " + " | ".join(parts))
        lines.append("\n")

    # Hooks/narratives if passed in
    if hooks and isinstance(hooks, dict):
        def _dump_hook(arr_key: str, title: str):
            arr = hooks.get(arr_key, [])
            if arr:
                lines.append(f"## {title}\n")
                for it in arr[:top_k]:
                    ref = it.get("ref")
                    hook = it.get("hook")
                    bbi = it.get("bbi")
                    lines.append(f"- **{ref}** — {hook} [BBI {bbi}]")
                lines.append("\n")
        _dump_hook("hooks_over", "Over/Under Hooks")
        _dump_hook("hooks_home", "Home-Edge Hooks")
        _dump_hook("hooks_road", "Home/Road Bias Hooks")

    # Footer with artifact index
    lines.append("## Artifacts\n")
    lines.append("- `ref_cards.csv` — full per-ref metrics\n"
                 "- `ref_leaderboards.csv` — leaderboard union\n"
                 "- `ref_copy.json` — bias bombs payload\n"
                 "- `referee_report_cards.md` — longform report\n"
                 "- `flag_files_*` — funnel-ready content\n")

    (path).write_text("\n".join(lines))


# ----------------------
# NEW: Export detailed penalty impact report
# ----------------------

def export_penalty_impact_report(penalty_impacts: pd.DataFrame, officials: pd.DataFrame):
    """Export detailed CSV of all penalties with impact metrics for advanced analysis."""
    refs = officials[officials["off_pos"] == "R"][["game_id", "name"]].rename(columns={"name": "ref_name"})
    pen_detail = penalty_impacts.merge(refs, on="game_id", how="left")
    
    output_cols = [
        "game_id", "ref_name", "penalty_type", "penalty_yards", "penalty_team",
        "home_team", "away_team", "pen_against_side",
        "down", "ydstogo", "yardline_100", "qtr", "game_seconds_remaining",
        "score_differential", "wp", "epa", "wpa", "epa_abs", "wpa_abs",
        "is_high_leverage", "is_red_zone", "is_4th_qtr", "is_final_2min",
        "is_close_game", "is_3rd_down", "is_4th_down", "is_passing_down",
        "fd_by_pen"
    ]
    
    available_cols = [c for c in output_cols if c in pen_detail.columns]
    pen_detail[available_cols].to_csv(OUTDIR / "penalty_impact_details.csv", index=False)


# ----------------------
# Main (ENHANCED)
# ----------------------

def infer_current_week(ref_games: pd.DataFrame) -> int | None:
    """
    Prefer the week that has games within a small window around 'today'.
    Falls back to the max observed week when dates are missing.
    """
    # Try to use a date column if present
    df = ref_games.copy()
    date_cols = [c for c in ["gameday", "game_date", "gamedate", "start_time", "start_time_et"] if c in df.columns]
    week_col = "week" if "week" in df.columns else None
    if week_col and date_cols:
        for dc in date_cols:
            try:
                dt = pd.to_datetime(df[dc], errors="coerce")
                if dt.notna().any():
                    today = pd.Timestamp("now").tz_localize(None)
                    # Choose the nearest upcoming week or the most recent past week
                    df2 = df.loc[dt.notna(), [week_col]].copy()
                    df2["date"] = dt[dt.notna()]
                    ahead = df2[df2["date"] >= today].sort_values("date")
                    if not ahead.empty:
                        return int(ahead.iloc[0][week_col])
                    behind = df2[df2["date"] < today].sort_values("date")
                    if not behind.empty:
                        return int(behind.iloc[-1][week_col])
            except Exception:
                continue
    # Fallback: max week in the table
    try:
        w = pd.to_numeric(df.get("week"), errors="coerce")
        if w.notna().any():
            return int(w.max())
    except Exception:
        pass
    return None

def main():
    print("Loading data...")
    years = load_years()
    globals()["_current_season"] = int(years[-1]) if years else None
    schedules, pbp, officials = load_frames(years)

    crews_df = load_2025_crews()
    if crews_df is not None:
        globals()["_crews_df"] = crews_df.copy()

    needed_cols = {"home_score", "away_score", "spread_line", "total_line"}
    if not needed_cols.issubset(set(schedules.columns)):
        print("Schedules missing required columns for ATS/O-U computations. Aborting.")
        return

    print("Building referee game table with EPA/WPA impact analysis...")
    ref_games, type_counts, penalty_impacts = build_ref_game_table(schedules, pbp, officials)
    
    print("Enriching with odds history...")
    odds_df = load_odds_history_from_excel()
    ref_games = merge_odds_into_ref_games(ref_games, odds_df)
    
    globals()["_ref_games_for_md"] = ref_games.copy()
    
    current_week = infer_current_week(ref_games)
    if current_week is not None:
        globals()["_current_week"] = current_week

    try:
        fbz_url = globals().get("FBZ_URL", None)  # optional override
        if current_week is not None:
            print(f"Attempting Football Zebras assignment fetch for Week {current_week}...")
            fbz = get_ref_assignments_from_fbz(
                week=int(current_week),
                season=int(years[-1]) if years else CURRENT_YEAR,
                url=fbz_url
            )
            if fbz is not None and not fbz.empty:
                print(f"  -> Fetched {len(fbz)} assignments from FBZ; merging into ref_games")
                ref_games = merge_assignments_into_ref_games(ref_games, fbz)
                globals()["_fbz_assignments"] = fbz.copy()
            else:
                print("  -> No FBZ assignments fetched or below threshold; skipping merge.")
        else:
            print("Current week unknown; skipping FBZ assignment fetch.")
    except Exception as e:
        print(f"Warning: FBZ assignment integration failed: {e}")
    
    print("Computing league baselines...")
    baselines = league_baselines(ref_games, type_counts)
    
    print("Aggregating per-referee statistics...")
    ref_summary, eligible = aggregate_refs(ref_games, type_counts, baselines)

    ref_summary = attach_crews_to_summary(ref_summary, globals().get("_crews_df", None))

    print("Creating leaderboards...")
    leaderboards = make_leaderboards(ref_summary)
    
    # NEW: Add impact leaderboards
    print("Creating impact leaderboards...")
    impact_leaderboards = make_impact_leaderboards(ref_summary, penalty_impacts, officials)
    leaderboards.update(impact_leaderboards)

    leaderboards.update(compute_prop_leaderboards(ref_summary))

    print("Generating narratives and exporting files...")
    export_public_csv(ref_summary)
    hooks = generate_narratives(ref_summary, baselines)

    export_csv_json(ref_summary, leaderboards)
    
    copy_path = OUTDIR / "ref_copy.json"
    if copy_path.exists():
        with open(copy_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(hooks)
    
    tier_counts = ref_summary[["tier_over", "tier_ats", "tier_flags", "tier_fairness"]].apply(lambda s: s.value_counts().to_dict())
    data["tier_counts"] = {k: dict(v) for k, v in tier_counts.to_dict().items()}
    
    top_bbi_df = ref_summary[ref_summary["games"] >= MIN_GAMES_TOTAL].sort_values("bbi", ascending=False)[["ref_name", "games", "bbi"]].head(15)
    data["bbi_top"] = top_bbi_df.to_dict(orient="records")
    
    with open(copy_path, "w") as f:
        json.dump(data, f, indent=2)

    try:
        print("Plotting referee trends...")
        top_refs = ref_summary.sort_values("bbi", ascending=False)["ref_name"].head(5).tolist()
        plot_ref_trends(ref_games, baselines, top_refs, ref_summary=ref_summary)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    try:
        print("Rendering per-ref image cards...")
        export_ref_card_images(ref_summary, ref_games, penalty_impacts, current_week)
    except Exception as e:
        print(f"Warning: Could not render ref card images: {e}")

    print("Exporting markdown report...")
    export_markdown(ref_summary, baselines, leaderboards)

    print("Generating Flag Files content pack...")
    export_flag_files_pack(ref_summary, baselines, leaderboards)

    print("Exporting Ref Intelligence Digest...")
    export_ref_intelligence_digest(ref_summary, baselines, leaderboards, hooks)
    
    # NEW: Export detailed penalty impact report
    print("Exporting detailed penalty impact report...")
    export_penalty_impact_report(penalty_impacts, officials)

    print(f"\n✓ Referee report cards generated successfully!")
    print(f"  Years analyzed: {years}")
    print(f"  Total games: {len(ref_games)}")
    print(f"  Referees tracked: {len(ref_summary)}")
    print(f"  Output directory: {OUTDIR.resolve()}")
    print(f"\nGenerated files:")
    print(f"  - referee_report_cards.md (comprehensive markdown report)")
    print(f"  - ref_cards.csv (full referee statistics)")
    print(f"  - ref_cards_public.csv (public-facing metrics)")
    print(f"  - ref_leaderboards.csv (all leaderboards)")
    print(f"  - ref_copy.json (content for social/marketing)")
    print(f"  - penalty_impact_details.csv (NEW: individual penalty EPA/WPA analysis)")
    print(f"  - flag_files_*.md (content pack for Flag Files brand)")
    print(f"  - plots/*.png (referee trend visualizations)")
    print(f"  - ref_cards_images/*.png (per-ref assignment cards)")


if __name__ == "__main__":
    main()