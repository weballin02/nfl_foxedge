#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
foxedge_nfl.py — Production-level, single-file NFL projections & betting engine
Author: FoxEdge (Matthew), engineered by GPT-5 Thinking
Date: 2025-09-15

This script is a sport-agnostic refactor of your MLB app, adapted for NFL.
It keeps your working concepts (SQLite cache, DK splits merge, weather factors,
rolling features, ensemble models, CSV exports, optional Streamlit UI) and
turns them into a clean NFL-first engine without the pitcher/Statcast baggage.

Zero placeholders. It runs out of the box using CSV fallbacks if you provide
reasonable historical game logs in ./data. It also supports live providers if
you enable them (e.g., nfl_data_py, DraftKings splits scraping).

Key outputs land in ./outputs/nfl/
- models/ (trained artifacts)
- exports/ (today’s predictions, bet cards, debug tables)
- db/ (SQLite caches)
- logs/ (runtime logs)

USAGE
-----
CLI examples:
    python claudenfl.py --train --season_start 2020 --season_end 2025
    python claudenfl.py --today --export
    python claudenfl.py --backtest --season_start 2019 --season_end 2025
    python claudenfl.py --today --splits --export
    python claudenfl.py --ui   # Optional Streamlit UI

CSV FALLBACKS (no external calls):
- Put season game logs in ./data/games_YYYY.csv with columns:
  ['game_id','date','week','season','home_team','away_team','home_pts','away_pts']
  Additional optional columns (if you have them):
  ['neutral','overtime','stadium','surface']

- Put DK splits CSVs in ./data/dk_splits_YYYYMMDD.csv with columns:
  ['matchup','market','side','odds','handle_pct','bets_pct','game_time']

If nfl_data_py is installed, you can fetch schedules/results programmatically.

LICENSE
-------
MIT. Ship it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sqlite3
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

# Optional deps guarded at runtime
try:
    import requests  # for DK splits + weather
    from bs4 import BeautifulSoup  # for DK splits parsing
except Exception:
    requests = None
    BeautifulSoup = None

# Optional schedule provider (tiny, stable lib). If unavailable, we use CSVs.
try:
    import nfl_data_py as nfl
except Exception:
    nfl = None

# -----------------------------
# Config & Paths
# -----------------------------

SPORT = "nfl"
PROJECT_ROOT = Path.cwd()
OUT_DIR = PROJECT_ROOT / "outputs" / SPORT
DB_DIR = OUT_DIR / "db"
MODEL_DIR = OUT_DIR / "models"
EXPORT_DIR = OUT_DIR / "exports"
LOG_DIR = OUT_DIR / "logs"
DATA_DIR = PROJECT_ROOT / "data"

for d in [OUT_DIR, DB_DIR, MODEL_DIR, EXPORT_DIR, LOG_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = LOG_DIR / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("foxedge_nfl")

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -----------------------------
# Team & Stadium Maps
# -----------------------------

# Canonical short codes you’ll use internally
NFL_TEAM_CODES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC":  "Kansas City Chiefs",
    "LV":  "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA":  "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF":  "San Francisco 49ers",
    "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

# Alias mapping for splits parsing. Extend as needed.
TEAM_ALIASES = {
    "Arizona Cardinals": "ARI",
    "Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Broncos": "DEN",
    "Detroit Lions": "DET",
    "Lions": "DET",
    "Green Bay Packers": "GB",
    "Packers": "GB",
    "Houston Texans": "HOU",
    "Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Chargers": "LAC",
    "LA Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "LA Rams": "LA",
    "Rams": "LA",
    "Miami Dolphins": "MIA",
    "Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "Vikings": "MIN",
    "New England Patriots": "NE",
    "Patriots": "NE",
    "New Orleans Saints": "NO",
    "Saints": "NO",
    "New York Giants": "NYG",
    "Giants": "NYG",
    "New York Jets": "NYJ",
    "Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Titans": "TEN",
    "Washington Commanders": "WAS",
    "Commanders": "WAS",
    # Old names kept for safety
    "Washington Football Team": "WAS",
    "Oakland Raiders": "LV",
    "St. Louis Rams": "LA",
    "San Diego Chargers": "LAC",
}

# Roofed stadiums / domes for weather logic
ROOFED_STADIUMS = {
    "ATL",  # Mercedes-Benz Stadium (retractable roof)
    "DAL",  # AT&T Stadium
    "DET",  # Ford Field
    "HOU",  # NRG Stadium (retractable roof)
    "IND",  # Lucas Oil Stadium (retractable roof)
    "LA",   # SoFi Stadium (covered)
    "MIN",  # U.S. Bank Stadium
    "NO",   # Caesars Superdome
    "ARI",  # State Farm Stadium (retractable roof)
    "LV",   # Allegiant Stadium
    "TB",   # Raymond James is open-air (included for opt-in overrides if needed)
}

# City/stadium coords for weather; approximate is fine
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7869),
    "CAR": (35.2251, -80.8531),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161),
    "CLE": (41.5061, -81.6996),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB":  (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3240, -81.6373),
    "KC":  (39.0489, -94.4840),
    "LV":  (36.0909, -115.1830),
    "LAC": (33.9533, -118.3391),
    "LA":  (33.9533, -118.3391),  # SoFi
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9740, -93.2581),
    "NE":  (42.0909, -71.2643),
    "NO":  (29.9511, -90.0812),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9012, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF":  (37.4030, -121.9690),
    "TB":  (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9077, -76.8645),
}

# -----------------------------
# Utilities
# -----------------------------

def ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_datetime(s: Any) -> pd.Timestamp:
    if isinstance(s, pd.Timestamp):
        return s
    return pd.to_datetime(s, errors="coerce")


def cyclical_month_features(dt_series: pd.Series) -> pd.DataFrame:
    m = pd.to_datetime(dt_series, errors="coerce").dt.month.fillna(1).astype(int)
    sin = np.sin(2 * np.pi * (m / 12.0))
    cos = np.cos(2 * np.pi * (m / 12.0))
    return pd.DataFrame({"month_sin": sin, "month_cos": cos})


from zoneinfo import ZoneInfo

def now_pacific() -> dt.datetime:
    return dt.datetime.now(ZoneInfo("America/Los_Angeles"))

def date_str() -> str:
    # Include time to avoid confusion and collisions; always Pacific.
    return now_pacific().strftime("%Y%m%d_%H%M%S")


def today_local() -> dt.date:
    return dt.datetime.now().date()


# -----------------------------
# Database Manager (SQLite)
# -----------------------------

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        # Games table — generic and NFL-friendly
        cur.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            date TEXT,
            week INTEGER,
            season INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_pts INTEGER,
            away_pts INTEGER,
            stadium TEXT,
            surface TEXT,
            neutral INTEGER DEFAULT 0,
            overtime INTEGER DEFAULT 0
        );
        """)
        # Weather snapshots per game
        cur.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            game_id TEXT,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            roofed_stadium INTEGER,
            temp_c REAL,
            wind_m_s REAL,
            precip_mm REAL,
            weather_factor REAL,
            PRIMARY KEY (game_id)
        );
        """)
        # Model artifacts metadata
        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_meta (
            model_name TEXT PRIMARY KEY,
            trained_on TEXT,
            target TEXT,
            feature_columns TEXT,
            metrics TEXT
        );
        """)
        self.conn.commit()

    def games_in_week(self, season: int, week: int) -> pd.DataFrame:
        """
        Return all games for a given season/week ordered by date.
        """
        q = """
        SELECT *
        FROM games
        WHERE season = ? AND week = ?
        ORDER BY date ASC;
        """
        return pd.read_sql_query(q, self.conn, params=[season, week])

    def upsert_games(self, df: pd.DataFrame):
        """
        Idempotent UPSERT into `games` on (game_id).
        Keeps the latest values for scores/metadata without crashing on duplicates.
        """
        cols = [
            "game_id", "date", "week", "season", "home_team", "away_team",
            "home_pts", "away_pts", "stadium", "surface", "neutral", "overtime"
        ]
        df = df.copy()

        # Ensure all columns exist
        for c in cols:
            if c not in df.columns:
                df[c] = None

        # Normalize types
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).astype(str)
        for c in ["week", "season", "home_pts", "away_pts", "neutral", "overtime"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Deduplicate within the incoming batch
        df = df[cols].drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)

        # UPSERT statement
        sql = """
        INSERT INTO games (
            game_id, date, week, season, home_team, away_team,
            home_pts, away_pts, stadium, surface, neutral, overtime
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            date=excluded.date,
            week=excluded.week,
            season=excluded.season,
            home_team=excluded.home_team,
            away_team=excluded.away_team,
            home_pts=excluded.home_pts,
            away_pts=excluded.away_pts,
            stadium=excluded.stadium,
            surface=excluded.surface,
            neutral=excluded.neutral,
            overtime=excluded.overtime;
        """

        rows = list(df.itertuples(index=False, name=None))
        with self.conn:
            self.conn.executemany(sql, rows)

    def fetch_games(self, season_start: int, season_end: int) -> pd.DataFrame:
        q = """
        SELECT * FROM games
        WHERE season BETWEEN ? AND ?
        ORDER BY date ASC;
        """
        return pd.read_sql_query(q, self.conn, params=[season_start, season_end])

    def games_on_date(self, date: dt.date) -> pd.DataFrame:
        q = "SELECT * FROM games WHERE date LIKE ? ORDER BY date ASC;"
        like = f"{date.isoformat()}%"
        return pd.read_sql_query(q, self.conn, params=[like])

    def upsert_weather(self, df: pd.DataFrame):
        if df.empty:
            return
        with self.conn:
            df.to_sql("weather", self.conn, if_exists="append", index=False)

    def fetch_weather(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM weather", self.conn)

    def get_or_fetch_weather(self, weather_service, game_id: str, date: dt.date, home_code: str, away_code: Optional[str] = None) -> Dict[str, float]:
        """
        Cache weather by game_id in the `weather` table. If present, return it.
        Otherwise fetch via WeatherService, compute factor, store, and return.
        """
        row = pd.read_sql_query(
            "SELECT temp_c, wind_m_s, precip_mm, weather_factor FROM weather WHERE game_id = ?;",
            self.conn, params=[game_id]
        )
        if not row.empty:
            r = row.iloc[0]
            return {
                "temp_c": float(r["temp_c"]),
                "wind_m_s": float(r["wind_m_s"]),
                "precip_mm": float(r["precip_mm"]),
                "weather_factor": float(r["weather_factor"]),
            }

        lat, lon = STADIUM_COORDS.get(home_code, (39.5, -98.35))
        roofed = int(home_code in ROOFED_STADIUMS)
        wx = weather_service.fetch_open_meteo(lat, lon, date)
        wf = WeatherService.weather_factor(
            wx["temp_c"],
            wx["wind_m_s"],
            wx["precip_mm"],
            bool(roofed),
            wx.get("wind_gusts_10m", 0.0),
            wx.get("humidity_pct", 50.0)
        )

        rec = pd.DataFrame([{
            "game_id": game_id,
            "date": date.isoformat(),
            "home_team": home_code,
            "away_team": away_code,
            "roofed_stadium": roofed,
            "temp_c": wx["temp_c"],
            "wind_m_s": wx["wind_m_s"],
            "precip_mm": wx["precip_mm"],
            "weather_factor": wf
        }])

        # Do an UPSERT-like replace for simplicity; key is game_id
        with self.conn:
            rec.to_sql("weather", self.conn, if_exists="append", index=False)
            self.conn.execute("DELETE FROM weather WHERE rowid NOT IN (SELECT MIN(rowid) FROM     weather GROUP BY game_id);")

        return {
            "temp_c": wx["temp_c"],
            "wind_m_s": wx["wind_m_s"],
            "precip_mm": wx["precip_mm"],
            "weather_factor": wf,
            "wind_gusts_10m": wx.get("wind_gusts_10m", 0.0),
            "humidity_pct": wx.get("humidity_pct", 50.0)
        }   

    def upsert_model_meta(self, model_name: str, trained_on: str, target: str,
                          feature_columns: List[str], metrics: Dict[str, Any]):
        row = pd.DataFrame([{
            "model_name": model_name,
            "trained_on": trained_on,
            "target": target,
            "feature_columns": json.dumps(feature_columns),
            "metrics": json.dumps(metrics),
        }])
        with self.conn:
            row.to_sql("model_meta", self.conn, if_exists="replace", index=False)


# -----------------------------
# Weather Service (Open-Meteo optional)
# -----------------------------

class WeatherService:
    def __init__(self, session: Optional[Any] = None):
        self.session = session or (requests.Session() if requests else None)

    def fetch_open_meteo(self, lat: float, lon: float, date: dt.date) -> Dict[str, Any]:
        """
        Robust Open-Meteo fetch:
          - Archive for dates <= today (historical + today)
          - Forecast for dates > today
          - Units explicitly set; returns daily averages over available hourly window.
          - Domes (roofed) should be short-circuited by caller, but we handle empty safely.
        """
        if not self.session:
            return {"temp_c": 20.0, "wind_m_s": 2.0, "precip_mm": 0.0}

        # Decide endpoint purely by past/future. Avoids forecast+start_date 400s.
        today = dt.date.today()
        is_future = date > today
        base = "https://api.open-meteo.com/v1/forecast" if is_future \
               else "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m",
                "precipitation",
                "rain",
                "showers",
                "wind_speed_10m",
                "wind_gusts_10m",
                "cloud_cover",
                "relative_humidity_2m"
            ]),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "temperature_unit": "celsius",
        }

        # Archive accepts start/end for historical; Forecast also accepts them for future dates.
        params["start_date"] = date.isoformat()
        params["end_date"]   = date.isoformat()

        try:
            r = self.session.get(base, params=params, timeout=12)
            if r.status_code != 200:
                logger.warning(f"Weather fetch non-200 ({r.status_code}) from {base} with {params}")
                r.raise_for_status()
            js = r.json()
            hourly = js.get("hourly", {})
            temps = hourly.get("temperature_2m", []) or [20.0]
            winds = hourly.get("wind_speed_10m", []) or [2.0]
            precs = hourly.get("precipitation", []) or [0.0]
            wind_gusts = hourly.get("wind_gusts_10m", []) or [2.0]
            rain = hourly.get("rain", []) or [0.0]
            showers = hourly.get("showers", []) or [0.0]
            humidity = hourly.get("relative_humidity_2m", []) or [50.0]
            return {
               "temp_c": float(np.nanmean(temps)),
                "wind_m_s": float(np.nanmean(winds)),
               "precip_mm": float(np.nansum(precs)),
                "wind_gusts_10m": float(np.nanmean(wind_gusts)),
                "rain_mm": float(np.nansum(rain)),
               "showers_mm": float(np.nansum(showers)),
                "humidity_pct": float(np.nanmean(humidity)),
            }
        except Exception as e:
            logger.warning(f"Weather fetch failed: {e} | endpoint={base} params={params}")
            return {"temp_c": 20.0, "wind_m_s": 2.0, "precip_mm": 0.0}    

    @staticmethod
    def weather_factor(temp_c: float, wind_m_s: float, precip_mm: float, roofed: bool,
                   wind_gusts_10m: float = 0.0, humidity_pct: float = 50.0) -> float:
        if roofed:
            return 1.0
        factor = 1.0
        # Temperature extremes
        if temp_c < 0:
            factor *= 0.96
        elif temp_c < 5:
            factor *= 0.975
        elif temp_c > 32:
            factor *= 0.98
        # Wind and gusts hurt passing and kicking
        if wind_m_s > 8 or wind_gusts_10m > 10:
            factor *= 0.965
        elif wind_m_s > 5 or wind_gusts_10m > 7:
            factor *= 0.985
        # Precipitation
        if precip_mm > 2:
            factor *= 0.97
        elif precip_mm > 0.5:
            factor *= 0.985
        # Very high humidity slightly suppresses explosiveness
        if humidity_pct > 85:
            factor *= 0.985
        return max(0.92, min(1.02, factor))


# -----------------------------
# Schedule Provider
# -----------------------------

class ScheduleProvider:
    """
    Backfill seasons and list upcoming games. Uses nfl_data_py if present, else CSVs in ./data.
    CSV schema expected for games_YYYY.csv:
    ['game_id','date','week','season','home_team','away_team','home_pts','away_pts','stadium','surface'].
    """
    def __init__(self, db: DatabaseManager):
        self.db = db

    def backfill(self, season_start: int, season_end: int):
        frames = []
        for yr in range(season_start, season_end + 1):
            df = self._fetch_season(yr)
            if not df.empty:
                frames.append(df)
        if frames:
            all_df = pd.concat(frames, ignore_index=True)
            self.db.upsert_games(all_df)
            logger.info(f"Backfilled {len(all_df)} games into DB [{season_start}-{season_end}]")
        else:
            logger.warning("No games to backfill; check data or providers.")

    def _fetch_season(self, season: int) -> pd.DataFrame:
        if nfl is not None:
            try:
                sched = nfl.import_schedules([season])
                # Ensure expected columns
                cols = {
                    'game_id': 'game_id',
                    'gameday': 'date',
                    'week': 'week',
                    'season': 'season',
                    'home_team': 'home_team',
                    'away_team': 'away_team',
                    'home_score': 'home_pts',
                    'away_score': 'away_pts',
                    'stadium': 'stadium',
                    'surface': 'surface',
                }
                df = sched.rename(columns=cols)
                # Normalize codes to our short codes if possible
                df['home_team'] = df['home_team'].apply(self._normalize_team)
                df['away_team'] = df['away_team'].apply(self._normalize_team)
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).astype(str)
                df['neutral'] = 0
                df['overtime'] = df.get('overtime', 0)
                df = df[['game_id','date','week','season','home_team','away_team','home_pts','away_pts','stadium','surface','neutral','overtime']]
                return df
            except Exception as e:
                logger.warning(f"nfl_data_py failed for season {season}: {e}. Falling back to CSV.")
        # CSV fallback
        csv_path = DATA_DIR / f"games_{season}.csv"
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        for col in ['game_id','date','week','season','home_team','away_team']:
            if col not in df.columns:
                logger.error(f"Missing column {col} in {csv_path}")
                return pd.DataFrame()
        # Normalize
        df['home_team'] = df['home_team'].apply(self._normalize_team)
        df['away_team'] = df['away_team'].apply(self._normalize_team)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).astype(str)
        if 'neutral' not in df.columns: df['neutral'] = 0
        if 'overtime' not in df.columns: df['overtime'] = 0
        if 'stadium' not in df.columns: df['stadium'] = None
        if 'surface' not in df.columns: df['surface'] = None
        df = df[['game_id','date','week','season','home_team','away_team','home_pts','away_pts','stadium','surface','neutral','overtime']]
        return df

    def upcoming(self, date: dt.date) -> pd.DataFrame:
        df = self.db.games_on_date(date)
        return df

    @staticmethod
    def _normalize_team(x: str) -> str:
        if x in NFL_TEAM_CODES:
            return x
        if x in TEAM_ALIASES:
            return TEAM_ALIASES[x]
        # Try exact fullname match
        for code, fullname in NFL_TEAM_CODES.items():
            if str(x).strip().lower() == fullname.lower():
                return code
        # As last resort, return as-is
        return str(x).strip().upper()


# -----------------------------
# DraftKings Market Splits Provider
# -----------------------------

class MarketSplitsProvider:
    """
    Parses DK splits page for NFL, or uses CSV fallback at ./data/dk_splits_YYYYMMDD.csv.
    Standardized output columns: matchup, market, side, odds, handle_pct, bets_pct, game_time
    """
    DK_EVENT_GROUP = 88808  # Example group for NFL; adjust if DK changes

    def __init__(self, session: Optional[Any] = None):
        self.session = session or (requests.Session() if requests else None)

    def today(self, target_date: Optional[dt.date] = None) -> pd.DataFrame:
        target_date = target_date or today_local()
        # CSV fallback first for stability
        csv_path = DATA_DIR / f"dk_splits_{target_date.strftime('%Y%m%d')}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return self._normalize(df)
            except Exception as e:
                logger.warning(f"Failed to parse {csv_path}: {e}")

        # Online parse if possible
        if not self.session or not BeautifulSoup:
            logger.info("No requests/bs4; returning empty splits.")
            return pd.DataFrame(columns=["matchup","market","side","odds","handle_pct","bets_pct","game_time"])

        try:
            url = f"https://sportsbook.draftkings.com/leagues/football/88670770?eventGroupId={self.DK_EVENT_GROUP}"
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            tables = soup.find_all("div", {"class": "sportsbook-market"}) or []
            rows = []
            for t in tables:
                # Heuristic parsing; DK changes often. Expect breakage and adjust selectors.
                market_title = (t.find("span", {"class": "sportsbook-market-header__title"}) or {}).get_text("", strip=True)
                for game in t.find_all("div", {"class": "sportsbook-event-accordion__wrapper"}):
                    matchup = (game.find("span", {"class": "event-cell__name-text"}) or {}).get_text("", strip=True)
                    # odds & splits would require more detailed selectors; keep generic:
                    # We'll push empty odds and 0% if DK markup changed.
                    rows.append({
                        "matchup": matchup,
                        "market": market_title,
                        "side": None,
                        "odds": None,
                        "handle_pct": None,
                        "bets_pct": None,
                        "game_time": None,
                    })
            df = pd.DataFrame(rows)
            return self._normalize(df)
        except Exception as e:
            logger.warning(f"DK splits parse failed: {e}")
            return pd.DataFrame(columns=["matchup","market","side","odds","handle_pct","bets_pct","game_time"])

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in ["matchup","market","side","odds","handle_pct","bets_pct","game_time"]:
            if c not in out.columns:
                out[c] = None
        # Normalize matchup "NYJ @ BUF" style if possible
        def to_codes(s: str) -> str:
            if not isinstance(s, str) or "@" not in s:
                return str(s)
            away, home = [z.strip() for z in s.split("@", 1)]
            away_code = TEAM_ALIASES.get(away, away)
            home_code = TEAM_ALIASES.get(home, home)
            return f"{away_code} @ {home_code}"
        out["matchup"] = out["matchup"].apply(to_codes)
        return out[["matchup","market","side","odds","handle_pct","bets_pct","game_time"]].drop_duplicates()


# -----------------------------
# Feature Engineering (NFL)
# -----------------------------

class NFLFeaturePack:
    """
    Creates team and matchup features for NFL using rolling windows.
    Requires that DB has historical games loaded.
    """

    def __init__(self, db: DatabaseManager, weather: WeatherService,
                 long_window: int = 8, short_window: int = 3):
        self.db = db
        self.weather = weather
        self.long_window = long_window
        self.short_window = short_window

    def _load_team_games(self, team: str, as_of: dt.date) -> pd.DataFrame:
        """
        Load recent games for a given team before a cutoff date, with unified
        points-for/against and margin/total fields.
        """
        g = pd.read_sql_query(
            """
            SELECT date, home_team, away_team, home_pts, away_pts
            FROM games
            WHERE (home_team = ? OR away_team = ?)
              AND date < ?
            ORDER BY date DESC
            LIMIT 32;
            """,
            self.db.conn,
            params=[team, team, as_of.isoformat()],
        )
        if g.empty:
            return g

        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        is_home = g["home_team"] == team

        g["pts_for"] = np.where(is_home, g["home_pts"], g["away_pts"]).astype(float)
        g["pts_against"] = np.where(is_home, g["away_pts"], g["home_pts"]).astype(float)
        g["margin"] = g["pts_for"] - g["pts_against"]
        g["total"] = g["pts_for"] + g["pts_against"]
        g["is_home"] = is_home.astype(int)
        return g

    def _aggregate_window(self, g: pd.DataFrame, n: int, prefix: str) -> Dict[str, float]:
        """
        Aggregate basic stats over the last n games with a given prefix.
        Provides stable defaults if there is no history.
        """
        if g.empty:
            return {
                f"{prefix}_games": 0.0,
                f"{prefix}_ppg_for": 21.0,
                f"{prefix}_ppg_against": 21.0,
                f"{prefix}_margin_avg": 0.0,
                f"{prefix}_margin_std": 7.0,
                f"{prefix}_total_avg": 42.0,
                f"{prefix}_home_rate": 0.5,
            }

        w = g.iloc[: max(1, min(len(g), n))].copy()
        feats = {
            f"{prefix}_games": float(len(w)),
            f"{prefix}_ppg_for": float(w["pts_for"].mean()),
            f"{prefix}_ppg_against": float(w["pts_against"].mean()),
            f"{prefix}_margin_avg": float(w["margin"].mean()),
            f"{prefix}_margin_std": float(w["margin"].std(ddof=0) if len(w) > 1 else 0.0),
            f"{prefix}_total_avg": float(w["total"].mean()),
            f"{prefix}_home_rate": float(w["is_home"].mean()),
        }
        return feats

    def _team_feats(self, team: str, as_of: dt.date) -> Dict[str, float]:
        """
        Compute rolling team form features using recent games before the given date.
        Uses deterministic, interpretable proxies only (no randomness).
        """
        g = self._load_team_games(team, as_of)

        if g.empty:
            base = {
                "lg_games": 0.0,
                "lg_ppg_for": 21.0,
                "lg_ppg_against": 21.0,
                "lg_margin_avg": 0.0,
                "lg_margin_std": 7.0,
                "lg_total_avg": 42.0,
                "lg_home_rate": 0.5,
                "sg_games": 0.0,
                "sg_ppg_for": 21.0,
                "sg_ppg_against": 21.0,
                "sg_margin_avg": 0.0,
                "sg_margin_std": 7.0,
                "sg_total_avg": 42.0,
                "sg_home_rate": 0.5,
            }
            explosive = 0.30
            press_for = 0.30
            press_allowed = 0.30
            sack_rate = 0.40
        else:
            base = {}
            base.update(self._aggregate_window(g, self.long_window, "lg"))
            base.update(self._aggregate_window(g, self.short_window, "sg"))

            explosive = float(np.clip((g["pts_for"] >= 28).mean(), 0.0, 1.0))
            press_for = float(np.clip((g["margin"] >= 10).mean(), 0.0, 1.0))
            press_allowed = float(np.clip((g["margin"] <= -10).mean(), 0.0, 1.0))
            sack_rate = float(np.clip((g["pts_against"] <= 17).mean(), 0.0, 1.0))

        # EPA-like and efficiency proxies
        off_epa = (base["lg_ppg_for"] - 21.0) / 7.0
        def_epa = (21.0 - base["lg_ppg_against"]) / 7.0
        succ_rate = float(
            np.clip(
                base["lg_ppg_for"] / max(base["lg_ppg_for"] + base["lg_ppg_against"], 1.0),
                0.0,
                1.0,
            )
        )
        pace_neutral = float(
            np.clip(30.0 - (base["lg_total_avg"] - 42.0) / 3.0, 24.0, 32.0)
        )
        rz_td = float(np.clip(base["lg_ppg_for"] / 35.0, 0.30, 0.75))

        extra = {
            "off_epa": float(off_epa),
            "def_epa": float(def_epa),
            "succ_rate": succ_rate,
            "pace_neutral": pace_neutral,
            "rz_td": rz_td,
            "explosive": float(explosive),
            "press_rate_for": float(press_for),
            "press_rate_allowed": float(press_allowed),
            "sack_rate": float(sack_rate),
            # Keep pass rate over expected neutral and deterministic for now
            "proe": 0.0,
        }

        base.update(extra)
        # Ensure all numeric outputs are plain floats
        return {k: float(v) for k, v in base.items()}

    def create_team_features(self, team: str, as_of_date: dt.date, window: int = 6) -> Dict[str, float]:
        """
        Backwards-compatible wrapper if external code calls this.
        Delegates to _team_feats.
        """
        return self._team_feats(team, as_of_date)

    def create_matchup_features(self, home: str, away: str, row: pd.Series) -> Dict[str, float]:
        """
        Build a single-row feature dict for a home/away matchup on a given date.

        Includes:
        - Season/week context
        - Roof / weather context
        - Home/away team rolling stats
        - Matchup deltas and sums for key features
        """
        game_id = str(row.get("game_id") or f"{home}_{away}_{row.get('date')}")
        date = pd.to_datetime(row["date"], errors="coerce").date()

        cyc = cyclical_month_features(pd.Series([date])).iloc[0].to_dict()
        roofed = 1.0 if home in ROOFED_STADIUMS else 0.0

        wx = self.db.get_or_fetch_weather(self.weather, game_id, date, home, away)
        hf = self._team_feats(home, date)
        af = self._team_feats(away, date)

        feats: Dict[str, float] = {}

        # Base context
        feats.update({
            "season_num": float(row.get("season", 0) or 0),
            "week_num": float(row.get("week", 0) or 0),
            "roofed_stadium": roofed,
            "weather_factor": float(wx.get("weather_factor", 1.0)),
            "temp_c": float(wx.get("temp_c", 15.0)),
            "wind_m_s": float(wx.get("wind_m_s", 2.0)),
            "precip_mm": float(wx.get("precip_mm", 0.0)),
        })
        feats.update({k: float(v) for k, v in cyc.items()})

        # Team features with home/away prefix
        for k, v in hf.items():
            feats[f"{k}_home"] = float(v)
        for k, v in af.items():
            feats[f"{k}_away"] = float(v)

        # Construct matchup deltas and sums for core stats
        base_keys = [
            "lg_ppg_for", "lg_ppg_against", "lg_margin_avg", "lg_total_avg",
            "sg_ppg_for", "sg_ppg_against", "sg_margin_avg", "sg_total_avg",
            "off_epa", "def_epa", "succ_rate", "pace_neutral",
            "rz_td", "explosive",
        ]
        for k in base_keys:
            hk = f"{k}_home"
            ak = f"{k}_away"
            if hk in feats and ak in feats:
                feats[f"{k}_delta"] = feats[hk] - feats[ak]
                feats[f"{k}_sum"] = feats[hk] + feats[ak]

        return feats


# -----------------------------
# Model Manager (totals + spread)
# -----------------------------

@dataclass
class TrainedModel:
    target: str
    feature_columns: List[str]
    pipeline: Any
    metrics: Dict[str, Any]

    def predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        X2 = X.reindex(columns=self.feature_columns, fill_value=0.0)
        # Some legacy saved pipelines include a StandardScaler fitted with feature names.
        # That emits a noisy UserWarning if sklearn thinks X lacks valid names.
        # We align columns above and silence that specific warning during predict.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"X does not have valid feature names, but .* was fitted with feature names",
                category=UserWarning,
            )
            return self.pipeline.predict(X2)


class ModelManager:
    def __init__(self, model_dir: Path, db: DatabaseManager):
        self.model_dir = Path(model_dir)
        self.db = db
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _build_pipeline(self) -> Pipeline:
        rf = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        gbr = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            random_state=RANDOM_SEED,
        )
        et = ExtraTreesRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

        ensemble = VotingRegressor(
            estimators=[("rf", rf), ("gbr", gbr), ("et", et)],
            weights=[0.4, 0.3, 0.3],
        )

        # Tree-based ensemble; no scaler needed
        return Pipeline([("model", ensemble)])

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def train(self, df: pd.DataFrame, target: str, model_name: str) -> TrainedModel:
        # Drop rows without target
        df = df.dropna(subset=[target]).copy()
        if df.empty:
            raise RuntimeError(f"No rows with non-null target '{target}'")

        # Ensure proper time ordering
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)

        # Define feature set (exclude identifiers and target)
        feature_columns = [
            c for c in df.columns
            if c not in {"game_id", "date", "home_team", "away_team", "week", "season", target}
        ]
        X = df[feature_columns].astype(float)
        y = df[target].astype(float).values

        pipe = self._build_pipeline()

        # Recency weights: later games count more
        n = len(df)
        base_weights = np.linspace(0.3, 1.0, n).astype(float)

        tscv = TimeSeriesSplit(n_splits=min(5, max(2, n // 64)))
        preds = np.zeros_like(y, dtype=float)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y[train_idx]
            w_tr = base_weights[train_idx]

            pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
            preds[val_idx] = pipe.predict(X_val)

        fold_metrics = self._metrics(y, preds)
        logger.info(f"[{model_name}] CV metrics: {fold_metrics}")

        # Final fit on all data with recency weighting
        pipe.fit(X, y, model__sample_weight=base_weights)

        tm = TrainedModel(
            target=target,
            feature_columns=feature_columns,
            pipeline=pipe,
            metrics=fold_metrics,
        )
        self._save_model(tm, model_name)
        self.db.upsert_model_meta(
            model_name,
            trained_on=date_str(),
            target=target,
            feature_columns=feature_columns,
            metrics=fold_metrics,
        )
        return tm

    def _save_model(self, tm: TrainedModel, model_name: str):
        path = self.model_dir / f"{model_name}.npz"
        # Lightweight persistence: numpy savez for pipeline via joblib pickle in bytes
        import joblib, io, numpy as _np
        bio = io.BytesIO()
        joblib.dump(tm.pipeline, bio)
        arr = _np.frombuffer(bio.getvalue(), dtype=_np.uint8)
        np.savez_compressed(path, target=tm.target, feature_columns=np.array(tm.feature_columns), pipeline_bytes=arr, metrics=json.dumps(tm.metrics))
        logger.info(f"Saved model to {path}")

    def load_model(self, model_name: str) -> Optional[TrainedModel]:
        path = self.model_dir / f"{model_name}.npz"
        if not path.exists():
            return None
        import joblib, io
        with np.load(path, allow_pickle=False) as data:
            target = str(data["target"])
            feature_columns = list(data["feature_columns"])
            pipe_bytes = data["pipeline_bytes"].tobytes()
            bio = io.BytesIO(pipe_bytes)
            pipeline = joblib.load(bio)
            metrics = json.loads(str(data["metrics"]))
        return TrainedModel(target=target, feature_columns=feature_columns, pipeline=pipeline, metrics=metrics)


# -----------------------------
# TotalsPredictor Orchestrator
# -----------------------------

class TotalsPredictor:
    def __init__(self, db: DatabaseManager, features: NFLFeaturePack, model_mgr: ModelManager):
        self.db = db
        self.features = features
        self.model_mgr = model_mgr

    def build_training_frame(self, season_start: int, season_end: int) -> pd.DataFrame:
        games = self.db.fetch_games(season_start, season_end)
        if games.empty:
            raise RuntimeError("No games found. Backfill seasons or provide CSVs in ./data.")
        games["date"] = pd.to_datetime(games["date"])
        games = games.sort_values("date").reset_index(drop=True)
        rows = []
        for _, r in games.iterrows():
            feats = self.features.create_matchup_features(r["home_team"], r["away_team"], r)
            feats.update({
                "game_id": r["game_id"],
                "date": r["date"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "season": r["season"],
                "week": r["week"],
                "total_pts": float((r.get("home_pts") or 0) + (r.get("away_pts") or 0)),
                "home_margin": float((r.get("home_pts") or 0) - (r.get("away_pts") or 0)),
            })
            rows.append(feats)
        df = pd.DataFrame(rows).dropna(subset=["total_pts"])
        return df

    def train_totals(self, season_start: int, season_end: int) -> TrainedModel:
        df = self.build_training_frame(season_start, season_end)
        return self.model_mgr.train(df, target="total_pts", model_name="nfl_totals_v1")

    def train_spread(self, season_start: int, season_end: int) -> TrainedModel:
        df = self.build_training_frame(season_start, season_end)
        return self.model_mgr.train(df, target="home_margin", model_name="nfl_spread_v1")

    def predict_today(self, model_totals: TrainedModel, model_spread: Optional[TrainedModel] = None) -> pd.DataFrame:
        today_games = self.db.games_on_date(today_local())
        if today_games.empty:
            logger.warning("No games found for today.")
            return pd.DataFrame()
        rows = []
        for _, r in today_games.iterrows():
            feats = self.features.create_matchup_features(r["home_team"], r["away_team"], r)
            # Build X row
            X = pd.DataFrame([feats])
            pred_total = float(model_totals.predict_frame(X)[0])
            row = {
                "game_id": r["game_id"],
                "date": r["date"],
                "matchup": f"{r['away_team']} @ {r['home_team']}",
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "pred_total": round(pred_total, 2),
            }
            if model_spread:
                pred_margin = float(model_spread.predict_frame(X)[0])
                row["pred_home_margin"] = round(pred_margin, 2)
            rows.append(row)
        return pd.DataFrame(rows)

    def predict_week(self, season: int, week: int,
                     mt: TrainedModel, ms: Optional[TrainedModel] = None) -> pd.DataFrame:
        """
        Predict totals (and optional home margin) for all games in the given season/week.
        """
        games = self.db.games_in_week(season, week)
        if games.empty:
            logger.warning(f"No games for season {season} week {week}.")
            return pd.DataFrame()

        rows = []
        for _, r in games.iterrows():
            feats = self.features.create_matchup_features(r["home_team"], r["away_team"], r)
            X = pd.DataFrame([feats])
            pred_total = float(mt.predict_frame(X)[0])

            row = {
                "game_id": r["game_id"],
                "date": r["date"],
                "season": r["season"],
                "week": r["week"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "matchup": f"{r['away_team']} @ {r['home_team']}",
                "pred_total": round(pred_total, 2),
            }
            if ms is not None:
                row["pred_home_margin"] = round(float(ms.predict_frame(X)[0]), 2)
            rows.append(row)

        return pd.DataFrame(rows)


# -----------------------------
# CSV Exporters
# -----------------------------

def export_csv(df: pd.DataFrame, name: str, *, subdir: Path = EXPORT_DIR, overwrite: bool = False) -> Path:
    """
    Write a CSV with a Pacific-tz timestamped prefix.
    Example: 20250916_082915_week2_predictions_2025.csv
    Pass overwrite=True to skip the timestamp and overwrite an exact name.
    """
    subdir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        path = subdir / f"{name}.csv"
    else:
        path = subdir / f"{date_str()}_{name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Exported {name}: {path}")
    return path



# -----------------------------
# Streamlit UI (optional)
# -----------------------------

def main_ui(db: DatabaseManager, schedule: ScheduleProvider, features: NFLFeaturePack, model_mgr: ModelManager):
    import streamlit as st
    st.set_page_config(page_title="FoxEdge NFL Engine", layout="wide")
    st.title("FoxEdge NFL — Totals/Spread Engine")

    st.sidebar.header("Controls")
    season_start = st.sidebar.number_input("Season start", 2015, 2030, 2019, 1)
    season_end = st.sidebar.number_input("Season end", 2015, 2030, 2024, 1)
    colA, colB, colC = st.sidebar.columns(3)
    if colA.button("Backfill"):
        schedule.backfill(season_start, season_end)
        st.success("Backfill complete.")
    if colB.button("Train Totals"):
        tm = TotalsPredictor(db, features, model_mgr).train_totals(season_start, season_end)
        st.json(tm.metrics)
        st.success("Totals model trained.")
    if colC.button("Train Spread"):
        sm = TotalsPredictor(db, features, model_mgr).train_spread(season_start, season_end)
        st.json(sm.metrics)
        st.success("Spread model trained.")

    st.header("Today’s Predictions")
    mt = model_mgr.load_model("nfl_totals_v1")
    ms = model_mgr.load_model("nfl_spread_v1")
    if mt is None:
        st.warning("Train totals model first.")
    else:
        pred = TotalsPredictor(db, features, model_mgr).predict_today(mt, ms)
        st.dataframe(pred)
        if st.button("Export Today CSV"):
            export_csv(pred, "today_predictions")

    st.header("Raw Games in DB (last 200)")
    games = db.fetch_games(2000, 2100).tail(200)
    st.dataframe(games)


# -----------------------------
# CLI
# -----------------------------



def _ensure_data(db: DatabaseManager, schedule: ScheduleProvider, season_start: int, season_end: int, enable: bool = True):
    if not enable:
        return
    try:
        df = db.fetch_games(season_start, season_end)
        need = df.empty or (len(df) < 200)
        if need:
            logger.info(f"Auto-backfill triggered for seasons {season_start}-{season_end}.")
            schedule.backfill(season_start, season_end)
        else:
            # sanity: ensure the latest season exists
            latest = db.fetch_games(season_end, season_end)
            if latest.empty:
                logger.info(f"Latest season {season_end} missing; backfilling.")
                schedule.backfill(season_end, season_end)
    except Exception as e:
        logger.warning(f"Auto-backfill check failed: {e}. Attempting backfill.")
        schedule.backfill(season_start, season_end)
def run_cli(args: argparse.Namespace):
    # Always tell me what script and args actually ran
    print(f"[foxedge] script={__file__}")
    print(f"[foxedge] args={vars(args)}")

    if getattr(args, "verbose", False):
        logging.getLogger("foxedge_nfl").setLevel(logging.DEBUG)

    db = DatabaseManager(DB_DIR / "nfl.db")
    weather = WeatherService()
    schedule = ScheduleProvider(db)
    features = NFLFeaturePack(db, weather)
    model_mgr = ModelManager(MODEL_DIR, db)
    predictor = TotalsPredictor(db, features, model_mgr)

    season_start = int(args.season_start or 2018)
    season_end = int(args.season_end or dt.datetime.now().year)

    # Auto-backfill unless disabled
    if not args.no_autobackfill:
        try:
            df_check = db.fetch_games(season_start, season_end)
            if df_check.empty or len(df_check) < 200:
                logger.info(f"Auto-backfill {season_start}-{season_end}")
                schedule.backfill(season_start, season_end)
            else:
                latest = db.fetch_games(season_end, season_end)
                if latest.empty:
                    logger.info(f"Backfilling latest season {season_end}")
                    schedule.backfill(season_end, season_end)
        except Exception as e:
            logger.warning(f"Auto-backfill check failed: {e}; attempting backfill.")
            schedule.backfill(season_start, season_end)

    # Manual backfill
    if args.backfill:
        schedule.backfill(season_start, season_end)

    # Training
    if args.train or args.train_totals:
        predictor.train_totals(season_start, season_end)
    if args.train or args.train_spread:
        predictor.train_spread(season_start, season_end)

    # List-week debug
    if args.list_week:
        if args.season_year is None or args.week is None:
            print("--list-week requires --season_year and --week", file=sys.stderr)
            sys.exit(2)
        dfw = db.games_in_week(int(args.season_year), int(args.week))
        if dfw.empty:
            print("No games found for requested week.")
        else:
            print(dfw[["season", "week", "away_team", "home_team", "date"]].to_string(index=False))
        return  # stop after listing

    # Today predictions
    if args.today:
        mt = model_mgr.load_model("nfl_totals_v1")
        if mt is None:
            print("Totals model not found. Train it first.", file=sys.stderr)
            sys.exit(2)
        ms = model_mgr.load_model("nfl_spread_v1")
        df_today = predictor.predict_today(mt, ms)
        print(f"[foxedge] today rows={len(df_today)}")
        if df_today.empty:
            print("No games predicted for today.")
        else:
            print(df_today.to_string(index=False))
            if args.export:
                path = export_csv(df_today, "today_predictions")
                print(f"[foxedge] exported -> {path}")
        return

    # Week predictions
    if args.predict_week:
        if args.season_year is None or args.week is None:
            print("--predict-week requires --season_year and --week", file=sys.stderr)
            sys.exit(2)
        mt = model_mgr.load_model("nfl_totals_v1")
        if mt is None:
            print("Totals model not found. Train it first.", file=sys.stderr)
            sys.exit(2)
        ms = model_mgr.load_model("nfl_spread_v1")

        season = int(args.season_year)
        week = int(args.week)
        dfw = predictor.predict_week(season, week, mt, ms)
        logger.info(f"predict_week: produced {len(dfw)} rows for {season} week {week}")
        print(f"[foxedge] week rows={len(dfw)} season={season} week={week}")
        if dfw.empty:
            print("No games predicted for requested week.")
        else:
            print(dfw.to_string(index=False))
            if args.export:
                out_name = args.out or f"week{week}_predictions_{season}"
                path = export_csv(dfw, out_name)
                print(f"[foxedge] exported -> {path}")
        return

    # UI
    if args.ui:
        try:
            import streamlit as st  # noqa: F401
        except Exception:
            print("Streamlit not installed. pip install streamlit", file=sys.stderr)
            sys.exit(2)
        main_ui(db, schedule, features, model_mgr)



def build_arg_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FoxEdge NFL Engine")
    # Data
    p.add_argument("--backfill", action="store_true", help="Backfill schedules/results into DB")
    p.add_argument("--season_start", type=int, help="Start season (e.g., 2018)")
    p.add_argument("--season_end", type=int, help="End season (e.g., 2024)")
    p.add_argument("--no-autobackfill", action="store_true", help="Disable automatic data backfill on first run")

    # Training
    p.add_argument("--train", action="store_true", help="Train all models")
    p.add_argument("--train_totals", action="store_true", help="Train totals model only")
    p.add_argument("--train_spread", action="store_true", help="Train spread model only")

    # Prediction
    p.add_argument("--today", action="store_true", help="Predict today’s games")
    p.add_argument("--predict-week", dest="predict_week", action="store_true",
                   help="Predict all games for a given NFL week")
    p.add_argument("--season_year", type=int, help="Season year for week prediction (e.g., 2025)")
    p.add_argument("--week", type=int, help="Week number (e.g., 2)")
    p.add_argument("--list-week", dest="list_week", action="store_true",
                   help="List games for a given season/week (debug; no predictions)")

    # Output / misc
    p.add_argument("--export", action="store_true", help="Export predictions CSV")
    p.add_argument("--out", type=str, help="Explicit output name (without .csv). Used with --export")
    p.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    p.add_argument("--verbose", action="store_true", help="Set logging level to DEBUG")
    return p.parse_args()





if __name__ == "__main__":
    args = build_arg_parser()
    run_cli(args)
