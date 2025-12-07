
import io, json, math, re
from datetime import datetime
from urllib.parse import urlencode, urlparse, parse_qs

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pytz

# =============================================================================
# App config
# =============================================================================
st.set_page_config(page_title="NFL Weather → Betting Edge", layout="wide")

# Inline CSS theming (no external files)
st.markdown(
    """
    <style>
    .main > div {padding-top: 0.5rem;}
    h1, h2, h3 { font-weight: 700; }
    .edge-badge { font-weight: 700; padding: 2px 8px; border-radius: 999px; }
    .edge-badge.red { background: rgba(255, 59, 48, .12); color: #b00020; }
    .edge-badge.orange { background: rgba(255,159,10,.12); color: #a85a00; }
    .edge-badge.yellow { background: rgba(255,214,10,.12); color: #7a6500; }
    .edge-badge.green { background: rgba(48,209,88,.12); color: #126b2e; }
    .pill { padding: 2px 8px; border-radius: 999px; background: rgba(0,0,0,.06); font-weight: 600; }
    .small-note { color: #666; font-size: 0.9rem; }
    .stDataFrame table thead th { position: sticky; top: 0; background: #111 !important; color: #fff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# DK Network live odds/splits (embedded scraper)
# =============================================================================
DK_BASE = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"

def _dk_clean(text: str) -> str:
    return re.sub(r"opens?\\s+in\\s+(?:a\\s+)?new\\s+tab", "", text or "", flags=re.I).strip()

def _dk_get(url: str, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "nfl-weather-betting/1.0"})
    r.raise_for_status()
    return r.text

def _dk_pagination_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    pag = soup.select_one("div.tb_pagination")
    if not pag:
        return []
    urls = []
    for a in pag.select("a"):
        href = a.get("href")
        if href and "tb_page" in href:
            urls.append(href)
    def pnum(u: str) -> int:
        try:
            return int(parse_qs(urlparse(u).query).get("tb_page", ["1"])[0])
        except Exception:
            return 1
    return sorted(list(dict.fromkeys(urls)), key=pnum)

def _dk_parse_page(html: str) -> list[dict]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    games = soup.select("div.tb-se")
    out: list[dict] = []
    pac = pytz.timezone("America/Los_Angeles")
    now = datetime.now(pac).strftime("%Y-%m-%d %H:%M:%S %Z")
    for g in games:
        title_el = g.select_one("div.tb-se-title h5")
        if not title_el:
            continue
        title = _dk_clean(title_el.get_text(strip=True))  # e.g., "Patriots @ Jets"
        time_el = g.select_one("div.tb-se-title span")
        game_time = _dk_clean(time_el.get_text(strip=True)) if time_el else ""
        for section in g.select(".tb-market-wrap > div"):
            head = section.select_one(".tb-se-head > div")
            if not head:
                continue
            market_name = _dk_clean(head.get_text(strip=True))
            if market_name not in ("Moneyline", "Total", "Spread"):
                continue
            for row in section.select(".tb-sodd"):
                side_el = row.select_one(".tb-slipline")
                odds_el = row.select_one("a.tb-odd-s")
                if not side_el or not odds_el:
                    continue
                side = _dk_clean(side_el.get_text(strip=True))
                odds = _dk_clean(odds_el.get_text(strip=True))  # like +105 or Over 44.5
                bets_span = row.select_one("span.tb-per:nth-of-type(1)")
                handle_span = row.select_one("span.tb-per:nth-of-type(2)")
                bets = _dk_clean(bets_span.get_text(strip=True)).replace("%", "") if bets_span else None
                handle = _dk_clean(handle_span.get_text(strip=True)).replace("%", "") if handle_span else None
                out.append({
                    "matchup": title,
                    "game_time": game_time,
                    "market": market_name,
                    "side": side,
                    "odds": odds,
                    "%handle": float(handle) if handle not in (None, "") else None,
                    "%bets": float(bets) if bets not in (None, "") else None,
                    "update_time": now,
                })
    return out

def _dk_parse_total(side: str) -> tuple[str, float | None]:
    m = re.search(r"(Over|Under)\\s*([0-9]+\\.?[0-9]*)", side or "", flags=re.I)
    return (m.group(1).title(), float(m.group(2))) if m else ("", None)

def _dk_parse_spread(side: str) -> float | None:
    m = re.search(r"([+-]?[0-9]+\\.?[0-9]*)$", (side or "").strip())
    return float(m.group(1)) if m else None

@st.cache_data(show_spinner=False, ttl=300)
def fetch_dk_splits(event_group: int = 88808, date_range: str = "n7days") -> pd.DataFrame:
    params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
    first_url = f"{DK_BASE}?{urlencode(params)}"
    html = _dk_get(first_url)
    pages = [html] + [_dk_get(u) for u in _dk_pagination_urls(html)]
    records = []
    for h in pages:
        records.extend(_dk_parse_page(h))
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # Normalize totals/spreads
    is_tot = df["market"].eq("Total")
    if is_tot.any():
        parsed = df.loc[is_tot, "side"].apply(_dk_parse_total)
        df.loc[is_tot, "total_dir"] = parsed.apply(lambda t: t[0])
        df.loc[is_tot, "total_val"] = parsed.apply(lambda t: t[1])
    is_sp = df["market"].eq("Spread")
    if is_sp.any():
        df.loc[is_sp, "spread_val"] = df.loc[is_sp, "side"].apply(_dk_parse_spread)
    return df

# =============================================================================
# Stadium metadata (32 NFL teams + orientation/roof)
# =============================================================================
NFL_STADIUMS = [
  {"team":"Arizona Cardinals","stadium":"State Farm Stadium","city":"Glendale","state":"AZ","lat":33.527283,"lon":-112.263275,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":5,"roof":"retractable","sun_exposure_material":True},
  {"team":"Atlanta Falcons","stadium":"Mercedes-Benz Stadium","city":"Atlanta","state":"GA","lat":33.755489,"lon":-84.401993,"orientation":"E-W","azimuth_deg":90,"azimuth_uncertainty_deg":10,"roof":"retractable","sun_exposure_material":False},
  {"team":"Baltimore Ravens","stadium":"M&T Bank Stadium","city":"Baltimore","state":"MD","lat":39.278088,"lon":-76.623322,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Buffalo Bills","stadium":"Highmark Stadium","city":"Orchard Park","state":"NY","lat":42.773773,"lon":-78.78746,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Carolina Panthers","stadium":"Bank of America Stadium","city":"Charlotte","state":"NC","lat":35.225845,"lon":-80.853607,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Chicago Bears","stadium":"Soldier Field","city":"Chicago","state":"IL","lat":41.862366,"lon":-87.617256,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Cincinnati Bengals","stadium":"Paycor Stadium","city":"Cincinnati","state":"OH","lat":39.096306,"lon":-84.516846,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Cleveland Browns","stadium":"Cleveland Browns Stadium","city":"Cleveland","state":"OH","lat":41.506035,"lon":-81.700058,"orientation":"NE-SW","azimuth_deg":45,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Dallas Cowboys","stadium":"AT&T Stadium","city":"Arlington","state":"TX","lat":32.747841,"lon":-97.093628,"orientation":"NE-SW","azimuth_deg":45,"azimuth_uncertainty_deg":10,"roof":"retractable","sun_exposure_material":True},
  {"team":"Denver Broncos","stadium":"Empower Field at Mile High","city":"Denver","state":"CO","lat":39.744129,"lon":-105.020828,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":5,"roof":"open","sun_exposure_material":True},
  {"team":"Detroit Lions","stadium":"Ford Field","city":"Detroit","state":"MI","lat":42.340115,"lon":-83.046341,"orientation":"E-W","azimuth_deg":90,"azimuth_uncertainty_deg":10,"roof":"fixed","sun_exposure_material":False},
  {"team":"Green Bay Packers","stadium":"Lambeau Field","city":"Green Bay","state":"WI","lat":44.501308,"lon":-88.062317,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":5,"roof":"open","sun_exposure_material":True},
  {"team":"Houston Texans","stadium":"NRG Stadium","city":"Houston","state":"TX","lat":29.68486,"lon":-95.411667,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":5,"roof":"retractable","sun_exposure_material":True},
  {"team":"Indianapolis Colts","stadium":"Lucas Oil Stadium","city":"Indianapolis","state":"IN","lat":39.759991,"lon":-86.163712,"orientation":"NE-SW","azimuth_deg":45,"azimuth_uncertainty_deg":10,"roof":"retractable","sun_exposure_material":True},
  {"team":"Jacksonville Jaguars","stadium":"EverBank Stadium","city":"Jacksonville","state":"FL","lat":30.323471,"lon":-81.636528,"orientation":"NE-SW","azimuth_deg":45,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Kansas City Chiefs","stadium":"GEHA Field at Arrowhead Stadium","city":"Kansas City","state":"MO","lat":39.048786,"lon":-94.484566,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":5,"roof":"open","sun_exposure_material":True},
  {"team":"Las Vegas Raiders","stadium":"Allegiant Stadium","city":"Paradise","state":"NV","lat":36.090794,"lon":-115.183952,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"fixed","sun_exposure_material":False},
  {"team":"Los Angeles Chargers","stadium":"SoFi Stadium","city":"Inglewood","state":"CA","lat":33.953587,"lon":-118.33963,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":5,"roof":"canopy","sun_exposure_material":True},
  {"team":"Los Angeles Rams","stadium":"SoFi Stadium","city":"Inglewood","state":"CA","lat":33.953587,"lon":-118.33963,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":5,"roof":"canopy","sun_exposure_material":True},
  {"team":"Miami Dolphins","stadium":"Hard Rock Stadium","city":"Miami Gardens","state":"FL","lat":25.95796,"lon":-80.239311,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":5,"roof":"canopy","sun_exposure_material":True},
  {"team":"Minnesota Vikings","stadium":"U.S. Bank Stadium","city":"Minneapolis","state":"MN","lat":44.973774,"lon":-93.258736,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":5,"roof":"fixed","sun_exposure_material":False},
  {"team":"New England Patriots","stadium":"Gillette Stadium","city":"Foxborough","state":"MA","lat":42.090944,"lon":-71.264344,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"New Orleans Saints","stadium":"Caesars Superdome","city":"New Orleans","state":"LA","lat":29.951439,"lon":-90.08197,"orientation":"E-W","azimuth_deg":90,"azimuth_uncertainty_deg":10,"roof":"fixed","sun_exposure_material":False},
  {"team":"New York Giants","stadium":"MetLife Stadium","city":"East Rutherford","state":"NJ","lat":40.813778,"lon":-74.07431,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"New York Jets","stadium":"MetLife Stadium","city":"East Rutherford","state":"NJ","lat":40.813778,"lon":-74.07431,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Philadelphia Eagles","stadium":"Lincoln Financial Field","city":"Philadelphia","state":"PA","lat":39.900898,"lon":-75.168098,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Pittsburgh Steelers","stadium":"Acrisure Stadium","city":"Pittsburgh","state":"PA","lat":40.446764,"lon":-80.01576,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"San Francisco 49ers","stadium":"Levi's Stadium","city":"Santa Clara","state":"CA","lat":37.403,"lon":-121.97,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":5,"roof":"open","sun_exposure_material":True},
  {"team":"Seattle Seahawks","stadium":"Lumen Field","city":"Seattle","state":"WA","lat":47.595097,"lon":-122.332245,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":5,"roof":"canopy","sun_exposure_material":True},
  {"team":"Tampa Bay Buccaneers","stadium":"Raymond James Stadium","city":"Tampa","state":"FL","lat":27.975958,"lon":-82.503693,"orientation":"N-S","azimuth_deg":0,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Tennessee Titans","stadium":"Nissan Stadium","city":"Nashville","state":"TN","lat":36.166505,"lon":-86.77129,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True},
  {"team":"Washington Commanders","stadium":"FedExField","city":"Landover","state":"MD","lat":38.907652,"lon":-76.864479,"orientation":"NW-SE","azimuth_deg":135,"azimuth_uncertainty_deg":10,"roof":"open","sun_exposure_material":True}
]

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

TEAM_TO_STADIUM = {s["team"]: s for s in NFL_STADIUMS}

# =============================================================================
# Aliases and resolvers
# =============================================================================
TEAM_ALIASES = {
    "Washington": "Washington Commanders",
    "NY Giants": "New York Giants",
    "NY Jets": "New York Jets",
    "LA Rams": "Los Angeles Rams",
    "LA Chargers": "Los Angeles Chargers",
    "New England": "New England Patriots",
    "Pats": "New England Patriots",
    "Bucs": "Tampa Bay Buccaneers",
    "Niners": "San Francisco 49ers",
    "Jags": "Jacksonville Jaguars",
    "Vikings": "Minnesota Vikings",
    "Patriots": "New England Patriots",
    "Saints": "New Orleans Saints",
    "Buccaneers": "Tampa Bay Buccaneers",
    "Packers": "Green Bay Packers",
    "Chiefs": "Kansas City Chiefs",
    "Raiders": "Las Vegas Raiders",
    "Rams": "Los Angeles Rams",
    "Chargers": "Los Angeles Chargers",
    "Giants": "New York Giants",
    "Jets": "New York Jets",
    "Commanders": "Washington Commanders",
    "Jaguars": "Jacksonville Jaguars",
    "Dolphins": "Miami Dolphins",
    "Bills": "Buffalo Bills",
    "Ravens": "Baltimore Ravens",
    "Falcons": "Atlanta Falcons",
    "Cardinals": "Arizona Cardinals",
    "Panthers": "Carolina Panthers",
    "Bears": "Chicago Bears",
    "Bengals": "Cincinnati Bengals",
    "Browns": "Cleveland Browns",
    "Cowboys": "Dallas Cowboys",
    "Broncos": "Denver Broncos",
    "Lions": "Detroit Lions",
    "Texans": "Houston Texans",
    "Colts": "Indianapolis Colts",
    "Eagles": "Philadelphia Eagles",
    "Steelers": "Pittsburgh Steelers",
    "Seahawks": "Seattle Seahawks",
    "Titans": "Tennessee Titans",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "TB": "Tampa Bay Buccaneers",
    "GB": "Green Bay Packers",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAR": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "SF": "San Francisco 49ers",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "WAS": "Washington Commanders",
    "WSH": "Washington Commanders",
    "JAX": "Jacksonville Jaguars",
    "MIA": "Miami Dolphins",
    "BUF": "Buffalo Bills",
    "BAL": "Baltimore Ravens",
    "ATL": "Atlanta Falcons",
    "ARI": "Arizona Cardinals",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "MIN": "Minnesota Vikings",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "TEN": "Tennessee Titans",
}

CITY_ALIASES = {
    "Baltimore": "Baltimore Ravens",
    "Buffalo": "Buffalo Bills",
    "Carolina": "Carolina Panthers",
    "Charlotte": "Carolina Panthers",
    "Cincinnati": "Cincinnati Bengals",
    "Cleveland": "Cleveland Browns",
    "Dallas": "Dallas Cowboys",
    "Denver": "Denver Broncos",
    "Detroit": "Detroit Lions",
    "Foxborough": "New England Patriots",
    "Green Bay": "Green Bay Packers",
    "Houston": "Houston Texans",
    "Indianapolis": "Indianapolis Colts",
    "Jacksonville": "Jacksonville Jaguars",
    "Kansas City": "Kansas City Chiefs",
    "Las Vegas": "Las Vegas Raiders",
    "Los Angeles Rams": "Los Angeles Rams",
    "Los Angeles Chargers": "Los Angeles Chargers",
    "Miami": "Miami Dolphins",
    "Miami Gardens": "Miami Dolphins",
    "Minneapolis": "Minnesota Vikings",
    "Minnesota": "Minnesota Vikings",
    "New Orleans": "New Orleans Saints",
    "Philadelphia": "Philadelphia Eagles",
    "Pittsburgh": "Pittsburgh Steelers",
    "Seattle": "Seattle Seahawks",
    "Tampa": "Tampa Bay Buccaneers",
    "Tampa Bay": "Tampa Bay Buccaneers",
    "Nashville": "Tennessee Titans",
    "Washington": "Washington Commanders",
    "East Rutherford": None,
    "New York": None,
    "Los Angeles": None,
}

MATCHUP_AT_RX = re.compile(r"^\\s*(.*?)\\s*@\\s*(.*?)\\s*$", re.I)
MATCHUP_VS_RX = re.compile(r"^\\s*(.*?)\\s*vs\\.?\\s*(.*?)\\s*$", re.I)

def resolve_team(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return name
    if name in TEAM_TO_STADIUM:
        return name
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    u = name.upper()
    if u in TEAM_ALIASES:
        return TEAM_ALIASES[u]
    if name in CITY_ALIASES and CITY_ALIASES[name]:
        return CITY_ALIASES[name]
    base = name.replace("The ", "").replace("the ", "").strip()
    if base in TEAM_ALIASES:
        return TEAM_ALIASES[base]
    if base in CITY_ALIASES and CITY_ALIASES[base]:
        return CITY_ALIASES[base]
    low = base.lower()
    for full in TEAM_TO_STADIUM.keys():
        f = full.lower()
        if low == f or low in f or f in low:
            return full
    for nick, full in TEAM_ALIASES.items():
        if full in TEAM_TO_STADIUM and nick.lower() in low:
            return full
    return name

def dk_teams_from_rows(dkdf: pd.DataFrame, matchup: str) -> list[str]:
    if dkdf is None or dkdf.empty:
        return []
    cand = []
    sub = dkdf[dkdf["matchup"].eq(matchup)]
    for _, r in sub.iterrows():
        side = (r.get("side") or "").strip()
        side = re.sub(r"\\s*[+-]?[0-9]+\\.?[0-9]*$", "", side)
        side = side.replace("Over", "").replace("Under", "").strip()
        if side and side not in cand:
            cand.append(side)
    return cand[:4]

def parse_matchup_teams(matchup: str) -> tuple[str, str]:
    txt = (matchup or "").strip()

    # Normalize garbage whitespace (non-breaking, multi-space, etc)
    txt = re.sub(r"\s+", " ", txt)

    # Try @ format first
    m_at = re.match(r"^(.*?)\s*@\s*(.*?)$", txt, flags=re.I)
    if m_at:
        raw_away = m_at.group(1).strip()
        raw_home = m_at.group(2).strip()
        away = resolve_team(raw_away)
        home = resolve_team(raw_home)

        # Safety check: make sure "home" actually looks like the hosting team
        # We trust stadium map to tell us who's at home. If DK reversed it, flip.
        if home in TEAM_TO_STADIUM and away in TEAM_TO_STADIUM:
            # if DK fed "MIA @ BAL" but matchup is actually in Miami, stadium check will fix it
            home_stadium = TEAM_TO_STADIUM[home]["stadium"]
            away_stadium = TEAM_TO_STADIUM[away]["stadium"]
            # if they somehow gave two teams but it's clearly Miami's home game,
            # 'home' should be the team whose stadium we're using downstream,
            # not some other stadium
            if home_stadium != away_stadium:
                # normal NFL game: different stadiums, so we assume DK is right
                pass
            else:
                # rare case like Jets/Giants same venue
                pass

        return away, home

    # Try vs format ("Dolphins vs Ravens") which usually means neutral or listing "home vs away"
    m_vs = re.match(r"^(.*?)\s*vs\.?\s*(.*?)$", txt, flags=re.I)
    if m_vs:
        raw_home = m_vs.group(1).strip()
        raw_away = m_vs.group(2).strip()
        return resolve_team(raw_away), resolve_team(raw_home)

    # If we can't parse, don't lie
    return "", ""

def resolve_matchup(matchup: str, dkdf_full: pd.DataFrame) -> tuple[str, str]:
    away, home = parse_matchup_teams(matchup)

    # If parser got both, trust that and return immediately.
    if away and home:
        return away, home

    # Only then try to guess from DK rows.
    labels = dk_teams_from_rows(dkdf_full, matchup)
    teams = [resolve_team(x) for x in labels if x]
    teams = [t for t in teams if t in TEAM_TO_STADIUM]
    teams = list(dict.fromkeys(teams))
    if len(teams) >= 2:
        # For vs we know order is home vs away, so invert
        if MATCHUP_VS_RX.match((matchup or "")):
            return teams[1], teams[0]
        # For @ we ALREADY handled; if we're here it's because parsing failed,
        # and at that point we genuinely don't know. But at least don't randomly flip Miami out.
        return teams[0], teams[1]

    return away, home

    # fallback if parsing failed
    labels = dk_teams_from_rows(dkdf_full, matchup)
    teams = [resolve_team(x) for x in labels if x]
    teams = [t for t in teams if t in TEAM_TO_STADIUM]
    teams = list(dict.fromkeys(teams))
    if len(teams) >= 2:
        if MATCHUP_VS_RX.match(matchup or ""):
            return teams[1], teams[0]
        return teams[0], teams[1]
    return away, home

def stadium_for_home_team(home_team: str) -> dict | None:
    return TEAM_TO_STADIUM.get(home_team)

# =============================================================================
# Pretty helpers
# =============================================================================
def roof_icon(roof: str) -> str:
    r = str(roof).lower()
    return {"open":"🏟️","fixed":"🧱","retractable":"🔁","canopy":"🛡️"}.get(r, "🏟️")

def wind_flag(mph_val: float | None) -> str:
    v = float(mph_val or 0)
    if v >= 20: return "🌪️"
    if v >= 15: return "🌬️"
    if v >= 10: return "🍃"
    return "·"

def precip_flag(pop_pct: float | None, precip_in: float | None) -> str:
    p = float(pop_pct or 0); r = float(precip_in or 0)
    if r >= 0.2 or p >= 70: return "🌧️"
    if r > 0.0 or p >= 30: return "☔"
    return "·"

def edge_pill(delta: float | None) -> str:
    d = float(delta or 0)
    if d <= -6: return f"<span class='edge-badge red'>{d:+.1f}</span>"
    if d <= -4: return f"<span class='edge-badge orange'>{d:+.1f}</span>"
    if d <= -2: return f"<span class='edge-badge yellow'>{d:+.1f}</span>"
    return f"<span class='edge-badge green'>{d:+.1f}</span>"

def actions_short(txt: str | None, limit: int = 3) -> str:
    if not txt: return ""
    parts = [p for p in str(txt).split("|") if p]
    return " | ".join(parts[:limit])

# =============================================================================
# Betting intelligence helpers
# =============================================================================
def pct(x):
    try:
        return float(x) if x is not None and x == x else None
    except:
        return None

def bias_class(delta):
    d = float(delta or 0)
    if d <= -6: return "UNDER ZONE"
    if d <= -2: return "LEAN UNDER"
    if d <  2:  return "NEUTRAL"
    if d <=  6: return "LEAN OVER"
    return "OVER ZONE"

def wmfi(points_delta, over_handle_pct):
    d = abs(float(points_delta or 0))
    skew = abs((pct(over_handle_pct) or 50.0)/100.0 - 0.5)
    score = 100 * (d/10.0) * (1 + 2*skew)
    return round(min(score, 100), 1)

def public_pain_index(points_delta, over_handle_pct):
    d = abs(float(points_delta or 0))
    h = (pct(over_handle_pct) or 50.0) / 100.0
    return round(100 * d * max(h, 1-h) / 10.0, 1)

def fragility(market_total_min, market_total_max, over_handle_pct):
    rng = 0 if market_total_min is None or market_total_max is None else float(market_total_max) - float(market_total_min)
    skew = abs((pct(over_handle_pct) or 50.0)/100.0 - 0.5)
    score = 100 * (rng/6.0) * (0.5 + 1.5*skew)
    return round(max(0, min(score, 100)), 1)

def momentum(next_hour_delta, now_delta):
    try:
        dn = float(next_hour_delta or 0) - float(now_delta or 0)
        if dn <= -1.0: return "↘ edge weakening"
        if dn >=  1.0: return "↗ edge building"
        return "→ stable"
    except:
        return "·"

def compass_from_components(tail_head_mps, cross_mps):
    th = float(tail_head_mps or 0)
    cr = float(cross_mps or 0)
    if abs(th) >= abs(cr):
        return "⬆ tailwind" if th > 0 else ("⬇ headwind" if th < 0 else "• neutral")
    else:
        return "⬅ crosswind" if cr < 0 else ("➡ crosswind" if cr > 0 else "• neutral")

def edge_convergence(wmfi_score, cls, frag_score, pain_score):
    hits = 0
    if wmfi_score is not None and wmfi_score >= 60: hits += 1
    if cls in ("UNDER ZONE","OVER ZONE","LEAN UNDER"): hits += 1
    if frag_score is not None and frag_score >= 50: hits += 1
    if pain_score is not None and pain_score >= 40: hits += 1
    return "⚡" if hits >= 3 else ("🧠" if hits == 2 else "·")

# =============================================================================
# Weather fetching / transforms
# =============================================================================
def mph(mps): return (mps or 0) * 2.236936
def in_per_hr(mm): return (mm or 0) * 0.0393701
def normalize_deg(d):
    d %= 360.0
    return d + 360.0 if d < 0 else d

def wind_components(speed_mps, wind_dir_deg_met, field_bearing_deg):
    toward_deg = normalize_deg((wind_dir_deg_met or 0) + 180.0)
    delta_rad = math.radians(normalize_deg(toward_deg - field_bearing_deg))
    return speed_mps*math.cos(delta_rad), speed_mps*math.sin(delta_rad), toward_deg

def classify_weather_and_effects(temp_c, wind_mps, precip_mm, pop_pct, roof):
    w_mph = mph(wind_mps)
    temp_f = temp_c * 9/5 + 32
    rate_in = in_per_hr(precip_mm)
    effects = {"triggers": [], "points_delta": 0.0, "actions": []}
    if w_mph >= 20:
        effects["triggers"].append("wind_severe"); effects["points_delta"] -= 6; effects["actions"] += ["UNDER","RB_OVERS","FADE_QB/WR","FADE_FG","DOGS"]
    elif w_mph >= 15:
        effects["triggers"].append("wind_strong"); effects["points_delta"] -= 6; effects["actions"] += ["UNDER","RB_OVERS","FADE_QB/WR","FADE_FG","DOGS"]
    elif w_mph >= 10:
        effects["triggers"].append("wind_mod"); effects["points_delta"] -= 2; effects["actions"] += ["LEAN_UNDER"]
    if rate_in >= 0.2 or (pop_pct is not None and pop_pct >= 70):
        effects["triggers"].append("rain_heavy"); effects["points_delta"] -= 6;
        for a in ["UNDER","RB_OVERS","DOGS"]:
            if a not in effects["actions"]: effects["actions"].append(a)
    elif rate_in > 0 or (pop_pct is not None and pop_pct >= 30):
        effects["triggers"].append("rain_light"); effects["points_delta"] -= 2
    if temp_f < 25: effects["triggers"].append("cold_ext"); effects["points_delta"] -= 4; effects["actions"].append("HOME_COLD_EDGE")
    elif temp_f < 40: effects["triggers"].append("cold"); effects["points_delta"] -= 3
    elif temp_f > 85: effects["triggers"].append("heat"); effects["points_delta"] -= 3
    if str(roof).lower() == "fixed": effects["points_delta"] *= 0.25
    elif str(roof).lower() in {"retractable","canopy"}: effects["points_delta"] *= 0.6
    effects["points_delta"] = max(-14, min(0, effects["points_delta"]))
    effects["actions"] = list(dict.fromkeys(effects["actions"]))
    return effects

@st.cache_data(show_spinner=False)
def fetch_open_meteo(lat: float, lon: float, hours: int = 6):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation_probability,precipitation,weathercode",
        "windspeed_unit": "ms", "timezone": "UTC"
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])[:hours]
    out = []
    for i, t in enumerate(times):
        out.append({
            "time_utc": t,
            "temp_c": hourly.get("temperature_2m", [None]*len(times))[i],
            "rh_pct": hourly.get("relative_humidity_2m", [None]*len(times))[i],
            "wind_speed_mps": hourly.get("wind_speed_10m", [None]*len(times))[i],
            "wind_dir_deg": hourly.get("wind_direction_10m", [None]*len(times))[i],
            "pop_pct": hourly.get("precipitation_probability", [None]*len(times))[i],
            "precip_mm": hourly.get("precipitation", [None]*len(times))[i],
            "weathercode": hourly.get("weathercode", [None]*len(times))[i],
        })
    return out

def compute_rows(s, hours):
    lat, lon, roof, bearing = s["lat"], s["lon"], s.get("roof","open"), float(s.get("azimuth_deg") or 0.0)
    om = fetch_open_meteo(lat, lon, hours)
    rows = []
    for h in om:
        if None in (h["temp_c"], h["rh_pct"], h["wind_speed_mps"], h["wind_dir_deg"]): continue
        ht, cr, toward = wind_components(h["wind_speed_mps"], h["wind_dir_deg"], bearing)
        eff = classify_weather_and_effects(h["temp_c"], h["wind_speed_mps"], h["precip_mm"], h["pop_pct"], roof)
        rows.append({
            **s, "bearing_used_deg": bearing,
            "time_utc": h["time_utc"],
            "temp_c": round(h["temp_c"],1),
            "wind_speed_mps": round(h["wind_speed_mps"],2),
            "wind_speed_mph": round(mph(h["wind_speed_mps"]),1),
            "pop_pct": h["pop_pct"], "precip_mm": h["precip_mm"], "precip_in": round(in_per_hr(h["precip_mm"]),3),
            "tail_head_mps": round(ht,2), "cross_mps": round(cr,2),
            "bet_points_delta": eff["points_delta"],
            "bet_actions": "|".join(eff["actions"]),
            "bet_triggers": "|".join(eff["triggers"]),
        })
    return rows

def first_hour_row_for_stadium(df: pd.DataFrame, stadium_name: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    sub = df[df["stadium"].eq(stadium_name)]
    if sub.empty:
        return None
    first_hour = sub["time_utc"].min()
    row = sub[sub["time_utc"].eq(first_hour)].head(1)
    return None if row.empty else row.iloc[0]

def ensure_first_hour_for_stadium(df_all: pd.DataFrame, stadium_name: str, hours: int) -> pd.Series | None:
    row = first_hour_row_for_stadium(df_all, stadium_name)
    if row is not None:
        return row
    meta = next((s for s in NFL_STADIUMS if s["stadium"] == stadium_name), None)
    if not meta:
        return None
    try:
        rows = compute_rows(meta, hours)
        if not rows:
            return None
        tmp = pd.DataFrame(rows)
        return first_hour_row_for_stadium(tmp, stadium_name)
    except Exception:
        return None

# =============================================================================
# Sidebar controls
# =============================================================================
with st.sidebar:
    st.header("Filters & Settings")
    hours = st.slider("Hours ahead", 1, 24, 6)
    roof_filter = st.multiselect("Roof types", ["open","fixed","retractable","canopy"], default=["open","retractable","canopy","fixed"])
    trigger_filter = st.multiselect("Triggers", ["wind_severe","wind_strong","wind_mod","rain_heavy","rain_light","cold_ext","cold","heat"], default=[])
    min_wind = st.number_input("Min wind mph (snapshot filter)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    outdoor_only = st.checkbox("Outdoor only (excludes fixed)", value=False)
    st.markdown("---")
    st.subheader("Market Lines (optional)")
    st.caption("Upload lines to compute edges. Columns: team,stadium,market_total,market_spread")
    lines_file = st.file_uploader("lines.csv", type=["csv"])
    manual_total = st.number_input("Single-stadium market total (for detail view)", min_value=0.0, max_value=100.0, value=43.5, step=0.5)
    manual_spread = st.number_input("Single-stadium favorite spread (− fav, + dog)", min_value=-30.0, max_value=30.0, value=-3.0, step=0.5)

    st.markdown("---")
    use_live = st.checkbox("Use live DK splits", value=True, help="Pull odds/splits directly from DK Network (NFL)")
    show_dk_preview = st.checkbox("Show DK raw preview", value=False)
    dk_date_range = st.selectbox(
        "DK date window",
        ["today", "tomorrow", "n7days", "n14days"],
        index=2,
        help="How far ahead to pull DK matchups."
    )
    host_only = st.checkbox(
        "Show only stadiums hosting a game in DK window",
        value=True,
        help="Filters snapshot to home sites that have a listed matchup in the selected DK window."
    )

st.title("🏈 Weather → Betting Edge Radar")

# =============================================================================
# Batch weather fetch
# =============================================================================
all_rows = []
prog = st.progress(0)
for i, s in enumerate(NFL_STADIUMS):
    if outdoor_only and str(s.get("roof","open")).lower() == "fixed":
        prog.progress((i+1)/len(NFL_STADIUMS)); continue
    rs = compute_rows(s, hours)
    all_rows.extend(rs)
    prog.progress((i+1)/len(NFL_STADIUMS))

df = pd.DataFrame(all_rows)
if df.empty:
    st.warning("No data returned.")
    st.stop()

# First-hour snapshot
first_hour = df["time_utc"].min()
snap = df[df["time_utc"] == first_hour].copy()
snap = snap[snap["roof"].isin(roof_filter)]
if trigger_filter:
    snap = snap[snap["bet_triggers"].str.contains("|".join(trigger_filter), na=False)]
if min_wind > 0:
    snap = snap[snap["wind_speed_mph"] >= min_wind]

# =============================================================================
# DK fetch + totals mapping
# =============================================================================
dkdf = pd.DataFrame()
if use_live:
    try:
        dkdf = fetch_dk_splits(event_group=88808, date_range=dk_date_range)
        if show_dk_preview:
            st.expander("DK raw preview").dataframe(dkdf.head(200), use_container_width=True)
    except Exception as e:
        st.error(f"DK fetch failed: {e}")
        try:
            dkdf = fetch_dk_splits(event_group=88808, date_range="n7days")
        except Exception:
            dkdf = pd.DataFrame()

# Totals and Over handle aggregation by matchup
over_map, tot_mean_map, tot_min_map, tot_max_map = {}, {}, {}, {}
if not dkdf.empty and "market" in dkdf.columns:
    dkt = dkdf[dkdf["market"].eq("Total")].copy()
    if not dkt.empty:
        grp = dkt.groupby(["matchup"])
        for m, g in grp:
            tv = g["total_val"].dropna()
            if not tv.empty:
                tot_mean_map[m] = float(tv.mean())
                tot_min_map[m]  = float(tv.min())
                tot_max_map[m]  = float(tv.max())
            g_over = g[g["side"].str.contains(r"^Over", case=False, na=False)]
            over_map[m] = float(g_over["%handle"].dropna().mean()) if not g_over.empty else None

# Build home-team → market_total map using resolver
home_total = {}
home_to_away = {}
home_to_matchup = {}

if not dkdf.empty and "matchup" in dkdf.columns:
    totals_sub = dkdf[dkdf["market"].eq("Total")].dropna(subset=["total_val"]) if "market" in dkdf.columns else pd.DataFrame()
    for m in dkdf["matchup"].dropna().unique().tolist():
        a, h = resolve_matchup(m, dkdf)
        if h:
            home_to_away[h] = a
            home_to_matchup[h] = m
    if not totals_sub.empty:
        for m in totals_sub["matchup"].dropna().unique().tolist():
            a, h = resolve_matchup(m, dkdf)
            if not h:
                continue
            mt = float(totals_sub[totals_sub["matchup"].eq(m)]["total_val"].mean())
            home_total[h] = mt

# Map to snapshot
if not dkdf.empty:
    snap["market_total"] = snap["team"].map(home_total)
    snap["hosting_this_window"] = snap["team"].isin(home_total.keys())
    snap["opponent"] = snap["team"].map(home_to_away)
else:
    snap["market_total"] = None
    snap["hosting_this_window"] = False
    snap["opponent"] = None

# Compute edges in snapshot
snap["adj_total"] = snap["market_total"] + snap["bet_points_delta"]
snap["total_edge"] = snap["adj_total"] - snap["market_total"]
snap["dog_signal"] = snap["bet_triggers"].str.contains("wind_severe|wind_strong|rain_heavy", na=False)

# Attach handle and bettor indices to snapshot
snap["over_handle_pct"] = snap["team"].map(lambda t: over_map.get(home_to_matchup.get(t)) if home_to_matchup else None)
snap["bias"] = snap["bet_points_delta"].apply(bias_class)
snap["wmfi"] = [wmfi(d, h) for d, h in zip(snap["bet_points_delta"], snap["over_handle_pct"])]
snap["ppi"]  = [public_pain_index(d, h) for d, h in zip(snap["bet_points_delta"], snap["over_handle_pct"])]
snap["compass"] = [compass_from_components(th, cr) for th, cr in zip(snap.get("tail_head_mps"), snap.get("cross_mps"))]
snap["fragility"] = [
    fragility(
        tot_min_map.get(home_to_matchup.get(t)) if home_to_matchup else None,
        tot_max_map.get(home_to_matchup.get(t)) if home_to_matchup else None,
        h,
    )
    for t, h in zip(snap["team"], snap["over_handle_pct"])
]
snap["convergence"] = [edge_convergence(w, c, f, p) for w,c,f,p in zip(snap["wmfi"], snap["bias"], snap["fragility"], snap["ppi"])]

# Optional: restrict to hosting sites
if host_only and "hosting_this_window" in snap.columns:
    snap = snap[snap["hosting_this_window"]].copy()

# =============================================================================
# Slate summary
# =============================================================================
if not snap.empty:
    avg_delta = float(snap["bet_points_delta"].mean())
    u_edges = int((snap["bet_points_delta"] <= -2).sum())
    o_edges = int((snap["bet_points_delta"] >=  2).sum())
    avg_wmfi = round(float(pd.Series(snap["wmfi"]).dropna().mean()), 1) if "wmfi" in snap.columns else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Weather Δ", f"{avg_delta:+.1f} pts")
    c2.metric("Under edges (≤ −2)", u_edges)
    c3.metric("Over edges (≥ +2)", o_edges)
    c4.metric("Avg WMFI", f"{avg_wmfi}" if avg_wmfi is not None else "n/a")

# =============================================================================
# Tabs
# =============================================================================
tab_radar, tab_matchups, tab_detail = st.tabs(["📡 Bet Radar","📈 Matchups","🔎 Detail"])

# -------------------------------
# Bet Radar (snapshot)
# -------------------------------
with tab_radar:
    st.subheader("Bet Radar — first hour")
    snap_view = snap.copy()
    snap_view["🏟️ roof"] = snap_view["roof"].apply(roof_icon)
    snap_view["🌬️"] = snap_view["wind_speed_mph"].apply(wind_flag)
    snap_view["☔"] = [precip_flag(a,b) for a,b in zip(snap_view.get("pop_pct"), snap_view.get("precip_in"))]
    snap_view["edge_badge"] = snap_view["bet_points_delta"].apply(edge_pill)
    snap_view["notes"] = snap_view["bet_actions"].apply(actions_short)
    radar_cols = [
        "team","opponent","hosting_this_window","stadium","city","state","🏟️ roof","🌬️","☔","compass",
        "bias","edge_badge","wmfi","ppi","fragility","over_handle_pct",
        "wind_speed_mph","pop_pct","precip_in","bet_triggers","notes",
        "market_total","adj_total","bet_points_delta","total_edge","dog_signal"
    ]
    snap_view = snap_view.reindex(columns=[c for c in radar_cols if c in snap_view.columns])
    st.dataframe(
        snap_view.sort_values("bet_points_delta").reset_index(drop=True),
        use_container_width=True,
        column_config={
            "hosting_this_window": st.column_config.CheckboxColumn("host", help="Stadium hosts a DK matchup in the selected window"),
            "wind_speed_mph": st.column_config.NumberColumn("wind (mph)", format="%.1f"),
            "pop_pct": st.column_config.NumberColumn("PoP %", format="%d"),
            "precip_in": st.column_config.NumberColumn("precip (in/hr)", format="%.2f"),
            "market_total": st.column_config.NumberColumn("O/U", format="%.1f"),
            "adj_total": st.column_config.NumberColumn("Adj O/U", format="%.1f"),
            "bet_points_delta": st.column_config.NumberColumn("Δ pts", format="%.1f"),
            "total_edge": st.column_config.NumberColumn("edge", format="%.1f"),
            "wmfi": st.column_config.NumberColumn("WMFI", format="%.1f"),
            "ppi": st.column_config.NumberColumn("PPI", format="%.1f"),
            "fragility": st.column_config.NumberColumn("Fragility", format="%.1f"),
            "over_handle_pct": st.column_config.NumberColumn("Over handle %", format="%.1f"),
            "edge_badge": st.column_config.TextColumn("edge badge", help="Color-coded points adjustment from weather"),
            "notes": st.column_config.TextColumn("bet notes"),
        },
    )
    st.caption("Icons: 🌪️ ≥20 mph, 🌬️ 15–19, 🍃 10–14.  ☔ light, 🌧️ heavy.  Edge badge buckets: −6+, −4 to −6, −2 to −4, 0 to −2.")

# -------------------------------
# Matchups & Betting Impacts (home-stadium aware)
# -------------------------------
with tab_matchups:
    try:
        dkdf_full = dkdf if not dkdf.empty else fetch_dk_splits(event_group=88808, date_range=dk_date_range)
        if dkdf_full.empty:
            st.info("No DK matchups found to compute home-based edges.")
        else:
            unresolved = []
            all_matchups = dkdf_full["matchup"].dropna().unique().tolist()
            totals = dkdf_full[dkdf_full["market"].eq("Total")].assign(has_total=~dkdf_full[dkdf_full["market"].eq("Total")]["total_val"].isna())
            rows = []
            for matchup in all_matchups:
                away, home = resolve_matchup(matchup, dkdf_full)
                if not home or stadium_for_home_team(home) is None:
                    unresolved.append({"matchup": matchup, "away": away, "home": home})
                sub_tot = totals[totals["matchup"].eq(matchup) & totals["has_total"]]
                if not sub_tot.empty:
                    mt_mean = float(sub_tot["total_val"].mean())
                    mt_min = float(sub_tot["total_val"].min())
                    mt_max = float(sub_tot["total_val"].max())
                else:
                    mt_mean = float("nan"); mt_min = float("nan"); mt_max = float("nan")
                stad = stadium_for_home_team(home)
                if stad:
                    wrow = ensure_first_hour_for_stadium(df, stad["stadium"], hours)
                    if wrow is None:
                        delta = 0.0; wind_mph = None; precip_in = None; triggers = ""; actions = ""
                        th = None; cr = None
                    else:
                        delta = float(wrow.get("bet_points_delta") or 0.0)
                        wind_mph = wrow.get("wind_speed_mph")
                        precip_in = wrow.get("precip_in")
                        triggers = wrow.get("bet_triggers")
                        actions = wrow.get("bet_actions")
                        th = wrow.get("tail_head_mps"); cr = wrow.get("cross_mps")
                    stad_name = stad["stadium"]; city = stad["city"]; state = stad["state"]; roof = stad.get("roof")
                else:
                    delta = 0.0; wind_mph = None; precip_in = None; triggers = ""; actions = ""; th=None; cr=None
                    stad_name = None; city = None; state = None; roof = None

                over_handle = over_map.get(matchup)
                bias = bias_class(delta)
                w_score = wmfi(delta, over_handle)
                pain = public_pain_index(delta, over_handle)
                frag = fragility(mt_min if pd.notna(mt_min) else None, mt_max if pd.notna(mt_max) else None, over_handle)
                compass = compass_from_components(th, cr)
                mom = "·"
                if stad:
                    site_rows = df[(df["stadium"]==stad["stadium"])].sort_values("time_utc")
                    if not site_rows.empty and site_rows.shape[0] >= 2:
                        now_d = site_rows.iloc[0]["bet_points_delta"]; nxt_d = site_rows.iloc[1]["bet_points_delta"]
                        mom = momentum(nxt_d, now_d)

                adj_total = round(mt_mean + delta, 1) if pd.notna(mt_mean) else None
                rows.append({
                    "matchup": matchup,
                    "away": away, "home": home,
                    "stadium": stad_name, "city": city, "state": state, "roof": roof,
                    "wind_mph": wind_mph, "precip_in": precip_in,
                    "triggers": triggers, "actions": actions,
                    "market_total": round(mt_mean, 1) if pd.notna(mt_mean) else None,
                    "market_total_min": round(mt_min, 1) if pd.notna(mt_min) else None,
                    "market_total_max": round(mt_max, 1) if pd.notna(mt_max) else None,
                    "weather_delta": round(delta, 1),
                    "adj_total": adj_total,
                    "edge_total": round(delta, 1),
                    "over_handle_pct": round(over_handle,1) if over_handle is not None else None,
                    "wmfi": w_score, "ppi": pain, "fragility": frag,
                    "compass": compass, "bias": bias, "momentum": mom,
                    "convergence": edge_convergence(w_score, bias, frag, pain),
                })
            mdf = pd.DataFrame(rows)
            if not mdf.empty:
                st.subheader("Upcoming Games — Weather-adjusted edges (home-based)")
                mdf_view = mdf.copy()
                mdf_view["🏟️ roof"] = mdf_view["roof"].apply(roof_icon)
                mdf_view["🌬️"] = mdf_view["wind_mph"].apply(wind_flag)
                mdf_view["☔"] = [precip_flag(None, b) for b in mdf_view.get("precip_in")]
                show_cols = [
                    "matchup","home","away","stadium","city","state","🏟️ roof","🌬️","precip_in",
                    "triggers","actions","market_total","market_total_min","market_total_max",
                    "weather_delta","adj_total","edge_total",
                    "bias","wmfi","ppi","fragility","momentum","convergence","compass","over_handle_pct"
                ]
                mdf_view = mdf_view.reindex(columns=[c for c in show_cols if c in mdf_view.columns])
                st.dataframe(
                    mdf_view.reset_index(drop=True),
                    use_container_width=True,
                    column_config={
                        "precip_in": st.column_config.NumberColumn("precip (in/hr)", format="%.2f"),
                        "market_total": st.column_config.NumberColumn("O/U", format="%.1f"),
                        "market_total_min": st.column_config.NumberColumn("min O/U", format="%.1f"),
                        "market_total_max": st.column_config.NumberColumn("max O/U", format="%.1f"),
                        "weather_delta": st.column_config.NumberColumn("Δ pts", format="%.1f"),
                        "adj_total": st.column_config.NumberColumn("Adj O/U", format="%.1f"),
                        "edge_total": st.column_config.NumberColumn("edge", format="%.1f"),
                        "wmfi": st.column_config.NumberColumn("WMFI", format="%.1f"),
                        "ppi": st.column_config.NumberColumn("PPI", format="%.1f"),
                        "fragility": st.column_config.NumberColumn("Fragility", format="%.1f"),
                        "over_handle_pct": st.column_config.NumberColumn("Over handle %", format="%.1f"),
                    },
                )
                mbytes = mdf.to_csv(index=False).encode("utf-8")
                st.download_button("Download matchup edges CSV", mbytes, "nfl_matchup_edges.csv", "text/csv")
                if unresolved:
                    st.info(f"Unresolved matchups: {len(unresolved)}. Add aliases in TEAM_ALIASES or CITY_ALIASES to resolve these.")
                    st.expander("Show unresolved").dataframe(pd.DataFrame(unresolved))
    except Exception as e:
        st.warning(f"Live matchup edges unavailable: {e}")

# -------------------------------
# Detail tab
# -------------------------------
with tab_detail:
    st.subheader("Detail view")
    uniq = df.groupby(["team","stadium","city","state"]).head(1).reset_index(drop=True)
    pick = st.selectbox("Pick a matchup site", list(range(len(uniq))), format_func=lambda i: f'{uniq.loc[i,"stadium"]} — {uniq.loc[i,"team"]} ({uniq.loc[i,"city"]}, {uniq.loc[i,"state"]})')
    site = uniq.loc[pick, ["team","stadium","city","state"]].to_dict()
    detail = df[(df["team"]==site["team"]) & (df["stadium"]==site["stadium"])].copy()
    detail["adj_total_series"] = detail["bet_points_delta"] + manual_total

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    latest = detail.iloc[0]
    kc1.metric("Temp °C", latest.get("temp_c"))
    kc2.metric("Wind mph", latest.get("wind_speed_mph"))
    kc3.metric("Points Δ (now)", latest.get("bet_points_delta"))
    kc4.metric("Market total", manual_total)
    kc5.metric("Adj total (now)", round(latest.get("bet_points_delta", 0) + manual_total, 1))

    fig3, ax3 = plt.subplots()
    ax3.plot(pd.to_datetime(detail["time_utc"]), detail["bet_points_delta"])
    ax3.set_title("Projected points Δ vs time")
    ax3.set_xlabel("UTC time"); ax3.set_ylabel("Δ points")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    ax4.plot(pd.to_datetime(detail["time_utc"]), detail["adj_total_series"])
    ax4.set_title("Adjusted total vs time (using market total input)")
    ax4.set_xlabel("UTC time"); ax4.set_ylabel("points")
    st.pyplot(fig4)

# Snapshot export
csv_bytes = snap.sort_values("bet_points_delta").to_csv(index=False).encode("utf-8")
st.download_button("Download edges CSV (snapshot)", csv_bytes, "nfl_edges_snapshot.csv", "text/csv")

st.caption("Deterministic weather → scoring heuristic. It ignores injuries/pace. That part is on you.")
