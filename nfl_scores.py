#!/usr/bin/env python3
"""
ESPN NFL Scoreboard Scraper (v4.1)
----------------------------------
Fetches full NFL scoreboard data for any date:
- Teams, scores, status, venue, odds, weather, links
- For in-progress games: quarter, time remaining, down & distance, yard line, possession
"""

import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup

API_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
HTML_URL = "https://www.espn.com/nfl/scoreboard"

def fetch_json(date: str):
    """Fetch JSON scoreboard data."""
    r = requests.get(f"{API_URL}?dates={date}", headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()

def fetch_html(date: str):
    """Fetch scoreboard HTML page for live situation parsing."""
    r = requests.get(f"{HTML_URL}/_/date/{date}", headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def parse_game(event: dict) -> dict:
    """Parse structured data for one game."""
    comp = event.get("competitions", [{}])[0]
    status = comp.get("status", {}).get("type", {}).get("description", "")
    start_time = comp.get("date", "")
    kickoff = (
        datetime.fromisoformat(start_time.replace("Z", "+00:00")).strftime("%I:%M %p ET")
        if start_time
        else "TBD"
    )
    venue = comp.get("venue", {}).get("fullName", "")
    city = comp.get("venue", {}).get("address", {}).get("city", "")
    odds_info = comp.get("odds", [{}])[0]
    odds = {
        "spread": odds_info.get("details", "N/A"),
        "over_under": odds_info.get("overUnder", "N/A"),
        "provider": odds_info.get("provider", {}).get("name", "N/A"),
    }
    network = comp.get("geoBroadcasts", [{}])[0].get("media", {}).get("shortName", "")

    def team(t):
        return {
            "name": t.get("team", {}).get("displayName", ""),
            "abbrev": t.get("team", {}).get("abbreviation", ""),
            "score": int(t.get("score", 0) or 0),
            "record": t.get("records", [{}])[0].get("summary", ""),
            "winner": t.get("winner", False),
        }

    teams = comp.get("competitors", [])
    home = next((team(t) for t in teams if t.get("homeAway") == "home"), {})
    away = next((team(t) for t in teams if t.get("homeAway") == "away"), {})

    return {
        "game_id": comp.get("id"),
        "status": status,
        "kickoff": kickoff,
        "venue": venue,
        "city": city,
        "network": network,
        "odds": odds,
        "home_team": home,
        "away_team": away,
        "links": {
            "gamecast": f"https://www.espn.com/nfl/game/_/gameId/{comp.get('id')}",
            "boxscore": f"https://www.espn.com/nfl/boxscore/_/gameId/{comp.get('id')}",
            "playbyplay": f"https://www.espn.com/nfl/playbyplay/_/gameId/{comp.get('id')}",
        },
    }

def enrich_with_live_context(games: list, soup: BeautifulSoup):
    """
    Augment in-progress games with down, distance, yard line, possession, and clock.
    Works off ESPN scoreboard's live HTML layout.
    """
    raw = soup.get_text(" ", strip=True)
    raw = re.sub(r"\s+", " ", raw)

    for g in games:
        gid = g["game_id"]
        if not gid:
            continue

        # Match game block near team abbreviations (DAL, DEN, etc.)
        abbrevs = [g["away_team"]["abbrev"], g["home_team"]["abbrev"]]
        if not all(abbrevs):
            continue

        # Build a block pattern around the matchup
        block_pattern = rf"{abbrevs[0]}.*?{abbrevs[1]}.*?(?=Gamecast|Box)"
        block = re.search(block_pattern, raw)
        if not block:
            continue

        segment = block.group(0)

        # Extract quarter/time like "7:14 - 4th"
        quarter_time = re.search(r"\b\d{1,2}:\d{2}\s*-\s*\d+(?:st|nd|rd|th)\b", segment)

        # Extract down & distance like "3rd & 3 at IND 4"
        down_distance = re.search(r"\b\d+(?:st|nd|rd|th)\s*&\s*\d+\s*at\s*[A-Z]{2,3}\s*\d+\b", segment)

        # Extract possession team from yard line (e.g. "at DAL 5")
        possession = None
        if down_distance:
            poss_match = re.search(r"at\s+([A-Z]{2,3})\s*\d+", down_distance.group(0))
            possession = poss_match.group(1) if poss_match else None

        if any([quarter_time, down_distance]):
            g["live"] = {
                "quarter_time": quarter_time.group(0) if quarter_time else "N/A",
                "down_distance": down_distance.group(0) if down_distance else "N/A",
                "possession": possession if possession else "N/A",
            }
        else:
            g["live"] = None

def display(games):
    """Nicely print games."""
    print("\n🏈 NFL SCOREBOARD")
    print("=" * 80)
    for g in games:
        line = f"{g['away_team']['name']} [{g['away_team']['score']}] @ {g['home_team']['name']} [{g['home_team']['score']}]"
        print(line)
        print(f"   Status: {g['status']} | Kickoff: {g['kickoff']} | Network: {g['network']}")
        print(f"   Venue: {g['venue']} ({g['city']})")
        if g["odds"]["spread"] != "N/A":
            print(f"   Odds: {g['odds']['spread']} | O/U {g['odds']['over_under']}")
        if g.get("live"):
            live = g["live"]
            print(f"   ⏱  {live['quarter_time']} | {live['down_distance']} | Poss: {live['possession']}")
        print(f"   Gamecast: {g['links']['gamecast']}")
        print("-" * 80)

def main():
    date = datetime.now().strftime("%Y%m%d")
    print(f"Fetching NFL scoreboard for {date}...\n")
    data = fetch_json(date)
    soup = fetch_html(date)

    games = [parse_game(e) for e in data.get("events", [])]
    enrich_with_live_context(games, soup)
    display(games)

if __name__ == "__main__":
    main()
