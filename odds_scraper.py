import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

BASE_URL = "https://www.sportsoddshistory.com/nfl-game-season/?y={year}"
START_YEAR = 2000
END_YEAR = 2024
OUTPUT_FILE = "nfl_game_closing_lines.csv"

def parse_game_row(tr, week, season):
    tds = tr.find_all("td")
    if len(tds) < 10:
        return []

    # Extract favorite and underdog
    fav = tds[4].get_text(strip=True)
    dog = tds[8].get_text(strip=True)

    # Spread (e.g., "W -3.5", "L -7", etc.)
    spread_raw = tds[6].get_text(strip=True)
    spread_match = re.search(r'(-?\d+\.?\d*)', spread_raw)
    spread_line = float(spread_match.group(1)) if spread_match else None

    # Total (e.g., "O 46", "U 43.5")
    total_raw = tds[9].get_text(strip=True)
    total_match = re.search(r'(\d+\.?\d*)', total_raw)
    total_line = float(total_match.group(1)) if total_match else None

    # Determine home team (look for '@')
    at_symbol = tds[3].get_text(strip=True)
    neutral = at_symbol == 'N'
    home_team = fav if at_symbol == '@' else dog
    away_team = dog if at_symbol == '@' else fav

    # Default placeholders (SportsOddsHistory doesn’t list ML or odds)
    spread_odds = -110
    total_odds = -110
    ml_home = None
    ml_away = None

    rows = []

    # Spread market
    if spread_line is not None:
        rows.append({
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "market": "spread",
            "label": home_team,
            "price": spread_odds,
            "point": spread_line
        })

    # Total market
    if total_line is not None:
        rows.append({
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "market": "total",
            "label": "over",
            "price": total_odds,
            "point": total_line
        })

        rows.append({
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "market": "total",
            "label": "under",
            "price": total_odds,
            "point": total_line
        })

    # Moneyline placeholders (can integrate from other data sources)
    # rows.append({
    #     "season": season,
    #     "week": week,
    #     "home_team": home_team,
    #     "away_team": away_team,
    #     "market": "ml",
    #     "label": home_team,
    #     "price": ml_home,
    #     "point": None
    # })

    # rows.append({
    #     "season": season,
    #     "week": week,
    #     "home_team": home_team,
    #     "away_team": away_team,
    #     "market": "ml",
    #     "label": away_team,
    #     "price": ml_away,
    #     "point": None
    # })

    return rows


def scrape_season(year):
    url = BASE_URL.format(year=year)
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")

    all_rows = []
    week = None

    for tag in soup.find_all(["h3", "table"]):
        if tag.name == "h3":
            header_text = tag.get_text(strip=True)
            week_match = re.search(r"Week\s+(\d+)", header_text)
            if week_match:
                week = int(week_match.group(1))
        elif tag.name == "table" and week is not None:
            for tr in tag.find_all("tr")[1:]:
                row_data = parse_game_row(tr, week, year)
                all_rows.extend(row_data)

    return all_rows


def main():
    all_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Fetching {year}...")
        season_data = scrape_season(year)
        all_data.extend(season_data)
        time.sleep(1.5)  # respectful delay

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
