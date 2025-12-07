import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import re
import os

# ---- Hardcoded 1H Odds CSV Path ----
HALF_CSV_PATH = "/Users/matthewfox/Documents/FoxEdgeAI_LOCAL.bak/NFL/pages/1h_odds.csv"

# ---- Fetch Functions ----

def clean_odds(odds_str):
    try:
        return int(str(odds_str).replace("−", "-").strip())
    except Exception:
        try:
            return int(float(odds_str))
        except Exception:
            return None

def _parse_total_from_side(side_text: str):
    try:
        s = str(side_text).strip()
        if s.lower().startswith("over ") or s.lower().startswith("under "):
            parts = s.split()
            if len(parts) >= 2:
                return parts[0].title(), float(parts[1])
    except Exception:
        pass
    return None, None

def _parse_spread_from_side(side_text: str):
    try:
        s = str(side_text).strip()
        parts = s.split()
        if len(parts) >= 2:
            last = parts[-1].replace("−", "-")
            if last.startswith(("+", "-")):
                return float(last)
    except Exception:
        pass
    return None

def fetch_dk_splits(event_group: int = 88808, date_range: str = "today", market: str = "All") -> pd.DataFrame:
    from urllib.parse import urlencode, urlparse, parse_qs

    base = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
    params = {"tb_eg": event_group, "tb_edate": date_range, "tb_emt": "0"}
    first_url = f"{base}?{urlencode(params)}"

    def clean(text: str) -> str:
        return re.sub(r"opens?\s+in\s+(?:a\s+)?new\s+tab", "", text or "", flags=re.I).strip()

    def _get_html(url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
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
                    urls.add(base + href if not href.startswith('http') else href)
        def pnum(u: str) -> int:
            try:
                return int(parse_qs(urlparse(u).query).get("tb_page", ["1"])[0])
            except Exception:
                return 1
        return sorted(list(urls), key=pnum)

    def _parse_page(html: str) -> list[dict]:
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        games = soup.select("div.tb-se")
        out = []
        now = datetime.now(timezone.utc)
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
                    odds_val = clean_odds(oddstxt)
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

    first_html = _get_html(first_url)
    all_urls = _discover_page_urls(first_html)
    records = []
    for url in all_urls:
        html = first_html if url == first_url else _get_html(url)
        records.extend(_parse_page(html))
    if not records:
        if date_range == "today":
            return fetch_dk_splits(event_group, "tomorrow", market)
        if date_range == "tomorrow":
            return fetch_dk_splits(event_group, "n7days", market)
    df = pd.DataFrame(records)
    return df

# ---- 1H Analysis Functions ----

def implied_prob_from_odds(odds):
    if odds is None or not isinstance(odds, (int, float)):
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 0.5

def calculate_expected_1h_spread(full_spread, r=0.55, key_adjust=0):
    if full_spread is None:
        return 0
    return r * abs(full_spread) - key_adjust if full_spread < 0 else r * full_spread + key_adjust

def calculate_expected_1h_total(full_total, r=0.50):
    if full_total is None:
        return 0
    return r * full_total

def is_key_number(n):
    if n is None:
        return False
    key_nums = [3, 7, 10, 6, 4, 1, 2, 0]
    return abs(n) in key_nums

def model_prob_cover(spread, sigma=6.5, discrepancy_adjust=0):
    if spread is None:
        return 0.5
    return 0.5 + (abs(spread) / (2 * sigma)) + discrepancy_adjust

def normalize_team_name(name):
    """Normalize team names for consistent matching."""
    if not name:
        return ""
    name = name.strip().lower()
    team_map = {
        "la chargers": "los angeles chargers",
        "cle browns": "cleveland browns",
        "det lions": "detroit lions",
        "phi eagles": "philadelphia eagles",
        "tb buccaneers": "tampa bay buccaneers",
        "no saints": "new orleans saints",
        "buf bills": "buffalo bills",
        "was commanders": "washington commanders",
        "atl falcons": "atlanta falcons",
        "ten titans": "tennessee titans",
        "hou texans": "houston texans",
        "car panthers": "carolina panthers",
        "ne patriots": "new england patriots",
        "ind colts": "indianapolis colts",
        "la rams": "los angeles rams",
        "jax jaguars": "jacksonville jaguars",
        "sf 49ers": "san francisco 49ers",
        "bal ravens": "baltimore ravens",
        "kc chiefs": "kansas city chiefs",
        "chi bears": "chicago bears",
        "lv raiders": "las vegas raiders",
        "gb packers": "green bay packers",
        "dal cowboys": "dallas cowboys",
        "ny jets": "new york jets",
        "mia dolphins": "miami dolphins",
        "cin bengals": "cincinnati bengals",
        "den broncos": "denver broncos",
        "ny giants": "new york giants"
    }
    return team_map.get(name, name)

def normalize_matchup(matchup):
    """Normalize matchup strings to 'home vs away' (lowercase) regardless of
    whether the source uses '@', 'at', or 'vs', and regardless of spacing.
    Examples accepted:
      'LA Chargers @NY Giants' -> 'new york giants vs los angeles chargers'
      'phi eagles @ tb buccaneers' -> 'tampa bay buccaneers vs philadelphia eagles'
      'Kansas City Chiefs vs Baltimore Ravens' -> 'kansas city chiefs vs baltimore ravens'
    """
    if not matchup:
        return ""

    s = str(matchup).strip().lower()
    # collapse weird unicode/minus/whitespace artifacts
    s = s.replace("\u2212", "-")
    s = re.sub(r"\s+", " ", s)

    # Prefer explicit home/away from '@' or 'at' if present
    if "@" in s:
        parts = re.split(r"\s*@\s*", s)
        if len(parts) == 2:
            away_raw, home_raw = parts[0], parts[1]
            away = normalize_team_name(away_raw)
            home = normalize_team_name(home_raw)
            return f"{home} vs {away}"

    if " at " in s:
        parts = re.split(r"\s+at\s+", s)
        if len(parts) == 2:
            away_raw, home_raw = parts[0], parts[1]
            away = normalize_team_name(away_raw)
            home = normalize_team_name(home_raw)
            return f"{home} vs {away}"

    # Fall back to 'vs' format, assume left is home
    if " vs " in s:
        parts = re.split(r"\s+vs\s+", s)
        if len(parts) == 2:
            home_raw, away_raw = parts[0], parts[1]
            home = normalize_team_name(home_raw)
            away = normalize_team_name(away_raw)
            return f"{home} vs {away}"

    # Final fallback: try splitting on common separators without strict spacing
    parts = re.split(r"\s*(?:@|vs|at)\s*", s)
    if len(parts) == 2:
        # default to treating right side as home if an '@' was present originally
        if "@" in s or " at " in s:
            away = normalize_team_name(parts[0])
            home = normalize_team_name(parts[1])
            return f"{home} vs {away}"
        # otherwise assume left is home
        home = normalize_team_name(parts[0])
        away = normalize_team_name(parts[1])
        return f"{home} vs {away}"

    # If we couldn't confidently parse, return cleaned string (lowercase)
    return s

def analyze_1h_bets(full_df, half_df):
    # Normalize matchups
    full_df = full_df.copy()
    half_df = half_df.copy()
    print("Applying normalization to full-game matchups...")
    full_df['matchup'] = full_df['matchup'].apply(normalize_matchup)
    print("First few normalized full-game matchups:", list(full_df['matchup'].head(5)))
    half_df['matchup'] = (half_df['home_team'].map(normalize_team_name) + ' vs ' + half_df['away_team'].map(normalize_team_name)).str.lower()
    
    # Debug: List matchups
    print("Full-game matchups:", sorted(list(full_df['matchup'].unique())))
    print("1H odds matchups:", sorted(list(half_df['matchup'].unique())))
    
    results = []
    matched_games = set()
    print("Analyzing 1H bets...")
    for matchup in full_df['matchup'].unique():
        print(f"\nMatchup: {matchup}")
        full_spread_rows = full_df[(full_df['matchup'] == matchup) & (full_df['market'] == 'Spread')]
        full_total_rows = full_df[(full_df['matchup'] == matchup) & (full_df['market'] == 'Total')]
        half_spread_rows = half_df[(half_df['matchup'] == matchup) & (half_df['market'] == 'spreads_h1')]
        half_total_rows = half_df[(half_df['matchup'] == matchup) & (half_df['market'] == 'totals_h1')]

        if not half_spread_rows.empty or not half_total_rows.empty:
            matched_games.add(matchup)
            print(f"  Matched with 1H data (spreads: {len(half_spread_rows)}, totals: {len(half_total_rows)})")

        # Spread Analysis
        if not full_spread_rows.empty:
            full_fav_row = full_spread_rows.iloc[0]
            full_spread = full_fav_row['spread']
            full_odds = full_fav_row['odds']
            key_adjust = 0.5 if is_key_number(full_spread) else 0
            expected_1h_spread = calculate_expected_1h_spread(full_spread, key_adjust=key_adjust)

            if not half_spread_rows.empty:
                half_fav_row = half_spread_rows[half_spread_rows['point'] < 0]
                if not half_fav_row.empty:
                    half_fav_row = half_fav_row.iloc[0]
                    half_spread = half_fav_row['point']
                    half_odds = half_fav_row['price']
                    discrepancy = expected_1h_spread - abs(half_spread)
                    print(f"  Spread: Expected 1H {expected_1h_spread:.2f}, Posted 1H {half_spread:.2f}, Discrepancy {discrepancy:.2f}")
                    if discrepancy >= 0.2:
                        implied_p = implied_prob_from_odds(half_odds)
                        adjust = 0.03 if discrepancy > 1 else 0.02
                        model_p = model_prob_cover(half_spread, discrepancy_adjust=adjust)
                        edge = model_p - implied_p
                        if edge > 0.02:
                            results.append({
                                'game_id': half_fav_row['game_id'],
                                'matchup': matchup,
                                'bet_type': '1H Spread',
                                'side': half_fav_row['label'],
                                'odds': half_odds,
                                'expected_1h': expected_1h_spread,
                                'posted_1h': half_spread,
                                'discrepancy': discrepancy,
                                'model_p': model_p,
                                'implied_p': implied_p,
                                'edge': edge
                            })

        # Total Analysis
        if not full_total_rows.empty:
            full_over_row = full_total_rows[full_total_rows['side'].str.lower().str.startswith('over')]
            if not full_over_row.empty:
                _, full_total = _parse_total_from_side(full_over_row.iloc[0]['side'])
                full_odds = full_over_row.iloc[0]['odds']
                expected_1h_total = calculate_expected_1h_total(full_total)
                half_over_row = half_total_rows[half_total_rows['label'].str.lower() == 'over']
                if not half_over_row.empty:
                    half_over_row = half_over_row.iloc[0]
                    half_total = half_over_row['point']
                    half_odds = half_over_row['price']
                    discrepancy = expected_1h_total - half_total
                    print(f"  Total: Expected 1H {expected_1h_total:.2f}, Posted 1H {half_total:.2f}, Discrepancy {discrepancy:.2f}")
                    if discrepancy >= 0.2:
                        implied_p = implied_prob_from_odds(half_odds)
                        adjust = 0.03 if discrepancy > 1 else 0.02
                        model_p = 0.5 + adjust
                        edge = model_p - implied_p
                        if edge > 0.02:
                            results.append({
                                'game_id': half_over_row['game_id'],
                                'matchup': matchup,
                                'bet_type': '1H Total',
                                'side': 'Over',
                                'odds': half_odds,
                                'expected_1h': expected_1h_total,
                                'posted_1h': half_total,
                                'discrepancy': discrepancy,
                                'model_p': model_p,
                                'implied_p': implied_p,
                                'edge': edge
                            })

    # Debug: Report unmatched games
    half_matchups = set(half_df['matchup'].unique())
    unmatched = half_matchups - matched_games
    if unmatched:
        print("\nUnmatched 1H odds games:", sorted(list(unmatched)))
        for um in unmatched:
            um_teams = set(um.split(" vs "))
            for fm in full_df['matchup'].unique():
                fm_teams = set(fm.split(" vs "))
                if um_teams & fm_teams:
                    print(f"  Possible match for '{um}': '{fm}'")

    return pd.DataFrame(results)

# ---- Market Mood Functions ----

def compute_market_mood(full_df):
    full_df = full_df.copy()
    full_df["irrationality"] = abs(full_df["%bets"] - full_df["%handle"])
    mood_score = full_df["irrationality"].mean()
    print(f"Market Irrationality Index: {mood_score:.2f}%")

# ---- Main Script ----

if __name__ == "__main__":
    print("Fetching full-game NFL betting splits from DraftKings...")
    full_df = fetch_dk_splits(event_group=88808)
    if full_df.empty:
        print("No data fetched. Attempting to load from full_game_splits.csv...")
        try:
            full_df = pd.read_csv("full_game_splits.csv")
            print(f"Loaded {len(full_df['matchup'].unique())} games from full_game_splits.csv")
        except Exception as e:
            print(f"Failed to load full_game_splits.csv: {e}")
            full_df = pd.DataFrame()
    else:
        print(f"Fetched {len(full_df['matchup'].unique())} games.")
        full_df = full_df.copy()
        print("Normalizing full-game matchups after fetch...")
        full_df['matchup'] = full_df['matchup'].apply(normalize_matchup)
        print("First few normalized full-game matchups after fetch:", list(full_df['matchup'].head(5)))
        full_df.to_csv("full_game_splits.csv", index=False)
        print("Saved to full_game_splits.csv")
        compute_market_mood(full_df)

    if os.path.exists(HALF_CSV_PATH):
        try:
            half_df = pd.read_csv(HALF_CSV_PATH)
            required_cols = ['game_id', 'home_team', 'away_team', 'market', 'label', 'price', 'point']
            if not all(col in half_df.columns for col in required_cols):
                print(f"CSV at {HALF_CSV_PATH} missing required columns: {required_cols}")
            else:
                half_df['matchup'] = (half_df['home_team'].map(normalize_team_name) + ' vs ' + half_df['away_team'].map(normalize_team_name)).str.lower()
                print(f"Loaded 1H data from {HALF_CSV_PATH} with {len(half_df['matchup'].unique())} games.")
                results_df = analyze_1h_bets(full_df, half_df)
                if not results_df.empty:
                    print("High-likelihood 1H bets found:")
                    print(results_df[['game_id', 'matchup', 'bet_type', 'side', 'odds', 'expected_1h', 'posted_1h', 'discrepancy', 'edge']])
                    results_df.to_csv("suggested_1h_bets.csv", index=False)
                    print("Saved to suggested_1h_bets.csv")
                else:
                    print("No high-likelihood 1H bets identified. Check discrepancies in output above.")
        except Exception as e:
            print(f"Error reading CSV at {HALF_CSV_PATH}: {e}")
            print("Ensure the CSV has the correct format and numeric values for 'price' and 'point'.")
    else:
        print(f"CSV not found at {HALF_CSV_PATH}. Skipping 1H analysis.")