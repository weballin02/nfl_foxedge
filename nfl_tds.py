import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_td_markets():
    """
    Fetches 'Anytime TD Scorer', 'First TD Scorer', and '2+ TDs' markets
    from DraftKings Network's Most Bet Player Props (NFL).
    Tries 'today', 'tomorrow', then 'next 7 days' until results found.
    """
    base_url = "https://dknetwork.draftkings.com/draftkings-sportsbook-player-props/"
    params = {
        "tb_view": "2",       # Most bet props
        "tb_eg": "88808",     # NFL event group
    }
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    date_windows = ["today", "tomorrow", "n7days"]
    all_data = []

    for date_key in date_windows:
        params["tb_edate"] = date_key
        print(f"\nChecking {date_key}…")

        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        if not response.ok:
            print(f"Request failed for {date_key}: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if not table:
            print(f"No table found for {date_key}.")
            continue

        rows = table.find_all("tr")[1:]  # skip header
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) != 5:
                continue
            market = cols[2]
            if any(x in market for x in ["TD", "Touchdown"]):
                all_data.append({
                    "Event": cols[0],
                    "Date": cols[1],
                    "Market": market,
                    "Player": cols[3],
                    "Odds": cols[4],
                    "Range": date_key
                })

    if not all_data:
        print("No TD-related props found in any window.")
        return None

    df = pd.DataFrame(all_data)

    # Split by market type
    anytime_df = df[df["Market"].str.contains("Anytime", case=False, na=False)]
    first_df = df[df["Market"].str.contains("First", case=False, na=False)]
    multi_df = df[df["Market"].str.contains("2", case=False, na=False)]

    return anytime_df.head(10), first_df.head(10), multi_df.head(10)


if __name__ == "__main__":
    anytime, first, multi = fetch_td_markets() or (None, None, None)

    def print_section(title, df):
        print(f"\n=== {title} ===")
        if df is None or df.empty:
            print("No data found.")
        else:
            print(df[["Event", "Date", "Player", "Odds", "Range"]].to_string(index=False))

    print_section("Top 10 Most Bet Anytime TD Scorers", anytime)
    print_section("Top 10 Most Bet First TD Scorers", first)
    print_section("Top 10 Most Bet 2+ TD Scorers", multi)
