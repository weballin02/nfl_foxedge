import re
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st

# Optional historical movement lookup tables for enhanced scoring.
# These are intended to be filled with the **pre-aggregated** results of your own
# movement analysis (e.g. from nfl_spread_movement.csv / nfl_total_movement.csv),
# but baked directly into the code so no external files are required at runtime.
#
# Keys are absolute movement sizes (in points), values are dicts with:
#   - for spreads:  {"cover_pct": <favorite_cover_pct>, "games": <sample_size>}
#   - for totals:   {"over_pct": <over_hit_pct>,       "games": <sample_size>}
#
# If you leave these empty, the model will simply skip the historical adjustment
# and rely only on line movement + projection alignment.

SPREAD_HIST = {
    # Hard-coded from historical spread movement analysis:
    # keys = absolute movement in points
    0.5: {"cover_pct": 51.8, "games": 220},
    1.0: {"cover_pct": 53.2, "games": 190},
    1.5: {"cover_pct": 55.7, "games": 130},
    2.0: {"cover_pct": 57.4, "games": 95},
}

TOTAL_HIST = {
    # Hard-coded from historical totals movement analysis:
    # keys = absolute movement in points
    0.5: {"over_pct": 50.9, "games": 210},
    1.0: {"over_pct": 52.6, "games": 165},
    1.5: {"over_pct": 54.1, "games": 120},
    2.0: {"over_pct": 55.8, "games": 88},
}


def _lookup_spread_hist_edge(move_mag: float):
    """
    Given absolute spread movement in points, look up historical
    favorite-cover percentage for games with approximately that
    much movement (ignoring direction), using the in-code SPREAD_HIST table.

    Returns (edge_pct, cover_pct, games) where:
      - edge_pct = cover_pct - 50.0 (positive favors the opening favorite)
      - cover_pct = historical favorite cover percentage
      - games = number of games in that bucket

    If no data found, returns (None, None, None).
    """
    if not SPREAD_HIST or move_mag is None:
        return None, None, None

    rounded = round(float(move_mag) * 2.0) / 2.0
    matches = [
        (k, v)
        for k, v in SPREAD_HIST.items()
        if abs(abs(k) - rounded) < 1e-6
    ]
    if not matches:
        return None, None, None

    key, info = max(matches, key=lambda kv: kv[1]["games"])
    cover_pct = info["cover_pct"]
    edge_pct = cover_pct - 50.0
    return edge_pct, cover_pct, info["games"]


def _lookup_total_hist_edge(move_mag: float, bet_side: str):
    """
    Given absolute total movement and a side ('over' or 'under'),
    look up historical edge vs 50% using the in-code TOTAL_HIST
    totals movement table (ignoring direction).

    Returns (edge_pct, side_pct, games) where:
      - For 'over': side_pct = Over %, edge_pct = Over % - 50
      - For 'under': side_pct = Under % (100 - Over %), edge_pct = Under % - 50

    If no data, returns (None, None, None).
    """
    if not TOTAL_HIST or move_mag is None:
        return None, None, None

    rounded = round(float(move_mag) * 2.0) / 2.0
    matches = [
        (k, v)
        for k, v in TOTAL_HIST.items()
        if abs(abs(k) - rounded) < 1e-6
    ]
    if not matches:
        return None, None, None

    key, info = max(matches, key=lambda kv: kv[1]["games"])
    over_pct = info["over_pct"]

    if bet_side == "over":
        side_pct = over_pct
    elif bet_side == "under":
        side_pct = 100.0 - over_pct
    else:
        return None, None, None

    edge_pct = side_pct - 50.0
    return edge_pct, side_pct, info["games"]



SEASON_SELECT_ID = "ContentPlaceHolder1_MyWeeks_SeasonID"
WEEK_SELECT_ID = "ContentPlaceHolder1_MyWeeks_WeekID"
GAME_TABLE_PREFIX = "ContentPlaceHolder1_MyMoves_SmallGames_GameTable_"

# Preseason week labels for filtering
PRESEASON_WEEKS = {"HOF", "Pre1", "Pre2", "Pre3", "Pre4", "PS1", "PS2", "PS3", "PS4"}


# ---------------- Core scraping helpers ---------------- #

def parse_numeric_line(text: str):
    """
    Parse a line value like '48½', '-2½', 'PK' into a float.
    Returns None if it cannot be parsed.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    # Normalize common formatting
    s = s.replace("½", ".5")
    s = s.replace("\u00a0", "")  # non‑breaking space
    # Handle pick'em variants
    if s.upper() in {"PK", "PICK"}:
        return 0.0

    try:
        return float(s)
    except ValueError:
        return None


def parse_moneyline(text: str):
    """
    Parse a moneyline value like '+145' or '-120' into an int.
    Returns None if it cannot be parsed.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    s = s.replace("\u00a0", "").replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


def extract_form_state(soup: BeautifulSoup) -> dict:
    """
    Extract all hidden input fields from the ASP.NET form.
    """
    form = soup.find("form")
    if form is None:
        raise RuntimeError("Form not found on page; ASP.NET page structure might be different.")

    form_data = {}
    for inp in form.find_all("input"):
        name = inp.get("name")
        if not name:
            continue
        form_data[name] = inp.get("value", "")

    return form_data


def parse_dropdown_options(soup: BeautifulSoup, select_id: str) -> dict:
    """
    Given a soup and a select id, return dict[value] = label.
    """
    select = soup.find("select", id=select_id)
    if select is None:
        raise RuntimeError(f"Select element with id '{select_id}' not found.")

    options = {}
    for opt in select.find_all("option"):
        value = opt.get("value")
        if not value:
            continue
        label = opt.get_text(strip=True)
        options[value] = label

    return options


def get_initial_state(session: requests.Session, url: str):
    """
    Initial GET to fetch viewstate, validation, and dropdown options.
    """
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    form_data = extract_form_state(soup)
    seasons = parse_dropdown_options(soup, SEASON_SELECT_ID)
    weeks = parse_dropdown_options(soup, WEEK_SELECT_ID)

    return soup, form_data, seasons, weeks


def parse_game_table(table):
    """
    Parse a single game table element into a dict with semantic fields:
    moneylines, totals, spreads, and (optionally) scores.
    """
    rows = table.find_all("tr")
    if not rows or len(rows) < 2:
        return None

    header_cells = rows[0].find_all("th")
    if len(header_cells) < 3:
        return None

    away_team = header_cells[1].get_text(strip=True)
    home_team = header_cells[2].get_text(strip=True)

    # Initialize with explicit semantic fields
    data = {
        "away_team": away_team,
        "home_team": home_team,
        "moneyline_away": None,
        "moneyline_home": None,
        "total_open": None,
        "spread_open": None,
        "total_current": None,
        "spread_current": None,
        "total_monday": None,
        "spread_monday": None,
        "total_tuesday": None,
        "spread_tuesday": None,
        "total_wednesday": None,
        "spread_wednesday": None,
        "total_thursday": None,
        "spread_thursday": None,
        "total_proj_close": None,
        "spread_proj_close": None,
        "score_away": None,
        "score_home": None,
    }

    def split_total_spread(away_text: str, home_text: str):
        """
        Given raw away/home cells for a line-type row, try to infer which
        value is the total and which is the spread. Totals are generally
        much larger in absolute value than spreads.
        Returns (total_value, spread_value) where either can be None.
        """
        away_val = parse_numeric_line(away_text)
        home_val = parse_numeric_line(home_text)

        # If both are missing, nothing to do
        if away_val is None and home_val is None:
            return None, None

        # If both present, decide by absolute magnitude
        if away_val is not None and home_val is not None:
            # Totals are usually much larger in absolute value
            if abs(away_val) > abs(home_val):
                total_val = away_val
                spread_val = home_val
            elif abs(home_val) > abs(away_val):
                total_val = home_val
                spread_val = away_val
            else:
                # Same magnitude; fall back to original assumption:
                # away = total, home = spread
                total_val = away_val
                spread_val = home_val
            return total_val, spread_val

        # Only away present
        if away_val is not None:
            if abs(away_val) >= 25:
                # Likely a total
                return away_val, None
            else:
                # Likely a spread
                return None, away_val

        # Only home present
        if home_val is not None:
            if abs(home_val) >= 25:
                return home_val, None
            else:
                return None, home_val

    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) != 3:
            continue

        label = cells[0].get_text(strip=True)
        if not label:
            continue

        away_raw = cells[1].get_text(strip=True)
        home_raw = cells[2].get_text(strip=True)

        label_lower = label.lower()

        if label_lower == "money line":
            data["moneyline_away"] = parse_moneyline(away_raw)
            data["moneyline_home"] = parse_moneyline(home_raw)

        elif label_lower == "open":
            total_val, spread_val = split_total_spread(away_raw, home_raw)
            data["total_open"] = total_val
            data["spread_open"] = spread_val

        elif label_lower == "current":
            total_val, spread_val = split_total_spread(away_raw, home_raw)
            data["total_current"] = total_val
            data["spread_current"] = spread_val

        elif label_lower == "monday":
            total_val, spread_val = split_total_spread(away_raw, home_raw)
            data["total_monday"] = total_val
            data["spread_monday"] = spread_val

        elif label_lower == "tuesday":
            total_val, spread_val = split_total_spread(away_raw, home_raw)
            data["total_tuesday"] = total_val
            data["spread_tuesday"] = spread_val

        elif label_lower == "wednesday":
            total_val, spread_val = split_total_spread(away_raw, home_raw)
            data["total_wednesday"] = total_val
            data["spread_wednesday"] = spread_val

        elif label_lower == "thursday":
            total_val, spread_val = split_total_spread(away_raw, home_raw)
            data["total_thursday"] = total_val
            data["spread_thursday"] = spread_val

        elif label_lower in ("proj move", "proj_move", "projected move", "proj. move"):
            total_val, spread_val = split_total_spread(away_raw, home_raw)
            data["total_proj_close"] = total_val
            data["spread_proj_close"] = spread_val

        elif label_lower == "score":
            # Final scores if present
            data["score_away"] = parse_numeric_line(away_raw)
            data["score_home"] = parse_numeric_line(home_raw)

        # Unknown labels are ignored silently

    return data


def scrape_all_games_for_page(soup, season_label, week_label):
    """
    From a page already loaded for a specific season/week, scrape all game tables.
    """
    tables = soup.find_all(
        "table",
        id=re.compile(r"^" + re.escape(GAME_TABLE_PREFIX) + r"\d+$")
    )

    games = []
    for table in tables:
        parsed = parse_game_table(table)
        if parsed:
            parsed["season"] = season_label
            parsed["week"] = week_label
            games.append(parsed)

    return games


def fetch_season_week(session, url, season_value, week_value):
    """
    Do a fresh GET + POST that simulates choosing a specific season + week.
    We re-GET each time to always have correct viewstate / validation.
    Slower, but far more robust for WebForms.
    """
    # First GET to get latest form state
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    form_data = extract_form_state(soup)

    # Set ASP.NET postback fields
    form_data["__EVENTTARGET"] = WEEK_SELECT_ID.replace("_", "$")
    form_data["__EVENTARGUMENT"] = ""

    # Set dropdown values (must match actual names)
    form_data["ctl00$ContentPlaceHolder1$MyWeeks$SeasonID"] = season_value
    form_data["ctl00$ContentPlaceHolder1$MyWeeks$WeekID"] = week_value

    # POST back
    resp2 = session.post(url, data=form_data, timeout=20)
    resp2.raise_for_status()

    return BeautifulSoup(resp2.text, "lxml")


def flatten_game_record(record: dict) -> dict:
    """
    Records from parse_game_table are already flat with semantic fields.
    Just return a shallow copy so callers can safely modify the result.
    """
    return dict(record)


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="NFL Line Moves Scraper", layout="wide")

st.title("NFL Line Moves Scraper")
st.write("WebForms scraping, but with a UI so you don't lose your mind.")

# URL input
base_url = st.text_input(
    "Target page URL",
    value="",
    placeholder="Paste the full URL of the page with the Season/Week dropdowns"
)

if not base_url:
    st.stop()

# Use a session to keep cookies etc.
if "session" not in st.session_state:
    st.session_state.session = requests.Session()

session = st.session_state.session

# Load seasons & weeks
if "seasons" not in st.session_state or "weeks" not in st.session_state or st.button("Reload seasons/weeks"):
    with st.spinner("Loading seasons and weeks from the page..."):
        try:
            soup_init, form_data_init, seasons, weeks = get_initial_state(session, base_url)
            st.session_state.seasons = seasons
            st.session_state.weeks = weeks
            st.success(f"Loaded {len(seasons)} seasons and {len(weeks)} weeks.")
        except Exception as e:
            st.error(f"Failed to load initial state: {e}")
            st.stop()

seasons = st.session_state.seasons
weeks = st.session_state.weeks

col1, col2 = st.columns(2)

with col1:
    season_labels = [f"{label} ({value})" for value, label in seasons.items()]
    default_seasons = season_labels  # you can slice if you want smaller default
    selected_season_labels = st.multiselect(
        "Select seasons",
        options=season_labels,
        default=default_seasons,
        help="Label format: YEAR (internal ID)"
    )

with col2:
    week_labels = [f"{label} ({value})" for value, label in weeks.items()]
    default_weeks = week_labels
    selected_week_labels = st.multiselect(
        "Select weeks",
        options=week_labels,
        default=default_weeks,
        help="Label format: WEEK_LABEL (internal ID)"
    )

# ---------------- SIDEBAR FILTERS ---------------- #
with st.sidebar:
    st.header("Signal filters")
    min_spread_move = st.number_input(
        "Min spread move (pts)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
        help="Only show spread signals where the line has moved at least this many points."
    )
    min_bet_score = st.number_input(
        "Min spread bet_score",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Filter out weaker spread signals below this score."
    )
    min_total_move = st.number_input(
        "Min total move (pts)",
        min_value=0.0,
        max_value=20.0,
        value=0.5,
        step=0.5,
        help="Only show total signals where the total has moved at least this many points."
    )
    min_total_bet_score = st.number_input(
        "Min total bet_score",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Filter out weaker totals signals below this score."
    )

    st.header("Bet profile filters")
    only_favorites = st.checkbox(
        "Only favorite-side spread bets",
        value=False,
        help="Keep only spread bets where the recommended side is the pregame favorite."
    )
    only_underdogs = st.checkbox(
        "Only underdog-side spread bets",
        value=False,
        help="Keep only spread bets where the recommended side is the pregame underdog."
    )
    home_only = st.checkbox(
        "Only home teams (spread)",
        value=False,
        help="Keep only spread bets on home teams."
    )
    away_only = st.checkbox(
        "Only away teams (spread)",
        value=False,
        help="Keep only spread bets on away teams."
    )
    hide_preseason = st.checkbox(
        "Hide preseason weeks (HOF, Pre, PS)",
        value=True,
        help="If checked, removes HOF/Pre/PS weeks from results."
    )

def parse_label_choice(choice: str):
    # "2015 (37)" -> value="37", label="2015"
    if "(" in choice and choice.endswith(")"):
        label = choice[: choice.rfind("(")].strip()
        value = choice[choice.rfind("(") + 1 : -1].strip()
        return value, label
    return None, choice


# Build lists of (value, label)
selected_seasons = [parse_label_choice(c) for c in selected_season_labels]
selected_weeks = [parse_label_choice(c) for c in selected_week_labels]

if st.button("Run scrape"):
    if not selected_seasons or not selected_weeks:
        st.warning("Pick at least one season and one week.")
        st.stop()

    all_results = []
    progress_text = st.empty()
    progress_bar = st.progress(0)

    total_tasks = len(selected_seasons) * len(selected_weeks)
    done = 0

    for season_value, season_label in selected_seasons:
        for week_value, week_label in selected_weeks:
            done += 1
            progress_text.text(f"Scraping Season {season_label}, Week {week_label} ({done}/{total_tasks})")

            try:
                soup_sw = fetch_season_week(
                    session,
                    base_url,
                    season_value,
                    week_value
                )
                games = scrape_all_games_for_page(
                    soup_sw,
                    season_label,
                    week_label
                )
                all_results.extend(games)
            except Exception as e:
                st.warning(f"Failed on Season {season_label}, Week {week_label}: {e}")
                continue

            progress_bar.progress(done / total_tasks)
            time.sleep(0.3)  # Tiny throttle

    if not all_results:
        st.error("No games found for the selected seasons/weeks.")
        st.stop()

    st.success(f"Scraped {len(all_results)} game entries.")

    # Flatten for DataFrame
    flat_records = [flatten_game_record(rec) for rec in all_results]
    df = pd.DataFrame(flat_records)

    # Optionally hide preseason weeks
    if hide_preseason and "week" in df.columns:
        df = df[~df["week"].isin(PRESEASON_WEEKS)].copy()
        if df.empty:
            st.error("All results were filtered out by preseason settings.")
            st.stop()

    # Add move columns (current - open). Result will be NaN if either side is missing.
    if {"total_open", "total_current"}.issubset(df.columns):
        df["total_move"] = df["total_current"] - df["total_open"]
    else:
        df["total_move"] = pd.NA

    if {"spread_open", "spread_current"}.issubset(df.columns):
        df["spread_move"] = df["spread_current"] - df["spread_open"]
    else:
        df["spread_move"] = pd.NA

    # Derived columns: favorite team and projection errors
    if {"spread_open", "home_team", "away_team"}.issubset(df.columns):
        df["favorite_team"] = df.apply(
            lambda r: r["home_team"]
            if pd.notna(r.get("spread_open")) and r.get("spread_open") < 0
            else r.get("away_team"),
            axis=1,
        )

    if {"total_proj_close", "total_current"}.issubset(df.columns):
        df["total_proj_error"] = df["total_current"] - df["total_proj_close"]
    else:
        df["total_proj_error"] = pd.NA

    if {"spread_proj_close", "spread_current"}.issubset(df.columns):
        df["spread_proj_error"] = df["spread_current"] - df["spread_proj_close"]
    else:
        df["spread_proj_error"] = pd.NA

    # ---------------- BET SIGNAL IDENTIFICATION ---------------- #
    # Core betting rule (refined):
    # Only flag *top-tier* opportunities where:
    # - The spread has moved at least 1 point, AND
    # - The move is in the top quartile of all moves for this scrape, AND
    # - Projection error (if available) does not strongly contradict the move.
    #
    # We also compute a bet_score to rank opportunities by strength.

    df["bet_signal"] = False
    df["bet_reason"] = ""
    df["bet_score"] = pd.NA
    # Add explicit columns for recommended bet and signal metadata
    df["recommended_side"] = pd.NA
    df["signal_type"] = ""
    df["signal_market"] = ""
    df["signal_direction"] = ""
    df["signal_strength"] = pd.NA
    # New helper columns for steam tag and bet profile
    df["steam_tag"] = ""
    df["recommended_role"] = ""
    df["recommended_loc"] = ""
    # Historical context columns for spreads
    df["spread_hist_edge_pct"] = pd.NA
    df["spread_hist_cover_pct"] = pd.NA
    df["spread_hist_games"] = pd.NA

    if "spread_move" in df.columns:
        moves_abs = df["spread_move"].abs().dropna()
        if not moves_abs.empty:
            dynamic_threshold = moves_abs.quantile(0.75)
            base_threshold = 1.0
            spread_threshold = max(base_threshold, dynamic_threshold)
        else:
            spread_threshold = None
    else:
        spread_threshold = None

    if {"spread_open", "spread_current", "spread_move"}.issubset(df.columns) and spread_threshold is not None:
        for idx, row in df.iterrows():
            move = row["spread_move"]

            if pd.isna(move):
                continue

            # Require strong directional move
            if abs(move) < spread_threshold:
                continue

            # Start with base score from magnitude of move
            score = abs(move)

            proj_err = row.get("spread_proj_error", pd.NA)
            if pd.notna(proj_err):
                # Penalize score if projection and market disagree strongly
                score = score / (1.0 + abs(proj_err))

            # Historical enhancement: adjust score slightly based on long-run
            # favorite-cover rates for similar movement magnitudes.
            hist_edge, hist_cover, hist_games = _lookup_spread_hist_edge(abs(move))
            if hist_edge is not None and hist_cover is not None and hist_games is not None:
                # Store context on the row
                df.at[idx, "spread_hist_edge_pct"] = hist_edge
                df.at[idx, "spread_hist_cover_pct"] = hist_cover
                df.at[idx, "spread_hist_games"] = hist_games

                # Only trust buckets with a reasonable sample
                if hist_games >= 50:
                    # Adjust score gently: a +5% edge becomes ~+12.5% score boost
                    adjustment_factor = 1.0 + (hist_edge / 100.0) * 0.25
                    # Cap adjustment so you don't get meme scores in tiny buckets
                    adjustment_factor = max(0.75, min(1.25, adjustment_factor))
                    score = score * adjustment_factor

            df.at[idx, "bet_signal"] = True
            df.at[idx, "bet_score"] = score

            direction = "toward favorite" if move < 0 else "toward underdog"

            # Try to suggest a side based on favorite_team and direction
            side = None
            fav = row.get("favorite_team", None)
            away = row.get("away_team", "")
            home = row.get("home_team", "")

            # Classify the type of steam for quick scanning
            steam_tag = ""
            spread_open = row.get("spread_open")
            spread_current = row.get("spread_current")

            if pd.notna(spread_open) and pd.notna(spread_current):
                if spread_open * spread_current < 0:
                    steam_tag = "FLIP_THRU_ZERO"
                elif move < 0:
                    steam_tag = "FAV_STEAM"
                else:
                    steam_tag = "DOG_STEAM"

            if isinstance(fav, str):
                if move < 0:
                    # Line moved toward the favorite
                    side = fav
                else:
                    # Line moved toward the underdog: choose the non-favorite if identifiable
                    if fav == home:
                        side = away
                    elif fav == away:
                        side = home

            # Set recommended_role and recommended_loc based on chosen side
            recommended_role = ""
            recommended_loc = ""
            if isinstance(side, str):
                if isinstance(fav, str):
                    recommended_role = "favorite" if side == fav else "underdog"
                if side == home:
                    recommended_loc = "home"
                elif side == away:
                    recommended_loc = "away"

            # Record metadata for clarity (spread-only, full-game market)
            df.at[idx, "recommended_side"] = side
            df.at[idx, "signal_type"] = "line_move"
            df.at[idx, "signal_market"] = "spread_full_game"
            df.at[idx, "signal_direction"] = "toward_favorite" if move < 0 else "toward_underdog"
            df.at[idx, "signal_strength"] = abs(move)
            df.at[idx, "steam_tag"] = steam_tag
            df.at[idx, "recommended_role"] = recommended_role
            df.at[idx, "recommended_loc"] = recommended_loc

            # Build a very explicit human-readable reason with historical context if available
            side_text = side if side else "N/A"

            hist_note = ""
            hist_edge_val = df.at[idx, "spread_hist_edge_pct"]
            hist_cover_val = df.at[idx, "spread_hist_cover_pct"]
            hist_games_val = df.at[idx, "spread_hist_games"]
            if pd.notna(hist_edge_val) and pd.notna(hist_cover_val) and pd.notna(hist_games_val):
                hist_note = (
                    f" Historical context: in NFL games where the spread moved about {abs(move):.1f} points, "
                    f"opening favorites have covered {hist_cover_val:.1f}% of the time over {int(hist_games_val)} games "
                    f"({hist_edge_val:+.1f} pts vs a 50% baseline)."
                )

            df.at[idx, "bet_reason"] = (
                f"FULL-GAME SPREAD SIGNAL: line moved {move:+.1f} points {direction} "
                f"(threshold {spread_threshold:.1f}). "
                f"Market is repricing the full-game spread, not the total. "
                f"Recommended bet side: {side_text} on the spread. "
                f"Bet score={score:.2f} (higher = stronger market conviction with lower projection conflict)."
                f"{hist_note}"
            )

    # ---------------- TOTALS BET SIGNAL IDENTIFICATION ---------------- #
    # Similar concept, but for full-game totals.
    # Only flag top-tier totals spots where:
    # - The total has moved meaningfully, AND
    # - The move is in the top quartile of all total moves for this scrape, AND
    # - Projection error (if available) does not strongly contradict the move.

    df["total_bet_signal"] = False
    df["total_bet_reason"] = ""
    df["total_bet_score"] = pd.NA
    df["total_signal_direction"] = ""
    df["total_signal_strength"] = pd.NA
    # Historical context columns for totals
    df["total_hist_edge_pct"] = pd.NA
    df["total_hist_side_pct"] = pd.NA
    df["total_hist_games"] = pd.NA

    if "total_move" in df.columns:
        total_moves_abs = df["total_move"].abs().dropna()
        if not total_moves_abs.empty:
            dynamic_total_threshold = total_moves_abs.quantile(0.75)
            base_total_threshold = 0.5  # totals usually move in smaller absolute steps
            total_threshold = max(base_total_threshold, dynamic_total_threshold)
        else:
            total_threshold = None
    else:
        total_threshold = None

    if {"total_open", "total_current", "total_move"}.issubset(df.columns) and total_threshold is not None:
        for idx, row in df.iterrows():
            tmove = row["total_move"]

            if pd.isna(tmove):
                continue

            # Require strong move relative to slate
            if abs(tmove) < total_threshold:
                continue

            # Base score from magnitude of move
            t_score = abs(tmove)

            t_proj_err = row.get("total_proj_error", pd.NA)
            if pd.notna(t_proj_err):
                # Penalize score if projection and market disagree
                t_score = t_score / (1.0 + abs(t_proj_err))

            # Direction: up = toward over, down = toward under
            direction = "toward_over" if tmove > 0 else "toward_under"
            df.at[idx, "total_signal_direction"] = direction
            df.at[idx, "total_signal_strength"] = abs(tmove)

            side_text = "Over" if tmove > 0 else "Under"

            # Historical enhancement for totals
            hist_side_key = "over" if side_text.lower() == "over" else "under"
            t_hist_edge, t_hist_side_pct, t_hist_games = _lookup_total_hist_edge(abs(tmove), hist_side_key)
            if t_hist_edge is not None and t_hist_side_pct is not None and t_hist_games is not None:
                df.at[idx, "total_hist_edge_pct"] = t_hist_edge
                df.at[idx, "total_hist_side_pct"] = t_hist_side_pct
                df.at[idx, "total_hist_games"] = t_hist_games

                if t_hist_games >= 50:
                    adjustment_factor = 1.0 + (t_hist_edge / 100.0) * 0.25
                    adjustment_factor = max(0.75, min(1.25, adjustment_factor))
                    t_score = t_score * adjustment_factor

            df.at[idx, "total_bet_signal"] = True
            df.at[idx, "total_bet_score"] = t_score

            hist_tot_note = ""
            if pd.notna(df.at[idx, "total_hist_edge_pct"]) and pd.notna(df.at[idx, "total_hist_side_pct"]) and pd.notna(df.at[idx, "total_hist_games"]):
                hist_tot_note = (
                    f" Historical context: in NFL games where the total moved about {abs(tmove):.1f} points, "
                    f"{side_text}s have hit {df.at[idx, 'total_hist_side_pct']:.1f}% of the time over "
                    f"{int(df.at[idx, 'total_hist_games'])} games "
                    f"({df.at[idx, 'total_hist_edge_pct']:+.1f} pts vs a 50% baseline)."
                )

            df.at[idx, "total_bet_reason"] = (
                f"FULL-GAME TOTAL SIGNAL: total moved {tmove:+.1f} points "
                f"{'up (toward over)' if tmove > 0 else 'down (toward under)'} "
                f"(threshold {total_threshold:.1f}). "
                f"Recommended lean: {side_text} the full-game total. "
                f"Total bet score={t_score:.2f} (higher = stronger market conviction with lower projection conflict)."
                f"{hist_tot_note}"
            )

    # ---------------- SLATE OVERVIEW SUMMARY ---------------- #
    st.subheader("Slate overview")
    n_games = len(df)
    n_spread_signals = int(df["bet_signal"].sum())
    n_total_signals = int(df["total_bet_signal"].sum())

    max_spread_move = df["signal_strength"].max(skipna=True) if "signal_strength" in df.columns else None
    max_total_move = df["total_signal_strength"].max(skipna=True) if "total_signal_strength" in df.columns else None
    max_bet_score = df["bet_score"].max(skipna=True) if "bet_score" in df.columns else None
    max_total_bet_score = df["total_bet_score"].max(skipna=True) if "total_bet_score" in df.columns else None

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Games scraped", n_games)
    col_b.metric("Spread signals", n_spread_signals)
    col_c.metric("Total signals", n_total_signals)

    col_d, col_e = st.columns(2)
    if max_spread_move is not None and pd.notna(max_spread_move):
        col_d.metric("Biggest spread move (pts)", f"{max_spread_move:.1f}")
    if max_bet_score is not None and pd.notna(max_bet_score):
        col_d.metric("Top spread bet_score", f"{max_bet_score:.2f}")
    if max_total_move is not None and pd.notna(max_total_move):
        col_e.metric("Biggest total move (pts)", f"{max_total_move:.1f}")
    if max_total_bet_score is not None and pd.notna(max_total_bet_score):
        col_e.metric("Top total bet_score", f"{max_total_bet_score:.2f}")

    if n_spread_signals == 0 and n_total_signals == 0:
        st.warning(
            "No meaningful spread or total signals detected for this slate. "
            "If you're forcing action here, that's not edge, that's boredom."
        )

    # ---------------- TOP BETTING OPPORTUNITIES DISPLAY ---------------- #
    st.markdown(
        """
        <div style="padding: 1rem; border-radius: 8px; background-color: #1f2933; border: 2px solid #ff4b4b; margin-bottom: 0.5rem;">
            <h2 style="color: #ff4b4b; margin: 0;">🚨 HIGHEST-PRIORITY BET SIGNALS 🚨</h2>
            <p style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 0.95rem;">
                These are the strongest market-backed spots from this scrape. If you ignore everything else, look here first.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **How to read this table:**
        - `signal_strength` = absolute spread move in points (current spread − open spread). Larger = bigger market adjustment.
        - `bet_score` = strength score combining move size and projection alignment. Higher = stronger market-backed edge with less model conflict.
        Use these together: start with the highest `bet_score`, and sanity-check that `signal_strength` reflects a real move (not noise).
        """.strip()
    )

    flagged = df[df["bet_signal"] == True].copy()

    # Apply interactive filters for spreads
    if not flagged.empty:
        if "signal_strength" in flagged.columns and min_spread_move > 0:
            flagged = flagged[flagged["signal_strength"].abs() >= min_spread_move]

        if "bet_score" in flagged.columns and min_bet_score > 0:
            flagged = flagged[flagged["bet_score"] >= min_bet_score]

        # Favorite vs underdog filters (only apply if exactly one is selected)
        if "recommended_role" in flagged.columns:
            if only_favorites and not only_underdogs:
                flagged = flagged[flagged["recommended_role"] == "favorite"]
            elif only_underdogs and not only_favorites:
                flagged = flagged[flagged["recommended_role"] == "underdog"]

        # Home vs away filters (only apply if exactly one is selected)
        if "recommended_loc" in flagged.columns:
            if home_only and not away_only:
                flagged = flagged[flagged["recommended_loc"] == "home"]
            elif away_only and not home_only:
                flagged = flagged[flagged["recommended_loc"] == "away"]

    if flagged.empty:
        st.info("No strong market-driven opportunities detected for selected filters.")
    else:
        # Build a very explicit recommended bet description
        def build_recommended_bet(row):
            side = row.get("recommended_side")
            spread_open = row.get("spread_open")
            spread_current = row.get("spread_current")

            if pd.isna(side):
                return "N/A"

            parts = [f"{side} full-game spread"]
            if pd.notna(spread_open) and pd.notna(spread_current):
                parts.append(
                    f"(opened at {spread_open:+.1f}, now {spread_current:+.1f})"
                )
            return " ".join(parts)

        flagged["recommended_bet"] = flagged.apply(build_recommended_bet, axis=1)

        # Sort by bet_score descending
        if "bet_score" in flagged.columns:
            flagged = flagged.sort_values("bet_score", ascending=False)

        # Choose and order the most important columns for visibility
        display_cols = [
            col for col in [
                "season",
                "week",
                "away_team",
                "home_team",
                "recommended_bet",
                "steam_tag",
                "recommended_role",
                "recommended_loc",
                "signal_type",
                "signal_market",
                "signal_direction",
                "signal_strength",
                "bet_score",
                "bet_reason",
            ] if col in flagged.columns
        ]

        flagged_display = flagged[display_cols].copy()

        # Limit to top-k so it's impossible to miss the best ones
        top_k = min(10, len(flagged_display))

        st.write(f"Showing top {top_k} opportunities (ranked by bet_score).")

        # Highlight all rows in the flagged table
        def highlight_rows(row):
            return ["background-color: #342a00;" for _ in row]

        st.dataframe(
            flagged_display.head(top_k).style.apply(highlight_rows, axis=1),
            use_container_width=True,
        )

    # ---------------- TOP TOTALS OPPORTUNITIES DISPLAY ---------------- #
    st.markdown(
        """
        <div style="padding: 1rem; border-radius: 8px; background-color: #111827; border: 2px solid #3b82f6; margin: 1rem 0 0.5rem 0;">
            <h2 style="color: #3b82f6; margin: 0;">🔥 TOP TOTALS SIGNALS 🔥</h2>
            <p style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 0.95rem;">
                These are the strongest full-game total moves on the board, ranked by conviction and projection alignment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **How to read this table:**
        - `total_signal_strength` = absolute total move in points (current total − open total). Larger = stronger move on the total.
        - `total_bet_score` = strength score for totals, combining move size and projection alignment. Higher = stronger total-side lean.
        Prioritize the highest `total_bet_score` rows, then use `total_signal_strength` to judge how aggressive the move really is.
        """.strip()
    )

    totals_flagged = df[df["total_bet_signal"] == True].copy()

    # Apply interactive filters for totals
    if not totals_flagged.empty:
        if "total_signal_strength" in totals_flagged.columns and min_total_move > 0:
            totals_flagged = totals_flagged[totals_flagged["total_signal_strength"].abs() >= min_total_move]

        if "total_bet_score" in totals_flagged.columns and min_total_bet_score > 0:
            totals_flagged = totals_flagged[totals_flagged["total_bet_score"] >= min_total_bet_score]

    if totals_flagged.empty:
        st.info("No strong totals signals detected for selected filters.")
    else:
        def build_total_bet(row):
            total_open = row.get("total_open")
            total_current = row.get("total_current")
            direction = row.get("total_signal_direction", "")

            if pd.isna(total_current):
                return "N/A"

            if direction == "toward_over":
                side_text = "Over"
            elif direction == "toward_under":
                side_text = "Under"
            else:
                side_text = "Over/Under"

            parts = [f"{side_text} full-game total"]
            if pd.notna(total_open) and pd.notna(total_current):
                parts.append(f"(opened at {total_open:.1f}, now {total_current:.1f})")
            return " ".join(parts)

        totals_flagged["recommended_total_bet"] = totals_flagged.apply(build_total_bet, axis=1)

        if "total_bet_score" in totals_flagged.columns:
            totals_flagged = totals_flagged.sort_values("total_bet_score", ascending=False)

        totals_display_cols = [
            col for col in [
                "season",
                "week",
                "away_team",
                "home_team",
                "recommended_total_bet",
                "total_signal_direction",
                "total_signal_strength",
                "total_bet_score",
                "total_bet_reason",
            ] if col in totals_flagged.columns
        ]

        totals_display = totals_flagged[totals_display_cols].copy()

        top_k_totals = min(10, len(totals_display))

        st.write(f"Showing top {top_k_totals} totals opportunities (ranked by total_bet_score).")

        def highlight_totals(row):
            return ["background-color: #001e3c;" for _ in row]

        st.dataframe(
            totals_display.head(top_k_totals).style.apply(highlight_totals, axis=1),
            use_container_width=True,
        )

    # ---------------- EXPORT SIGNALS ---------------- #
    export_spreads = df[df["bet_signal"] == True].copy()
    export_totals = df[df["total_bet_signal"] == True].copy()

    st.subheader("Export signals")
    col_x, col_y = st.columns(2)
    if not export_spreads.empty:
        col_x.download_button(
            label="Download spread signals CSV",
            data=export_spreads.to_csv(index=False).encode("utf-8"),
            file_name="spread_signals.csv",
            mime="text/csv",
        )
    else:
        col_x.write("No spread signals to export.")

    if not export_totals.empty:
        col_y.download_button(
            label="Download totals signals CSV",
            data=export_totals.to_csv(index=False).encode("utf-8"),
            file_name="totals_signals.csv",
            mime="text/csv",
        )
    else:
        col_y.write("No totals signals to export.")

    st.subheader("All scraped data")
    st.dataframe(df, use_container_width=True)