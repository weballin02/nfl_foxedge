
# bdl_streamlit_explorer.py
# Streamlit explorer for NFL games and injuries via the balldontlie Python SDK

import os
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import date, timedelta

import streamlit as st

# Optional: if the SDK isn't installed, show guidance
try:
    from balldontlie import BalldontlieAPI
    from balldontlie.exceptions import (
        BallDontLieException,
        AuthenticationError,
        RateLimitError,
        ValidationError,
        NotFoundError,
        ServerError,
    )
except Exception as e:
    BalldontlieAPI = None
    st.error("The 'balldontlie' package is not installed. Install it with:\n\npip install balldontlie")
    st.stop()


# ------------------------------
# Utilities
# ------------------------------

def get_api_key() -> str:
    """
    Resolve API key from Streamlit secrets, env var, or user input.
    """
    key = st.session_state.get("bdl_api_key", "") or os.environ.get("BDL_API_KEY", "")
    return key.strip()


def build_client(api_key: str, base_url: str = "https://api.balldontlie.io") -> BalldontlieAPI:
    """
    Instantiate the SDK client.
    """
    return BalldontlieAPI(api_key=api_key, base_url=base_url)


def to_rows(obj: Any) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Convert SDK responses to a list of dict rows plus optional meta.
    Handles both typed objects (pydantic) and raw dicts.
    """
    meta = None
    # Paginated/List responses usually carry .data and optional .meta
    if hasattr(obj, "data"):
        data = getattr(obj, "data")
        meta = getattr(obj, "meta", None)
    else:
        # Fallback: assume it is already a list or dict
        data = obj

    rows: List[Dict[str, Any]] = []

    def dump_item(x: Any) -> Dict[str, Any]:
        # pydantic BaseModel v2 has model_dump()
        if hasattr(x, "model_dump"):
            return x.model_dump()
        # dict-like
        if isinstance(x, dict):
            return x
        # Dataclass or random object: fallback to jsonable
        try:
            return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            return {"value": str(x)}

    if isinstance(data, list):
        rows = [dump_item(x) for x in data]
    elif isinstance(data, dict):
        rows = [data]
    else:
        rows = [dump_item(data)]

    # Flatten nested 'team' objects commonly seen in injuries or games
    # Keep original nested structures; Streamlit can still expand JSON.
    return rows, meta


def sanitize_kwargs(**kwargs) -> Dict[str, Any]:
    """
    Remove None/empty values so the SDK only sees meaningful params.
    Also convert empty lists to None.
    """
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 0:
            continue
        out[k] = v
    return out


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="BallDontLie NFL Explorer", layout="wide")

st.title("BallDontLie NFL Explorer")
st.caption("Review and investigate NFL games and injuries using the official Python SDK")

with st.sidebar:
    st.subheader("Connection")
    api_key_input = st.text_input(
        "API Key",
        value=get_api_key(),
        type="password",
        help="Your BallDontLie API key. Reads from BDL_API_KEY env if present."
    )
    st.session_state["bdl_api_key"] = api_key_input

    base_url = st.text_input("Base URL", value="https://api.balldontlie.io")

    st.divider()
    endpoint = st.radio(
        "Endpoint",
        options=["nfl.games", "nfl.injuries"],
        help="Choose which NFL endpoint to query"
    )

    st.subheader("Common options")
    per_page = st.number_input("Per page", value=25, min_value=1, max_value=200, step=1)
    cursor = st.text_input("Cursor (for next page)", value="")
    fetch_all = st.checkbox("Fetch all pages", value=False, help="Iterate through pages until exhausted. Respects per_page.")

    st.divider()
    st.subheader("Filters")

    if endpoint == "nfl.games":
        seasons = st.multiselect("Seasons (e.g. 2021, 2022, 2023, 2024)", [])
        weeks = st.multiselect("Weeks", [])
        team_ids = st.text_input("Team IDs (comma-separated)", value="")
        postseason = st.selectbox("Postseason", options=["Any", "True", "False"], index=0)
        use_dates = st.checkbox("Filter by date range", value=False)
        if use_dates:
            start_date = st.date_input("Start date", value=date.today() - timedelta(days=30))
            end_date = st.date_input("End date", value=date.today())
        else:
            start_date, end_date = None, None

    elif endpoint == "nfl.injuries":
        inj_team_ids = st.text_input("Team IDs (comma-separated)", value="")
        inj_player_ids = st.text_input("Player IDs (comma-separated)", value="")


col1, col2 = st.columns([1, 3], gap="large")

with col1:
    st.subheader("Query")
    if not api_key_input:
        st.warning("Provide an API key to continue.")
        st.stop()

    # Build client
    try:
        client = build_client(api_key_input, base_url=base_url)
    except Exception as e:
        st.error(f"Failed to initialize client: {e}")
        st.stop()

    query_params: Dict[str, Any] = {"per_page": per_page}
    if cursor.strip():
        query_params["cursor"] = cursor.strip()

    if endpoint == "nfl.games":
        # Seasons and weeks to int lists
        if seasons:
            try:
                seasons_list = [int(s) for s in seasons]
            except Exception:
                st.error("Seasons must be integers.")
                st.stop()
            query_params["seasons"] = seasons_list

        if weeks:
            try:
                weeks_list = [int(w) for w in weeks]
            except Exception:
                st.error("Weeks must be integers.")
                st.stop()
            query_params["weeks"] = weeks_list

        # Team IDs from CSV
        if team_ids.strip():
            try:
                ids_list = [int(x.strip()) for x in team_ids.split(",") if x.strip()]
            except Exception:
                st.error("Team IDs must be integers separated by commas.")
                st.stop()
            query_params["team_ids"] = ids_list

        # Postseason handling
        if postseason == "True":
            query_params["postseason"] = True
        elif postseason == "False":
            query_params["postseason"] = False

        # Dates list
        if start_date and end_date:
            if start_date > end_date:
                st.error("Start date must be before end date.")
                st.stop()
            date_cursor = start_date
            dates_list: List[str] = []
            while date_cursor <= end_date:
                dates_list.append(date_cursor.isoformat())
                date_cursor += timedelta(days=1)
            query_params["dates"] = dates_list

    elif endpoint == "nfl.injuries":
        if inj_team_ids.strip():
            try:
                ids_list = [int(x.strip()) for x in inj_team_ids.split(",") if x.strip()]
            except Exception:
                st.error("Team IDs must be integers separated by commas.")
                st.stop()
            query_params["team_ids"] = ids_list

        if inj_player_ids.strip():
            try:
                pids = [int(x.strip()) for x in inj_player_ids.split(",") if x.strip()]
            except Exception:
                st.error("Player IDs must be integers separated by commas.")
                st.stop()
            query_params["player_ids"] = pids

    query_params = sanitize_kwargs(**query_params)

    st.code(json.dumps({"endpoint": endpoint, "params": query_params}, indent=2), language="json")

    # Action buttons
    do_fetch = st.button("Fetch")

with col2:
    st.subheader("Results")

    @st.cache_data(show_spinner=False, ttl=300)
    def fetch_pages(endpoint: str, query_params: Dict[str, Any], fetch_all: bool) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """
        Fetch one page or all pages, returning rows, last meta, and cursors encountered.
        """
        rows: List[Dict[str, Any]] = []
        last_meta: Optional[Dict[str, Any]] = None
        cursors: List[str] = []

        # helper to call SDK
        def call_once(params: Dict[str, Any]):
            if endpoint == "nfl.games":
                return client.nfl.games.list(**params)
            elif endpoint == "nfl.injuries":
                return client.nfl.injuries.list(**params)
            else:
                raise ValueError(f"Unsupported endpoint: {endpoint}")

        # pagination loop
        current_params = dict(query_params)
        seen = 0
        max_pages = 200 if fetch_all else 1  # hard safety cap

        for _ in range(max_pages):
            resp = call_once(current_params)
            page_rows, meta = to_rows(resp)
            rows.extend(page_rows)
            last_meta = meta

            # Track cursor info if available
            if isinstance(meta, dict):
                next_cursor = meta.get("next_cursor") or meta.get("next", None)
                if next_cursor:
                    cursors.append(str(next_cursor))

            # Decide if we continue
            if not fetch_all:
                break

            # If no more cursor, break
            if not isinstance(meta, dict):
                break
            next_cursor = meta.get("next_cursor") or meta.get("next", None)
            if not next_cursor:
                break

            current_params["cursor"] = next_cursor

            # Safety bound on total rows
            seen += len(page_rows)
            if seen >= 200000:
                break

        return rows, last_meta, cursors

    if do_fetch:
        try:
            data_rows, meta, curs = fetch_pages(endpoint, query_params, fetch_all)
            st.success(f"Fetched {len(data_rows)} row(s).")
            if meta:
                with st.expander("Meta (pagination/info)", expanded=False):
                    st.json(meta)

            # Display tables
            if len(data_rows) > 0:
                st.dataframe(data_rows, use_container_width=True)

                # Exports
                import pandas as pd
                df = pd.DataFrame(data_rows)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes, file_name=f"{endpoint.replace('.','_')}.csv", mime="text/csv")

                try:
                    import io
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    table = pa.Table.from_pandas(df)
                    buf = io.BytesIO()
                    pq.write_table(table, buf, compression="snappy")
                    st.download_button("Download Parquet", data=buf.getvalue(), file_name=f"{endpoint.replace('.','_')}.parquet", mime="application/octet-stream")
                except Exception:
                    st.info("Install pyarrow to enable Parquet export: pip install pyarrow")
            else:
                st.info("No rows returned for the given filters.")

            if curs:
                with st.expander("Cursors encountered", expanded=False):
                    st.write(curs)

        except AuthenticationError as e:
            st.error(f"Authentication error: {e}")
        except RateLimitError as e:
            st.error(f"Rate limit hit: {e}")
        except ValidationError as e:
            st.error(f"Validation error: {e}")
        except NotFoundError as e:
            st.error(f"Not found: {e}")
        except ServerError as e:
            st.error(f"Server error: {e}")
        except BallDontLieException as e:
            st.error(f"API error: {e}")
        except Exception as e:
            st.exception(e)
    else:
        st.info("Set your parameters in the sidebar and click Fetch.")
