#!/usr/bin/env python3
"""
Fetch live NFL Injury Report data from TheInjuryExpertz public Google Sheet.
"""

import pandas as pd
from datetime import datetime
import os
import sys

# Public Google Sheet endpoint (already verified)
SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRqhxPKXwyMUvFswaA3ru1Cd87-PtgcyoLKjPvdG5O87e78UabD8XyVMgGET1fw6zLKo3HuhazIKj6u/"
    "pub?output=csv"
)

def fetch_injury_data(url: str = SHEET_URL) -> pd.DataFrame:
    """Download and clean the injury report CSV from Google Sheets."""
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}", file=sys.stderr)
        sys.exit(1)

    # Basic cleanup
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all")

    # Normalize column naming conventions if present
    rename_map = {
        "name": "player",
        "team": "team",
        "injury": "injury",
        "status": "status",
        "in/out": "status",
        "level_of_concern": "concern_level"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Optional cleanup
    if "concern_level" in df.columns:
        df["concern_level"] = df["concern_level"].astype(str).str.title()

    return df


def main(save_csv: bool = True) -> None:
    df = fetch_injury_data()

    print(f"✅ Successfully fetched {len(df)} rows of injury data.")

    # Print preview
    print(df.head())

    if save_csv:
        out_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"nfl_injury_report_{datetime.now():%Y%m%d}.csv")
        df.to_csv(out_path, index=False)
        print(f"💾 Saved to: {out_path}")


if __name__ == "__main__":
    main()
