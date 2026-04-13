"""
Step 1b: Collect KTC (KeepTradeCut) dynasty values to use as target variable.

KTC embeds player data as a JavaScript array in their rankings page.
We extract it and save as CSV.
"""
import json
import re
import subprocess
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def fetch_ktc_values():
    """Scrape KTC dynasty rankings from the embedded playersArray."""
    url = "https://keeptradecut.com/dynasty-rankings"

    print("Fetching KTC dynasty rankings page...")
    # Use curl to avoid Python 3.9 SSL issues with LibreSSL
    result = subprocess.run(
        ["curl", "-s", url, "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    html = result.stdout

    # Extract the playersArray JSON from the page
    match = re.search(r"var\s+playersArray\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if not match:
        raise ValueError("Could not find playersArray in KTC page. Site structure may have changed.")

    print("Parsing player data...")
    players_json = json.loads(match.group(1))

    rows = []
    for p in players_json:
        row = {
            "ktc_player_name": p.get("playerName", ""),
            "ktc_id": p.get("playerID", ""),
            "ktc_slug": p.get("slug", ""),
            "team": p.get("team", ""),
            "position": p.get("position", ""),
            "age": p.get("age", None),
            "rookie": p.get("rookie", False),
            "college": p.get("college", ""),
            "draft_year": p.get("draftYear", None),
            # Values - oneQB format
            "ktc_value_1qb": p.get("oneQBValues", {}).get("value", 0) if isinstance(p.get("oneQBValues"), dict) else 0,
            "ktc_rank_1qb": p.get("oneQBValues", {}).get("rank", 0) if isinstance(p.get("oneQBValues"), dict) else 0,
            # Values - superflex format
            "ktc_value_sf": p.get("superflexValues", {}).get("value", 0) if isinstance(p.get("superflexValues"), dict) else 0,
            "ktc_rank_sf": p.get("superflexValues", {}).get("rank", 0) if isinstance(p.get("superflexValues"), dict) else 0,
            # Trend data
            "ktc_trend_1qb": p.get("oneQBValues", {}).get("overallTrend", 0) if isinstance(p.get("oneQBValues"), dict) else 0,
            "ktc_trend_sf": p.get("superflexValues", {}).get("overallTrend", 0) if isinstance(p.get("superflexValues"), dict) else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Filter to dynasty-relevant positions
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df = df[df["ktc_value_sf"] > 0].copy()

    print(f"Collected {len(df)} player KTC values")
    print(f"  Value range (SF): {df['ktc_value_sf'].min()} - {df['ktc_value_sf'].max()}")
    print(f"  Positions: {df['position'].value_counts().to_dict()}")

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "ktc_values.csv"), index=False)
    print(f"Saved to {os.path.join(DATA_DIR, 'ktc_values.csv')}")

    return df


if __name__ == "__main__":
    fetch_ktc_values()
