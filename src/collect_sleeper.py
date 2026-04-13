"""
Collect 2025 season stats from Sleeper API and convert to nfl_data_py format.

Hybrid approach: Sleeper provides current-season base stats, while nfl_data_py
provides the historical training data (2010-2024). We map Sleeper's column names
to match what build_features.py expects so the model can use 2025 data seamlessly.

Player ID matching uses 3 strategies (in order):
  1. gsis_id from Sleeper metadata (direct match)
  2. espn_id cross-reference with our players.csv
  3. Normalized name + position match (handles Jr., III, etc.)

Fields NOT available from Sleeper (will be NaN — XGBoost handles this natively):
  - EPA metrics (passing_epa, rushing_epa, receiving_epa)
  - Advanced analytics (target_share, air_yards_share, wopr_x, dom, w8dom, yptmpa, pacr, dakota)
"""
import json
import os
import re
import subprocess
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

SLEEPER_STATS_URL = "https://api.sleeper.app/v1/stats/nfl/regular/2025"
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

DYNASTY_POSITIONS = {"QB", "RB", "WR", "TE"}

# Map Sleeper stat field names → nfl_data_py seasonal_stats column names
STAT_COLUMN_MAP = {
    "pass_cmp": "completions",
    "pass_att": "attempts",
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "interceptions",
    "pass_sack": "sacks",
    "pass_sack_yds": "sack_yards",
    "pass_air_yd": "passing_air_yards",
    "pass_fd": "passing_first_downs",
    "pass_2pt": "passing_2pt_conversions",
    "rush_att": "carries",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rush_fd": "rushing_first_downs",
    "rec": "receptions",
    "rec_tgt": "targets",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "rec_air_yd": "receiving_air_yards",
    "rec_yar": "receiving_yards_after_catch",
    "rec_fd": "receiving_first_downs",
    "fum": "rushing_fumbles",  # approximate — Sleeper doesn't split by type
    "fum_lost": "rushing_fumbles_lost",
    "pts_ppr": "fantasy_points_ppr",
    "pts_std": "fantasy_points",
    "gp": "games",
}


def fetch_json(url):
    """Fetch JSON from URL using curl (avoids Python SSL issues)."""
    result = subprocess.run(
        ["curl", "-s", "-f", url],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {url}: {result.stderr}")
    return json.loads(result.stdout)


def normalize_name(name):
    """Normalize player name for fuzzy matching: lowercase, strip suffixes and punctuation."""
    name = name.lower().strip()
    name = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv|v)\b", "", name)
    name = re.sub(r"[.'\-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def build_id_lookups(players_csv):
    """Build multiple lookup dicts from players.csv for matching Sleeper → gsis_id."""
    # espn_id → gsis_id
    espn_to_gsis = {}
    for _, p in players_csv.iterrows():
        if pd.notna(p.get("espn_id")) and pd.notna(p.get("gsis_id")):
            espn_to_gsis[int(p["espn_id"])] = p["gsis_id"]

    # (normalized_name, position) → gsis_id
    name_pos_to_gsis = {}
    for _, p in players_csv.iterrows():
        if pd.notna(p.get("display_name")) and pd.notna(p.get("gsis_id")) and pd.notna(p.get("position")):
            key = (normalize_name(p["display_name"]), p["position"])
            name_pos_to_gsis[key] = p["gsis_id"]

    return espn_to_gsis, name_pos_to_gsis


def resolve_gsis_id(sleeper_info, espn_to_gsis, name_pos_to_gsis):
    """Resolve a Sleeper player to a gsis_id using 3 strategies."""
    # Strategy 1: direct gsis_id from Sleeper metadata
    gsis = sleeper_info.get("gsis_id")
    if gsis:
        return gsis, "gsis_id"

    # Strategy 2: cross-reference via espn_id
    espn = sleeper_info.get("espn_id")
    if espn and int(espn) in espn_to_gsis:
        return espn_to_gsis[int(espn)], "espn_id"

    # Strategy 3: normalized name + position
    name = sleeper_info.get("full_name", "")
    pos = sleeper_info.get("position", "")
    norm = normalize_name(name)
    if (norm, pos) in name_pos_to_gsis:
        return name_pos_to_gsis[(norm, pos)], "name"

    return None, None


def collect_sleeper_2025():
    """Fetch 2025 stats from Sleeper and save in nfl_data_py format."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Load our players.csv for ID cross-referencing ---
    players_path = os.path.join(DATA_DIR, "players.csv")
    players_csv = pd.read_csv(players_path) if os.path.exists(players_path) else pd.DataFrame()
    espn_to_gsis, name_pos_to_gsis = build_id_lookups(players_csv)

    # --- Fetch Sleeper player metadata ---
    print("  Fetching Sleeper player metadata...")
    players_raw = fetch_json(SLEEPER_PLAYERS_URL)

    # --- Fetch 2025 season stats ---
    print("  Fetching 2025 season stats from Sleeper...")
    stats_raw = fetch_json(SLEEPER_STATS_URL)

    # --- Build rows in nfl_data_py format ---
    rows = []
    match_counts = {"gsis_id": 0, "espn_id": 0, "name": 0, "unmatched": 0}

    for sleeper_id, stats in stats_raw.items():
        if sleeper_id.startswith("TEAM_"):
            continue

        info = players_raw.get(sleeper_id)
        if not info:
            continue

        pos = info.get("position")
        if pos not in DYNASTY_POSITIONS:
            continue

        gp = stats.get("gp", 0)
        if not gp or gp < 1:
            continue

        gsis_id, method = resolve_gsis_id(info, espn_to_gsis, name_pos_to_gsis)
        if not gsis_id:
            match_counts["unmatched"] += 1
            continue

        match_counts[method] += 1

        row = {
            "player_id": gsis_id,
            "season": 2025,
            "season_type": "REG",
        }

        # Map all stat columns
        for sleeper_col, nfl_col in STAT_COLUMN_MAP.items():
            row[nfl_col] = stats.get(sleeper_col, 0)

        rows.append(row)

    sleeper_df = pd.DataFrame(rows)

    print(f"    ID matching: {match_counts['gsis_id']} gsis_id, "
          f"{match_counts['espn_id']} espn_id, {match_counts['name']} name, "
          f"{match_counts['unmatched']} unmatched")
    print(f"    {len(sleeper_df)} players with 2025 stats")

    # --- Merge into existing seasonal_stats.csv ---
    seasonal_path = os.path.join(DATA_DIR, "seasonal_stats.csv")
    if os.path.exists(seasonal_path):
        existing = pd.read_csv(seasonal_path)
        # Remove any prior 2025 rows (in case we re-run)
        existing = existing[existing["season"] != 2025]
        combined = pd.concat([existing, sleeper_df], ignore_index=True)
    else:
        combined = sleeper_df

    combined.to_csv(seasonal_path, index=False)
    print(f"    seasonal_stats.csv now has {combined['season'].nunique()} seasons "
          f"({int(combined['season'].min())}-{int(combined['season'].max())})")

    # --- Update players.csv: set last_season=2025 for matched players ---
    if os.path.exists(players_path) and len(players_csv):
        active_2025 = set(sleeper_df["player_id"].unique())
        mask = players_csv["gsis_id"].isin(active_2025)
        players_csv.loc[mask, "last_season"] = 2025
        players_csv.to_csv(players_path, index=False)
        print(f"    Updated last_season=2025 for {mask.sum()} players in players.csv")

    # Save Sleeper-specific data for reference
    sleeper_df.to_csv(os.path.join(DATA_DIR, "sleeper_2025_stats.csv"), index=False)

    return sleeper_df


if __name__ == "__main__":
    print("Collecting 2025 data from Sleeper API...")
    df = collect_sleeper_2025()
    print(f"\nDone! {len(df)} players with 2025 season stats.")
