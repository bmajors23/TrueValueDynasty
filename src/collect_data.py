"""
Step 1: Collect raw data from nflverse and nfl_data_py, save locally.

Uses nflverse-data parquet files directly for player stats (2010-2025),
which provides EPA, target share, WOPR, and other advanced metrics
that nfl_data_py's import_seasonal_data() was missing for 2025.

Still uses nfl_data_py for player info, draft picks, and snap counts
(these are stable and don't change year-to-year).
"""
import nfl_data_py as nfl
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# nflverse-data parquet base URL
NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player"

# Full range for seasonal stats
STAT_SEASONS = list(range(2010, 2026))
# Snap counts available from 2013
SNAP_SEASONS = list(range(2013, 2025))
# NGS available from 2016
NGS_SEASONS = list(range(2016, 2025))

CURRENT_SEASON = 2025


def fetch_nflverse_stats(seasons):
    """Fetch per-season player stats directly from nflverse parquet files.

    These have EPA, target_share, air_yards_share, wopr, pacr, racr
    which nfl_data_py's import_seasonal_data() doesn't include.
    """
    dfs = []
    for season in seasons:
        url = f"{NFLVERSE_BASE}/stats_player_reg_{season}.parquet"
        try:
            df = pd.read_parquet(url)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not fetch {season} stats: {e}")
    combined = pd.concat(dfs, ignore_index=True)

    # Rename columns to match what build_features.py expects
    # (nflverse uses slightly different names than nfl_data_py)
    combined = combined.rename(columns={
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
        "sack_yards_lost": "sack_yards",
        "wopr": "wopr_x",
        "passing_cpoe": "dakota",  # closest equivalent
    })

    # Add season_type column for compatibility
    if "season_type" not in combined.columns:
        combined["season_type"] = "REG"

    return combined


def collect_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching player info...")
    players = nfl.import_players()
    players.to_csv(os.path.join(DATA_DIR, "players.csv"), index=False)
    print(f"  Players: {len(players)} rows")

    print(f"Fetching seasonal stats from nflverse ({STAT_SEASONS[0]}-{STAT_SEASONS[-1]})...")
    seasonal = fetch_nflverse_stats(STAT_SEASONS)
    seasonal.to_csv(os.path.join(DATA_DIR, "seasonal_stats.csv"), index=False)
    print(f"  Seasonal stats: {len(seasonal)} rows, {seasonal['season'].nunique()} seasons")

    # Update last_season for players with 2025 data
    s2025_ids = set(seasonal[seasonal["season"] == 2025]["player_id"].unique())
    mask = players["gsis_id"].isin(s2025_ids)
    players.loc[mask, "last_season"] = 2025
    players.to_csv(os.path.join(DATA_DIR, "players.csv"), index=False)
    print(f"  Updated last_season=2025 for {mask.sum()} players")

    print("Fetching draft picks...")
    draft = nfl.import_draft_picks()
    draft.to_csv(os.path.join(DATA_DIR, "draft_picks.csv"), index=False)
    print(f"  Draft picks: {len(draft)} rows")

    print(f"Fetching snap counts ({SNAP_SEASONS[0]}-{SNAP_SEASONS[-1]})...")
    snaps = nfl.import_snap_counts(SNAP_SEASONS)
    snaps.to_csv(os.path.join(DATA_DIR, "snap_counts.csv"), index=False)
    print(f"  Snap counts: {len(snaps)} rows")

    print("Fetching weekly player stats from nflverse...")
    weekly_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet"
    try:
        weekly = pd.read_parquet(weekly_url)
        weekly = weekly[weekly["season"] >= 2010]
        weekly = weekly[weekly["season_type"] == "REG"]
        weekly.to_csv(os.path.join(DATA_DIR, "weekly_stats.csv"), index=False)
        print(f"  Weekly stats: {len(weekly)} rows, {weekly['season'].nunique()} seasons")
    except Exception as e:
        print(f"  Warning: could not fetch weekly stats: {e}")

    print(f"Fetching NGS data ({NGS_SEASONS[0]}-{NGS_SEASONS[-1]})...")
    for stat_type in ["passing", "rushing", "receiving"]:
        print(f"  NGS {stat_type}...")
        ngs = nfl.import_ngs_data(stat_type, NGS_SEASONS)
        ngs.to_csv(os.path.join(DATA_DIR, f"ngs_{stat_type}.csv"), index=False)
        print(f"    {len(ngs)} rows")

    print(f"\nAll data collected. Current season: {CURRENT_SEASON}")


if __name__ == "__main__":
    collect_all()
