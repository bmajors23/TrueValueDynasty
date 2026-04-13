"""
Step 1: Collect raw data from nfl_data_py and save locally.

Pulls seasonal stats, player info, draft picks, snap counts, and NGS data
across all available historical seasons for training.
"""
import nfl_data_py as nfl
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Full range for seasonal stats (used for age curves + training pairs)
# 2025 data not yet available in nfl_data_py — update when it drops
STAT_SEASONS = list(range(2010, 2025))
# Snap counts available from 2013
SNAP_SEASONS = list(range(2013, 2025))
# NGS available from 2016
NGS_SEASONS = list(range(2016, 2025))

# The most recent complete season available
CURRENT_SEASON = 2024


def collect_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching player info...")
    players = nfl.import_players()
    players.to_csv(os.path.join(DATA_DIR, "players.csv"), index=False)
    print(f"  Players: {len(players)} rows")

    print(f"Fetching seasonal stats ({STAT_SEASONS[0]}-{STAT_SEASONS[-1]})...")
    seasonal = nfl.import_seasonal_data(STAT_SEASONS)
    seasonal.to_csv(os.path.join(DATA_DIR, "seasonal_stats.csv"), index=False)
    print(f"  Seasonal stats: {len(seasonal)} rows")

    print("Fetching draft picks...")
    draft = nfl.import_draft_picks()
    draft.to_csv(os.path.join(DATA_DIR, "draft_picks.csv"), index=False)
    print(f"  Draft picks: {len(draft)} rows")

    print(f"Fetching snap counts ({SNAP_SEASONS[0]}-{SNAP_SEASONS[-1]})...")
    snaps = nfl.import_snap_counts(SNAP_SEASONS)
    snaps.to_csv(os.path.join(DATA_DIR, "snap_counts.csv"), index=False)
    print(f"  Snap counts: {len(snaps)} rows")

    print(f"Fetching NGS data ({NGS_SEASONS[0]}-{NGS_SEASONS[-1]})...")
    for stat_type in ["passing", "rushing", "receiving"]:
        print(f"  NGS {stat_type}...")
        ngs = nfl.import_ngs_data(stat_type, NGS_SEASONS)
        ngs.to_csv(os.path.join(DATA_DIR, f"ngs_{stat_type}.csv"), index=False)
        print(f"    {len(ngs)} rows")

    print(f"\nAll data collected. Current season: {CURRENT_SEASON}")
    print("(Update CURRENT_SEASON when 2025 data becomes available)")


if __name__ == "__main__":
    collect_all()
