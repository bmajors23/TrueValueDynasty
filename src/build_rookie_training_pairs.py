"""
Build rookie model training pairs: features → Year-1 NFL PPG target.

For each rookie in rookie_features.csv:
  - Find their Year-1 NFL season (= nfl_draft_year)
  - Pull actual fantasy_points_ppr and games from seasonal_stats.csv
  - Compute Year-1 PPG = fantasy_points_ppr / max(games, 1)
  - Also compute Year-2 PPG (for optional multi-task training)
  - Mark players who never played as never_played=1 (excluded from training
    by default but saved for inspection)

Inputs:
  data/college/rookie_features.csv   — 756 players × 62 features
  data/seasonal_stats.csv            — NFL seasonal stats (nflverse)

Output:
  data/college/rookie_training_pairs.csv — features + year1_ppg target

Usage:
    python build_rookie_training_pairs.py
    python build_rookie_training_pairs.py --verify Bijan
"""
import argparse
import re
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLLEGE_DIR = DATA_DIR / "college"


def normalize(text):
    """Name normalization — same logic as link_college_to_nfl.py."""
    if pd.isna(text) or not text:
        return ""
    s = str(text).lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?\s*$", "", s)
    s = re.sub(r"[.'\-,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_ppg_row(stats_row):
    """Compute PPG from a seasonal stats row."""
    g = stats_row.get("games", 0) or 0
    fp = stats_row.get("fantasy_points_ppr", 0) or 0
    if g <= 0:
        return np.nan
    return fp / g


def find_year_ppg(seasonal_df, player_name_norm, season, pos=None):
    """Find a player's PPG + games for a given NFL season.

    Returns (ppg, games, fp_ppr) or (np.nan, 0, 0) if not found.
    Name match is exact on normalized name + season; position filter is
    a soft check (we use it to disambiguate collisions, not to hard-gate).
    """
    # Primary match: normalized name + season
    cand = seasonal_df[
        (seasonal_df["name_norm"] == player_name_norm) &
        (seasonal_df["season"] == season)
    ]
    if len(cand) == 0:
        return np.nan, 0, 0.0

    # If multiple rows (name collision on that year), prefer the positional match
    if len(cand) > 1 and pos:
        pos_match = cand[cand["position"] == pos]
        if len(pos_match) >= 1:
            cand = pos_match

    # Take first row. If still multiple, take the one with most games.
    if len(cand) > 1:
        cand = cand.sort_values("games", ascending=False)

    row = cand.iloc[0]
    games = int(row.get("games", 0) or 0)
    fp = float(row.get("fantasy_points_ppr", 0) or 0)
    ppg = fp / games if games > 0 else np.nan
    return ppg, games, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", help="Print training pair for player (name substring)")
    args = ap.parse_args()

    print("Loading inputs...")
    features = pd.read_csv(COLLEGE_DIR / "rookie_features.csv")
    seasonal = pd.read_csv(DATA_DIR / "seasonal_stats.csv")
    print(f"  rookie features: {len(features)} players")
    print(f"  seasonal stats:  {len(seasonal):,} rows, {seasonal['season'].min()}-{seasonal['season'].max()}")

    # Normalize name for matching (use display_name — 'player_display_name' is cleaner than 'player_name')
    name_col = "player_display_name" if "player_display_name" in seasonal.columns else "player_name"
    seasonal["name_norm"] = seasonal[name_col].apply(normalize)
    features["name_norm"] = features["nfl_player_name"].apply(normalize)

    # Pull Year-1 and Year-2 outcomes for each rookie
    print("\nJoining Year-1 NFL outcomes...")
    year1_ppg = []
    year1_games = []
    year1_fp = []
    year2_ppg = []
    year2_games = []
    never_played = []

    for _, row in features.iterrows():
        name_n = row["name_norm"]
        draft_yr = int(row["nfl_draft_year"]) if pd.notna(row["nfl_draft_year"]) else None
        pos = row["nfl_pos"]
        if draft_yr is None:
            year1_ppg.append(np.nan); year1_games.append(0); year1_fp.append(0.0)
            year2_ppg.append(np.nan); year2_games.append(0); never_played.append(1)
            continue
        ppg1, g1, fp1 = find_year_ppg(seasonal, name_n, draft_yr, pos)
        ppg2, g2, _  = find_year_ppg(seasonal, name_n, draft_yr + 1, pos)
        year1_ppg.append(ppg1)
        year1_games.append(g1)
        year1_fp.append(fp1)
        year2_ppg.append(ppg2)
        year2_games.append(g2)
        # "Never played" = 0 games in Year 1 AND Year 2
        never_played.append(int(g1 == 0 and g2 == 0))

    features["year1_ppg"] = year1_ppg
    features["year1_games"] = year1_games
    features["year1_fp_ppr"] = year1_fp
    features["year2_ppg"] = year2_ppg
    features["year2_games"] = year2_games
    features["never_played"] = never_played

    # Summary
    total = len(features)
    with_y1 = (features["year1_games"] > 0).sum()
    never = features["never_played"].sum()
    limited = ((features["year1_games"] > 0) & (features["year1_games"] < 4)).sum()

    print(f"\nYear-1 outcome coverage:")
    print(f"  Total rookies:            {total}")
    print(f"  Played Year 1 (≥1 game):  {with_y1} ({with_y1/total:.1%})")
    print(f"  Never played (Y1+Y2=0g):  {never} ({never/total:.1%})")
    print(f"  Limited Y1 (1-3 games):   {limited}")

    # Position breakdown of Year-1 PPG distribution
    print(f"\nYear-1 PPG distribution (players with ≥4 games):")
    played = features[features["year1_games"] >= 4]
    print(played.groupby("nfl_pos")["year1_ppg"].agg(["count", "mean", "median", "std", "max"]).round(2))

    # Save
    out = COLLEGE_DIR / "rookie_training_pairs.csv"
    features.to_csv(out, index=False)
    print(f"\n✅ Saved {len(features)} training pairs → {out}")
    print(f"   For model training, filter to year1_games >= 4 → {(features['year1_games'] >= 4).sum()} training rows")

    # Show some extreme cases
    print(f"\nTop 10 Year-1 PPG performers (proof the data is right):")
    top = features[features["year1_games"] >= 4].nlargest(10, "year1_ppg")
    for _, r in top.iterrows():
        print(f"  {r['nfl_player_name']:25} {r['nfl_pos']} {int(r['nfl_draft_year'])}  "
              f"{r['year1_games']:2}g  {r['year1_ppg']:.2f} PPG")

    if args.verify:
        target = features[features["nfl_player_name"].str.contains(args.verify, case=False, na=False)]
        if len(target) > 0:
            print(f"\nVerification — {args.verify}:")
            for _, r in target.iterrows():
                print(f"  {r['nfl_player_name']} ({r['nfl_pos']}, drafted {int(r['nfl_draft_year'])})")
                print(f"    Y1: {r['year1_games']} games, {r['year1_fp_ppr']:.1f} fp, {r['year1_ppg']:.2f} PPG")
                print(f"    Y2: {r['year2_games']} games, {r['year2_ppg']:.2f} PPG" if pd.notna(r['year2_ppg']) else "    Y2: N/A")


if __name__ == "__main__":
    main()
