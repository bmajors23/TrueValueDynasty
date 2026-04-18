"""
Build landing-spot features for rookies — the "offensive context" of
the NFL team they land on. These are the highest-signal features after
college production for post-draft rookie rankings.

For each rookie (historical or 2026):
  - Identify their NFL team (draft_team for historical; post-draft lookup for 2026)
  - Pull that team's prior-season stats: QB quality, pass rate, offensive
    EPA-ish proxies, total receiving/rushing opportunity
  - Compute "vacated production" — what percentage of last year's team
    skill-position opportunity is now available

Inputs:
  data/combine_data.csv            — draft_team per player (full name)
  data/seasonal_stats.csv          — team+season skill-position stats
  data/college/rookie_features.csv — existing feature rows to augment

Output:
  data/college/rookie_features_with_landing.csv
     = rookie_features.csv + ~10 landing-spot features

Usage:
    python build_landing_features.py
    python build_landing_features.py --verify "Bijan"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLLEGE_DIR = DATA_DIR / "college"


# Map full team names (combine_data) → seasonal_stats abbreviations
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Oakland Raiders": "LV",            # renamed 2020
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",         # renamed 2017
    "Los Angeles Rams": "LA",
    "St. Louis Rams": "LA",              # renamed 2016
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Redskins": "WAS",        # renamed 2020 / 2022
    "Washington Football Team": "WAS",
    "Washington Commanders": "WAS",
}


def map_team(name):
    if pd.isna(name):
        return None
    return TEAM_NAME_TO_ABBR.get(str(name).strip(), None)


def build_team_year_features(seasonal_df):
    """Compute per-(team, season) offensive context features.

    Returns DataFrame indexed by (team, season) with columns:
      team_fppr, team_pass_att, team_rush_att, team_rec_yds, team_rush_yds,
      team_pass_tds, team_rush_tds, team_rec_tds, team_pass_yds,
      team_qb_best_ppg, team_pass_rate,
      vacated_fppr, vacated_targets, vacated_carries
    """
    df = seasonal_df[seasonal_df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df = df[df["recent_team"].notna() & df["season"].notna()].copy()
    df["season"] = df["season"].astype(int)

    # Fill NaN numerics with 0 for aggregations
    for c in ["fantasy_points_ppr", "games", "attempts", "completions",
              "passing_yards", "passing_tds", "carries", "rushing_yards",
              "rushing_tds", "targets", "receptions", "receiving_yards",
              "receiving_tds"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Team-season aggregates
    agg = df.groupby(["recent_team", "season"]).agg(
        team_fppr=("fantasy_points_ppr", "sum"),
        team_pass_att=("attempts", "sum"),
        team_pass_yds=("passing_yards", "sum"),
        team_pass_tds=("passing_tds", "sum"),
        team_rush_att=("carries", "sum"),
        team_rush_yds=("rushing_yards", "sum"),
        team_rush_tds=("rushing_tds", "sum"),
        team_targets=("targets", "sum"),
        team_rec=("receptions", "sum"),
        team_rec_yds=("receiving_yards", "sum"),
        team_rec_tds=("receiving_tds", "sum"),
    ).reset_index()
    agg["team_pass_rate"] = agg["team_pass_att"] / (agg["team_pass_att"] + agg["team_rush_att"]).clip(lower=1)

    # Best QB PPG per team-season (primary starter's production)
    qbs = df[df["position"] == "QB"].copy()
    qbs["ppg"] = qbs["fantasy_points_ppr"] / qbs["games"].clip(lower=1)
    qb_best = qbs.loc[qbs.groupby(["recent_team", "season"])["ppg"].idxmax()][
        ["recent_team", "season", "ppg"]
    ].rename(columns={"ppg": "team_qb_best_ppg"})
    agg = agg.merge(qb_best, on=["recent_team", "season"], how="left")

    # Compute "vacated" production: for team T going into season Y+1,
    # sum of production in Y from players NOT on team T in Y+1.
    # (Those players either left or were released.)
    vacated_rows = []
    for (team, season), sub in df.groupby(["recent_team", "season"]):
        # Who's on this team this year?
        current_roster_ids = set(sub["player_id"].tolist())
        # Who was on this team last year?
        prior = df[(df["recent_team"] == team) & (df["season"] == season - 1)]
        if prior.empty:
            vacated_rows.append({"recent_team": team, "season": season,
                                  "vacated_fppr": 0, "vacated_targets": 0,
                                  "vacated_carries": 0})
            continue
        departed = prior[~prior["player_id"].isin(current_roster_ids)]
        vacated_rows.append({
            "recent_team": team,
            "season": season,  # Y — the features apply to team T heading into Y
            "vacated_fppr": departed["fantasy_points_ppr"].sum(),
            "vacated_targets": departed["targets"].sum(),
            "vacated_carries": departed["carries"].sum(),
        })
    vacated = pd.DataFrame(vacated_rows)
    agg = agg.merge(vacated, on=["recent_team", "season"], how="left")

    return agg


def attach_landing_to_rookies(rookie_features_path, combine_path, team_year_df):
    """For each rookie, look up their draft_team's prior-season features."""
    rookies = pd.read_csv(rookie_features_path)
    combine = pd.read_csv(combine_path)

    # Build name + draft_year → draft_team mapping
    combine_slim = combine[combine["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    combine_slim["draft_team_abbr"] = combine_slim["draft_team"].apply(map_team)
    team_map = combine_slim.set_index(["player_name", "draft_year"])["draft_team_abbr"].to_dict()

    # For each rookie, look up their draft team
    rookies["landing_team"] = rookies.apply(
        lambda r: team_map.get((r["nfl_player_name"], r["nfl_draft_year"])), axis=1
    )

    # Join to team-year features from the PRIOR season (draft_year - 1).
    # That's the context they're walking into.
    rookies["_prior_season"] = (rookies["nfl_draft_year"] - 1).astype("Int64")
    team_year_ren = team_year_df.rename(columns={
        "recent_team": "landing_team",
        "season": "_prior_season",
    })
    # Prefix all landing features for clarity
    landing_cols = [c for c in team_year_ren.columns
                    if c not in ["landing_team", "_prior_season"]]
    team_year_ren = team_year_ren.rename(columns={c: f"landing_{c}" for c in landing_cols})

    merged = rookies.merge(team_year_ren, on=["landing_team", "_prior_season"], how="left")
    merged = merged.drop(columns=["_prior_season"])

    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", help="Print landing features for a player (name substring)")
    args = ap.parse_args()

    print("Loading inputs...")
    seasonal = pd.read_csv(DATA_DIR / "seasonal_stats.csv")
    print(f"  seasonal_stats: {len(seasonal):,} rows")

    print("\nComputing team-year landing context features...")
    team_year = build_team_year_features(seasonal)
    print(f"  → {len(team_year)} (team, season) rows · {len(team_year.columns)} columns")
    print(f"  team_year columns: {list(team_year.columns)}")

    print("\nAttaching landing features to rookies...")
    rookies_with_landing = attach_landing_to_rookies(
        COLLEGE_DIR / "rookie_features.csv",
        DATA_DIR / "combine_data.csv",
        team_year,
    )

    total = len(rookies_with_landing)
    has_landing = rookies_with_landing["landing_team_fppr"].notna().sum()
    print(f"  {has_landing}/{total} rookies have landing features ({has_landing/total:.1%})")

    # Save
    out = COLLEGE_DIR / "rookie_features_with_landing.csv"
    rookies_with_landing.to_csv(out, index=False)
    print(f"\n✅ Saved → {out}")
    print(f"   {len(rookies_with_landing.columns)} total columns")

    if args.verify:
        target = rookies_with_landing[
            rookies_with_landing["nfl_player_name"].str.contains(args.verify, case=False, na=False)
        ]
        if len(target) == 0:
            print(f"\nNo player matching '{args.verify}'")
        else:
            for _, row in target.iterrows():
                print(f"\n{'=' * 60}")
                print(f"  {row['nfl_player_name']} ({row['nfl_pos']}, draft {int(row['nfl_draft_year'])})")
                print(f"  Landing team: {row['landing_team']}")
                print(f"{'=' * 60}")
                landing_cols = [c for c in row.index if c.startswith("landing_")]
                for c in landing_cols:
                    val = row[c]
                    if pd.notna(val):
                        print(f"  {c:35} = {val:.2f}" if isinstance(val, float) else f"  {c:35} = {val}")


if __name__ == "__main__":
    main()
