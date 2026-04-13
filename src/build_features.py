"""
Step 2: Build training dataset for next-season PPG prediction.

Creates season-pairs: features from season N -> target PPG from season N+1.

Also builds a current player roster that includes ALL dynasty-relevant players,
even those who missed the most recent season (injured, suspended, etc.) by
using their most recent season of data.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DYNASTY_POSITIONS = ["QB", "RB", "WR", "TE"]
CURRENT_SEASON = 2024


def load_raw_data():
    players = pd.read_csv(os.path.join(DATA_DIR, "players.csv"))
    seasonal = pd.read_csv(os.path.join(DATA_DIR, "seasonal_stats.csv"))
    draft = pd.read_csv(os.path.join(DATA_DIR, "draft_picks.csv"))
    snaps = pd.read_csv(os.path.join(DATA_DIR, "snap_counts.csv"))
    return players, seasonal, draft, snaps


def build_season_features(seasonal, players, draft, snaps):
    """
    For each player-season, build a feature row from that season's stats.
    """
    reg = seasonal[seasonal["season_type"] == "REG"].copy()

    # Compute PPG and per-game stats
    reg["ppg"] = reg["fantasy_points_ppr"] / reg["games"].clip(lower=1)
    g = reg["games"].clip(lower=1)
    reg["yards_per_game"] = (reg["rushing_yards"].fillna(0) + reg["receiving_yards"].fillna(0) + reg["passing_yards"].fillna(0)) / g
    reg["td_per_game"] = (reg["rushing_tds"].fillna(0) + reg["receiving_tds"].fillna(0) + reg["passing_tds"].fillna(0)) / g
    reg["receptions_per_game"] = reg["receptions"].fillna(0) / g
    reg["targets_per_game"] = reg["targets"].fillna(0) / g
    reg["carries_per_game"] = reg["carries"].fillna(0) / g

    # Efficiency
    reg["yards_per_carry"] = reg["rushing_yards"].fillna(0) / reg["carries"].clip(lower=1)
    reg["yards_per_target"] = reg["receiving_yards"].fillna(0) / reg["targets"].clip(lower=1)
    reg["catch_rate"] = reg["receptions"].fillna(0) / reg["targets"].clip(lower=1)
    reg["yards_per_reception"] = reg["receiving_yards"].fillna(0) / reg["receptions"].clip(lower=1)
    reg["yac_per_reception"] = reg["receiving_yards_after_catch"].fillna(0) / reg["receptions"].clip(lower=1)
    reg["passing_ypa"] = reg["passing_yards"].fillna(0) / reg["attempts"].clip(lower=1)
    reg["td_rate"] = (reg["rushing_tds"].fillna(0) + reg["receiving_tds"].fillna(0) + reg["passing_tds"].fillna(0)) / g

    # EPA per game
    reg["total_epa"] = reg["rushing_epa"].fillna(0) + reg["receiving_epa"].fillna(0) + reg["passing_epa"].fillna(0)
    reg["epa_per_game"] = reg["total_epa"] / g

    # Player info join
    player_info = players[players["position"].isin(DYNASTY_POSITIONS)].copy()
    player_info["birth_date"] = pd.to_datetime(player_info["birth_date"], errors="coerce")
    player_info["rookie_season"] = pd.to_numeric(player_info["rookie_season"], errors="coerce")
    player_info = player_info[["gsis_id", "position", "birth_date", "rookie_season", "height", "weight"]].copy()
    player_info = player_info.rename(columns={"gsis_id": "player_id"})

    reg = reg.merge(player_info, on="player_id", how="inner")

    # Age at the time of the season
    reg["age"] = ((pd.to_datetime(reg["season"].astype(str) + "-09-01") - reg["birth_date"]).dt.days / 365.25).round(1)
    reg["years_in_league"] = reg["season"] - reg["rookie_season"]

    # Position one-hot
    for pos in DYNASTY_POSITIONS:
        reg[f"pos_{pos}"] = (reg["position"] == pos).astype(int)

    # Draft capital
    draft_info = draft[["gsis_id", "round", "pick", "season"]].dropna(subset=["gsis_id"]).copy()
    draft_info = draft_info.rename(columns={"gsis_id": "player_id", "round": "draft_round",
                                             "pick": "draft_pick", "season": "draft_season"})
    draft_info = draft_info.drop_duplicates(subset=["player_id"], keep="first")
    reg = reg.merge(draft_info[["player_id", "draft_round", "draft_pick"]], on="player_id", how="left")

    reg["draft_round"] = reg["draft_round"].fillna(7)
    reg["draft_pick"] = reg["draft_pick"].fillna(224)
    reg["draft_capital"] = (8 - reg["draft_round"].clip(upper=7)) / 7
    reg["draft_capital_decayed"] = reg["draft_capital"] * (0.8 ** reg["years_in_league"].clip(lower=0))

    # Snap percentage (season-level aggregate)
    snap_agg = snaps.groupby(["pfr_player_id", "season"]).agg(
        total_off_snaps=("offense_snaps", "sum"),
        avg_snap_pct=("offense_pct", "mean"),
    ).reset_index()
    pfr_map = players[["gsis_id", "pfr_id"]].dropna().rename(columns={"gsis_id": "player_id"})
    snap_agg = snap_agg.merge(pfr_map, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    snap_agg = snap_agg.drop(columns=["pfr_player_id", "pfr_id"])
    reg = reg.merge(snap_agg, on=["player_id", "season"], how="left")

    # Relevance flag (for survival model)
    reg["relevant"] = ((reg["games"] >= 6) & (reg["ppg"] >= 5)).astype(int)

    # Feature columns
    feature_cols = [
        "age", "years_in_league",
        "pos_QB", "pos_RB", "pos_WR", "pos_TE",
        "games", "fantasy_points_ppr", "ppg",
        "yards_per_game", "td_per_game", "receptions_per_game",
        "targets_per_game", "carries_per_game",
        "yards_per_carry", "yards_per_target", "catch_rate",
        "yards_per_reception", "yac_per_reception", "passing_ypa",
        "td_rate", "epa_per_game",
        "target_share", "air_yards_share", "wopr_x",
        "dom", "w8dom", "yptmpa",
        "rushing_epa", "receiving_epa", "passing_epa",
        "pacr", "dakota",
        "draft_capital_decayed", "draft_round", "draft_pick",
        "total_off_snaps", "avg_snap_pct",
        "relevant",
    ]

    id_cols = ["player_id", "season", "position"]
    output = reg[id_cols + feature_cols].copy()
    output["ppg_target_col"] = reg["ppg"]

    return output


def build_training_pairs(season_features):
    """
    Create training pairs: features from season N, target = PPG from season N+1.
    """
    seasons = sorted(season_features["season"].unique())
    pairs = []

    for i in range(len(seasons) - 1):
        curr_season = seasons[i]
        next_season = seasons[i + 1]

        curr = season_features[season_features["season"] == curr_season].copy()
        next_ppg = season_features[season_features["season"] == next_season][
            ["player_id", "ppg_target_col", "games"]
        ].rename(columns={"ppg_target_col": "next_season_ppg", "games": "next_season_games"})

        merged = curr.merge(next_ppg, on="player_id", how="inner")
        merged = merged[merged["next_season_games"] >= 4].copy()
        pairs.append(merged)

    training_df = pd.concat(pairs, ignore_index=True)

    # Previous season stats for trajectory features
    prev = season_features[["player_id", "season", "ppg", "fantasy_points_ppr", "games",
                             "yards_per_game", "td_per_game", "epa_per_game"]].copy()
    prev["season"] = prev["season"] + 1
    prev.columns = ["player_id", "season"] + [f"prev_{c}" for c in prev.columns[2:]]

    training_df = training_df.merge(prev, on=["player_id", "season"], how="left")

    # YoY deltas
    training_df["ppg_delta"] = training_df["ppg"] - training_df["prev_ppg"].fillna(0)
    training_df["ypg_delta"] = training_df["yards_per_game"] - training_df["prev_yards_per_game"].fillna(0)
    training_df["epg_delta"] = training_df["epa_per_game"] - training_df["prev_epa_per_game"].fillna(0)

    return training_df


def build_current_player_features(season_features, players):
    """
    Build feature matrix for ALL dynasty-relevant current players.

    Key fix: includes players who missed the most recent season (injured, etc.)
    by using their most recent season of data with age adjusted to current.
    """
    player_info = players[players["position"].isin(DYNASTY_POSITIONS)].copy()
    # Filter by last_season instead of status (status field is unreliable —
    # nfl_data_py marks retired players like Tony Gonzalez as "ACT")
    # Allow 1 year buffer for players who missed a season due to injury
    player_info["last_season"] = pd.to_numeric(player_info["last_season"], errors="coerce")
    active_players = player_info[player_info["last_season"] >= CURRENT_SEASON - 1].copy()
    active_ids = set(active_players["gsis_id"])

    # For each active player, find their most recent season of stats
    player_seasons = season_features[season_features["player_id"].isin(active_ids)].copy()

    # Get most recent season per player
    latest = player_seasons.sort_values("season").groupby("player_id").last().reset_index()

    # Adjust age to current: add years since their last data season
    latest["seasons_missed"] = CURRENT_SEASON - latest["season"]
    latest["age"] = latest["age"] + latest["seasons_missed"]
    latest["years_in_league"] = latest["years_in_league"] + latest["seasons_missed"]

    # Decay draft capital further for missed time
    latest["draft_capital_decayed"] = latest["draft_capital_decayed"] * (0.8 ** latest["seasons_missed"])

    # For players who missed time, reduce confidence in their stats
    # Flag how stale their data is
    latest["data_staleness"] = latest["seasons_missed"]

    # Add previous season for deltas
    prev = season_features[["player_id", "season", "ppg", "fantasy_points_ppr", "games",
                             "yards_per_game", "td_per_game", "epa_per_game"]].copy()
    prev_lookup = prev.sort_values("season").groupby("player_id").nth(-2).reset_index()
    prev_lookup.columns = ["player_id"] + [f"prev_{c}" for c in prev_lookup.columns[1:]]
    latest = latest.merge(prev_lookup, on="player_id", how="left")

    latest["ppg_delta"] = latest["ppg"] - latest["prev_ppg"].fillna(0)
    latest["ypg_delta"] = latest["yards_per_game"] - latest["prev_yards_per_game"].fillna(0)
    latest["epg_delta"] = latest["epa_per_game"] - latest["prev_epa_per_game"].fillna(0)

    # Also include rookies/players drafted in 2025 who have no stats yet
    # (They'd come from draft picks data — future enhancement)

    print(f"  Players with {CURRENT_SEASON} data: {(latest['seasons_missed']==0).sum()}")
    print(f"  Players using older data (missed {CURRENT_SEASON}): {(latest['seasons_missed']>0).sum()}")

    return latest


def main():
    print("Loading raw data...")
    players, seasonal, draft, snaps = load_raw_data()

    print("Building per-season features...")
    season_features = build_season_features(seasonal, players, draft, snaps)
    print(f"  {len(season_features)} player-seasons across {season_features['season'].nunique()} seasons")

    print("Building training pairs (season N -> season N+1 PPG)...")
    training_df = build_training_pairs(season_features)
    print(f"  {len(training_df)} training samples")

    print("Building current player features...")
    current_df = build_current_player_features(season_features, players)
    print(f"  {len(current_df)} total current players")

    # Save
    season_features.to_csv(os.path.join(DATA_DIR, "season_features.csv"), index=False)
    training_df.to_csv(os.path.join(DATA_DIR, "training_data.csv"), index=False)
    current_df.to_csv(os.path.join(DATA_DIR, "current_features.csv"), index=False)
    print("\nAll feature data saved!")

    return season_features, training_df, current_df


if __name__ == "__main__":
    main()
