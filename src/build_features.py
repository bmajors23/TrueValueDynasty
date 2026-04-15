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
CURRENT_SEASON = 2025


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

    # === Volume / Opportunity features ===
    # Total touches per game (carries + receptions) — measures opportunity
    reg["touches_per_game"] = (reg["carries"].fillna(0) + reg["receptions"].fillna(0)) / g
    # Total opportunities per game (carries + targets) — measures involvement
    reg["opportunities_per_game"] = (reg["carries"].fillna(0) + reg["targets"].fillna(0)) / g
    # Raw volume totals (XGBoost can learn that 350 carries means a workhorse)
    reg["total_touches"] = reg["carries"].fillna(0) + reg["receptions"].fillna(0)
    reg["total_opportunities"] = reg["carries"].fillna(0) + reg["targets"].fillna(0)

    # === First down / efficiency features ===
    reg["first_downs_per_game"] = (
        reg["rushing_first_downs"].fillna(0) +
        reg["receiving_first_downs"].fillna(0) +
        reg["passing_first_downs"].fillna(0)
    ) / g
    reg["receiving_air_yards_per_game"] = reg["receiving_air_yards"].fillna(0) / g
    reg["racr"] = reg.get("racr", pd.Series(dtype=float))

    # === Team offensive context ===
    # Derive team offensive quality from total team fantasy points per game
    # This lets the model know if a player is on a high-powered offense
    team_season_fpts = reg.groupby(["season", "recent_team"])["fantasy_points_ppr"].sum().reset_index()
    team_season_fpts.columns = ["season", "recent_team", "team_total_fpts"]
    # Normalize per-season to a 0-1 scale (relative to other teams that year)
    team_season_fpts["team_offense_rank_pct"] = team_season_fpts.groupby("season")["team_total_fpts"].rank(pct=True)
    reg = reg.merge(team_season_fpts[["season", "recent_team", "team_total_fpts", "team_offense_rank_pct"]],
                     on=["season", "recent_team"], how="left")
    reg["team_fpts_per_game"] = reg["team_total_fpts"] / 17  # approx games per team

    # Player's share of team production — measures how central they are
    reg["team_production_share"] = reg["fantasy_points_ppr"] / reg["team_total_fpts"].clip(lower=1)

    # Filter to dynasty positions (nflverse data includes position column)
    if "position" in reg.columns:
        reg = reg[reg["position"].isin(DYNASTY_POSITIONS)].copy()

    # Player info join (for birth_date, rookie_season — position already in stats)
    player_info = players[players["position"].isin(DYNASTY_POSITIONS)].copy()
    player_info["birth_date"] = pd.to_datetime(player_info["birth_date"], errors="coerce")
    player_info["rookie_season"] = pd.to_numeric(player_info["rookie_season"], errors="coerce")
    player_info = player_info[["gsis_id", "birth_date", "rookie_season", "height", "weight"]].copy()
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
        # Volume / opportunity
        "touches_per_game", "opportunities_per_game",
        "total_touches", "total_opportunities",
        # First downs / air yards
        "first_downs_per_game", "receiving_air_yards_per_game", "racr",
        # Team context
        "team_offense_rank_pct", "team_fpts_per_game", "team_production_share",
        # Game-level consistency (from weekly data)
        "ppg_std", "ppg_cv", "ppg_floor", "ppg_ceiling", "ceiling_floor_ratio",
        "boom_games_pct", "bust_games_pct",
        # Trend / momentum (from weekly data)
        "late_season_trend", "last_4_avg", "momentum",
        "target_trend", "carry_trend", "target_share_trend",
        # Situation features
        "qb_quality", "team_pass_rate", "positional_dominance", "is_positional_alpha",
        # Rookie / combine features
        "forty_time", "combine_weight", "height_inches",
        "vertical_jump", "broad_jump", "bench_reps",
        "athleticism_score", "draft_capital_score",
        "forty_time_zscore", "combine_weight_zscore", "height_inches_zscore",
        "vertical_jump_zscore", "broad_jump_zscore", "bench_reps_zscore",
    ]

    id_cols = ["player_id", "season", "position"]
    # Only include feature columns that exist (Sleeper 2025 data won't have
    # EPA or advanced analytics — XGBoost handles NaN natively)
    available_features = [c for c in feature_cols if c in reg.columns]
    missing_features = [c for c in feature_cols if c not in reg.columns]
    output = reg[id_cols + available_features].copy()
    for col in missing_features:
        output[col] = np.nan
    output["ppg_target_col"] = reg["ppg"]

    return output


def build_weekly_features(weekly_stats, players):
    """Build game-level features from weekly data that seasonal aggregates miss.

    These capture consistency, trends, and usage patterns that per-season
    averages hide. Each feature is aggregated to the player-season level.
    """
    weekly = weekly_stats.copy()

    # Filter to dynasty positions and regular season
    if "position" in weekly.columns:
        weekly = weekly[weekly["position"].isin(DYNASTY_POSITIONS)].copy()

    # Only weeks where the player had some activity
    weekly = weekly[weekly["fantasy_points_ppr"] > 0].copy()

    features_list = []
    for (pid, season), games in weekly.groupby(["player_id", "season"]):
        if len(games) < 4:
            continue  # need minimum sample

        fpts = games["fantasy_points_ppr"].values
        n_games = len(fpts)

        row = {"player_id": pid, "season": season}

        # --- Consistency features ---
        row["ppg_std"] = np.std(fpts)
        row["ppg_cv"] = np.std(fpts) / np.mean(fpts) if np.mean(fpts) > 0 else 1.0
        row["ppg_floor"] = np.percentile(fpts, 10)
        row["ppg_ceiling"] = np.percentile(fpts, 90)
        row["ceiling_floor_ratio"] = row["ppg_ceiling"] / max(row["ppg_floor"], 0.1)
        row["boom_games_pct"] = (fpts >= 20).sum() / n_games  # boom = 20+ PPR pts
        row["bust_games_pct"] = (fpts < 5).sum() / n_games    # bust = <5 PPR pts

        # --- Trend features (late season vs early season) ---
        if n_games >= 8:
            half = n_games // 2
            first_half = np.mean(fpts[:half])
            second_half = np.mean(fpts[half:])
            row["late_season_trend"] = second_half - first_half
            # Last 4 games momentum
            row["last_4_avg"] = np.mean(fpts[-4:])
            row["first_4_avg"] = np.mean(fpts[:4])
            row["momentum"] = row["last_4_avg"] - row["first_4_avg"]
        else:
            row["late_season_trend"] = 0
            row["last_4_avg"] = np.mean(fpts[-4:]) if n_games >= 4 else np.mean(fpts)
            row["first_4_avg"] = np.mean(fpts[:4]) if n_games >= 4 else np.mean(fpts)
            row["momentum"] = 0

        # --- Usage trend features ---
        if "targets" in games.columns:
            targets = games["targets"].fillna(0).values
            if n_games >= 8:
                half = n_games // 2
                row["target_trend"] = np.mean(targets[half:]) - np.mean(targets[:half])
            else:
                row["target_trend"] = 0

        if "carries" in games.columns:
            carries = games["carries"].fillna(0).values
            if n_games >= 8:
                half = n_games // 2
                row["carry_trend"] = np.mean(carries[half:]) - np.mean(carries[:half])
            else:
                row["carry_trend"] = 0

        # --- Target share trend ---
        if "target_share" in games.columns:
            ts = games["target_share"].fillna(0).values
            row["target_share_avg"] = np.mean(ts)
            if n_games >= 8:
                half = n_games // 2
                row["target_share_trend"] = np.mean(ts[half:]) - np.mean(ts[:half])
            else:
                row["target_share_trend"] = 0

        features_list.append(row)

    if not features_list:
        return pd.DataFrame()

    weekly_features = pd.DataFrame(features_list)
    print(f"  Weekly features computed for {len(weekly_features)} player-seasons")
    return weekly_features


def build_situation_features(seasonal_stats, players):
    """Build situation-aware features from existing data.

    - QB quality index: how good is the QB on this player's team?
    - Team pass/run tendency: scheme indicator
    - Opportunity concentration: is this player the alpha on their team?
    """
    reg = seasonal_stats[seasonal_stats["season_type"] == "REG"].copy()
    reg["ppg"] = reg["fantasy_points_ppr"] / reg["games"].clip(lower=1)

    features_list = []

    for season in reg["season"].unique():
        season_data = reg[reg["season"] == season]

        # Compute QB quality per team
        qb_data = season_data[season_data["position"] == "QB"]
        qb_quality = {}
        for team in season_data["recent_team"].unique():
            team_qbs = qb_data[qb_data["recent_team"] == team]
            if len(team_qbs) > 0:
                # Best QB on the team (by fantasy points)
                best_qb = team_qbs.sort_values("fantasy_points_ppr", ascending=False).iloc[0]
                qb_quality[team] = best_qb["ppg"]
            else:
                qb_quality[team] = 0

        # Team pass/run ratio
        team_tendency = {}
        for team in season_data["recent_team"].unique():
            team_players = season_data[season_data["recent_team"] == team]
            total_pass_yards = team_players["passing_yards"].fillna(0).sum()
            total_rush_yards = team_players["rushing_yards"].fillna(0).sum()
            total_yards = total_pass_yards + total_rush_yards
            if total_yards > 0:
                team_tendency[team] = total_pass_yards / total_yards  # pass rate
            else:
                team_tendency[team] = 0.5

        # Per-player situation features
        for _, player in season_data.iterrows():
            team = player["recent_team"]
            pos = player["position"]
            if pos not in DYNASTY_POSITIONS:
                continue

            row = {
                "player_id": player["player_id"],
                "season": season,
                "qb_quality": qb_quality.get(team, 0),
                "team_pass_rate": team_tendency.get(team, 0.5),
            }

            # Opportunity concentration: this player's share vs next best at same position
            same_pos_team = season_data[
                (season_data["recent_team"] == team) &
                (season_data["position"] == pos)
            ].sort_values("fantasy_points_ppr", ascending=False)

            if len(same_pos_team) >= 2:
                top_fpts = same_pos_team.iloc[0]["fantasy_points_ppr"]
                second_fpts = same_pos_team.iloc[1]["fantasy_points_ppr"]
                # How dominant is the #1 at this position on this team?
                row["positional_dominance"] = top_fpts / max(top_fpts + second_fpts, 1)
                # Is this player THE guy?
                row["is_positional_alpha"] = 1 if player["player_id"] == same_pos_team.iloc[0]["player_id"] else 0
            else:
                row["positional_dominance"] = 1.0
                row["is_positional_alpha"] = 1

            features_list.append(row)

    if not features_list:
        return pd.DataFrame()

    situation_features = pd.DataFrame(features_list)
    print(f"  Situation features computed for {len(situation_features)} player-seasons")
    return situation_features


def build_rookie_features(players, draft, combine):
    """Build rookie-specific features from combine + draft data.

    For players with <2 NFL seasons, these features provide signal that
    seasonal stats can't — athletic profile, draft capital context, and
    combine performance relative to position peers.
    """
    # Map combine data to gsis_id via pfr_id
    pfr_map = players[["gsis_id", "pfr_id"]].dropna().rename(
        columns={"gsis_id": "player_id"}
    )

    combine_slim = combine[combine["pos"].isin(DYNASTY_POSITIONS)].copy()
    combine_slim = combine_slim.rename(columns={
        "pfr_id": "pfr_id_combine",
        "pos": "combine_pos",
        "ht": "combine_height",
        "wt": "combine_weight",
        "forty": "forty_time",
        "bench": "bench_reps",
        "vertical": "vertical_jump",
        "broad_jump": "broad_jump",
        "cone": "cone_drill",
        "shuttle": "shuttle_time",
        "season": "combine_season",
        "draft_round": "combine_draft_round",
        "draft_ovr": "combine_draft_ovr",
    })

    # Parse height to inches
    def parse_height(h):
        if pd.isna(h) or not isinstance(h, str):
            return np.nan
        parts = str(h).split("-")
        if len(parts) == 2:
            try:
                return int(parts[0]) * 12 + int(parts[1])
            except ValueError:
                return np.nan
        return np.nan

    combine_slim["height_inches"] = combine_slim["combine_height"].apply(parse_height)

    # Merge with player IDs via pfr_id
    combine_slim = combine_slim.merge(
        pfr_map, left_on="pfr_id_combine", right_on="pfr_id", how="inner"
    ).drop(columns=["pfr_id"])

    # Compute position-relative athleticism scores
    # (how does this player compare to others at their position?)
    for metric in ["forty_time", "combine_weight", "vertical_jump", "broad_jump",
                   "bench_reps", "height_inches"]:
        if metric in combine_slim.columns:
            pos_stats = combine_slim.groupby("combine_pos")[metric].agg(["mean", "std"])
            for pos in DYNASTY_POSITIONS:
                if pos in pos_stats.index:
                    mask = combine_slim["combine_pos"] == pos
                    mean_val = pos_stats.loc[pos, "mean"]
                    std_val = pos_stats.loc[pos, "std"]
                    if std_val > 0:
                        # Z-score relative to position (flip forty so higher = better)
                        if metric == "forty_time":
                            combine_slim.loc[mask, f"{metric}_zscore"] = (
                                mean_val - combine_slim.loc[mask, metric]
                            ) / std_val
                        else:
                            combine_slim.loc[mask, f"{metric}_zscore"] = (
                                combine_slim.loc[mask, metric] - mean_val
                            ) / std_val

    # Compute composite athleticism score
    zscore_cols = [c for c in combine_slim.columns if c.endswith("_zscore")]
    if zscore_cols:
        combine_slim["athleticism_score"] = combine_slim[zscore_cols].mean(axis=1)

    # Draft capital score (more granular than the existing draft_round)
    # Top 10 pick = elite capital, picks 11-32 = first round, etc.
    combine_slim["draft_capital_score"] = np.where(
        combine_slim["combine_draft_ovr"] <= 10, 1.0,
        np.where(combine_slim["combine_draft_ovr"] <= 32, 0.8,
        np.where(combine_slim["combine_draft_ovr"] <= 64, 0.6,
        np.where(combine_slim["combine_draft_ovr"] <= 100, 0.4,
        np.where(combine_slim["combine_draft_ovr"] <= 160, 0.2, 0.1)
    ))))

    # Select output columns
    output_cols = ["player_id", "combine_season",
                   "forty_time", "combine_weight", "height_inches",
                   "vertical_jump", "broad_jump", "bench_reps",
                   "athleticism_score", "draft_capital_score"]
    # Add z-score columns
    output_cols += [c for c in combine_slim.columns if c.endswith("_zscore")]

    available = [c for c in output_cols if c in combine_slim.columns]
    result = combine_slim[available].copy()

    print(f"  Rookie/combine features for {len(result)} players")
    print(f"  Features: {[c for c in available if c not in ['player_id', 'combine_season']]}")
    return result


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


def build_multiyear_training_pairs(season_features, horizon=2):
    """
    Create training pairs: features from season N, target = PPG from season N+horizon.

    Only includes players who actually played in the target season (inner join).
    This makes the model predict E[PPG | still playing in year N], NOT the
    unconditional expectation. Attrition is handled separately by the survival model.
    """
    seasons = sorted(season_features["season"].unique())
    pairs = []

    for i in range(len(seasons) - horizon):
        curr_season = seasons[i]
        target_season = seasons[i + horizon]

        curr = season_features[season_features["season"] == curr_season].copy()
        target = season_features[season_features["season"] == target_season][
            ["player_id", "ppg_target_col", "games"]
        ].rename(columns={"ppg_target_col": f"target_ppg_{horizon}yr",
                          "games": f"target_games_{horizon}yr"})

        # Inner join: only players who actually played in the target season.
        # Survival/attrition is modeled separately.
        merged = curr.merge(target, on="player_id", how="inner")

        # Filter: must have played meaningful snaps in both seasons
        merged = merged[
            (merged["games"] >= 4) &
            (merged[f"target_games_{horizon}yr"] >= 4)
        ].copy()
        pairs.append(merged)

    training_df = pd.concat(pairs, ignore_index=True)

    # Previous season stats for trajectory features
    prev = season_features[["player_id", "season", "ppg", "fantasy_points_ppr", "games",
                             "yards_per_game", "td_per_game", "epa_per_game"]].copy()
    prev["season"] = prev["season"] + 1
    prev.columns = ["player_id", "season"] + [f"prev_{c}" for c in prev.columns[2:]]
    training_df = training_df.merge(prev, on=["player_id", "season"], how="left")

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

    # Weekly game-level features
    weekly_path = os.path.join(DATA_DIR, "weekly_stats.csv")
    if os.path.exists(weekly_path):
        print("Building weekly game-level features...")
        weekly_stats = pd.read_csv(weekly_path)
        weekly_features = build_weekly_features(weekly_stats, players)
        if len(weekly_features) > 0:
            season_features = season_features.merge(
                weekly_features, on=["player_id", "season"], how="left"
            )
            print(f"  Merged weekly features: {weekly_features.columns.tolist()}")
    else:
        print("  No weekly_stats.csv found — skipping game-level features")

    # Situation features
    print("Building situation features...")
    situation_features = build_situation_features(seasonal, players)
    if len(situation_features) > 0:
        season_features = season_features.merge(
            situation_features, on=["player_id", "season"], how="left"
        )
        print(f"  Merged situation features: {situation_features.columns.tolist()}")

    # Rookie / combine features (static per player, merged to every season)
    combine_path = os.path.join(DATA_DIR, "combine_data.csv")
    if os.path.exists(combine_path):
        print("Building rookie/combine features...")
        combine = pd.read_csv(combine_path)
        rookie_features = build_rookie_features(players, draft, combine)
        if len(rookie_features) > 0:
            # Drop combine_season — these are static player attributes
            rookie_cols = [c for c in rookie_features.columns if c != "combine_season"]
            season_features = season_features.merge(
                rookie_features[rookie_cols], on="player_id", how="left"
            )
    else:
        print("  No combine_data.csv found — skipping rookie features")

    print("Building training pairs (season N -> season N+1 PPG)...")
    training_df = build_training_pairs(season_features)
    print(f"  {len(training_df)} training samples")

    print("Building multi-year training pairs...")
    for horizon in [2, 3]:
        df = build_multiyear_training_pairs(season_features, horizon=horizon)
        df.to_csv(os.path.join(DATA_DIR, f"training_data_{horizon}yr.csv"), index=False)
        print(f"  {horizon}-year horizon: {len(df)} samples")

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
