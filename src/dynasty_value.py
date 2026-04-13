"""
Step 5: Calculate dynasty true value using the 3-model approach.

Dynasty Value = Σ [ P(relevant in year N) × E(PPG in year N | relevant) - ReplacementPPG ] × Discount^N

Three models, all data-derived:
1. PPG Predictor  — predicts next-season PPG (XGBoost regression)
2. Survival Model — predicts P(still fantasy-relevant) in future years (XGBoost classifier)
3. Age Curves     — adjusts predicted PPG for aging effects

No manual weights. No KTC in the formula.
KTC is loaded only for comparison at the end.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os

from age_curves import get_retention_factor

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# How many future seasons to project
PROJECTION_YEARS = 5

# Discount rate: how much less a future point is worth vs a current point
# 0.90 = a point 1 year from now is worth 90% of a point today
DISCOUNT_RATE = 0.90

# League settings (configurable)
DEFAULT_LEAGUE = {
    "teams": 12,
    "qb_slots": 2,  # 2 = superflex
}

REPLACEMENT_LEVEL_MULTIPLIER = {
    "QB": {"1qb": 1.0, "sf": 2.0},
    "RB": {"1qb": 2.0, "sf": 2.0},
    "WR": {"1qb": 3.0, "sf": 3.0},
    "TE": {"1qb": 1.0, "sf": 1.0},
}


def compute_replacement_levels(season_features, league):
    """Compute replacement-level PPG by position from historical data."""
    teams = league["teams"]
    is_sf = league["qb_slots"] >= 2
    format_key = "sf" if is_sf else "1qb"

    recent = season_features[season_features["season"] >= season_features["season"].max() - 2].copy()
    recent = recent[recent["games"] >= 8].copy()

    replacement = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        rank_cutoff = int(REPLACEMENT_LEVEL_MULTIPLIER[pos][format_key] * teams)
        pos_data = recent[recent["position"] == pos]
        season_repls = []
        for season in pos_data["season"].unique():
            s = pos_data[pos_data["season"] == season].sort_values("ppg", ascending=False)
            if len(s) >= rank_cutoff:
                season_repls.append(s.iloc[rank_cutoff - 1]["ppg"])
            elif len(s) > 0:
                season_repls.append(s["ppg"].min())
        replacement[pos] = np.mean(season_repls) if season_repls else 0

    return replacement


def calculate_dynasty_values(league=None):
    """Calculate dynasty true value for all current players."""
    if league is None:
        league = DEFAULT_LEAGUE

    # Load models
    ppg_model = xgb.XGBRegressor()
    ppg_model.load_model(os.path.join(MODEL_DIR, "xgb_ppg_predictor.json"))

    survival_model = xgb.XGBClassifier()
    survival_model.load_model(os.path.join(MODEL_DIR, "xgb_survival.json"))

    # Load data
    current_df = pd.read_csv(os.path.join(DATA_DIR, "current_features.csv"))
    season_features = pd.read_csv(os.path.join(DATA_DIR, "season_features.csv"))
    age_curves = pd.read_csv(os.path.join(DATA_DIR, "age_curves.csv"))

    # Replacement levels
    replacement = compute_replacement_levels(season_features, league)
    print("Replacement-level PPG by position:")
    for pos, ppg in replacement.items():
        print(f"  {pos}: {ppg:.1f} PPG")

    # Define feature columns for each model
    ppg_non_feature = ["player_id", "season", "position", "ppg_target_col",
                       "next_season_ppg", "next_season_games",
                       "seasons_missed", "data_staleness"]
    ppg_feature_cols = [c for c in current_df.columns
                        if c not in ppg_non_feature
                        and c in ppg_model.get_booster().feature_names]

    survival_feature_cols = [
        "age", "years_in_league",
        "pos_QB", "pos_RB", "pos_WR", "pos_TE",
        "games", "ppg", "fantasy_points_ppr", "relevant",
        "draft_capital_decayed", "draft_round", "draft_pick",
        "prev_ppg", "prev_games", "ppg_delta",
    ]

    # Predict next-season PPG
    X_ppg = current_df[ppg_feature_cols].fillna(0)
    current_df["predicted_ppg"] = ppg_model.predict(X_ppg).clip(min=0)

    # For each player, project dynasty value
    results = []
    for idx, player in current_df.iterrows():
        pos = player["position"]
        age = player["age"]
        repl_ppg = replacement.get(pos, 0)
        seasons_missed = player.get("seasons_missed", 0)

        # Chain survival probability and projected PPG forward
        total_value = 0
        projected_ppg = player["predicted_ppg"]

        # Build survival features that we'll age forward
        surv_features = {}
        for col in survival_feature_cols:
            surv_features[col] = player.get(col, 0)
            if pd.isna(surv_features[col]):
                surv_features[col] = 0

        cumulative_survival = 1.0

        for year in range(PROJECTION_YEARS):
            future_age = age + year + 1

            # --- Survival probability ---
            surv_features["age"] = age + year
            surv_features["years_in_league"] = player["years_in_league"] + year
            surv_features["draft_capital_decayed"] = (
                player.get("draft_capital_decayed", 0) * (0.8 ** year)
            )
            surv_features["ppg"] = projected_ppg
            surv_features["fantasy_points_ppr"] = projected_ppg * 17
            surv_features["relevant"] = 1 if projected_ppg >= 5 else 0

            X_surv = pd.DataFrame([surv_features])[survival_feature_cols].fillna(0)
            p_survive = survival_model.predict_proba(X_surv)[0][1]
            cumulative_survival *= p_survive

            # --- Projected PPG ---
            if year == 0:
                season_ppg = projected_ppg
            else:
                retention = get_retention_factor(pos, future_age - 1, age_curves)
                projected_ppg = projected_ppg * retention
                season_ppg = max(0, projected_ppg)

            # --- Value above replacement ---
            # Soft replacement: full credit above replacement, small baseline
            # for being at replacement level, tapered partial credit below.
            if season_ppg >= repl_ppg:
                par = (season_ppg - repl_ppg) + repl_ppg * 0.12
            elif season_ppg >= repl_ppg * 0.6:
                # Below replacement but still rosterable — tapered credit
                fraction = (season_ppg - repl_ppg * 0.6) / (repl_ppg * 0.4)
                par = repl_ppg * 0.12 * fraction
            else:
                par = 0

            # --- Discount ---
            discount = DISCOUNT_RATE ** year

            # --- Season value = P(alive) × value × discount ---
            season_value = cumulative_survival * par * discount
            total_value += season_value

        # Penalty for stale data (player missed recent seasons)
        if seasons_missed > 0:
            staleness_penalty = 0.85 ** seasons_missed
            total_value *= staleness_penalty

        results.append({
            "player_id": player["player_id"],
            "position": pos,
            "age": round(age, 1),
            "current_ppg": round(player["ppg"], 1),
            "predicted_next_ppg": round(player["predicted_ppg"], 1),
            "replacement_ppg": round(repl_ppg, 1),
            "seasons_missed": int(seasons_missed),
            "dynasty_value_raw": round(total_value, 2),
        })

    results_df = pd.DataFrame(results)

    # Normalize to 0-9999 scale using a log-scaled approach
    # This spreads the distribution more naturally instead of letting
    # one outlier compress everyone else to the bottom
    raw = results_df["dynasty_value_raw"].copy()
    raw_positive = raw[raw > 0]

    if len(raw_positive) > 0:
        # Log transform to compress the top and spread the middle
        log_raw = np.log1p(raw)  # log(1 + x) to handle zeros
        log_max = log_raw.max()
        if log_max > 0:
            results_df["dynasty_value"] = (
                (log_raw / log_max) * 9999
            ).round(0).astype(int)
        else:
            results_df["dynasty_value"] = 0
        # Ensure 0 raw stays 0
        results_df.loc[raw <= 0, "dynasty_value"] = 0
    else:
        results_df["dynasty_value"] = 0

    # Map player names
    players = pd.read_csv(os.path.join(DATA_DIR, "players.csv"))
    name_map = players.set_index("gsis_id")["display_name"].to_dict()
    results_df["player_name"] = results_df["player_id"].map(name_map)

    # Load KTC for comparison
    ktc_path = os.path.join(DATA_DIR, "ktc_values.csv")
    if os.path.exists(ktc_path):
        ktc = pd.read_csv(ktc_path)
        ktc["name_norm"] = ktc["ktc_player_name"].str.lower().str.strip().str.replace(
            r"[.'\-]", "", regex=True).str.replace(r"\s+", " ", regex=True)
        results_df["name_norm"] = results_df["player_name"].str.lower().str.strip().str.replace(
            r"[.'\-]", "", regex=True).str.replace(r"\s+", " ", regex=True)

        is_sf = league["qb_slots"] >= 2
        ktc_col = "ktc_value_sf" if is_sf else "ktc_value_1qb"
        ktc_slim = ktc[["name_norm", "position", ktc_col]].rename(columns={ktc_col: "ktc_value"})
        results_df = results_df.merge(ktc_slim, on=["name_norm", "position"], how="left")
        results_df.drop(columns=["name_norm"], inplace=True)

        results_df["value_vs_ktc"] = results_df["dynasty_value"] - results_df["ktc_value"]
        results_df["value_vs_ktc_pct"] = (
            (results_df["value_vs_ktc"] / results_df["ktc_value"].clip(lower=1)) * 100
        ).round(1)

    # Sort and rank
    results_df = results_df.sort_values("dynasty_value", ascending=False).reset_index(drop=True)
    results_df.index += 1
    results_df.index.name = "rank"

    # Save
    results_df.to_csv(os.path.join(DATA_DIR, "dynasty_values.csv"))

    # Print rankings
    print("\n" + "=" * 95)
    print("TRUE VALUE DYNASTY RANKINGS (3-Model Approach)")
    print("=" * 95)

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = results_df[results_df["position"] == pos].head(20)
        print(f"\n{'─' * 95}")
        print(f"  TOP 20 {pos}s")
        print(f"{'─' * 95}")
        print(f"  {'Rk':<5}{'Player':<25}{'Age':<6}{'CurPPG':<8}{'PrdPPG':<8}{'Value':<8}{'KTC':<8}{'Diff':<8}{'Miss'}")
        print(f"  {'─'*5}{'─'*25}{'─'*6}{'─'*8}{'─'*8}{'─'*8}{'─'*8}{'─'*8}{'─'*4}")
        for rank, row in pos_df.iterrows():
            ktc_str = f"{row['ktc_value']:.0f}" if pd.notna(row.get('ktc_value')) else "—"
            diff_str = f"{row['value_vs_ktc']:+.0f}" if pd.notna(row.get('value_vs_ktc')) else ""
            miss_str = f"{row['seasons_missed']}" if row['seasons_missed'] > 0 else ""
            print(f"  {rank:<5}{row['player_name']:<25}{row['age']:<6.1f}"
                  f"{row['current_ppg']:<8.1f}{row['predicted_next_ppg']:<8.1f}"
                  f"{row['dynasty_value']:<8}{ktc_str:<8}{diff_str:<8}{miss_str}")

    # Market inefficiencies
    if "value_vs_ktc" in results_df.columns:
        has_ktc = results_df.dropna(subset=["value_vs_ktc"])
        # Only show players with meaningful value on at least one side
        meaningful = has_ktc[(has_ktc["dynasty_value"] > 500) | (has_ktc["ktc_value"] > 2000)]

        print(f"\n{'=' * 95}")
        print("BIGGEST MARKET INEFFICIENCIES")
        print(f"{'=' * 95}")

        print(f"\n  UNDERVALUED BY KTC (True Value >> Market Price):")
        under = meaningful.nlargest(15, "value_vs_ktc_pct")
        for _, row in under.iterrows():
            print(f"    {row['player_name']:<25}{row['position']:<4}Age {row['age']:<5.1f}"
                  f"  True: {row['dynasty_value']:<6}  KTC: {row['ktc_value']:<6.0f}"
                  f"  Diff: {row['value_vs_ktc']:+6.0f} ({row['value_vs_ktc_pct']:+5.1f}%)")

        print(f"\n  OVERVALUED BY KTC (Market Price >> True Value):")
        over = meaningful.nsmallest(15, "value_vs_ktc_pct")
        for _, row in over.iterrows():
            print(f"    {row['player_name']:<25}{row['position']:<4}Age {row['age']:<5.1f}"
                  f"  True: {row['dynasty_value']:<6}  KTC: {row['ktc_value']:<6.0f}"
                  f"  Diff: {row['value_vs_ktc']:+6.0f} ({row['value_vs_ktc_pct']:+5.1f}%)")

    print(f"\n\nFull rankings saved to {os.path.join(DATA_DIR, 'dynasty_values.csv')}")
    return results_df


if __name__ == "__main__":
    calculate_dynasty_values()
