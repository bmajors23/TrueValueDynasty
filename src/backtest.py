"""
Backtesting framework for TrueValue Dynasty.

For each test season, trains the PPG model using ONLY data available before
that season, generates predictions, then measures how well those predictions
matched actual outcomes. This proves (or disproves) that the model adds
value beyond naive baselines.

Usage:
    python backtest.py              # Run full backtest
    python backtest.py --year 2024  # Backtest single year
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
import json
import argparse

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Columns that aren't features
NON_FEATURE_COLS = [
    "player_id", "season", "position",
    "ppg_target_col", "next_season_ppg", "next_season_games",
    "target_ppg_2yr", "target_games_2yr",
    "target_ppg_3yr", "target_games_3yr",
    "seasons_missed", "data_staleness",
]

# Model hyperparams (same as train_model.py)
MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_weight=5,
    random_state=42,
)


def build_training_pairs_for_backtest(season_features, max_season):
    """Build training pairs using only data up to max_season.

    Features from season N, target = PPG from season N+1.
    Only includes seasons where both N and N+1 are <= max_season.
    """
    sf = season_features[season_features["season"] <= max_season].copy()
    seasons = sorted(sf["season"].unique())
    pairs = []

    for i in range(len(seasons) - 1):
        curr_season = seasons[i]
        next_season = seasons[i + 1]

        curr = sf[sf["season"] == curr_season].copy()
        next_ppg = sf[sf["season"] == next_season][
            ["player_id", "ppg_target_col", "games"]
        ].rename(columns={"ppg_target_col": "next_season_ppg", "games": "next_season_games"})

        merged = curr.merge(next_ppg, on="player_id", how="inner")
        merged = merged[merged["next_season_games"] >= 4].copy()
        pairs.append(merged)

    if not pairs:
        return pd.DataFrame()

    training_df = pd.concat(pairs, ignore_index=True)

    # Previous season stats for trajectory features
    prev = sf[["player_id", "season", "ppg", "fantasy_points_ppr", "games",
               "yards_per_game", "td_per_game", "epa_per_game"]].copy()
    prev["season"] = prev["season"] + 1
    prev.columns = ["player_id", "season"] + [f"prev_{c}" for c in prev.columns[2:]]
    training_df = training_df.merge(prev, on=["player_id", "season"], how="left")

    training_df["ppg_delta"] = training_df["ppg"] - training_df["prev_ppg"].fillna(0)
    training_df["ypg_delta"] = training_df["yards_per_game"] - training_df["prev_yards_per_game"].fillna(0)
    training_df["epg_delta"] = training_df["epa_per_game"] - training_df["prev_epa_per_game"].fillna(0)

    return training_df


def backtest_single_year(season_features, test_season, verbose=True):
    """Train model on data before test_season, predict test_season outcomes.

    Returns a DataFrame with predictions vs actuals for each player.
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"  BACKTESTING: Train on <={test_season - 1}, Predict {test_season}")
        print(f"{'='*80}")

    # Build training data using only seasons before test_season
    # Training pairs: features from season N, target = PPG from N+1
    # So max training season = test_season - 2 (features) -> test_season - 1 (target)
    training_df = build_training_pairs_for_backtest(season_features, test_season - 1)

    if len(training_df) == 0:
        print(f"  No training data available for test season {test_season}")
        return None

    feature_cols = [c for c in training_df.columns if c not in NON_FEATURE_COLS]
    X_train = training_df[feature_cols].fillna(0)
    y_train = training_df["next_season_ppg"]

    if verbose:
        print(f"  Training samples: {len(X_train)} ({training_df['season'].min()}-{training_df['season'].max()})")
        print(f"  Features: {len(feature_cols)}")

    # Train model
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Build test set: players from test_season - 1, predicting test_season PPG
    prev_season = test_season - 1
    test_features = season_features[season_features["season"] == prev_season].copy()

    # Get actual outcomes from test_season
    actual_outcomes = season_features[season_features["season"] == test_season][
        ["player_id", "ppg_target_col", "games", "position"]
    ].rename(columns={"ppg_target_col": "actual_ppg", "games": "actual_games",
                       "position": "actual_pos"})

    # Merge: only players who played in both seasons
    test_df = test_features.merge(actual_outcomes, on="player_id", how="inner")
    test_df = test_df[test_df["actual_games"] >= 4].copy()  # meaningful sample

    if len(test_df) == 0:
        print(f"  No test data for season {test_season}")
        return None

    # Add previous season features for trajectory
    prev_prev = season_features[season_features["season"] == prev_season - 1][
        ["player_id", "ppg", "fantasy_points_ppr", "games",
         "yards_per_game", "td_per_game", "epa_per_game"]
    ].copy()
    prev_prev.columns = ["player_id"] + [f"prev_{c}" for c in prev_prev.columns[1:]]
    test_df = test_df.merge(prev_prev, on="player_id", how="left")
    test_df["ppg_delta"] = test_df["ppg"] - test_df["prev_ppg"].fillna(0)
    test_df["ypg_delta"] = test_df["yards_per_game"] - test_df["prev_yards_per_game"].fillna(0)
    test_df["epg_delta"] = test_df["epa_per_game"] - test_df["prev_epa_per_game"].fillna(0)

    # Predict
    available_features = [c for c in feature_cols if c in test_df.columns]
    missing = [c for c in feature_cols if c not in test_df.columns]
    X_test = test_df[available_features].copy()
    for col in missing:
        X_test[col] = 0
    X_test = X_test[feature_cols].fillna(0)

    test_df["predicted_ppg"] = model.predict(X_test).clip(min=0)

    # Naive baseline: last season's PPG (the simplest possible prediction)
    test_df["naive_prediction"] = test_df["ppg"]

    # Position-mean baseline: average PPG for that position last season
    pos_means = test_features.groupby("position")["ppg"].mean().to_dict()
    test_df["pos_mean_prediction"] = test_df["position"].map(pos_means)

    # Blended baseline: 70% last PPG + 30% position mean (a common naive approach)
    test_df["blended_naive"] = 0.7 * test_df["ppg"] + 0.3 * test_df["pos_mean_prediction"]

    return test_df


def evaluate_predictions(results_df, label="Model", verbose=True):
    """Compute accuracy metrics for a set of predictions."""
    actual = results_df["actual_ppg"]
    predicted = results_df[label.lower().replace(" ", "_") + "_prediction" if label != "Model" else "predicted_ppg"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    # Correlation (how well does the model rank players?)
    rank_corr = actual.corr(predicted, method="spearman")

    # Top-24 accuracy: what % of predicted top-24 are actually top-24?
    actual_top24 = set(actual.nlargest(24).index)
    pred_top24 = set(predicted.nlargest(24).index)
    top24_overlap = len(actual_top24 & pred_top24) / 24 * 100

    # Directional accuracy for players with prior year data
    if "ppg" in results_df.columns:
        went_up = actual > results_df["ppg"]
        pred_up = predicted > results_df["ppg"]
        directional = (went_up == pred_up).mean() * 100
    else:
        directional = None

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "rank_correlation": rank_corr,
        "top24_overlap_pct": top24_overlap,
        "directional_accuracy_pct": directional,
    }

    if verbose:
        print(f"    {label:25s}  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.3f}  "
              f"Rank Corr: {rank_corr:.3f}  Top24: {top24_overlap:.0f}%"
              + (f"  Dir: {directional:.1f}%" if directional else ""))

    return metrics


def run_backtest(test_seasons=None, by_position=False):
    """Run full backtesting across multiple seasons."""
    print("Loading season features...")
    season_features = pd.read_csv(os.path.join(DATA_DIR, "season_features.csv"))
    print(f"  {len(season_features)} player-seasons, "
          f"seasons {season_features['season'].min()}-{season_features['season'].max()}")

    if test_seasons is None:
        # Default: test 2022-2025 (need enough training history)
        available = sorted(season_features["season"].unique())
        test_seasons = [s for s in available if s >= 2022]

    all_results = []
    season_metrics = {}

    for test_season in test_seasons:
        results = backtest_single_year(season_features, test_season)
        if results is None:
            continue

        print(f"\n  Results for {test_season} ({len(results)} players):")
        print(f"  {'─'*90}")

        # Evaluate all models
        model_metrics = evaluate_predictions(results, "Model")

        # Naive baselines
        # Rename for the evaluate function
        results["naive_prediction"] = results["ppg"]  # already set
        results_eval = results.copy()
        results_eval["predicted_ppg"] = results_eval["naive_prediction"]
        naive_metrics = evaluate_predictions(results_eval, "Model")
        # Print manually for baselines
        actual = results["actual_ppg"]
        for baseline_name, baseline_col in [("Naive (last PPG)", "naive_prediction"),
                                              ("Pos Mean", "pos_mean_prediction"),
                                              ("Blended (70/30)", "blended_naive")]:
            pred = results[baseline_col]
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            rank_corr = actual.corr(pred, method="spearman")
            pred_top24 = set(pred.nlargest(24).index)
            actual_top24 = set(actual.nlargest(24).index)
            top24 = len(actual_top24 & pred_top24) / 24 * 100
            print(f"    {baseline_name:25s}  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.3f}  "
                  f"Rank Corr: {rank_corr:.3f}  Top24: {top24:.0f}%")

        # Model improvement over naive
        naive_mae = mean_absolute_error(actual, results["naive_prediction"])
        model_mae = model_metrics["mae"]
        improvement = (1 - model_mae / naive_mae) * 100
        print(f"\n  Model improvement over naive: {improvement:+.1f}% MAE reduction")

        results["test_season"] = test_season
        all_results.append(results)
        season_metrics[test_season] = model_metrics

        # Per-position breakdown
        if by_position:
            print(f"\n  Per-position breakdown:")
            for pos in ["QB", "RB", "WR", "TE"]:
                pos_results = results[results["position"] == pos]
                if len(pos_results) < 10:
                    continue
                pos_actual = pos_results["actual_ppg"]
                pos_pred = pos_results["predicted_ppg"]
                pos_naive = pos_results["naive_prediction"]
                pos_mae = mean_absolute_error(pos_actual, pos_pred)
                naive_mae = mean_absolute_error(pos_actual, pos_naive)
                r2 = r2_score(pos_actual, pos_pred)
                print(f"    {pos}: Model MAE {pos_mae:.2f} vs Naive MAE {naive_mae:.2f} "
                      f"({(1 - pos_mae/naive_mae)*100:+.1f}%)  R²: {r2:.3f}  n={len(pos_results)}")

    if not all_results:
        print("No backtest results generated.")
        return

    # Aggregate results
    combined = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*80}")
    print(f"  AGGREGATE RESULTS ({len(test_seasons)} seasons, {len(combined)} total predictions)")
    print(f"{'='*80}")

    actual = combined["actual_ppg"]
    for name, col in [("XGBoost Model", "predicted_ppg"),
                       ("Naive (last PPG)", "naive_prediction"),
                       ("Position Mean", "pos_mean_prediction"),
                       ("Blended (70/30)", "blended_naive")]:
        pred = combined[col]
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        rank_corr = actual.corr(pred, method="spearman")
        print(f"  {name:25s}  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.3f}  Rank Corr: {rank_corr:.3f}")

    naive_mae = mean_absolute_error(actual, combined["naive_prediction"])
    model_mae = mean_absolute_error(actual, combined["predicted_ppg"])
    blended_mae = mean_absolute_error(actual, combined["blended_naive"])

    print(f"\n  Model vs Naive:    {(1 - model_mae/naive_mae)*100:+.1f}% MAE improvement")
    print(f"  Model vs Blended:  {(1 - model_mae/blended_mae)*100:+.1f}% MAE improvement")

    # Elite player analysis: how well does the model predict top performers?
    print(f"\n  ELITE PLAYER ANALYSIS (PPG >= 15 in test season):")
    elite = combined[combined["actual_ppg"] >= 15]
    if len(elite) > 0:
        for name, col in [("XGBoost Model", "predicted_ppg"),
                           ("Naive (last PPG)", "naive_prediction"),
                           ("Blended (70/30)", "blended_naive")]:
            mae = mean_absolute_error(elite["actual_ppg"], elite[col])
            r2 = r2_score(elite["actual_ppg"], elite[col])
            print(f"    {name:25s}  MAE: {mae:.2f}  R²: {r2:.3f}  n={len(elite)}")

    # Breakout analysis: players who improved 5+ PPG
    print(f"\n  BREAKOUT ANALYSIS (PPG improved 5+ from prior year):")
    breakouts = combined[combined["actual_ppg"] - combined["ppg"] >= 5]
    if len(breakouts) > 0:
        for name, col in [("XGBoost Model", "predicted_ppg"),
                           ("Naive (last PPG)", "naive_prediction")]:
            mae = mean_absolute_error(breakouts["actual_ppg"], breakouts[col])
            print(f"    {name:25s}  MAE: {mae:.2f}  n={len(breakouts)}")

    # Bust analysis: players who declined 5+ PPG
    print(f"\n  BUST ANALYSIS (PPG declined 5+ from prior year):")
    busts = combined[combined["ppg"] - combined["actual_ppg"] >= 5]
    if len(busts) > 0:
        for name, col in [("XGBoost Model", "predicted_ppg"),
                           ("Naive (last PPG)", "naive_prediction")]:
            mae = mean_absolute_error(busts["actual_ppg"], busts[col])
            print(f"    {name:25s}  MAE: {mae:.2f}  n={len(busts)}")

    # Save results
    output_path = os.path.join(DATA_DIR, "backtest_results.csv")
    combined.to_csv(output_path, index=False)
    print(f"\n  Detailed results saved to {output_path}")

    # Save summary
    summary = {
        "test_seasons": [int(s) for s in test_seasons],
        "total_predictions": int(len(combined)),
        "aggregate": {
            "model_mae": round(model_mae, 3),
            "naive_mae": round(naive_mae, 3),
            "blended_mae": round(blended_mae, 3),
            "improvement_vs_naive_pct": round((1 - model_mae / naive_mae) * 100, 1),
            "improvement_vs_blended_pct": round((1 - model_mae / blended_mae) * 100, 1),
        },
        "per_season": {str(k): {kk: round(vv, 3) for kk, vv in v.items() if vv is not None}
                       for k, v in season_metrics.items()},
    }
    summary_path = os.path.join(DATA_DIR, "backtest_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    return combined, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest TrueValue Dynasty model")
    parser.add_argument("--year", type=int, help="Single year to backtest")
    parser.add_argument("--positions", action="store_true", help="Show per-position breakdown")
    args = parser.parse_args()

    if args.year:
        run_backtest(test_seasons=[args.year], by_position=args.positions)
    else:
        run_backtest(by_position=args.positions)
