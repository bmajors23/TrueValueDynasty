"""
Step 3: Train XGBoost PPG predictors at multiple horizons.

Trains separate models for 1-year, 2-year, and 3-year predictions.
Each horizon gets a mean model + quantile models (10th/90th percentile).
Years 4-5 use the 3-year model with additional regression to mean.

Also computes player-specific regression strength based on career consistency.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Columns to exclude from features
NON_FEATURE_COLS = [
    "player_id", "season", "position",
    "ppg_target_col", "next_season_ppg", "next_season_games",
    "target_ppg_2yr", "target_games_2yr",
    "target_ppg_3yr", "target_games_3yr",
]


def compute_position_priors(training_df):
    """Compute position-level PPG priors for regression to the mean."""
    recent = training_df[training_df["season"] >= training_df["season"].max() - 3].copy()
    priors = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_data = recent[recent[f"pos_{pos}"] == 1]
        relevant = pos_data[pos_data["next_season_ppg"] >= 5]
        priors[pos] = {
            "mean_ppg": float(relevant["next_season_ppg"].mean()) if len(relevant) > 0 else 8.0,
            "std_ppg": float(relevant["next_season_ppg"].std()) if len(relevant) > 0 else 4.0,
            "all_mean_ppg": float(pos_data["next_season_ppg"].mean()) if len(pos_data) > 0 else 5.0,
        }
    return priors


def train_single_horizon(X, y, label, X_train=None, X_test=None, y_train=None, y_test=None):
    """Train mean + quantile models for one horizon. Returns (mean_model, q10_model, q90_model)."""
    base_params = dict(
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

    # Mean model
    model = xgb.XGBRegressor(**base_params)

    if X_train is not None:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  {label} holdout — MAE: {mae:.2f} PPG, R²: {r2:.3f}")

    # Train final on all data
    model.fit(X, y)

    # Quantile models
    q_params = {**base_params, "n_estimators": 200, "max_depth": 4}
    q10 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10, **q_params)
    q90 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90, **q_params)
    q10.fit(X, y)
    q90.fit(X, y)

    return model, q10, q90


def train_ppg_model():
    """Train horizon-specific PPG models (1yr, 2yr, 3yr)."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 1-year model (existing) ---
    print("=== 1-Year PPG Model ===")
    training_df = pd.read_csv(os.path.join(DATA_DIR, "training_data.csv"))
    feature_cols = [c for c in training_df.columns if c not in NON_FEATURE_COLS]
    X = training_df[feature_cols].fillna(0)
    y = training_df["next_season_ppg"]

    print(f"  {len(X)} samples, {len(feature_cols)} features")

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=5)
    sort_idx = training_df["season"].argsort()
    X_sorted, y_sorted = X.iloc[sort_idx], y.iloc[sort_idx]
    temp_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                                   subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                                   reg_lambda=1.0, min_child_weight=5, random_state=42)
    cv_scores = cross_val_score(temp_model, X_sorted, y_sorted, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"  CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    # Holdout
    last_season = training_df["season"].max()
    train_mask = training_df["season"] < last_season
    test_mask = training_df["season"] == last_season

    model_1yr, q10_1yr, q90_1yr = train_single_horizon(
        X, y, "1yr",
        X[train_mask], X[test_mask], y[train_mask], y[test_mask],
    )

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model_1yr.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  Top 15 features:")
    for _, row in importance.head(15).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:30s} {row['importance']:.4f} {bar}")

    # --- 2-year model ---
    print("\n=== 2-Year PPG Model ===")
    path_2yr = os.path.join(DATA_DIR, "training_data_2yr.csv")
    if os.path.exists(path_2yr):
        df_2yr = pd.read_csv(path_2yr)
        feat_2yr = [c for c in df_2yr.columns if c not in NON_FEATURE_COLS]
        X_2 = df_2yr[feat_2yr].fillna(0)
        y_2 = df_2yr["target_ppg_2yr"]
        print(f"  {len(X_2)} samples")

        last_s2 = df_2yr["season"].max()
        m2 = df_2yr["season"] < last_s2
        t2 = df_2yr["season"] == last_s2
        model_2yr, q10_2yr, q90_2yr = train_single_horizon(
            X_2, y_2, "2yr", X_2[m2], X_2[t2], y_2[m2], y_2[t2],
        )
    else:
        print("  No 2yr training data found — skipping")
        model_2yr = q10_2yr = q90_2yr = None

    # --- 3-year model ---
    print("\n=== 3-Year PPG Model ===")
    path_3yr = os.path.join(DATA_DIR, "training_data_3yr.csv")
    if os.path.exists(path_3yr):
        df_3yr = pd.read_csv(path_3yr)
        feat_3yr = [c for c in df_3yr.columns if c not in NON_FEATURE_COLS]
        X_3 = df_3yr[feat_3yr].fillna(0)
        y_3 = df_3yr["target_ppg_3yr"]
        print(f"  {len(X_3)} samples")

        last_s3 = df_3yr["season"].max()
        m3 = df_3yr["season"] < last_s3
        t3 = df_3yr["season"] == last_s3
        model_3yr, q10_3yr, q90_3yr = train_single_horizon(
            X_3, y_3, "3yr", X_3[m3], X_3[t3], y_3[m3], y_3[t3],
        )
    else:
        print("  No 3yr training data found — skipping")
        model_3yr = q10_3yr = q90_3yr = None

    # Compute position priors
    priors = compute_position_priors(training_df)
    print("\nPosition PPG priors:")
    for pos, vals in priors.items():
        print(f"  {pos}: mean={vals['mean_ppg']:.1f}, std={vals['std_ppg']:.1f}")

    # Save all models
    model_1yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_predictor.json"))
    q10_1yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_q10.json"))
    q90_1yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_q90.json"))

    if model_2yr is not None:
        model_2yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_2yr.json"))
        q10_2yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_2yr_q10.json"))
        q90_2yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_2yr_q90.json"))

    if model_3yr is not None:
        model_3yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_3yr.json"))
        q10_3yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_3yr_q10.json"))
        q90_3yr.save_model(os.path.join(MODEL_DIR, "xgb_ppg_3yr_q90.json"))

    importance.to_csv(os.path.join(DATA_DIR, "feature_importance.csv"), index=False)
    with open(os.path.join(MODEL_DIR, "position_priors.json"), "w") as f:
        json.dump(priors, f, indent=2)

    print(f"\nAll models saved to {MODEL_DIR}")
    return model_1yr, feature_cols, importance


if __name__ == "__main__":
    train_ppg_model()
