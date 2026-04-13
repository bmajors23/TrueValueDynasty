"""
Step 3: Train XGBoost to predict next-season PPG.

This is a pure production prediction model — no KTC in the loop.
The model learns: given a player's current stats/age/situation -> what will
their PPG be next season?
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Columns to exclude from features
NON_FEATURE_COLS = [
    "player_id", "season", "position",
    "ppg_target_col", "next_season_ppg", "next_season_games",
]


def train_ppg_model():
    """Train XGBoost to predict next-season PPG."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    training_df = pd.read_csv(os.path.join(DATA_DIR, "training_data.csv"))

    # Feature columns = everything except IDs and target
    feature_cols = [c for c in training_df.columns if c not in NON_FEATURE_COLS]
    X = training_df[feature_cols].fillna(0)
    y = training_df["next_season_ppg"]

    print(f"Training data: {len(X)} samples, {len(feature_cols)} features")
    print(f"Target (next_season_ppg): mean={y.mean():.1f}, median={y.median():.1f}, std={y.std():.1f}")

    # Train model
    model = xgb.XGBRegressor(
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

    # Time-aware cross-validation (don't leak future into past)
    print("\nRunning time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    # Sort by season for proper time-series split
    sort_idx = training_df["season"].argsort()
    X_sorted = X.iloc[sort_idx]
    y_sorted = y.iloc[sort_idx]

    cv_mae_scores = cross_val_score(model, X_sorted, y_sorted, cv=tscv, scoring="neg_mean_absolute_error")
    cv_r2_scores = cross_val_score(model, X_sorted, y_sorted, cv=tscv, scoring="r2")
    print(f"  CV MAE: {-cv_mae_scores.mean():.2f} PPG (+/- {cv_mae_scores.std():.2f})")
    print(f"  CV R²:  {cv_r2_scores.mean():.3f} (+/- {cv_r2_scores.std():.3f})")

    # Holdout test: train on everything except last season pair, test on last
    last_season = training_df["season"].max()
    train_mask = training_df["season"] < last_season
    test_mask = training_df["season"] == last_season

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nHoldout test (trained on <{last_season}, tested on {last_season}):")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.2f} PPG")
    print(f"  R²:   {r2_score(y_test, y_pred):.3f}")

    # Train final model on ALL data
    print("\nTraining final model on all data...")
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 most important features for predicting next-season PPG:")
    for _, row in importance.head(20).iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"  {row['feature']:35s} {row['importance']:.4f} {bar}")

    # Save
    model.save_model(os.path.join(MODEL_DIR, "xgb_ppg_predictor.json"))
    importance.to_csv(os.path.join(DATA_DIR, "feature_importance.csv"), index=False)

    print(f"\nModel saved to {os.path.join(MODEL_DIR, 'xgb_ppg_predictor.json')}")
    return model, feature_cols, importance


if __name__ == "__main__":
    train_ppg_model()
