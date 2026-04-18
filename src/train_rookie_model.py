"""
Train the rookie model.

TRAIN:     draft classes 2012-2022 (11 classes)
VALIDATE:  draft classes 2023-2024 (2 classes — holdout for honest eval)
FINAL:     retrained on 2012-2024 for production use

Target: Year-1 NFL PPG (filter to players with year1_games >= 4)

Features: 62 from rookie_features.csv, minus identifiers and target-adjacent:
  - Drop: nfl_player_name, recruit_committed_school (high-card), nfl_draft_year
  - Drop: year1_*, year2_*, never_played (outcome leakage)
  - One-hot: nfl_pos

Baselines compared:
  - Position mean (predict mean Y1 PPG for that position)
  - Draft-pick-only (1-feature linear regression: PPG ~ draft_ovr)
  - Naive draft_capital_score

Reports MAE / RMSE / R² / Spearman rank correlation and per-position
breakdown. Saves model to models/rookie_model.json and feature
importance to data/rookie_feature_importance.csv.

Usage:
    python train_rookie_model.py
    python train_rookie_model.py --final   # retrain on 2012-2024 for prod
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLLEGE_DIR = DATA_DIR / "college"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PARAMS = dict(
    n_estimators=400,
    max_depth=4,          # shallow — training set is only ~500 rows
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=4,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
    early_stopping_rounds=30,
)

# Columns to EXCLUDE from the feature matrix
EXCLUDE_COLS = [
    "nfl_player_name",          # ID
    "name_norm",                # ID helper
    "recruit_committed_school", # high-cardinality string
    "landing_team",             # string (we'll use the landing_* numeric features)
    "nfl_draft_year",           # used only for splitting
    "first_college_season",     # absolute year leakage
    "last_college_season",      # absolute year leakage
    "recruit_class_year",       # absolute year redundancy
    # Target-adjacent (leakage!)
    "year1_ppg", "year1_games", "year1_fp_ppr",
    "year2_ppg", "year2_games",
    "never_played",
]


def prepare_xy(df):
    """Build X matrix (features) and y vector (year1_ppg)."""
    df = df.copy()
    # One-hot encode position
    pos_dummies = pd.get_dummies(df["nfl_pos"], prefix="pos").astype(int)
    df = pd.concat([df.drop(columns=["nfl_pos"]), pos_dummies], axis=1)
    # Select feature cols
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].fillna(np.nan)  # XGBoost handles NaN natively
    # Convert object columns (strings) — if any leaked in, drop them
    non_numeric = X.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"  Dropping non-numeric leftover columns: {non_numeric}")
        X = X.drop(columns=non_numeric)
        feat_cols = [c for c in feat_cols if c not in non_numeric]
    return X, feat_cols


def evaluate(name, y_true, y_pred, verbose=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rank_corr = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    if verbose:
        print(f"  {name:30} MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:+.3f}  Rank={rank_corr:+.3f}  n={len(y_true)}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "rank_corr": rank_corr, "n": len(y_true)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="Retrain on all data (2012-2024) for production")
    ap.add_argument("--min-games", type=int, default=4,
                    help="Filter training rows to players with >= N Year-1 games")
    ap.add_argument("--with-landing", action="store_true",
                    help="Include landing-spot features (post-draft model). "
                         "Requires build_landing_features.py to have been run.")
    args = ap.parse_args()

    print("Loading training pairs...")
    df = pd.read_csv(COLLEGE_DIR / "rookie_training_pairs.csv")

    # If using landing features, merge them in by player
    if args.with_landing:
        landing_path = COLLEGE_DIR / "rookie_features_with_landing.csv"
        if not landing_path.exists():
            raise FileNotFoundError(
                f"{landing_path} not found. Run build_landing_features.py first."
            )
        landing = pd.read_csv(landing_path)
        # Pick out just the landing_ columns + the join key
        landing_cols = [c for c in landing.columns if c.startswith("landing_")]
        landing_slim = landing[["nfl_player_name", "nfl_draft_year"] + landing_cols]
        df = df.merge(landing_slim, on=["nfl_player_name", "nfl_draft_year"], how="left")
        print(f"  Merged {len(landing_cols)} landing features")

    # Filter to usable training examples
    df = df[df["year1_games"] >= args.min_games].copy()
    df = df[df["nfl_draft_year"].notna()].copy()
    df["nfl_draft_year"] = df["nfl_draft_year"].astype(int)
    print(f"  {len(df)} rows with year1_games >= {args.min_games}")
    print(f"  Draft years range: {df['nfl_draft_year'].min()}-{df['nfl_draft_year'].max()}")

    if args.final:
        train = df.copy()
        val = None
    else:
        train = df[df["nfl_draft_year"] <= 2022].copy()
        val = df[df["nfl_draft_year"].isin([2023, 2024])].copy()
        print(f"  Train: {len(train)} rows (2012-2022)")
        print(f"  Val:   {len(val)} rows (2023-2024)")

    # Prepare features
    X_train, feat_cols = prepare_xy(train)
    y_train = train["year1_ppg"].values

    if val is not None:
        X_val, _ = prepare_xy(val)
        # Align columns (val may have missing pos dummies if a pos class is absent)
        for c in feat_cols:
            if c not in X_val.columns:
                X_val[c] = np.nan
        X_val = X_val[feat_cols]
        y_val = val["year1_ppg"].values
    else:
        X_val, y_val = None, None

    print(f"\n  Features: {len(feat_cols)}")

    # ── Train XGBoost model ──
    print("\nTraining rookie model (XGBoost)...")
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    if val is not None and len(val) > 0:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iter = getattr(model, "best_iteration", None)
        print(f"  Best iteration: {best_iter}")
    else:
        # Final model — no validation set, just train all estimators
        final_params = {k: v for k, v in MODEL_PARAMS.items() if k != "early_stopping_rounds"}
        model = xgb.XGBRegressor(**final_params)
        model.fit(X_train, y_train, verbose=False)

    # ── Evaluation ──
    print("\n" + "=" * 75)
    print("  EVALUATION — training set (in-sample)")
    print("=" * 75)
    pred_train = model.predict(X_train)
    evaluate("Rookie Model", y_train, pred_train)

    # Baseline: position mean
    pos_cols = [c for c in X_train.columns if c.startswith("pos_")]
    pos_means = {}
    for pc in pos_cols:
        mask = X_train[pc] == 1
        if mask.sum() > 0:
            pos_means[pc] = y_train[mask].mean()
    def pos_mean_pred(X):
        preds = np.zeros(len(X))
        for pc, m in pos_means.items():
            preds[X[pc].values == 1] = m
        return preds
    evaluate("Position Mean baseline", y_train, pos_mean_pred(X_train))

    # Baseline: draft-pick-only linear regression (1 feature: draft_ovr)
    lr = LinearRegression()
    lr.fit(train[["draft_ovr"]].fillna(256), y_train)  # 256 = typical undrafted filler
    evaluate("Draft-pick only", y_train, lr.predict(train[["draft_ovr"]].fillna(256)))

    # ── Validation set metrics ──
    if val is not None and len(val) > 0:
        print("\n" + "=" * 75)
        print(f"  EVALUATION — validation set (2023-2024, n={len(val)})")
        print("=" * 75)
        pred_val = model.predict(X_val)
        rookie_metrics = evaluate("Rookie Model", y_val, pred_val)
        pos_mean_val = pos_mean_pred(X_val)
        pm_metrics = evaluate("Position Mean baseline", y_val, pos_mean_val)
        dp_val = lr.predict(val[["draft_ovr"]].fillna(256))
        dp_metrics = evaluate("Draft-pick only", y_val, dp_val)

        print(f"\n  Model vs Position Mean: {(1 - rookie_metrics['mae']/pm_metrics['mae'])*100:+.1f}% MAE improvement")
        print(f"  Model vs Draft-only:    {(1 - rookie_metrics['mae']/dp_metrics['mae'])*100:+.1f}% MAE improvement")

        # Per-position breakdown
        print(f"\n  Per-position breakdown (validation set):")
        for pos in ["QB", "RB", "WR", "TE"]:
            mask = val["nfl_pos"] == pos
            if mask.sum() >= 3:
                mae = mean_absolute_error(val.loc[mask, "year1_ppg"], pred_val[mask])
                naive_mae = mean_absolute_error(val.loc[mask, "year1_ppg"], pos_mean_val[mask])
                r2 = r2_score(val.loc[mask, "year1_ppg"], pred_val[mask])
                imp = (1 - mae/naive_mae)*100 if naive_mae > 0 else 0
                print(f"    {pos}: MAE {mae:.2f} (vs pos-mean {naive_mae:.2f}, {imp:+.1f}%)  R²={r2:+.3f}  n={mask.sum()}")

        # Top 10 best/worst predictions on val
        val_out = val[["nfl_player_name", "nfl_pos", "nfl_draft_year", "year1_ppg"]].copy()
        val_out["predicted"] = pred_val
        val_out["error"] = val_out["predicted"] - val_out["year1_ppg"]
        print(f"\n  Top 5 biggest misses (val set):")
        for _, r in val_out.reindex(val_out["error"].abs().sort_values(ascending=False).index).head(5).iterrows():
            print(f"    {r['nfl_player_name']:25} {r['nfl_pos']}  pred={r['predicted']:.1f}  actual={r['year1_ppg']:.1f}  miss={r['error']:+.1f}")

        print(f"\n  Top 5 best predictions (val set):")
        for _, r in val_out.reindex(val_out["error"].abs().sort_values(ascending=True).index).head(5).iterrows():
            print(f"    {r['nfl_player_name']:25} {r['nfl_pos']}  pred={r['predicted']:.1f}  actual={r['year1_ppg']:.1f}  miss={r['error']:+.1f}")

    # ── Feature importance ──
    print("\n" + "=" * 75)
    print("  FEATURE IMPORTANCE (top 25)")
    print("=" * 75)
    importance = pd.DataFrame({
        "feature": feat_cols,
        "gain": model.feature_importances_,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    for _, row in importance.head(25).iterrows():
        bar = "█" * int(row["gain"] * 300)
        print(f"  {row['feature']:32} {row['gain']:.4f}  {bar}")

    # Save artifacts (name differs for pre/post-draft variants)
    suffix = "_postdraft" if args.with_landing else ""
    model_file = f"rookie_model_final{suffix}.json" if args.final else f"rookie_model{suffix}.json"
    feat_file = f"rookie_model_features{suffix}.txt"
    imp_file = f"rookie_feature_importance{suffix}.csv"
    model.save_model(MODEL_DIR / model_file)
    importance.to_csv(DATA_DIR / imp_file, index=False)
    (MODEL_DIR / feat_file).write_text("\n".join(feat_cols))

    print(f"\n✅ Saved: {MODEL_DIR / model_file}")
    print(f"   Feature list: {MODEL_DIR / feat_file}")
    print(f"   Importance:   {DATA_DIR / imp_file}")


if __name__ == "__main__":
    main()
