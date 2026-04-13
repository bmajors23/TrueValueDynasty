"""
Survival Model: Predict the probability that a player remains fantasy-relevant
in each future season.

Fantasy-relevant = plays 6+ games AND scores 5+ PPG (PPR).

This answers: "Given this player's age, position, production level, and draft
capital, what's the probability they're still a useful dynasty asset in year N?"

Uses historical data to build transition probabilities, then chains them
forward to get multi-year survival curves.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Minimum threshold to be "fantasy relevant"
MIN_GAMES = 6
MIN_PPG = 5.0


def build_survival_training_data():
    """
    Build training data for the survival model.

    For each player-season where the player was on an NFL roster,
    determine if they were fantasy-relevant that season and the next.

    Features: age, position, current PPG, games played, years in league,
    draft capital, recent trajectory.
    """
    seasonal = pd.read_csv(os.path.join(DATA_DIR, "seasonal_stats.csv"))
    players = pd.read_csv(os.path.join(DATA_DIR, "players.csv"))

    reg = seasonal[seasonal["season_type"] == "REG"].copy()
    reg["ppg"] = reg["fantasy_points_ppr"] / reg["games"].clip(lower=1)

    # Join player info
    player_info = players[players["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    player_info["birth_date"] = pd.to_datetime(player_info["birth_date"], errors="coerce")
    player_info["rookie_season"] = pd.to_numeric(player_info["rookie_season"], errors="coerce")

    reg = reg.merge(
        player_info[["gsis_id", "position", "birth_date", "rookie_season",
                      "draft_round", "draft_pick"]].rename(columns={"gsis_id": "player_id"}),
        on="player_id", how="inner",
    )

    reg["age"] = (
        (pd.to_datetime(reg["season"].astype(str) + "-09-01") - reg["birth_date"]).dt.days / 365.25
    ).round(1)
    reg["years_in_league"] = reg["season"] - reg["rookie_season"]

    # Define relevance
    reg["relevant"] = ((reg["games"] >= MIN_GAMES) & (reg["ppg"] >= MIN_PPG)).astype(int)

    # Position encoding
    for pos in ["QB", "RB", "WR", "TE"]:
        reg[f"pos_{pos}"] = (reg["position"] == pos).astype(int)

    # Draft capital
    reg["draft_round"] = reg["draft_round"].fillna(7)
    reg["draft_pick"] = reg["draft_pick"].fillna(224)
    reg["draft_capital"] = (8 - reg["draft_round"].clip(upper=7)) / 7
    reg["draft_capital_decayed"] = reg["draft_capital"] * (0.8 ** reg["years_in_league"].clip(lower=0))

    # Build pairs: current season -> next season relevance
    seasons = sorted(reg["season"].unique())
    pairs = []

    for i in range(len(seasons) - 1):
        curr_season = seasons[i]
        next_season = seasons[i + 1]

        curr = reg[reg["season"] == curr_season].copy()
        nxt = reg[reg["season"] == next_season][["player_id", "relevant"]].rename(
            columns={"relevant": "next_relevant"}
        )

        merged = curr.merge(nxt, on="player_id", how="left")
        # Players not found next season = not relevant (retired, cut, etc.)
        merged["next_relevant"] = merged["next_relevant"].fillna(0).astype(int)

        pairs.append(merged)

    pairs_df = pd.concat(pairs, ignore_index=True)

    # Add previous season PPG for trajectory
    prev_ppg = reg[["player_id", "season", "ppg", "games", "fantasy_points_ppr"]].copy()
    prev_ppg["season"] = prev_ppg["season"] + 1
    prev_ppg.columns = ["player_id", "season", "prev_ppg", "prev_games", "prev_fpts"]
    pairs_df = pairs_df.merge(prev_ppg, on=["player_id", "season"], how="left")
    pairs_df["ppg_delta"] = pairs_df["ppg"] - pairs_df["prev_ppg"].fillna(pairs_df["ppg"])

    return pairs_df


def train_survival_model():
    """Train XGBoost classifier to predict next-season fantasy relevance."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Building survival training data...")
    pairs_df = build_survival_training_data()

    feature_cols = [
        "age", "years_in_league",
        "pos_QB", "pos_RB", "pos_WR", "pos_TE",
        "games", "ppg", "fantasy_points_ppr", "relevant",
        "draft_capital_decayed", "draft_round", "draft_pick",
        "prev_ppg", "prev_games", "ppg_delta",
    ]

    X = pairs_df[feature_cols].fillna(0)
    y = pairs_df["next_relevant"]

    print(f"Survival training data: {len(X)} samples")
    print(f"  Relevant next season: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Not relevant: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=10,
        scale_pos_weight=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    # Time-series CV
    print("\nRunning time-series cross-validation...")
    sort_idx = pairs_df["season"].argsort()
    X_sorted = X.iloc[sort_idx]
    y_sorted = y.iloc[sort_idx]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_sorted, y_sorted, cv=tscv, scoring="roc_auc")
    print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Train final model
    print("Training final survival model...")
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nSurvival model feature importance:")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

    model.save_model(os.path.join(MODEL_DIR, "xgb_survival.json"))
    print(f"\nSurvival model saved to {os.path.join(MODEL_DIR, 'xgb_survival.json')}")

    return model, feature_cols


def predict_survival_probability(player_row, model, feature_cols, age_curves, years_ahead=5):
    """
    Chain survival probabilities forward for a single player.

    For each future year, estimate P(relevant) by simulating the player
    aging forward and predicting survival at each step.

    Returns list of probabilities: [P(relevant year 1), P(relevant year 2), ...]
    """
    probs = []
    # Start with current features
    features = player_row[feature_cols].fillna(0).copy()

    cumulative_prob = 1.0

    for year in range(years_ahead):
        # Predict P(relevant next season)
        X = pd.DataFrame([features])
        p_survive = model.predict_proba(X)[0][1]

        cumulative_prob *= p_survive
        probs.append(cumulative_prob)

        # Age the player forward for next iteration
        features["age"] = features["age"] + 1
        features["years_in_league"] = features["years_in_league"] + 1
        features["draft_capital_decayed"] = features["draft_capital_decayed"] * 0.8
        # Assume PPG stays roughly the same (the age curve handles decline separately)
        # but reduce games slightly for aging
        features["prev_ppg"] = features["ppg"]
        features["ppg_delta"] = 0

    return probs


if __name__ == "__main__":
    train_survival_model()
