"""
Step 5: Calculate dynasty true value.

Dynasty Value = Σ [ P(alive year N) × E(PPG year N | alive) - ReplacementPPG ] × Discount^N

Architecture:
- Horizon-specific PPG models predict E[PPG | still playing] at 1yr, 2yr, 3yr.
  Trained on inner-join data (only players who actually played in target season).
- Survival model predicts P(still playing) — chained forward year by year.
- Player-specific regression to mean applied at ALL horizons, scaled by
  career consistency (multi-year elite producers regress less).
- Linear percentile-based scaling: 99th percentile maps to ~9500, preserving
  trade math (values are additive) without one outlier compressing everyone.
- Replacement levels calibrated for dynasty roster construction.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

PROJECTION_YEARS = 8
DISCOUNT_RATE = 0.90

DEFAULT_LEAGUE = {
    "teams": 12,
    "qb_slots": 2,  # 2 = superflex
}

# Replacement level: reflects what's actually available on dynasty waivers.
# In a 12-team league with 25-man rosters, ~300 players are rostered.
# Replacement = the best player NOT rostered, i.e. the waiver wire ceiling.
# SF QB: ~2.5 per team → QB30 replacement (scarce position, hoarded)
# RB: ~3.5 per team → RB42 replacement (dynasty rosters hoard RBs heavily)
# WR: ~4.0 per team → WR48 replacement (deep position but many rostered)
# TE: ~1.5 per team → TE18 replacement (shallow position)
REPLACEMENT_LEVEL_MULTIPLIER = {
    "QB": {"1qb": 1.0, "sf": 2.5},
    "RB": {"1qb": 3.0, "sf": 3.5},
    "WR": {"1qb": 3.5, "sf": 4.0},
    "TE": {"1qb": 1.0, "sf": 1.5},
}

# Base regression schedule by projection year.
# Every year gets some regression; further out = more.
# These are then MULTIPLIED by the player's consistency score.
BASE_REGRESSION = [0.05, 0.12, 0.22, 0.32, 0.40, 0.48, 0.55, 0.60]


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


def compute_player_consistency(season_features, current_df):
    """Compute per-player regression strength based on career consistency.

    Returns a dict: {player_id: regression_multiplier} where lower = more
    consistent (regress less) and higher = less track record (regress more).
    Range: 0.3 (elite multi-year producer) to 1.0 (unknown/one-year sample).
    """
    consistency = {}
    for _, player in current_df.iterrows():
        pid = player["player_id"]
        player_history = season_features[season_features["player_id"] == pid]

        if len(player_history) == 0:
            consistency[pid] = 1.0
            continue

        # Count seasons with meaningful production
        good_seasons = (player_history["ppg"] >= 8).sum()
        total_seasons = len(player_history)

        # PPG coefficient of variation
        ppg_values = player_history["ppg"].values
        if len(ppg_values) >= 2 and ppg_values.mean() > 0:
            cv = ppg_values.std() / ppg_values.mean()
        else:
            cv = 0.5

        # Track record: more good seasons = lower regression
        track_record_factor = max(0, 1.0 - (good_seasons - 1) * 0.2)  # 5+ good seasons → 0.0
        variance_factor = min(1.0, cv / 0.5)

        multiplier = 0.3 + 0.7 * (0.6 * track_record_factor + 0.4 * variance_factor)
        consistency[pid] = min(1.0, max(0.3, multiplier))

    return consistency


# Career phase curves: peak trade value age by position.
# Before peak = appreciating asset (multiplier > 1.0)
# After peak = depreciating asset (multiplier < 1.0)
# Derived from when dynasty market values are highest relative to production.
CAREER_PHASE = {
    "QB": {"peak_age": 26, "appreciation_rate": 0.08, "depreciation_rate": 0.06},
    "RB": {"peak_age": 23, "appreciation_rate": 0.12, "depreciation_rate": 0.10},
    "WR": {"peak_age": 24, "appreciation_rate": 0.10, "depreciation_rate": 0.07},
    "TE": {"peak_age": 24, "appreciation_rate": 0.10, "depreciation_rate": 0.06},
}


def compute_asset_multiplier(age, position, ppg_mean, ppg_q90, elite_counts,
                              league_teams=12, current_ppg=0, years_in_league=0):
    """Compute the asset value multiplier for a player.

    Three components:
    1. Career phase: pre-peak players appreciate, post-peak depreciate
    2. Ceiling premium: high upside relative to expectation gets a bonus
    3. Positional scarcity: fewer elite options at position = higher premium

    Guardrail: the youth/phase premium is gated by proven production.
    Players with no meaningful NFL production don't get full appreciation bonus.

    Returns (total_multiplier, phase_mult, ceiling_mult, scarcity_mult) for transparency.
    """
    phase = CAREER_PHASE.get(position, CAREER_PHASE["WR"])
    peak = phase["peak_age"]

    # --- 1. Career phase multiplier ---
    if age <= peak:
        years_to_peak = peak - age
        # Appreciating: younger = higher multiplier, caps at ~1.8
        phase_mult = 1.0 + years_to_peak * phase["appreciation_rate"]
        phase_mult = min(phase_mult, 1.8)
    else:
        years_past = age - peak
        # Depreciating: asymmetric — decline accelerates with age
        base_depreciation = years_past * phase["depreciation_rate"] * (1 + years_past * 0.03)

        # --- Production-aware depreciation ---
        # If a player is still producing at elite level, slow the decline.
        # A 27-yr-old RB scoring 20 PPG shouldn't depreciate like one scoring 10.
        # "Elite" threshold: 15+ PPG means you're a top producer at your position.
        if current_ppg >= 15:
            # Strong production shields up to 40% of the depreciation
            production_shield = min(0.40, (current_ppg - 15) / 20)
            base_depreciation *= (1.0 - production_shield)
        elif current_ppg >= 10:
            # Moderate production shields up to 15%
            production_shield = (current_ppg - 10) / 50
            base_depreciation *= (1.0 - production_shield)

        phase_mult = 1.0 - base_depreciation
        phase_mult = max(phase_mult, 0.35)

    # --- GUARDRAIL: Gate youth premium by proven production ---
    # Unproven players (low PPG or <2 NFL seasons) get a dampened phase multiplier.
    # This prevents rookies/unproven players from getting massive appreciation boosts.
    if phase_mult > 1.0:
        # Production gate: how much of the premium to apply (0 to 1)
        # Ramps from 0.3 at 0 PPG to 1.0 at 12+ PPG
        production_gate = min(1.0, 0.3 + (current_ppg / 12.0) * 0.7)
        # Experience gate: first-year players get less premium than 2+ year players
        experience_gate = min(1.0, 0.4 + years_in_league * 0.3)
        # Combined gate (take the lower — need both production AND experience)
        gate = min(production_gate, experience_gate)
        # Apply: phase_mult above 1.0 is scaled by the gate
        phase_mult = 1.0 + (phase_mult - 1.0) * gate

    # --- 2. Ceiling premium ---
    # How much upside does q90 show relative to mean prediction?
    if ppg_mean > 0:
        ceiling_ratio = (ppg_q90 - ppg_mean) / ppg_mean
    else:
        ceiling_ratio = 0

    # Youth amplifies ceiling value (young + high ceiling = dynasty gold)
    youth_factor = max(0, (28 - age) / 10)  # peaks at age 18, zero at 28+
    ceiling_mult = 1.0 + min(ceiling_ratio * (1 + youth_factor * 0.5), 0.25)

    # --- 3. Positional scarcity ---
    # Fewer elite players at this position = each one is more valuable.
    # In dynasty, the gap between positions with few elite options (RB) and
    # many (QB in SF) is massive — worth up to 1.6x premium.
    elite_at_pos = elite_counts.get(position, 10)
    # Benchmark: 15+ elite players = no scarcity premium (e.g. QB in SF)
    # Below that, premium scales up steeply
    if elite_at_pos >= 15:
        scarcity_mult = 1.0
    elif elite_at_pos > 0:
        scarcity_mult = 1.0 + max(0, (12 - elite_at_pos) / 12) * 0.6
    else:
        scarcity_mult = 1.4
    scarcity_mult = min(scarcity_mult, 1.60)

    total = phase_mult * ceiling_mult * scarcity_mult
    # Cap total asset multiplier to prevent runaway values
    total = min(total, 2.5)
    return total, phase_mult, ceiling_mult, scarcity_mult


def compute_weighted_career_ppg(player_id, season_features):
    """Compute a recency-weighted career PPG for anchoring.

    Uses up to 3 most recent seasons with weights: 50% / 30% / 20%.
    If fewer seasons exist, renormalizes across what's available.
    Returns (weighted_ppg, total_games) or (0, 0) if no data.
    """
    history = season_features[season_features["player_id"] == player_id].sort_values("season", ascending=False)
    if len(history) == 0:
        return 0, 0

    # Weights for most recent, 2nd most recent, 3rd most recent
    raw_weights = [0.50, 0.30, 0.20]
    n = min(len(history), 3)
    weights = raw_weights[:n]
    # Renormalize so they sum to 1
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    weighted_ppg = 0
    total_games = 0
    for i in range(n):
        row = history.iloc[i]
        weighted_ppg += weights[i] * row["ppg"]
        total_games += row["games"]

    return weighted_ppg, total_games


def anchor_prediction(model_pred, anchor_ppg, games_played, consistency,
                       seasons_missed=0, horizon=1):
    """Blend model prediction with career-weighted PPG as an anchor.

    The XGBoost model systematically regresses elite performers toward the mean
    (R² ~0.67). For players with strong track records, their weighted career PPG
    is a better starting point than a pure model prediction.

    anchor_ppg should be the recency-weighted career PPG (not just last season).

    Anchor weight depends on:
    - Games played: more career games = more reliable anchor
    - Consistency: lower variance = more trustworthy anchor
    - Staleness: missed seasons reduce trust in the anchor
    - Horizon: further-out predictions rely more on the model

    Returns blended PPG prediction.
    """
    if anchor_ppg <= 0 or games_played < 6:
        return model_pred  # No reliable anchor — trust the model

    # Base anchor weight: how much to trust career PPG vs model
    base_weight = 0.45

    # Games bonus: more total games = more reliable anchor
    games_factor = min(1.0, games_played / 30.0)  # full credit at 30+ career games
    base_weight += 0.15 * games_factor

    # Consistency bonus: low-variance players get more anchor weight
    # consistency is 0.3 (very consistent) to 1.0 (unknown/volatile)
    consistency_bonus = max(0, (1.0 - consistency) / 0.7) * 0.10
    base_weight += consistency_bonus

    # Staleness penalty: each missed season halves the anchor weight
    if seasons_missed > 0:
        base_weight *= 0.5 ** seasons_missed

    # Horizon decay: further out predictions rely more on model
    # Year 1: full anchor, Year 2: 70%, Year 3+: 50%
    horizon_decay = {1: 1.0, 2: 0.70, 3: 0.50}
    decay = horizon_decay.get(horizon, 0.50)

    anchor_weight = base_weight * decay
    anchor_weight = min(anchor_weight, 0.70)  # never more than 70% anchor

    return model_pred * (1 - anchor_weight) + anchor_ppg * anchor_weight


def load_horizon_model(filename):
    """Try to load an XGBoost model, return None if not found."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def calculate_dynasty_values(league=None):
    """Calculate dynasty true value for all current players."""
    if league is None:
        league = DEFAULT_LEAGUE

    # Load horizon-specific PPG models (predict E[PPG | still playing])
    models = {
        1: {"mean": load_horizon_model("xgb_ppg_predictor.json"),
            "q10": load_horizon_model("xgb_ppg_q10.json"),
            "q90": load_horizon_model("xgb_ppg_q90.json")},
        2: {"mean": load_horizon_model("xgb_ppg_2yr.json"),
            "q10": load_horizon_model("xgb_ppg_2yr_q10.json"),
            "q90": load_horizon_model("xgb_ppg_2yr_q90.json")},
        3: {"mean": load_horizon_model("xgb_ppg_3yr.json"),
            "q10": load_horizon_model("xgb_ppg_3yr_q10.json"),
            "q90": load_horizon_model("xgb_ppg_3yr_q90.json")},
    }

    if models[1]["mean"] is None:
        raise RuntimeError("1-year PPG model not found. Run train_model.py first.")
    for h in [2, 3]:
        if models[h]["mean"] is None:
            print(f"  Warning: {h}yr model not found, falling back to 1yr model")
            models[h] = models[1]

    # Load survival model
    survival_model = xgb.XGBClassifier()
    survival_model.load_model(os.path.join(MODEL_DIR, "xgb_survival.json"))

    # Position priors
    priors_path = os.path.join(MODEL_DIR, "position_priors.json")
    if os.path.exists(priors_path):
        with open(priors_path) as f:
            position_priors = json.load(f)
    else:
        position_priors = {
            "QB": {"mean_ppg": 14.0}, "RB": {"mean_ppg": 10.0},
            "WR": {"mean_ppg": 10.0}, "TE": {"mean_ppg": 8.0},
        }

    # Load data
    current_df = pd.read_csv(os.path.join(DATA_DIR, "current_features.csv"))
    season_features = pd.read_csv(os.path.join(DATA_DIR, "season_features.csv"))

    # Replacement levels
    replacement = compute_replacement_levels(season_features, league)
    print("Replacement-level PPG by position:")
    for pos, ppg in replacement.items():
        print(f"  {pos}: {ppg:.1f} PPG")

    # Player consistency
    print("Computing player consistency scores...")
    consistency = compute_player_consistency(season_features, current_df)

    # Feature columns
    ppg_non_feature = ["player_id", "season", "position", "ppg_target_col",
                       "next_season_ppg", "next_season_games",
                       "target_ppg_2yr", "target_games_2yr",
                       "target_ppg_3yr", "target_games_3yr",
                       "seasons_missed", "data_staleness"]

    survival_feature_cols = [
        "age", "years_in_league",
        "pos_QB", "pos_RB", "pos_WR", "pos_TE",
        "games", "ppg", "fantasy_points_ppr", "relevant",
        "draft_capital_decayed", "draft_round", "draft_pick",
        "prev_ppg", "prev_games", "ppg_delta",
    ]

    # Pre-compute PPG predictions at each horizon for all players
    predictions = {}
    for horizon in [1, 2, 3]:
        m = models[horizon]
        model_features = m["mean"].get_booster().feature_names
        available = [c for c in current_df.columns
                     if c not in ppg_non_feature and c in model_features]
        X_h = current_df[available].fillna(0)

        predictions[horizon] = {
            "mean": m["mean"].predict(X_h).clip(min=0),
            "q10": m["q10"].predict(X_h).clip(min=0) if m["q10"] else None,
            "q90": m["q90"].predict(X_h).clip(min=0) if m["q90"] else None,
        }

    # Year → which horizon model to use
    # Years 0-2 use direct models; years 3+ use the 3yr model + regression
    horizon_map = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3}

    # Compute elite counts for positional scarcity
    # "Elite" = predicted year-1 PPG above 2× replacement level
    elite_counts = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        repl = replacement.get(pos, 10)
        pos_mask = current_df["position"] == pos
        pos_preds = predictions[1]["mean"][pos_mask.values]
        elite_counts[pos] = int((pos_preds >= repl * 1.5).sum())
    print("Elite player counts by position (PPG >= 1.5× replacement):")
    for pos, count in elite_counts.items():
        print(f"  {pos}: {count}")

    # Pre-compute career-weighted PPG for anchoring (multi-season average)
    print("Computing career-weighted PPG anchors...")
    career_anchors = {}
    for _, player in current_df.iterrows():
        pid = player["player_id"]
        w_ppg, w_games = compute_weighted_career_ppg(pid, season_features)
        career_anchors[pid] = {"ppg": w_ppg, "games": w_games}

    # Project dynasty value for each player
    results = []
    for idx, player in current_df.iterrows():
        pos = player["position"]
        age = player["age"]
        repl_ppg = replacement.get(pos, 0)
        seasons_missed = player.get("seasons_missed", 0)
        prior_mean = position_priors.get(pos, {"mean_ppg": 8.0}).get("mean_ppg", 8.0)
        player_reg = consistency.get(player["player_id"], 0.7)

        total_value = 0
        total_value_low = 0
        total_value_high = 0

        # Build survival features for chaining forward
        surv_features = {}
        for col in survival_feature_cols:
            surv_features[col] = player.get(col, 0)
            if pd.isna(surv_features[col]):
                surv_features[col] = 0

        cumulative_survival = 1.0
        # Use career-weighted PPG as anchor (not just most recent season)
        anchor = career_anchors.get(player["player_id"], {"ppg": 0, "games": 0})
        anchor_ppg = anchor["ppg"]
        anchor_games = anchor["games"]

        # Capture year-by-year projection data for player detail pages
        yearly_projections = []

        for year in range(PROJECTION_YEARS):
            horizon = horizon_map[year]

            # --- Survival probability (chained forward) ---
            surv_features["age"] = age + year
            surv_features["years_in_league"] = player["years_in_league"] + year
            surv_features["draft_capital_decayed"] = (
                player.get("draft_capital_decayed", 0) * (0.8 ** year)
            )
            # Use anchored year-1 PPG prediction as the survival model's PPG input
            yr1_ppg_raw = predictions[1]["mean"][idx]
            yr1_ppg = anchor_prediction(
                yr1_ppg_raw, anchor_ppg, anchor_games, player_reg,
                seasons_missed, horizon=1,
            )
            surv_features["ppg"] = yr1_ppg
            surv_features["fantasy_points_ppr"] = yr1_ppg * 17
            surv_features["relevant"] = 1 if yr1_ppg >= 5 else 0

            X_surv = pd.DataFrame([surv_features])[survival_feature_cols].fillna(0)
            p_survive = survival_model.predict_proba(X_surv)[0][1]

            # Survival floor for elite young players: the model is too pessimistic
            # for high-PPG players in their physical prime. A 23-yr-old scoring
            # 20+ PPG has ~97% chance of being relevant next year, not 91%.
            projected_age = age + year
            if projected_age < 28 and yr1_ppg >= 15:
                elite_floor = 0.92 + min(0.05, (yr1_ppg - 15) / 100)
                p_survive = max(p_survive, elite_floor)
            elif projected_age < 30 and yr1_ppg >= 12:
                p_survive = max(p_survive, 0.88)

            cumulative_survival *= p_survive

            # --- PPG prediction (conditional on still playing) ---
            # Anchor model predictions to actual recent PPG
            ppg_mean_raw = predictions[horizon]["mean"][idx]
            ppg_q10_raw = predictions[horizon]["q10"][idx] if predictions[horizon]["q10"] is not None else ppg_mean_raw * 0.6
            ppg_q90_raw = predictions[horizon]["q90"][idx] if predictions[horizon]["q90"] is not None else ppg_mean_raw * 1.4

            ppg_mean = anchor_prediction(
                ppg_mean_raw, anchor_ppg, anchor_games, player_reg,
                seasons_missed, horizon=horizon,
            )
            ppg_q10 = anchor_prediction(
                ppg_q10_raw, anchor_ppg * 0.8, anchor_games, player_reg,
                seasons_missed, horizon=horizon,
            )
            ppg_q90 = anchor_prediction(
                ppg_q90_raw, anchor_ppg * 1.15, anchor_games, player_reg,
                seasons_missed, horizon=horizon,
            )

            # --- Player-specific regression to mean ---
            reg_weight = BASE_REGRESSION[year] * player_reg
            ppg_mean = ppg_mean * (1 - reg_weight) + prior_mean * reg_weight
            ppg_q10 = ppg_q10 * (1 - reg_weight) + prior_mean * 0.6 * reg_weight
            ppg_q90 = ppg_q90 * (1 - reg_weight) + prior_mean * 1.4 * reg_weight

            # --- PAR: pure value above replacement ---
            par = max(0, ppg_mean - repl_ppg)
            par_low = max(0, ppg_q10 - repl_ppg)
            par_high = max(0, ppg_q90 - repl_ppg)

            # --- Discount ---
            discount = DISCOUNT_RATE ** year

            # --- Season value = P(alive) × PAR × discount ---
            total_value += cumulative_survival * par * discount
            total_value_low += cumulative_survival * par_low * discount
            total_value_high += cumulative_survival * par_high * discount

            # Store projection for this year
            yearly_projections.append({
                "year": int(year + 1),
                "projAge": round(float(age + year), 1),
                "ppgMean": round(float(ppg_mean), 1),
                "ppgLow": round(float(ppg_q10), 1),
                "ppgHigh": round(float(ppg_q90), 1),
                "survival": round(float(cumulative_survival), 3),
                "seasonValue": round(float(cumulative_survival * par * discount), 2),
            })

        # Staleness penalty
        if seasons_missed > 0:
            staleness_penalty = 0.85 ** seasons_missed
            total_value *= staleness_penalty
            total_value_low *= staleness_penalty
            total_value_high *= staleness_penalty

        # Asset multiplier: converts production value → dynasty value
        ppg_mean_1yr = float(predictions[1]["mean"][idx])
        ppg_q90_1yr = float(predictions[1]["q90"][idx]) if predictions[1]["q90"] is not None else ppg_mean_1yr * 1.3
        # Anchor the 1yr prediction for the asset multiplier too
        ppg_mean_1yr_anchored = anchor_prediction(
            ppg_mean_1yr, anchor_ppg, anchor_games, player_reg, seasons_missed, horizon=1,
        )
        asset_mult, phase_mult, ceiling_mult, scarcity_mult = compute_asset_multiplier(
            age, pos, ppg_mean_1yr_anchored, ppg_q90_1yr, elite_counts, league["teams"],
            current_ppg=anchor_ppg, years_in_league=player["years_in_league"],
        )

        results.append({
            "player_id": player["player_id"],
            "position": pos,
            "age": round(age, 1),
            "current_ppg": round(player["ppg"], 1),
            "predicted_next_ppg": round(ppg_mean_1yr_anchored, 1),
            "replacement_ppg": round(repl_ppg, 1),
            "seasons_missed": int(seasons_missed),
            "consistency": round(player_reg, 2),
            "production_value": round(total_value, 2),
            "dynasty_value_raw": round(total_value * asset_mult, 2),
            "dynasty_value_low": round(total_value_low * asset_mult, 2),
            "dynasty_value_high": round(total_value_high * asset_mult, 2),
            "asset_multiplier": round(asset_mult, 2),
            "phase_mult": round(phase_mult, 2),
            "ceiling_mult": round(ceiling_mult, 2),
            "scarcity_mult": round(scarcity_mult, 2),
            "yearly_projections": yearly_projections,
        })

    results_df = pd.DataFrame(results)

    # Percentile-based linear scaling.
    # 99th percentile → 9500, allowing elite outliers to exceed 9999.
    # Values are linearly proportional to raw PAR — trade math works.
    raw = results_df["dynasty_value_raw"].clip(lower=0)
    p99 = np.percentile(raw[raw > 0], 99) if (raw > 0).sum() > 0 else 1.0
    scale_factor = 9500.0 / p99 if p99 > 0 else 1.0

    results_df["dynasty_value"] = (raw * scale_factor).round(0).astype(int)

    # Range shrinkage: accumulating q10/q90 independently over 8 years
    # overstates realistic uncertainty (worst-case compounds unrealistically).
    # Shrink the range toward the point estimate for a tighter, more usable band.
    # shrink=0.65 blends 65% mean + 35% raw quantile → ~50% confidence interval.
    RANGE_SHRINK = 0.65

    raw_low = results_df["dynasty_value_low"].clip(lower=0) * scale_factor
    raw_high = results_df["dynasty_value_high"].clip(lower=0) * scale_factor
    center = results_df["dynasty_value"].astype(float)

    tight_low = center * RANGE_SHRINK + raw_low * (1 - RANGE_SHRINK)
    tight_high = center * RANGE_SHRINK + raw_high * (1 - RANGE_SHRINK)

    results_df["value_low"] = tight_low.clip(lower=0).round(0).astype(int)
    results_df["value_high"] = tight_high.round(0).astype(int)

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

    # Assign value tiers based on percentile within the population
    def assign_tier(value, thresholds):
        if value >= thresholds["elite"]:
            return "Elite"
        elif value >= thresholds["star"]:
            return "Star"
        elif value >= thresholds["starter"]:
            return "Starter"
        elif value >= thresholds["bench"]:
            return "Bench"
        else:
            return "Waiver"

    positive = results_df[results_df["dynasty_value"] > 0]["dynasty_value"]
    tier_thresholds = {
        "elite": np.percentile(positive, 95),
        "star": np.percentile(positive, 85),
        "starter": np.percentile(positive, 65),
        "bench": np.percentile(positive, 40),
    }
    results_df["tier"] = results_df["dynasty_value"].apply(lambda v: assign_tier(v, tier_thresholds))

    # Print rankings
    print("\n" + "=" * 120)
    print("TRUEVALUE DYNASTY RANKINGS")
    print("Model-driven valuations — KTC shown as market reference, not ground truth")
    print("=" * 120)
    print(f"  Tiers: Elite (top 5%) | Star (top 15%) | Starter (top 35%) | Bench | Waiver")
    print(f"  Thresholds: Elite {tier_thresholds['elite']:.0f}+ | Star {tier_thresholds['star']:.0f}+ "
          f"| Starter {tier_thresholds['starter']:.0f}+ | Bench {tier_thresholds['bench']:.0f}+")

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = results_df[results_df["position"] == pos].head(20)
        print(f"\n{'─' * 120}")
        print(f"  TOP 20 {pos}s")
        print(f"{'─' * 120}")
        print(f"  {'Rk':<5}{'Player':<25}{'Tier':<9}{'Age':<6}{'PPG':<7}{'Pred':<7}"
              f"{'Value':<8}{'Low':<7}{'High':<7}"
              f"{'Asset':<7}{'Consist':<9}"
              f"{'Mkt':<7}{'Miss'}")
        print(f"  {'─'*5}{'─'*25}{'─'*9}{'─'*6}{'─'*7}{'─'*7}"
              f"{'─'*8}{'─'*7}{'─'*7}"
              f"{'─'*7}{'─'*9}"
              f"{'─'*7}{'─'*4}")
        for rank, row in pos_df.iterrows():
            ktc_str = f"{row['ktc_value']:.0f}" if pd.notna(row.get('ktc_value')) else "—"
            miss_str = f"{row['seasons_missed']}" if row['seasons_missed'] > 0 else ""
            print(f"  {rank:<5}{row['player_name']:<25}{row['tier']:<9}{row['age']:<6.1f}"
                  f"{row['current_ppg']:<7.1f}{row['predicted_next_ppg']:<7.1f}"
                  f"{row['dynasty_value']:<8}{row['value_low']:<7}{row['value_high']:<7}"
                  f"{row['asset_multiplier']:<7.2f}{row['consistency']:<9.2f}"
                  f"{ktc_str:<7}{miss_str}")

    # Trade alpha: where model disagrees most with market consensus
    if "value_vs_ktc" in results_df.columns:
        has_ktc = results_df.dropna(subset=["value_vs_ktc"])
        # Only show players with meaningful value on at least one side
        meaningful = has_ktc[
            (has_ktc["dynasty_value"] >= 1000) | (has_ktc["ktc_value"] >= 3000)
        ]
        # Compute rank difference (more intuitive than value difference)
        ktc_ranked = has_ktc.sort_values("ktc_value", ascending=False).reset_index(drop=True)
        ktc_ranked["ktc_rank"] = ktc_ranked.index + 1
        ktc_rank_map = ktc_ranked.set_index("player_id")["ktc_rank"].to_dict()
        meaningful = meaningful.copy()
        meaningful["ktc_rank"] = meaningful["player_id"].map(ktc_rank_map)
        meaningful["rank_diff"] = meaningful["ktc_rank"] - meaningful.index  # positive = model ranks higher

        print(f"\n{'=' * 120}")
        print("POTENTIAL TRADE ALPHA (where model disagrees with market)")
        print("Positive rank diff = model values higher than market (potential buy)")
        print("Negative rank diff = model values lower than market (potential sell)")
        print(f"{'=' * 120}")

        print(f"\n  BUY TARGETS (model ranks significantly higher than market):")
        buys = meaningful.nlargest(15, "rank_diff")
        buys = buys[buys["rank_diff"] > 10]  # only show meaningful gaps
        for _, row in buys.iterrows():
            print(f"    {row['player_name']:<25}{row['position']:<4}Age {row['age']:<5.1f}"
                  f"  Model Rk: {row.name:<5} Mkt Rk: {row['ktc_rank']:<5.0f}"
                  f"  ({row['rank_diff']:+.0f} spots)  Value: {row['dynasty_value']}")

        print(f"\n  SELL CANDIDATES (market ranks significantly higher than model):")
        sells = meaningful.nsmallest(15, "rank_diff")
        sells = sells[sells["rank_diff"] < -10]  # only show meaningful gaps
        for _, row in sells.iterrows():
            print(f"    {row['player_name']:<25}{row['position']:<4}Age {row['age']:<5.1f}"
                  f"  Model Rk: {row.name:<5} Mkt Rk: {row['ktc_rank']:<5.0f}"
                  f"  ({row['rank_diff']:+.0f} spots)  Value: {row['dynasty_value']}")

    print(f"\n\nFull rankings saved to {os.path.join(DATA_DIR, 'dynasty_values.csv')}")
    export_frontend_json(results_df, season_features=season_features)
    return results_df


def build_player_history(season_features):
    """Build historical season-by-season data per player for detail pages."""
    history = {}
    cols_of_interest = [
        "season", "games", "ppg", "recent_team",
        "passing_yards", "passing_tds", "interceptions", "completions", "attempts",
        "rushing_yards", "rushing_tds", "carries",
        "receiving_yards", "receiving_tds", "receptions", "targets",
        "target_share", "fantasy_points_ppr",
    ]
    available = [c for c in cols_of_interest if c in season_features.columns]

    for pid, group in season_features.groupby("player_id"):
        group = group.sort_values("season")
        seasons = []
        for _, row in group.iterrows():
            s = {}
            for col in available:
                val = row.get(col)
                if pd.isna(val):
                    s[col] = None
                elif isinstance(val, (np.integer, np.int64)):
                    s[col] = int(val)
                elif isinstance(val, (np.floating, np.float64, float)):
                    s[col] = round(float(val), 2)
                else:
                    s[col] = val
            seasons.append(s)
        history[pid] = seasons
    return history


def derive_strengths_weaknesses(row):
    """Derive human-readable strengths and weaknesses from player data."""
    strengths = []
    weaknesses = []

    age = row.get("age", 30)
    ppg = row.get("current_ppg", 0)
    pred = row.get("predicted_next_ppg", 0)
    phase = row.get("phase_mult", 1.0)
    ceiling = row.get("ceiling_mult", 1.0)
    scarcity = row.get("scarcity_mult", 1.0)
    consistency = row.get("consistency", 1.0)
    missed = row.get("seasons_missed", 0)
    pos = row.get("position", "")

    # Youth / age
    peak_ages = {"QB": 26, "RB": 23, "WR": 24, "TE": 24}
    peak = peak_ages.get(pos, 25)
    if age <= peak:
        strengths.append({"label": "Youth Premium", "detail": f"Age {age} — still appreciating toward peak ({peak})", "impact": "high"})
    elif age <= peak + 3:
        strengths.append({"label": "Prime Window", "detail": f"Age {age} — in or near prime years", "impact": "medium"})
    elif age >= peak + 6:
        weaknesses.append({"label": "Age Decline", "detail": f"Age {age} — well past positional peak of {peak}", "impact": "high"})
    elif age >= peak + 3:
        weaknesses.append({"label": "Post-Peak", "detail": f"Age {age} — past positional peak of {peak}", "impact": "medium"})

    # Production
    if ppg >= 20:
        strengths.append({"label": "Elite Producer", "detail": f"{ppg} PPG — top-tier fantasy production", "impact": "high"})
    elif ppg >= 15:
        strengths.append({"label": "Strong Producer", "detail": f"{ppg} PPG — solid starter-level output", "impact": "medium"})
    elif ppg < 8 and ppg > 0:
        weaknesses.append({"label": "Low Production", "detail": f"{ppg} PPG — below starter threshold", "impact": "medium"})

    # Projected growth/decline
    if pred > ppg * 1.15 and ppg > 0:
        strengths.append({"label": "Projected Growth", "detail": f"Model projects {pred} PPG (up from {ppg})", "impact": "medium"})
    elif pred < ppg * 0.85 and ppg > 5:
        weaknesses.append({"label": "Projected Decline", "detail": f"Model projects {pred} PPG (down from {ppg})", "impact": "medium"})

    # Consistency
    if consistency <= 0.4:
        strengths.append({"label": "Rock Solid", "detail": "Extremely consistent year-to-year — low regression risk", "impact": "medium"})
    elif consistency <= 0.55:
        strengths.append({"label": "Consistent", "detail": "Reliable multi-year track record", "impact": "low"})
    elif consistency >= 0.85:
        weaknesses.append({"label": "Unproven", "detail": "Limited track record — high regression risk", "impact": "medium"})
    elif consistency >= 0.7:
        weaknesses.append({"label": "Volatile", "detail": "Production has varied significantly year-to-year", "impact": "low"})

    # Ceiling
    if ceiling >= 1.15:
        strengths.append({"label": "High Ceiling", "detail": "Significant upside in projection range", "impact": "medium"})

    # Scarcity
    if scarcity >= 1.3:
        strengths.append({"label": "Positional Scarcity", "detail": f"Few elite {pos}s available — scarcity premium", "impact": "high"})
    elif scarcity >= 1.15:
        strengths.append({"label": "Scarce Position", "detail": f"Limited elite {pos} supply adds value", "impact": "low"})

    # Phase (depreciation)
    if phase < 0.7:
        weaknesses.append({"label": "Heavy Depreciation", "detail": "Age-based value decline significantly impacts dynasty worth", "impact": "high"})
    elif phase < 0.85:
        weaknesses.append({"label": "Depreciating", "detail": "Past peak — dynasty value declining with age", "impact": "medium"})

    # Missed time
    if missed >= 2:
        weaknesses.append({"label": "Extended Absence", "detail": f"Missed {missed} season(s) — staleness penalty applied", "impact": "high"})
    elif missed == 1:
        weaknesses.append({"label": "Missed Time", "detail": "Missed a season — some uncertainty in projection", "impact": "low"})

    return strengths, weaknesses


def export_frontend_json(results_df, season_features=None):
    """Export dynasty values to frontend/data.json with enriched player detail data."""
    import json as json_mod
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    if not os.path.isdir(frontend_dir):
        return

    # Build historical data from raw seasonal_stats.csv (has full stat totals)
    player_history = {}
    raw_stats_path = os.path.join(DATA_DIR, "seasonal_stats.csv")
    if os.path.exists(raw_stats_path):
        raw_stats = pd.read_csv(raw_stats_path)
        # Aggregate weekly rows per player-season (nflverse data is per-week sometimes)
        # The seasonal file should already be per-season, but ensure we have PPG
        if "fantasy_points_ppr" in raw_stats.columns and "games" in raw_stats.columns:
            raw_stats["ppg"] = raw_stats["fantasy_points_ppr"] / raw_stats["games"].clip(lower=1)
        player_history = build_player_history(raw_stats)
    elif season_features is not None:
        player_history = build_player_history(season_features)

    # Load player info for teams/headshots
    players_path = os.path.join(DATA_DIR, "players.csv")
    team_map = {}
    headshot_map = {}
    if os.path.exists(players_path):
        players_info = pd.read_csv(players_path)
        team_col = "team_abbr" if "team_abbr" in players_info.columns else "latest_team"
        team_map = players_info.set_index("gsis_id")[team_col].to_dict()
        if "headshot" in players_info.columns:
            headshot_map = players_info.set_index("gsis_id")["headshot"].to_dict()
        elif "headshot_url" in players_info.columns:
            headshot_map = players_info.set_index("gsis_id")["headshot_url"].to_dict()

    records = []
    for rank, row in results_df.iterrows():
        pid = row.get("player_id", "")
        strengths, weaknesses = derive_strengths_weaknesses(row)

        rec = {
            "rank": int(rank),
            "id": pid,
            "name": row.get("player_name", ""),
            "pos": row.get("position", ""),
            "team": team_map.get(pid, ""),
            "tier": row.get("tier", ""),
            "age": round(float(row.get("age", 0)), 1),
            "ppg": round(float(row.get("current_ppg", 0)), 1),
            "predPpg": round(float(row.get("predicted_next_ppg", 0)), 1),
            "replacementPpg": round(float(row.get("replacement_ppg", 0)), 1),
            "value": int(row.get("dynasty_value", 0)),
            "valueLow": int(row.get("value_low", 0)),
            "valueHigh": int(row.get("value_high", 0)),
            "productionValue": round(float(row.get("production_value", 0)), 1),
            "consistency": round(float(row.get("consistency", 0)), 2),
            "assetMult": round(float(row.get("asset_multiplier", 1.0)), 2),
            "phaseMult": round(float(row.get("phase_mult", 1.0)), 2),
            "ceilingMult": round(float(row.get("ceiling_mult", 1.0)), 2),
            "scarcityMult": round(float(row.get("scarcity_mult", 1.0)), 2),
            "marketValue": None if pd.isna(row.get("ktc_value")) else int(row["ktc_value"]),
            "missed": int(row.get("seasons_missed", 0)),
            "headshot": headshot_map.get(pid, None),
            # Enriched detail data
            "projections": row.get("yearly_projections", []),
            "history": player_history.get(pid, []),
            "strengths": strengths,
            "weaknesses": weaknesses,
        }
        # Clean NaN headshots
        if pd.isna(rec.get("headshot")):
            rec["headshot"] = None
        if pd.isna(rec.get("team")):
            rec["team"] = ""
        records.append(rec)

    out_path = os.path.join(frontend_dir, "data.json")
    with open(out_path, "w") as f:
        json_mod.dump(records, f)
    print(f"Frontend data exported to {out_path} ({len(records)} players)")


if __name__ == "__main__":
    calculate_dynasty_values()
