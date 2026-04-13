"""
Derive position-specific age curves from historical data.

These curves tell us: given a player's current age and production,
how is their PPG expected to change in future seasons?

The curves are used to project production forward across remaining career years.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def build_age_curves():
    """
    Build age curves using the delta method:
    For each player who played consecutive seasons, compute the % change in PPG.
    Average these deltas by age and position to get expected YoY change.
    """
    season_features = pd.read_csv(os.path.join(DATA_DIR, "season_features.csv"))

    # Only keep players with meaningful production (avoid noise from bench players)
    df = season_features[season_features["games"] >= 6].copy()

    # Create consecutive season pairs per player
    df_sorted = df.sort_values(["player_id", "season"])

    pairs = []
    for pid, group in df_sorted.groupby("player_id"):
        group = group.sort_values("season")
        for i in range(len(group) - 1):
            curr = group.iloc[i]
            nxt = group.iloc[i + 1]
            # Must be consecutive seasons
            if nxt["season"] - curr["season"] == 1 and curr["ppg"] > 1:
                pairs.append({
                    "player_id": pid,
                    "position": curr["position"],
                    "age": int(round(curr["age"])),
                    "curr_ppg": curr["ppg"],
                    "next_ppg": nxt["ppg_target_col"],
                    "ppg_ratio": nxt["ppg_target_col"] / curr["ppg"],
                })

    pairs_df = pd.DataFrame(pairs)
    print(f"Age curve pairs: {len(pairs_df)}")

    # Compute median ratio at each age/position combo
    # Use median to reduce outlier impact
    age_curves = pairs_df.groupby(["position", "age"]).agg(
        ppg_retention=("ppg_ratio", "median"),
        sample_size=("ppg_ratio", "count"),
    ).reset_index()

    # Only keep ages with enough samples (at least 10 player-seasons)
    age_curves = age_curves[age_curves["sample_size"] >= 10].copy()

    # Smooth the curves slightly using rolling average within position
    smoothed = []
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_data = age_curves[age_curves["position"] == pos].sort_values("age").copy()
        pos_data["ppg_retention_smooth"] = pos_data["ppg_retention"].rolling(3, center=True, min_periods=1).mean()
        smoothed.append(pos_data)

    age_curves_smooth = pd.concat(smoothed, ignore_index=True)

    print("\nAge curves (smoothed PPG retention ratio by position):")
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_data = age_curves_smooth[age_curves_smooth["position"] == pos]
        print(f"\n  {pos}:")
        for _, row in pos_data.iterrows():
            bar = "█" * int(row["ppg_retention_smooth"] * 20)
            print(f"    Age {row['age']:2.0f}: {row['ppg_retention_smooth']:.3f} {bar}  (n={row['sample_size']:.0f})")

    age_curves_smooth.to_csv(os.path.join(DATA_DIR, "age_curves.csv"), index=False)
    print(f"\nSaved to {os.path.join(DATA_DIR, 'age_curves.csv')}")

    return age_curves_smooth


def get_retention_factor(position, age, age_curves_df):
    """
    Get the expected PPG retention factor for a given position and age.
    Returns the ratio: expected_next_ppg / current_ppg
    If age is outside the curve range, extrapolate conservatively.
    """
    pos_curve = age_curves_df[age_curves_df["position"] == position].sort_values("age")

    if len(pos_curve) == 0:
        return 0.95  # default conservative decline

    age_rounded = int(round(age))

    match = pos_curve[pos_curve["age"] == age_rounded]
    if len(match) > 0:
        return match.iloc[0]["ppg_retention_smooth"]

    # Extrapolate: if older than curve, assume accelerating decline
    max_age = pos_curve["age"].max()
    min_age = pos_curve["age"].min()

    if age_rounded > max_age:
        last_retention = pos_curve.iloc[-1]["ppg_retention_smooth"]
        years_beyond = age_rounded - max_age
        return last_retention * (0.93 ** years_beyond)  # steeper decline
    elif age_rounded < min_age:
        return pos_curve.iloc[0]["ppg_retention_smooth"]

    return 0.95


if __name__ == "__main__":
    build_age_curves()
