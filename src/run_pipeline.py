"""
Run the full TrueValueDynasty pipeline:
1. Collect NFL data + KTC values
2. Build features and training pairs
3. Derive age curves
4. Train survival model
5. Train next-season PPG predictor
6. Calculate dynasty true values and compare vs KTC
"""
from collect_data import collect_all
from collect_ktc import fetch_ktc_values
from build_features import main as build_features
from age_curves import build_age_curves
from survival_model import train_survival_model
from train_model import train_ppg_model
from dynasty_value import calculate_dynasty_values


def main():
    print("=" * 95)
    print("TRUEVALUE DYNASTY - Full Pipeline (3-Model Approach)")
    print("=" * 95)

    print("\n[1/7] Collecting NFL data (2010-2025 via nflverse + nfl_data_py)...")
    collect_all()

    print("\n[2/7] Collecting KTC values (for comparison only)...")
    fetch_ktc_values()

    print("\n[3/7] Building features and training pairs...")
    build_features()

    print("\n[4/7] Deriving position-specific age curves...")
    build_age_curves()

    print("\n[5/7] Training survival model...")
    train_survival_model()

    print("\n[6/7] Training next-season PPG predictor...")
    train_ppg_model()

    print("\n[7/7] Calculating dynasty true values...")
    results = calculate_dynasty_values()

    print("\n" + "=" * 95)
    print("Pipeline complete!")
    print("=" * 95)


if __name__ == "__main__":
    main()
