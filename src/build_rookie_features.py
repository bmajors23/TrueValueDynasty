"""
Aggregate per-season college data into a single feature row per NFL player
for training the rookie model.

Inputs:
  data/college/nfl_college_career.csv      — one row per (NFL player, season)
  data/college/nfl_recruit_linkage.csv     — one row per NFL player (recruit)
  data/college/YYYY/team_stats.parquet     — team totals for dominator denom
  data/combine_data.csv                    — combine metrics + draft info

Output:
  data/college/rookie_features.csv         — one row per drafted NFL player,
                                             ready to join with Year-1 PPG
                                             target for rookie model training

Feature families built:
  Volume    — career totals (rush/rec/pass)
  Peak      — best single-season production
  Trajectory — first vs last year usage/PPA (development slope)
  Dominator — market share at team level (yds/TDs as % of team)
  Breakout  — first year of usage ≥ 20% (when they "broke out")
  Context   — seasons played, early declare, transferred
  Recruit   — stars, rating, national rank, HS size
  Combine   — athletic profile (already in combine_data)
  Draft     — round, pick, overall capital score

Usage:
    python build_rookie_features.py
    python build_rookie_features.py --verify Bijan    # inspect a specific player
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLLEGE_DIR = DATA_DIR / "college"


def load_team_stats_all_years():
    """Load team totals across all years — denominators for dominator calcs."""
    frames = []
    for year_dir in sorted(COLLEGE_DIR.iterdir()):
        if not year_dir.is_dir() or year_dir.name == "recruits":
            continue
        path = year_dir / "team_stats.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["season"] = int(year_dir.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def safe_div(num, denom):
    """Divide, returning NaN if denom is 0/NaN."""
    try:
        if pd.isna(denom) or denom == 0:
            return np.nan
        return num / denom
    except Exception:
        return np.nan


def aggregate_player(player_name, career_df, team_stats_df, pos):
    """Compute all career features for one NFL player.

    career_df is the per-season rows for THIS player only, sorted by season.
    team_stats_df is all team-season totals.
    """
    if career_df.empty:
        return {}

    career_df = career_df.sort_values("season").copy()
    seasons = career_df["season"].tolist()

    out = {
        "nfl_player_name": player_name,
        "nfl_pos": pos,
        "college_seasons_played": len(career_df),
        "college_schools": career_df["team"].nunique(),
        "transferred": int(career_df["team"].nunique() > 1),
        "first_college_season": int(min(seasons)),
        "last_college_season": int(max(seasons)),
        "left_for_draft_early": int(len(career_df) < 4),
    }

    # ── Volume (career totals) ──
    for stat in ["rus_car", "rus_yds", "rus_td",
                 "rec_rec", "rec_yds", "rec_td",
                 "pas_att", "pas_yds", "pas_td", "pas_int",
                 "pas_comp"]:
        if stat in career_df.columns:
            total = career_df[stat].fillna(0).sum()
            out[f"career_{stat}"] = float(total) if total else 0.0

    # Efficiency (career rates)
    c = career_df
    out["career_ypc"] = safe_div(out.get("career_rus_yds", 0), out.get("career_rus_car", 0))
    out["career_ypr"] = safe_div(out.get("career_rec_yds", 0), out.get("career_rec_rec", 0))
    out["career_comp_pct"] = safe_div(out.get("career_pas_comp", 0), out.get("career_pas_att", 0))
    out["career_td_int_ratio"] = safe_div(out.get("career_pas_td", 0), max(1, out.get("career_pas_int", 1)))

    # ── Peak single season ──
    out["peak_rus_yds"] = c["rus_yds"].max() if "rus_yds" in c.columns else np.nan
    out["peak_rec_yds"] = c["rec_yds"].max() if "rec_yds" in c.columns else np.nan
    out["peak_pas_yds"] = c["pas_yds"].max() if "pas_yds" in c.columns else np.nan
    out["peak_usage_overall"] = c["usage_overall"].max() if "usage_overall" in c.columns else np.nan
    out["peak_ppa_per_play"] = c["ppa_avg_all"].max() if "ppa_avg_all" in c.columns else np.nan
    out["peak_ppa_total"] = c["ppa_total_all"].max() if "ppa_total_all" in c.columns else np.nan

    # ── Trajectory (slope from first to last season) ──
    if "usage_overall" in c.columns and c["usage_overall"].notna().sum() >= 2:
        u = c.dropna(subset=["usage_overall"])
        out["usage_first_season"] = float(u.iloc[0]["usage_overall"])
        out["usage_last_season"] = float(u.iloc[-1]["usage_overall"])
        out["usage_slope"] = out["usage_last_season"] - out["usage_first_season"]
    else:
        out["usage_first_season"] = np.nan
        out["usage_last_season"] = np.nan
        out["usage_slope"] = np.nan

    if "ppa_total_all" in c.columns and c["ppa_total_all"].notna().sum() >= 2:
        p = c.dropna(subset=["ppa_total_all"])
        out["ppa_first_season"] = float(p.iloc[0]["ppa_total_all"])
        out["ppa_last_season"] = float(p.iloc[-1]["ppa_total_all"])
        out["ppa_slope"] = out["ppa_last_season"] - out["ppa_first_season"]
    else:
        out["ppa_first_season"] = np.nan
        out["ppa_last_season"] = np.nan
        out["ppa_slope"] = np.nan

    # ── Breakout age proxy ──
    # First season with usage >= 20%. Approximates "when they broke out" as
    # season-index (freshman=0, sophomore=1, etc.). Lower = earlier breakout.
    breakout_idx = np.nan
    if "usage_overall" in c.columns:
        c_indexed = c.reset_index(drop=True)
        breakout_rows = c_indexed[c_indexed["usage_overall"] >= 0.20]
        if len(breakout_rows) > 0:
            breakout_idx = int(breakout_rows.index[0])  # 0 = freshman year
    out["breakout_season_index"] = breakout_idx
    out["dominant_seasons"] = int((c.get("usage_overall", pd.Series([])).fillna(0) >= 0.20).sum())

    # ── Dominator rating (market share of team yards/TDs) ──
    # For each season, join to team totals and compute (player_yds / team_yds).
    # Then take peak across seasons.
    if team_stats_df is not None and not team_stats_df.empty:
        peak_yds_share_rec = 0.0
        peak_td_share_rec = 0.0
        peak_yds_share_rush = 0.0
        peak_td_share_rush = 0.0
        peak_dominator_rec = 0.0
        peak_dominator_rush = 0.0

        for _, row in c.iterrows():
            season = row["season"]
            team = row["team"]
            team_row = team_stats_df[
                (team_stats_df["season"] == season) & (team_stats_df["team"] == team)
            ]
            if team_row.empty:
                continue
            t = team_row.iloc[0]

            # Receiving share (relevant for WR/TE and pass-catching RB)
            team_pass_yds = t.get("netPassingYards", 0) or 0
            team_pass_tds = t.get("passingTDs", 0) or 0
            yds_share_rec = safe_div(row.get("rec_yds", 0) or 0, team_pass_yds) or 0
            td_share_rec = safe_div(row.get("rec_td", 0) or 0, team_pass_tds) or 0
            dom_rec = (yds_share_rec + td_share_rec) / 2

            # Rushing share (relevant for RB)
            team_rush_yds = t.get("rushingYards", 0) or 0
            team_rush_tds = t.get("rushingTDs", 0) or 0
            yds_share_rush = safe_div(row.get("rus_yds", 0) or 0, team_rush_yds) or 0
            td_share_rush = safe_div(row.get("rus_td", 0) or 0, team_rush_tds) or 0
            dom_rush = (yds_share_rush + td_share_rush) / 2

            peak_yds_share_rec = max(peak_yds_share_rec, yds_share_rec or 0)
            peak_td_share_rec = max(peak_td_share_rec, td_share_rec or 0)
            peak_yds_share_rush = max(peak_yds_share_rush, yds_share_rush or 0)
            peak_td_share_rush = max(peak_td_share_rush, td_share_rush or 0)
            peak_dominator_rec = max(peak_dominator_rec, dom_rec or 0)
            peak_dominator_rush = max(peak_dominator_rush, dom_rush or 0)

        out["peak_rec_yds_share"] = peak_yds_share_rec
        out["peak_rec_td_share"] = peak_td_share_rec
        out["peak_rush_yds_share"] = peak_yds_share_rush
        out["peak_rush_td_share"] = peak_td_share_rush
        out["peak_dominator_rec"] = peak_dominator_rec
        out["peak_dominator_rush"] = peak_dominator_rush
        # Positional best — use rec dominator for WR/TE, rush for RB, either for QB
        if pos in ("WR", "TE"):
            out["peak_dominator"] = peak_dominator_rec
        elif pos == "RB":
            out["peak_dominator"] = peak_dominator_rush
        else:  # QB
            out["peak_dominator"] = max(peak_dominator_rec, peak_dominator_rush)
    else:
        for k in ["peak_rec_yds_share", "peak_rec_td_share", "peak_rush_yds_share",
                  "peak_rush_td_share", "peak_dominator_rec", "peak_dominator_rush",
                  "peak_dominator"]:
            out[k] = np.nan

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", help="Print detailed features for one player (name substring)")
    args = ap.parse_args()

    print("Loading inputs...")
    career = pd.read_csv(COLLEGE_DIR / "nfl_college_career.csv")
    recruit = pd.read_csv(COLLEGE_DIR / "nfl_recruit_linkage.csv")
    combine = pd.read_csv(DATA_DIR / "combine_data.csv")
    team_stats = load_team_stats_all_years()

    print(f"  college career rows: {len(career):,}")
    print(f"  recruit profiles:    {len(recruit):,}")
    print(f"  team-season totals:  {len(team_stats):,}")

    # Aggregate per NFL player
    print("\nAggregating per-player career features...")
    agg_rows = []
    grouped = career.groupby("nfl_player_name")
    for player_name, group in grouped:
        pos = group["nfl_pos"].iloc[0]
        features = aggregate_player(player_name, group, team_stats, pos)
        features["nfl_draft_year"] = group["nfl_draft_year"].iloc[0]
        agg_rows.append(features)

    agg_df = pd.DataFrame(agg_rows)
    print(f"  → {len(agg_df)} players aggregated")

    # Join recruit profile
    recruit_slim = recruit[[
        "nfl_player_name", "stars", "rating", "national_rank",
        "recruit_height", "recruit_weight", "committed_to", "recruit_year",
    ]].copy()
    recruit_slim.columns = [
        "nfl_player_name", "recruit_stars", "recruit_rating",
        "recruit_national_rank", "recruit_height_in", "recruit_weight_lb",
        "recruit_committed_school", "recruit_class_year",
    ]
    merged = agg_df.merge(recruit_slim, on="nfl_player_name", how="left")
    print(f"  recruit profile joined: {merged['recruit_stars'].notna().sum()} / {len(merged)} have recruit data")

    # Join combine (via name match — ideally by cfb_id/pfr_id but we keep it simple)
    combine_slim = combine[combine["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    combine_slim = combine_slim.rename(columns={"player_name": "nfl_player_name"})
    combine_keep = ["nfl_player_name", "draft_year", "draft_round", "draft_ovr",
                    "ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
    combine_slim = combine_slim[[c for c in combine_keep if c in combine_slim.columns]]
    combine_slim = combine_slim.rename(columns={
        "draft_year": "nfl_draft_year",
        "ht": "combine_height",
        "wt": "combine_weight",
        "forty": "combine_forty",
        "bench": "combine_bench",
        "vertical": "combine_vertical",
        "broad_jump": "combine_broad",
        "cone": "combine_cone",
        "shuttle": "combine_shuttle",
    })
    # Combine has "6-2" format height — convert to inches
    def parse_height(h):
        if pd.isna(h): return np.nan
        m = re.match(r"(\d+)[-'](\d+)", str(h))
        if m: return int(m.group(1)) * 12 + int(m.group(2))
        try: return float(h)
        except: return np.nan
    combine_slim["combine_height"] = combine_slim["combine_height"].apply(parse_height)

    merged = merged.merge(combine_slim, on=["nfl_player_name", "nfl_draft_year"], how="left")
    print(f"  combine joined: {merged['combine_forty'].notna().sum()} / {len(merged)} have combine data")

    # Draft capital score (matching existing dynasty_value.py logic)
    def draft_capital(ovr):
        if pd.isna(ovr): return 0.1
        if ovr <= 10: return 1.0
        if ovr <= 32: return 0.8
        if ovr <= 64: return 0.6
        if ovr <= 100: return 0.4
        if ovr <= 160: return 0.2
        return 0.1
    merged["draft_capital_score"] = merged["draft_ovr"].apply(draft_capital)

    # Save
    out = COLLEGE_DIR / "rookie_features.csv"
    merged.to_csv(out, index=False)
    print(f"\n✅ Saved {len(merged)} rookie feature rows → {out}")
    print(f"   Total columns: {len(merged.columns)}")

    if args.verify:
        target = merged[merged["nfl_player_name"].str.contains(args.verify, case=False, na=False)]
        if len(target) == 0:
            print(f"\nNo player matching '{args.verify}'")
        else:
            for _, row in target.iterrows():
                print(f"\n{'=' * 60}")
                print(f"  {row['nfl_player_name']} ({row['nfl_pos']}, draft {int(row['nfl_draft_year'])})")
                print(f"{'=' * 60}")
                # Group features by family for readability
                for col, val in row.items():
                    if pd.notna(val) and col not in ["nfl_player_name", "nfl_pos"]:
                        if isinstance(val, float):
                            print(f"  {col:30} = {val:.3f}")
                        else:
                            print(f"  {col:30} = {val}")


if __name__ == "__main__":
    main()
