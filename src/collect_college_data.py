"""
Pull college football data from CollegeFootballData API for the rookie model.

This data will be used ONLY by a dedicated rookie model, which is blended
with the main model via career-games-decay weight (exponential). College
features' influence shrinks as NFL sample accumulates.

Requires CFBD_API_KEY in .env file. Signup free at collegefootballdata.com/key.

Endpoints used (all batched by year — single call returns every FBS player):
  /stats/player/season     — basic box-score stats per (player, category)
  /player/usage            — usage share by down/situation
  /ppa/players/season      — Predicted Points Added (college's EPA)
  /recruiting/players      — HS recruit pedigree (stars, rating, rank)

API budget: ~7 calls per year × 14 years = ~100 calls (one-time setup).
Monthly refresh during CFB season: ~5-10 calls.
Free tier: 200/month. We're well within.

Usage:
    python collect_college_data.py --year 2022           # single year test
    python collect_college_data.py --years 2011-2024     # full historical
    python collect_college_data.py --match-coverage      # how many of our rookies linked?
"""
import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

import pandas as pd

BASE = "https://api.collegefootballdata.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLLEGE_DIR = DATA_DIR / "college"
COLLEGE_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key():
    """Load CFBD_API_KEY from .env file in project root."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        raise RuntimeError(f"No .env file at {env_path}. Add CFBD_API_KEY=...")
    for line in env_path.read_text().splitlines():
        if line.startswith("CFBD_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("CFBD_API_KEY not found in .env")


API_KEY = load_api_key()
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "User-Agent": "TrueValueDynasty/1.0",
}


def api_get(path, params=None, retries=2):
    """Thin wrapper around CFBD API. Returns parsed JSON or None on failure."""
    url = BASE + path
    if params:
        url += "?" + "&".join(
            f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items() if v is not None
        )
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            print(f"  HTTP {e.code} on {path}: {body}")
            if e.code == 429:  # rate-limited
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"  Error fetching {path}: {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(2)
                continue
            return None
    return None


# ─────────────────────────────────────────────────────────────────────
# Per-endpoint pullers — each returns a tidy DataFrame
# ─────────────────────────────────────────────────────────────────────

def pull_player_season_stats(year, category):
    """Basic box-score stats. CFBD returns long format — we pivot to wide."""
    rows = api_get("/stats/player/season", {"year": year, "category": category})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Long → wide pivot: one row per (player, team, season)
    # Key columns: playerId, player, team, conference, category, statType, stat
    wide = df.pivot_table(
        index=["playerId", "player", "team", "conference", "category"],
        columns="statType",
        values="stat",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide["season"] = year
    # Prefix stat columns with category for clarity when we merge across categories
    cat_prefix = category[:3]  # "rus", "rec", "pas"
    rename = {c: f"{cat_prefix}_{c.lower()}" for c in wide.columns
              if c not in ["playerId", "player", "team", "conference", "category", "season"]}
    wide = wide.rename(columns=rename)
    return wide


def pull_player_usage(year):
    """Usage % by situation — the 'market share done right' metric."""
    rows = api_get("/player/usage", {"year": year})
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        usage = r.get("usage") or {}
        records.append({
            "season": year,
            "playerId": r.get("id"),
            "player": r.get("name"),
            "team": r.get("team"),
            "conference": r.get("conference"),
            "position": r.get("position"),
            "usage_overall": usage.get("overall"),
            "usage_pass": usage.get("pass"),
            "usage_rush": usage.get("rush"),
            "usage_first_down": usage.get("firstDown"),
            "usage_second_down": usage.get("secondDown"),
            "usage_third_down": usage.get("thirdDown"),
            "usage_standard_downs": usage.get("standardDowns"),
            "usage_passing_downs": usage.get("passingDowns"),
        })
    return pd.DataFrame(records)


def pull_player_ppa(year):
    """PPA — college football's EPA. Efficiency above context per play."""
    rows = api_get("/ppa/players/season", {"year": year})
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        avg = r.get("averagePPA") or {}
        tot = r.get("totalPPA") or {}
        records.append({
            "season": year,
            "playerId": r.get("id"),
            "player": r.get("name"),
            "team": r.get("team"),
            "conference": r.get("conference"),
            "position": r.get("position"),
            "ppa_avg_all": avg.get("all"),
            "ppa_avg_pass": avg.get("pass"),
            "ppa_avg_rush": avg.get("rush"),
            "ppa_total_all": tot.get("all"),
            "ppa_total_pass": tot.get("pass"),
            "ppa_total_rush": tot.get("rush"),
        })
    return pd.DataFrame(records)


def pull_recruits(year):
    """HS recruit rankings — pedigree signal."""
    rows = api_get("/recruiting/players", {"year": year, "classification": "HighSchool"})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "recruit_year": year,
        "recruit_id": r.get("id"),
        "athlete_id": r.get("athleteId"),
        "name": r.get("name"),
        "committed_to": r.get("committedTo"),
        "position": r.get("position"),
        "stars": r.get("stars"),
        "rating": r.get("rating"),
        "national_rank": r.get("ranking"),
        "recruit_height": r.get("height"),
        "recruit_weight": r.get("weight"),
    } for r in rows])


def pull_team_stats(year):
    """Team-season totals — denominators for dominator rating."""
    rows = api_get("/stats/season", {"year": year})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    wide = df.pivot_table(
        index=["team", "conference"],
        columns="statName",
        values="statValue",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide["season"] = year
    return wide


# ─────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────

def pull_year(year, verbose=True):
    """Pull all college data for one year. Returns a dict of DataFrames."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Pulling college data for season {year}")
        print(f"{'=' * 60}")

    results = {}

    for cat in ["rushing", "receiving", "passing"]:
        if verbose:
            print(f"  /stats/player/season category={cat}...")
        df = pull_player_season_stats(year, cat)
        results[f"stats_{cat}"] = df
        if verbose:
            print(f"    {len(df)} player-season rows")

    if verbose:
        print(f"  /player/usage...")
    results["usage"] = pull_player_usage(year)
    if verbose:
        print(f"    {len(results['usage'])} player-season rows")

    if verbose:
        print(f"  /ppa/players/season...")
    results["ppa"] = pull_player_ppa(year)
    if verbose:
        print(f"    {len(results['ppa'])} player-season rows")

    if verbose:
        print(f"  /stats/season (team totals)...")
    results["team_stats"] = pull_team_stats(year)
    if verbose:
        print(f"    {len(results['team_stats'])} team rows")

    return results


def save_year(year, results):
    """Save per-year parquet files. Compact + efficient."""
    year_dir = COLLEGE_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if not df.empty:
            path = year_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)


def check_match_coverage():
    """
    Sanity check: how many of our combine/rookie players can we link to CFBD?
    Uses fuzzy name+school matching as a proxy for true ID linkage (CFBD
    doesn't share cfb_id with sports-reference's cfb_id).
    """
    combine = pd.read_csv(DATA_DIR / "combine_data.csv")
    # Keep skill positions only
    skill = combine[combine["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    skill["name_lower"] = skill["player_name"].str.lower().str.strip()

    # Load most recent year of CFBD player data we have
    year_dirs = sorted([p for p in COLLEGE_DIR.iterdir() if p.is_dir()])
    if not year_dirs:
        print("No college data pulled yet — run with --year YYYY first")
        return
    latest = year_dirs[-1]
    year = int(latest.name)
    print(f"Checking coverage using {year} CFBD data...")

    usage_path = latest / "usage.parquet"
    if not usage_path.exists():
        print(f"  No usage data for {year}")
        return
    cfbd = pd.read_parquet(usage_path)
    cfbd["name_lower"] = cfbd["player"].str.lower().str.strip()

    # Which of our combine players (drafted in year+1) show up in CFBD year?
    draft_year = year + 1
    our_rookies = skill[skill["draft_year"] == draft_year]
    matched = our_rookies.merge(cfbd, left_on="name_lower", right_on="name_lower", how="inner")

    print(f"  Our combine players drafted in {draft_year}: {len(our_rookies)}")
    print(f"  Matched in CFBD {year} data: {len(matched)}")
    print(f"  Match rate: {len(matched) / max(1, len(our_rookies)):.1%}")
    if len(matched) < len(our_rookies):
        missing = our_rookies[~our_rookies["name_lower"].isin(matched["name_lower"])]
        print(f"  Sample unmatched ({len(missing)}):")
        for _, row in missing.head(8).iterrows():
            print(f"    {row['player_name']} ({row['pos']}, {row['school']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="Pull a single year")
    ap.add_argument("--years", help="Range 'YYYY-YYYY' to pull")
    ap.add_argument("--match-coverage", action="store_true",
                    help="Report % of our rookies linkable to CFBD")
    args = ap.parse_args()

    if args.match_coverage:
        check_match_coverage()
        return

    if args.year:
        years = [args.year]
    elif args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        print("Pass --year YYYY or --years YYYY-YYYY")
        sys.exit(1)

    for y in years:
        results = pull_year(y)
        save_year(y, results)
        print(f"  ✅ Saved {y} → {COLLEGE_DIR / str(y)}")
        time.sleep(1)  # be gentle with the API


if __name__ == "__main__":
    main()
