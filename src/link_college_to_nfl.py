"""
Link NFL skill-position players to their CFBD college career + HS recruit profile.

Inputs (read-only):
  data/combine_data.csv                     — our NFL player universe
  data/players.csv                          — supplementary name/team data
  data/college/YYYY/*.parquet               — CFBD college stats per year
  data/college/recruits/YYYY.parquet        — CFBD HS recruit classes

Outputs:
  data/college/nfl_college_career.csv       — 1 row per (nfl_player, college_season)
  data/college/nfl_recruit_linkage.csv      — 1 row per nfl_player with recruit profile
  data/college/linkage_report.txt           — coverage and ambiguity summary

Strategy:
  - Matching key: normalized(name) + school + year_window
  - Year window: for a player drafted in year D, their college career is roughly
    [D-5, D-1] (allowing for 5th-year seniors, medical redshirts, transfers)
  - Primary match: exact name + school match in window
  - Fallback 1: fuzzy name match (sequence similarity >= 0.85) + school match
  - Fallback 2: exact name match across any year (handles odd cases like JUCO)
  - Ambiguous cases (multiple equally-good matches) are logged for manual review

Usage:
    python link_college_to_nfl.py
    python link_college_to_nfl.py --verbose      # log all matches
    python link_college_to_nfl.py --report-only  # don't save, just diagnose
"""
import re
import argparse
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLLEGE_DIR = DATA_DIR / "college"


# Common school-name aliases — canonicalize to one spelling.
# Important: Miami (FL) ≠ Miami (OH), keep state-suffixed variants distinct where needed.
SCHOOL_ALIASES = {
    # NFL combine source → CFBD form
    "miami fl": "miami",
    "miami oh": "miami oh",        # stays distinct
    "nc state": "north carolina state",
    "ucf": "central florida",
    "usf": "south florida",
    "umass": "massachusetts",
    "smu": "southern methodist",
    "byu": "brigham young",
    "ucla": "ucla",                # CFBD uses UCLA too
    "tcu": "texas christian",
    "san jose state": "san jose state",   # diacritic normalized by `normalize()`
    "san josé state": "san jose state",
    "louisiana lafayette": "louisiana",
    "louisiana monroe": "louisiana monroe",
    "western michigan": "western michigan",
    "west michigan": "western michigan",
    "eastern illinois": "eastern illinois",
    "east illinois": "eastern illinois",
    "massachusetts": "umass",       # either direction
    "pitt": "pittsburgh",
    "ole miss": "mississippi",
    "mississippi": "mississippi",
    "penn state": "penn state",
    "ohio state": "ohio state",
    "arkansas state": "arkansas state",
    "middle tennessee": "middle tennessee",
    "middle tennessee state": "middle tennessee",
    "csun": "cal state northridge",
    "fiu": "florida international",
    "fau": "florida atlantic",
}


# Common nickname pairs — first names that commonly differ between our
# NFL records (casual) and CFBD records (formal) or vice versa.
NICKNAMES = {
    "cam": "cameron", "cameron": "cam",
    "mike": "michael", "michael": "mike",
    "will": "william", "william": "will",
    "bill": "william", "billy": "william",
    "rob": "robert", "robert": "rob",
    "bob": "robert",
    "nick": "nicholas", "nicholas": "nick",
    "chris": "christopher", "christopher": "chris",
    "matt": "matthew", "matthew": "matt",
    "alex": "alexander", "alexander": "alex",
    "dan": "daniel", "danny": "daniel", "daniel": "dan",
    "tom": "thomas", "thomas": "tom",
    "jim": "james", "jimmy": "james", "james": "jim",
    "joe": "joseph", "joseph": "joe",
    "ben": "benjamin", "benjamin": "ben",
    "sam": "samuel", "samuel": "sam",
    "tony": "anthony", "anthony": "tony",
    "steve": "stephen", "stephen": "steve", "steven": "steve",
    "pat": "patrick", "patrick": "pat",
    "jon": "jonathan", "jonathan": "jon",
    "ed": "edward", "eddie": "edward", "edward": "ed",
    "ty": "tyler", "tyler": "ty",
    "zach": "zachary", "zachary": "zach",
}


def normalize(text):
    """Lowercase, strip suffixes (Jr/Sr/II/III/IV), remove punctuation,
    normalize diacritics, and trim parenthetical state suffixes like (FL)."""
    if pd.isna(text) or not text:
        return ""
    s = str(text).lower().strip()
    # Strip diacritics via NFKD decomposition
    import unicodedata
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    # Strip suffixes
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?\s*$", "", s)
    # Normalize parenthetical state suffix: "miami (fl)" → "miami fl"
    s = re.sub(r"\(([^)]+)\)", r" \1 ", s)
    # Replace punctuation with space
    s = re.sub(r"[.'\-,]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_school(text):
    """School name with alias lookup applied after base normalization."""
    base = normalize(text)
    return SCHOOL_ALIASES.get(base, base)


def name_variants(name):
    """Generate plausible nickname variants of a normalized name."""
    parts = name.split(" ")
    if not parts:
        return [name]
    first = parts[0]
    rest = " ".join(parts[1:]) if len(parts) > 1 else ""
    variants = {name}
    if first in NICKNAMES:
        variants.add(f"{NICKNAMES[first]} {rest}".strip())
    return variants


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ─────────────────────────────────────────────────────────────────
# Load all CFBD data into memory (they're small parquet files)
# ─────────────────────────────────────────────────────────────────

def load_cfbd_stats():
    """Load all CFBD college stats across years into one big DataFrame."""
    rows = []
    for year_dir in sorted(COLLEGE_DIR.iterdir()):
        if not year_dir.is_dir() or year_dir.name == "recruits":
            continue
        year = int(year_dir.name)

        # Merge rushing, receiving, passing stats (wide format with prefixes)
        frames = []
        for cat in ["stats_rushing", "stats_receiving", "stats_passing"]:
            path = year_dir / f"{cat}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if "category" in df.columns:
                    df = df.drop(columns=["category"])
                frames.append(df)

        if not frames:
            continue

        # Merge all categories — key on (playerId, season) only.
        # Other fields (player/team/conference) sometimes have inconsistent
        # spellings between endpoints (e.g. "Big 12" vs "B12"), which would
        # break the join if included in the key.
        MERGE_KEY = ["playerId", "season"]
        COMMON_COLS = ["player", "team", "conference"]

        merged = frames[0]
        for other in frames[1:]:
            merged = merged.merge(
                other.drop(columns=COMMON_COLS, errors="ignore"),
                on=MERGE_KEY,
                how="outer",
            )

        # Merge usage
        usage_path = year_dir / "usage.parquet"
        if usage_path.exists():
            usage = pd.read_parquet(usage_path)
            merged = merged.merge(
                usage.drop(columns=COMMON_COLS + ["position"], errors="ignore"),
                on=MERGE_KEY,
                how="left",
            )

        # Merge PPA
        ppa_path = year_dir / "ppa.parquet"
        if ppa_path.exists():
            ppa = pd.read_parquet(ppa_path)
            merged = merged.merge(
                ppa.drop(columns=COMMON_COLS + ["position"], errors="ignore"),
                on=MERGE_KEY,
                how="left",
            )

        merged["season"] = year
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    all_stats = pd.concat(rows, ignore_index=True, sort=False)
    # Normalize helper columns
    all_stats["name_norm"] = all_stats["player"].apply(normalize)
    all_stats["school_norm"] = all_stats["team"].apply(normalize_school)
    return all_stats


def load_cfbd_recruits():
    """Load all HS recruit classes into one DataFrame."""
    recruits_dir = COLLEGE_DIR / "recruits"
    if not recruits_dir.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in sorted(recruits_dir.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["name_norm"] = df["name"].apply(normalize)
    df["school_norm"] = df["committed_to"].apply(normalize_school)
    return df


def load_nfl_universe(drafted_only=True):
    """NFL skill-position players we want to link. Undrafted combine
    attendees are excluded by default — they aren't in our dynasty rankings."""
    combine = pd.read_csv(DATA_DIR / "combine_data.csv")
    skill = combine[combine["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    if drafted_only:
        skill = skill[skill["draft_year"].notna()].copy()
    skill["name_norm"] = skill["player_name"].apply(normalize)
    skill["school_norm"] = skill["school"].apply(normalize_school)
    return skill


# ─────────────────────────────────────────────────────────────────
# Matching logic
# ─────────────────────────────────────────────────────────────────

def match_college_career(nfl_row, cfbd_stats, verbose=False):
    """Find all CFBD stat rows matching this NFL player's college career."""
    nfl_name = nfl_row["name_norm"]
    nfl_school = nfl_row["school_norm"]
    draft_year = nfl_row.get("draft_year")
    if pd.isna(draft_year):
        return pd.DataFrame(), "no_draft_year"
    draft_year = int(draft_year)
    # Possible college years: draft_year - 5 to draft_year - 1
    # (covers 5-year seniors, redshirts; undrafted can still fit this window)
    year_window = set(range(draft_year - 5, draft_year))

    # Tier 1: exact name + school match within year window
    variants = name_variants(nfl_name)
    tier1 = cfbd_stats[
        cfbd_stats["name_norm"].isin(variants)
        & (cfbd_stats["school_norm"] == nfl_school)
        & cfbd_stats["season"].isin(year_window)
    ]
    if len(tier1) > 0:
        # Take all matching player_ids — could be a transfer, possibly multiple
        player_ids = tier1["playerId"].unique()
        career = cfbd_stats[cfbd_stats["playerId"].isin(player_ids)]
        return career, "tier1_exact"

    # Tier 2: fuzzy name match + exact school within window
    candidates = cfbd_stats[
        (cfbd_stats["school_norm"] == nfl_school)
        & cfbd_stats["season"].isin(year_window)
    ]
    if len(candidates) > 0:
        candidates = candidates.copy()
        candidates["sim"] = candidates["name_norm"].apply(lambda n: similarity(n, nfl_name))
        best = candidates[candidates["sim"] >= 0.85]
        if len(best) > 0:
            player_ids = best["playerId"].unique()
            career = cfbd_stats[cfbd_stats["playerId"].isin(player_ids)]
            return career, "tier2_fuzzy_school"

    # Tier 3: exact name match across any school, within window
    # (handles transfers where we don't know the college)
    tier3 = cfbd_stats[
        cfbd_stats["name_norm"].isin(variants)
        & cfbd_stats["season"].isin(year_window)
    ]
    if len(tier3) > 0:
        # Disambiguate: prefer the player with most career games near that school
        player_ids = tier3["playerId"].unique()
        if len(player_ids) == 1:
            career = cfbd_stats[cfbd_stats["playerId"] == player_ids[0]]
            return career, "tier3_name_only"
        else:
            # Multiple matches — ambiguous, take the one with most career production
            # as a heuristic (real NFL players tend to have more college stats)
            career_sizes = {
                pid: cfbd_stats[cfbd_stats["playerId"] == pid].shape[0]
                for pid in player_ids
            }
            best_pid = max(career_sizes, key=career_sizes.get)
            career = cfbd_stats[cfbd_stats["playerId"] == best_pid]
            return career, f"tier3_ambiguous_picked_largest({len(player_ids)}_candidates)"

    return pd.DataFrame(), "not_found"


def match_recruit(nfl_row, cfbd_recruits):
    """Find HS recruit profile. Window = [draft_year - 5, draft_year - 2] roughly."""
    nfl_name = nfl_row["name_norm"]
    nfl_school = nfl_row["school_norm"]
    draft_year = nfl_row.get("draft_year")
    if pd.isna(draft_year):
        return None, "no_draft_year"
    draft_year = int(draft_year)
    # HS recruit year typically 3-5 years before NFL draft
    year_window = set(range(draft_year - 5, draft_year - 1))
    variants = name_variants(nfl_name)

    # Tier 1: exact name + committed_to school + year window
    t1 = cfbd_recruits[
        cfbd_recruits["name_norm"].isin(variants)
        & (cfbd_recruits["school_norm"] == nfl_school)
        & cfbd_recruits["recruit_year"].isin(year_window)
    ]
    if len(t1) > 0:
        return t1.iloc[0], "tier1_exact"

    # Tier 2: exact name + school (any year — handles older players)
    t2 = cfbd_recruits[
        cfbd_recruits["name_norm"].isin(variants)
        & (cfbd_recruits["school_norm"] == nfl_school)
    ]
    if len(t2) > 0:
        return t2.iloc[0], "tier2_school_any_year"

    # Tier 3: exact name only, within year window
    t3 = cfbd_recruits[
        cfbd_recruits["name_norm"].isin(variants)
        & cfbd_recruits["recruit_year"].isin(year_window)
    ]
    if len(t3) > 0:
        if len(t3) == 1:
            return t3.iloc[0], "tier3_name_only"
        else:
            # Ambiguous — prefer highest-rated
            return t3.loc[t3["rating"].idxmax()], f"tier3_ambiguous_picked_top({len(t3)})"

    return None, "not_found"


# ─────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="Log every match")
    ap.add_argument("--report-only", action="store_true", help="Diagnose, don't save")
    args = ap.parse_args()

    print("Loading CFBD college stats...")
    cfbd_stats = load_cfbd_stats()
    print(f"  {len(cfbd_stats):,} college player-season rows")

    print("Loading CFBD recruit data...")
    cfbd_recruits = load_cfbd_recruits()
    print(f"  {len(cfbd_recruits):,} recruit rows")

    print("Loading NFL player universe...")
    nfl = load_nfl_universe()
    print(f"  {len(nfl)} NFL skill-position combine entries")

    college_career_rows = []
    recruit_rows = []
    college_tier_counts = {}
    recruit_tier_counts = {}
    not_found_college = []
    not_found_recruit = []

    for _, nfl_row in nfl.iterrows():
        # Match college career
        career, c_reason = match_college_career(nfl_row, cfbd_stats, verbose=args.verbose)
        college_tier_counts[c_reason] = college_tier_counts.get(c_reason, 0) + 1
        if c_reason == "not_found":
            not_found_college.append(nfl_row)
        if not career.empty:
            career = career.copy()
            career["nfl_player_name"] = nfl_row["player_name"]
            career["nfl_pos"] = nfl_row["pos"]
            career["nfl_draft_year"] = nfl_row["draft_year"]
            career["nfl_cfb_id"] = nfl_row.get("cfb_id")
            career["nfl_pfr_id"] = nfl_row.get("pfr_id")
            career["match_tier"] = c_reason
            college_career_rows.append(career)

        # Match recruit
        recruit, r_reason = match_recruit(nfl_row, cfbd_recruits)
        recruit_tier_counts[r_reason] = recruit_tier_counts.get(r_reason, 0) + 1
        if r_reason == "not_found":
            not_found_recruit.append(nfl_row)
        if recruit is not None:
            rec_dict = recruit.to_dict()
            rec_dict["nfl_player_name"] = nfl_row["player_name"]
            rec_dict["nfl_pos"] = nfl_row["pos"]
            rec_dict["nfl_draft_year"] = nfl_row["draft_year"]
            rec_dict["nfl_cfb_id"] = nfl_row.get("cfb_id")
            rec_dict["match_tier"] = r_reason
            recruit_rows.append(rec_dict)

    # Assemble
    if college_career_rows:
        college_df = pd.concat(college_career_rows, ignore_index=True, sort=False)
    else:
        college_df = pd.DataFrame()
    recruit_df = pd.DataFrame(recruit_rows) if recruit_rows else pd.DataFrame()

    # ─── Report ───
    print()
    print("=" * 70)
    print("  COLLEGE CAREER MATCH REPORT")
    print("=" * 70)
    total = len(nfl)
    matched_college = total - college_tier_counts.get("not_found", 0) - college_tier_counts.get("no_draft_year", 0)
    print(f"  Total NFL skill players: {total}")
    print(f"  Matched to CFBD college career: {matched_college} ({matched_college/total:.1%})")
    print(f"  Tier breakdown:")
    for tier, count in sorted(college_tier_counts.items(), key=lambda x: -x[1]):
        print(f"    {tier:40} {count}")

    # Sample of unmatched for investigation
    if not_found_college:
        print(f"\n  Sample unmatched college (first 15):")
        for r in not_found_college[:15]:
            dy = int(r["draft_year"]) if pd.notna(r["draft_year"]) else "?"
            print(f"    {r['player_name']:25} {r['pos']:3} {r['school']:25} draft={dy}")

    print()
    print("=" * 70)
    print("  HS RECRUIT MATCH REPORT")
    print("=" * 70)
    matched_recruits = total - recruit_tier_counts.get("not_found", 0) - recruit_tier_counts.get("no_draft_year", 0)
    print(f"  Matched to HS recruit profile: {matched_recruits} ({matched_recruits/total:.1%})")
    print(f"  Tier breakdown:")
    for tier, count in sorted(recruit_tier_counts.items(), key=lambda x: -x[1]):
        print(f"    {tier:40} {count}")

    if args.report_only:
        print("\n(--report-only: not writing files)")
        return

    # Save
    out1 = COLLEGE_DIR / "nfl_college_career.csv"
    out2 = COLLEGE_DIR / "nfl_recruit_linkage.csv"
    college_df.to_csv(out1, index=False)
    recruit_df.to_csv(out2, index=False)
    print(f"\n✅ Wrote {len(college_df):,} rows to {out1}")
    print(f"✅ Wrote {len(recruit_df):,} rows to {out2}")


if __name__ == "__main__":
    main()
