import os
import json
import pandas as pd
import duckdb
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(os.getcwd())
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "open-data/data"
DUCKDB_PATH  = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"

TOP5_LEAGUES = [
    "Premier League",
    "La Liga",
    "Serie A",
    "1. Bundesliga",
    "Ligue 1"
]

TARGET_SEASON = "2015/2016"

print(DATA_DIR)
print(DUCKDB_PATH)

# ── Load match_scores from DuckDB ─────────────────────────────────────────
conn = duckdb.connect(str(DUCKDB_PATH))
match_scores = conn.execute("SELECT * FROM match_scores").df()

print("match_scores columns:", match_scores.columns.tolist())
print(f"Total matches in DB: {len(match_scores)}")

# ── Load matches JSON to get league names + filter to top 5 / 2015-16 ────
match_rows = []

for matches_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(matches_file, encoding="utf-8") as f:
        matches = json.load(f)

    for m in matches:
        competition = m.get("competition", {})
        season      = m.get("season", {})
        match_rows.append({
            "match_id":        m["match_id"],
            "league_name":     competition.get("competition_name"),
            "season_name":     season.get("season_name"),
        })

matches_meta = pd.DataFrame(match_rows)

# Filter to top 5 leagues + 2015/2016 season
filtered_meta = matches_meta[
    (matches_meta["league_name"].isin(TOP5_LEAGUES)) &
    (matches_meta["season_name"] == TARGET_SEASON)
]

# Keep only match_ids that also exist in match_scores (inner join)
valid_match_ids = set(filtered_meta["match_id"]) & set(match_scores["match_id"])
MATCHES_TO_PROCESS = sorted(valid_match_ids)

print(f"\nMatches found in JSON meta: {len(filtered_meta)}")
print(f"Matches found in DuckDB:    {len(match_scores)}")
print(f"Matches to process:         {len(MATCHES_TO_PROCESS)}")


# ── Helper ────────────────────────────────────────────────────────────────
def add_match_score(shots_df):
    shots_df = shots_df.sort_values(["minute", "second"]).copy()

    score = {}
    score_at_shot = []

    for _, row in shots_df.iterrows():
        if row["goal"] == 1:
            tid = row["team_id"]
            score[tid] = score.get(tid, 0) + 1
        score_at_shot.append(dict(score))

    team_ids = shots_df["team_id"].unique()
    for tid in team_ids:
        shots_df[f"score_team_{tid}"] = [s.get(tid, 0) for s in score_at_shot]

    return shots_df, team_ids


# ── Main loop ─────────────────────────────────────────────────────────────
all_rows = []

for i, match_id in enumerate(MATCHES_TO_PROCESS):
    try:
        with open(DATA_DIR / "events" / f"{match_id}.json", encoding="utf-8") as f:
            events = json.load(f)

        events_df = pd.DataFrame(events)
        shots_df  = events_df[events_df["shot"].notna()].copy()

        if shots_df.empty:
            continue

        shots_df["shot_statsbomb_xg"] = shots_df["shot"].apply(
            lambda x: x.get("statsbomb_xg") if isinstance(x, dict) else None
        )
        shots_df["shot_outcome_name"] = shots_df["shot"].apply(
            lambda x: x.get("outcome", {}).get("name") if isinstance(x, dict) else None
        )
        shots_df["goal"] = (shots_df["shot_outcome_name"] == "Goal").astype(int)

        shots_df["scorer_id"] = shots_df["player"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else None
        )
        shots_df["team_id"] = shots_df["team"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else None
        )

        if "second" not in shots_df.columns:
            shots_df["second"] = shots_df["timestamp"].apply(
                lambda x: int(x.split(":")[1]) if isinstance(x, str) else 0
            )

        match_row = match_scores[match_scores["match_id"] == match_id].iloc[0]

        shots_df, team_ids = add_match_score(shots_df)

        # Identify home/away team by matching final score to DB values
        if len(team_ids) >= 2:
            tid_a, tid_b       = team_ids[0], team_ids[1]
            final_a            = shots_df[f"score_team_{tid_a}"].iloc[-1]
            home_score_final   = match_row["home_score"]
            home_tid = tid_a if final_a == home_score_final else tid_b
            away_tid = tid_b if home_tid == tid_a else tid_a
        else:
            # Only one team took shots
            home_tid = team_ids[0]
            away_tid = None

        shots_df = shots_df.rename(columns={
            f"score_team_{home_tid}": "home_score",
            **({f"score_team_{away_tid}": "away_score"} if away_tid else {})
        })

        if away_tid is None:
            shots_df["away_score"] = 0

        # Drop any leftover score_team_ columns
        for tid in team_ids:
            col = f"score_team_{tid}"
            if col in shots_df.columns:
                shots_df.drop(columns=[col], inplace=True)

        shots_df["match_id"]  = match_id
        shots_df["league_id"] = match_row["league_id"]

        all_rows.append(shots_df[[
            "match_id", "league_id", "minute", "second",
            "shot_statsbomb_xg", "goal", "scorer_id",
            "home_score", "away_score"
        ]])

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(MATCHES_TO_PROCESS)} matches...")

    except Exception as e:
        print(f"ERROR on match_id {match_id}: {e}")
        continue

final_df = pd.concat(all_rows, ignore_index=True)
conn.execute("CREATE OR REPLACE TABLE temporal_xg AS SELECT * FROM final_df")
conn.close()
print(f"Written {len(final_df)} rows to 'temporal_xg' table in DuckDB")

