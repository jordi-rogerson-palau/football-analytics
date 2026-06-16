import os
import json
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(os.getcwd())
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "open-data/data"
DUCKDB_PATH  = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"

TOP5_LEAGUES = [
    "Premier League",
    "La Liga",
    "Serie A",
    "1. Bundesliga",
    "Ligue 1",
]
TARGET_SEASON = "2015/2016"

# ── Build match metadata ───────────────────────────────────────────────────
match_rows = []
for matches_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(matches_file, encoding="utf-8") as f:
        matches = json.load(f)
    for m in matches:
        competition = m.get("competition", {})
        season      = m.get("season", {})
        if (competition.get("competition_name") in TOP5_LEAGUES and
                season.get("season_name") == TARGET_SEASON):
            match_rows.append({
                "match_id":    m["match_id"],
                "league_name": competition.get("competition_name"),
                "league_id":   competition.get("competition_id"),
            })

valid_match_ids = sorted(set(r["match_id"] for r in match_rows))
matches_meta    = pd.DataFrame(match_rows).drop_duplicates("match_id")

print(f"Processing {len(valid_match_ids)} matches across top 5 leagues {TARGET_SEASON}")

# ── Per-match event parsing ────────────────────────────────────────────────
all_rows = []

for i, match_id in enumerate(valid_match_ids):
    try:
        events_path = DATA_DIR / "events" / f"{match_id}.json"
        with open(events_path, encoding="utf-8") as f:
            raw = json.load(f)

        events    = pd.json_normalize(raw)
        type_col  = events.get("type.name", pd.Series("", index=events.index)).fillna("")

        def make_row(row, is_goal=0, is_assist=0, is_key_pass=0, is_pass_success=0, xg=0):
            return {
                "player_id":        row.get("player.id"),
                "player_name":      row.get("player.name"),
                "team_id":          row.get("team.id"),
                "position":         row.get("position.name"),
                "is_goal":          is_goal,
                "is_assist":        is_assist,
                "is_key_pass":      is_key_pass,
                "xg":               xg,
                "is_pass_success":  is_pass_success,
            }

        # ── Goals ──────────────────────────────────────────────────────────
        shots        = events[type_col == "Shot"].copy()
        shot_outcome = shots.get("shot.outcome.name", pd.Series("", index=shots.index)).fillna("")
        goals        = shots[shot_outcome == "Goal"]

        for _, row in goals.iterrows():
            all_rows.append(make_row(
                row,
                is_goal=1,
                xg=row.get("shot.statsbomb_xg", 0) or 0,
            ))

        # ── xG for non-goal shots (still counts toward cumulative xG) ──────
        non_goals = shots[shot_outcome != "Goal"]
        for _, row in non_goals.iterrows():
            all_rows.append(make_row(
                row,
                xg=row.get("shot.statsbomb_xg", 0) or 0,
            ))

        # ── Assists & key passes ───────────────────────────────────────────
        if "shot.key_pass_id" in shots.columns and "id" in events.columns:

            # Assists: key pass leading to a goal
            goal_key_pass_ids = shots.loc[
                shot_outcome == "Goal", "shot.key_pass_id"
            ].dropna().unique()

            for _, row in events[events["id"].isin(goal_key_pass_ids)].iterrows():
                all_rows.append(make_row(row, is_assist=1))

            # Key passes: key pass NOT leading to a goal
            non_goal_key_pass_ids = shots.loc[
                shot_outcome != "Goal", "shot.key_pass_id"
            ].dropna().unique()

            for _, row in events[events["id"].isin(non_goal_key_pass_ids)].iterrows():
                all_rows.append(make_row(row, is_key_pass=1))

        # ── Successful passes ──────────────────────────────────────────────
        passes       = events[type_col == "Pass"].copy()
        pass_outcome = passes.get("pass.outcome.name", pd.Series("", index=passes.index)).fillna("")

        for _, row in passes[pass_outcome == ""].iterrows():
            all_rows.append(make_row(row, is_pass_success=1))

    except Exception as e:
        print(f"ERROR on match_id {match_id}: {e}")
        continue

    if (i + 1) % 50 == 0 or (i + 1) == len(valid_match_ids):
        print(f"  Processed {i + 1}/{len(valid_match_ids)} matches...")

# ── Aggregate: one row per (player_id, team_id) ───────────────────────────
df = pd.DataFrame(all_rows).dropna(subset=["player_id"])

first_position = (
    df.dropna(subset=["position"])
    .groupby("player_id")["position"]
    .first()
    .rename("position")
)

first_name = (
    df.dropna(subset=["player_name"])
    .groupby("player_id")["player_name"]
    .first()
    .rename("player_name")
)

stats = (
    df.groupby(["player_id", "team_id"])
    .agg(
        goals            =("is_goal",         "sum"),
        assists          =("is_assist",        "sum"),
        key_passes       =("is_key_pass",      "sum"),
        xg               =("xg",               "sum"),
        successful_passes=("is_pass_success",  "sum"),
    )
    .reset_index()
)

result = (
    stats
    .join(first_name,     on="player_id")
    .join(first_position, on="player_id")
)

result = result[[
    "player_id", "player_name", "team_id", "position",
    "goals", "assists", "key_passes", "xg", "successful_passes"
]].sort_values(["player_id", "team_id"]).reset_index(drop=True)

# ── Write to DuckDB ────────────────────────────────────────────────────────
conn = duckdb.connect(str(DUCKDB_PATH))
conn.execute("CREATE OR REPLACE TABLE players AS SELECT * FROM result")
conn.close()

print(f"\nWritten {len(result)} rows to 'players' table in DuckDB")

# ── Sanity checks ──────────────────────────────────────────────────────────
print(f"\nTotal rows (player × team):  {len(result)}")
print(f"Unique players:              {result['player_id'].nunique()}")
print(f"Total goals:                 {result['goals'].sum()}")
print(f"Total assists:               {result['assists'].sum()}")
print(f"Total key passes:            {result['key_passes'].sum()}")
print(f"Total xG:                    {result['xg'].sum():.2f}")
print(f"Total successful passes:     {result['successful_passes'].sum()}")

for metric, label in [
    ("goals",      "GOALS"),
    ("assists",    "ASSISTS"),
    ("xg",         "xG"),
    ("key_passes", "KEY PASSES"),
]:
    print(f"\n=== TOP 10 by {label} ===")
    top10 = (
        result[["player_name", "team_id", metric]]
        .sort_values(metric, ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10.index += 1
    fmt = ".2f" if metric == "xg" else ".0f"
    top10[metric] = top10[metric].map(lambda x: format(x, fmt))
    print(top10.to_string())