import pandas as pd
import duckdb
import json
from pathlib import Path

# -----------------------------
# Resolve project root directory
# -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "open-data/data"
DUCKDB_PATH = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"

# -----------------------------
# Configuration
# -----------------------------

TOP5_LEAGUES = [
    "Premier League",
    "La Liga",
    "Serie A",
    "1. Bundesliga",
    "Ligue 1"
]

TARGET_SEASON = "2015/2016"

# -----------------------------
# Load competitions metadata
# -----------------------------

with open(DATA_DIR / "competitions.json", "r", encoding="utf-8") as f:
    competitions = json.load(f)

competitions_df = pd.DataFrame(competitions)

filtered_comps = competitions_df[
    (competitions_df["competition_name"].isin(TOP5_LEAGUES)) &
    (competitions_df["season_name"] == TARGET_SEASON)
]

# -----------------------------
# Extract passes from all matches
# -----------------------------

all_passes = []

for _, row in filtered_comps.iterrows():
    competition_id = row["competition_id"]
    season_id = row["season_id"]
    league_name = row["competition_name"]

    matches_path = DATA_DIR / "matches" / str(competition_id) / f"{season_id}.json"

    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    print(f"📂 Processing {league_name} — {len(matches)} matches...")

    for match in matches:
        match_id = match["match_id"]
        events_path = DATA_DIR / "events" / f"{match_id}.json"

        if not events_path.exists():
            print(f"  ⚠️  Missing events file: {match_id}")
            continue

        with open(events_path, "r", encoding="utf-8") as f:
            events = json.load(f)

        # Filter pass events and normalize into flat DataFrame
        pass_events = [e for e in events if e.get("type", {}).get("name") == "Pass"]

        if not pass_events:
            continue

        rows = []
        for e in pass_events:
            p = e.get("pass", {})
            location = e.get("location", [None, None])
            end_location = p.get("end_location", [None, None])
            outcome = p.get("outcome", {})

            rows.append({
                "team_id":        e.get("team", {}).get("id"),
                "match_id":       match_id,
                "player_id":      e.get("player", {}).get("id"),
                "pass_successful": outcome == {},   # no outcome dict = successful
                "pass_length":    p.get("length"),
                "pass_height":    p.get("height", {}).get("name"),
                "start_x":        location[0] if isinstance(location, list) else None,
                "start_y":        location[1] if isinstance(location, list) else None,
                "end_x":          end_location[0] if isinstance(end_location, list) else None,
                "end_y":          end_location[1] if isinstance(end_location, list) else None,
            })

        all_passes.append(pd.DataFrame(rows))

# -----------------------------
# Combine & write to DuckDB
# -----------------------------

passes_df = pd.concat(all_passes, ignore_index=True)

con = duckdb.connect(DUCKDB_PATH)
con.register("passes_df", passes_df)

con.execute("""
    CREATE OR REPLACE TABLE passes AS
    SELECT * FROM passes_df
""")

con.close()

print(f"✅ passes table created — {len(passes_df):,} rows.")