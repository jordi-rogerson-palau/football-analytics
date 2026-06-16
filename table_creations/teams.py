import json
import pandas as pd
import duckdb
from pathlib import Path

# -----------------------------
# Resolve project root directory
# -----------------------------

# Path to this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Parent directory → "projects"
PROJECT_ROOT = SCRIPT_DIR.parent

# Define data + output locations
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

teams_list = []

# -----------------------------
# Extract teams from matches
# -----------------------------

for _, row in filtered_comps.iterrows():

    competition_id = row["competition_id"]
    season_id = row["season_id"]
    league_name = row["competition_name"]
    league_id = competition_id

    matches_path = DATA_DIR / "matches" / str(competition_id) / f"{season_id}.json"

    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    for match in matches:
        teams_list.append({
            "team_id": match["home_team"]["home_team_id"],
            "team_name": match["home_team"]["home_team_name"],
            "league_id": league_id,
            "league_name": league_name
        })

        teams_list.append({
            "team_id": match["away_team"]["away_team_id"],
            "team_name": match["away_team"]["away_team_name"],
            "league_id": league_id,
            "league_name": league_name
        })

teams_df = pd.DataFrame(teams_list).drop_duplicates()

# -----------------------------
# Write to DuckDB
# -----------------------------

con = duckdb.connect(DUCKDB_PATH)

con.register("teams_df", teams_df)

con.execute("""
CREATE OR REPLACE TABLE teams AS
SELECT * FROM teams_df
""")

con.close()

print("✅ Teams table created successfully.")