import json
import pandas as pd
import duckdb
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

matches_list = []

# -----------------------------
# Extract matches
# -----------------------------

for _, row in filtered_comps.iterrows():

    competition_id = row["competition_id"]
    season_id = row["season_id"]

    league_id = competition_id

    matches_path = DATA_DIR / "matches" / str(competition_id) / f"{season_id}.json"

    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    for match in matches:
        matches_list.append({
            "match_id": match["match_id"],
            "league_id": league_id,
            "home_score": match["home_score"],
            "away_score": match["away_score"]
        })

matches_df = pd.DataFrame(matches_list)

# -----------------------------
# Write to DuckDB
# -----------------------------

con = duckdb.connect(DUCKDB_PATH)

con.register("matches_df", matches_df)

con.execute("""
CREATE OR REPLACE TABLE match_scores AS
SELECT * FROM matches_df
""")

con.close()

print("✅ Match scores table created successfully.")