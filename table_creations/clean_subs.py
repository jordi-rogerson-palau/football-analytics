import os
import duckdb
import pandas as pd
from pathlib import Path

# --- Path setup (consistent with project convention) ---
SCRIPT_DIR = Path(os.getcwd())
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"

print(f"Connecting to DuckDB at: {DB_PATH}")
con = duckdb.connect(str(DB_PATH))

# -------------------------------------------------------
# 1. Load tables
# -------------------------------------------------------
subs    = con.execute("SELECT * FROM substitutions").df()
matches = con.execute("SELECT match_id, home_team_id, away_team_id FROM matches").df()
teams   = con.execute("SELECT team_id, team_name FROM teams").df()
players = con.execute("SELECT player_id, player_name FROM players").df()

# player_id is stored as float → cast to int for clean merging
players["player_id"] = players["player_id"].astype("Int64")

print(f"  substitutions  → {subs.shape[0]:>6} rows, {subs.shape[1]} cols")
print(f"  matches        → {matches.shape[0]:>6} rows, {matches.shape[1]} cols")
print(f"  teams          → {teams.shape[0]:>6} rows, {teams.shape[1]} cols")
print(f"  players        → {players.shape[0]:>6} rows, {players.shape[1]} cols")

# -------------------------------------------------------
# 2. Merge subs with matches on game_id = match_id
#    → appends home_team_id and away_team_id to each sub event
# -------------------------------------------------------
subs_enriched = subs.merge(
    matches.rename(columns={"match_id": "game_id"}),
    on="game_id",
    how="left"
)

null_check = subs_enriched[["home_team_id", "away_team_id"]].isna().sum()
print(f"\nAfter merging matches  → {subs_enriched.shape[0]} rows | nulls: {null_check.to_dict()}")

# -------------------------------------------------------
# 3. Join teams twice to get home and away team names
# -------------------------------------------------------
subs_enriched = subs_enriched.merge(
    teams.rename(columns={"team_id": "home_team_id", "team_name": "home_team_name"}),
    on="home_team_id",
    how="left"
)

subs_enriched = subs_enriched.merge(
    teams.rename(columns={"team_id": "away_team_id", "team_name": "away_team_name"}),
    on="away_team_id",
    how="left"
)

null_names = subs_enriched[["home_team_name", "away_team_name"]].isna().sum()
print(f"After joining team names → {subs_enriched.shape[0]} rows | nulls: {null_names.to_dict()}")

# -------------------------------------------------------
# 4. Join players twice to get sub_in and sub_out names
# -------------------------------------------------------
subs_enriched = subs_enriched.merge(
    players.rename(columns={"player_id": "sub_in_id", "player_name": "sub_in_name"}),
    on="sub_in_id",
    how="left"
)

subs_enriched = subs_enriched.merge(
    players.rename(columns={"player_id": "sub_out_id", "player_name": "sub_out_name"}),
    on="sub_out_id",
    how="left"
)

null_players = subs_enriched[["sub_in_name", "sub_out_name"]].isna().sum()
print(f"After joining player names → {subs_enriched.shape[0]} rows | nulls: {null_players.to_dict()}")

# -------------------------------------------------------
# 5. Preview
# -------------------------------------------------------
print("\nSample output (5 rows):")
print(subs_enriched[[
    "game_id", "team_id",
    "home_team_id", "home_team_name",
    "away_team_id", "away_team_name",
    "sub_in_id", "sub_in_name",
    "sub_out_id", "sub_out_name",
    "position_in", "position_out",
    "minute", "current_result"
]].head().to_string(index=False))

# -------------------------------------------------------
# 6. Overwrite substitutions table in DuckDB
# -------------------------------------------------------
con.execute("DROP TABLE IF EXISTS substitutions")
con.execute("CREATE TABLE substitutions AS SELECT * FROM subs_enriched")

row_count = con.execute("SELECT COUNT(*) AS n FROM substitutions").df()["n"].iloc[0]
print(f"\n✓ substitutions overwritten → {row_count} rows")

final_schema = con.execute("DESCRIBE substitutions").df()
print("\nFinal schema:")
print(final_schema[["column_name", "column_type"]].to_string(index=False))

con.close()
print("\nDone. Connection closed.")