import json
import os
import pandas as pd
import duckdb
from pathlib import Path


# Path to this script
SCRIPT_DIR = Path(os.getcwd())

# Parent directory → "projects"
PROJECT_ROOT = SCRIPT_DIR.parent

# Define data + output locations
DUCKDB_PATH     = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"
NEW_SEQ_PATH    = PROJECT_ROOT / "new_seq.duckdb"

# ── 1. Pull new sequences from new_seq.duckdb ──────────────────────────────────
new_conn = duckdb.connect(str(NEW_SEQ_PATH), read_only=True)
sequences = new_conn.execute("SELECT * FROM sequences_premier_league").df()
new_conn.close()

# ── 2. Connect to main DB and load supporting tables ───────────────────────────
conn = duckdb.connect(str(DUCKDB_PATH))
teams = conn.execute("SELECT * FROM teams").df()

# ── 3. Cleaning steps (unchanged) ─────────────────────────────────────────────
sequences = sequences[sequences['duration_seconds'] <= 1000]

sequences['average_speed'] = sequences['distance_progressed'] / sequences['duration_seconds']
sequences.loc[sequences['duration_seconds'] == 0, 'average_speed'] = 0

# ── 4. sequences_teams aggregation (unchanged) ────────────────────────────────
cols_to_drop = ['sequence_id', 'match_id', 'possession', 'end_type']
sequences_teams = sequences.drop(columns=cols_to_drop)

group_cols = ['team_id']

sequences_teams = (sequences_teams
       .groupby(group_cols, dropna=False)
       .mean()
       .reset_index())

sequences_teams = sequences_teams.merge(teams, on=['team_id'])

# ── 5. sequences_league aggregation (unchanged) ───────────────────────────────
sequences_league_ = (
    sequences_teams
    .drop(columns=['team_id', 'team_name'])
    .groupby(['league_id', 'league_name'], as_index=False)
    .mean(numeric_only=True)
)

sequences_league_['league_name'] = sequences_league_['league_name'].astype('category')

# ── 6. Write back to main DB, replacing old tables ────────────────────────────
conn.execute("DROP TABLE IF EXISTS sequences")
conn.execute("CREATE TABLE sequences AS SELECT * FROM sequences")

conn.execute("CREATE OR REPLACE TABLE sequences_teams AS SELECT * FROM sequences_teams")
conn.execute("CREATE OR REPLACE TABLE sequences_league AS SELECT * FROM sequences_league_")

conn.execute("CHECKPOINT")
conn.close()

print("✅ Done! sequences, sequences_teams and sequences_league updated in statsbomb_2015_2016.duckdb")