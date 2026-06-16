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
DATA_DIR = PROJECT_ROOT / "open-data/data"
DUCKDB_PATH = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"


# Connect to the DuckDB file
conn = duckdb.connect(DUCKDB_PATH)

teams = conn.execute("SELECT * FROM teams").df()
sequences = conn.execute("SELECT * FROM sequences").df()

shots_ = sequences[sequences['end_type'].isin(['shot_scored', 'shot_blocked', 'shot_saved', 'shot_missed'])]

# merge shots_ with teams on team_id to get league name
shots_ = shots_.merge(teams[["team_id", "league_name"]], left_on="team_id", right_on="team_id", how="left")

shots_teams = shots_.drop(columns=['sequence_id', 'match_id', 'possession'])

shots_teams = shots_.groupby(['team_id', 'league_name', 'end_type']).agg(
    count=('end_type', 'size'),
    avg_num_passes=('num_passes', 'mean'),
    avg_num_carries=('num_carries', 'mean'),
    avg_num_dribbles=('num_dribbles', 'mean'),
    avg_num_duels=('num_duels', 'mean'),
    avg_distance_progressed=('distance_progressed', 'mean'),
    avg_duration_seconds=('duration_seconds', 'mean'),
    avg_start_x=('start_x', 'mean'),
    avg_start_y=('start_y', 'mean'),
    avg_end_x=('end_x', 'mean'),
    avg_end_y=('end_y', 'mean')
).reset_index()

shots_league = shots_.drop(columns=['sequence_id', 'match_id', 'possession', 'team_id'])

#group by league name and end type, have a count and rest of attributes compute avg
shots_league = shots_.groupby(['league_name', 'end_type']).agg(
    count=('end_type', 'size'),
    avg_num_passes=('num_passes', 'mean'),
    avg_num_carries=('num_carries', 'mean'),
    avg_num_dribbles=('num_dribbles', 'mean'),
    avg_num_duels=('num_duels', 'mean'),
    avg_distance_progressed=('distance_progressed', 'mean'),
    avg_duration_seconds=('duration_seconds', 'mean'),
    avg_start_x=('start_x', 'mean'),
    avg_start_y=('start_y', 'mean'),
    avg_end_x=('end_x', 'mean'),
    avg_end_y=('end_y', 'mean')
).reset_index()

#copy to duckdb the clean shots_, shots_teams and shots_league tables; shots_ has to replace the already existing one
conn.execute("CREATE OR REPLACE TABLE shots AS SELECT * FROM shots_")
conn.execute("CREATE OR REPLACE TABLE shots_teams AS SELECT * FROM shots_teams")
conn.execute("CREATE OR REPLACE TABLE shots_league AS SELECT * FROM shots_league")

conn.close()
