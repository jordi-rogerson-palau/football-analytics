# clean_passes.py

import os
import pandas as pd
import duckdb
from pathlib import Path

SCRIPT_DIR   = Path(os.getcwd())
PROJECT_ROOT = SCRIPT_DIR.parent
DUCKDB_PATH  = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"

conn = duckdb.connect(DUCKDB_PATH)

passes_ = conn.execute("SELECT * FROM passes").df()
teams   = conn.execute("SELECT * FROM teams").df()

# ── out-of-bounds filtering (still needed) ────────────────────────────────────
passes_ = passes_[
    (passes_["start_x"] >= 0) & (passes_["start_x"] <= 120) &
    (passes_["start_y"] >= 0) & (passes_["start_y"] <= 80)  &
    (passes_["end_x"]   >= 0) & (passes_["end_x"]   <= 120) &
    (passes_["end_y"]   >= 0) & (passes_["end_y"]   <= 80)
]

# ── merge with teams to get league_name ───────────────────────────────────────
league_passes = passes_.merge(
    teams[["team_id", "league_name"]],
    on="team_id",
    how="left"
)

# ── write back to DuckDB ──────────────────────────────────────────────────────
conn.execute("DROP TABLE IF EXISTS passes")
conn.execute("CREATE TABLE passes AS SELECT * FROM passes_")

conn.execute("DROP TABLE IF EXISTS league_passes")
conn.execute("CREATE TABLE league_passes AS SELECT * FROM league_passes")

conn.close()
print(f"✅ passes: {len(passes_):,} rows")
print(f"✅ league_passes: {len(league_passes):,} rows")