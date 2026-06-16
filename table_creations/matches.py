import json
import pandas as pd
import duckdb
from pathlib import Path
import os

SCRIPT_DIR   = Path(os.getcwd())
PROJECT_ROOT = SCRIPT_DIR.parent
MATCHES_DIR  = PROJECT_ROOT / "open-data" / "data" / "matches"
DB_PATH      = PROJECT_ROOT / "statsbomb_2015_2016.duckdb"

# ── Parse all match JSONs ─────────────────────────────────────────────────────
records = []
for comp_folder in MATCHES_DIR.iterdir():
    if not comp_folder.is_dir():
        continue
    for season_file in comp_folder.glob("*.json"):
        with open(season_file, encoding="utf-8") as f:
            matches = json.load(f)
        for m in matches:
            records.append({
                "match_id":     m["match_id"],
                "home_team_id": m["home_team"]["home_team_id"],
                "away_team_id": m["away_team"]["away_team_id"],
            })

match_teams = pd.DataFrame(records)
print(f"Parsed {len(match_teams)} matches")

# ── Store in DuckDB ───────────────────────────────────────────────────────────
con = duckdb.connect(str(DB_PATH))
con.execute("DROP TABLE IF EXISTS matches")
con.execute("CREATE TABLE matches AS SELECT * FROM match_teams")
con.close()

print(f"Table 'matches' stored in {DB_PATH}")