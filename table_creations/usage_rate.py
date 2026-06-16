import os
import json
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
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

# ── Build match metadata ──────────────────────────────────────────────────
match_rows = []
for matches_file in (DATA_DIR / "matches").glob("**/*.json"):
    with open(matches_file, encoding="utf-8") as f:
        matches = json.load(f)
    for m in matches:
        competition = m.get("competition", {})
        season      = m.get("season", {})
        match_rows.append({
            "match_id":    m["match_id"],
            "league_name": competition.get("competition_name"),
            "league_id":   competition.get("competition_id"),
            "season_name": season.get("season_name"),
        })

matches_meta = pd.DataFrame(match_rows)

filtered_meta = matches_meta[
    (matches_meta["league_name"].isin(TOP5_LEAGUES)) &
    (matches_meta["season_name"] == TARGET_SEASON)
]

valid_match_ids = sorted(filtered_meta["match_id"].unique())

print(f"Processing {len(valid_match_ids)} matches across top 5 leagues {TARGET_SEASON}")

# ── Ball-loss classification ───────────────────────────────────────────────
# Mirrors classify_end_vectorized from the benchmark but we apply it to
# every event (not just the last of a possession) and only keep rows
# that represent a loss of possession by the acting team.
#
# Loss-of-possession event types and their ball_lost_type label:
#
#   Explicit losses (the acting player directly loses the ball):
#     failed_pass        – Pass with non-Complete outcome
#     failed_dribble     – Dribble that is not Complete
#     lost_duel          – Duel that is not Won
#     dispossessed       – Dispossessed event
#     miscontrol         – Miscontrol event
#     out_of_bounds      – Any event where out == True
#
#   Possession-transfer events (ball moves to opponent):
#     interception       – Interception by opponent (we record the opponent's player)
#     clearance          – Clearance (ball leaves the zone, possession unclear)
#     goalkeeper_action  – Goalkeeper action that ends possession
#
#   Structural losses:
#     shot_saved         – Shot saved by goalkeeper
#     shot_blocked       – Shot blocked
#     shot_missed        – Shot off target / wide / post
#     offside            – Offside
#     foul               – Foul committed (possession ends)

LOSS_TYPE_MAP = {
    # (condition_key): label
    "out_of_bounds":    "out_of_bounds",
    "failed_pass":      "failed_pass",
    "failed_dribble":   "failed_dribble",
    "lost_duel":        "lost_duel",
    "dispossessed":     "dispossessed",
    "miscontrol":       "miscontrol",
    "interception":     "interception",
    "clearance":        "clearance",
    "goalkeeper_action":"goalkeeper_action",
    "shot_saved":       "shot_saved",
    "shot_blocked":     "shot_blocked",
    "shot_missed":      "shot_missed",
    "offside":          "offside",
    "foul":             "foul",
}


def classify_ball_losses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a normalised events DataFrame (pd.json_normalize output),
    return only the rows that represent a loss of possession,
    with a 'ball_lost_type' column added.
    """
    t = df.get("type.name", pd.Series("", index=df.index)).fillna("")

    out_col          = df.get("out",                       pd.Series(False, index=df.index)).fillna(False).astype(bool)
    pass_outcome     = df.get("pass.outcome.name",         pd.Series("", index=df.index)).fillna("")
    dribble_outcome  = df.get("dribble.outcome.name",      pd.Series("", index=df.index)).fillna("")
    duel_outcome     = df.get("duel.outcome.name",         pd.Series("", index=df.index)).fillna("")
    shot_outcome     = df.get("shot.outcome.name",         pd.Series("", index=df.index)).fillna("")
    foul_type        = df.get("foul_committed.type.name",  pd.Series("", index=df.index)).fillna("")
    foul_card        = df.get("bad_behaviour.card.name",   pd.Series("", index=df.index)).fillna("")
    interc_outcome   = df.get("interception.outcome.name", pd.Series("", index=df.index)).fillna("")
    gk_outcome       = df.get("goalkeeper.success_out",    pd.Series("", index=df.index)).fillna("")

    conditions = [
        out_col,
        (t == "Pass")         & pass_outcome.isin(["Incomplete", "Out", "Pass Offside", "Unknown"]),
        (t == "Dribble")      & (dribble_outcome != "") & (dribble_outcome != "Complete"),
        (t == "Duel")         & duel_outcome.isin(["Lost In Play", "Lost Out", "Success Out"]),
        t == "Dispossessed",
        t == "Miscontrol",
        (t == "Interception") & (interc_outcome != ""),
        t.str.contains("Clearance", na=False),
        t.str.contains("Goalkeeper", na=False) | (gk_outcome != ""),
        (t == "Shot") & shot_outcome.isin(["Saved", "Saved Off Target", "Saved to Post"]),
        (t == "Shot") & (shot_outcome == "Blocked"),
        (t == "Shot") & shot_outcome.isin(["Off T", "Post", "Wayward"]),
        t == "Offside",
        (t == "Foul Committed") | (foul_card != "") | (foul_type != ""),
    ]

    labels = [
        "out_of_bounds",
        "failed_pass",
        "failed_dribble",
        "lost_duel",
        "dispossessed",
        "miscontrol",
        "interception",
        "clearance",
        "goalkeeper_action",
        "shot_saved",
        "shot_blocked",
        "shot_missed",
        "offside",
        "foul",
    ]

    ball_lost_type = pd.Series(
        np.select(conditions, labels, default=None),
        index=df.index
    )

    loss_mask = ball_lost_type.notna()
    result = df[loss_mask].copy()
    result["ball_lost_type"] = ball_lost_type[loss_mask]
    return result


# ── Main loop ─────────────────────────────────────────────────────────────
all_rows = []

for i, match_id in enumerate(valid_match_ids):
    try:
        match_meta = filtered_meta[filtered_meta["match_id"] == match_id].iloc[0]
        league_id  = match_meta["league_id"]

        events_path = DATA_DIR / "events" / f"{match_id}.json"
        with open(events_path, encoding="utf-8") as f:
            raw = json.load(f)

        events = pd.json_normalize(raw)
        events["match_id"] = match_id

        # Extract location_x / location_y
        events["location_x"] = events["location"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) >= 2 else None
        )
        events["location_y"] = events["location"].apply(
            lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else None
        )

        # Extract player_id (null if event has no player, e.g. some GK/team events)
        if "player.id" in events.columns:
            events["player_id"] = events["player.id"]
        else:
            events["player_id"] = None

        # Extract team_id of the acting team
        if "team.id" in events.columns:
            events["team_id"] = events["team.id"]
        else:
            events["team_id"] = None

        # Classify and filter to ball-loss rows only
        losses = classify_ball_losses(events)

        if losses.empty:
            continue

        losses["match_id"]  = match_id
        losses["league_id"] = league_id

        all_rows.append(losses[[
            "match_id", "league_id", "team_id", "player_id",
            "ball_lost_type", "location_x", "location_y",
        ]])

    except Exception as e:
        print(f"ERROR on match_id {match_id}: {e}")
        continue

    if (i + 1) % 5 == 0 or (i + 1) == len(valid_match_ids):
        print(f"  Processed {i + 1}/{len(valid_match_ids)} matches...")

# ── Assemble & write ──────────────────────────────────────────────────────
ball_losses_df = pd.concat(all_rows, ignore_index=True)

conn = duckdb.connect(str(DUCKDB_PATH))
conn.execute("CREATE OR REPLACE TABLE ball_losses AS SELECT * FROM ball_losses_df")
conn.close()

print(f"\nWritten {len(ball_losses_df)} rows to 'ball_losses' table in DuckDB")

# ── Sanity checks ─────────────────────────────────────────────────────────
print("\n=== COLUMNS ===")
print(ball_losses_df.columns.tolist())

print("\n=== SHAPE ===")
print(ball_losses_df.shape)

print("\n=== SAMPLE (5 rows) ===")
print(ball_losses_df.head().to_string(index=False))

print("\n=== NULLS ===")
print(ball_losses_df.isnull().sum())

print("\n=== DISTINCT ball_lost_type ===")
print(ball_losses_df["ball_lost_type"].value_counts())

print("\n=== LOSSES WITH NULL player_id ===")
null_player = ball_losses_df["player_id"].isna().sum()
total       = len(ball_losses_df)
print(f"{null_player} / {total} ({100*null_player/total:.1f}%) have no player_id")

print("\n=== LOCATION RANGE ===")
print(f"location_x: {ball_losses_df['location_x'].min():.1f} – {ball_losses_df['location_x'].max():.1f}")
print(f"location_y: {ball_losses_df['location_y'].min():.1f} – {ball_losses_df['location_y'].max():.1f}")

print("\n=== LOSSES PER MATCH ===")
print(ball_losses_df.groupby("match_id").size().describe())