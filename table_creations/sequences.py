import os
import json
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

# -----------------------------------
# KAGGLE PATHS
# -----------------------------------

EVENTS_DIR = Path("/kaggle/input/datasets/saurabhshahane/statsbomb-football-data/data/events")
OUTPUT_DIR = Path("/kaggle/working")
DB_PATH    = OUTPUT_DIR / "statsbomb_2015_2016.duckdb"

# -----------------------------------
# HELPERS — vectorized end locations
# -----------------------------------

def get_team_attack_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine attack direction per (match_id, period, possession_team.name).
    
    Primary signal: average carry dx per team per half.
      - Teams consistently move the ball toward higher x when attacking right.
      - Aggregating over an entire half gives a robust, low-noise signal.
    
    Fallback (per possession): for any team/half combo still unresolved,
      use the mean carry dx within each individual possession.
    
    Pitch: 120 x 80. Attack right → x increases toward 120.
    """
    carry_mask = (
        (df["type.name"] == "Carry") &
        df["location"].apply(lambda x: isinstance(x, list)) &
        df["carry.end_location"].apply(lambda x: isinstance(x, list)) &
        (df["team.name"] == df["possession_team.name"])  # only possessing team's carries
    )
    carries = df[carry_mask].copy()
    carries["carry_dx"] = (
        carries["carry.end_location"].apply(lambda x: x[0]) -
        carries["location"].apply(lambda x: x[0])
    )

    # --- Primary: aggregate by match + period + possession_team ---
    half_dir = (
        carries
        .groupby(["match_id", "period", "possession_team.name"])["carry_dx"]
        .mean()
        .reset_index()
        .rename(columns={"carry_dx": "mean_dx"})
    )
    half_dir["attack_right"] = half_dir["mean_dx"] > 0

    # Merge primary signal back onto original df possessions
    possession_teams = (
        df[["match_id", "period", "possession", "possession_team.name"]]
        .drop_duplicates()
    )
    direction = possession_teams.merge(
        half_dir[["match_id", "period", "possession_team.name", "attack_right"]],
        on=["match_id", "period", "possession_team.name"],
        how="left"
    )

    # --- Fallback: per-possession carry dx for still-missing rows ---
    missing_mask = direction["attack_right"].isna()
    if missing_mask.any():
        poss_dir = (
            carries
            .groupby(["match_id", "possession"])["carry_dx"]
            .mean()
            .reset_index()
            .rename(columns={"carry_dx": "poss_mean_dx"})
        )
        direction = direction.merge(poss_dir, on=["match_id", "possession"], how="left")
        direction.loc[missing_mask, "attack_right"] = (
            direction.loc[missing_mask, "poss_mean_dx"] > 0
        )
        direction = direction.drop(columns=["poss_mean_dx"])

    # Final fallback: assume attacking right
    direction["attack_right"] = direction["attack_right"].fillna(True)

    return direction[["match_id", "possession", "attack_right"]]


def normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip x and y coordinates for possessions where the team is attacking left,
    so all possessions are normalized to attacking right (x increases toward goal).
    Pitch dimensions: 120 (length) x 80 (width).
    """
    direction = get_team_attack_direction(df)
    df = df.merge(direction, on=["match_id", "possession"], how="left")

    df["attack_right"] = df["attack_right"].fillna(True)
    needs_flip = ~df["attack_right"]

    for x_col, y_col in [
        ("start_x",     "start_y"),
        ("event_end_x", "event_end_y"),
    ]:
        if x_col in df.columns:
            df.loc[needs_flip, x_col] = 120 - df.loc[needs_flip, x_col]
        if y_col in df.columns:
            df.loc[needs_flip, y_col] = 80  - df.loc[needs_flip, y_col]

    df = df.drop(columns=["attack_right"])
    return df


   


def extract_end_locations(df: pd.DataFrame) -> pd.DataFrame:
    def safe_col(col, idx):
        if col not in df.columns:
            return pd.Series([None] * len(df), index=df.index)
        return df[col].apply(
            lambda x: x[idx] if isinstance(x, list) and len(x) > idx else None
        )

    pass_x  = safe_col("pass.end_location", 0)
    pass_y  = safe_col("pass.end_location", 1)
    carry_x = safe_col("carry.end_location", 0)
    carry_y = safe_col("carry.end_location", 1)
    gk_x    = safe_col("goalkeeper.end_location", 0)
    gk_y    = safe_col("goalkeeper.end_location", 1)
    loc_x   = safe_col("location", 0)
    loc_y   = safe_col("location", 1)

    t = df["type.name"] if "type.name" in df.columns else df.get("type", pd.Series(dtype=str))

    is_pass  = t == "Pass"
    is_carry = t == "Carry"
    is_shot  = t == "Shot"   # ← shots now fall through to loc_x/loc_y below
    is_gk    = t == "Goal Keeper"

    end_x = (
        pd.Series(None, index=df.index, dtype=object)
        .where(~is_pass,  pass_x)
        .where(~is_carry, carry_x)
        # no shot branch here — shots use location (where shot was taken)
        .where(~is_gk,    gk_x)
    )
    end_x = end_x.combine_first(loc_x)  # shots fall back to their own location

    end_y = (
        pd.Series(None, index=df.index, dtype=object)
        .where(~is_pass,  pass_y)
        .where(~is_carry, carry_y)
        .where(~is_gk,    gk_y)
    )
    end_y = end_y.combine_first(loc_y)

    df["event_end_x"] = pd.to_numeric(end_x, errors="coerce")
    df["event_end_y"] = pd.to_numeric(end_y, errors="coerce")
    return df



def classify_end_vectorized(df: pd.DataFrame) -> pd.Series:
    t = df.get("type.name", pd.Series("", index=df.index)).fillna("")

    out_col            = df.get("out",                        pd.Series(False, index=df.index)).fillna(False).astype(bool)
    pass_outcome       = df.get("pass.outcome.name",          pd.Series("", index=df.index)).fillna("")
    dribble_outcome    = df.get("dribble.outcome.name",       pd.Series("", index=df.index)).fillna("")
    duel_outcome       = df.get("duel.outcome.name",          pd.Series("", index=df.index)).fillna("")
    shot_outcome       = df.get("shot.outcome.name",          pd.Series("", index=df.index)).fillna("")
    foul_card          = df.get("bad_behaviour.card.name",    pd.Series("", index=df.index)).fillna("")
    foul_type          = df.get("foul_committed.type.name",   pd.Series("", index=df.index)).fillna("")
    interception_out   = df.get("interception.outcome.name",  pd.Series("", index=df.index)).fillna("")
    goalkeeper_outcome = df.get("goalkeeper.success_out",     pd.Series("", index=df.index)).fillna("")

    # -------------------
    # SHOT CLASSIFICATION
    # -------------------

    shot_mask = t == "Shot"
    shot_type = pd.Series("shot_other", index=df.index)

    shot_type.loc[shot_mask & (shot_outcome == "Goal")]                                          = "shot_scored"
    shot_type.loc[shot_mask & shot_outcome.isin(["Saved", "Saved Off Target", "Saved to Post"])] = "shot_saved"
    shot_type.loc[shot_mask & (shot_outcome == "Blocked")]                                       = "shot_blocked"
    shot_type.loc[shot_mask & shot_outcome.isin(["Off T", "Post", "Wayward"])]                   = "shot_missed"

     # -------------------
    # OTHER CONDITIONS
    # -------------------

    conditions = [
        shot_mask,
        out_col,
        (t == "Pass")         & (pass_outcome    != "") & (pass_outcome    != "Complete"),
        (t == "Dribble")      & (dribble_outcome != "") & (dribble_outcome != "Complete"),
        (t == "Duel")         & (duel_outcome    != "") & (duel_outcome    != "Won"),
        t == "Dispossessed",
        (t == "Interception") & (interception_out != ""),
        (foul_card != "") | (foul_type != ""),
        t.str.contains("Goalkeeper", na=False) | (goalkeeper_outcome != ""),
        t == "Miscontrol",
        t.str.contains("Clearance", na=False),
        t == "Offside",
        t == "Substitution",
        t == "Foul Won",
        t == "Block",
        t == "Ball Receipt*",
    ]

    choices = [
        None,               # shot rows handled separately below
        "out_of_bounds",
        "failed_pass",
        "failed_dribble",
        "lost_duel",
        "dispossessed",
        "interception",
        "foul",
        "goalkeeper_action",
        "miscontrol",
        "clearance",
        "offside",
        "substitution",
        "foul_won",
        "block",
        "natural_transition",
    ]

    end_type = pd.Series(
        np.select(conditions, choices, default="other"),
        index=df.index
    )

    # Overwrite shot rows with detailed shot classification
    end_type.loc[shot_mask] = shot_type.loc[shot_mask]

    return end_type


def build_last_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Shot-aware last event selection with priority-based shot picking.

    StatsBomb appends bookkeeping events (e.g. "Ball Receipt*") after a goal
    within the same possession number, so naive tail(1) misses the shot.

    For possessions WITH a shot:
      - If multiple shots exist (e.g. saved rebound then goal), keep the
        highest-priority outcome (Goal > Saved > Blocked > missed variants)
        so goals are never discarded in favour of an earlier saved shot.

    For possessions WITHOUT a shot → use the true last event.
    """

    COLS = ["match_id", "possession", "possession_team_id",
            "event_end_x", "event_end_y", "end_type"]

    SHOT_PRIORITY = {
        "Goal":             0,
        "Saved":            1,
        "Saved to Post":    1,
        "Saved Off Target": 1,
        "Blocked":          2,
        "Post":             3,
        "Off T":            4,
        "Wayward":          5,
    }

    # 1. Shot possessions — pick highest-priority outcome per possession
    shot_df = events[events["type.name"] == "Shot"].copy()
    shot_df["end_type"] = classify_end_vectorized(shot_df)
    shot_df["_outcome_priority"] = (
        shot_df["shot.outcome.name"]
        .map(SHOT_PRIORITY)
        .fillna(99)
    )

    shot_last = (
        shot_df
        .sort_values(["match_id", "possession", "_outcome_priority"])
        .drop_duplicates(subset=["match_id", "possession"], keep="first")
        .drop(columns=["_outcome_priority"])
    )[COLS]

    # 2. Non-shot possessions — use true last event
    # Fast merge-based mask (avoids float/int tuple mismatch from apply(axis=1))
    shot_keys   = shot_last[["match_id", "possession"]].drop_duplicates().assign(_has_shot=True)
    events_flagged = events.merge(shot_keys, on=["match_id", "possession"], how="left")
    no_shot_mask   = events_flagged["_has_shot"].isna().values

    no_shot_last = (
        events[no_shot_mask]
        .groupby(["match_id", "possession", "possession_team_id"], sort=False)
        .tail(1)
        .copy()
    )
    no_shot_last["end_type"] = classify_end_vectorized(no_shot_last)
    no_shot_last = no_shot_last[COLS]

    return pd.concat([shot_last, no_shot_last], ignore_index=True)







# -----------------------------------
# FILTER TO TARGET COMPETITIONS/SEASON
# -----------------------------------

MATCHES_DIR = Path("/kaggle/input/datasets/saurabhshahane/statsbomb-football-data/data/matches")

TARGET_COMPETITIONS = {
    "Premier League",
    "La Liga",
    "1. Bundesliga",
    "Ligue 1",
    "Serie A",
}
TARGET_SEASON = "2015/2016"

target_match_ids = set()
competition_counts = {}

for competition_dir in MATCHES_DIR.iterdir():
    for season_file in competition_dir.iterdir():
        if season_file.suffix != ".json":
            continue
        with open(season_file, encoding="utf-8") as f:
            matches = json.load(f)
        for match in matches:
            competition_name = match.get("competition", {}).get("competition_name", "")
            season_name      = match.get("season", {}).get("season_name", "")
            if competition_name in TARGET_COMPETITIONS and season_name == TARGET_SEASON:
                target_match_ids.add(match["match_id"])
                competition_counts[competition_name] = competition_counts.get(competition_name, 0) + 1

print(f"Matches found for season {TARGET_SEASON}:")
for comp, count in sorted(competition_counts.items()):
    print(f"  {comp}: {count} matches")
print(f"  TOTAL: {len(target_match_ids)} matches")

# -----------------------------------
# PROCESS FILES IN BATCHES
# -----------------------------------

all_files = [
    f for f in os.listdir(EVENTS_DIR)
    if f.endswith(".json") and int(f.replace(".json", "")) in target_match_ids
]

print(f"Event files to process: {len(all_files)}")

BATCH_SIZE = 50

con = duckdb.connect(str(DB_PATH))
con.execute("DROP TABLE IF EXISTS sequences_premier_league")
table_created = False

for batch_start in range(0, len(all_files), BATCH_SIZE):
    batch_files = all_files[batch_start: batch_start + BATCH_SIZE]
    print(f"  Processing files {batch_start + 1}–{batch_start + len(batch_files)} / {len(all_files)} ...")

    batch_events = []
    for file in batch_files:
        match_id = int(file.replace(".json", ""))
        with open(EVENTS_DIR / file, encoding="utf-8") as f:
            match_events = json.load(f)
        df = pd.json_normalize(match_events)
        df["match_id"] = match_id
        batch_events.append(df)

    events = pd.concat(batch_events, ignore_index=True)

    events = events[
        events["possession"].notna() &
        events["possession_team.id"].notna()
    ].copy()

    events = events.rename(columns={"possession_team.id": "possession_team_id"})

    type_col = "type.name" if "type.name" in events.columns else "type"

    events["start_x"] = pd.to_numeric(
        events["location"].apply(lambda x: x[0] if isinstance(x, list) else None),
        errors="coerce"
    )
    events["start_y"] = pd.to_numeric(
        events["location"].apply(lambda x: x[1] if isinstance(x, list) else None),
        errors="coerce"
    )

    events = extract_end_locations(events)
    events = normalize_direction(events)
    events["absolute_time"] = events["minute"] * 60 + events["second"]
    events = events.sort_values(["match_id", "possession", "absolute_time"])

    # null/zero possession cleaning
    events = events[events["possession"] != 0]
    events = events[events["start_x"].notna() & events["start_y"].notna()]

    # out of bounds cleaning
    events = events[
        events["start_x"].between(0, 120) &
        events["start_y"].between(0, 80)
    ]
    events = events[
        events["event_end_x"].isna() |
        (events["event_end_x"].between(0, 120) & events["event_end_y"].between(0, 80))
    ]

   # last_events = events.groupby(
    #    ["match_id", "possession", "possession_team_id"], sort=False
    #).tail(1).copy()

    #last_events["end_type"] = classify_end_vectorized(last_events)
    #last_events = last_events[[
     #   "match_id", "possession", "possession_team_id",
      #  "event_end_x", "event_end_y", "end_type"
    #]]

    last_events = build_last_events(events) 
    sequences = (
        events
        .assign(_type=events[type_col])
        .groupby(["match_id", "possession", "possession_team_id"], sort=False)
        .agg(
            num_passes   =("_type", lambda x: (x == "Pass").sum()),
            num_carries  =("_type", lambda x: (x == "Carry").sum()),
            num_dribbles =("_type", lambda x: (x == "Dribble").sum()),
            num_duels    =("_type", lambda x: (x == "Duel").sum()),
            start_x      =("start_x", "first"),
            start_y      =("start_y", "first"),
            start_time   =("absolute_time", "first"),
            end_time     =("absolute_time", "last"),
        )
        .reset_index()
    )

    sequences = sequences.merge(
        last_events,
        on=["match_id", "possession", "possession_team_id"],
        how="left"
    )

    sequences["duration_seconds"]    = sequences["end_time"] - sequences["start_time"]
    sequences["distance_progressed"] = sequences["event_end_x"] - sequences["start_x"]
    sequences["sequence_id"] = (
        sequences["match_id"].astype(str) + "_" +
        sequences["possession_team_id"].astype(str) + "_" +
        sequences["possession"].astype(str)
    )

    sequences = sequences.rename(columns={
        "possession_team_id": "team_id",
        "event_end_x": "end_x",
        "event_end_y": "end_y",
    })

    sequences = sequences[[
        "sequence_id", "match_id", "team_id", "possession",
        "num_passes", "num_carries", "num_dribbles", "num_duels",
        "start_x", "start_y", "end_x", "end_y",
        "duration_seconds", "distance_progressed", "end_type"
    ]]
    print(sequences.head())

    # Quick sanity check: goals should always end near x=120
    goals = events[
        (events["type.name"] == "Shot") &
        (events.get("shot.outcome.name", pd.Series(dtype=str)) == "Goal")
    ]
    if len(goals):
        print(f"    [sanity] goal end_x min={goals['event_end_x'].min():.1f} "
              f"mean={goals['event_end_x'].mean():.1f} "
              f"max={goals['event_end_x'].max():.1f}  (expect ~120)")

    if not table_created:
        con.execute("CREATE TABLE sequences_premier_league AS SELECT * FROM sequences")
        table_created = True
    else:
        con.execute("INSERT INTO sequences_premier_league SELECT * FROM sequences")

    del events, last_events, sequences, batch_events

con.execute("CHECKPOINT")
con.close()
print(f"✅ Done! DuckDB saved to: {DB_PATH}")
print("   Download it from the Kaggle output panel on the right.")