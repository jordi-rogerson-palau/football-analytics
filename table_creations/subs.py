import os
import json
import pandas as pd
import duckdb
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
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
TARGET_SEASON  = "2015/2016"
TEST_LIMIT     = None   # set to an integer to process only N matches

# ── Connect to DuckDB ─────────────────────────────────────────────────────
conn = duckdb.connect(str(DUCKDB_PATH))

# ── Build match metadata (match_id → league + season) ────────────────────
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

if TEST_LIMIT:
    valid_match_ids = valid_match_ids[:TEST_LIMIT]

print(f"Processing {len(valid_match_ids)} matches from {TOP5_LEAGUES} {TARGET_SEASON}")


# ── Helper: rolling goal tally → current score at each event ─────────────
def build_score_series(events_df):
    """
    Returns a dict mapping index → (home_goals, away_goals) scored
    *before* that event was processed (i.e. score at the time of the event).
    Requires events_df sorted by [period, minute, second, timestamp].
    """
    # Identify home team from the first kick-off event
    kickoff = events_df[events_df["type"].apply(
        lambda x: x.get("name") == "Kick Off" if isinstance(x, dict) else False
    )]
    if kickoff.empty:
        # Fallback: first team mentioned
        first_team_id = events_df["team"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else None
        ).dropna().iloc[0]
        home_team_id = first_team_id
    else:
        home_team_id = kickoff.iloc[0]["team"]
        if isinstance(home_team_id, dict):
            home_team_id = home_team_id.get("id")

    home_goals = 0
    away_goals = 0
    score_at = {}  # index → (home, away) before this event

    for idx, row in events_df.iterrows():
        score_at[idx] = (home_goals, away_goals)

        event_type = row.get("type", {})
        if isinstance(event_type, dict) and event_type.get("name") == "Shot":
            shot = row.get("shot", {})
            if isinstance(shot, dict):
                outcome = shot.get("outcome", {})
                if isinstance(outcome, dict) and outcome.get("name") == "Goal":
                    team_id = row.get("team", {})
                    if isinstance(team_id, dict):
                        team_id = team_id.get("id")
                    if team_id == home_team_id:
                        home_goals += 1
                    else:
                        away_goals += 1

    return score_at, home_team_id


# ── Helper: extract position from lineup positions list ──────────────────
def get_last_position_before(positions: list, event_time_sec: float) -> str | None:
    """
    Given a player's `positions` array from lineups, return the position name
    that was active at `event_time_sec` (seconds from match start).

    Each position entry has `from` (HH:MM:SS or MM:SS) and `to` fields.
    """
    if not positions:
        return None

    def to_seconds(t: str | None) -> float:
        if t is None:
            return float("inf")
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float("inf")

    active = None
    for p in positions:
        start = to_seconds(p.get("from"))
        end   = to_seconds(p.get("to"))   # None → inf (still playing)
        if start <= event_time_sec < end:
            return p.get("position")
        # Keep track of the most recent position started before event
        if start <= event_time_sec:
            active = p.get("position")

    return active


# ── Main loop ─────────────────────────────────────────────────────────────
all_rows = []

for i, match_id in enumerate(valid_match_ids):
    try:
        # -- Load events --------------------------------------------------
        events_path = DATA_DIR / "events" / f"{match_id}.json"
        with open(events_path, encoding="utf-8") as f:
            events = json.load(f)

        events_df = pd.DataFrame(events)

        # Sort chronologically
        for col in ["period", "minute", "second"]:
            if col not in events_df.columns:
                events_df[col] = 0
        events_df = events_df.sort_values(["period", "minute", "second"]).reset_index(drop=True)

        # Build rolling score lookup
        score_at, home_team_id = build_score_series(events_df)

        # -- Load lineups -------------------------------------------------
        lineup_path = DATA_DIR / "lineups" / f"{match_id}.json"
        with open(lineup_path, encoding="utf-8") as f:
            lineups_raw = json.load(f)

        # Build player_id → {positions, team_id} map
        player_info = {}
        for team_entry in lineups_raw:
            team_id = team_entry.get("team_id")
            for player in team_entry.get("lineup", []):
                player_info[player["player_id"]] = {
                    "positions": player.get("positions", []),
                    "team_id":   team_id,
                    "name":      player.get("player_name"),
                }

        # -- Find substitution events -------------------------------------
        sub_events = events_df[
            events_df["type"].apply(
                lambda x: x.get("name") == "Substitution" if isinstance(x, dict) else False
            )
        ]

        match_meta = filtered_meta[filtered_meta["match_id"] == match_id].iloc[0]
        league_id  = match_meta["league_id"]

        for _, ev in sub_events.iterrows():
            minute  = ev.get("minute", 0)
            second  = ev.get("second", 0)
            period  = ev.get("period", 1)

            team_raw = ev.get("team", {})
            team_id  = team_raw.get("id") if isinstance(team_raw, dict) else None

            # StatsBomb time: period 2 starts at 45', ET etc.
            # We compute seconds-from-kickoff for position lookup
            period_offsets = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60, 5: 120 * 60}
            event_time_sec = period_offsets.get(period, 0) + minute * 60 + second

            # sub_out player (the player performing the "Substitution" event)
            player_out_info = ev.get("player", {})
            sub_out_id = player_out_info.get("id") if isinstance(player_out_info, dict) else None

            # sub_in player lives inside the substitution detail dict
            sub_detail = ev.get("substitution", {})
            if isinstance(sub_detail, dict):
                replacement = sub_detail.get("replacement", {})
                sub_in_id   = replacement.get("id") if isinstance(replacement, dict) else None
            else:
                sub_in_id = None

            # Positions
            out_info = player_info.get(sub_out_id, {})
            in_info  = player_info.get(sub_in_id,  {})

            position_out = get_last_position_before(
                out_info.get("positions", []), event_time_sec
            )
            # The sub-in player's first position entry = what they came on as
            in_positions = in_info.get("positions", [])
            position_in  = in_positions[0].get("position") if in_positions else None

            # Current score at moment of substitution
            home_g, away_g = score_at.get(ev.name, (0, 0))  # ev.name is the df index
            current_result = f"{home_g}-{away_g}"

            timestamp = ev.get("timestamp", f"{minute:02d}:{second:02d}")

            all_rows.append({
                "game_id":        match_id,
                "league_id":      league_id,
                "team_id":        team_id,
                "sub_in_id":      sub_in_id,
                "sub_out_id":     sub_out_id,
                "position_in":    position_in,
                "position_out":   position_out,
                "period":         period,
                "minute":         minute,
                "second":         second,
                "timestamp":      timestamp,
                "current_result": current_result,
            })

    except Exception as e:
        print(f"ERROR on match_id {match_id}: {e}")
        continue

    if (i + 1) % 5 == 0 or (i + 1) == len(valid_match_ids):
        print(f"  Processed {i + 1}/{len(valid_match_ids)} matches...")

# ── Assemble & write ──────────────────────────────────────────────────────
substitutions_df = pd.DataFrame(all_rows, columns=[
    "game_id", "league_id", "team_id",
    "sub_in_id", "sub_out_id",
    "position_in", "position_out",
    "period", "minute", "second", "timestamp",
    "current_result",
])

print(f"\nTotal substitutions found: {len(substitutions_df)}")
print(substitutions_df.head(10).to_string(index=False))

conn.execute("CREATE OR REPLACE TABLE substitutions AS SELECT * FROM substitutions_df")
conn.close()
print(f"\nWritten {len(substitutions_df)} rows to 'substitutions' table in DuckDB")

# ── Sanity checks ─────────────────────────────────────────────────────────
print("\n=== COLUMNS ===")
print(substitutions_df.columns.tolist())

print("\n=== SHAPE ===")
print(substitutions_df.shape)

print("\n=== SAMPLE (5 rows) ===")
print(substitutions_df.head().to_string(index=False))

print("\n=== NULLS ===")
print(substitutions_df.isnull().sum())

print("\n=== DISTINCT position_in ===")
print(sorted(substitutions_df["position_in"].dropna().unique()))

print("\n=== DISTINCT position_out ===")
print(sorted(substitutions_df["position_out"].dropna().unique()))

print("\n=== DISTINCT current_result ===")
print(sorted(substitutions_df["current_result"].unique()))

print("\n=== SUBS PER GAME (expect 3-6 per match) ===")
print(substitutions_df.groupby("game_id").size().describe())

print("\n=== MINUTE DISTRIBUTION ===")
print(substitutions_df["minute"].describe())