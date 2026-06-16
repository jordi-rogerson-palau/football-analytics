import pandas as pd
import duckdb
import json
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

# -----------------------------
# Extract duels from all matches
# -----------------------------

all_duels = []

for _, row in filtered_comps.iterrows():
    competition_id = row["competition_id"]
    season_id      = row["season_id"]
    league_name    = row["competition_name"]

    matches_path = DATA_DIR / "matches" / str(competition_id) / f"{season_id}.json"

    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    print(f"📂 Processing {league_name} — {len(matches)} matches...")

    for match in matches:
        match_id    = match["match_id"]
        events_path = DATA_DIR / "events" / f"{match_id}.json"

        if not events_path.exists():
            print(f"  ⚠️  Missing events file: {match_id}")
            continue

        with open(events_path, "r", encoding="utf-8") as f:
            events = json.load(f)

        # A duel can be signalled by three event types:
        #   - "Duel"             → contested 50/50, tackle attempts, aerial duels
        #   - "Foul Committed"   → duel that resulted in a foul (has card info)
        #   - "Foul Won"         → receiving end of the foul (no card info here)
        # We capture Duel + Foul Committed so we never miss a carded event.
        duel_events = [
            e for e in events
            if e.get("type", {}).get("name") in ("Duel", "Foul Committed")
        ]

        if not duel_events:
            continue

        rows = []
        for e in duel_events:
            event_type    = e.get("type", {}).get("name")
            duel          = e.get("duel", {})
            foul          = e.get("foul_committed", {})
            bad_behaviour = e.get("bad_behaviour", {})   # card on non-foul events (e.g. dissent)

            # --- card resolution -------------------------------------------------
            # Cards live in foul_committed.card  OR  bad_behaviour.card
            card_name = (
                foul.get("card", {}).get("name")
                or bad_behaviour.get("card", {}).get("name")
                or ""
            )

            rows.append({
                "match_id":        match_id,
                "event_id":        e.get("id"),
                "team_id":         e.get("team",   {}).get("id"),
                "player_id":       e.get("player", {}).get("id"),
                "event_type":      event_type,
                # duel sub-type (e.g. "Tackle", "Aerial Lost") — None for fouls
                "duel_type":       duel.get("type",    {}).get("name"),
                "duel_outcome":    duel.get("outcome", {}).get("name"),
                # bool flags
                "foul":            event_type == "Foul Committed",
                "yellow_card":     card_name in ("Yellow Card", "Second Yellow"),
                "red_card":        card_name in ("Red Card",    "Second Yellow"),
            })

        all_duels.append(pd.DataFrame(rows))

# -----------------------------
# Combine & write to DuckDB
# -----------------------------

duels_df = pd.concat(all_duels, ignore_index=True)

con = duckdb.connect(DUCKDB_PATH)
con.register("duels_df", duels_df)

con.execute("""
    CREATE OR REPLACE TABLE duels AS
    SELECT * FROM duels_df
""")

con.close()

print(f"✅ duels table created — {len(duels_df):,} rows.")
print("\nFoul / card distribution:")
print(duels_df[["foul", "yellow_card", "red_card"]].sum().to_string())

print(f"\n✅ duels table created — {len(duels_df):,} rows.")
print("\nFoul / card distribution:")
print(duels_df[["foul", "yellow_card", "red_card"]].sum().to_string())

# --- sanity checks ---

print("\n--- Sanity Checks ---")

# 1. Row counts
print(f"\nTotal duel rows:         {len(duels_df):,}")
print(f"  of which Duel:         {(duels_df['event_type'] == 'Duel').sum():,}")
print(f"  of which Foul:         {(duels_df['event_type'] == 'Foul Committed').sum():,}")

# 2. Foul / card rates (should be plausible: ~25-30 fouls/game, ~3-4 yellows/game)
n_matches = duels_df["match_id"].nunique()
print(f"\nMatches covered:         {n_matches}")
print(f"Fouls per match:         {duels_df['foul'].sum() / n_matches:.1f}  (expect ~25–30)")
print(f"Yellow cards per match:  {duels_df['yellow_card'].sum() / n_matches:.1f}  (expect ~3–4)")
print(f"Red cards per match:     {duels_df['red_card'].sum() / n_matches:.1f}  (expect ~0.1–0.3)")

# 3. No match_id or player_id should be null
print(f"\nNull match_id:           {duels_df['match_id'].isna().sum()}")
print(f"Null player_id:          {duels_df['player_id'].isna().sum()}")

# 4. Sample of carded events — verify card info looks right
print("\nSample yellow card events:")
print(duels_df[duels_df["yellow_card"]][
    ["match_id", "player_id", "event_type", "duel_type", "foul", "yellow_card", "red_card"]
].head(5).to_string(index=False))

print("\nSample red card events:")
red_sample = duels_df[duels_df["red_card"]]
if len(red_sample):
    print(red_sample[
        ["match_id", "player_id", "event_type", "duel_type", "foul", "yellow_card", "red_card"]
    ].head(5).to_string(index=False))
else:
    print("  ⚠️  No red cards found — check bad_behaviour card parsing")

# 5. Duel outcome distribution
print("\nDuel outcome distribution:")
print(duels_df["duel_outcome"].value_counts(dropna=False).head(10).to_string())

# 6. Duel type distribution
print("\nDuel type distribution:")
print(duels_df["duel_type"].value_counts(dropna=False).head(10).to_string())

