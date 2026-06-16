"""
Run this script ONCE from your project root before starting Streamlit.
It creates two pre-aggregated tables in statsbomb_2015_2016.duckdb:

    pass_bin_agg  — 60 rows  (5 leagues × 2 halves × 6 distance bins)
    pass_metrics  — 5 rows   (one per league)

These replace the 1,777,390-row league_passes scan that barchart_pass.py
previously did on every page load.

Usage:
    python create_pass_tables.py
"""

from pathlib import Path
import table_creations.duck_check as duck_check

# ── Locate the database ───────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent.parent   # wherever you put this script
DB_PATH      = SCRIPT_DIR / "statsbomb_2015_2016.duckdb"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found at {DB_PATH}")

con = duck_check.connect(str(DB_PATH))

print("Connected to", DB_PATH)

# ── 1. pass_bin_agg ───────────────────────────────────
print("Creating pass_bin_agg …", end=" ", flush=True)

con.execute("DROP TABLE IF EXISTS pass_bin_agg")
con.execute("""
CREATE TABLE pass_bin_agg AS
WITH successful AS (
    SELECT
        league_name,
        match_id,
        pass_length,
        CASE WHEN start_x < 60 THEN 'Own Half' ELSE 'Opposition Half' END AS half,
        CASE
            WHEN pass_length <  10 THEN '0–10'
            WHEN pass_length <  20 THEN '10–20'
            WHEN pass_length <  30 THEN '20–30'
            WHEN pass_length <  40 THEN '30–40'
            WHEN pass_length <  50 THEN '40–50'
            WHEN pass_length <  60 THEN '50–60'
        END AS bin_label
    FROM league_passes
    WHERE pass_successful = true
      AND pass_length  IS NOT NULL
      AND pass_length  <  60
      AND start_x      IS NOT NULL
),
games AS (
    SELECT league_name, COUNT(DISTINCT match_id) AS num_games
    FROM league_passes
    GROUP BY league_name
),
agg AS (
    SELECT
        s.league_name,
        s.half,
        s.bin_label,
        COUNT(*)           AS raw_count,
        AVG(s.pass_length) AS avg_length
    FROM successful s
    GROUP BY s.league_name, s.half, s.bin_label
)
SELECT
    a.league_name,
    a.half,
    a.bin_label,
    a.raw_count * 1.0 / g.num_games AS passes_per_game,
    a.avg_length
FROM agg a
JOIN games g ON a.league_name = g.league_name
ORDER BY a.league_name, a.half, a.bin_label
""")

n = con.execute("SELECT COUNT(*) FROM pass_bin_agg").fetchone()[0]
print(f"done — {n} rows")

# ── 2. pass_metrics ───────────────────────────────────
print("Creating pass_metrics …", end=" ", flush=True)

con.execute("DROP TABLE IF EXISTS pass_metrics")
con.execute("""
CREATE TABLE pass_metrics AS
WITH games AS (
    SELECT league_name, COUNT(DISTINCT match_id) AS num_games
    FROM league_passes
    GROUP BY league_name
)
SELECT
    lp.league_name,
    COUNT(*)                                                               * 1.0
        / g.num_games                                              AS total_passes_per_game,
    AVG(lp.pass_successful::INTEGER)                               * 100   AS completion_rate,
    SUM(CASE WHEN lp.end_x > lp.start_x THEN 1 ELSE 0 END)        * 1.0
        / g.num_games                                              AS prog_passes_per_game,
    SUM(CASE WHEN lp.end_x > lp.start_x AND lp.pass_successful
             THEN 1 ELSE 0 END)                                    * 100.0
        / NULLIF(SUM(CASE WHEN lp.end_x > lp.start_x
                          THEN 1 ELSE 0 END), 0)                   AS prog_completion
FROM league_passes lp
JOIN games g ON lp.league_name = g.league_name
GROUP BY lp.league_name, g.num_games
ORDER BY lp.league_name
""")

n = con.execute("SELECT COUNT(*) FROM pass_metrics").fetchone()[0]
print(f"done — {n} rows")

# ── Verify ────────────────────────────────────────────
print("\npass_bin_agg sample:")
print(con.execute("SELECT * FROM pass_bin_agg LIMIT 6").df().to_string(index=False))

print("\npass_metrics:")
print(con.execute("SELECT * FROM pass_metrics").df().to_string(index=False))

con.close()
print("\nAll done. You can now run Streamlit.")