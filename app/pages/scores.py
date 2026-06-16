import duckdb
import pandas as pd
import altair as alt
import streamlit as st
from pathlib import Path
import os

#st.set_page_config(layout="wide")

# ── Spacing constants from STREAMLIT_STYLE_CONSTANTS ──
st.markdown(
    """
    <style>
        #MainMenu        { visibility: hidden; }
        header           { visibility: hidden; }
        footer           { visibility: hidden; }
        .block-container {
            padding-top:  0rem  !important;
            margin-top:  -1rem  !important;
            padding-left: 1rem  !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"


@st.cache_data
def load_data(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    match_scores = con.execute("SELECT * FROM match_scores").df()
    matches = con.execute("SELECT * FROM matches").df()
    teams = con.execute("SELECT * FROM teams").df()
    con.close()
    return match_scores, matches, teams

match_scores, matches, teams = load_data(str(DB_PATH))

name_map   = teams.set_index("team_id")["team_name"]
league_map = teams.drop_duplicates("league_id").set_index("league_id")["league_name"]

df = match_scores.merge(matches, on="match_id", how="inner")
df["home_team"]   = df["home_team_id"].map(name_map)
df["away_team"]   = df["away_team_id"].map(name_map)
df = df.dropna(subset=["home_team", "away_team"])
df["league_name"] = df["league_id"].map(league_map)
df["result"]      = df.apply(
    lambda r: "Home Win" if r.home_score > r.away_score
              else ("Draw" if r.home_score == r.away_score else "Away Win"),
    axis=1
)
df["goal_diff"] = abs(df["home_score"] - df["away_score"])

LEAGUE_ORDER  = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

CHART_W = 210
CHART_H = 195

st.markdown("### Home vs Away Scorelines")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each scatter plot shows the **distribution of final scorelines** for a given league. "
        "The **x-axis** represents away team goals and the **y-axis** represents home team goals, "
        "so any point above the diagonal dashed line is a home win, on the line is a draw, "
        "and below the line is an away win. "
        "**Dot size** encodes how frequently that exact scoreline occurred — larger dots mean more "
        "games ended with that result. "
        "The **🏠 / ═ / ✈ percentages** above each chart show the overall split of home wins, draws "
        "and away wins for that league, giving a quick read on home advantage and competitive balance. "
        "Points concentrated in the top-left area indicate dominant home performances; "
        "a more even spread across both sides of the diagonal reflects a more competitive and "
        "unpredictable league. "
        "The **min goal difference slider** removes low-margin results (draws and 1-goal games) "
        "to isolate more decisive outcomes and make the home vs away asymmetry easier to see "
        "without clutter along the diagonal."
    )

row1_cols = st.columns(3, gap="large")
row2_cols = st.columns(3, gap="large")
grid      = row1_cols + row2_cols

for idx, league in enumerate(LEAGUE_ORDER):
    with grid[idx]:
        st.empty()

max_diff = int(df["goal_diff"].max())

with grid[5]:
    st.markdown("#### Filters")
    min_diff = st.slider("Min goal difference", 0, max_diff, 0, 1)

df_filtered = df[df["goal_diff"] >= min_diff].copy()

agg = (
    df_filtered
    .groupby(["league_name", "home_score", "away_score", "result"])
    .size()
    .reset_index(name="count")
)

summary = (
    df_filtered.groupby(["league_name", "result"])
    .size()
    .reset_index(name="n")
    .merge(df_filtered.groupby("league_name").size().reset_index(name="total"), on="league_name")
)
summary["pct"] = (summary["n"] / summary["total"] * 100).round(1)

max_goals = int(max(agg["home_score"].max(), agg["away_score"].max())) if len(agg) else 8
global_max_count = int(agg["count"].max()) if len(agg) else 1
diag      = pd.DataFrame({"x": [0, max_goals], "y": [0, max_goals]})

def make_chart(league):
    color = LEAGUE_COLORS[league]
    data  = agg[agg["league_name"] == league].copy()
    s     = summary[summary["league_name"] == league].set_index("result")["pct"]

    hw = s.get("Home Win", 0)
    d  = s.get("Draw",     0)
    aw = s.get("Away Win", 0)

    stats_data = pd.DataFrame([{"label": f"🏠 {hw}%    ═ {d}%    ✈ {aw}%"}])

    stats_text = (
        alt.Chart(stats_data)
        .mark_text(align="center", baseline="middle", fontSize=11, color="#444444")
        .encode(text="label:N")
        .properties(width=CHART_W, height=16)
    )

    diag_line = (
        alt.Chart(diag)
        .mark_line(color="#cccccc", strokeDash=[4, 4], strokeWidth=1)
        .encode(x=alt.X("x:Q"), y=alt.Y("y:Q"))
    )

    scatter = (
        alt.Chart(data)
        .mark_circle(opacity=0.85, color=color)
        .encode(
            x=alt.X("away_score:Q", title="Away Goals",
                    scale=alt.Scale(domain=[-0.5, max_goals + 0.5]),
                    axis=alt.Axis(tickMinStep=1, labelFontSize=7, titleFontSize=8)),
            y=alt.Y("home_score:Q", title="Home Goals",
                    scale=alt.Scale(domain=[-0.5, max_goals + 0.5]),
                    axis=alt.Axis(tickMinStep=1, labelFontSize=7, titleFontSize=8)),
            size=alt.Size("count:Q", title="# Matches",
                          scale=alt.Scale(domain=[0, global_max_count], range=[20, 400]), legend=None),
            tooltip=[
                alt.Tooltip("home_score:Q", title="Home Goals"),
                alt.Tooltip("away_score:Q", title="Away Goals"),
                alt.Tooltip("result:N",     title="Result"),
                alt.Tooltip("count:Q",      title="# Matches"),
            ]
        )
    )

    return (
        alt.vconcat(
            stats_text,
            (diag_line + scatter).properties(
                title=alt.TitleParams(league, fontSize=22, fontWeight="bold", color=color),
                width=CHART_W,
                height=CHART_H
            ),
            spacing=2
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#eeeeee")
    )

for idx, league in enumerate(LEAGUE_ORDER):
    with grid[idx]:
        st.altair_chart(make_chart(league), use_container_width=False)

with grid[5]:
    pass