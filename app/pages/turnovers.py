import os
import duckdb
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

LEAGUE_ORDER  = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

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

st.markdown("### Turnovers Leading to the Other Half")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "This chart shows two metrics per league, both averaged per game: "
        "**Total Turnovers** (faded bar) and **Turnovers Leading to the Opposite Half** (solid bar). "
        "A **turnover** is a change of possession where play continues without the ball going out of bounds "
        "or the referee stopping the game — it is therefore a measure of in-game freneticism. "
        "The **opposite half** metric counts only those turnovers where the ball then crossed the halfway "
        "line, meaning the previously attacking team was immediately forced into a defensive position; "
        "a high value here indicates a more open, end-to-end and frenetic style of play. "
        "Use the **turnover type filter** to isolate specific ways possession is lost — note that "
        "*dispossessed* means a player had the ball stolen without attempting a dribble, while "
        "*interception* means an opponent actively cut off a pass trajectory, and "
        "*clearance* means a player kicked the ball away with no passing option available. "
        "The **order leagues by** selector re-sorts the bars by either total volume or opposite-half "
        "impact, which can reveal different league rankings depending on which dimension of "
        "freneticism you want to compare."
    )

@st.cache_data
def load_data():
    con = duckdb.connect(str(DB_PATH), read_only = True)
    sequences = con.execute("SELECT * FROM sequences").df()
    teams     = con.execute("SELECT team_id, league_name FROM teams").df()
    con.close()

    sequences_with_league = sequences.merge(teams[["team_id", "league_name"]], on="team_id", how="left")
    sequences_sorted = sequences_with_league.sort_values(["match_id", "possession"]).reset_index(drop=True)
    sequences_sorted["next_start_x"] = sequences_sorted.groupby("match_id")["start_x"].shift(-1)
    sequences_sorted["next_team_id"]  = sequences_sorted.groupby("match_id")["team_id"].shift(-1)
    return sequences_sorted

sequences_sorted = load_data()

ALL_TURNOVER_TYPES = [
    "failed_pass", "interception", "failed_dribble",
    "lost_duel", "miscontrol", "dispossessed",
    "clearance", "block"
]

if "selected_types" not in st.session_state:
    st.session_state["selected_types"] = ALL_TURNOVER_TYPES
if "sort_criterion" not in st.session_state:
    st.session_state["sort_criterion"] = "Total Turnovers"

selected_types = st.session_state["selected_types"]
sort_criterion = st.session_state["sort_criterion"]

if not selected_types:
    st.warning("Select at least one turnover type.")
    st.stop()

active_turnovers = sequences_sorted[sequences_sorted["end_type"].isin(selected_types)].copy()
active_turnovers["opposite_half"] = (
    (active_turnovers["next_team_id"] != active_turnovers["team_id"]) &
    (
        ((active_turnovers["end_x"] < 60) & (active_turnovers["next_start_x"] >= 60)) |
        ((active_turnovers["end_x"] >= 60) & (active_turnovers["next_start_x"] < 60))
    )
)

per_match = (
    active_turnovers.groupby(["match_id", "league_name"])
    .agg(
        total_turnovers=("sequence_id", "count"),
        opposite_half_turnovers=("opposite_half", "sum")
    )
    .reset_index()
)

agg = (
    per_match.groupby("league_name")
    .agg(
        total_per_game=("total_turnovers", "mean"),
        opposite_half_per_game=("opposite_half_turnovers", "mean")
    )
    .reset_index()
)

sort_col    = "total_per_game" if sort_criterion == "Total Turnovers" else "opposite_half_per_game"
league_order = agg.sort_values(sort_col, ascending=False)["league_name"].tolist()

agg_long = pd.concat([
    agg[["league_name", "total_per_game"]].rename(columns={"total_per_game": "value"}).assign(metric="Total Turnovers"),
    agg[["league_name", "opposite_half_per_game"]].rename(columns={"opposite_half_per_game": "value"}).assign(metric="Led to Opposite Half"),
], ignore_index=True)

chart = alt.Chart(agg_long).mark_bar().encode(
    y=alt.Y("league_name:N", sort=league_order, title=None, axis=alt.Axis(labelFontSize=11)),
    x=alt.X("value:Q", title="Turnovers per Game", axis=alt.Axis(labelFontSize=10)),
    color=alt.Color(
        "league_name:N",
        scale=alt.Scale(domain=LEAGUE_ORDER, range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER]),
        legend=None
    ),
    opacity=alt.Opacity(
        "metric:N",
        scale=alt.Scale(domain=["Total Turnovers", "Led to Opposite Half"], range=[0.35, 0.9]),
        legend=alt.Legend(title="Metric", orient="bottom", offset = 10)
    ),
    yOffset=alt.YOffset("metric:N",
                        scale=alt.Scale(domain=["Total Turnovers", "Led to Opposite Half"])),
    tooltip=[
        alt.Tooltip("league_name:N", title="League"),
        alt.Tooltip("metric:N",      title="Metric"),
        alt.Tooltip("value:Q",       title="Per Game", format=".1f")
    ]
).properties(
    width=900,
    height=380,
    title=f"Ordered by: {sort_criterion}"
).configure_view(
    stroke=None
).configure_axis(
    grid=False
)

st.altair_chart(chart, use_container_width=True)

st.markdown("---")
col_left, col_right = st.columns([2, 1])
with col_left:
    st.multiselect(
        "Turnover types",
        options=ALL_TURNOVER_TYPES,
        format_func=lambda x: x.replace("_", " ").title(),
        key="selected_types"
    )
with col_right:
    st.selectbox(
        "Order leagues by",
        options=["Total Turnovers", "Led to Opposite Half"],
        key="sort_criterion"
    )