import duckdb
import pandas as pd
import altair as alt
import streamlit as st
from pathlib import Path
import os

#st.set_page_config(layout="wide")

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
    temporal_xg = con.execute("SELECT * FROM temporal_xg").df()
    teams_df = con.execute("SELECT team_id, team_name, league_id FROM teams").df()
    con.close()
    return temporal_xg, teams_df

temporal_xg, teams_df = load_data(str(DB_PATH))

LEAGUE_ORDER  = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}
LEAGUE_ID_MAP = {
    11: "La Liga",
    2:  "Premier League",
    12: "Serie A",
    9:  "1. Bundesliga",
    7:  "Ligue 1",
}

# Teams to highlight with a late-goals line (80+)
HIGHLIGHT_TEAMS = {
    "La Liga":       "Real Madrid",
    "Premier League":"Leicester City",
    "1. Bundesliga": "Bayer Leverkusen",
}

temporal_xg["league_name"] = temporal_xg["league_id"].map(LEAGUE_ID_MAP)

# ── League-level aggregation ──────────────────────────
matches_per_league = temporal_xg.groupby("league_name")["match_id"].nunique()

agg = (
    temporal_xg
    .groupby(["league_name", "minute"])
    .agg(total_xg=("shot_statsbomb_xg", "sum"), total_goals=("goal", "sum"))
    .reset_index()
    .sort_values(["league_name", "minute"])
)
agg["n_matches"] = agg["league_name"].map(matches_per_league)
agg["avg_xg"]    = agg["total_xg"]   / agg["n_matches"]
agg["avg_goals"] = agg["total_goals"] / agg["n_matches"]

# Same rolling smoothing as league lines
agg["avg_xg"]    = agg.groupby("league_name")["avg_xg"].transform(
    lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
)
agg["avg_goals"] = agg.groupby("league_name")["avg_goals"].transform(
    lambda x: x.rolling(window=5, center=True, min_periods=1).mean()
)

# ── Team-level aggregation ─────────────────────────────
def build_team_agg(temporal_xg, teams_df, team_name):
    row = teams_df[teams_df["team_name"] == team_name]
    if len(row) == 0:
        return None
    team_id = int(row["team_id"].iloc[0])

    
    team_match_ids = set(
    temporal_xg[temporal_xg["match_id"].isin(
        temporal_xg["match_id"]  
    )]["match_id"].unique()
)

    if not team_match_ids:
        return None

    team_xg   = temporal_xg[temporal_xg["match_id"].isin(team_match_ids)].copy()
    n_matches = team_xg["match_id"].nunique()

    team_agg = (
        team_xg
        .groupby("minute")
        .agg(total_goals=("goal", "sum"))
        .reset_index()
        .sort_values("minute")
    )
    team_agg["avg_goals"] = team_agg["total_goals"] / n_matches

    # ── Same rolling window as league data ──
    team_agg["avg_goals"] = (
        team_agg["avg_goals"]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )
    return team_agg

# Pre-build and cache team aggregations
team_agg_cache = {}
for league, team_name in HIGHLIGHT_TEAMS.items():
    result = build_team_agg(temporal_xg, teams_df, team_name)
    if result is not None:
        team_agg_cache[league] = (team_name, result)

# ── Shared y_max: include both league AND team data ────
team_vals = [
    t_agg["avg_goals"].max()
    for _, (_, t_agg) in team_agg_cache.items()
]
y_max = max(
    agg["avg_xg"].max(),
    agg["avg_goals"].max(),
    *team_vals if team_vals else [0]
) * 1.1

st.markdown("### xG vs Goals per Minute")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each line chart shows how **average goals** (solid line) and **average xG** (dashed line) "
        "are distributed across match minutes for a given league. "
        "**xG (Expected Goals)** is a StatsBomb metric that quantifies the quality of a chance — "
        "every shot attempt carries an xG value regardless of whether it resulted in a goal, "
        "so the xG line represents chance creation while the goals line represents reality. "
        "The **shaded area** between both lines makes it easy to see at a glance whether a league "
        "tends to over- or under-convert its chances at different stages of the game. "
        "The **HT** and **ET** vertical markers indicate half-time (45') and end of regular time (90'), "
        "helping identify if leagues are particularly active in the final minutes or in stoppage time. "
        "For three leagues a **dark dashed team line** appears from minute 80 onwards, showing the "
        "average goals per minute for a highlighted team — useful to compare a specific team's "
        "late-game behaviour against the league average. "
        "The **minute range slider** lets you zoom into any phase of the game, and the "
        "**bottom-right barchart** updates accordingly to show the cumulative goals vs xG "
        "totals within the selected window, making it easy to compare finishing efficiency across leagues."
    )

row1_cols = st.columns(3, gap="large")
row2_cols = st.columns(3, gap="large")
grid      = row1_cols + row2_cols

for idx, league in enumerate(LEAGUE_ORDER):
    with grid[idx]:
        st.empty()

with grid[5]:
    minute_min, minute_max = st.slider("Minute range", 0, 95, (0, 95), 1)

agg_filtered = agg[
    (agg["minute"] >= minute_min) & (agg["minute"] <= minute_max)
].copy()

summary_rows = []
for league in LEAGUE_ORDER:
    d = agg_filtered[agg_filtered["league_name"] == league].copy()
    if len(d) < 2:
        continue
    summary_rows.append({
        "league_name": league,
        "area_goals":  float(d["avg_goals"].values.sum()),
        "area_xg":     float(d["avg_xg"].values.sum()),
    })
summary = pd.DataFrame(summary_rows)


def make_chart(league, data):
    color = LEAGUE_COLORS[league]

    area = (
        alt.Chart(data).mark_area(opacity=0.25, color=color)
        .encode(
            x=alt.X("minute:Q", title="Match Minute",
                    scale=alt.Scale(domain=[minute_min, minute_max]),
                    axis=alt.Axis(tickMinStep=5)),
            y=alt.Y("avg_xg:Q",   scale=alt.Scale(domain=[0, y_max]), title="Avg per Game"),
            y2=alt.Y2("avg_goals:Q")
        )
    )

    goals_line = (
        alt.Chart(data).mark_line(strokeWidth=2.5, color=color, opacity=1.0)
        .encode(
            x=alt.X("minute:Q", scale=alt.Scale(domain=[minute_min, minute_max])),
            y=alt.Y("avg_goals:Q", scale=alt.Scale(domain=[0, y_max])),
            tooltip=[
                alt.Tooltip("minute:Q",    title="Minute"),
                alt.Tooltip("avg_goals:Q", title="Avg Goals", format=".4f"),
                alt.Tooltip("avg_xg:Q",    title="Avg xG",    format=".4f"),
            ]
        )
    )

    xg_line = (
        alt.Chart(data).mark_line(strokeWidth=1.8, color=color, opacity=0.45, strokeDash=[6, 3])
        .encode(
            x=alt.X("minute:Q", scale=alt.Scale(domain=[minute_min, minute_max])),
            y=alt.Y("avg_xg:Q", scale=alt.Scale(domain=[0, y_max])),
            tooltip=[
                alt.Tooltip("minute:Q",    title="Minute"),
                alt.Tooltip("avg_goals:Q", title="Avg Goals", format=".4f"),
                alt.Tooltip("avg_xg:Q",    title="Avg xG",    format=".4f"),
            ]
        )
    )

    layers = [area, goals_line, xg_line]

    # ── Team late-goals line (80+), same smoothing, same y-scale ──
    if league in team_agg_cache:
        team_name, t_agg = team_agg_cache[league]
        t_filtered = t_agg[
            (t_agg["minute"] >= max(minute_min, 80)) &
            (t_agg["minute"] <= minute_max)
        ].copy()

        if len(t_filtered) > 1:
            team_line = (
                alt.Chart(t_filtered)
                .mark_line(strokeWidth=2.5, color="#222222", opacity=0.55,
                           strokeDash=[6, 3])
                .encode(
                    x=alt.X("minute:Q",
                            scale=alt.Scale(domain=[minute_min, minute_max])),
                    y=alt.Y("avg_goals:Q",
                            scale=alt.Scale(domain=[0, y_max])),
                    tooltip=[
                        alt.Tooltip("minute:Q",    title="Minute"),
                        alt.Tooltip("avg_goals:Q",
                                    title=f"{team_name} Avg Goals",
                                    format=".4f"),
                    ]
                )
            )
            # Short label at the first visible point
            label_row = t_filtered.iloc[[0]].copy()
            label_row["label"] = team_name.split()[-1]
            team_label = (
                alt.Chart(label_row)
                .mark_text(align="right", dx=-4, fontSize=6.5,
                           color="#222222", fontWeight="bold", opacity=0.75)
                .encode(
                    x=alt.X("minute:Q"),
                    y=alt.Y("avg_goals:Q"),
                    text="label:N"
                )
            )
            layers += [team_line, team_label]

    if minute_min <= 45 <= minute_max:
        halftime_rule = (
            alt.Chart(pd.DataFrame({"minute": [45]}))
            .mark_rule(color="#777777", strokeDash=[3, 3], strokeWidth=1.2, opacity=0.6)
            .encode(x="minute:Q")
        )
        halftime_label = (
            alt.Chart(pd.DataFrame({"minute": [45], "y": [y_max * 0.98], "label": ["HT"]}))
            .mark_text(align="left", dx=3, fontSize=6.5, color="#777777", opacity=0.8)
            .encode(x="minute:Q", y="y:Q", text="label:N")
        )
        layers += [halftime_rule, halftime_label]

    if minute_min <= 90 <= minute_max:
        fulltime_rule = (
            alt.Chart(pd.DataFrame({"minute": [90]}))
            .mark_rule(color="#777777", strokeDash=[3, 3], strokeWidth=1.2, opacity=0.6)
            .encode(x="minute:Q")
        )
        fulltime_label = (
            alt.Chart(pd.DataFrame({"minute": [90], "y": [y_max * 0.98], "label": ["ET"]}))
            .mark_text(align="left", dx=3, fontSize=6.5, color="#777777", opacity=0.8)
            .encode(x="minute:Q", y="y:Q", text="label:N")
        )
        layers += [fulltime_rule, fulltime_label]

    # Build subtitle only for leagues with a highlighted team
    subtitle = ""
    if league in team_agg_cache:
        team_name, _ = team_agg_cache[league]
        subtitle = f"── {team_name} goals (80'+)"

    return (
        alt.layer(*layers)
        .properties(
            title=alt.TitleParams(
                text=league,
                subtitle=subtitle,
                fontSize=22,
                fontWeight="bold",
                color=color,
                subtitleFontSize=7.5,
                subtitleColor="#555555",
                subtitleFontStyle="italic",
                offset=4,
                anchor="start",
            ),
            width=360, height=260
        )
    )


def make_summary_chart(summary):
    rows = []
    for _, row in summary.iterrows():
        rows.append({"league_name": row["league_name"], "metric": "Goals", "value": row["area_goals"]})
        rows.append({"league_name": row["league_name"], "metric": "xG",    "value": row["area_xg"]})
    df_long = pd.DataFrame(rows)
    y_max_shared = df_long["value"].max() * 1.15

    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("league_name:N", sort=LEAGUE_ORDER, title=None,
                    axis=alt.Axis(labelAngle=-60, labelFontSize=8)),
            y=alt.Y("value:Q", title="Cumulative Avg",
                    scale=alt.Scale(domain=[0, y_max_shared]),
                    axis=alt.Axis(grid=True, gridColor="#eeeeee")),
            color=alt.Color(
                "league_name:N",
                scale=alt.Scale(domain=LEAGUE_ORDER,
                                range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER]),
                legend=None
            ),
            opacity=alt.Opacity(
                "metric:N",
                scale=alt.Scale(domain=["Goals", "xG"], range=[1.0, 0.4]),
                legend=alt.Legend(title="Metric")
            ),
            xOffset=alt.XOffset("metric:N", scale=alt.Scale(domain=["Goals", "xG"])),
            tooltip=[
                alt.Tooltip("league_name:N", title="League"),
                alt.Tooltip("metric:N",      title="Metric"),
                alt.Tooltip("value:Q",       title="Value", format=".2f"),
            ]
        )
        .properties(width=300, height=200)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    return chart


for idx, league in enumerate(LEAGUE_ORDER):
    data  = agg_filtered[agg_filtered["league_name"] == league].copy()
    chart = make_chart(league, data)
    with grid[idx]:
        st.altair_chart(
            chart
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor="#eeeeee"),
            use_container_width=False
        )

with grid[5]:
    if len(summary) > 0:
        st.altair_chart(make_summary_chart(summary))