import pandas as pd
import numpy as np
import altair as alt
import duckdb
import streamlit as st
from pathlib import Path
import os

#st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

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

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

LEAGUE_ORDER = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

# -------------------------------------------------------
# Data loading — cached, one connection
# -------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data(db_path: str):
    con         = duckdb.connect(db_path, read_only=True)
    temporal_xg = con.execute("SELECT match_id, minute, goal, scorer_id, home_score, away_score FROM temporal_xg").df()
    subs        = con.execute("SELECT game_id, team_id, sub_in_id FROM substitutions").df()
    teams       = con.execute("SELECT team_id, team_name, league_name FROM teams").df()
    matches     = con.execute("SELECT match_id, home_team_id, away_team_id FROM matches").df()
    players     = con.execute("SELECT player_id, team_id FROM players").df()
    passes      = con.execute("SELECT match_id, team_id, player_id, pass_successful, end_x FROM passes").df()
    con.close()
    return temporal_xg, subs, teams, matches, players, passes


# -------------------------------------------------------
# Precompute result-changing goals for ALL leagues
# Returns one df with all leagues — filter at render time
# -------------------------------------------------------
@st.cache_data(show_spinner="Preparing data…")
def precompute(temporal_xg, subs, teams, matches, players, passes):

    all_team_ids = set(teams["team_id"].tolist())

    # ── Per-team player sets ───────────────────────────
    sub_players = subs[["team_id", "sub_in_id"]].rename(columns={"sub_in_id": "player_id"})
    reg_players = players[["team_id", "player_id"]].copy()
    reg_players["player_id"] = reg_players["player_id"].astype("Int64")
    all_team_players = pd.concat([sub_players, reg_players]).drop_duplicates()

    # ── Sub lookup set: (match_id, player_id) ─────────
    subs_lookup = (
        subs[["game_id", "sub_in_id", "team_id"]]
        .rename(columns={"game_id": "match_id", "sub_in_id": "player_id"})
        .drop_duplicates()
    )
    sub_set = set(zip(subs_lookup["match_id"], subs_lookup["player_id"]))

    # ── Match → team + is_home ─────────────────────────
    home = matches[["match_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"})
    home["is_home"] = True
    away = matches[["match_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"})
    away["is_home"] = False
    match_team = pd.concat([home, away])

    # ── Goals — filter to known scorers ───────────────
    goals = temporal_xg[
        (temporal_xg["goal"] == 1) &
        (temporal_xg["scorer_id"].notna())
    ].copy()
    goals["scorer_id"] = goals["scorer_id"].astype(int)
    goals = goals.merge(match_team, on="match_id", how="inner")
    goals = goals.merge(
        all_team_players.rename(columns={"player_id": "scorer_id"}),
        on=["team_id", "scorer_id"], how="inner"
    )

    # ── Score before goal → situation ─────────────────
    goals["home_before"] = np.where(goals["is_home"], goals["home_score"] - 1, goals["home_score"])
    goals["away_before"] = np.where(goals["is_home"], goals["away_score"],     goals["away_score"] - 1)
    goals["team_before"] = np.where(goals["is_home"], goals["home_before"], goals["away_before"])
    goals["opp_before"]  = np.where(goals["is_home"], goals["away_before"], goals["home_before"])
    goals["situation_before"] = np.select(
        [goals["team_before"] < goals["opp_before"],
         goals["team_before"] == goals["opp_before"]],
        ["Losing", "Drawing"], default="Winning"
    )

    rc = goals[
        (goals["situation_before"].isin(["Losing", "Drawing"])) &
        (goals["minute"] >= 45)
    ].copy().reset_index(drop=True)

    # ── Assist proxy: highest end_x pass per match+team
    good_passes = passes[passes["pass_successful"] == True].copy()
    assist_proxy = (
        good_passes
        .sort_values("end_x", ascending=False)
        .groupby(["match_id", "team_id"])
        .first()
        .reset_index()
        [["match_id", "team_id", "player_id"]]
        .rename(columns={"player_id": "assister_id"})
    )
    rc = rc.merge(assist_proxy, on=["match_id", "team_id"], how="left")

    # ── Tag sub involvement ────────────────────────────
    rc["scorer_is_sub"] = [
        (mid, int(sid)) in sub_set
        for mid, sid in zip(rc["match_id"], rc["scorer_id"])
    ]
    rc["assister_is_sub"] = [
        (mid, int(aid)) in sub_set if pd.notna(aid) else False
        for mid, aid in zip(rc["match_id"], rc["assister_id"])
    ]

    # ── Join league name ───────────────────────────────
    rc = rc.merge(teams[["team_id", "team_name", "league_name"]], on="team_id", how="left")

    return rc


# -------------------------------------------------------
# Build aggregation for a given league + involvement filter
# -------------------------------------------------------
def build_agg(rc: pd.DataFrame, league: str, involvement: str) -> pd.DataFrame:
    lg = rc[rc["league_name"] == league].copy()

    # Apply involvement filter to define "sub_involved"
    if involvement == "Scorer only":
        lg["sub_involved"] = lg["scorer_is_sub"]
    elif involvement == "Assister only":
        lg["sub_involved"] = lg["assister_is_sub"]
    else:  # Both (scorer OR assister)
        lg["sub_involved"] = lg["scorer_is_sub"] | lg["assister_is_sub"]

    total = lg.groupby("team_id").size().reset_index(name="total_goals")
    sub_g = lg[lg["sub_involved"]].groupby("team_id").size().reset_index(name="sub_goals")

    teams_lg = rc[rc["league_name"] == league][["team_id", "team_name"]].drop_duplicates()

    agg = (
        total
        .merge(sub_g,    on="team_id", how="left")
        .merge(teams_lg, on="team_id", how="left")
    )
    agg["sub_goals"]     = agg["sub_goals"].fillna(0).astype(int)
    agg["non_sub_goals"] = agg["total_goals"] - agg["sub_goals"]
    agg["sub_pct"]       = (agg["sub_goals"] / agg["total_goals"] * 100).round(1)
    return agg


# -------------------------------------------------------
# Build Altair chart
# -------------------------------------------------------
def make_chart(agg: pd.DataFrame, league: str, sort_by: str) -> alt.Chart:
    base_color = LEAGUE_COLORS[league]

    sort_col   = "sub_pct" if sort_by == "% Sub involvement" else "total_goals"
    agg        = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
    team_order = agg["team_name"].tolist()

    long = pd.melt(
        agg,
        id_vars=["team_name", "total_goals", "sub_pct"],
        value_vars=["non_sub_goals", "sub_goals"],
        var_name="type", value_name="goals"
    )
    long["label"] = long["type"].map({
        "non_sub_goals": "No substitute involved",
        "sub_goals":     "Substitute involved",
    })

    bars = (
        alt.Chart(long)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("team_name:N", sort=team_order, title=None,
                    axis=alt.Axis(
                        labelAngle=-38,
                        labelFontSize=9,       # kept small so all names fit
                        labelLimit=200,        # allow full name length
                        labelOverlap=False,    # never skip a label
                    )),
            y=alt.Y("goals:Q", title="Result-changing goals (2nd half)",
                    axis=alt.Axis(tickMinStep=1, labelFontSize=12.4, titleFontSize=13.8)),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=["No substitute involved", "Substitute involved"],
                    range=[base_color, "#333333"]
                ),
                legend=alt.Legend(title="", orient="bottom",
                                  labelFontSize=12.4, titleFontSize=12.4,
                                  direction="horizontal", titleOrient="left")
            ),
            order=alt.Order("type:N", sort="descending"),
            tooltip=[
                alt.Tooltip("team_name:N",   title="Team"),
                alt.Tooltip("total_goals:Q", title="Total result-changing goals"),
                alt.Tooltip("goals:Q",       title="Goals (segment)"),
                alt.Tooltip("sub_pct:Q",     title="% sub involved", format=".1f"),
            ]
        )
    )

    # % label inside the dark (sub) segment, centred vertically within it
    pct_df = agg[agg["sub_goals"] > 0].copy()
    pct_df["pct_label"]   = pct_df["sub_pct"].apply(lambda x: f"{x:.0f}%")
    pct_df["label_y"]     = pct_df["sub_goals"] / 2
    text = (
        alt.Chart(pct_df)
        .mark_text(dy=0, fontSize=11, color="white", fontWeight="bold")
        .encode(
            x=alt.X("team_name:N", sort=team_order),
            y=alt.Y("label_y:Q"),
            text=alt.Text("pct_label:N"),
        )
    )

    return (
        (bars + text)
        .properties(width=938, height=469)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


# -------------------------------------------------------
# App
# -------------------------------------------------------
temporal_xg, subs, teams, matches, players, passes = load_data(str(DB_PATH))
rc = precompute(temporal_xg, subs, teams, matches, players, passes)

# ── Session state: shared with subs_min / subs_pos ────
if "selected_league" not in st.session_state:
    st.session_state.selected_league = LEAGUE_ORDER[0]

st.markdown("### Result-Changing Goals by Team — 2nd Half")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "This chart focuses on **result-changing goals scored in the second half** — "
        "defined as goals scored from a Losing or Drawing situation from minute 45 onwards, "
        "meaning goals that either equalised or turned the game around. "
        "Each bar represents a team's total count of such goals, split into two segments: "
        "the **league-coloured portion** shows goals with no substitute involved, "
        "and the **dark portion** shows goals where a substitute was involved. "
        "The **percentage label** above each bar indicates what share of that team's "
        "result-changing goals had substitute involvement. "
        "**Substitute involvement** is defined as the substitute being either the scorer or "
        "the assister — the assist here is approximated as the most advanced successful pass "
        "by the team in that match, which is a proxy rather than an exact assist. "
        "Use the **Sub involvement filter** to narrow down to scorer-only or assister-only "
        "contributions, and the **Sort by** selector to rank teams either by total volume "
        "or by percentage of substitute involvement."
    )

# Tighter ratio so chart fills more of the screen
chart_col, sel_col = st.columns([6, 1], gap="medium")

with sel_col:
    st.markdown("**League**")
    new_league = st.selectbox("League", LEAGUE_ORDER,
                              index=LEAGUE_ORDER.index(st.session_state.selected_league),
                              label_visibility="collapsed")
    if new_league != st.session_state.selected_league:
        st.session_state.selected_league = new_league
        st.rerun()
    selected_league = st.session_state.selected_league

    st.markdown("**Sort by**")
    sort_by = st.selectbox(
        "Sort by",
        ["Total result-changing goals", "% Sub involvement"],
        label_visibility="collapsed"
    )

    st.markdown("**Sub involvement**")
    involvement = st.selectbox(
        "Involvement",
        ["Scorer or assister", "Scorer only", "Assister only"],
        label_visibility="collapsed"
    )


agg = build_agg(rc, selected_league, involvement)

with chart_col:
    st.altair_chart(
        make_chart(agg, selected_league, sort_by),
        use_container_width=False,
    )