import os
import duckdb
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from pathlib import Path

# -------------------------------------------------------
# Path setup
# -------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DUCKDB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
POSITION_GROUP_MAP = {
    "Goalkeeper":                 "Goalkeeper",
    "Center Back":                "Center Backs",
    "Left Center Back":           "Center Backs",
    "Right Center Back":          "Center Backs",
    "Left Back":                  "Left Backs",
    "Left Wing Back":             "Left Backs",
    "Right Back":                 "Right Backs",
    "Right Wing Back":            "Right Backs",
    "Left Defensive Midfield":    "Defensive Midfielders",
    "Center Defensive Midfield":  "Defensive Midfielders",
    "Right Defensive Midfield":   "Defensive Midfielders",
    "Left Center Midfield":       "Center Midfielders",
    "Center Midfield":            "Center Midfielders",
    "Right Center Midfield":      "Center Midfielders",
    "Left Midfield":              "Left Wingers",
    "Right Midfield":             "Right Wingers",
    "Left Attacking Midfield":    "Attacking Midfielders",
    "Center Attacking Midfield":  "Attacking Midfielders",
    "Right Attacking Midfield":   "Attacking Midfielders",
    "Left Wing":                  "Left Wingers",
    "Right Wing":                 "Right Wingers",
    "Left Center Forward":        "Strikers",
    "Right Center Forward":       "Strikers",
    "Center Forward":             "Strikers",
}

GROUP_ORDER = [
    "Goalkeeper",
    "Center Backs",
    "Left Backs",
    "Right Backs",
    "Defensive Midfielders",
    "Center Midfielders",
    "Left Wingers",
    "Right Wingers",
    "Attacking Midfielders",
    "Strikers",
]

SITUATIONS    = ["Winning", "Drawing", "Losing"]
LEAGUE_ORDER  = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

# row-1 only barcharts: increased 10% from previous
BAR_W = 350
BAR_H = 205

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------
@st.cache_data(show_spinner="Loading substitution data…")
def load_data(db_path: str):
    con     = duckdb.connect(DUCKDB_PATH, read_only=True)
    subs    = con.execute("SELECT * FROM substitutions").df()
    teams   = con.execute("SELECT team_id, team_name, league_name FROM teams").df()
    players = con.execute("SELECT player_id, player_name, team_id, position FROM players").df()
    con.close()
    players["player_id"] = players["player_id"].astype("Int64")
    return subs, teams, players


# -------------------------------------------------------
# Precompute
# -------------------------------------------------------
@st.cache_data(show_spinner="Preparing data for all teams…")
def precompute_all(subs: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame):

    def get_situation(row):
        try:
            home_score, away_score = map(int, row["current_result"].split("-"))
        except (ValueError, AttributeError):
            return None
        is_home    = row["home_team_id"] == row["team_id"]
        team_score = home_score if is_home else away_score
        opp_score  = away_score if is_home else home_score
        if team_score > opp_score:    return "Winning"
        elif team_score == opp_score: return "Drawing"
        else:                         return "Losing"

    # ---- Barchart data ----
    enriched = subs.copy()
    enriched["position_group"] = enriched["position_in"].map(POSITION_GROUP_MAP)
    #enriched["situation"]      = enriched.apply(get_situation, axis=1)

    #vectorized version of get_situation
    scores = enriched["current_result"].str.split("-", expand=True)
    enriched["home_score"] = pd.to_numeric(scores[0], errors="coerce")
    enriched["away_score"] = pd.to_numeric(scores[1], errors="coerce")
    is_home = enriched["home_team_id"] == enriched["team_id"]
    enriched["team_score"] = np.where(is_home, enriched["home_score"], enriched["away_score"])
    enriched["opp_score"]  = np.where(is_home, enriched["away_score"], enriched["home_score"])
    enriched["situation"]  = np.select(
        [enriched["team_score"] > enriched["opp_score"],
        enriched["team_score"] == enriched["opp_score"]],
        ["Winning", "Drawing"], default="Losing"
    )

    enriched = enriched.dropna(subset=["situation", "position_group"])

    all_combos = pd.MultiIndex.from_product(
        [subs["team_id"].unique(), SITUATIONS, GROUP_ORDER],
        names=["team_id", "situation", "position_group"]
    )
    counts = (
        enriched
        .groupby(["team_id", "situation", "position_group"])
        .size()
        .reset_index(name="count")
    )
    bar_df = (
        counts
        .set_index(["team_id", "situation", "position_group"])
        .reindex(all_combos, fill_value=0)
        .reset_index()
        .merge(teams[["team_id", "league_name"]], on="team_id", how="left")
    )

    # ---- Minutes data ----
    all_games = subs[["team_id", "game_id"]].drop_duplicates()

    subbed_off = (
        subs[["team_id", "game_id", "sub_out_id", "sub_out_name", "position_out", "minute"]]
        .rename(columns={
            "sub_out_id":   "player_id",
            "sub_out_name": "player_name",
            "position_out": "position",
            "minute":       "minutes_played",
        })
    )

    subbed_on = subs[["team_id", "game_id", "sub_in_id", "sub_in_name", "position_in", "minute"]].copy()
    subbed_on["minutes_played"] = 90 - subbed_on["minute"]
    subbed_on = subbed_on.rename(columns={
        "sub_in_id":   "player_id",
        "sub_in_name": "player_name",
        "position_in": "position",
    }).drop(columns="minute")

    touched = pd.concat([
        subs[["team_id", "game_id", "sub_out_id"]].rename(columns={"sub_out_id": "player_id"}),
        subs[["team_id", "game_id", "sub_in_id"]].rename(columns={"sub_in_id":  "player_id"}),
    ])

    team_players = players[["player_id", "player_name", "team_id", "position"]].copy()
    team_players["player_id"] = team_players["player_id"].astype("Int64")

    full90_parts = []
    for _, row in all_games.iterrows():
        tid, gid    = row["team_id"], row["game_id"]
        touched_ids = touched[
            (touched["team_id"] == tid) & (touched["game_id"] == gid)
        ]["player_id"].values
        squad       = team_players[team_players["team_id"] == tid].copy()
        full90      = squad[~squad["player_id"].isin(touched_ids)].copy()
        full90["game_id"]        = gid
        full90["minutes_played"] = 90
        full90_parts.append(full90)

    full90_df   = pd.concat(full90_parts, ignore_index=True) if full90_parts else pd.DataFrame()
    all_minutes = pd.concat([subbed_off, subbed_on, full90_df], ignore_index=True)
    all_minutes["player_id"] = all_minutes["player_id"].astype("Int64")
    all_minutes["position"]  = all_minutes["position"].map(POSITION_GROUP_MAP).fillna(
        all_minutes["position"]
    )

    minutes_df = (
        all_minutes
        .groupby(["team_id", "player_id", "player_name", "position"], dropna=False)
        .agg(total_minutes=("minutes_played", "sum"))
        .reset_index()
    )

    return bar_df, minutes_df


# -------------------------------------------------------
# Chart builders
# -------------------------------------------------------
def make_bar_charts(bar_df: pd.DataFrame, team_id: int, league_name: str) -> alt.Chart:
    team_df   = bar_df[bar_df["team_id"] == team_id].copy()
    y_max     = int(team_df["count"].max()) + 1
    bar_color = LEAGUE_COLORS.get(league_name, "#4C72B0")

    charts = []
    for sit in SITUATIONS:
        sit_df = team_df[team_df["situation"] == sit].copy()
        sit_df["count_display"] = sit_df["count"].clip(lower=0.01)
        n_subs = int(sit_df["count"].sum())

        chart = (
            alt.Chart(sit_df)
            .mark_bar(color=bar_color, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X(
                    "position_group:O",
                    sort=GROUP_ORDER,
                    title=None,
                    scale=alt.Scale(domain=GROUP_ORDER, padding=0.2),
                    axis=alt.Axis(
                        labelAngle=-45,
                        labelFontSize=6.6,
                        labelLimit=200,
                        labelOverlap=False,
                    ),
                ),
                y=alt.Y(
                    "count_display:Q",
                    title="Substitutions",
                    scale=alt.Scale(domain=[0, y_max]),
                    axis=alt.Axis(tickMinStep=1, labelFontSize=6.6, titleFontSize=7.7),
                ),
                tooltip=[
                    alt.Tooltip("position_group:O", title="Position"),
                    alt.Tooltip("count:Q",          title="Count"),
                ],
            )
            .properties(
                width=BAR_W,
                height=BAR_H,
                title=alt.TitleParams(
                    text=f"{sit}  (n={n_subs})",
                    fontSize=8.8,
                    fontWeight="bold",
                    color="#333333",
                    offset=5,
                    anchor="start",
                ),
            )
        )
        charts.append(chart)

    return (
        alt.hconcat(*charts, spacing=24)
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )


def make_subbed_in_table(subs: pd.DataFrame, minutes_df: pd.DataFrame,
                          players: pd.DataFrame, team_id: int) -> pd.DataFrame:
    counts = (
        subs[subs["team_id"] == team_id]
        .groupby("sub_in_id")
        .agg(Player=("sub_in_name", "first"), **{"Times Subbed In": ("sub_in_id", "count")})
        .reset_index()
        .rename(columns={"sub_in_id": "player_id"})
    )
    counts["player_id"] = counts["player_id"].astype("Int64")

    mins = (
        minutes_df[minutes_df["team_id"] == team_id]
        .groupby("player_id", as_index=False)["total_minutes"].sum()
        .rename(columns={"total_minutes": "Total Mins"})
    )

    pos = players[players["team_id"] == team_id][["player_id", "position"]].copy()
    pos["player_id"] = pos["player_id"].astype("Int64")
    pos["position"]  = pos["position"].map(POSITION_GROUP_MAP).fillna(pos["position"])
    pos = pos.drop_duplicates("player_id")

    result = (
        counts
        .merge(mins, on="player_id", how="left")
        .merge(pos,  on="player_id", how="left")
        .rename(columns={"position": "Position"})
        .sort_values("Times Subbed In", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )
    return result[["Player", "Position", "Times Subbed In", "Total Mins"]]


def make_subbed_out_table(subs: pd.DataFrame, minutes_df: pd.DataFrame,
                           players: pd.DataFrame, team_id: int) -> pd.DataFrame:
    counts = (
        subs[subs["team_id"] == team_id]
        .groupby("sub_out_id")
        .agg(Player=("sub_out_name", "first"), **{"Times Subbed Out": ("sub_out_id", "count")})
        .reset_index()
        .rename(columns={"sub_out_id": "player_id"})
    )
    counts["player_id"] = counts["player_id"].astype("Int64")

    mins = (
        minutes_df[minutes_df["team_id"] == team_id]
        .groupby("player_id", as_index=False)["total_minutes"].sum()
        .rename(columns={"total_minutes": "Total Mins"})
    )

    pos = players[players["team_id"] == team_id][["player_id", "position"]].copy()
    pos["player_id"] = pos["player_id"].astype("Int64")
    pos["position"]  = pos["position"].map(POSITION_GROUP_MAP).fillna(pos["position"])
    pos = pos.drop_duplicates("player_id")

    result = (
        counts
        .merge(mins, on="player_id", how="left")
        .merge(pos,  on="player_id", how="left")
        .rename(columns={"position": "Position"})
        .sort_values("Times Subbed Out", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )
    return result[["Player", "Position", "Times Subbed Out", "Total Mins"]]


# -------------------------------------------------------
# App
# -------------------------------------------------------
def main():
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

    st.markdown("### Positions Subbed In by Match Situation")

    with st.expander("ℹ️ Chart information"):
        st.markdown(
            "The **three barcharts** show which positions are most frequently subbed in for the "
            "selected team, split by match situation at the time of the substitution: "
            "Winning, Drawing, and Losing. The title of each chart includes the total number of "
            "substitutions made in that situation. "
            "Comparing across the three panels reveals the team's tactical substitution logic — "
            "for example, a high volume of attacking positions subbed in when Losing suggests "
            "an aggressive chase for goals, while more defensive or midfield substitutions when "
            "Winning indicates a game-management approach. "
            "The positions within each bar chart are ordered from left-to-right based on how attacking the position is (ascending order)."
            "The **two tables below** show the top 3 most subbed in and most subbed out players "
            "for the selected team, along with their position and total minutes played across "
            "the season — useful to identify rotation patterns and key impact players from the bench. "
            "Use the **league and team selectors** on the right to switch between teams."
        )

    if not Path(DUCKDB_PATH).exists():
        st.error(f"Database not found at `{DUCKDB_PATH}`.")
        st.stop()

    subs, teams, players = load_data(str(DUCKDB_PATH))
    bar_df, minutes_df   = precompute_all(subs, players, teams)

    # ── Session state ──────────────────────────────────
    if "selected_league" not in st.session_state:
        st.session_state.selected_league = LEAGUE_ORDER[0]
    if "selected_team" not in st.session_state:
        st.session_state.selected_team = (
            teams[teams["league_name"] == st.session_state.selected_league]
            .sort_values("team_name")["team_name"].iloc[0]
        )

    selected_league = st.session_state.selected_league
    selected_team   = st.session_state.selected_team
    team_id         = int(teams.loc[teams["team_name"] == selected_team, "team_id"].iloc[0])
    league_name     = teams.loc[teams["team_name"] == selected_team, "league_name"].iloc[0]

    # ── Row 1: 3 barcharts full width ─────────────────
    st.altair_chart(
        make_bar_charts(bar_df, team_id, league_name),
        use_container_width=False,
    )

    # ── Row 2: table_in | table_out | selector ─────────
    TABLE_CSS = """
        <style>
            [data-testid="stDataFrame"] * { font-size: 11px !important; }
        </style>
    """
    SELECTOR_CSS = """
        <style>
            [data-testid="stSelectbox"] label {
                font-size: 13px !important;
                font-weight: bold !important;
            }
            [data-testid="stSelectbox"] div[data-baseweb="select"] * {
                font-size: 13px !important;
            }
        </style>
    """
    TABLE_H = 140   # height for 3 rows + header

    #in_col, out_col, selector_col = st.columns([5, 5, 2], gap="small")
    in_col, out_col, selector_col = st.columns([5, 5, 2], gap="medium")

    with in_col:
        st.markdown(TABLE_CSS, unsafe_allow_html=True)
        st.markdown("**Top 3 most subbed in**")
        st.dataframe(
            make_subbed_in_table(subs, minutes_df, players, team_id),
            use_container_width=True,
            hide_index=True,
            height=TABLE_H,
        )

    with out_col:
        st.markdown(TABLE_CSS, unsafe_allow_html=True)
        st.markdown("**Top 3 most subbed out**")
        st.dataframe(
            make_subbed_out_table(subs, minutes_df, players, team_id),
            use_container_width=True,
            hide_index=True,
            height=TABLE_H,
        )

    with selector_col:
        st.markdown(SELECTOR_CSS, unsafe_allow_html=True)
        st.markdown(
            "<div style='padding-top:0.3cm'></div>",
            unsafe_allow_html=True,
        )
        new_league = st.selectbox("League", LEAGUE_ORDER,
                                  index=LEAGUE_ORDER.index(selected_league))
        league_teams = (
            teams[teams["league_name"] == new_league]
            .sort_values("team_name")["team_name"].tolist()
        )
        idx      = league_teams.index(selected_team) if selected_team in league_teams else 0
        new_team = st.selectbox("Team", league_teams, index=idx)

        if new_league != selected_league or new_team != selected_team:
            st.session_state.selected_league = new_league
            st.session_state.selected_team   = new_team
            st.rerun()

main()