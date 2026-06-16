import os
import duckdb
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from pathlib import Path
import io

# -------------------------------------------------------
# Path setup
# -------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DUCKDB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
LEAGUE_ORDER = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

BIN_EDGES  = list(range(45, 91, 5))
BIN_LABELS = [f"{b}–{b+5}" for b in BIN_EDGES[:-1]]
ROW_LABELS = ["1st Half"] + BIN_LABELS + ["Extra Time"]
COLUMNS    = ["Overall", "Winning", "Drawing", "Losing"]

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------
@st.cache_data(show_spinner="Loading substitution data…")
def load_data(db_path: str):
    con   = duckdb.connect(DUCKDB_PATH, read_only=True)
    subs  = con.execute("SELECT * FROM substitutions").df()
    teams = con.execute("SELECT team_id, team_name, league_name, league_id FROM teams").df()
    con.close()
    return subs, teams

# -------------------------------------------------------
# Precompute all metrics
# -------------------------------------------------------
@st.cache_data(show_spinner="Preparing data…")
def precompute_all(subs: pd.DataFrame, teams: pd.DataFrame):

    enriched = subs.copy()

    scores = enriched["current_result"].str.split("-", expand=True)
    enriched["home_score"] = pd.to_numeric(scores[0], errors="coerce")
    enriched["away_score"] = pd.to_numeric(scores[1], errors="coerce")
    is_home = enriched["home_team_id"] == enriched["team_id"]
    enriched["team_score"] = np.where(is_home, enriched["home_score"], enriched["away_score"])
    enriched["opp_score"]  = np.where(is_home, enriched["away_score"], enriched["home_score"])
    enriched["situation"]  = np.select(
        [enriched["team_score"] > enriched["opp_score"],
         enriched["team_score"] == enriched["opp_score"]],
        ["Winning", "Drawing"],
        default="Losing"
    )
    bad = enriched["home_score"].isna() | enriched["away_score"].isna()
    enriched.loc[bad, "situation"] = np.nan

    enriched["time_bin"] = pd.cut(
        enriched["minute"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=False,
    ).astype(object)
    enriched.loc[enriched["period"] == 1,     "time_bin"] = "1st Half"
    enriched.loc[enriched["minute"] > 90,     "time_bin"] = "Extra Time"
    enriched.loc[enriched["time_bin"].isna(),  "time_bin"] = "Extra Time"

    enriched = enriched.merge(
        teams[["team_id", "team_name", "league_name", "league_id"]],
        on="team_id", how="left"
    )

    subs_per_match = (
        enriched.groupby(["team_id", "game_id"]).size()
        .reset_index(name="n")
        .groupby("team_id")["n"].mean()
        .reset_index(name="avg_subs")
    )
    first_sub = (
        enriched.groupby(["team_id", "game_id"])["minute"].min()
        .reset_index(name="first_min")
        .groupby("team_id")["first_min"].mean()
        .reset_index(name="avg_first_sub")
    )
    bar_df = (
        subs_per_match
        .merge(first_sub, on="team_id")
        .merge(teams[["team_id", "team_name", "league_name"]], on="team_id")
    )

    hm_team = enriched.dropna(subset=["situation", "time_bin"])

    team_hm_lookup   = {tid: grp.copy() for tid, grp in hm_team.groupby("team_id")}
    league_hm_lookup = {lg:  grp.copy() for lg,  grp in hm_team.groupby("league_name")}

    return bar_df, team_hm_lookup, league_hm_lookup

# -------------------------------------------------------
# Heatmap matrix builder
# -------------------------------------------------------
def build_matrix(df: pd.DataFrame):
    def col_counts(subset):
        return subset.groupby("time_bin").size().reindex(ROW_LABELS, fill_value=0)
    matrix = pd.DataFrame({
        "Overall": col_counts(df),
        "Winning": col_counts(df[df["situation"] == "Winning"]),
        "Drawing": col_counts(df[df["situation"] == "Drawing"]),
        "Losing":  col_counts(df[df["situation"] == "Losing"]),
    })
    matrix.index = pd.CategoricalIndex(matrix.index, categories=ROW_LABELS, ordered=True)
    matrix       = matrix.sort_index()
    pct_matrix   = matrix.div(matrix.sum(axis=0), axis=1).multiply(100).fillna(0)
    norm_matrix  = matrix.div(matrix.max(axis=0), axis=1).fillna(0)
    return matrix, pct_matrix, norm_matrix

# -------------------------------------------------------
# Heatmap renderer → PNG bytes
# -------------------------------------------------------
_HM_FIG, _HM_AX = plt.subplots(figsize=(5.5, 6.5), facecolor="white")

def render_heatmap(matrix, pct_matrix, norm_matrix, title: str, hm_color: str) -> bytes:
    cmap = mcolors.LinearSegmentedColormap.from_list("hm", ["#ffffff", hm_color], N=256)
    n_rows, n_cols = matrix.shape

    _HM_FIG.set_facecolor("white")
    _HM_AX.clear()
    _HM_AX.imshow(norm_matrix.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for r in range(n_rows):
        for c in range(n_cols):
            count    = int(matrix.iloc[r, c])
            pct      = pct_matrix.iloc[r, c]
            norm_val = norm_matrix.iloc[r, c]
            txt_col  = "white" if norm_val > 0.55 else "#222222"
            _HM_AX.text(c, r, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=8.5,
                        color=txt_col, fontweight="bold", linespacing=1.3)

    _HM_AX.set_xticks(range(n_cols))
    _HM_AX.set_xticklabels(COLUMNS, fontsize=10, fontweight="bold")
    _HM_AX.xaxis.set_ticks_position("top")
    _HM_AX.xaxis.set_label_position("top")
    _HM_AX.set_yticks(range(n_rows))
    _HM_AX.set_yticklabels(ROW_LABELS, fontsize=9.5)
    _HM_AX.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    _HM_AX.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    _HM_AX.grid(which="minor", color="white", linewidth=1.5)
    _HM_AX.tick_params(which="minor", length=0)
    _HM_AX.tick_params(which="major", length=0)
    for spine in _HM_AX.spines.values():
        spine.set_visible(False)
    _HM_AX.set_title(title, fontsize=11, fontweight="bold", pad=28, color="#333333")

    _HM_FIG.tight_layout(pad=0.4)
    buf = io.BytesIO()
    _HM_FIG.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()

# -------------------------------------------------------
# Altair barcharts
# -------------------------------------------------------
def make_barcharts(bar_df: pd.DataFrame, team_name: str, league_name: str) -> alt.Chart:
    color     = LEAGUE_COLORS[league_name]
    league_df = bar_df[bar_df["league_name"] == league_name].copy()
    league_df["highlighted"] = (league_df["team_name"] == team_name).astype(int)

    def single_bar(df_sorted, x_field, x_title, chart_title):
        return (
            alt.Chart(df_sorted)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                y=alt.Y("team_name:N", sort=list(df_sorted["team_name"]), title=None,
                        scale=alt.Scale(padding=0),
                        axis=alt.Axis(labelFontSize=6, labelLimit=108, labelOverlap=False)),
                x=alt.X(f"{x_field}:Q", title=None,
                        axis=alt.Axis(labelFontSize=6, grid=True, gridOpacity=0.3)),
                color=alt.condition(
                    alt.datum.highlighted == 1,
                    alt.value(color),
                    alt.value("#cccccc"),
                ),
                opacity=alt.condition(
                    alt.datum.highlighted == 1,
                    alt.value(1.0),
                    alt.value(0.55),
                ),
                tooltip=[
                    alt.Tooltip("team_name:N",  title="Team"),
                    alt.Tooltip(f"{x_field}:Q", title=x_title, format=".2f"),
                ],
            )
            .properties(
                width=166, height=216,
                title=alt.TitleParams(
                    text=chart_title,
                    fontSize=8,
                    fontWeight="bold",
                    color="#333333",
                    offset=5,
                    anchor="start",
                )
            )
        )

    df1 = league_df.sort_values("avg_subs",      ascending=False)
    df2 = league_df.sort_values("avg_first_sub", ascending=False)

    return (
        alt.vconcat(
            single_bar(df1, "avg_subs",      "Avg subs / match",        "Avg Substitutions per Match"),
            single_bar(df2, "avg_first_sub", "Avg minute of first sub", "Avg Minute of First Sub"),
            spacing=20,
        )
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )

# -------------------------------------------------------
# Main app
# -------------------------------------------------------
def main():
    st.markdown(
        """
        <style>
            #MainMenu        { visibility: hidden; }
            header           { visibility: hidden; }
            footer           { visibility: hidden; }
            .block-container {
                padding-top:   0rem  !important;
                margin-top:   -1rem  !important;
                padding-left:  1rem  !important;
                padding-right: 2rem  !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not Path(DUCKDB_PATH).exists():
        st.error(f"Database not found at `{DUCKDB_PATH}`.")
        st.stop()

    subs, teams                              = load_data(str(DUCKDB_PATH))
    bar_df, team_hm_lookup, league_hm_lookup = precompute_all(subs, teams)

    st.markdown("### Substitution Behaviour by Team")

    with st.expander("ℹ️ Chart information"):
        st.markdown(
            "This page combines three views to analyse substitution timing behaviour. "
            "The **two barcharts on the left** show all teams in the selected league ranked by "
            "average substitutions per match and average minute of first substitution — "
            "the selected team is highlighted in colour, all others are greyed out. "
            "The **two heatmaps on the right** show when substitutions are made broken down by "
            "match situation (Overall, Winning, Drawing, Losing): the left heatmap is for the "
            "selected team only, the right one aggregates all teams in the league for comparison. "
            "Each cell shows the raw count and its percentage of all substitutions in that column — "
            "colour intensity is normalised per column so the darkest cell always marks the most "
            "frequent time window for that situation. "
            "A darker cell in the **Losing** column at an early time bin means the team tends to "
            "react quickly when behind; a darker cell in **Winning** at a late bin suggests "
            "more conservative, game-management substitutions. "
            "Use the **league and team selectors below the left barcharts** to switch between "
            "teams and compare individual behaviour against the league aggregate."
        )

    # Session state defaults
    if "selected_league" not in st.session_state:
        st.session_state.selected_league = LEAGUE_ORDER[0]
    if "selected_team" not in st.session_state:
        st.session_state.selected_team = (
            teams[teams["league_name"] == st.session_state.selected_league]
            .sort_values("team_name")["team_name"].iloc[0]
        )

    selected_league = st.session_state.selected_league
    selected_team   = st.session_state.selected_team
    hm_color        = LEAGUE_COLORS[selected_league]
    team_row        = teams.loc[teams["team_name"] == selected_team, "team_id"]
    team_id         = int(team_row.iloc[0]) if not team_row.empty else None

    left_col, right_col = st.columns([2, 5], gap="large")

    # ── Left: barcharts + selectors below ─────────────
    with left_col:
        bar_key = f"barchart_{selected_league}_{selected_team}"
        if bar_key not in st.session_state:
            st.session_state[bar_key] = make_barcharts(bar_df, selected_team, selected_league)
        st.altair_chart(st.session_state[bar_key], use_container_width=False)

        new_league = st.selectbox(
            "League", LEAGUE_ORDER,
            index=LEAGUE_ORDER.index(selected_league),
        )
        if new_league != selected_league:
            st.session_state.selected_league = new_league
            st.session_state.selected_team = (
                teams[teams["league_name"] == new_league]
                .sort_values("team_name")["team_name"].iloc[0]
            )
            st.rerun()

        league_teams = (
            teams[teams["league_name"] == selected_league]
            .sort_values("team_name")["team_name"].tolist()
        )
        idx      = league_teams.index(selected_team) if selected_team in league_teams else 0
        new_team = st.selectbox("Team", league_teams, index=idx)
        if new_team != selected_team:
            st.session_state.selected_team = new_team
            st.rerun()

    # ── Right: heatmaps ───────────────────────────────
    with right_col:
        if team_id is not None:

            team_hm_key = f"team_heatmap_{team_id}"
            if team_hm_key not in st.session_state:
                team_df = team_hm_lookup.get(team_id, pd.DataFrame())
                tm_mat, tm_pct, tm_norm = build_matrix(team_df)
                st.session_state[team_hm_key] = render_heatmap(
                    tm_mat, tm_pct, tm_norm, selected_team, hm_color
                )
            team_img = st.session_state[team_hm_key]

            league_hm_key = f"league_heatmap_{selected_league}"
            if league_hm_key not in st.session_state:
                league_df = league_hm_lookup.get(selected_league, pd.DataFrame())
                lg_mat, lg_pct, lg_norm = build_matrix(league_df)
                st.session_state[league_hm_key] = render_heatmap(
                    lg_mat, lg_pct, lg_norm,
                    f"{selected_league} (all teams)", hm_color
                )
            league_img = st.session_state[league_hm_key]

            hm1, hm2 = st.columns(2, gap="small")
            with hm1:
                st.image(team_img,   use_container_width=True)
            with hm2:
                st.image(league_img, use_container_width=True)


main()