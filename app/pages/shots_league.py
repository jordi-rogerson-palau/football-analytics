import io
import os
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from pathlib import Path

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

LEAGUE_ORDER  = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

L_SCALE = 120 / 105
W_SCALE = 80  / 68


@st.cache_data
def load_data():
    conn         = duckdb.connect(str(DB_PATH), read_only=True)
    shots_league = conn.execute("SELECT * FROM shots_league").df()
    shots        = conn.execute("SELECT * FROM shots").df()
    match_scores = conn.execute("SELECT * FROM match_scores").df()
    teams        = conn.execute("SELECT * FROM teams").df()
    conn.close()
    return shots_league, shots, match_scores, teams

shots_league, shots, match_scores, teams = load_data()

if "shot_outcome" not in st.session_state:
    st.session_state["shot_outcome"] = "All"


def build_table_df(shots_filtered, games_per_league):
    outcome_map = {
        "shot_scored":  "Scored",
        "shot_saved":   "Saved",
        "shot_missed":  "Missed",
        "shot_blocked": "Blocked",
    }
    raw = shots_filtered.groupby(["league_name", "end_type"]).size().reset_index(name="count")
    raw["end_type"] = raw["end_type"].map(outcome_map).fillna(raw["end_type"])

    pivot = raw.pivot(index="league_name", columns="end_type", values="count").fillna(0).reset_index()
    for col in ["Scored", "Saved", "Missed", "Blocked"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["Total"]  = pivot[["Scored", "Saved", "Missed", "Blocked"]].sum(axis=1)
    pivot           = pivot.merge(games_per_league, on="league_name", how="left")
    pivot["Shots"]  = (pivot["Total"] / pivot["num_games"]).round(1)
    for col in ["Scored", "Saved", "Missed", "Blocked"]:
        pivot[f"{col}"] = (pivot[col] / pivot["Total"] * 100).round(1)

    return (
        pivot[["league_name", "Shots", "Scored", "Saved", "Missed", "Blocked"]]
        .rename(columns={"league_name": "League"})
        .sort_values("League")
        .reset_index(drop=True)
    )


def _draw_pitch_on_ax(ax, lines_color="#444444", bg="#f8f5f0"):
    ax.set_facecolor(bg)

    def sp(x, y):
        return x * L_SCALE, y * W_SCALE

    for (x1, y1), (x2, y2) in [
        [(0, 0),   (0, 80)],
        [(120, 0), (120, 80)],
        [(0, 80),  (120, 80)],
        [(0, 0),   (120, 0)],
        [(60, 0),  (60, 80)],
        [sp(0, 13.85),    sp(16.5, 13.85)],
        [sp(0, 54.15),    sp(16.5, 54.15)],
        [sp(16.5, 13.85), sp(16.5, 54.15)],
        [sp(0, 24.85),    sp(5.5, 24.85)],
        [sp(0, 43.15),    sp(5.5, 43.15)],
        [sp(5.5, 24.85),  sp(5.5, 43.15)],
        [sp(88.5, 13.85), sp(105, 13.85)],
        [sp(88.5, 54.15), sp(105, 54.15)],
        [sp(88.5, 13.85), sp(88.5, 54.15)],
        [sp(99.5, 24.85), sp(105, 24.85)],
        [sp(99.5, 43.15), sp(105, 43.15)],
        [sp(99.5, 24.85), sp(99.5, 43.15)],
    ]:
        ax.plot([x1, x2], [y1, y2], "-", lw=1.2, color=lines_color, alpha=0.8, zorder=3)

    cx = 52.5 * L_SCALE
    cy = 34   * W_SCALE
    ax.add_patch(patches.Wedge((94.0 * L_SCALE, 34 * W_SCALE), 9 * L_SCALE, 128, 232,
                               fill=False, edgecolor=lines_color, width=0.02, zorder=3))
    ax.add_patch(patches.Wedge((11.0 * L_SCALE, 34 * W_SCALE), 9 * L_SCALE, 308, 52,
                               fill=False, edgecolor=lines_color, width=0.02, zorder=3))
    ax.add_patch(patches.Wedge((cx, cy), 9.15 * L_SCALE, 0, 360,
                               fill=False, edgecolor=lines_color, width=0.02, zorder=3))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 87)
    ax.set_aspect("equal")
    ax.axis("off")


def render_league_pitch(league, shots_df):
    colour = LEAGUE_COLORS[league]
    lg     = shots_df[shots_df["league_name"] == league].dropna(subset=["end_x", "end_y"])

    fig, ax = plt.subplots(figsize=(4.5, 3.0), facecolor="white")
    _draw_pitch_on_ax(ax)

    if not lg.empty:
        ax.scatter(
            lg["end_x"], lg["end_y"],
            s=6,
            color=colour, edgecolors="none",
            alpha=0.15, zorder=6,
            marker="o"
        )

    # Title drawn inside axes headroom (y=85, above pitch top at y=80)
    # so PNG bounding box is unaffected — fontsize=22 matches other pitch scripts
    ax.text(60, 83.5, league, color=colour, fontsize=14,
            fontweight="bold", ha="center", va="center", zorder=5)
    fig.tight_layout(pad=0.4)
    return fig



def make_table_figure(table_df):
    # Fractions measured from a rendered pitch panel:
    #   figsize=(4.5, 3.0), xlim=0-120, ylim=0-80, aspect=equal,
    #   tight_layout(pad=0.4), set_title(fontsize=8, pad=3, loc="center")
    # add_axes pins the position absolutely so the title above does NOT
    # shift the axes — the table top/bottom align exactly with the pitch borders.
    AX_LEFT   = 0.038796
    AX_BOTTOM = 0.018519
    AX_WIDTH  = 0.922408   # 0.961204 - 0.038796
    AX_HEIGHT = 0.922407   # 0.940926 - 0.018519

    fig = plt.figure(figsize=(4.5, 3.0), facecolor="white")
    ax  = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_WIDTH, AX_HEIGHT])
    ax.axis("off")
    ax.set_title("Per Game Metrics", fontsize=8, fontweight="bold", pad=3,
                 loc="center", color="#333333")

    col_labels  = ["League", "Shots", "Scored%", "Saved%", "Missed%", "Blocked%"]
    col_widths  = [0.26, 0.12, 0.155, 0.155, 0.155, 0.155]
    cell_data   = []
    cell_colors = []

    for _, row in table_df.iterrows():
        c = LEAGUE_COLORS.get(row["League"], "#ffffff")
        cell_data.append([
            row["League"],
            f"{row['Shots']:.1f}",
            f"{row['Scored']:.1f}%",
            f"{row['Saved']:.1f}%",
            f"{row['Missed']:.1f}%",
            f"{row['Blocked']:.1f}%",
        ])
        cell_colors.append([c + "22"] * 6)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
        colWidths=col_widths,
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row_idx == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold", fontsize=7.5)
        elif col_idx == 0:
            league_name = cell_data[row_idx - 1][0]
            c           = LEAGUE_COLORS.get(league_name, "#333333")
            cell.set_text_props(color=c, fontweight="bold", fontsize=7)

    return fig


games_per_league = (
    match_scores
    .merge(
        teams[["team_id", "league_id", "league_name"]].drop_duplicates("league_id"),
        on="league_id", how="left"
    )
    .groupby("league_name")["match_id"]
    .nunique()
    .reset_index()
    .rename(columns={"match_id": "num_games"})
)

st.markdown("### Spatial Distribution of Shots and per-game Metrics")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each pitch shows the **spatial distribution of locations where shots have been taken** for a given league."
        "Each dot is a single shot, with opacity set low so denser areas become naturally darker."
        "The **bottom-right table** summarises per-game shot volume and outcome breakdown: "
        "**Shots** is the average number of shots per game; "
        "**Scored**, **Saved**, **Missed** and **Blocked** are the percentage of all shots falling "
        "into each outcome. "
        "Use the **filter at the bottom** to restrict all six panels to a single outcome — for example, "
        "selecting *Scored* shows only where goals ended up spatially, and updates the table to reflect "
        "only that subset. "
        "A higher **Scored %** indicates better shot quality or finishing; a higher **Blocked %** "
        "reflects more defensive pressure at the moment of the shot."
    )

outcome_options     = ["All", "Scored", "Saved", "Missed", "Blocked"]
outcome_map_reverse = {
    "Scored":  "shot_scored",
    "Saved":   "shot_saved",
    "Missed":  "shot_missed",
    "Blocked": "shot_blocked",
}

selected = st.session_state["shot_outcome"]
if selected == "All":
    shots_filtered = shots.copy()
else:
    shots_filtered = shots[shots["end_type"] == outcome_map_reverse[selected]].copy()

table_df = build_table_df(shots_filtered, games_per_league)

row1  = st.columns(3, gap="small")
row2  = st.columns(3, gap="small")
slots = row1 + row2

for slot, league in zip(slots[:5], LEAGUE_ORDER):
    with slot:
        fig = render_league_pitch(league, shots_filtered)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with slots[5]:
    fig_t = make_table_figure(table_df)
    st.pyplot(fig_t, use_container_width=True)
    plt.close(fig_t)

st.markdown("---")
st.selectbox(
    "Filter by shot outcome",
    options=outcome_options,
    key="shot_outcome",
)