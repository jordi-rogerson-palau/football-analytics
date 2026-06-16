import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from PIL import Image
import pandas as pd
import duckdb
import altair as alt
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
            padding-right: 2rem !important;
        }
        [data-testid="stImage"] img { border-radius: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Path setup
# -------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"


# -------------------------------------------------------
# Constants
# -------------------------------------------------------
PITCH_W, PITCH_H = 120.0, 80.0
APOTHEM          = 2.0
L_SCALE          = 120 / 105
W_SCALE          = 80  / 68
FIG_W, FIG_H, DPI = 5.78, 3.85, 120

LEAGUE_ORDER  = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

POSITION_GROUP_MAP = {
    "Goalkeeper":                "Goalkeeper",
    "Center Back":               "Center Backs",
    "Left Center Back":          "Center Backs",
    "Right Center Back":         "Center Backs",
    "Left Back":                 "Left Backs",
    "Left Wing Back":            "Left Backs",
    "Right Back":                "Right Backs",
    "Right Wing Back":           "Right Backs",
    "Left Defensive Midfield":   "Defensive Midfielders",
    "Center Defensive Midfield": "Defensive Midfielders",
    "Right Defensive Midfield":  "Defensive Midfielders",
    "Left Center Midfield":      "Center Midfielders",
    "Center Midfield":           "Center Midfielders",
    "Right Center Midfield":     "Center Midfielders",
    "Left Midfield":             "Left Wingers",
    "Right Midfield":            "Right Wingers",
    "Left Attacking Midfield":   "Attacking Midfielders",
    "Center Attacking Midfield": "Attacking Midfielders",
    "Right Attacking Midfield":  "Attacking Midfielders",
    "Left Wing":                 "Left Wingers",
    "Right Wing":                "Right Wingers",
    "Left Center Forward":       "Strikers",
    "Right Center Forward":      "Strikers",
    "Center Forward":            "Strikers",
}

POSITION_ORDER = [
    "Goalkeeper", "Center Backs", "Left Backs", "Right Backs",
    "Defensive Midfielders", "Center Midfielders", "Attacking Midfielders",
    "Left Wingers", "Right Wingers", "Strikers",
]

# -------------------------------------------------------
# Data loading — one connection, everything needed
# -------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data(db_path: str):
    con      = duckdb.connect(db_path, read_only=True)
    usage    = con.execute("SELECT match_id, team_id, player_id, location_x, location_y FROM ball_losses").df()
    teams    = con.execute("SELECT team_id, team_name, league_name FROM teams").df()
    players  = con.execute("SELECT player_id, player_name, position FROM players").df()
    seq_dirs = con.execute("SELECT match_id, team_id, AVG(start_x) AS avg_start_x FROM sequences GROUP BY match_id, team_id").df()
    con.close()

    # Pre-split usage by team_id — O(1) lookup
    usage_by_team = {tid: grp.copy() for tid, grp in usage.groupby("team_id")}

    return usage_by_team, teams, players, seq_dirs


# -------------------------------------------------------
# Hex grid — built once, cached
# -------------------------------------------------------
@st.cache_data
def build_hex_grid():
    pitch_box = shapely_box(0, 0, PITCH_W, PITCH_H)
    r      = APOTHEM * 2.0 / np.sqrt(3.0)
    col_dx = 1.5 * r
    row_dy = 2.0 * APOTHEM

    valid_centres = []
    hex_polys     = []

    col_start = int(np.floor(0 / col_dx))
    col_end   = int(np.ceil(PITCH_W / col_dx)) + 1

    for col in range(col_start, col_end + 1):
        cx       = col * col_dx
        y_offset = APOTHEM if (col % 2 == 1) else 0.0
        row_start = int(np.floor(-y_offset / row_dy)) - 1
        row_end   = int(np.ceil((PITCH_H - y_offset) / row_dy)) + 1
        for row in range(row_start, row_end + 1):
            cy      = row * row_dy + APOTHEM + y_offset
            angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
            verts_x = cx + r * np.cos(angles)
            verts_y = cy + r * np.sin(angles)
            poly    = ShapelyPolygon(zip(verts_x, verts_y))
            clipped = poly.intersection(pitch_box)
            if clipped.is_empty or clipped.area < 1e-6:
                continue
            valid_centres.append((cx, cy))
            hex_polys.append(clipped)

    centres = np.array(valid_centres)
    cells   = []
    for clipped in hex_polys:
        cx_c  = np.mean([p[0] for p in clipped.exterior.coords])
        cy_c  = np.mean([p[1] for p in clipped.exterior.coords])
        dists = (centres[:, 0] - cx_c)**2 + (centres[:, 1] - cy_c)**2
        idx   = int(np.argmin(dists))
        ext   = np.array(clipped.exterior.coords).copy()
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(ext) - 2) + [MplPath.CLOSEPOLY]
        cells.append((ext, codes, idx))

    # ── Corner exclusion ──────────────────────────────────────────────────
    # Attacking direction is always left→right after normalisation, so only
    # the RIGHT end corners matter: (120, 0) = bottom-right, (120, 80) = top-right.
    # Any hex whose centre falls within CORNER_RADIUS units of either corner
    # is flagged and excluded from both computation and rendering.
    CORNER_RADIUS = 6.0   # metres in the 120×80 system — covers the corner arc
    corner_points = np.array([[120.0, 0.0], [120.0, 80.0]])
    corner_hex_indices = set()
    for cp in corner_points:
        dists_to_corner = np.hypot(centres[:, 0] - cp[0], centres[:, 1] - cp[1])
        corner_hex_indices.update(np.where(dists_to_corner <= CORNER_RADIUS)[0].tolist())

    return centres, cells, corner_hex_indices


# -------------------------------------------------------
# Precompute per-team usage data — direction-normalised,
# hex-assigned, position-grouped. Cached once.
# -------------------------------------------------------
@st.cache_data(show_spinner="Preparing team data…")
def precompute_team_data(_usage_by_team, _teams, _players, _seq_dirs):
    player_info = _players[["player_id", "player_name", "position"]].drop_duplicates("player_id")
    player_info = player_info.copy()
    player_info["player_id"]      = player_info["player_id"].astype(float)
    player_info["position_group"] = player_info["position"].map(POSITION_GROUP_MAP)

    match_dirs = _seq_dirs.copy()   # match_id, team_id, avg_start_x

    team_data = {}
    for _, row in _teams.iterrows():
        tid   = row["team_id"]
        tname = row["team_name"]
        lg    = row["league_name"]

        df = _usage_by_team.get(tid, pd.DataFrame())
        if df.empty:
            continue

        # Direction normalisation per match (vectorized)
        df = df.merge(
            match_dirs[match_dirs["team_id"] == tid][["match_id", "avg_start_x"]],
            on="match_id", how="left"
        )
        df["location_x"] = np.where(
            df["avg_start_x"] > 60,
            PITCH_W - df["location_x"],
            df["location_x"]
        )
        df = df.dropna(subset=["location_x", "location_y"])

        # Position group join
        df["player_id"] = df["player_id"].astype(float)
        df = df.merge(player_info[["player_id", "position_group"]], on="player_id", how="left")

        team_data[(lg, tname)] = {
            "df":        df,
            "n_matches": df["match_id"].nunique(),
            "team_id":   tid,
        }

    return team_data


# -------------------------------------------------------
# Hex assignment (vectorized, chunked)
# -------------------------------------------------------
def assign_hex(coords: np.ndarray, centres: np.ndarray) -> list:
    cx, cy   = centres[:, 0], centres[:, 1]
    hex_idxs = []
    chunk    = 5000
    for i in range(0, len(coords), chunk):
        c    = coords[i:i+chunk]
        dx   = cx[None, :] - c[:, 0:1]
        dy   = cy[None, :] - c[:, 1:2]
        hex_idxs.extend(np.argmin(dx**2 + dy**2, axis=1).tolist())
    return hex_idxs


# -------------------------------------------------------
# Pitch drawing helper
# -------------------------------------------------------
def draw_pitch(ax, lines_color="#bcbcbc"):
    ax.set_facecolor("white")
    def sp(x, y): return x * L_SCALE, y * W_SCALE
    lines = [
        [(0,0),(0,80)], [(120,0),(120,80)], [(0,80),(120,80)], [(0,0),(120,0)],
        [(60,0),(60,80)],
        [sp(0,13.85),sp(16.5,13.85)], [sp(0,54.15),sp(16.5,54.15)],
        [sp(16.5,13.85),sp(16.5,54.15)],
        [sp(0,24.85),sp(5.5,24.85)], [sp(0,43.15),sp(5.5,43.15)],
        [sp(5.5,24.85),sp(5.5,43.15)],
        [sp(88.5,13.85),sp(105,13.85)], [sp(88.5,54.15),sp(105,54.15)],
        [sp(88.5,13.85),sp(88.5,54.15)],
        [sp(99.5,24.85),sp(105,24.85)], [sp(99.5,43.15),sp(105,43.15)],
        [sp(99.5,24.85),sp(99.5,43.15)],
    ]
    for (x1,y1),(x2,y2) in lines:
        ax.plot([x1,x2],[y1,y2],"-",lw=1.5,color=lines_color,alpha=0.9,zorder=3)
    cx_p, cy_p = 52.5*L_SCALE, 34*W_SCALE
    for centre, t1, t2 in [
        ((94.0*L_SCALE,34*W_SCALE),128,232),
        ((11.0*L_SCALE,34*W_SCALE),308,52),
        ((cx_p,cy_p),0,360),
    ]:
        r = 9.15*L_SCALE if t2==360 else 9*L_SCALE
        ax.add_patch(patches.Wedge(centre,r,t1,t2,
                                   fill=False,edgecolor=lines_color,width=0.02,zorder=3))
    ax.set_xlim(0,120); ax.set_ylim(80,0)
    ax.set_aspect("equal"); ax.axis("off")


# -------------------------------------------------------
# Render heatmap → PNG bytes (fresh fig per call, page-safe)
# -------------------------------------------------------
def render_pitch_heatmap(df: pd.DataFrame, n_matches: int,
                         centres: np.ndarray, hex_cells: list,
                         corner_hex_indices: set,
                         color: str) -> bytes:
    # Exclude own penalty area (x < 16.5m scaled) — keeper + defenders
    # inside the box dominate and wash out all other pitch zones
    CUTOFF_X = 16.5 * L_SCALE   # ≈ 18.9 in 120×80 system
    df = df[df["location_x"] >= CUTOFF_X].copy()

    coords = df[["location_x","location_y"]].to_numpy()
    idxs   = assign_hex(coords, centres)
    df["hex_idx"] = idxs

    # Drop corner-kick hexes from the count before normalisation so they
    # don't inflate the max and don't appear in the rendered heatmap.
    df = df[~df["hex_idx"].isin(corner_hex_indices)]

    hc = df.groupby("hex_idx").size().reset_index(name="count")

    # Per-team min-max: each team's scale fills the full colour range
    # so spatial patterns are visible regardless of volume differences.
    count_min = hc["count"].min()
    count_max = hc["count"].max()
    if count_max == count_min:
        hc["norm"] = 1.0
    else:
        hc["norm"] = (hc["count"] - count_min) / (count_max - count_min)
    count_map = dict(zip(hc["hex_idx"], hc["norm"]))

    cmap = mcolors.LinearSegmentedColormap.from_list("hm", ["#ffffff", color], N=256)

    # Gamma stretch: pushes low normalised values upward so bins that are
    # only slightly below the team's maximum still render with visible colour.
    # Without this, the pale-color AND low-alpha applied simultaneously
    # (double-dimming) makes most hexes look identical and washed out.
    # gamma < 1 expands the bottom of the distribution; 0.5 (square-root)
    # is a good default — tweak between 0.4 (aggressive) and 0.7 (subtle).
    GAMMA = 0.5

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw_pitch(ax)

    for ext, codes, hex_idx in hex_cells:
        if hex_idx in corner_hex_indices:
            continue
        val = count_map.get(hex_idx, 0.0)
        if val <= 0:
            continue
        # Apply gamma stretch so the full colour range is always used,
        # regardless of how peaked or flat this team's distribution is.
        val_stretched = val ** GAMMA
        c_rgba = cmap(val_stretched)
        # Single dimming source — alpha only.  No more double-dimming
        # (previously: pale cmap colour * low alpha at the same time).
        alpha = 0.15 + 0.80 * val_stretched
        ax.add_patch(PathPatch(MplPath(ext, codes),
                               facecolor=c_rgba, edgecolor="none",
                               alpha=alpha, linewidth=0.0, zorder=1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI,
                bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------
# Build Altair position barchart
# -------------------------------------------------------
def make_position_chart(df: pd.DataFrame, color: str, team_name: str) -> alt.Chart:
    df = df.copy()
    df = df[df["position_group"].notna()]
    total = len(df)
    if total == 0:
        return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No data"))

    pct_df = (
        df["position_group"].value_counts()
        .reindex(POSITION_ORDER, fill_value=0)
        .reset_index()
    )
    pct_df.columns = ["position_group", "count"]
    pct_df["pct"]   = (pct_df["count"] / total * 100).round(1)
    pct_df["label"] = pct_df["pct"].apply(lambda x: f"{x:.1f}%" if x > 0 else "")
    pct_df["position_group"] = pd.Categorical(
        pct_df["position_group"], categories=POSITION_ORDER[::-1], ordered=True
    )

    bars = (
        alt.Chart(pct_df)
        .mark_bar(cornerRadiusTopRight=2, cornerRadiusBottomRight=2)
        .encode(
            y=alt.Y("position_group:O", sort=None, title=None,
                    axis=alt.Axis(labelFontSize=10, labelLimit=180)),
            x=alt.X("pct:Q", title="% of ball losses",
                    axis=alt.Axis(labelFontSize=9, titleFontSize=10)),
            color=alt.value(color),
            tooltip=[
                alt.Tooltip("position_group:O", title="Position"),
                alt.Tooltip("pct:Q",            title="%", format=".1f"),
                alt.Tooltip("count:Q",          title="Count"),
            ]
        )
    )

    text = (
        alt.Chart(pct_df[pct_df["pct"] > 0])
        .mark_text(align="left", dx=4, fontSize=9, color="#444444")
        .encode(
            y=alt.Y("position_group:O", sort=None),
            x=alt.X("pct:Q"),
            text=alt.Text("label:N"),
        )
    )

    return (
        (bars + text)
        .properties(
            width=420, height=320,
            title=alt.TitleParams(
                text="Ball Losses by Position",
                fontSize=11, fontWeight="bold", color="#333333",
                anchor="start", offset=5, limit = 400,
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


# -------------------------------------------------------
# App
# -------------------------------------------------------
centres, hex_cells, corner_hex_indices    = build_hex_grid()
usage_by_team, teams, players, seq_dirs  = load_data(str(DB_PATH))
team_data                                 = precompute_team_data(
    usage_by_team, teams, players, seq_dirs
)

# ── Session state ──────────────────────────────────────
if "ur_league" not in st.session_state:
    st.session_state.ur_league = "1. Bundesliga"
if "ur_team" not in st.session_state:
    st.session_state.ur_team = (
        teams[teams["league_name"] == "1. Bundesliga"]
        .sort_values("team_name")["team_name"].iloc[0]
    )

selected_league = st.session_state.ur_league
selected_team   = st.session_state.ur_team
color           = LEAGUE_COLORS[selected_league]

# ── Title row ─────────────────────────────────────────

title_col, _ = st.columns([3, 5], gap="small")
with title_col:
    st.markdown(
        '<h3 style="white-space: nowrap;">Ball Losses by Position & Location - Team Analysis</h3>',
        unsafe_allow_html=True
    )

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "**Usage Rate** measures the percentage of a team's possessions that end with a given player — "
        "it identifies who the ball consistently flows through before possession is lost. "
        "A ball loss occurs whenever a player ends a possession sequence, whether through a missed pass, "
        "failed dribble, miscontrol, or any other action that gives the ball to the opponent. "
        "The **pitch heatmap on the left** shows the spatial distribution of all ball losses for the "
        "selected team across the season, normalised to the team's own maximum so spatial patterns "
        "are always visible regardless of volume. "
        "All pitches are direction-normalised so the team always attacks left to right, and the "
        "own penalty area is excluded to prevent the goalkeeper from dominating the colour scale. "
        "The **barchart on the right** breaks down ball losses by position group, showing which "
        "positions are responsible for the highest share of possession endings — a high percentage "
        "for a given position can indicate both high involvement and high risk-taking in that area. "
        "Use the **league and team selectors** to switch between teams."
    )
# ── Layout: pitch left | barchart+selectors right ─────
pitch_col, right_col = st.columns([4, 3], gap="large")

# Resolve team data
entry      = team_data.get((selected_league, selected_team))
pitch_key  = f"ur_pitch_{selected_league}_{selected_team}"
bar_key    = f"ur_bar_{selected_league}_{selected_team}"

# ── Left: pitch heatmap — never disappears on team switch ─────
with pitch_col:
    ph = st.empty()

    # Always show something immediately — last rendered image if available
    last_img_key = st.session_state.get("ur_last_pitch_img")
    if pitch_key in st.session_state:
        ph.image(st.session_state[pitch_key], width=713)
    elif last_img_key and last_img_key in st.session_state:
        ph.image(st.session_state[last_img_key], width=713)

    if pitch_key not in st.session_state and entry is not None:
        img = render_pitch_heatmap(
            entry["df"], entry["n_matches"], centres, hex_cells, corner_hex_indices, color
        )
        st.session_state[pitch_key] = img
        ph.image(img, width=713)

    st.session_state["ur_last_pitch_img"] = pitch_key

# ── Right: selectors on top, barchart below ───────────
with right_col:
    new_league = st.selectbox(
        "League", LEAGUE_ORDER,
        index=LEAGUE_ORDER.index(selected_league),
        label_visibility="visible",
    )
    if new_league != selected_league:
        st.session_state.ur_league = new_league
        st.session_state.ur_team   = (
            teams[teams["league_name"] == new_league]
            .sort_values("team_name")["team_name"].iloc[0]
        )
        st.rerun()

    league_teams = (
        teams[teams["league_name"] == selected_league]
        .sort_values("team_name")["team_name"].tolist()
    )
    idx      = league_teams.index(selected_team) if selected_team in league_teams else 0
    new_team = st.selectbox("Team", league_teams, index=idx, label_visibility="visible")
    if new_team != selected_team:
        st.session_state.ur_team = new_team
        st.rerun()

    st.markdown("---")

    if entry is not None:
        if bar_key not in st.session_state:
            st.session_state[bar_key] = make_position_chart(entry["df"], color, selected_team)
        st.altair_chart(st.session_state[bar_key], use_container_width=False)
    else:
        st.info("No data available for this team.")