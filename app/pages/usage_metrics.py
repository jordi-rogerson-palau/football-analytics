import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
import pandas as pd
import duckdb
import altair as alt
import streamlit as st
from pathlib import Path as FilePath

st.markdown(
    """
    <style>
        #MainMenu        { visibility: hidden; }
        header           { visibility: hidden; }
        footer           { visibility: hidden; }

        /* Equal left/right screen margins */
        .block-container {
            padding-top:   0rem  !important;
            margin-top:   -1rem  !important;
            padding-left:  1rem  !important;
            padding-right: 1rem  !important;
            max-width:    100%   !important;
        }

        /* Equal gutters between all columns */
        [data-testid="column"] {
            padding-left:  0.75rem !important;
            padding-right: 0.75rem !important;
        }

        [data-testid="stImage"] img { border-radius: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Path setup
# -------------------------------------------------------
SCRIPT_DIR   = FilePath(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
PITCH_W, PITCH_H   = 120.0, 80.0
APOTHEM            = 2.0
L_SCALE            = 120 / 105
W_SCALE            = 80  / 68
FIG_W, FIG_H, DPI  = 4.2, 2.8, 120    # compact — shares row with lollipop

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

TOP_N = 5

LOLLIPOP_METRICS = [
    ("Goals",             "goals"),
    ("Assists",           "assists"),
    ("Key Passes",        "key_passes"),
    ("Successful Passes", "successful_passes"),
    ("xG",                "xg"),
]

# -------------------------------------------------------
# Ordinal suffix helper
# -------------------------------------------------------
def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th','th','th','th','th','th'][n % 10]}"


# -------------------------------------------------------
# Data loading
# -------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data(db_path: str):
    con = duckdb.connect(db_path, read_only=True)

    # ball_losses with locations for hexbin
    losses_raw = con.execute(
        "SELECT match_id, team_id, player_id, location_x, location_y FROM ball_losses"
    ).df()

    # players: full stats for lollipop
    players = con.execute(
        "SELECT player_id, player_name, team_id, position, "
        "goals, assists, xg, key_passes, successful_passes FROM players"
    ).df()

    teams = con.execute("SELECT team_id, team_name, league_name FROM teams").df()

    # direction normalisation: avg start_x per match per team
    seq_dirs = con.execute(
        "SELECT match_id, team_id, AVG(start_x) AS avg_start_x "
        "FROM sequences GROUP BY match_id, team_id"
    ).df()

    con.close()

    losses_raw["player_id"] = losses_raw["player_id"].astype(float)
    players["player_id"]    = players["player_id"].astype(float)

    # ── Position group ─────────────────────────────────
    players["position_group"] = players["position"].map(POSITION_GROUP_MAP)

    # ── Usage % per player per team ────────────────────
    team_totals   = losses_raw.groupby("team_id").size().reset_index(name="team_total")
    player_losses = (
        losses_raw.groupby(["team_id", "player_id"]).size()
        .reset_index(name="losses")
        .merge(team_totals, on="team_id")
    )
    player_losses["usage_pct"] = (
        player_losses["losses"] / player_losses["team_total"] * 100
    ).round(2)
    player_losses["player_id"] = player_losses["player_id"].astype(float)

    # join name + position_group
    pinfo = players[["player_id", "player_name", "position_group"]].drop_duplicates("player_id")
    player_losses = player_losses.merge(pinfo, on="player_id", how="left")

    # top-5 per team, sorted descending by usage_pct
    top5_by_team = {
        tid: grp.sort_values("usage_pct", ascending=False).head(TOP_N).reset_index(drop=True)
        for tid, grp in player_losses.groupby("team_id")
    }

    # ── Direction-normalised losses split by player ────
    # merge direction per match
    losses_dir = losses_raw.merge(seq_dirs, on=["match_id", "team_id"], how="left")
    losses_dir["location_x"] = np.where(
        losses_dir["avg_start_x"] > 60,
        PITCH_W - losses_dir["location_x"],
        losses_dir["location_x"],
    )
    losses_dir = losses_dir.dropna(subset=["location_x", "location_y"])

    losses_by_player = {
        pid: grp.copy()
        for pid, grp in losses_dir.groupby("player_id")
    }

    return top5_by_team, players, teams, losses_by_player


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
        codes = [Path.MOVETO] + [Path.LINETO] * (len(ext) - 2) + [Path.CLOSEPOLY]
        cells.append((ext, codes, idx))

    CORNER_RADIUS  = 6.0
    corner_points  = np.array([[120.0, 0.0], [120.0, 80.0]])
    corner_hex_set = set()
    for cp in corner_points:
        dists_to_corner = np.hypot(centres[:, 0] - cp[0], centres[:, 1] - cp[1])
        corner_hex_set.update(np.where(dists_to_corner <= CORNER_RADIUS)[0].tolist())

    return centres, cells, corner_hex_set


# -------------------------------------------------------
# Hex assignment (vectorized, chunked)
# -------------------------------------------------------
def assign_hex(coords: np.ndarray, centres: np.ndarray) -> list:
    cx, cy   = centres[:, 0], centres[:, 1]
    hex_idxs = []
    chunk    = 5000
    for i in range(0, len(coords), chunk):
        c = coords[i:i + chunk]
        dx = cx[None, :] - c[:, 0:1]
        dy = cy[None, :] - c[:, 1:2]
        hex_idxs.extend(np.argmin(dx**2 + dy**2, axis=1).tolist())
    return hex_idxs


# -------------------------------------------------------
# Pitch drawing helper
# -------------------------------------------------------
def draw_pitch(ax, lines_color="#bcbcbc"):
    ax.set_facecolor("white")

    def sp(x, y): return x * L_SCALE, y * W_SCALE

    lines = [
        [(0, 0), (0, 80)], [(120, 0), (120, 80)],
        [(0, 80), (120, 80)], [(0, 0), (120, 0)],
        [(60, 0), (60, 80)],
        [sp(0, 13.85), sp(16.5, 13.85)], [sp(0, 54.15), sp(16.5, 54.15)],
        [sp(16.5, 13.85), sp(16.5, 54.15)],
        [sp(0, 24.85), sp(5.5, 24.85)], [sp(0, 43.15), sp(5.5, 43.15)],
        [sp(5.5, 24.85), sp(5.5, 43.15)],
        [sp(88.5, 13.85), sp(105, 13.85)], [sp(88.5, 54.15), sp(105, 54.15)],
        [sp(88.5, 13.85), sp(88.5, 54.15)],
        [sp(99.5, 24.85), sp(105, 24.85)], [sp(99.5, 43.15), sp(105, 43.15)],
        [sp(99.5, 24.85), sp(99.5, 43.15)],
    ]
    for (x1, y1), (x2, y2) in lines:
        ax.plot([x1, x2], [y1, y2], "-", lw=1.5, color=lines_color, alpha=0.9, zorder=3)

    cx_p, cy_p = 52.5 * L_SCALE, 34 * W_SCALE
    for centre, t1, t2 in [
        ((94.0 * L_SCALE, 34 * W_SCALE), 128, 232),
        ((11.0 * L_SCALE, 34 * W_SCALE), 308, 52),
        ((cx_p, cy_p), 0, 360),
    ]:
        r = 9.15 * L_SCALE if t2 == 360 else 9 * L_SCALE
        ax.add_patch(patches.Wedge(centre, r, t1, t2,
                                   fill=False, edgecolor=lines_color, width=0.02, zorder=3))

    ax.set_xlim(0, 120)
    ax.set_ylim(80, 0)
    ax.set_aspect("equal")
    ax.axis("off")


# -------------------------------------------------------
# Render static base pitch → PNG bytes (cached globally)
# -------------------------------------------------------
@st.cache_data
def render_base_pitch() -> bytes:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    draw_pitch(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI,
                bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------
# Render hexbin overlay → RGBA PNG bytes (transparent bg)
# -------------------------------------------------------
def render_hex_overlay(player_id: float, losses_by_player: dict,
                       centres: np.ndarray, hex_cells: list,
                       corner_hex_indices: set, color: str) -> bytes | None:
    """Returns a transparent-background PNG with only the hexbin layer,
    or None if the player has no data."""
    df = losses_by_player.get(player_id, pd.DataFrame())
    if df.empty:
        return None

    CUTOFF_X = 16.5 * L_SCALE
    df = df[df["location_x"] >= CUTOFF_X].copy()
    if df.empty:
        return None

    coords = df[["location_x", "location_y"]].to_numpy()
    idxs   = assign_hex(coords, centres)
    df["hex_idx"] = idxs
    df = df[~df["hex_idx"].isin(corner_hex_indices)]

    hc = df.groupby("hex_idx").size().reset_index(name="count")
    count_min, count_max = hc["count"].min(), hc["count"].max()
    hc["norm"] = (
        1.0 if count_max == count_min
        else (hc["count"] - count_min) / (count_max - count_min)
    )
    count_map = dict(zip(hc["hex_idx"], hc["norm"]))

    cmap  = mcolors.LinearSegmentedColormap.from_list("hm", ["#ffffff", color], N=256)
    GAMMA = 0.5

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 120); ax.set_ylim(80, 0)
    ax.set_aspect("equal"); ax.axis("off")

    for ext, codes, hex_idx in hex_cells:
        if hex_idx in corner_hex_indices:
            continue
        val = count_map.get(hex_idx, 0.0)
        if val <= 0:
            continue
        val_s  = val ** GAMMA
        c_rgba = cmap(val_s)
        alpha  = 0.15 + 0.80 * val_s
        ax.add_patch(PathPatch(Path(ext, codes),
                               facecolor=c_rgba, edgecolor="none",
                               alpha=alpha, linewidth=0.0, zorder=1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI,
                bbox_inches="tight", pad_inches=0,
                transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------
# Composite base pitch + hex overlay → final PNG bytes
# -------------------------------------------------------
def composite_pitch(base_bytes: bytes, overlay_bytes: bytes | None) -> bytes:
    from PIL import Image as PILImage
    base = PILImage.open(io.BytesIO(base_bytes)).convert("RGBA")
    if overlay_bytes is None:
        out = io.BytesIO()
        base.convert("RGB").save(out, format="png")
        out.seek(0)
        return out.getvalue()
    overlay = PILImage.open(io.BytesIO(overlay_bytes)).convert("RGBA")
    if overlay.size != base.size:
        overlay = overlay.resize(base.size, PILImage.LANCZOS)
    base.paste(overlay, (0, 0), mask=overlay)
    out = io.BytesIO()
    base.convert("RGB").save(out, format="png")
    out.seek(0)
    return out.getvalue()


# -------------------------------------------------------
# Build horizontal bar chart — % of team total per metric
# -------------------------------------------------------
def make_metric_chart(player_row: pd.Series, team_players: pd.DataFrame,
                      color: str) -> alt.Chart:
    rows = []
    for label, col in LOLLIPOP_METRICS:
        team_total = team_players[col].sum()
        player_val = float(player_row[col]) if pd.notna(player_row[col]) else 0.0
        pct        = (player_val / team_total * 100) if team_total > 0 else 0.0
        # Median % share across all squad players for this specific metric —
        # genuinely different per metric because distributions vary widely
        # (goals/xG are skewed towards attackers; passes are more evenly spread)
        if team_total > 0:
            median_pct = float(
                (team_players[col].fillna(0) / team_total * 100).median()
            )
        else:
            median_pct = 0.0
        rows.append({
            "metric":      label,
            "pct":         round(pct, 1),
            "median_pct":  round(median_pct, 1),
            "player_val":  player_val,
            "team_total":  team_total,
        })

    df = pd.DataFrame(rows)
    metric_order = [r["metric"] for r in rows][::-1]   # bottom→top
    x_max = 40

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("metric:N", sort=metric_order, title=None,
                    axis=alt.Axis(labelFontSize=10, labelLimit=160)),
            x=alt.X("pct:Q", title="% of team total",
                    scale=alt.Scale(domain=[0, x_max]),
                    axis=alt.Axis(labelFontSize=9, titleFontSize=10, tickMinStep=1)),
            color=alt.value(color),
            tooltip=[
                alt.Tooltip("metric:N",      title="Metric"),
                alt.Tooltip("player_val:Q",  title="Player value", format=".1f"),
                alt.Tooltip("team_total:Q",  title="Team total",   format=".1f"),
                alt.Tooltip("pct:Q",         title="% of team",    format=".1f"),
            ]
        )
    )

    median_tick = (
        alt.Chart(df)
        .mark_tick(
            color="#888888",
            strokeDash=[4, 3],
            thickness=1.5,
            bandSize=18,
        )
        .encode(
            y=alt.Y("metric:N", sort=metric_order),
            x=alt.X("median_pct:Q"),
            tooltip=[alt.Tooltip("median_pct:Q", title="Squad median %", format=".1f")],
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(align="left", dx=4, fontSize=9.5, color="#333333")
        .encode(
            y=alt.Y("metric:N", sort=metric_order),
            x=alt.X("pct:Q"),
            text=alt.Text("pct:Q", format=".1f"),
        )
    )

    return (
        (bars + median_tick + text)
        .properties(
            width=220, height=280,
            title=alt.TitleParams(
                text="% of Team Total per Metric",
                fontSize=11, fontWeight="bold", color="#333333",
                anchor="start", offset=5,
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


# -------------------------------------------------------
# Info card HTML helper
# -------------------------------------------------------
def info_card(label: str, value: str, color: str) -> str:
    return f"""
    <div style="
        background:#f8f8f8;
        border-left: 4px solid {color};
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
    ">
        <div style="font-size:11px; color:#888888; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.05em;">
            {label}
        </div>
        <div style="font-size:20px; font-weight:700; color:#222222; margin-top:3px;">
            {value}
        </div>
    </div>
    """


# -------------------------------------------------------
# App
# -------------------------------------------------------
centres, hex_cells, corner_hex_set                   = build_hex_grid()
top5_by_team, players_df, teams_df, losses_by_player = load_data(str(DB_PATH))

# ── Session state defaults ─────────────────────────────
if "ur_league" not in st.session_state:
    st.session_state.ur_league = "1. Bundesliga"
if "ur_team" not in st.session_state:
    st.session_state.ur_team = (
        teams_df[teams_df["league_name"] == "1. Bundesliga"]
        .sort_values("team_name")["team_name"].iloc[0]
    )
if "ur2_player_label" not in st.session_state:
    st.session_state.ur2_player_label = None

# ── on_change callbacks (no st.rerun() needed) ────────
def _on_league_change():
    new = st.session_state._ur2_league_widget
    st.session_state.ur_league = new
    st.session_state.ur_team   = (
        teams_df[teams_df["league_name"] == new]
        .sort_values("team_name")["team_name"].iloc[0]
    )
    st.session_state.ur2_player_label = None

def _on_team_change():
    st.session_state.ur_team         = st.session_state._ur2_team_widget
    st.session_state.ur2_player_label = None

def _on_player_change():
    st.session_state.ur2_player_label = st.session_state._ur2_player_widget

# ── Resolve current selections ─────────────────────────
selected_league = st.session_state.ur_league
selected_team   = st.session_state.ur_team
color           = LEAGUE_COLORS[selected_league]

team_row = teams_df[teams_df["team_name"] == selected_team]
team_id  = int(team_row["team_id"].iloc[0]) if not team_row.empty else None

# ── Title ──────────────────────────────────────────────
st.markdown("### Top Usage Players Study")


with st.expander("ℹ️ Chart information"):
    st.markdown(
        "**Usage Rate** measures the percentage of a team's possessions that end with a given player — "
        "it identifies who the ball consistently flows through before possession is lost. "
        "The **top 5 players by usage** for the selected team are shown in the left selector; "
        "switching between them updates both the pitch heatmap and the metrics barchart. "
        "The **pitch heatmap** shows the spatial distribution of ball losses for the selected player "
        "across the season, direction-normalised so the team always attacks left to right. "
        "Darker hexes indicate zones where the player more frequently ended possessions — "
        "this reveals whether a player operates centrally, wide, deep, or in advanced areas. "
        "The **barchart on the right** shows the selected player's share of the team's total "
        "for five metrics: Goals, Assists, Key Passes, Successful Passes, and xG. "
        "The **grey tick mark** on each bar represents the squad median percentage for that metric, "
        "so you can quickly see whether the player is above or below average for each contribution type. "
        "Use the league and team selectors to compare across different teams."
    )

# ── Layout — columns created once per run ──────────────
left_col, pitch_col, bar_col = st.columns([2, 4, 3], gap="small")

# Base pitch: globally cached, never redrawn
base_pitch_bytes = render_base_pitch()

# ── Left: selectors + position card ───────────────────
with left_col:
    st.markdown("**Select team**")

    st.selectbox(
        "League", LEAGUE_ORDER,
        index=LEAGUE_ORDER.index(selected_league),
        key="_ur2_league_widget",
        on_change=_on_league_change,
        label_visibility="collapsed",
    )

    league_teams = (
        teams_df[teams_df["league_name"] == selected_league]
        .sort_values("team_name")["team_name"].tolist()
    )
    team_idx = league_teams.index(selected_team) if selected_team in league_teams else 0
    st.selectbox(
        "Team", league_teams,
        index=team_idx,
        key="_ur2_team_widget",
        on_change=_on_team_change,
        label_visibility="collapsed",
    )

    top5 = top5_by_team.get(team_id, pd.DataFrame()) if team_id is not None else pd.DataFrame()

    player_id  = None
    player_row = None

    if not top5.empty:
        player_labels = [
            f"{i+1}. {row['player_name']} ({row['usage_pct']:.1f}%)"
            for i, row in top5.iterrows()
        ]
        if st.session_state.ur2_player_label not in player_labels:
            st.session_state.ur2_player_label = player_labels[0]

        st.markdown("**Select player**")
        st.selectbox(
            "Player", player_labels,
            index=player_labels.index(st.session_state.ur2_player_label),
            key="_ur2_player_widget",
            on_change=_on_player_change,
            label_visibility="collapsed",
        )

        selected_label = st.session_state._ur2_player_widget
        sel_idx        = player_labels.index(selected_label)
        player_row     = top5.iloc[sel_idx]
        player_id      = float(player_row["player_id"])
        pos_group      = player_row["position_group"] if pd.notna(player_row["position_group"]) else "N/A"

        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
        st.markdown(info_card("Position", pos_group, color), unsafe_allow_html=True)
    else:
        st.info("No usage data for this team.")

# ── Compute new pitch/bar if needed, then ALWAYS render ──
# Pitch: ensure final composite is in session state
if player_id is not None:
    hex_cache_key   = f"ur2_hex_{team_id}_{player_id}"
    pitch_cache_key = f"ur2_pitch_final_{team_id}_{player_id}"

    if hex_cache_key not in st.session_state:
        st.session_state[hex_cache_key] = render_hex_overlay(
            player_id, losses_by_player,
            centres, hex_cells, corner_hex_set, color,
        )
    if pitch_cache_key not in st.session_state:
        st.session_state[pitch_cache_key] = composite_pitch(
            base_pitch_bytes, st.session_state[hex_cache_key]
        )
    # Track the latest key so fallback always has something to show
    st.session_state["ur2_current_pitch_key"] = pitch_cache_key

# Always render pitch — new composite if ready, else last known, else base
with pitch_col:
    current_pitch_key = st.session_state.get("ur2_current_pitch_key")
    if current_pitch_key and current_pitch_key in st.session_state:
        st.image(st.session_state[current_pitch_key], use_container_width=True)
    else:
        st.image(base_pitch_bytes, use_container_width=True)

# Bar chart: ensure chart is in session state
if player_id is not None and player_row is not None:
    team_players  = players_df[players_df["team_id"] == team_id]
    full_player   = players_df[players_df["player_id"] == player_id]
    if not full_player.empty:
        bar_cache_key = f"ur2_bar_{team_id}_{player_id}"
        if bar_cache_key not in st.session_state:
            st.session_state[bar_cache_key] = make_metric_chart(
                full_player.iloc[0], team_players, color
            )
        st.session_state["ur2_current_bar_key"] = bar_cache_key

# Always render bar — new chart if ready, else last known
with bar_col:
    current_bar_key = st.session_state.get("ur2_current_bar_key")
    if current_bar_key and current_bar_key in st.session_state:
        st.altair_chart(st.session_state[current_bar_key], use_container_width=True)