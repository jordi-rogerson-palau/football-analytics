import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from PIL import Image
import pandas as pd
import duckdb
import streamlit as st
from pathlib import Path as FilePath
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
        /* Remove oval/rounded clipping on images */
        [data-testid="stImage"] img {
            border-radius: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

SCRIPT_DIR   = FilePath(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

PITCH_W = 120.0
PITCH_H = 80.0
APOTHEM = 2.0
L_SCALE = 120 / 105
W_SCALE = 80  / 68
FIG_W, FIG_H, DPI = 7, 4.67, 110

LEAGUE_ORDER   = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLOURS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

TURNOVER_TYPES = [
    "failed_pass", "interception", "failed_dribble",
    "lost_duel", "miscontrol", "dispossessed",
    "clearance", "block"
]

if "rec_x_range" not in st.session_state:
    st.session_state["rec_x_range"] = (0.0, 120.0)

# -------------------------------------------------------
# Hex grid — built once, fully cached
# -------------------------------------------------------
@st.cache_data
def build_hex_grid():
    pitch_box = shapely_box(0, 0, PITCH_W, PITCH_H)
    r      = APOTHEM * 2.0 / np.sqrt(3.0)
    hex_h  = 2.0 * APOTHEM
    col_dx = 1.5 * r
    row_dy = hex_h

    col_start = int(np.floor(0 / col_dx))
    col_end   = int(np.ceil(PITCH_W / col_dx)) + 1

    valid_centres = []
    hex_polys     = []

    for col in range(col_start, col_end + 1):
        cx       = col * col_dx
        y_offset = APOTHEM if (col % 2 == 1) else 0.0
        row_start = int(np.floor(-y_offset / row_dy)) - 1
        row_end   = int(np.ceil((PITCH_H - y_offset) / row_dy)) + 1

        for row in range(row_start, row_end + 1):
            cy = row * row_dy + APOTHEM + y_offset
            angles   = np.linspace(0, 2 * np.pi, 7)[:-1]
            verts_x  = cx + r * np.cos(angles)
            verts_y  = cy + r * np.sin(angles)
            hex_poly = ShapelyPolygon(zip(verts_x, verts_y))
            clipped  = hex_poly.intersection(pitch_box)
            if clipped.is_empty or clipped.area < 1e-6:
                continue
            valid_centres.append((cx, cy))
            hex_polys.append(clipped)

    valid_centres = np.array(valid_centres)

    hex_cells = []
    for clipped in hex_polys:
        cx_c  = np.mean([p[0] for p in clipped.exterior.coords])
        cy_c  = np.mean([p[1] for p in clipped.exterior.coords])
        dists = (valid_centres[:, 0] - cx_c) ** 2 + (valid_centres[:, 1] - cy_c) ** 2
        hex_idx = int(np.argmin(dists))

        ext_coords = np.array(clipped.exterior.coords).copy()
        ext_coords[:, 0] = PITCH_W - ext_coords[:, 0]
        hex_cells.append((ext_coords, hex_idx, cx_c))

    # Pre-build path codes for every cell — never recomputed
    poly_paths = []
    for ext_coords, hex_idx, cx_c in hex_cells:
        codes = ([Path.MOVETO]
                 + [Path.LINETO] * (len(ext_coords) - 2)
                 + [Path.CLOSEPOLY])
        poly_paths.append((ext_coords, codes, hex_idx))

    return valid_centres, hex_cells, poly_paths


# -------------------------------------------------------
# Data loading — one DB call, split into per-league dfs
# pre-filtered to turnover types, hex assigned once
# -------------------------------------------------------
@st.cache_data
def load_raw(_valid_centres_bytes):
    valid_centres = np.frombuffer(_valid_centres_bytes).reshape(-1, 2)

    con       = duckdb.connect(str(DB_PATH), read_only=True)
    sequences = con.execute(
        "SELECT match_id, team_id, end_x, end_y, end_type FROM sequences"
    ).df()
    teams = con.execute("SELECT team_id, league_name FROM teams").df()
    con.close()

    pt = sequences[sequences["end_type"].isin(TURNOVER_TYPES)].copy()
    pt = pt.dropna(subset=["end_x", "end_y"])
    pt = pt.merge(teams[["team_id", "league_name"]], on="team_id", how="left")

    # Vectorized hex assignment — avoid row-by-row apply
    coords    = pt[["end_x", "end_y"]].to_numpy()
    cx        = valid_centres[:, 0]
    cy        = valid_centres[:, 1]
    hex_idxs  = []
    # Process in chunks to avoid huge memory spike
    chunk = 5000
    for i in range(0, len(coords), chunk):
        c    = coords[i:i+chunk]
        dx   = cx[None, :] - c[:, 0:1]
        dy   = cy[None, :] - c[:, 1:2]
        idxs = np.argmin(dx**2 + dy**2, axis=1)
        hex_idxs.extend(idxs.tolist())
    pt["hex_idx"] = hex_idxs

    # Pre-split by league — O(1) lookup at render time
    league_dfs = {}
    for league in LEAGUE_ORDER:
        lg           = pt[pt["league_name"] == league].copy()
        n_matches    = lg["match_id"].nunique()
        league_dfs[league] = (lg, n_matches)

    return league_dfs


# -------------------------------------------------------
# Compute per-league count maps for a given x_range
# -------------------------------------------------------
def compute_counts(league_dfs, x_range):
    min_end_x = PITCH_W - x_range[1]
    max_end_x = PITCH_W - x_range[0]

    league_count_maps = {}
    global_max        = 0.0

    for league in LEAGUE_ORDER:
        lg, n_matches = league_dfs[league]
        filtered      = lg[(lg["end_x"] >= min_end_x) & (lg["end_x"] <= max_end_x)]
        hc            = filtered.groupby("hex_idx").size().reset_index(name="count")
        hc["per_game"] = hc["count"] / n_matches if n_matches > 0 else 0.0
        league_count_maps[league] = dict(zip(hc["hex_idx"], hc["per_game"]))
        if not hc.empty:
            global_max = max(global_max, hc["per_game"].max())

    return league_count_maps, global_max


# -------------------------------------------------------
# Figure factory — fresh figure per render call,
# safe across Streamlit pages (no shared module-level state)
# -------------------------------------------------------
def _make_fig():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _save_buf(fig, facecolor="white", transparent=False) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI,
                bbox_inches="tight", pad_inches=0,
                facecolor=facecolor, transparent=transparent)
    buf.seek(0)
    return buf


def _draw_pitch_on_ax(ax):
    ax.set_facecolor("#f8f5f0")
    def sp(x, y): return x * L_SCALE, y * W_SCALE
    lines = [
        [(0, 0),    (0, 80)],   [(120, 0),  (120, 80)],
        [(0, 80),   (120, 80)], [(0, 0),    (120, 0)],
        [(60, 0),   (60, 80)],
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
    ]
    for (x1, y1), (x2, y2) in lines:
        ax.plot([x1, x2], [y1, y2], "-", lw=1.2, color="#444444", alpha=0.8, zorder=3)
    cx, cy = 52.5 * L_SCALE, 34 * W_SCALE
    ax.add_patch(patches.Wedge((94.0 * L_SCALE, 34 * W_SCALE), 9 * L_SCALE, 128, 232,
                               fill=False, edgecolor="#444444", width=0.02, zorder=3))
    ax.add_patch(patches.Wedge((11.0 * L_SCALE, 34 * W_SCALE), 9 * L_SCALE, 308, 52,
                               fill=False, edgecolor="#444444", width=0.02, zorder=3))
    ax.add_patch(patches.Wedge((cx, cy), 9.15 * L_SCALE, 0, 360,
                               fill=False, edgecolor="#444444", width=0.02, zorder=3))
    ax.set_xlim(0, 120); ax.set_ylim(0, 87)
    ax.set_aspect("equal"); ax.axis("off")


# -------------------------------------------------------
# LAYER 1: static base — pitch + all hex outlines
# Rendered once per league, never redrawn
# -------------------------------------------------------
def render_base(league: str, poly_paths) -> Image.Image:
    colour  = LEAGUE_COLOURS[league]
    fig, ax = _make_fig()
    _draw_pitch_on_ax(ax)

    for ext_coords, codes, _ in poly_paths:
        ax.add_patch(PathPatch(
            Path(ext_coords, codes),
            facecolor="#eeeeee", edgecolor="none",
            alpha=0.3, linewidth=0.0, zorder=1
        ))

    ax.text(60, 83.5, league, color=colour, fontsize=22,
            fontweight="bold", ha="center", va="center", zorder=5)
    buf = _save_buf(fig, facecolor="white")
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


# -------------------------------------------------------
# LAYER 2: dynamic intensity — coloured hex cells only
# Redrawn on every slider change (cheap — no pitch lines)
# -------------------------------------------------------
def render_intensity(league: str, poly_paths,
                     count_map: dict, global_max: float,
                     base_size: tuple) -> Image.Image:
    colour  = LEAGUE_COLOURS[league]
    fig, ax = _make_fig()
    fig.set_facecolor((0, 0, 0, 0))
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_xlim(0, 120); ax.set_ylim(0, 87)
    ax.set_aspect("equal"); ax.axis("off")

    for ext_coords, codes, hex_idx in poly_paths:
        val = count_map.get(hex_idx, 0.0)
        if val <= 0:
            continue
        alpha = 0.10 + 0.80 * (val / global_max) if global_max > 0 else 0.10
        ax.add_patch(PathPatch(
            Path(ext_coords, codes),
            facecolor=colour, edgecolor="none",
            alpha=alpha, linewidth=0.0, zorder=1
        ))

    buf       = _save_buf(fig, facecolor=(0, 0, 0, 0), transparent=True)
    intensity = Image.open(buf).copy().convert("RGBA")
    plt.close(fig)
    if intensity.size != base_size:
        intensity = intensity.resize(base_size, Image.LANCZOS)
    return intensity


def composite(base_img: Image.Image, intensity_img: Image.Image) -> bytes:
    result = base_img.copy().convert("RGBA")
    result.paste(intensity_img, (0, 0), mask=intensity_img)
    buf = io.BytesIO()
    result.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------
# Layout
# -------------------------------------------------------
st.markdown("### Recuperations Distribution by Pitch Location")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each pitch shows a **hexbin heatmap of ball recuperation locations** for a given league — "
        "where on the pitch teams regained possession across the 2015/16 season. "
        "Colour intensity is proportional to recuperations per game, with all five leagues sharing "
        "the same scale so cross-league comparisons are direct — a darker hex means more recuperations "
        "in that zone relative to the global maximum. "
        "The **x-range slider** filters by pitch location: set both handles to the attacking third "
        "(x > 80) to isolate high-press recuperations, or to the defensive third (x < 40) to focus "
        "on deep defensive recuperations. "
        "The **table** in the bottom right updates with the slider and shows the average recuperations "
        "per game within the selected zone for each league — useful to compare pressing volume "
        "independently from spatial distribution. "
        "A high volume of recuperations in the opponent's half indicates an aggressive high-press "
        "mentality, while a concentration in the defensive third suggests a more passive, "
        "deep-block defensive approach."
    )

with st.spinner("Loading…"):
    valid_centres, hex_cells, poly_paths = build_hex_grid()
    league_dfs                            = load_raw(valid_centres.tobytes())

row1  = st.columns(3)
row2  = st.columns(3)
cells = row1 + row2

# Placeholders — images never disappear on rerun
placeholders = [cells[i].empty() for i in range(5)]

# Fill immediately from last cached composite
for i, league in enumerate(LEAGUE_ORDER):
    last_key = st.session_state.get(f"rec_last_composite_{league}")
    if last_key and last_key in st.session_state:
        placeholders[i].image(st.session_state[last_key], use_container_width=True)

# Slider + summary table in cell[5]
with cells[5]:
    # Centered title above slider
    st.markdown(
        "<p style='text-align:center; font-weight:bold; margin-bottom:0px;'>"
        "Recuperation x range</p>",
        unsafe_allow_html=True,
    )
    x_range = st.slider(
        "Recuperation x range",
        min_value=0.0, max_value=120.0, step=1.0,
        key="rec_x_range",
        label_visibility="collapsed",
    )

# Compute count maps for current filter (cheap — just groupby on pre-split dfs)
x_range                       = st.session_state["rec_x_range"]
league_count_maps, global_max = compute_counts(league_dfs, x_range)

# Render loop — base cached forever, intensity cached per (league, x_range)
x_key = f"{x_range[0]:.0f}_{x_range[1]:.0f}"

for i, league in enumerate(LEAGUE_ORDER):
    composite_key = f"rec_composite_{league}_{x_key}"

    if composite_key not in st.session_state:
        base_key = f"rec_base_{league}"
        if base_key not in st.session_state:
            st.session_state[base_key] = render_base(league, poly_paths)
        base_img = st.session_state[base_key]

        intensity_img = render_intensity(
            league, poly_paths,
            league_count_maps[league], global_max,
            base_img.size
        )
        st.session_state[composite_key] = composite(base_img, intensity_img)

    st.session_state[f"rec_last_composite_{league}"] = composite_key
    placeholders[i].image(st.session_state[composite_key], use_container_width=True)

# Summary table
# Rendered as HTML so we can control exact height, equal row heights,
# and no scrollbar — st.dataframe has a fixed 35px row height that
# causes scroll when the available space is smaller than 6*35=210px.
# TABLE_HEIGHT matches available space: pitch_h(261px) - slider+title(103px) = 158px
TABLE_HEIGHT = 158
with cells[5]:
    summary_rows = []
    for league in LEAGUE_ORDER:
        total_per_game = sum(league_count_maps[league].values())
        summary_rows.append({
            "league":  league,
            "value":   f"{total_per_game:.2f}",
        })
    summary_rows = sorted(summary_rows, key=lambda r: r["league"])

    row_h = TABLE_HEIGHT / 6  # 1 header + 5 data rows

    header_html = (
        f"<tr style='height:{row_h:.1f}px;'>"
        f"<th style='text-align:left; padding:0 8px;"
        f"background:#f0f0f0; border-bottom:1px solid #ddd;'>League</th>"
        f"<th style='text-align:center; padding:0 8px;"
        f"background:#f0f0f0; border-bottom:1px solid #ddd;'>Recup / game</th>"
        f"</tr>"
    )

    rows_html = ""
    for r in summary_rows:
        c = LEAGUE_COLOURS.get(r["league"], "#888888")
        rows_html += (
            f"<tr style='height:{row_h:.1f}px; background:{c}22;'>"
            f"<td style='padding:0 8px; font-weight:bold;"
            f"color:{c};'>{r['league']}</td>"
            f"<td style='padding:0 8px; text-align:center;"
            f"color:#222222;'>{r['value']}</td>"
            f"</tr>"
        )

    table_html = (
        f"<div style='width:100%; height:{TABLE_HEIGHT}px; overflow:hidden;'>"
        f"<table style='width:100%; height:{TABLE_HEIGHT}px; border-collapse:collapse;"
        f"table-layout:fixed;'>"
        f"<thead>{header_html}</thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>"
    )

    st.markdown(table_html, unsafe_allow_html=True)