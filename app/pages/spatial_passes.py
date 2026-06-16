import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon as ShapelyPolygon
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
    </style>
    """,
    unsafe_allow_html=True,
)

SCRIPT_DIR   = FilePath(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

L_SCALE = 120 / 105
W_SCALE = 80  / 68

LEAGUE_ORDER   = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLOURS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

# Shared figure dimensions — must be consistent between base and arrows renders
FIG_W, FIG_H, DPI = 7, 4.67, 110

# ── Session state defaults ─────────────────────────────
for _k, _v in [("top_n", 5), ("min_start_x", 0.0)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# -------------------------------------------------------
# Data loading — all in one DB call, pre-sorted once
# -------------------------------------------------------
@st.cache_data
def build_hex_grid_from_db():
    con         = duckdb.connect(str(DB_PATH), read_only=True)
    centres_df  = con.execute("SELECT hex_idx, cx, cy FROM hex_centres ORDER BY hex_idx").df()
    polygons_df = con.execute(
        'SELECT hex_idx, vx, vy, "order" FROM hex_polygons ORDER BY hex_idx, "order"'
    ).df()
    con.close()

    centres   = centres_df[["cx", "cy"]].to_numpy()
    hex_ids   = centres_df["hex_idx"].tolist()   # actual hex_idx values in order
    clipped_polys = []
    for idx in hex_ids:
        verts = polygons_df[polygons_df["hex_idx"] == idx].sort_values("order")
        poly  = ShapelyPolygon(zip(verts["vx"], verts["vy"]))
        clipped_polys.append(poly)

    # Map hex_idx → positional index in centres/clipped_polys arrays
    hex_to_pos = {hid: pos for pos, hid in enumerate(hex_ids)}
    return centres, clipped_polys, hex_to_pos


@st.cache_data
def precompute_poly_paths(_clipped_polys):
    """Pre-extract (coords, codes) for every polygon — never recomputed."""
    result = []
    for poly in _clipped_polys:
        ext   = np.array(poly.exterior.coords)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(ext) - 2) + [Path.CLOSEPOLY]
        result.append((ext, codes))
    return result


@st.cache_data
def load_all_transitions(_centres, _hex_to_pos):
    """
    One DB call for all leagues.
    Uses hex_to_pos to map hex_idx → position in centres array,
    guaranteed consistent with build_hex_grid_from_db.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute("SELECT league_name, src, dst, count FROM hex_transitions").df()
    con.close()

    cx_map = {hid: _centres[pos, 0] for hid, pos in _hex_to_pos.items()}
    cy_map = {hid: _centres[pos, 1] for hid, pos in _hex_to_pos.items()}

    result = {}
    for league, grp in df.groupby("league_name"):
        g = grp.drop(columns="league_name").copy()
        g["src_cx"] = g["src"].map(cx_map)
        g["src_cy"] = g["src"].map(cy_map)
        g["dst_cx"] = g["dst"].map(cx_map)
        g["dst_cy"] = g["dst"].map(cy_map)
        g = g.dropna(subset=["src_cx", "src_cy", "dst_cx", "dst_cy"])
        g = g[g["src"] != g["dst"]]
        g = g.sort_values("count", ascending=False).reset_index(drop=True)
        result[league] = g
    return result


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


def _fig_to_pil(fig) -> Image.Image:
    buf = _save_buf(fig, facecolor="white")
    return Image.open(buf).copy()


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------
# LAYER 1: static base (pitch lines + all hex polygons)
# Cached per league in session_state — never redrawn once built
# -------------------------------------------------------
def _draw_pitch_on_ax(ax, lines_color="#444444"):
    ax.set_facecolor("white")
    def sp(x, y): return x * L_SCALE, y * W_SCALE
    lines = [
        [(0, 0),   (0, 80)],   [(120, 0), (120, 80)],
        [(0, 80),  (120, 80)], [(0, 0),   (120, 0)],
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
    ]
    for (x1, y1), (x2, y2) in lines:
        ax.plot([x1, x2], [y1, y2], "-", lw=1.2, color=lines_color, alpha=0.8, zorder=3)
    cx, cy = 52.5 * L_SCALE, 34 * W_SCALE
    ax.add_patch(patches.Wedge((94.0 * L_SCALE, 34 * W_SCALE), 9 * L_SCALE, 128, 232,
                               fill=False, edgecolor=lines_color, width=0.02, zorder=3))
    ax.add_patch(patches.Wedge((11.0 * L_SCALE, 34 * W_SCALE), 9 * L_SCALE, 308, 52,
                               fill=False, edgecolor=lines_color, width=0.02, zorder=3))
    ax.add_patch(patches.Wedge((cx, cy), 9.15 * L_SCALE, 0, 360,
                               fill=False, edgecolor=lines_color, width=0.02, zorder=3))
    ax.set_xlim(0, 120); ax.set_ylim(0, 87)
    ax.set_aspect("equal"); ax.axis("off")


def render_base(league: str, poly_paths) -> Image.Image:
    colour  = LEAGUE_COLOURS[league]
    fig, ax = _make_fig()
    _draw_pitch_on_ax(ax)

    for ext, codes in poly_paths:
        ax.add_patch(PathPatch(
            Path(ext, codes),
            facecolor="#eeeeee", edgecolor="#aaaaaa",
            alpha=0.6, linewidth=0.9, zorder=1
        ))

    # Title drawn inside axes headroom (y=83, above pitch top at y=80)
    # so PNG bounding box is unaffected — fontsize=22 matches barchart_pass
    ax.text(60, 83.5, league, color=colour, fontsize=22,
            fontweight="bold", ha="center", va="center", zorder=5)
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


# -------------------------------------------------------
# LAYER 2: dynamic arrows (only redrawn on filter change)
# Transparent background so it composites cleanly over base
# -------------------------------------------------------
def render_arrows(league: str, top: pd.DataFrame,
                  poly_paths, hex_to_pos: dict, base_size: tuple) -> Image.Image:
    colour  = LEAGUE_COLOURS[league]
    fig, ax = _make_fig()
    fig.set_facecolor((0, 0, 0, 0))
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_xlim(0, 120); ax.set_ylim(0, 87)
    ax.set_aspect("equal"); ax.axis("off")

    if not top.empty:
        max_count = top["count"].max()
        src_ids   = set(top["src"].tolist())
        dst_ids   = set(top["dst"].tolist())

        for hex_idx in src_ids | dst_ids:
            pos = hex_to_pos.get(hex_idx)
            if pos is None:
                continue
            ext, codes = poly_paths[pos]
            is_src     = hex_idx in src_ids
            ax.add_patch(PathPatch(
                Path(ext, codes),
                facecolor=colour if is_src else "#ffffff",
                edgecolor=colour,
                alpha=0.35 if is_src else 0.20,
                linewidth=1.2, zorder=2
            ))

        for _, row in top.iterrows():
            count = int(row["count"])
            lw    = 0.8 + 2.5 * (count / max_count)
            alpha = 0.45 + 0.50 * (count / max_count)
            ax.annotate(
                "",
                xy=(row["dst_cx"], row["dst_cy"]),
                xytext=(row["src_cx"], row["src_cy"]),
                arrowprops=dict(
                    arrowstyle="-|>", color=colour, lw=lw,
                    mutation_scale=10 + 4 * (count / max_count)
                ),
                alpha=alpha, zorder=5
            )

    buf        = _save_buf(fig, facecolor=(0, 0, 0, 0), transparent=True)
    arrows_img = Image.open(buf).copy().convert("RGBA")
    plt.close(fig)

    if arrows_img.size != base_size:
        arrows_img = arrows_img.resize(base_size, Image.LANCZOS)
    return arrows_img


# -------------------------------------------------------
# Composite: paste arrows layer over base layer
# -------------------------------------------------------
def composite(base_img: Image.Image, arrows_img: Image.Image) -> bytes:
    result = base_img.copy().convert("RGBA")
    result.paste(arrows_img, (0, 0), mask=arrows_img)
    buf = io.BytesIO()
    result.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------
# Layout
# -------------------------------------------------------
st.markdown("### Most Frequent Pass Transitions")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Each pitch shows the most frequent **pass transitions** between hexagonal zones for a given league. "
        "The pitch is divided into a hexbin grid — hexagons are used instead of squares to better approximate "
        "real spatial distances between players on the field. "
        "**Filled (coloured) hexagons** are the origin zones of the top N transitions; "
        "**outlined hexagons** are the destination zones. "
        "**Arrows** connect origin to destination, with thickness and opacity scaling with transition frequency — "
        "thicker and darker means that pair of zones exchanged the ball more often. "
        "Use the **Top N** slider to control how many transitions are shown (higher N adds clutter but reveals "
        "secondary patterns). "
        "Use the **Min starting x** slider to restrict transitions to a specific area of the pitch: "
        "0 shows the whole pitch, 60 filters to the opponent's half only, and ~101 isolates passes "
        "originating inside the penalty box. "
        "Note that with a very high Top N all leagues tend to converge to similar distributions — "
        "lower values of N reveal the most distinctive passing tendencies per league."
    )

with st.spinner("Loading…"):
    centres, clipped_polys, hex_to_pos = build_hex_grid_from_db()
    poly_paths                          = precompute_poly_paths(clipped_polys)
    all_transitions                     = load_all_transitions(centres, hex_to_pos)

row1  = st.columns(3)
row2  = st.columns(3)
cells = row1 + row2

# Create placeholders — images never disappear
placeholders = [cells[i].empty() for i in range(5)]

# Fill immediately from cache if available
for i, league in enumerate(LEAGUE_ORDER):
    last_key = st.session_state.get(f"sp_last_composite_{league}")
    if last_key and last_key in st.session_state:
        placeholders[i].image(st.session_state[last_key], use_container_width=True)

# Sliders
with cells[5]:
    top_n       = st.slider("Top N transitions", 1, 30, key="top_n")
    min_start_x = st.slider(
        "Min starting x",
        min_value=0.0, max_value=110.0, step=1.0,
        key="min_start_x"
    )

# ── Render loop ───────────────────────────────────────
for i, league in enumerate(LEAGUE_ORDER):
    composite_key = f"sp_composite_{league}_{top_n}_{int(min_start_x)}"

    if composite_key not in st.session_state:

        base_key = f"sp_base_{league}"
        if base_key not in st.session_state:
            st.session_state[base_key] = render_base(league, poly_paths)
        base_img = st.session_state[base_key]

        trans_df = all_transitions.get(league, pd.DataFrame())
        filtered = (
            trans_df[trans_df["src_cx"] >= min_start_x]
            .sort_values("count", ascending=False)
            .iloc[:top_n]
        )
        arrows_img = render_arrows(league, filtered, poly_paths, hex_to_pos, base_img.size)

        st.session_state[composite_key] = composite(base_img, arrows_img)

    st.session_state[f"sp_last_composite_{league}"] = composite_key
    placeholders[i].image(st.session_state[composite_key], use_container_width=True)