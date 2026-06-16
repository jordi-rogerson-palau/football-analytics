import io
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import streamlit as st

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent   # pages/
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH      = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
FIELD_LENGTH = 120
FIELD_WIDTH  = 80
HALFWAY_X    = 60

BIN_LABELS = ["0–10", "10–20", "20–30", "30–40", "40–50", "50–60"]

LEAGUE_ORDER  = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

HALF1_ALPHA = 0.88
HALF2_ALPHA = 0.45
L_SCALE     = 120 / 105
W_SCALE     = 80  / 68


# -------------------------------------------------------
# Data loading — queries pre-aggregated tables only.
# pass_bin_agg:  60 rows  (was 1,777,390)
# pass_metrics:   5 rows  (was 1,777,390)
# -------------------------------------------------------
@st.cache_data(show_spinner="Loading pass data…")
def load_data(db_path: str):
    con = duckdb.connect(db_path, read_only=True)

    bin_agg = con.execute("""
        SELECT league_name, half, bin_label, passes_per_game, avg_length
        FROM pass_bin_agg
    """).df()

    metrics = con.execute("""
        SELECT league_name,
               total_passes_per_game,
               completion_rate,
               prog_passes_per_game,
               prog_completion
        FROM pass_metrics
    """).df()

    con.close()

    global_max = float(bin_agg["passes_per_game"].max())

    wav = (
        bin_agg.assign(w_len=bin_agg["passes_per_game"] * bin_agg["avg_length"])
        .groupby(["league_name", "half"])
        .apply(lambda g: g["w_len"].sum() / g["passes_per_game"].sum() if g["passes_per_game"].sum() > 0 else 0)
        .reset_index(name="wavg_length")
    )
    avg_dict = {}
    for _, row in wav.iterrows():
        avg_dict.setdefault(row["league_name"], {})[row["half"]] = row["wavg_length"]

    return bin_agg, metrics, avg_dict, global_max


# -------------------------------------------------------
# Pitch drawing helper
# -------------------------------------------------------
def _pt(x_raw, y_raw):
    return x_raw * L_SCALE, y_raw * W_SCALE


def draw_pitch_ax(ax, lines_color="#b8b8b8", bg_color="#f4f4ec"):
    lw, la = 2.0, 0.80

    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], "-", color=lines_color, lw=lw, alpha=la, zorder=2)

    line(0, 0, FIELD_LENGTH, 0)
    line(0, FIELD_WIDTH, FIELD_LENGTH, FIELD_WIDTH)
    line(0, 0, 0, FIELD_WIDTH)
    line(FIELD_LENGTH, 0, FIELD_LENGTH, FIELD_WIDTH)
    line(HALFWAY_X, 0, HALFWAY_X, FIELD_WIDTH)

    raw_lines = [
        ((0, 13.85),    (16.5, 13.85)), ((0, 54.15),    (16.5, 54.15)),
        ((16.5, 13.85), (16.5, 54.15)),
        ((0, 24.85),    (5.5, 24.85)),  ((0, 43.15),    (5.5, 43.15)),
        ((5.5, 24.85),  (5.5, 43.15)),
        ((105, 13.85),  (88.5, 13.85)), ((105, 54.15),  (88.5, 54.15)),
        ((88.5, 13.85), (88.5, 54.15)),
        ((105, 24.85),  (99.5, 24.85)), ((105, 43.15),  (99.5, 43.15)),
        ((99.5, 24.85), (99.5, 43.15)),
    ]
    for (x1r, y1r), (x2r, y2r) in raw_lines:
        line(*_pt(x1r, y1r), *_pt(x2r, y2r))

    ax.add_patch(patches.Rectangle(
        (0, 0), FIELD_LENGTH, FIELD_WIDTH, facecolor=bg_color, zorder=0))
    ax.add_patch(patches.Wedge(
        (HALFWAY_X, FIELD_WIDTH / 2), 9.15 * L_SCALE, 0, 360,
        fill=False, edgecolor=lines_color, lw=lw, alpha=la, width=0.02, zorder=2))
    ax.add_patch(patches.Wedge(
        _pt(11, 34), 9 * L_SCALE, 308, 52,
        fill=False, edgecolor=lines_color, lw=lw, alpha=la, width=0.02, zorder=2))
    ax.add_patch(patches.Wedge(
        _pt(94, 34), 9 * L_SCALE, 128, 232,
        fill=False, edgecolor=lines_color, lw=lw, alpha=la, width=0.02, zorder=2))


# -------------------------------------------------------
# Per-league pitch figure
# -------------------------------------------------------
def make_league_figure_bytes(league_name: str,
                              bin_agg: pd.DataFrame,
                              avg_dict: dict,
                              global_max: float) -> bytes:
    color  = LEAGUE_COLORS[league_name]
    lg_agg = bin_agg[bin_agg["league_name"] == league_name]

    def get_counts(half_name: str) -> pd.Series:
        sub = lg_agg[lg_agg["half"] == half_name].set_index("bin_label")["passes_per_game"]
        return pd.Series([sub.get(lbl, 0.0) for lbl in BIN_LABELS], index=BIN_LABELS)

    c_own   = get_counts("Own Half")
    c_opp   = get_counts("Opposition Half")
    avgs    = avg_dict.get(league_name, {})
    avg_own = avgs.get("Own Half", None)
    avg_opp = avgs.get("Opposition Half", None)
    n_bins  = len(BIN_LABELS)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), facecolor="white")
    draw_pitch_ax(ax)

    MARGIN   = 2
    Z1_LEFT  = MARGIN
    Z1_RIGHT = HALFWAY_X - MARGIN
    Z2_LEFT  = HALFWAY_X + MARGIN
    Z1_W     = Z1_RIGHT - Z1_LEFT
    Z2_W     = (FIELD_LENGTH - MARGIN) - Z2_LEFT

    gap1   = Z1_W / n_bins
    gap2   = Z2_W / n_bins
    bar_w1 = gap1 * 0.76
    bar_w2 = gap2 * 0.76
    BASE_Y = 0

    MAX_BAR_H = FIELD_WIDTH * 0.72

    def ph(count):
        return (count / global_max) * MAX_BAR_H if global_max > 0 else 0

    for i, lbl in enumerate(BIN_LABELS):
        bx1 = Z1_LEFT + i * gap1 + (gap1 - bar_w1) / 2
        ax.add_patch(patches.Rectangle(
            (bx1, BASE_Y), bar_w1, ph(c_own[lbl]),
            facecolor=color, alpha=HALF1_ALPHA,
            edgecolor="white", linewidth=0.4, zorder=4))

        bx2 = Z2_LEFT + i * gap2 + (gap2 - bar_w2) / 2
        ax.add_patch(patches.Rectangle(
            (bx2, BASE_Y), bar_w2, ph(c_opp[lbl]),
            facecolor=color, alpha=HALF2_ALPHA,
            edgecolor="white", linewidth=0.4, zorder=4))

    TEXT_PAD = 1.2
    if avg_own is not None and c_own.max() > 0:
        ax.text(Z1_LEFT + Z1_W / 2, ph(c_own.max()) + TEXT_PAD,
                f"avg {avg_own:.1f} m",
                ha="center", va="bottom", fontsize=7.0,
                color=color, fontweight="bold", zorder=7)
    if avg_opp is not None and c_opp.max() > 0:
        ax.text(Z2_LEFT + Z2_W / 2, ph(c_opp.max()) + TEXT_PAD,
                f"avg {avg_opp:.1f} m",
                ha="center", va="bottom", fontsize=7.0,
                color=color, fontweight="bold", zorder=7)

    TOP_Y = FIELD_WIDTH + 1.0
    ax.text(Z1_LEFT + Z1_W / 2, TOP_Y, "Own Half",
            ha="center", va="bottom", fontsize=8.0,
            color="#222222", fontweight="bold", zorder=5)
    ax.text(Z2_LEFT + Z2_W / 2, TOP_Y, "Opposition Half",
            ha="center", va="bottom", fontsize=8.0,
            color="#222222", fontweight="bold", zorder=5)

    LABEL_Y = -1.8
    for i, lbl in enumerate(BIN_LABELS):
        for zone_left, gap in [(Z1_LEFT, gap1), (Z2_LEFT, gap2)]:
            tx = zone_left + i * gap + gap / 2
            ax.text(tx, LABEL_Y, lbl, ha="center", va="top",
                    fontsize=7.5, color="#555555", rotation=38, zorder=5)

    ax.set_xlim(-12, FIELD_LENGTH + 2)
    ax.set_ylim(-7, FIELD_WIDTH + 8)
    ax.set_aspect("equal")

    # Y-axis ticks
    n_ticks    = 5
    raw_step   = global_max / n_ticks
    magnitude  = 10 ** np.floor(np.log10(raw_step)) if raw_step > 0 else 1
    nice_steps = [1, 2, 5, 10, 20, 25, 50, 100]
    tick_step  = next(
        (s * magnitude for s in nice_steps if s * magnitude >= raw_step),
        magnitude * nice_steps[-1]
    )
    tick_vals, tv = [], 0.0
    while tv <= global_max * 1.05:
        tick_vals.append(tv)
        tv = round(tv + tick_step, 10)

    TICK_X, TICK_LEN, LABEL_X = 0, 1.5, -2.5
    ax.plot([TICK_X, TICK_X], [0, MAX_BAR_H], "-",
            color="#888888", lw=0.8, alpha=0.6, zorder=3)
    for tv in tick_vals:
        ty = ph(tv)
        ax.plot([TICK_X - TICK_LEN, TICK_X], [ty, ty], "-",
                color="#888888", lw=0.8, alpha=0.7, zorder=3)
        ax.plot([0, FIELD_LENGTH], [ty, ty], "-",
                color="#cccccc", lw=0.4, alpha=0.5, zorder=1)
        label = str(int(tv)) if tv == int(tv) else f"{tv:.1f}"
        ax.text(LABEL_X, ty, label, ha="right", va="center",
                fontsize=7.5, color="#555555", zorder=5)

    ax.text(-9, MAX_BAR_H / 2, "Successful passes per game",
            ha="center", va="center", fontsize=7.5,
            color="#555555", rotation=90, zorder=5)

    # League title drawn inside axes headroom so PNG bounding box is unaffected.
    # Y = FIELD_WIDTH + 5.5 (original) + 2.399 (one benchmark unit = fontsize-8
    # of the Own/Opp Half labels converted to data units at this figsize/dpi).
    ax.text(FIELD_LENGTH / 2, FIELD_WIDTH + 7.9, league_name,
            ha="center", va="center", fontsize=22.0,
            fontweight="bold", color=color, zorder=5)

    ax.axis("off")
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------
# Metrics chart
# -------------------------------------------------------
def make_metrics_chart_bytes(metrics: pd.DataFrame, global_max: float) -> bytes:
    FIELD_WIDTH  = 80
    MAX_BAR_H    = FIELD_WIDTH * 0.72
    Y_BOT        = -7
    Y_TOP        = FIELD_WIDTH + 8

    leagues  = LEAGUE_ORDER
    n        = len(leagues)
    totals   = {r["league_name"]: r["total_passes_per_game"]  for _, r in metrics.iterrows()}
    tot_comp = {r["league_name"]: r["completion_rate"]        for _, r in metrics.iterrows()}
    progs    = {r["league_name"]: r["prog_passes_per_game"]   for _, r in metrics.iterrows()}
    prg_comp = {r["league_name"]: r["prog_completion"]        for _, r in metrics.iterrows()}

    all_vals  = [totals[lg] for lg in leagues] + [progs[lg] for lg in leagues]
    local_max = max(all_vals) if all_vals else 1.0

    def to_data(v):
        return (v / local_max) * MAX_BAR_H

    FIELD_LENGTH = 120
    group_w  = FIELD_LENGTH / n
    bar_w    = group_w * 0.30
    gap      = group_w * 0.06
    x_starts = [i * group_w + group_w / 2 for i in range(n)]

    fig, ax = plt.subplots(figsize=(6.6, 4.4), facecolor="white")

    n_ticks   = 5
    tick_step_v = local_max / n_ticks
    magnitude = 10 ** np.floor(np.log10(tick_step_v)) if tick_step_v > 0 else 1
    nice_steps = [1, 2, 5, 10, 20, 25, 50, 100]
    tick_step_v = next(
        (s * magnitude for s in nice_steps if s * magnitude >= tick_step_v),
        magnitude * nice_steps[-1]
    )
    tick_vals_v, tv = [], 0.0
    while tv <= local_max * 1.05:
        tick_vals_v.append(tv)
        tv = round(tv + tick_step_v, 10)

    for tv in tick_vals_v:
        ty = to_data(tv)
        ax.plot([0, FIELD_LENGTH], [ty, ty], "-", color="#cccccc", lw=0.4, alpha=0.5, zorder=1)

    for i, lg in enumerate(leagues):
        cx    = x_starts[i]
        color = LEAGUE_COLORS[lg]

        tot      = totals[lg]
        c_rate   = tot_comp[lg] / 100.0
        h_comp   = to_data(tot * c_rate)
        h_incomp = to_data(tot * (1 - c_rate))
        bx_tot   = cx - gap / 2 - bar_w

        ax.add_patch(patches.Rectangle(
            (bx_tot, 0), bar_w, h_comp + h_incomp,
            facecolor=color, alpha=0.28, edgecolor="none", zorder=3))
        ax.add_patch(patches.Rectangle(
            (bx_tot, 0), bar_w, h_comp,
            facecolor=color, alpha=HALF1_ALPHA, edgecolor="none", zorder=4))

        prog      = progs[lg]
        p_rate    = prg_comp[lg] / 100.0
        h_pcomp   = to_data(prog * p_rate)
        h_pincomp = to_data(prog * (1 - p_rate))
        bx_prog   = cx + gap / 2

        ax.add_patch(patches.Rectangle(
            (bx_prog, 0), bar_w, h_pcomp + h_pincomp,
            facecolor=color, alpha=0.28, edgecolor="none", zorder=3))
        ax.add_patch(patches.Rectangle(
            (bx_prog, 0), bar_w, h_pcomp,
            facecolor=color, alpha=HALF1_ALPHA, edgecolor="none", zorder=4))

        TEXT_PAD = 1.0
        ax.text(bx_tot + bar_w / 2, to_data(tot) + TEXT_PAD,
                f"{tot_comp[lg]:.1f}%",
                ha="center", va="bottom", fontsize=6.0,
                color=color, fontweight="bold", zorder=7)
        ax.text(bx_prog + bar_w / 2, to_data(prog) + TEXT_PAD,
                f"{prg_comp[lg]:.1f}%",
                ha="center", va="bottom", fontsize=6.0,
                color=color, fontweight="bold", zorder=7)

        ax.text(cx, Y_BOT + 1.0, lg,
                ha="center", va="top", fontsize=7.0,
                color=color, fontweight="bold",
                rotation=0, zorder=5)

    TICK_X, TICK_LEN, LABEL_X = 0, 1.5, -2.5
    ax.plot([TICK_X, TICK_X], [0, MAX_BAR_H], "-",
            color="#888888", lw=0.8, alpha=0.6, zorder=3)
    for tv in tick_vals_v:
        ty = to_data(tv)
        ax.plot([TICK_X - TICK_LEN, TICK_X], [ty, ty], "-",
                color="#888888", lw=0.8, alpha=0.7, zorder=3)
        label = str(int(tv)) if tv == int(tv) else f"{tv:.1f}"
        ax.text(LABEL_X, ty, label, ha="right", va="center",
                fontsize=7.5, color="#555555", zorder=5)

    ax.text(-9, MAX_BAR_H / 2, "Avg passes per game",
            ha="center", va="center", fontsize=7.5,
            color="#555555", rotation=90, zorder=5)

    BOX_H    = 2.8
    legend_y = MAX_BAR_H + (Y_TOP - MAX_BAR_H) * 0.38
    items    = [("Total passes", HALF1_ALPHA), ("Progressive passes", HALF2_ALPHA)]
    item_w   = FIELD_LENGTH * 0.38
    total_w  = len(items) * item_w
    legend_x0 = (FIELD_LENGTH - total_w) / 2
    for j, (lbl, alpha) in enumerate(items):
        bx = legend_x0 + j * item_w
        ax.add_patch(patches.Rectangle(
            (bx, legend_y - BOX_H / 2), 4.5, BOX_H,
            facecolor="#555555", alpha=alpha, edgecolor="none", zorder=6))
        ax.text(bx + 6.0, legend_y, lbl,
                ha="left", va="center", fontsize=6.5,
                color="#444444", zorder=7)

    ax.set_xlim(-12, FIELD_LENGTH + 2)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Passes per Game · Completion Rate",
        fontsize=11.0, fontweight="bold", pad=4, color="#333333"
    )
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                pad_inches=0, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
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

    st.markdown("### Pass Length Distribution by Pitch Half")

    with st.expander("ℹ️ Chart information"):
        st.markdown(
            "Each pitch shows the **distribution of successful passes per game** binned by distance "
            "(0–10m, 10–20m … 50–60m), split between a team's **own half** (left, full opacity) "
            "and the **opposition half** (right, faded). "
            "Bar heights are scaled consistently across all leagues so cross-league comparisons are direct. "
            "The **avg Xm** label above each half marks the passes-per-game-weighted mean pass length. "
            "The bottom-right panel shows total and progressive passes per game for each league, "
            "with opacity reflecting completion rate — darker fill = higher % of successful passes."
        )

    if not Path(DB_PATH).exists():
        st.error(f"Database not found at `{DB_PATH}`.")
        st.stop()

    bin_agg, metrics, avg_dict, global_max = load_data(str(DB_PATH))

    row1  = st.columns(3, gap="small")
    row2  = st.columns(3, gap="small")
    slots = row1 + row2

    for slot, league in zip(slots[:5], LEAGUE_ORDER):
        cache_key = f"bcp_fig_{league}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = make_league_figure_bytes(
                league, bin_agg, avg_dict, global_max
            )
        with slot:
            st.image(st.session_state[cache_key], use_container_width=True)

    with slots[5]:
        metrics_key = "bcp_metrics_chart"
        if metrics_key not in st.session_state:
            st.session_state[metrics_key] = make_metrics_chart_bytes(metrics, global_max)
        st.image(st.session_state[metrics_key], use_container_width=True)


main()