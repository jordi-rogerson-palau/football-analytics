import altair as alt
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import io
from pathlib import Path
import streamlit as st

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DUCKDB_PATH  = PROJECT_ROOT.parent / "statsbomb_2015_2016.duckdb"

LEAGUE_ORDER  = ["1. Bundesliga", "La Liga", "Ligue 1", "Premier League", "Serie A"]
LEAGUE_COLORS = {
    "1. Bundesliga": "#d62728",
    "La Liga":       "#1f77b4",
    "Ligue 1":       "#9467bd",
    "Premier League":"#ff7f0e",
    "Serie A":       "#2ca02c",
}

# rows = leagues, columns = quadrants
QUADRANT_KEYS   = ["HH", "LH", "HL", "LL"]
QUADRANT_LABELS = [
    "High passes\nHigh speed",
    "Low passes\nHigh speed",
    "High passes\nLow speed",
    "Low passes\nLow speed",
]

st.markdown(
    """
    <style>
        #MainMenu            { visibility: hidden; }
        header               { visibility: hidden; }
        footer               { visibility: hidden; }
        .block-container     {
                                padding-top: 0rem !important;
                                margin-top: -1rem !important;
                                padding-left: 1rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Passes per Sequence vs Upfield Speed")

with st.expander("ℹ️ Information on the chart"):
    st.markdown(
        "Each point represents a team's average sequence profile across the 2015/16 season. "
        "All teams across all five leagues are shown. "
        "The **x-axis** measures the average number of passes per possession sequence — higher values "
        "suggest more patient, combinative build-up play. "
        "The **y-axis** measures the average speed at which teams advance the ball upfield. "
        "The **dashed lines** mark the overall mean for each axis, dividing the chart into four quadrants. "
        "Quadrant background colours follow a bivariate scheme to make the positioning more intuitive. "
        "The table shows the percentage of teams per league falling in each quadrant (unaffected by filter). "
        "Use the **league filter** to restrict the scatter to a single league."
    )

@st.cache_data
def load_data(db_path: str) -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM sequences_teams").df()
    conn.close()
    return df

sequences_teams = load_data(str(DUCKDB_PATH))

sequences_teams["league_name"] = pd.Categorical(
    sequences_teams["league_name"], categories=LEAGUE_ORDER, ordered=True
)

INF = 9999

mean_passes = sequences_teams["num_passes"].mean()
mean_speed  = sequences_teams["average_speed"].mean()

x_min = sequences_teams["num_passes"].min() - 0.1
x_max = sequences_teams["num_passes"].max() + 0.1
y_min = sequences_teams["average_speed"].min() - 0.05
y_max = sequences_teams["average_speed"].max() + 0.05

# ── Quadrant assignment (always on full data) ──────────────────────────────
def assign_quadrant(row):
    h_pass  = row["num_passes"]    >= mean_passes
    h_speed = row["average_speed"] >= mean_speed
    if   h_pass and h_speed:     return "HH"
    elif h_pass and not h_speed: return "HL"
    elif not h_pass and h_speed: return "LH"
    else:                        return "LL"

sequences_teams["quadrant"] = sequences_teams.apply(assign_quadrant, axis=1)

# ── Table data: rows=leagues, cols=quadrants ──────────────────────────────
@st.cache_data
def build_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["league_name", "quadrant"])
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby("league_name").size().reset_index(name="total")
    counts = counts.merge(totals, on="league_name")
    counts["pct"] = (counts["count"] / counts["total"] * 100).round(1)
    pivot = counts.pivot(index="league_name", columns="quadrant", values="pct").fillna(0.0)
    pivot = pivot.reindex(index=LEAGUE_ORDER, columns=QUADRANT_KEYS, fill_value=0.0)
    return pivot

table_df = build_table(sequences_teams)

# ── Matplotlib table — same style as shots_league.py ─────────────────────
@st.cache_data
def render_table(df: pd.DataFrame) -> bytes:
    n_rows = len(df)          # 5 leagues
    n_cols = len(QUADRANT_KEYS)  # 4 quadrants

    col_labels = QUADRANT_LABELS          # quadrant names as columns
    col_widths = [0.22] * n_cols

    cell_data   = []
    cell_colors = []

    for league in LEAGUE_ORDER:
        row = df.loc[league]
        c   = LEAGUE_COLORS[league]
        cell_data.append([f"{row[q]:.1f}%" for q in QUADRANT_KEYS])
        cell_colors.append([c + "22"] * n_cols)   # same tinted row as shots table

    # figure sized to match the shots table proportions
    fig = plt.figure(figsize=(4.5, 3.0), facecolor="white")
    ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.set_title("% of teams per quadrant", fontsize=8, fontweight="bold",
                 pad=3, loc="center", color="#333333")

    tbl = ax.table(
        cellText=cell_data,
        rowLabels=LEAGUE_ORDER,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        cellColours=cell_colors,
        colWidths=col_widths,
        bbox=[0.18, 0.0, 0.82, 0.92],   # leave room for row labels on the left
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row_idx == 0:
            # header row — grey background, bold
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold", fontsize=7.5)
        elif col_idx == -1:
            # row labels — coloured bold league name
            league_name = LEAGUE_ORDER[row_idx - 1]
            c           = LEAGUE_COLORS.get(league_name, "#333333")
            cell.set_text_props(color=c, fontweight="bold", fontsize=7)
            cell.set_edgecolor("none")

    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

table_img = render_table(table_df)

# ── Session state for filter ───────────────────────────────────────────────
if "scatter_league" not in st.session_state:
    st.session_state["scatter_league"] = "All"

# ── Layout ────────────────────────────────────────────────────────────────
col_chart, col_table, col_filter = st.columns([5, 4, 1.2], gap="medium")

with col_filter:
    st.markdown("<div style='padding-top:0.15cm'></div>", unsafe_allow_html=True)
    selected_league = st.selectbox(
        "League",
        options=["All"] + LEAGUE_ORDER,
        index=(["All"] + LEAGUE_ORDER).index(st.session_state["scatter_league"]),
        key="scatter_league",
    )

# ── Filter applied only to scatter ───────────────────────────────────────
plot_df = (
    sequences_teams.copy() if selected_league == "All"
    else sequences_teams[sequences_teams["league_name"] == selected_league].copy()
)

# ── Altair scatter ────────────────────────────────────────────────────────
def make_quad(x1, x2, y1, y2, hex_color):
    return (
        alt.Chart(pd.DataFrame({"x1": [x1], "x2": [x2], "y1": [y1], "y2": [y2]}))
        .mark_rect(color=hex_color, opacity=0.22)
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=[x_min, x_max])),
            x2=alt.X2("x2:Q"),
            y=alt.Y("y1:Q", scale=alt.Scale(domain=[y_min, y_max])),
            y2=alt.Y2("y2:Q"),
        )
    )

q_ll = make_quad(-INF,        mean_passes, -INF,        mean_speed, "#e8e8e8")
q_hl = make_quad(mean_passes,  INF,        -INF,        mean_speed, "#8fb3d0")
q_lh = make_quad(-INF,        mean_passes,  mean_speed,  INF,       "#d0a06e")
q_hh = make_quad(mean_passes,  INF,         mean_speed,  INF,       "#7a6b5a")

base = alt.Chart(plot_df)

points = base.mark_point(size=80, filled=True).encode(
    x=alt.X("num_passes:Q",
            scale=alt.Scale(domain=[x_min, x_max]),
            title="Avg Passes per Sequence"),
    y=alt.Y("average_speed:Q",
            scale=alt.Scale(domain=[y_min, y_max]),
            title="Avg Upfield Speed On-Ball"),
    color=alt.Color(
        "league_name:N",
        scale=alt.Scale(
            domain=LEAGUE_ORDER,
            range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER]
        ),
        legend=alt.Legend(title="League")
    ),
    tooltip=[
        alt.Tooltip("team_name:N",     title="Team"),
        alt.Tooltip("league_name:N",   title="League"),
        alt.Tooltip("num_passes:Q",    title="Avg Passes",  format=".2f"),
        alt.Tooltip("average_speed:Q", title="Avg Speed",   format=".2f"),
    ]
)

vline = (
    alt.Chart(pd.DataFrame({"v": [mean_passes]}))
    .mark_rule(strokeDash=[5, 5], color="#888888")
    .encode(x=alt.X("v:Q", scale=alt.Scale(domain=[x_min, x_max])))
)
hline = (
    alt.Chart(pd.DataFrame({"h": [mean_speed]}))
    .mark_rule(strokeDash=[5, 5], color="#888888")
    .encode(y=alt.Y("h:Q", scale=alt.Scale(domain=[y_min, y_max])))
)

chart = (
    alt.layer(q_ll, q_hl, q_lh, q_hh, points, vline, hline)
    .properties(width=600, height=400)
    .configure_view(strokeWidth=0)
    .configure_axis(grid=False)
    .interactive()
)

with col_chart:
    st.altair_chart(chart, use_container_width=False)

with col_table:
    st.image(table_img, use_container_width=True)