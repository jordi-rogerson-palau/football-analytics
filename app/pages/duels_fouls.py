import os
import duckdb
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde

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

st.markdown("### Duels & Fouls Distribution per Game")

with st.expander("ℹ️ Chart information"):
    st.markdown(
        "Both charts use a **ridge plot** (stacked kernel density estimates) to compare "
        "the per-game distribution of physical events across all five leagues. "
        "Each coloured area shows how frequently a given count occurs across all games in that league — "
        "a wide, flat shape means high variability between games, while a tall narrow peak means "
        "most games cluster around a consistent value. "
        "The **black vertical line** inside each distribution marks the median for that league, "
        "making cross-league comparison easier without needing to judge the peak position visually. "
        "Both charts share the same density scale so the height of areas is directly comparable "
        "between the two plots. "
        "Note that **duels and fouls are separate events**: a duel is a physical contest for the ball "
        "that the referee lets play on, while a foul is called when the referee stops play — "
        "a high duel count combined with a low foul count therefore suggests both physical intensity "
        "and either lenient refereeing or high-quality tackling. "
        "Hovering over the fouls chart also reveals the **card breakdown** for each league: "
        "the percentage of fouls that resulted in no card, a yellow card, or a red card."
    )

@st.cache_data
def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    duels = con.execute("SELECT * FROM duels").df()
    teams = con.execute("SELECT team_id, league_name FROM teams").df()
    con.close()

    duels_enriched = duels.merge(teams[["team_id", "league_name"]], on="team_id", how="left")

    duels_per_game = (
        duels_enriched[duels_enriched["event_type"] == "Duel"]
        .groupby(["match_id", "league_name"]).size().reset_index(name="count")
    )

    fouls_only = duels_enriched[duels_enriched["event_type"] == "Foul Committed"]

    fouls_per_game = (
        fouls_only.groupby(["match_id", "league_name"]).size().reset_index(name="count")
    )

    card_stats = (
        fouls_only.groupby("league_name")
        .agg(total=("yellow_card", "count"), yellow=("yellow_card", "sum"), red=("red_card", "sum"))
        .reset_index()
    )
    card_stats["no_card_pct"] = ((card_stats["total"] - card_stats["yellow"] - card_stats["red"])
                                  / card_stats["total"] * 100).round(1)
    card_stats["yellow_pct"]  = (card_stats["yellow"] / card_stats["total"] * 100).round(1)
    card_stats["red_pct"]     = (card_stats["red"]    / card_stats["total"] * 100).round(1)

    fouls_per_game = fouls_per_game.merge(
        card_stats[["league_name", "no_card_pct", "yellow_pct", "red_pct"]],
        on="league_name", how="left"
    )
    return duels_per_game, fouls_per_game

duels_per_game, fouls_per_game = load_data()

def alpha_order(df):
    return sorted(df["league_name"].unique().tolist())

def compute_density_peak(df):
    peaks = []
    for league in df["league_name"].unique():
        values = df[df["league_name"] == league]["count"].values
        if len(values) > 1:
            kde = gaussian_kde(values)
            x   = np.linspace(values.min(), values.max(), 200)
            peaks.append(kde(x).max())
    return max(peaks)

shared_peak = max(compute_density_peak(duels_per_game), compute_density_peak(fouls_per_game))

# ── Spacing constants ──
# Total y range is fixed at 2.2, chart height fixed at 300px — nothing grows.
# Increasing offset_step from 0.45 → 0.48 widens the gap between distributions.
# max_density_height shrinks accordingly to keep the total range unchanged:
#   (n_leagues-1) * offset_step + max_density_height = 2.2
#   4 * 0.48 + 0.28 = 2.20  ✓
offset_step        = 0.48
max_density_height = 2.2 - 4 * offset_step   # 0.28
density_scale      = max_density_height / shared_peak

Y_DOMAIN     = [0, 2.2]   # unchanged
CHART_HEIGHT = 300         # unchanged

def add_offsets(df, order):
    offset = {league: i * offset_step for i, league in enumerate(reversed(order))}
    df     = df.copy()
    df["offset"] = df["league_name"].map(offset)
    return df

def add_stats(df):
    stats = (
        df.groupby("league_name")["count"]
        .agg(median="median", std="std").round(2).reset_index()
    )
    return df.merge(stats, on="league_name", how="left")

def build_median_lines(df, order):
    rows = []
    for league in order:
        lg_df  = df[df["league_name"] == league]
        values = lg_df["count"].values
        if len(values) < 2:
            continue
        median_val = float(np.median(values))
        offset_val = float(lg_df["offset"].iloc[0])

        kde               = gaussian_kde(values)
        density_at_median = float(kde(np.array([median_val]))[0])
        top_val           = density_at_median * density_scale + offset_val

        rows.append({
            "league_name": league,
            "median":      median_val,
            "offset":      offset_val,
            "top":         top_val,
        })
    return pd.DataFrame(rows)

def ridge_chart(df, title, show_legend=True, show_cards=False, x_max=None, x_label="Count per Game", x_axis_color=None):
    order = alpha_order(df)
    df    = add_offsets(df, order)
    df    = add_stats(df)

    x_min_val = float(df["count"].min() - 3)
    x_max_val = float(x_max) if x_max is not None else float(df["count"].max() + 3)

    groupby_cols = ["league_name", "offset", "median", "std"]
    if show_cards:
        groupby_cols += ["no_card_pct", "yellow_pct", "red_pct"]

    tooltip = [
        alt.Tooltip("league_name:N", title="League"),
        alt.Tooltip("median:Q",      title="Median",  format=".1f"),
        alt.Tooltip("std:Q",         title="Std Dev", format=".1f"),
    ]
    if show_cards:
        tooltip += [
            alt.Tooltip("no_card_pct:Q", title="No Card %",   format=".1f"),
            alt.Tooltip("yellow_pct:Q",  title="🟡 Yellow %", format=".1f"),
            alt.Tooltip("red_pct:Q",     title="🔴 Red %",    format=".1f"),
        ]

    areas = alt.Chart(df).transform_density(
        density="count",
        groupby=groupby_cols,
        as_=["count", "density"],
        extent=[x_min_val, x_max_val],
        steps=200
    ).transform_calculate(
        density_scaled=f"datum.density * {density_scale}",
        shifted="datum.density_scaled + datum.offset"
    ).mark_area(fillOpacity=0.55, strokeWidth=1.5).encode(
        x=alt.X("count:Q", title=x_label,
                axis=alt.Axis(grid=False, titleFontSize=16, titleFontWeight="bold",
                              titleColor=x_axis_color if x_axis_color else alt.Undefined,
                              labelColor=x_axis_color if x_axis_color else alt.Undefined),
                scale=alt.Scale(domain=[x_min_val, x_max_val])),
        y=alt.Y("shifted:Q", axis=None,
                scale=alt.Scale(domain=Y_DOMAIN)),
        y2=alt.Y2("offset:Q"),
        color=alt.Color(
            "league_name:N",
            scale=alt.Scale(domain=LEAGUE_ORDER, range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER]),
            legend=alt.Legend(title="League") if show_legend else None
        ),
        stroke=alt.Stroke(
            "league_name:N",
            scale=alt.Scale(domain=LEAGUE_ORDER, range=[LEAGUE_COLORS[l] for l in LEAGUE_ORDER]),
            legend=None
        ),
        order=alt.Order("offset:Q", sort="ascending"),
        tooltip=tooltip
    )

    median_df    = build_median_lines(df, order)
    median_lines = alt.Chart(median_df).mark_rule(
        color="black", strokeWidth=1.2, opacity=0.7
    ).encode(
        x=alt.X("median:Q", scale=alt.Scale(domain=[x_min_val, x_max_val])),
        y=alt.Y("offset:Q", scale=alt.Scale(domain=Y_DOMAIN)),
        y2=alt.Y2("top:Q"),
    )

    chart = (
        alt.layer(areas, median_lines)
        .properties(
            width=400,
            height=CHART_HEIGHT,
        )
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )

    return chart

col1, sep, col2 = st.columns([20, 1, 20])
with col1:
    st.altair_chart(ridge_chart(duels_per_game, "Duels per Game",
                                show_legend=False, show_cards=False, x_max=145,
                                x_label="Duels per Game"))
with sep:
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            height: {CHART_HEIGHT}px;
            padding-top: 0px;
        ">
            <div style="
                width: 1px;
                height: 100%;
                background-color: #cccccc;
            "></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.altair_chart(ridge_chart(fouls_per_game, "Fouls Committed per Game",
                                show_legend=True, show_cards=True,
                                x_label="Fouls per Game", x_axis_color="black"))