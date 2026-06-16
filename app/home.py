import streamlit as st
import importlib.util
import sys
from pathlib import Path

st.set_page_config(
    page_title="Football Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CATEGORIES = {
    "⚡ Chance Creation": {
        "Up-field Speed"          : "pages/scatter.py",
        "Pass Length Distribution": "pages/barchart_pass.py",
        "Transition Frequency"    : "pages/spatial_passes.py",
        "Shots"                   : "pages/shots_league.py",
    },
    "🛡️ Defending": {
        "Duels vs Fouls": "pages/duels_fouls.py",
        "Recuperations" : "pages/recuperations.py",
        "Turnovers"     : "pages/turnovers.py",
    },
    "🏆 Competitiveness": {
        "Scores": "pages/scores.py",
        "xG"    : "pages/xg.py",
    },
    "🔄 Substitutions": {
        "Subs by Minute"  : "pages/subs_min.py",
        "Subs by Position": "pages/subs_pos.py",
        "Sub Impact"      : "pages/subs_impact.py",
    },
    "📊 Usage Rate": {
        "Team-level"  : "pages/usage_rate.py",
        "Player-level": "pages/usage_metrics.py",
    },
}

ALL_PAGES = {label: path for cat in CATEGORIES.values() for label, path in cat.items()}

first_cat  = list(CATEGORIES.keys())[0]
first_page = list(CATEGORIES[first_cat].keys())[0]

if "active_cat"  not in st.session_state:
    st.session_state.active_cat  = first_cat
if "active_page" not in st.session_state:
    st.session_state.active_page = first_page

active_cat  = st.session_state.active_cat
active_page = st.session_state.active_page

st.markdown("""
<style>
    /* ── Colour palette (UI chrome only — no overlap with league encodings) ────
       League hues already occupied:
         Bundesliga    #d62728  red      ~0°
         Premier League #ff7f0e  orange   ~30°
         Serie A       #2ca02c  green    ~120°
         La Liga       #1f77b4  blue     ~210°
         Ligue 1       #9467bd  purple   ~280°

       UI colours chosen in the open teal gap (~175°) and dark neutrals:
         Title          : #1a1a2e  dark navy-black  (authority, no hue clash)
         Cat btn inactive: #2e4057  steel blue-grey  (dark, desaturated, toolbar weight)
         Cat btn active  : #048a81  dark teal        (unambiguous selection signal)
         Sub btn inactive: #e8edf2  very light grey-blue (recedes, clear hierarchy)
         Sub btn active  : #048a81  same teal         (consistent "you are here" signal)
         Divider         : #2e4057  echoes cat buttons, frames the two rows as one unit
    ─────────────────────────────────────────────────────────────────────────── */

    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    .block-container {
        padding-top:  0rem  !important;
        margin-top:   0rem  !important;
        padding-left: 1rem  !important;
        padding-right: 2rem !important;
    }
    .block-container > div,
    .block-container > div > div {
        padding-top: 0 !important;
        margin-top:  0 !important;
    }

    /* ── Category button row layout ── */
    [data-testid="stHorizontalBlock"] {
        gap: 2px !important;
        margin-top:    0px !important;
        padding-top:   0px !important;
        margin-bottom: 0px !important;
        flex-wrap: nowrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        padding: 0 !important;
        min-width: 0 !important;
    }

    /* ── Category buttons — inactive: dark steel, white text ── */
    [data-testid="stHorizontalBlock"] button {
        width: 100% !important;
        border-radius: 6px !important;
        border: 1px solid #22303f !important;
        background: #2e4057 !important;
        color: #c8d6e5 !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
        padding: 5px 8px !important;
        line-height: 1.3 !important;
        margin: 0 !important;
        box-shadow: none !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        transition: background 0.15s, color 0.15s;
    }
    [data-testid="stHorizontalBlock"] button:hover {
        background: #3a5068 !important;
        color: #ffffff !important;
        border-color: #2e4057 !important;
        box-shadow: none !important;
    }
    /* Active category — teal, clearly selected ── */
    [data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: #048a81 !important;
        color: #ffffff !important;
        border-color: #036b63 !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px !important;
        box-shadow: none !important;
    }

    /* ── Divider between the two nav rows ── */
    .nav-divider {
        height: 1px;
        background: #2e4057;
        margin: 4px 0 4px 0;
    }

    /* ── Sub-page buttons: lighter, recessive — clear hierarchy below cat row ── */
    [data-testid="stHorizontalBlock"]:nth-child(2) button {
        background: #e8edf2 !important;
        color: #2e4057 !important;
        border: 1px solid #bbc8d6 !important;
        font-size: 11.5px !important;
        font-weight: 400 !important;
        letter-spacing: 0.2px !important;
        padding: 5px 14px !important;
        line-height: 1.3 !important;
        border-radius: 4px !important;
    }
    [data-testid="stHorizontalBlock"]:nth-child(2) button:hover {
        background: #d0dae6 !important;
        color: #1a2d40 !important;
        border-color: #9aafc2 !important;
        box-shadow: none !important;
    }
    /* Active sub-page — same teal as active category ── */
    [data-testid="stHorizontalBlock"]:nth-child(2) button[kind="primary"] {
        background: #048a81 !important;
        color: #ffffff !important;
        border-color: #036b63 !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    /* ── Reset chart toolbar buttons (expand/fullscreen) to neutral ────────────
       These sit inside stElementToolbar and must never inherit the nav palette. */
    [data-testid="stElementToolbar"] button,
    [data-testid="stElementToolbar"] button:hover,
    [data-testid="stElementToolbar"] button:focus {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #888888 !important;
    }

    /* ── Reset expander header to neutral — no nav colouring leaking in ── */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────
st.markdown(
    '<p style="font-size:32px; font-weight:700; color:#1a1a2e; '
    'text-align:center; margin:4px 0 8px 0; padding:0; line-height:1.2; '
    'letter-spacing:0.5px; font-family: sans-serif;">'
    'Visual Analytics of 15/16 Season Over the Top 5 Leagues</p>',
    unsafe_allow_html=True,
)

# ── Row 1: category buttons ────────────────────────────
cat_cols = st.columns(len(CATEGORIES))
for col, cat in zip(cat_cols, CATEGORIES.keys()):
    with col:
        if st.button(
            cat,
            key=f"cat_{cat}",
            use_container_width=True,
            type="primary" if cat == active_cat else "secondary",
        ):
            st.session_state.active_cat  = cat
            st.session_state.active_page = list(CATEGORIES[cat].keys())[0]
            st.rerun()

st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

# ── Row 2: sub-page buttons ────────────────────────────
cat_pages = list(CATEGORIES[active_cat].keys())
n         = len(cat_pages)

weights  = [2] * n + [max(1, 10 - 2 * n)]
all_cols = st.columns(weights)

for i, label in enumerate(cat_pages):
    with all_cols[i]:
        if st.button(
            label,
            key=f"page_{label}",
            use_container_width=True,
            type="primary" if label == active_page else "secondary",
        ):
            st.session_state.active_page = label
            st.rerun()
# remaining column(s) intentionally empty


# ── Page runner ────────────────────────────────────────
def run_page(script_path: str):
    path = Path(script_path)
    if not path.exists():
        st.error(f"Script not found: `{script_path}`")
        return

    module_name = f"_page_{path.stem}"

    if module_name in sys.modules:
        del sys.modules[module_name]
    if "_page_module" in sys.modules:
        del sys.modules["_page_module"]

    page_dir = str(path.parent.resolve())
    if page_dir not in sys.path:
        sys.path.insert(0, page_dir)

    spec   = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


run_page(ALL_PAGES[active_page])