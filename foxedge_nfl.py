import streamlit as st
from datetime import datetime, UTC
from importlib import import_module, util as importlib_util
from pathlib import Path
import sys

# ---- App config & minimal theming ----
st.set_page_config(
    page_title="FoxEdge NFL Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .fe-topbar {
      position: sticky; top: 0; z-index: 10;
      padding: 10px 14px; border-radius: 10px; margin-bottom: 10px;
      background: linear-gradient(90deg, rgba(0,255,170,0.12), rgba(255,0,170,0.10));
      border: 1px solid rgba(255,255,255,0.08);
      font-weight: 600; letter-spacing: .2px;
    }
    section[data-testid="stSidebar"] {width: 320px;}
    section[data-testid="stSidebar"] > div {padding-top: 1rem;}
    div[role="radiogroup"] label {
      padding: 10px 12px; border-radius: 8px; margin-bottom: 6px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
    }
    div[role="radiogroup"] label:hover {background: rgba(255,255,255,0.06);}
    div[role="radiogroup"] input:checked + div p {font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Dashboard",
    "Market Scanner",
    "Model Output",
    "Bet Tracker",
]

def _init_global_state():
    if "_page_states" not in st.session_state:
        st.session_state._page_states = {}
    if "active_page" not in st.session_state:
        st.session_state.active_page = PAGES[0]
    if "app_boot_ts" not in st.session_state:
        st.session_state.app_boot_ts = datetime.now(UTC).isoformat()


def page_state(page_name: str) -> dict:
    """Return a dict-like state bucket for a given page that persists across navigation."""
    _init_global_state()
    bucket = st.session_state._page_states.setdefault(page_name, {})
    return bucket


# ---------- Page implementations ----------

def page_dashboard():
    state = page_state("Dashboard")
    st.title("FoxEdge NFL – Dashboard")
    st.caption("Session started: " + st.session_state.app_boot_ts)

    # Example: sticky controls that keep values even after leaving the page
    state.setdefault("week", 1)
    state.setdefault("show_adv", False)

    state["week"] = st.slider("Week", 1, 23, value=state["week"], key="dash_week")
    state["show_adv"] = st.checkbox("Show advanced widgets", value=state["show_adv"], key="dash_adv")

    st.write({k: state[k] for k in state})
    if state["show_adv"]:
        st.info("Advanced widgets would render here. They will remember their values.")


def page_market_scanner():
    state = page_state("Market Scanner")
    st.title("Market Scanner")

    state.setdefault("book", "DK")
    state.setdefault("min_edge", 3.0)
    state.setdefault("totals_only", False)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        state["book"] = st.selectbox("Sportsbook", ["DK", "FD", "MG", "CZ"], index=["DK","FD","MG","CZ"].index(state["book"]), key="scan_book")
    with col2:
        state["min_edge"] = st.number_input("Min Edge %", value=state["min_edge"], step=0.5, key="scan_edge")
    with col3:
        state["totals_only"] = st.checkbox("Totals only", value=state["totals_only"], key="scan_totals")

    st.success("These selections persist when you switch pages.")


def page_model_output():
    state = page_state("Model Output")
    st.title("Model Output")

    state.setdefault("model_name", "foxedge_nfl_v1")
    state.setdefault("conf_cut", 0.70)

    state["model_name"] = st.text_input("Model", value=state["model_name"], key="model_name")
    state["conf_cut"] = st.slider("Confidence cutoff", 0.5, 0.99, value=float(state["conf_cut"]), key="conf_cut")

    st.write("Persisted state:", state)


def page_bet_tracker():
    state = page_state("Bet Tracker")
    st.title("Bet Tracker")

    state.setdefault("bankroll", 1000.0)
    state.setdefault("last_wager", 0.0)

    state["bankroll"] = st.number_input("Bankroll", min_value=0.0, value=float(state["bankroll"]), step=10.0, key="trk_roll")
    state["last_wager"] = st.number_input("Last Wager (u)", min_value=0.0, value=float(state["last_wager"]), step=0.1, key="trk_wager")

    st.write("Persisted state:", state)


# Map names to callables
PAGE_IMPL = {
    "Dashboard": page_dashboard,
    "Market Scanner": page_market_scanner,
    "Model Output": page_model_output,
    "Bet Tracker": page_bet_tracker,
}

# Baseline labels (optional pages will extend this)
PAGE_LABELS = {
    "Dashboard": "🏠 Dashboard",
    "Market Scanner": "📡 Market Scanner",
    "Model Output": "🧠 Model Output",
    "Bet Tracker": "📒 Bet Tracker",
}

# ---- Optional Page: DK Parlays (dynamic import so router doesn't crash) ----

def _load_page_module(preferred_names, file_stems=None):
    """Try to import a page module by dotted names; if that fails, try local files
    using provided stems (or derive stem from the first preferred name).
    """
    # Try normal imports first
    for name in preferred_names:
        try:
            return import_module(name)
        except Exception:
            continue
    # Try file-based discovery near this router for each candidate stem
    here = Path(__file__).resolve().parent
    stems = list(file_stems) if file_stems else [preferred_names[0].split(".")[-1]]
    for stem in stems:
        candidates = [
            here / f"{stem}.py",
            here.parent / f"{stem}.py",
            here / "pages" / f"{stem}_page.py",
        ]
        for p in candidates:
            if p.exists():
                spec = importlib_util.spec_from_file_location(p.stem, p)
                mod = importlib_util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                return mod
    return None

_mod_dk = _load_page_module((
    "dk_parlays",
    "pages.dk_parlays_page",
    "NFL.dk_parlays",
    "apps.dk_parlays",
), file_stems=("dk_parlays",))
if _mod_dk and hasattr(_mod_dk, "render"):
    PAGES.append("DK Parlays")
    PAGE_LABELS["DK Parlays"] = "🎰 DK Parlays"

    def page_dk_parlays():
        state = page_state("DK Parlays")
        _mod_dk.render(state)

    PAGE_IMPL["DK Parlays"] = page_dk_parlays
else:
    # Non-fatal hint in sidebar; app keeps running without this page
    try:
        st.sidebar.info("DK Parlays page not found. Place dk_parlays.py or pages/dk_parlays_page.py alongside this app to enable.")
    except Exception:
        pass

# ---- Optional Page: NFL Edge (dynamic import) ----
_mod_edge = _load_page_module((
    "nfl_edge_app",
    "pages.nfl_edge_app_page",
    "NFL.nfl_edge_app",
    "apps.nfl_edge_app",
), file_stems=("nfl_edge_app",))
if _mod_edge and hasattr(_mod_edge, "render"):
    PAGES.append("NFL Edge")
    PAGE_LABELS["NFL Edge"] = "🏈 NFL Edge"
    def page_nfl_edge():
        state = page_state("NFL Edge")
        _mod_edge.render(state)
    PAGE_IMPL["NFL Edge"] = page_nfl_edge
else:
    try:
        st.sidebar.info("NFL Edge page not found. Place nfl_edge_app.py or pages/nfl_edge_app_page.py alongside this app to enable.")
    except Exception:
        pass

# ---- Auto-discover wrapper pages under ./pages ----

def _auto_register_pages_from_pages_dir():
    here = Path(__file__).resolve().parent
    pages_dir = here / "pages"
    if not pages_dir.exists():
        return
    for p in sorted(pages_dir.glob("*_page.py")):
        base = p.stem.replace("_page", "")
        page_key = base.replace("_", " ").replace("-", " ").title()
        if page_key in PAGE_IMPL:
            continue
        try:
            spec = importlib_util.spec_from_file_location(p.stem, p)
            mod = importlib_util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "render"):
                PAGES.append(page_key)
                PAGE_LABELS[page_key] = f"🧩 {page_key}"
                def _make_page(mod=mod, page_key=page_key):
                    def _page():
                        state = page_state(page_key)
                        mod.render(state)
                    return _page
                PAGE_IMPL[page_key] = _make_page()
        except Exception:
            # Skip any broken module; keep the app running
            continue

# Run auto-discovery after explicit optional pages have been registered
_auto_register_pages_from_pages_dir()

# ---------- Sidebar Router ----------
LABEL_TO_KEY = {v: k for k, v in PAGE_LABELS.items()}

_init_global_state()

# Single, clean sidebar menu (label hidden for compactness)
selection_label = st.sidebar.radio(
    "Navigation",
    list(PAGE_LABELS.values()),
    index=list(PAGE_LABELS).index(st.session_state.active_page),
    label_visibility="collapsed",
    key="nav_radio",
)
selection = LABEL_TO_KEY[selection_label]

# Compact sticky top bar in main area
st.markdown(f"<div class='fe-topbar'>FoxEdge NFL · {PAGE_LABELS[selection]}</div>", unsafe_allow_html=True)

# Update active page
st.session_state.active_page = selection

# Render page
PAGE_IMPL[selection]()

st.markdown("---")
st.write("Powered by FoxEdge | Built with Streamlit")