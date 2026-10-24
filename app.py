import os
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE = os.getenv("CINEMATCH_API", "http://127.0.0.1:8000")
TMDB_IMAGE_BASE = os.getenv("TMDB_IMAGE_BASE", "https://image.tmdb.org/t/p")

st.set_page_config(
    page_title="CineMatch | Discover Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# The poster wall, hero, grids and carousels are one small local component
# (plain HTML/CSS/JS, no build step). It sends clicked movie titles back to Python.
_wall = components.declare_component(
    "cinematch_wall",
    path=str(Path(__file__).parent / "components" / "cinematch_wall"),
)

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
<style>
header[data-testid="stHeader"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stToolbar"] { display: none !important; }

.stApp {
    background:
        radial-gradient(1200px 600px at 12% -10%, rgba(88, 60, 190, .28), transparent 60%),
        radial-gradient(900px 500px at 95% 0%, rgba(242, 102, 59, .14), transparent 60%),
        #070b16;
    color: #f4f6ff;
    font-family: Inter, "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
}
.block-container { max-width: 1480px; padding: 0 2rem 3rem; }
h1, h2, h3 { color: #f4f6ff; }

/* Navigation bar */
.st-key-topnav {
    position: sticky; top: 0; z-index: 50; padding: 14px 0 10px;
    background: linear-gradient(to bottom, rgba(7,11,22,.96) 70%, rgba(7,11,22,0));
    backdrop-filter: blur(8px);
}
.brand { font-size: 1.55rem; font-weight: 850; letter-spacing: -.5px; line-height: 1; white-space: nowrap; }
.brand span { background: linear-gradient(90deg, #8b5cf6, #f2663b); -webkit-background-clip: text; color: transparent; }
.st-key-topnav button, .st-key-searchbar button, .stButton > button {
    width: 100%; border-radius: 10px; border: 1px solid transparent; background: transparent;
    color: #c9d0ea; font-weight: 650; transition: all .2s ease;
}
.st-key-topnav button:hover, .stButton > button:hover {
    color: #fff; border-color: rgba(139,92,246,.7); background: rgba(139,92,246,.14);
}
.st-key-topnav [data-testid="stBaseButton-primary"], .st-key-searchbar [data-testid="stBaseButton-primaryFormSubmit"],
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #8b5cf6, #f2663b); color: #fff; border: 0;
}
.st-key-topnav [data-testid="stBaseButton-primary"] { background: rgba(139,92,246,.22); border: 1px solid rgba(139,92,246,.6); }

/* Search */
.st-key-searchbar [data-testid="stForm"] { border: 0; padding: 0; }
.st-key-searchbar [data-testid="stTextInputRootElement"], .st-key-searchbar [data-baseweb="input"],
.st-key-searchbar [data-baseweb="base-input"] {
    background: rgba(18,24,43,.92) !important; border-radius: 14px !important; min-height: 50px;
}
.st-key-searchbar [data-testid="stTextInputRootElement"], .st-key-searchbar [data-baseweb="input"] {
    border: 1px solid rgba(255,255,255,.14) !important;
}
.st-key-searchbar [data-testid="stTextInputRootElement"]:focus-within, .st-key-searchbar [data-baseweb="input"]:focus-within {
    border-color: #8b5cf6 !important; box-shadow: 0 0 0 3px rgba(139,92,246,.25);
}
.st-key-searchbar input { color: #fff !important; font-size: 1.02rem !important; padding: 0 18px !important; height: 50px; background: transparent !important; }
.st-key-searchbar input::placeholder { color: #8f9abd !important; opacity: 1; }
.st-key-searchbar button { height: 50px; border-radius: 14px; }

/* Keep nav and search on one row on small screens */
.st-key-topnav [data-testid="stHorizontalBlock"], .st-key-searchbar [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important; gap: .4rem;
}
.st-key-topnav [data-testid="stColumn"], .st-key-searchbar [data-testid="stColumn"] { min-width: 0 !important; }
@media (max-width: 700px) {
    .block-container { padding: 0 .8rem 2rem; }
    .brand { font-size: 1.1rem; }
    .st-key-topnav button { font-size: .72rem; padding: 0 !important; min-height: 2.2rem; }
    .st-key-topnav [data-testid="stColumn"]:last-child { display: none; }
    .st-key-searchbar [data-testid="stColumn"]:last-child { flex: 0 0 84px !important; }
}

/* Page headings and states */
.page-title { font-size: 2rem; font-weight: 800; margin: 18px 0 2px; letter-spacing: -.5px; }
.page-sub { color: #a9b2cc; margin-bottom: 14px; }
.state-box {
    padding: 44px 24px; margin: 18px 0; text-align: center; border-radius: 18px;
    border: 1px dashed rgba(255,255,255,.18); background: rgba(18,24,43,.6); color: #b9c3e6;
}
.state-box h3 { margin: 0 0 6px; }
.state-box code { color: #f0abfc; }
.skeleton { height: 420px; border-radius: 22px; background: linear-gradient(100deg,#121832 30%,#1b2447 50%,#121832 70%);
    background-size: 200% 100%; animation: shimmer 1.4s linear infinite; }
@keyframes shimmer { to { background-position: -200% 0; } }
div[data-testid="stRadio"] label, div[data-testid="stPills"] { color: #c9d0ea; }
iframe { border: 0; color-scheme: normal; }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------
# API helpers
# -------------------------
@st.cache_data(ttl=120, show_spinner=False)
def api_get(path, params=None):
    response = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def img_url(path, size):
    """Build a TMDB image URL. The browser loads it directly (no token needed)."""
    if isinstance(path, str) and path.startswith("/"):
        return f"{TMDB_IMAGE_BASE}/{size}{path}"
    return None


def short(text, limit):
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def get_year(movie):
    release_date = movie.get("release_date")
    if release_date and len(str(release_date)) >= 4 and str(release_date)[:4].isdigit():
        return int(str(release_date)[:4])
    return None


def to_card(movie, size="w342", extra=None):
    rating = movie.get("vote_average")
    return {
        "title": movie.get("title") or "Untitled",
        "year": get_year(movie),
        "rating": float(rating) if isinstance(rating, (int, float)) else None,
        "genres": movie.get("genres_list") or [],
        "overview": short(movie.get("overview"), 170),
        "poster": img_url(movie.get("poster_path"), size),
        "fallback": img_url(movie.get("poster_fallback"), size),
        "extra": extra,
    }


# -------------------------
# Session state and navigation
# -------------------------
PAGES = [("home", "Home"), ("discover", "Discover"), ("search", "Search"), ("trending", "Trending")]

defaults = {
    "page": "home",
    "selected_movie_title": "",
    "selected_movie_year": None,
    "watched": [],  # recently opened titles, used to personalise the home rows
    "search_query": "",
    "browse_limit": 40,
    "browse_sig": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def go(page):
    st.session_state.page = page
    st.rerun()


def open_movie(title, year=None):
    st.session_state.selected_movie_title = title
    st.session_state.selected_movie_year = year
    watched = [t for t in st.session_state.watched if t != title] + [title]
    st.session_state.watched = watched[-5:]
    st.session_state.page = "details"
    st.rerun()


def wall(kind, key, **args):
    """Render the custom component and open a movie when a card or button is clicked."""
    event = _wall(kind=kind, key=key, default=None, **args)
    seen_key = f"_seen_{key}"
    if isinstance(event, dict) and event.get("title") and event.get("id") != st.session_state.get(seen_key):
        st.session_state[seen_key] = event["id"]
        open_movie(event["title"], event.get("year"))


def api_error(exc):
    detail = "The recommendation API did not respond."
    if isinstance(exc, requests.ConnectionError):
        detail = f"Could not reach the API at <code>{API_BASE}</code>. Start it with <code>uvicorn main:app --reload</code>."
    elif isinstance(exc, requests.Timeout):
        detail = "The API took too long to respond. Please try again."
    st.markdown(
        f"<div class='state-box'><h3>Something went wrong</h3><div>{detail}</div></div>",
        unsafe_allow_html=True,
    )
    if st.button("↻ Try again"):
        st.cache_data.clear()
        st.rerun()


def empty_state(title, message):
    st.markdown(
        f"<div class='state-box'><h3>{title}</h3><div>{message}</div></div>",
        unsafe_allow_html=True,
    )


def nav_bar():
    with st.container(key="topnav"):
        cols = st.columns([2.4, 1, 1.1, 1, 1.1, 3], vertical_alignment="center")
        cols[0].markdown("<div class='brand'>🎬 Cine<span>Match</span></div>", unsafe_allow_html=True)
        active = "home" if st.session_state.page == "details" else st.session_state.page
        for col, (page, label) in zip(cols[1:5], PAGES):
            if col.button(label, key=f"nav_{page}", type="primary" if active == page else "secondary"):
                go(page)


def search_bar():
    with st.container(key="searchbar"):
        with st.form("search_form", clear_on_submit=False):
            left, right = st.columns([6, 1], vertical_alignment="center")
            text = left.text_input(
                "Search movies",
                value=st.session_state.search_query,
                placeholder="Search thousands of movies: Batman, Toy Story, Titanic…",
                label_visibility="collapsed",
            )
            submitted = right.form_submit_button("Search", type="primary")
        if submitted:
            st.session_state.search_query = text.strip()
            go("search")


# -------------------------
# Pages
# -------------------------
def page_home():
    search_bar()
    slot = st.empty()
    slot.markdown("<div class='skeleton'></div>", unsafe_allow_html=True)
    try:
        feed = api_get("/home", {"watched": st.session_state.watched, "per_row": 20})
    except requests.RequestException as exc:
        slot.empty()
        api_error(exc)
        return
    slot.empty()

    hero = feed["hero"]
    hero_card = to_card(hero, "w780")
    hero_card["overview"] = short(hero.get("overview"), 300)
    hero_card["backdrop"] = img_url(hero.get("backdrop_path"), "w1280")
    wall("hero", "home_hero", movie=hero_card)

    st.write("")
    rows = [
        {"title": r["title"], "subtitle": r.get("subtitle"), "items": [to_card(m) for m in r["movies"]]}
        for r in feed["rows"]
    ]
    wall("rows", "home_rows", rows=rows)


def page_search():
    search_bar()
    query = st.session_state.search_query.strip()
    if not query:
        empty_state("Search the catalogue", "Type a movie title above to see matches.")
        return

    try:
        with st.spinner("Searching…"):
            result = api_get("/search", {"query": query, "limit": 40})
    except requests.RequestException as exc:
        api_error(exc)
        return

    movies = result.get("results", [])
    st.markdown(f"<div class='page-title'>Results for “{query}”</div>", unsafe_allow_html=True)
    if not movies:
        empty_state("No movies found", "Check the spelling or try a shorter title.")
        return
    st.markdown(f"<div class='page-sub'>{len(movies)} matching titles</div>", unsafe_allow_html=True)
    wall("grid", "search_grid", items=[to_card(m, "w500") for m in movies])


def browse_page(title, subtitle, genre, sort, key):
    signature = (key, genre, sort)
    if st.session_state.browse_sig != signature:
        st.session_state.browse_sig = signature
        st.session_state.browse_limit = 40

    try:
        with st.spinner("Loading movies…"):
            data = api_get("/browse", {"genre": genre, "sort": sort, "limit": st.session_state.browse_limit})
    except requests.RequestException as exc:
        api_error(exc)
        return

    movies = data.get("results", [])
    if not movies:
        empty_state("Nothing here yet", "No movies match this selection.")
        return
    wall("grid", f"{key}_grid", items=[to_card(m, "w500") for m in movies])

    if len(movies) < data.get("total", 0) and st.session_state.browse_limit < 100:
        _, middle, _ = st.columns([2, 1, 2])
        if middle.button("Show more", key=f"{key}_more"):
            st.session_state.browse_limit = min(100, st.session_state.browse_limit + 20)
            st.rerun()


def page_discover():
    st.markdown("<div class='page-title'>Discover</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Browse the catalogue by genre.</div>", unsafe_allow_html=True)

    try:
        genres = [g["name"] for g in api_get("/genres")["genres"]]
    except requests.RequestException as exc:
        api_error(exc)
        return

    left, right = st.columns([3, 2], vertical_alignment="bottom")
    genre = left.selectbox("Genre", ["All genres"] + genres)
    sort_labels = {"Popular": "popularity", "Top rated": "rating", "Most voted": "votes", "Newest": "newest"}
    sort_label = right.radio("Sort by", list(sort_labels), horizontal=True)

    browse_page(
        "Discover", "", None if genre == "All genres" else genre, sort_labels[sort_label], "discover"
    )


def page_trending():
    st.markdown("<div class='page-title'>Trending</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-sub'>The most popular movies in the dataset, ranked by its popularity score.</div>",
        unsafe_allow_html=True,
    )
    browse_page("Trending", "", None, "popularity", "trending")


def page_details():
    if st.button("← Back to Home", key="back_home"):
        go("home")

    title = st.session_state.selected_movie_title
    year = st.session_state.selected_movie_year

    try:
        with st.spinner("Loading movie…"):
            movie = api_get("/movie", {"title": title, "year": year})
            recs = api_get("/recommend", {"title": title, "year": year, "limit": 20})
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            empty_state("Movie not found", "This title is not in the local dataset. Return home and try another movie.")
        else:
            api_error(exc)
        return
    except requests.RequestException as exc:
        api_error(exc)
        return

    card = to_card(movie, "w780")
    card["overview"] = movie.get("overview") or ""
    card["backdrop"] = img_url(movie.get("backdrop_path"), "w1280")
    card["tagline"] = movie.get("tagline") or ""
    card["runtime"] = int(movie["runtime"]) if isinstance(movie.get("runtime"), (int, float)) and movie["runtime"] > 0 else None
    votes = movie.get("vote_count")
    card["votes"] = int(votes) if isinstance(votes, (int, float)) and votes > 0 else None

    def money(value):
        try:
            return f"${int(float(value)):,}" if float(value) > 0 else None
        except (TypeError, ValueError):
            return None

    facts = [
        ("Release date", movie.get("release_date")),
        ("Status", movie.get("status")),
        ("Original language", (movie.get("original_language") or "").upper() or None),
        ("Budget", money(movie.get("budget"))),
        ("Revenue", money(movie.get("revenue"))),
    ]
    wall("detail", "details_hero", movie=card, facts=[[k, str(v)] for k, v in facts if v])

    st.write("")
    similar = [
        to_card(m, extra=f"Similarity {m['similarity_score']:.2f}" if m.get("similarity_score") is not None else None)
        for m in recs.get("recommendations", [])
    ]
    wall("carousel", "details_similar", title="Movies you may also like", items=similar)


# -------------------------
# Router
# -------------------------
nav_bar()

page = st.session_state.page
if page == "details":
    page_details()
elif page == "discover":
    page_discover()
elif page == "search":
    page_search()
elif page == "trending":
    page_trending()
else:
    page_home()