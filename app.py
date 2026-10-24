import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CineMatch | Discover Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
<style>
.stApp {
    background: #0b1020;
    color: #f4f6ff;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #f4f6ff;
}
.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(120deg, #202b50, #41265a, #17213b);
    margin-bottom: 25px;
}
.hero p {
    color: #d1d7ef;
    font-size: 1.05rem;
}
.movie-title {
    font-size: 0.95rem;
    font-weight: 650;
    min-height: 2.6em;
    margin-top: 8px;
}
.movie-meta {
    font-size: 0.82rem;
    color: #b2bdd8;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #414d70;
    background: #202b49;
    color: white;
}
div.stButton > button:hover {
    border-color: #a78bfa;
    color: #ddd6fe;
}
section[data-testid="stSidebar"] {
    background: #11182b;
}
</style>
""",
    unsafe_allow_html=True,
)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

PLACEHOLDER_POSTER = (
    "https://placehold.co/500x750/20202b/ffffff" "?text=Poster+Unavailable"
)


# -------------------------
# API helpers
# -------------------------
@st.cache_data(ttl=60)
def api_get(path, params=None):
    response = requests.get(
        f"{API_BASE}{path}",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600)
def poster_url_works(url):
    try:
        response = requests.get(url, timeout=5, stream=True)
        valid = response.status_code == 200 and (
            response.headers.get("content-type", "").startswith("image/")
        )
        response.close()
        return valid
    except requests.RequestException:
        return False


def get_poster(movie):
    poster_path = movie.get("poster_path")

    if not isinstance(poster_path, str) or not poster_path.startswith("/"):
        return PLACEHOLDER_POSTER

    url = f"{TMDB_IMAGE_BASE}{poster_path}"

    if poster_url_works(url):
        return url

    return PLACEHOLDER_POSTER


def get_year(movie):
    release_date = movie.get("release_date")
    if release_date and len(str(release_date)) >= 4:
        return str(release_date)[:4]
    return "Year unknown"


def show_movie_grid(movies_list, section_key, columns=5):
    if not movies_list:
        st.info("No movies found. Try another search.")
        return

    cols = st.columns(columns)

    for index, movie in enumerate(movies_list):
        with cols[index % columns]:
            poster = get_poster(movie)

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(
                    "<div style='height:250px;display:flex;"
                    "align-items:center;justify-content:center;"
                    "background:#202b40;border-radius:12px;'>"
                    "🎬 No poster</div>",
                    unsafe_allow_html=True,
                )

            title = movie.get("title") or "Untitled"
            st.markdown(
                f"<div class='movie-title'>{title}</div>",
                unsafe_allow_html=True,
            )

            rating = movie.get("vote_average")
            rating_text = (
                f"⭐ {rating:.1f}"
                if isinstance(rating, (int, float))
                else "⭐ Not rated"
            )
            st.markdown(
                f"<div class='movie-meta'>{get_year(movie)} · " f"{rating_text}</div>",
                unsafe_allow_html=True,
            )

            if st.button("View details", key=f"{section_key}_{index}"):
                st.session_state.selected_movie_title = title
                st.session_state.page = "details"
                st.rerun()


# -------------------------
# Session state
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_movie_title" not in st.session_state:
    st.session_state.selected_movie_title = ""

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Your personal movie discovery space")
    st.divider()

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("### Explore")
    columns_count = st.select_slider(
        "Posters per row",
        options=[3, 4, 5, 6],
        value=5,
    )

    st.divider()
    st.caption("Recommendations powered by your saved TF-IDF model.")

# -------------------------
# Home page
# -------------------------
if st.session_state.page == "home":
    st.markdown(
        """
    <div class="hero">
        <h1>Find your next favourite movie.</h1>
        <p>Search thousands of movies and discover stories
        that match your taste.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Search movies",
        placeholder="Try Batman, Toy Story, Titanic...",
    )

    if query.strip():
        try:
            result = api_get(
                "/search",
                {"query": query.strip(), "limit": 30},
            )
            st.markdown(f"## Search results for “{query.strip()}”")
            show_movie_grid(
                result.get("results", []),
                "search",
                columns_count,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

    else:
        st.markdown("## Start exploring")
        st.write(
            "Search for a movie above, or choose one of these popular starting points."
        )

        starter_titles = [
            "Toy Story",
            "The Dark Knight",
            "Titanic",
            "The Matrix",
            "Finding Nemo",
        ]

        for title in starter_titles:
            if st.button(f"🎬 {title}", key=f"starter_{title}"):
                st.session_state.selected_movie_title = title
                st.session_state.page = "details"
                st.rerun()

# -------------------------
# Movie details page
# -------------------------
elif st.session_state.page == "details":
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    title = st.session_state.selected_movie_title

    try:
        movie = api_get("/movie", {"title": title})

        left, right = st.columns([1, 2], gap="large")

        with left:
            poster = get_poster(movie)
            st.image(poster, use_container_width=True)

        with right:
            st.markdown(f"# {movie.get('title', title)}")
            st.markdown(f"**Release date:** {movie.get('release_date') or 'Unknown'}")

            rating = movie.get("vote_average")
            if isinstance(rating, (int, float)):
                st.markdown(f"**Rating:** ⭐ {rating:.1f}/10")

            genres = movie.get("genres_clean")
            if genres:
                st.markdown(f"**Genres:** {genres}")

            st.markdown("### Overview")
            st.write(movie.get("overview") or "No overview available.")

        st.divider()
        st.markdown("## Movies you may also like")

        recommendations = api_get(
            "/recommend",
            {"title": title, "limit": 10},
        )
        show_movie_grid(
            recommendations.get("recommendations", []),
            "recommendations",
            columns_count,
        )

    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            st.error(
                "This exact title was not found in the local dataset. "
                "Return home and try another movie."
            )
        else:
            st.error(f"API error: {exc}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
