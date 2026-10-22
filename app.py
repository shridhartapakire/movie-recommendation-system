
import ast
import pandas as pd
import streamlit as st

from recommender import build_recommendation_model, recommend_movies


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favorites.")

@st.cache_data
def load_movies():
    def extract_genres(value):
        try:
            items = ast.literal_eval(value) if isinstance(value, str) else []
            return " ".join(
                item.get("name", "")
                for item in items
                if isinstance(item, dict)
            )
        except (ValueError, SyntaxError, TypeError):
            return ""

    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    movies = movies.drop_duplicates().copy()
    movies["title"] = movies["title"].fillna("").astype(str).str.strip()
    movies = movies[movies["title"] != ""].copy()
    movies["overview"] = movies["overview"].fillna("").astype(str)
    movies["genres_clean"] = movies["genres"].apply(extract_genres)
    movies["release_year"] = pd.to_datetime(
        movies["release_date"], errors="coerce"
    ).dt.year

    return movies.reset_index(drop=True)


@st.cache_resource
def load_recommendation_model():
    movies = load_movies()
    return build_recommendation_model(movies)


try:
    with st.spinner("Loading the movie database and recommendation model..."):
        final_movies, tfidf, tfidf_matrix = load_recommendation_model()
except FileNotFoundError:
    st.error(
        "The file movies_metadata.csv was not found. "
        "Place it in the project folder beside app.py."
    )
    st.stop()

st.success(f"Loaded {len(final_movies):,} movies.")

movie_titles = sorted(final_movies["title"].unique().tolist())
selected_title = st.selectbox(
    "Choose a movie",
    movie_titles,
    index=None,
    placeholder="Type a movie title to search..."
)

if selected_title is None:
    st.info("Search for and select a movie to get recommendations.")
    st.stop()

matching_movies = final_movies[
    final_movies["title"].str.casefold() == selected_title.casefold()
]

years = sorted(
    matching_movies["release_year"].dropna().astype(int).unique().tolist()
)

selected_year = None

if len(matching_movies) > 1:
    year_options = ["Choose a year"] + [str(year) for year in years]
    selected_year_option = st.selectbox("Select release year", year_options)

    if selected_year_option != "Choose a year":
        selected_year = int(selected_year_option)
elif len(years) == 1:
    selected_year = years[0]

top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("Recommend Movies", type="primary"):
    if len(matching_movies) > 1 and selected_year is None:
        st.warning("Please select the release year for this movie.")
    else:
        result = recommend_movies(
            final_movies,
            tfidf_matrix,
            selected_title,
            year=selected_year,
            top_n=top_n
        )

        st.subheader(result["message"])

        if result.get("versions"):
            st.write("Available versions:")
            st.write(result["versions"])

        for rank, movie in enumerate(result["recommendations"], start=1):
            st.markdown(f"### {rank}. {movie['title']} ({movie['year']})")
            st.write(f"Genres: {movie['genres'] or 'Unknown'}")
            st.write(f"Similarity score: {movie['similarity']:.3f}")
            st.divider()
