import os
import requests
from dotenv import load_dotenv
from functools import lru_cache

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

# -------------------------
# Configuration
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = FastAPI(
    title="Movie Recommendation API",
    description="Movie search and recommendations using the saved TF-IDF model.",
    version="1.0.0",
)

load_dotenv(BASE_DIR / ".env")

TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")

TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
    "accept": "application/json",
}

# -------------------------
# Load saved model files
# -------------------------
try:
    with open(MODEL_DIR / "movies.pkl", "rb") as f:
        movies = pickle.load(f)

    with open(MODEL_DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(MODEL_DIR / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

except Exception as exc:
    raise RuntimeError(
        f"Could not load saved model files from {MODEL_DIR}: {exc}"
    ) from exc

if not isinstance(movies, pd.DataFrame) or "title" not in movies.columns:
    raise RuntimeError("movies.pkl must contain a DataFrame with a title column.")

if tfidf_matrix.shape[0] != len(movies):
    raise RuntimeError(
        "Movie count does not match TF-IDF matrix rows. "
        "The saved files may not belong to the same model."
    )


# -------------------------
# Helpers
# -------------------------
def clean_value(value):
    """Convert pandas and NumPy values into JSON-safe values."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if pd.isna(value):
        return None

    return value


@lru_cache(maxsize=500)
def get_current_poster_path(tmdb_id):
    """Fetch and cache the current poster path from TMDB."""
    if not TMDB_BEARER_TOKEN or not tmdb_id:
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            headers=TMDB_HEADERS,
            timeout=10,
        )

        if response.status_code != 200:
            print("TMDB status:", response.status_code)
            print("TMDB response:", response.text[:300])
            return None

        return response.json().get("poster_path")

    except requests.RequestException as exc:
        print("TMDB connection error:", repr(exc))
        return None


def movie_to_dict(index):
    """Convert one movie row to a JSON-safe dictionary."""
    row = movies.iloc[index]
    result = {key: clean_value(value) for key, value in row.items()}

    tmdb_id = result.get("id")
    current_poster = get_current_poster_path(tmdb_id)

    if current_poster:
        result["poster_path"] = current_poster

    return result


def find_movie_index(title):
    """Find a movie by exact title, ignoring capitalization."""
    matches = movies.index[
        movies["title"].astype(str).str.strip().str.casefold()
        == title.strip().casefold()
    ]

    if len(matches) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Movie not found: {title}",
        )

    return int(matches[0])


# -------------------------
# API endpoints
# -------------------------
@app.get("/")
def home():
    return {
        "message": "Movie Recommendation API is running.",
        "movies_loaded": len(movies),
        "endpoints": ["/health", "/search", "/movie", "/recommend", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "movies_loaded": len(movies),
        "tfidf_rows": tfidf_matrix.shape[0],
    }


@app.get("/search")
def search_movies(
    query: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(20, ge=1, le=100),
):
    """Search local movie titles."""
    query = query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    titles = movies["title"].fillna("").astype(str)
    matches = movies[titles.str.contains(query, case=False, regex=False)].head(limit)

    indices = matches.index.tolist()

    return {
        "query": query,
        "count": len(indices),
        "results": [movie_to_dict(int(index)) for index in indices],
    }


@app.get("/movie")
def movie_details(
    title: str = Query(..., min_length=1, max_length=200),
):
    """Return details for an exact movie title."""
    index = find_movie_index(title)
    return movie_to_dict(index)


@app.get("/recommend")
def recommend_movies(
    title: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(10, ge=1, le=50),
):
    """Recommend similar movies using the saved TF-IDF matrix."""
    index = find_movie_index(title)

    query_vector = tfidf_matrix[index]
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()

    # Exclude the selected movie.
    scores[index] = -1

    # Select the highest-scoring movies.
    top_indices = np.argsort(-scores)[:limit]

    recommendations = []

    for movie_index in top_indices:
        if scores[movie_index] < 0:
            continue

        item = movie_to_dict(int(movie_index))
        item["similarity_score"] = round(float(scores[movie_index]), 4)
        recommendations.append(item)

    return {
        "selected_movie": str(movies.iloc[index]["title"]),
        "count": len(recommendations),
        "recommendations": recommendations,
    }
