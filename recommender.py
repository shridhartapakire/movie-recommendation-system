
"""Reusable movie recommendation module."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prepare_movie_text(movies):
    """Create text with extra weight given to movie genres."""
    genres = movies["genres_clean"].fillna("").astype(str)
    overview = movies["overview"].fillna("").astype(str)

    return (
        genres + " " + genres + " " + overview
    ).str.lower().str.strip()



def build_recommendation_model(movies):
    """Build and return the TF-IDF model and feature matrix."""
    final_movies = movies.copy()

    final_movies["weighted_tags"] = prepare_movie_text(final_movies)

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(final_movies["weighted_tags"])

    return final_movies, tfidf, tfidf_matrix



def recommend_movies(final_movies, tfidf_matrix, movie_title, year=None, top_n=5):
    """Return movie recommendations based on content similarity."""
    title = movie_title.strip().casefold()

    matches = final_movies.index[
        final_movies["title"].astype(str).str.casefold() == title
    ].tolist()

    if not matches:
        return {"message": f"Movie '{movie_title}' was not found.", "recommendations": []}

    if year is not None:
        matches = [
            idx for idx in matches
            if pd.notna(final_movies.loc[idx, "release_year"])
            and int(final_movies.loc[idx, "release_year"]) == year
        ]

        if not matches:
            return {
                "message": f"No movie found with title '{movie_title}' from {year}.",
                "recommendations": []
            }

    if len(matches) > 1:
        versions = []

        for idx in matches:
            release_year = final_movies.loc[idx, "release_year"]
            year_text = str(int(release_year)) if pd.notna(release_year) else "Unknown year"
            versions.append({
                "title": final_movies.loc[idx, "title"],
                "year": year_text
            })

        return {
            "message": f"Multiple versions of '{movie_title}' found. Please specify a year.",
            "versions": versions,
            "recommendations": []
        }

    movie_index = matches[0]

    scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).ravel()

    scores[movie_index] = -1
    ranked_indices = np.argsort(scores)[::-1][:max(0, top_n)]

    recommendations = []

    for idx in ranked_indices:
        movie = final_movies.iloc[idx]
        release_year = movie["release_year"]
        year_text = str(int(release_year)) if pd.notna(release_year) else "Unknown year"

        recommendations.append({
            "title": movie["title"],
            "year": year_text,
            "genres": movie["genres_clean"],
            "similarity": round(float(scores[idx]), 3)
        })

    return {
        "message": f"Recommendations for {final_movies.iloc[movie_index]['title']}.",
        "recommendations": recommendations
    }
