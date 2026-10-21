
import ast
import pandas as pd
from recommender import build_recommendation_model, recommend_movies


def extract_genres(value):
    """Convert the dataset's genre data into readable genre names."""
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        return " ".join(
            item.get("name", "")
            for item in items
            if isinstance(item, dict)
        )
    except (ValueError, SyntaxError, TypeError):
        return ""


# Load the movie dataset
movies = pd.read_csv("movies_metadata.csv", low_memory=False)

# Prepare the data
movies = movies.drop_duplicates().copy()
movies["title"] = movies["title"].fillna("").astype(str).str.strip()
movies = movies[movies["title"] != ""].copy()
movies["overview"] = movies["overview"].fillna("").astype(str)
movies["genres_clean"] = movies["genres"].apply(extract_genres)
movies["release_year"] = pd.to_datetime(
    movies["release_date"], errors="coerce"
).dt.year
movies = movies.reset_index(drop=True)

print("Dataset loaded:", len(movies), "movies")

# Build the recommendation model
final_movies, tfidf, tfidf_matrix = build_recommendation_model(movies)

print("TF-IDF matrix:", tfidf_matrix.shape)

# Test recommendations
result = recommend_movies(
    final_movies, tfidf_matrix, "Toy Story", year=1995, top_n=5
)

print("\n", result["message"])

for rank, movie in enumerate(result["recommendations"], start=1):
    print(
        f"{rank}. {movie['title']} ({movie['year']})"
        f" | Genres: {movie['genres']}"
        f" | Similarity: {movie['similarity']}"
    )


print("\n--- Duplicate-title test ---")

result = recommend_movies(
    final_movies, tfidf_matrix, "12 Angry Men", top_n=5
)

print(result["message"])
print("Versions:", result.get("versions", []))

print("\n--- Specific-year test ---")

result = recommend_movies(
    final_movies, tfidf_matrix, "12 Angry Men", year=1997, top_n=5
)

print(result["message"])

for rank, movie in enumerate(result["recommendations"], start=1):
    print(f"{rank}. {movie['title']} ({movie['year']})")
