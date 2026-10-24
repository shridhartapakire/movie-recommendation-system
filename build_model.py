
import ast
import pickle
from pathlib import Path

import pandas as pd

from recommender import build_recommendation_model


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "movies_metadata.csv"
MODEL_DIR = BASE_DIR / "model"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


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


print("Loading movie dataset...")
movies = pd.read_csv(DATA_PATH, low_memory=False)

movies = movies.drop_duplicates().copy()
movies["title"] = movies["title"].fillna("").astype(str).str.strip()
movies = movies[movies["title"] != ""].copy()
movies["overview"] = movies["overview"].fillna("").astype(str)
movies["genres_clean"] = movies["genres"].apply(extract_genres)
movies["release_year"] = pd.to_datetime(
    movies["release_date"], errors="coerce"
).dt.year
movies = movies.reset_index(drop=True)

print(f"Preparing {len(movies)} movies...")
final_movies, tfidf, tfidf_matrix = build_recommendation_model(movies)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

artifacts = {
    "movies.pkl": final_movies,
    "tfidf.pkl": tfidf,
    "tfidf_matrix.pkl": tfidf_matrix,
}

for filename, artifact in artifacts.items():
    path = MODEL_DIR / filename
    with path.open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {path}")

print("Model artifacts created successfully.")
print("Matrix shape:", tfidf_matrix.shape)
