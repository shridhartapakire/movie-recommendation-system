
import pickle
from pathlib import Path

from recommender import recommend_movies

MODEL_DIR = Path(__file__).resolve().parent / "model"


def load_artifact(filename):
    with (MODEL_DIR / filename).open("rb") as file:
        return pickle.load(file)


print("Loading saved model artifacts...")

movies = load_artifact("movies.pkl")
tfidf = load_artifact("tfidf.pkl")
tfidf_matrix = load_artifact("tfidf_matrix.pkl")

print("Movies:", len(movies))
print("TF-IDF matrix:", tfidf_matrix.shape)

result = recommend_movies(
    movies,
    tfidf_matrix,
    "Toy Story",
    year=1995,
    top_n=5,
)

print("\n", result["message"])

for rank, movie in enumerate(result["recommendations"], start=1):
    print(
        f"{rank}. {movie['title']} ({movie['year']})"
        f" | Similarity: {movie['similarity']}"
    )

assert len(movies) == 45443
assert tfidf_matrix.shape == (45443, 10000)
assert len(result["recommendations"]) == 5

print("\nSaved model test passed!")
