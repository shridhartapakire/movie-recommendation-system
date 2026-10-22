# 🎬 Movie Recommendation System

A content-based movie recommendation application built with Python, Pandas, Scikit-learn, and Streamlit.

The application recommends movies based on similarities between their genres and plot descriptions using TF-IDF vectorization and cosine similarity.

## Features

- Search and select movies from a dataset of over 45,000 movies.
- Get recommendations for similar movies.
- Select a release year when multiple movies share the same title.
- Choose how many recommendations to display.
- View movie genres and similarity scores.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit

## Project Structure

```text
movie-recommendation-system/
├── app.py
├── recommender.py
├── test_recommender.py
├── requirements.txt
├── README.md
└── movie_recommendation.ipynb
```

The movie dataset (`movies_metadata.csv`) is required to run the application but is not included in the repository.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/shridhartapakire/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add the dataset

Place `movies_metadata.csv` in the project root directory, alongside `app.py`.

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## How It Works

1. Movie metadata is cleaned and prepared.
2. Genre names are weighted by including them twice in the text representation.
3. TF-IDF converts movie descriptions and genres into numerical feature vectors.
4. Cosine similarity measures how similar movies are.
5. The application displays the highest-scoring recommendations.

## Future Improvements

- Add movie posters and richer movie details.
- Improve recommendation relevance.
- Deploy the application online.
- Add automated tests.

## Author

Shridhar Tapakire