# 🎬 CineMatch — Movie Recommendation System

CineMatch is a content-based movie discovery application built with Python, FastAPI, Streamlit, Pandas, and Scikit-learn. It helps users search a large movie dataset, explore movie details, and discover similar movies using a TF-IDF recommendation model.

## Features

- Search a dataset of more than 45,000 movies.
- View movie details, including genres, release date, ratings, and overview.
- Get similar movie recommendations using TF-IDF and cosine similarity.
- Retrieve updated movie poster paths from The Movie Database (TMDB).
- Cache poster lookups to reduce repeated API requests.
- Separate FastAPI backend and Streamlit frontend.

## Tech Stack

- Python
- FastAPI and Uvicorn
- Streamlit
- Pandas and NumPy
- Scikit-learn
- TF-IDF and cosine similarity
- TMDB API

## Project Structure

```text
movie-recommendation-system/
├── app.py
├── main.py
├── recommender.py
├── build_model.py
├── test_recommender.py
├── test_saved_model.py
├── movie_recommendation.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

The local `model/` directory contains generated model artifacts. The dataset, model artifacts, and `.env` file are not included in Git.

## Requirements

- Python compatible with the installed project dependencies
- The movie metadata CSV used to build the recommendation model
- A TMDB bearer token for current poster lookups

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/shridhartapakire/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies

```bash
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

### 3. Add the movie dataset

Place `movies_metadata.csv` in the project root.

### 4. Build the model artifacts

```bash
python build_model.py
```

This creates the files required by the API inside the local `model/` directory.

### 5. Configure TMDB

Create a `.env` file in the project root:

```text
TMDB_BEARER_TOKEN=your_tmdb_bearer_token
```

Keep this file private. Never commit your real token to GitHub.

### 6. Start the FastAPI backend

```bash
uvicorn main:app --reload
```

API documentation is available at:

`http://127.0.0.1:8000/docs`

### 7. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed by Streamlit.

## Testing

Validate the saved model:

```bash
python test_saved_model.py
```

Run the recommendation tests:

```bash
python test_recommender.py
```

The recommendation test script rebuilds the model and therefore requires the local dataset.

## How Recommendations Work

1. Movie metadata is cleaned and prepared.
2. Genres and plot descriptions are combined into text features.
3. TF-IDF converts the text into numerical vectors.
4. Cosine similarity measures the similarity between movie vectors.
5. The API returns movies with the highest similarity scores.

## Future Improvements

- Animated horizontal movie poster carousels.
- Improved poster availability and caching.
- Faster search and recommendation responses.
- Automated API tests.
- Deployment of the backend and frontend.

## Author

Shridhar Tapakire
