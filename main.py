import ast
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------
# Configuration
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

logger = logging.getLogger("cinematch")

app = FastAPI(
    title="Movie Recommendation API",
    description="Movie search and recommendations using the saved TF-IDF model.",
    version="2.0.0",
)

load_dotenv(BASE_DIR / ".env")

# The token is only ever used server-side. It is never logged or returned.
TMDB_BEARER_TOKEN = (os.getenv("TMDB_BEARER_TOKEN") or "").strip() or None
TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
    "accept": "application/json",
}

# -------------------------
# Get the saved model files (download from Hugging Face if missing)
# -------------------------
MODEL_FILES = ("movies.pkl", "tfidf_matrix.pkl", "tfidf.pkl")


def ensure_model_files():
    """
    Make sure the three model files exist in MODEL_DIR.

    Local development: the files already exist, so nothing is downloaded.
    Deployment (e.g. Render): the files are not in Git, so they are downloaded
    once at startup from a Hugging Face repo configured with these variables:
      HF_MODEL_REPO  e.g. "your-username/your-model-repo"
      HF_TOKEN       read token (only needed when the repo is private)
    The token is never printed or logged.
    """
    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if not missing:
        return

    repo_id = (os.getenv("HF_MODEL_REPO") or "").strip()
    token = (os.getenv("HF_TOKEN") or "").strip() or None

    if not repo_id:
        raise RuntimeError(
            f"Model files are missing ({', '.join(missing)}) and HF_MODEL_REPO is not set. "
            "Locally, run `python build_model.py`. On Render, set the HF_MODEL_REPO "
            "environment variable to your Hugging Face repo (username/repo-name)."
        )

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The 'huggingface_hub' package is required to download the model files. "
            "Add it to requirements.txt."
        ) from exc

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in missing:
        print(f"Downloading {name} from Hugging Face repo '{repo_id}'...", flush=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=name,
                token=token,
                local_dir=MODEL_DIR,
            )
        except RepositoryNotFoundError as exc:
            hint = (
                "HF_TOKEN is not set, which is required for a private repo."
                if token is None
                else "Check HF_MODEL_REPO for typos and that the token can read this repo."
            )
            raise RuntimeError(
                f"Hugging Face repo '{repo_id}' was not found or is not accessible. {hint}"
            ) from exc
        except EntryNotFoundError as exc:
            raise RuntimeError(
                f"'{name}' does not exist in Hugging Face repo '{repo_id}'. "
                f"The repo must contain: {', '.join(MODEL_FILES)}."
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            reason = (
                "authentication failed. The HF_TOKEN is invalid, expired or lacks read access"
                if status in (401, 403)
                else f"Hugging Face returned HTTP {status}"
            )
            raise RuntimeError(f"Could not download '{name}': {reason}.") from exc
        except Exception as exc:  # network errors, full disk, etc.
            raise RuntimeError(
                f"Could not download '{name}' from Hugging Face ({type(exc).__name__}). "
                "Check the network connection and available disk space."
            ) from exc

    still_missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if still_missing:
        raise RuntimeError(
            f"Download finished but files are still missing: {', '.join(still_missing)}"
        )


ensure_model_files()

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
# Precomputed lookup data (derived from the saved model, not hardcoded)
# -------------------------
def _numeric(column):
    return pd.to_numeric(movies[column], errors="coerce")


_title_cf = movies["title"].astype(str).str.strip().str.casefold()
_popularity = _numeric("popularity").fillna(0).to_numpy(dtype=float)
_votes = _numeric("vote_count").fillna(0).to_numpy(dtype=float)
_rating = _numeric("vote_average").fillna(0).to_numpy(dtype=float)
_year = _numeric("release_year").to_numpy(dtype=float)
_release = pd.to_datetime(movies["release_date"], errors="coerce")
_release_days = (_release - pd.Timestamp("1900-01-01")).dt.days.to_numpy(dtype=float)
_today_days = float((pd.Timestamp(date.today()) - pd.Timestamp("1900-01-01")).days)

_has_poster = movies["poster_path"].astype(str).str.startswith("/").to_numpy()
_is_adult = movies["adult"].astype(str).str.lower().eq("true").to_numpy()
# Titles with a poster, not adult-flagged: good candidates for visual rows.
_showable = _has_poster & ~_is_adult & (_title_cf.to_numpy() != "")

# Bayesian weighted rating (IMDB formula) so tiny vote counts don't dominate.
_voted = _votes > 0
_mean_rating = float(_rating[_voted].mean()) if _voted.any() else 0.0
_min_votes = float(np.quantile(_votes[_voted], 0.90)) if _voted.any() else 0.0
_weighted = (_votes / (_votes + _min_votes + 1e-9)) * _rating + (
    _min_votes / (_votes + _min_votes + 1e-9)
) * _mean_rating


def _parse_genres(value):
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
        return [i["name"] for i in items if isinstance(i, dict) and i.get("name")]
    except (ValueError, SyntaxError, TypeError):
        return []


_genre_lists = (
    [_parse_genres(v) for v in movies["genres"]]
    if "genres" in movies.columns
    else [str(v).split() for v in movies["genres_clean"]]
)
_genre_masks = {}
for _i, _names in enumerate(_genre_lists):
    for _name in _names:
        _genre_masks.setdefault(_name, np.zeros(len(movies), dtype=bool))[_i] = True


# -------------------------
# TMDB image lookups (server-side, cached, bounded, failure tolerant)
# -------------------------
_TTL_OK = 24 * 3600
_TTL_MISSING = 6 * 3600
_TTL_FAILED = 300  # transient failures are retried after 5 minutes
_BREAKER_FAILS = 4
_BREAKER_PAUSE = 60

_tmdb_cache = {}
_cache_lock = threading.Lock()
_breaker = {"fails": 0, "until": 0.0}

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=1,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
    ),
)
_executor = ThreadPoolExecutor(max_workers=12)


def valid_tmdb_id(value):
    """Return an int TMDB id, or None for missing / malformed dataset ids."""
    text = str(value).strip() if value is not None else ""
    if text.isdigit() and int(text) > 0:
        return int(text)
    return None


def _tmdb_available():
    return bool(TMDB_BEARER_TOKEN) and time.time() >= _breaker["until"]


def _cache_get(key):
    with _cache_lock:
        hit = _tmdb_cache.get(key)
    if hit and hit[0] > time.time():
        return True, hit[1]
    return False, None


def _cache_put(key, value, ttl):
    with _cache_lock:
        if len(_tmdb_cache) > 5000:
            _tmdb_cache.clear()
        _tmdb_cache[key] = (time.time() + ttl, value)


def tmdb_images(tmdb_id):
    """Current poster/backdrop paths for one TMDB id, or None."""
    movie_id = valid_tmdb_id(tmdb_id)
    if movie_id is None or not TMDB_BEARER_TOKEN:
        return None

    cached, value = _cache_get(movie_id)
    if cached:
        return value

    if time.time() < _breaker["until"]:
        return None

    try:
        response = _session.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            headers=TMDB_HEADERS,
            timeout=(3, 6),
        )
    except requests.RequestException as exc:
        _breaker["fails"] += 1
        if _breaker["fails"] >= _BREAKER_FAILS:
            _breaker["until"] = time.time() + _BREAKER_PAUSE
            _breaker["fails"] = 0
            logger.warning(
                "TMDB unreachable (%s); pausing lookups.", type(exc).__name__
            )
        _cache_put(movie_id, None, _TTL_FAILED)
        return None

    _breaker["fails"] = 0

    if response.status_code == 200:
        data = response.json()
        value = {
            "poster_path": data.get("poster_path"),
            "backdrop_path": data.get("backdrop_path"),
        }
        _cache_put(movie_id, value, _TTL_OK)
        return value

    if response.status_code == 404:
        _cache_put(movie_id, None, _TTL_MISSING)
    else:
        logger.warning(
            "TMDB returned HTTP %s for movie %s", response.status_code, movie_id
        )
        _cache_put(movie_id, None, _TTL_FAILED)
    return None


def attach_images(items, budget=4.0):
    """
    Add current TMDB images to a list of movie dicts, in parallel.

    Each item keeps the dataset poster as `poster_fallback`. Lookups that don't
    finish inside `budget` seconds are skipped (they still finish in the
    background and are cached for the next request).
    """
    by_id = {}
    for item in items:
        item["poster_fallback"] = item.get("poster_path")
        item.setdefault("backdrop_path", None)
        movie_id = valid_tmdb_id(item.get("id"))
        if movie_id:
            by_id.setdefault(movie_id, []).append(item)

    if not by_id or not _tmdb_available():
        return items

    futures = {_executor.submit(tmdb_images, mid): mid for mid in by_id}
    done, _ = wait(futures, timeout=budget)

    for future in done:
        try:
            data = future.result()
        except Exception:  # never let an image lookup break an API response
            continue
        if not data:
            continue
        for item in by_id[futures[future]]:
            item["poster_path"] = data.get("poster_path") or item["poster_path"]
            item["backdrop_path"] = data.get("backdrop_path")
    return items


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


def full_dict(index):
    """Every dataset column for one movie (the original response shape)."""
    row = movies.iloc[index]
    result = {key: clean_value(value) for key, value in row.items()}
    result["genres_list"] = _genre_lists[index]
    return result


def card_dict(index):
    """A lightweight movie dict for rows, grids and the hero."""
    row = movies.iloc[index]
    overview = clean_value(row.get("overview")) or ""
    return {
        "id": clean_value(row.get("id")),
        "title": clean_value(row.get("title")),
        "overview": overview,
        "poster_path": clean_value(row.get("poster_path")),
        "release_date": clean_value(row.get("release_date")),
        "vote_average": clean_value(row.get("vote_average")),
        "vote_count": clean_value(row.get("vote_count")),
        "runtime": clean_value(row.get("runtime")),
        "genres_clean": clean_value(row.get("genres_clean")),
        "genres_list": _genre_lists[index],
    }


def movie_to_dict(index):
    """Convert one movie row to a JSON-safe dictionary with current images."""
    return attach_images([full_dict(index)])[0]


def find_movie_index(title, year=None):
    """Find a movie by exact title (ignoring case), optionally by release year."""
    mask = _title_cf == title.strip().casefold()
    if year is not None:
        mask = mask & (_year == year)

    matches = np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else mask)

    if len(matches) == 0:
        suffix = f" ({year})" if year is not None else ""
        raise HTTPException(
            status_code=404,
            detail=f"Movie not found: {title}{suffix}",
        )

    return int(matches[0])


def find_movie_index_or_none(title):
    try:
        return find_movie_index(title)
    except HTTPException:
        return None


def pick_top(score, count, exclude=(), mask=None):
    """Indices of the highest scores, skipping excluded movies and repeat titles."""
    mask = _showable if mask is None else mask
    ranked = np.where(mask, score, -np.inf)
    exclude = set(exclude)
    seen_titles, picked = set(), []

    for index in np.argsort(-ranked, kind="stable"):
        if not np.isfinite(ranked[index]) or len(picked) >= count:
            break
        title = _title_cf.iloc[index]
        if index in exclude or title in seen_titles:
            continue
        seen_titles.add(title)
        picked.append(int(index))
    return picked


def similarity_to(indices):
    """Cosine similarity of every movie to the centroid of `indices`."""
    centroid = np.asarray(tfidf_matrix[indices].mean(axis=0)).ravel()
    return np.asarray(tfidf_matrix @ centroid).ravel()


# -------------------------
# API endpoints
# -------------------------
@app.get("/")
def home():
    return {
        "message": "Movie Recommendation API is running.",
        "movies_loaded": len(movies),
        "endpoints": [
            "/health",
            "/search",
            "/movie",
            "/recommend",
            "/home",
            "/browse",
            "/genres",
            "/docs",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "movies_loaded": len(movies),
        "tfidf_rows": tfidf_matrix.shape[0],
        "tmdb_configured": bool(TMDB_BEARER_TOKEN),
    }


@app.get("/search")
def search_movies(
    query: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(20, ge=1, le=100),
):
    """Search local movie titles (exact matches first, then most-voted)."""
    query = query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    titles = movies["title"].fillna("").astype(str)
    hits = np.flatnonzero(
        titles.str.contains(query, case=False, regex=False).to_numpy()
    )

    exact = (_title_cf.to_numpy()[hits] == query.casefold()).astype(int)
    order = np.lexsort((-_votes[hits], -exact))
    indices = [int(i) for i in hits[order][:limit]]

    results = attach_images([full_dict(i) for i in indices])

    return {
        "query": query,
        "count": len(indices),
        "results": results,
    }


@app.get("/movie")
def movie_details(
    title: str = Query(..., min_length=1, max_length=200),
    year: Optional[int] = Query(None, ge=1850, le=2100),
):
    """Return details for an exact movie title (optionally a specific year)."""
    index = find_movie_index(title, year)
    return movie_to_dict(index)


@app.get("/recommend")
def recommend_movies(
    title: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(10, ge=1, le=50),
    year: Optional[int] = Query(None, ge=1850, le=2100),
):
    """Recommend similar movies using the saved TF-IDF matrix."""
    index = find_movie_index(title, year)

    query_vector = tfidf_matrix[index]
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()

    # Exclude the selected movie.
    scores[index] = -1

    # Select the highest-scoring movies.
    top_indices = np.argsort(-scores)[:limit]

    items = []
    for movie_index in top_indices:
        if scores[movie_index] < 0:
            continue
        item = full_dict(int(movie_index))
        item["similarity_score"] = round(float(scores[movie_index]), 4)
        items.append(item)

    attach_images(items)

    return {
        "selected_movie": str(movies.iloc[index]["title"]),
        "count": len(items),
        "recommendations": items,
    }


@app.get("/genres")
def list_genres():
    """Genres present in the dataset, with movie counts."""
    return {
        "genres": sorted(
            (
                {"name": name, "count": int(mask.sum())}
                for name, mask in _genre_masks.items()
            ),
            key=lambda g: -g["count"],
        )
    }


@app.get("/browse")
def browse_movies(
    genre: Optional[str] = Query(None, max_length=50),
    sort: str = Query("popularity", pattern="^(popularity|rating|votes|newest)$"),
    limit: int = Query(40, ge=1, le=100),
):
    """Browse the catalogue by genre and sort order (Discover / Trending)."""
    mask = _showable.copy()

    if genre:
        if genre not in _genre_masks:
            raise HTTPException(status_code=404, detail=f"Unknown genre: {genre}")
        mask &= _genre_masks[genre]

    if sort == "popularity":
        score, mask = _popularity, mask & (_votes >= 5)
    elif sort == "rating":
        score, mask = _weighted, mask & (_votes >= 20)
    elif sort == "votes":
        score = _votes
    else:  # newest
        recent = np.nan_to_num(_release_days, nan=-1)
        score = recent
        mask = mask & (_votes >= 10) & (recent > 0) & (recent <= _today_days)

    indices = pick_top(score, limit, mask=mask)
    results = attach_images([card_dict(i) for i in indices])

    return {
        "genre": genre,
        "sort": sort,
        "total": int(mask.sum()),
        "count": len(results),
        "results": results,
    }


@app.get("/home")
def home_feed(
    watched: List[str] = Query(default=[], max_length=10),
    per_row: int = Query(20, ge=6, le=40),
):
    """
    Data for the discovery page: a featured movie plus content rows.

    `watched` is an optional list of titles the visitor recently opened. It
    personalises the "Because You Watched" and "Recommended For You" rows using
    the same TF-IDF model; without it those rows are seeded from popular movies.
    """
    seeds = []
    for title in watched[-5:]:
        index = find_movie_index_or_none(title)
        if index is not None and index not in seeds:
            seeds.append(index)

    # Featured movie: rotates daily among the most popular well-described titles.
    candidates = pick_top(
        _popularity,
        25,
        mask=_showable
        & (_votes >= 50)
        & (movies["overview"].astype(str).str.len() > 60).to_numpy(),
    )
    hero_index = candidates[date.today().toordinal() % len(candidates)]
    used = {hero_index, *seeds}

    rows = []

    def add_row(key, title, subtitle, indices):
        used.update(indices)
        rows.append(
            {
                "key": key,
                "title": title,
                "subtitle": subtitle,
                "movies": [card_dict(i) for i in indices],
            }
        )

    add_row(
        "trending",
        "Trending Movies",
        "Highest popularity score in the dataset",
        pick_top(_popularity, per_row, exclude=used, mask=_showable & (_votes >= 5)),
    )

    if seeds:
        anchor = seeds[-1]
        because_title = f"Because You Watched {movies.iloc[anchor]['title']}"
        because_sub = "Most similar plots and genres"
    else:
        anchor = hero_index
        because_title = f"If You Liked {movies.iloc[anchor]['title']}"
        because_sub = "Most similar plots and genres"
    add_row(
        "because",
        because_title,
        because_sub,
        pick_top(similarity_to([anchor]), per_row, exclude=used),
    )

    add_row(
        "top_rated",
        "Top Rated",
        "Weighted by rating and number of votes",
        pick_top(
            _weighted, per_row, exclude=used, mask=_showable & (_votes >= _min_votes)
        ),
    )

    if seeds:
        centroid_seeds = seeds
        rec_title, rec_sub = (
            "Recommended For You",
            "Based on movies you recently opened",
        )
    else:
        centroid_seeds = pick_top(_popularity, 3, exclude=used)
        rec_title = "Hand-Picked Recommendations"
        rec_sub = (
            "Based on today's popular movies. Open a few titles to personalise this row"
        )
    add_row(
        "recommended",
        rec_title,
        rec_sub,
        pick_top(
            similarity_to(centroid_seeds), per_row, exclude=used | set(centroid_seeds)
        ),
    )

    add_row(
        "audience",
        "Audience Favourites",
        "Most-voted movies",
        pick_top(_votes, per_row, exclude=used),
    )

    hero = card_dict(hero_index)
    everything = [hero] + [m for row in rows for m in row["movies"]]
    attach_images(everything, budget=4.0)

    return {"hero": hero, "rows": rows, "personalised": bool(seeds)}
