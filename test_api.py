"""
API checks that run without the network (TMDB is mocked).

    python test_api.py

Requires the saved model (python build_model.py) and `pip install httpx`
(FastAPI's TestClient needs it).
"""
import time
from unittest import mock

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code, self._payload = status, payload or {}
        self.text = ""

    def json(self):
        return self._payload


def reset_tmdb(token="test-token"):
    main._tmdb_cache.clear()
    main._breaker.update(fails=0, until=0.0)
    main.TMDB_BEARER_TOKEN = token
    main.TMDB_HEADERS["Authorization"] = f"Bearer {token}"


# --- existing endpoints keep working -------------------------------------
reset_tmdb(None)
assert client.get("/health").json()["status"] == "ok"
assert client.get("/movie", params={"title": "toy story"}).json()["title"] == "Toy Story"
assert client.get("/movie", params={"title": "Nope Nope Nope"}).status_code == 404
recs = client.get("/recommend", params={"title": "Toy Story", "limit": 5}).json()
assert recs["count"] == 5 and recs["recommendations"][0]["title"] == "Toy Story 2"
search = client.get("/search", params={"query": "batman", "limit": 10}).json()
assert search["count"] == 10 and "poster_path" in search["results"][0]
print("existing endpoints OK")

# --- duplicate titles can be disambiguated by year ------------------------
old = client.get("/movie", params={"title": "12 Angry Men", "year": 1957}).json()
new = client.get("/movie", params={"title": "12 Angry Men", "year": 1997}).json()
assert old["release_date"].startswith("1957") and new["release_date"].startswith("1997")
print("year disambiguation OK")

# --- home feed, browse, genres --------------------------------------------
feed = client.get("/home").json()
assert len(feed["rows"]) == 5 and feed["hero"]["title"]
assert all(len(r["movies"]) >= 10 for r in feed["rows"])
titles = [m["title"] for r in feed["rows"] for m in r["movies"]]
assert len(titles) == len(set(titles)), "rows should not repeat movies"
assert all(m["poster_path"] for r in feed["rows"] for m in r["movies"])
assert not feed["personalised"]

personal = client.get("/home", params={"watched": ["Toy Story", "Finding Nemo"]}).json()
assert personal["personalised"]
assert personal["rows"][1]["title"] == "Because You Watched Finding Nemo"
assert personal["rows"][3]["title"] == "Recommended For You"
print("home feed OK:", [r["title"] for r in feed["rows"]])

browse = client.get("/browse", params={"genre": "Animation", "sort": "rating", "limit": 12}).json()
assert browse["count"] == 12 and all("Animation" in m["genres_list"] for m in browse["results"])
for sort in ("popularity", "rating", "votes", "newest"):
    assert client.get("/browse", params={"sort": sort, "limit": 5}).json()["count"] == 5
assert client.get("/browse", params={"genre": "Nonsense"}).status_code == 404
assert client.get("/genres").json()["genres"][0]["count"] > 1000
print("browse/genres OK")

# --- TMDB id validation ----------------------------------------------------
assert main.valid_tmdb_id("862") == 862
assert all(main.valid_tmdb_id(v) is None for v in (None, "", "abc", "0", "1997-08-20", float("nan")))

# --- TMDB success: current poster + backdrop are attached ------------------
reset_tmdb()
ok = FakeResponse(200, {"poster_path": "/new.jpg", "backdrop_path": "/bd.jpg"})
with mock.patch.object(main._session, "get", return_value=ok) as get:
    movie = client.get("/movie", params={"title": "Toy Story"}).json()
    assert movie["poster_path"] == "/new.jpg" and movie["backdrop_path"] == "/bd.jpg"
    assert movie["poster_fallback"] == "/rhIRbceoE9lR4veEXuwCC2wARtG.jpg"
    client.get("/movie", params={"title": "Toy Story"})
    assert get.call_count == 1, "second lookup must come from cache"
    assert "test-token" not in str(movie)
print("TMDB success + cache OK")

# --- TMDB failure: API still answers, falls back to dataset poster ---------
import requests

reset_tmdb()
with mock.patch.object(main._session, "get", side_effect=requests.ConnectionError("boom")):
    t = time.time()
    movie = client.get("/movie", params={"title": "Toy Story"}).json()
    assert movie["poster_path"] == movie["poster_fallback"] and movie["backdrop_path"] is None
    assert client.get("/home").status_code == 200
    assert time.time() - t < 15
assert main._breaker["until"] > time.time(), "circuit breaker should open after repeated failures"
print("TMDB outage fallback + circuit breaker OK")

# --- transient failures are NOT cached forever (old lru_cache bug) ---------
reset_tmdb()
with mock.patch.object(main._session, "get", return_value=FakeResponse(500)):
    assert main.tmdb_images(862) is None
main._tmdb_cache[862] = (time.time() - 1, None)  # simulate TTL expiry
with mock.patch.object(main._session, "get", return_value=ok):
    assert main.tmdb_images(862)["poster_path"] == "/new.jpg"
print("failed lookups are retried OK")

print("\nAll API tests passed!")