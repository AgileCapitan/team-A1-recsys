import csv
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

from api.main import app, get_predictor
from app.api_client import APIError, RecommenderAPIClient
from src.models.batch_recommend import train_and_save
from src.serving.predictor import ModelPredictor


class _FastAPISessionAdapter:
    """Adapter allowing RecommenderAPIClient to run directly against FastAPI TestClient."""

    def __init__(self, test_client: TestClient) -> None:
        self._client = test_client

    def request(self, method: str, url: str, json=None, params=None, timeout=None):
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        return self._client.request(method, f"/{path}", json=json, params=params)


def _build_fixture_predictor(tmp_path: Path) -> ModelPredictor:
    ratings_csv = tmp_path / "ratings.csv"
    ratings_csv.write_text(
        "user_id,movie_id,rating,title\n"
        "1,10,5,Toy Action (1995)\n"
        "1,20,4,Comedy Night (1998)\n"
        "2,10,4,Toy Action (1995)\n"
        "2,30,5,Space Quest (2001)\n"
        "3,20,5,Comedy Night (1998)\n"
        "3,40,4,Action Hero (1999)\n"
        "4,30,4,Space Quest (2001)\n"
        "4,50,5,Funny Sci-Fi (2002)\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.joblib"
    train_and_save(ratings_csv, model_path)

    catalog_path = tmp_path / "movie-catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["movie_id", "title", "genres", "release_year"])
        writer.writerows(
            [
                ("10", "Toy Action (1995)", "Action|Adventure", "1995"),
                ("20", "Comedy Night (1998)", "Comedy", "1998"),
                ("30", "Space Quest (2001)", "Action|Sci-Fi", "2001"),
                ("40", "Action Hero (1999)", "Action", "1999"),
                ("50", "Funny Sci-Fi (2002)", "Comedy|Sci-Fi", "2002"),
            ]
        )

    return ModelPredictor(model_path=model_path, catalog_path=catalog_path)


def test_api_client_end_to_end_with_fastapi(tmp_path, monkeypatch):
    predictor = _build_fixture_predictor(tmp_path)
    posters_dir = tmp_path / "posters"
    posters_dir.mkdir()
    (posters_dir / "50.jpg").write_bytes(b"jpeg-bytes")
    monkeypatch.setenv("POSTERS_DIR", str(posters_dir))

    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        test_client = TestClient(app)
        api_client = RecommenderAPIClient(
            base_url="http://backend:8000",
            session=_FastAPISessionAdapter(test_client),
        )

        health = api_client.get_health()
        assert health["status"] == "healthy"
        assert health["model_version"] == "movielens-v1"

        genres = api_client.get_genres()
        assert "Action" in genres["genres"]

        options = api_client.get_rating_options(genres=["Action"], per_genre=10)
        assert options["movies"]

        new_user_recs = api_client.recommend_new_user(
            genres=["Action", "Comedy"],
            ratings={10: 5, 30: 4, 40: 3},
            top_k=3,
        )
        assert len(new_user_recs["recommendations"]) >= 1
        assert all(rec["movie_id"] not in {10, 30, 40} for rec in new_user_recs["recommendations"])

        user_recs = api_client.recommend_for_user(user_id=1, top_k=2, genres=["Action"])
        assert user_recs["user_id"] == 1
        assert len(user_recs["recommendations"]) == 2

        assert api_client.get_poster_bytes(50) == b"jpeg-bytes"
        assert api_client.get_poster_bytes(999) is None
    finally:
        app.dependency_overrides.clear()


def test_api_client_extracts_validation_and_server_errors(tmp_path):
    predictor = _build_fixture_predictor(tmp_path)
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        test_client = TestClient(app)
        api_client = RecommenderAPIClient(
            base_url="http://backend:8000",
            session=_FastAPISessionAdapter(test_client),
        )

        with pytest.raises(APIError) as exc_info:
            api_client.recommend_new_user(genres=["Action"], ratings={10: 5}, top_k=5)
        assert exc_info.value.status_code == 422
        assert "3 and 10" in exc_info.value.detail

        with pytest.raises(APIError) as exc_info_2:
            api_client.recommend_for_user(user_id=0, top_k=5)
        assert exc_info_2.value.status_code == 422
        assert "user_id" in exc_info_2.value.detail
    finally:
        app.dependency_overrides.clear()


def test_api_client_handles_connection_and_timeout_errors():
    session = Mock()
    session.request.side_effect = requests.exceptions.ConnectionError("refused")
    api_client = RecommenderAPIClient("http://backend:8000", session=session)

    with pytest.raises(APIError, match="Could not connect"):
        api_client.get_health()

    session.request.side_effect = requests.exceptions.Timeout("timed out")
    with pytest.raises(APIError, match="timed out"):
        api_client.get_genres()

    # Poster helper degrades gracefully on network errors.
    assert api_client.get_poster_bytes(1) is None
