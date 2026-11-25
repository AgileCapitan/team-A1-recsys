import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_predictor
from src.models.batch_recommend import train_and_save
from src.serving.predictor import BasePredictor, FallbackRecommender, ModelPredictor

client = TestClient(app)


class MockPredictor(BasePredictor):
    model_version = "mock-v1"

    def predict(self, user_id: int, top_k: int = 10):
        return [
            {"movie_id": i, "title": f"Test Movie {i}", "score": 5.0 - (0.1 * i)}
            for i in range(1, top_k + 1)
        ]


def _build_fixture_predictor(tmp_path: Path, *, with_catalog: bool = True) -> ModelPredictor:
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
        "4,50,5,Funny Sci-Fi (2002)\n"
        "5,40,3,Action Hero (1999)\n"
        "5,60,5,Drama Story (1994)\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.joblib"
    train_and_save(ratings_csv, model_path)

    catalog_path = None
    if with_catalog:
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
                    ("60", "Drama Story (1994)", "Drama", "1994"),
                ]
            )

    return ModelPredictor(model_path=model_path, catalog_path=catalog_path)


def test_health_endpoint_with_model_predictor(tmp_path):
    predictor = _build_fixture_predictor(tmp_path)
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_version"] == "movielens-v1"
        assert data["using_fallback"] is False
        assert data["artifact_loaded"] is True
        assert data["catalog_loaded"] is True
        assert data["fallback_reason"] is None
    finally:
        app.dependency_overrides.clear()


def test_health_endpoint_surfaces_explicit_fallback():
    fallback = FallbackRecommender(fallback_reason="FileNotFoundError: missing.joblib")
    app.dependency_overrides[get_predictor] = lambda: fallback
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_version"] == "fallback-v0.1"
        assert data["using_fallback"] is True
        assert data["artifact_loaded"] is False
        assert data["catalog_loaded"] is False
        assert "missing.joblib" in data["fallback_reason"]
    finally:
        app.dependency_overrides.clear()


def test_openapi_docs_available():
    """Verify OpenAPI and documentation endpoints are served (Acceptance Criteria 1)."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi_spec = response.json()
    assert "paths" in openapi_spec
    assert "/recommend" in openapi_spec["paths"]
    assert "/recommend/new-user" in openapi_spec["paths"]
    assert "/catalog/genres" in openapi_spec["paths"]
    assert "/catalog/options" in openapi_spec["paths"]

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200


def test_recommend_post_valid_request(tmp_path):
    """Verify valid recommendation request returns structured JSON (Acceptance Criteria 1 & 2)."""
    predictor = _build_fixture_predictor(tmp_path)
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        payload = {"user_id": 1, "top_k": 3}
        response = client.post("/recommend", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert data["model_version"] == "movielens-v1"
        assert data["using_fallback"] is False
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) == 3

        first_item = data["recommendations"][0]
        assert isinstance(first_item["movie_id"], int)
        assert isinstance(first_item["title"], str)
        assert isinstance(first_item["score"], (int, float))
        assert isinstance(first_item["genres"], list)
        assert first_item["movie_id"] not in {10, 20}
    finally:
        app.dependency_overrides.clear()


def test_recommend_get_valid_request(tmp_path):
    """Verify GET recommendation endpoint works as expected."""
    predictor = _build_fixture_predictor(tmp_path)
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        response = client.get("/recommend/1?top_k=2&genres=Action")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert len(data["recommendations"]) == 2
        assert all("Action" in rec["genres"] for rec in data["recommendations"])
    finally:
        app.dependency_overrides.clear()


def test_catalog_and_new_user_endpoints(tmp_path, monkeypatch):
    predictor = _build_fixture_predictor(tmp_path)
    posters_dir = tmp_path / "posters"
    posters_dir.mkdir()
    (posters_dir / "50.jpg").write_bytes(b"\xff\xd8\xff\xe0test-jpeg")
    monkeypatch.setenv("POSTERS_DIR", str(posters_dir))

    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        genres_resp = client.get("/catalog/genres")
        assert genres_resp.status_code == 200
        genres_data = genres_resp.json()
        assert genres_data["model_version"] == "movielens-v1"
        assert "Action" in genres_data["genres"]

        options_resp = client.get("/catalog/options", params=[("genres", "Action")])
        assert options_resp.status_code == 200
        options_data = options_resp.json()
        assert options_data["movies"]
        assert all("Action" in m["genres"] for m in options_data["movies"])

        new_user_resp = client.post(
            "/recommend/new-user",
            json={
                "genres": ["Action", "Comedy"],
                "ratings": {"10": 5, "30": 4, "40": 3},
                "top_k": 3,
            },
        )
        assert new_user_resp.status_code == 200
        rec_data = new_user_resp.json()
        assert rec_data["model_version"] == "movielens-v1"
        assert rec_data["using_fallback"] is False
        assert rec_data["recommendations"]
        rec_ids = {item["movie_id"] for item in rec_data["recommendations"]}
        assert rec_ids.isdisjoint({10, 30, 40})
        poster_flags = {
            item["movie_id"]: item["has_poster"] for item in rec_data["recommendations"]
        }
        if 50 in poster_flags:
            assert poster_flags[50] is True

        poster_ok = client.get("/posters/50")
        assert poster_ok.status_code == 200
        assert poster_ok.content == b"\xff\xd8\xff\xe0test-jpeg"

        poster_missing = client.get("/posters/999")
        assert poster_missing.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_new_user_and_catalog_validation_and_missing_catalog_errors(tmp_path):
    predictor = _build_fixture_predictor(tmp_path, with_catalog=False)
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        assert client.get("/catalog/genres").status_code == 503
        assert client.get("/catalog/options", params={"genres": "Action"}).status_code == 503
        assert (
            client.post(
                "/recommend/new-user",
                json={"genres": ["Action"], "ratings": {"10": 5, "30": 4, "40": 3}, "top_k": 3},
            ).status_code
            == 503
        )
    finally:
        app.dependency_overrides.clear()

    full_predictor = _build_fixture_predictor(tmp_path, with_catalog=True)
    app.dependency_overrides[get_predictor] = lambda: full_predictor
    try:
        # Insufficient ratings (empty preferences) returns 422.
        bad_ratings = client.post(
            "/recommend/new-user",
            json={"genres": ["Action"], "ratings": {}, "top_k": 3},
        )
        assert bad_ratings.status_code == 422

        bad_genre = client.get("/catalog/options", params={"genres": "UnknownGenre"})
        assert bad_genre.status_code == 422
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "invalid_payload,expected_error_field",
    [
        ({"user_id": 0, "top_k": 5}, "user_id"),  # user_id must be > 0
        ({"user_id": -1, "top_k": 5}, "user_id"),  # negative user_id
        ({"user_id": "not-an-int", "top_k": 5}, "user_id"),  # wrong type
        ({"top_k": 5}, "user_id"),  # missing required user_id
        ({"user_id": 1, "top_k": 0}, "top_k"),  # top_k must be >= 1
        ({"user_id": 1, "top_k": 101}, "top_k"),  # top_k exceeds max 100
        ({"user_id": 1, "top_k": "invalid"}, "top_k"),  # top_k invalid type
    ],
)
def test_recommend_invalid_requests_produce_meaningful_errors(
    invalid_payload, expected_error_field
):
    """Verify meaningful error responses for invalid inputs (Acceptance Criteria 3)."""
    response = client.post("/recommend", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
    error_data = response.json()
    assert "detail" in error_data

    # Check that error details mention the problematic field
    error_locations = [err["loc"][-1] for err in error_data["detail"]]
    assert expected_error_field in error_locations


def test_dependency_injection_override():
    """Verify the recommender can be easily swapped or mocked."""
    app.dependency_overrides[get_predictor] = lambda: MockPredictor()
    try:
        response = client.post("/recommend", json={"user_id": 99, "top_k": 2})
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "mock-v1"
        assert data["recommendations"][0]["title"] == "Test Movie 1"
    finally:
        app.dependency_overrides.clear()
