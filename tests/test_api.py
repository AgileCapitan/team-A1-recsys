import pytest
from fastapi.testclient import TestClient
from api.main import app, get_predictor
from src.serving.predictor import BasePredictor

client = TestClient(app)


class MockPredictor(BasePredictor):
    model_version = "mock-v1"

    def predict(self, user_id: int, top_k: int = 10):
        return [
            {"movie_id": i, "title": f"Test Movie {i}", "score": 5.0 - (0.1 * i)}
            for i in range(1, top_k + 1)
        ]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_openapi_docs_available():
    """Verify OpenAPI and documentation endpoints are served (Acceptance Criteria 1)."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi_spec = response.json()
    assert "paths" in openapi_spec
    assert "/recommend" in openapi_spec["paths"]

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200


def test_recommend_post_valid_request():
    """Verify valid recommendation request returns structured JSON (Acceptance Criteria 1 & 2)."""
    payload = {"user_id": 42, "top_k": 3}
    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 42
    assert "model_version" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) == 3

    first_item = data["recommendations"][0]
    assert "movie_id" in first_item
    assert "title" in first_item
    assert "score" in first_item
    assert isinstance(first_item["movie_id"], int)
    assert isinstance(first_item["title"], str)
    assert isinstance(first_item["score"], (int, float))


def test_recommend_get_valid_request():
    """Verify GET recommendation endpoint works as expected."""
    response = client.get("/recommend/42?top_k=2")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 42
    assert len(data["recommendations"]) == 2


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
def test_recommend_invalid_requests_produce_meaningful_errors(invalid_payload, expected_error_field):
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
