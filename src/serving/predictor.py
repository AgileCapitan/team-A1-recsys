from abc import ABC, abstractmethod
from typing import Any


class BasePredictor(ABC):
    """Abstract base class for recommendation predictors."""

    @abstractmethod
    def predict(self, user_id: int, top_k: int = 10) -> list[dict[str, Any]]:
        """Return top-k recommendations for a given user."""
        pass


class FallbackRecommender(BasePredictor):
    """
    Fallback / baseline recommender returning top popular movies.
    Used before a trained ML model artifact is loaded or as a graceful fallback.
    """

    POPULAR_MOVIES = [
        {"movie_id": 1, "title": "Toy Story (1995)", "score": 4.8},
        {"movie_id": 260, "title": "Star Wars: Episode IV - A New Hope (1977)", "score": 4.7},
        {"movie_id": 296, "title": "Pulp Fiction (1994)", "score": 4.6},
        {"movie_id": 318, "title": "The Shawshank Redemption (1994)", "score": 4.6},
        {"movie_id": 356, "title": "Forrest Gump (1994)", "score": 4.5},
        {"movie_id": 593, "title": "The Silence of the Lambs (1991)", "score": 4.5},
        {"movie_id": 2571, "title": "The Matrix (1999)", "score": 4.4},
        {"movie_id": 527, "title": "Schindler's List (1993)", "score": 4.4},
        {"movie_id": 858, "title": "The Godfather (1972)", "score": 4.3},
        {"movie_id": 50, "title": "The Usual Suspects (1995)", "score": 4.3},
    ]

    def __init__(self, model_version: str = "fallback-v0.1"):
        self.model_version = model_version

    def predict(self, user_id: int, top_k: int = 10) -> list[dict[str, Any]]:
        # Return popular items up to top_k
        return self.POPULAR_MOVIES[:top_k]
