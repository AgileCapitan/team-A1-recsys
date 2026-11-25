from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib

from src.models.recommender import recommend
from src.serving.new_user import NewUserRecommender

logger = logging.getLogger(__name__)

REQUIRED_BUNDLE_KEYS = ("model_version", "model", "history", "popularity", "titles")
DEFAULT_MODEL_ARTIFACT_PATH = "artifacts/movielens-v1.joblib"
DEFAULT_CATALOG_PATH = "artifacts/movie-catalog.csv"


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

    def __init__(
        self,
        model_version: str = "fallback-v0.1",
        fallback_reason: str | None = None,
    ) -> None:
        self.model_version = model_version
        self.using_fallback = True
        self.artifact_loaded = False
        self.catalog_loaded = False
        self.fallback_reason = fallback_reason
        self.catalog_error = fallback_reason

    def predict(self, user_id: int, top_k: int = 10) -> list[dict[str, Any]]:
        """Return popular items up to top_k."""
        return [dict(item) for item in self.POPULAR_MOVIES[:top_k]]


class ModelPredictor(BasePredictor):
    """Model-backed predictor loading a trained MovieLens bundle and optional catalog."""

    def __init__(
        self,
        model_path: str | Path,
        catalog_path: str | Path | None = None,
        regularization: float = 3.0,
    ) -> None:
        self.model_path = Path(model_path)
        bundle = joblib.load(self.model_path)
        if not isinstance(bundle, dict):
            raise ValueError(f"Model artifact at {self.model_path} must be a mapping.")

        missing_keys = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle]
        if missing_keys:
            raise ValueError(
                f"Model artifact at {self.model_path} is missing keys: {', '.join(missing_keys)}"
            )

        self.model_version = str(bundle["model_version"])
        self.model = bundle["model"]
        self.history = bundle["history"]
        self.popularity = bundle["popularity"]
        self.titles = bundle["titles"]
        self.using_fallback = False
        self.artifact_loaded = True
        self.fallback_reason: str | None = None

        self.new_user_recommender: NewUserRecommender | None = None
        self.catalog_loaded = False
        self.catalog_error: str | None = None

        resolved_catalog = Path(catalog_path) if catalog_path is not None else None
        if resolved_catalog is not None and resolved_catalog.is_file():
            self.new_user_recommender = NewUserRecommender.from_bundle(
                bundle,
                resolved_catalog,
                regularization=regularization,
            )
            self.catalog_loaded = True
        else:
            reason = (
                "No catalog path configured."
                if resolved_catalog is None
                else f"Movie catalog not found at {resolved_catalog}."
            )
            self.catalog_error = reason
            logger.warning("Catalog unavailable for %s: %s", self.model_version, reason)

    @property
    def genres(self) -> list[str]:
        if self.new_user_recommender is None:
            raise RuntimeError(self.catalog_error or "Movie catalog is not loaded.")
        return list(self.new_user_recommender.genres)

    def _format_movie(self, movie_id: int, score: float) -> dict[str, Any]:
        item: dict[str, Any] = {
            "movie_id": int(movie_id),
            "title": str(self.titles[movie_id]),
            "score": float(score),
        }
        if self.new_user_recommender is not None:
            meta = self.new_user_recommender.catalog.get(int(movie_id))
            if meta is not None:
                item["genres"] = list(meta["genres"])
                item["release_year"] = meta["release_year"]
        return item

    def predict(
        self,
        user_id: int,
        top_k: int = 10,
        genres: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank unseen movies for an existing user (MF) or unknown user (popularity)."""
        if top_k < 1:
            return []

        selected_genres = set(genres) if genres else None
        if selected_genres:
            if self.new_user_recommender is None:
                raise RuntimeError(
                    self.catalog_error or "Genre filtering requires a loaded movie catalog."
                )
            if selected_genres.difference(self.new_user_recommender.genres):
                raise ValueError("Choose at least one available genre.")
            candidate_count = len(self.model.movie_ids)
        else:
            candidate_count = top_k

        ranked = recommend(
            user_id=user_id,
            n=candidate_count,
            model=self.model,
            popularity=self.popularity,
            history=self.history,
            method="mf",
        )

        results: list[dict[str, Any]] = []
        for movie_id, score in ranked.items():
            mid = int(movie_id)
            if selected_genres and self.new_user_recommender is not None:
                meta = self.new_user_recommender.catalog.get(mid)
                if meta is None or selected_genres.isdisjoint(meta["genres"]):
                    continue
            results.append(self._format_movie(mid, float(score)))
            if len(results) >= top_k:
                break

        return results

    def rating_options(self, genres: list[str], per_genre: int = 25) -> list[dict[str, Any]]:
        """Return popular rateable movies matching the requested genres."""
        if self.new_user_recommender is None:
            raise RuntimeError(self.catalog_error or "Movie catalog is not loaded.")

        selected_genres = set(genres)
        if not selected_genres or selected_genres.difference(self.new_user_recommender.genres):
            raise ValueError("Choose at least one available genre.")

        movie_ids = self.new_user_recommender.rating_options(genres, per_genre=per_genre)
        return [
            {
                "movie_id": movie["movie_id"],
                "title": movie["title"],
                "genres": list(movie["genres"]),
                "release_year": movie["release_year"],
            }
            for movie_id in movie_ids
            for movie in [self.new_user_recommender.catalog[movie_id]]
        ]

    def recommend_new_user(
        self,
        genres: list[str],
        ratings: dict[int, int],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score unseen movies for a cold-start user from inline genre and rating inputs."""
        if self.new_user_recommender is None:
            raise RuntimeError(self.catalog_error or "Movie catalog is not loaded.")

        ranked = self.new_user_recommender.recommend(
            genres=genres,
            ratings=ratings,
            top_k=top_k,
        )
        return [
            {
                "movie_id": movie["movie_id"],
                "title": movie["title"],
                "score": float(movie["score"]),
                "genres": list(movie["genres"]),
                "release_year": movie["release_year"],
            }
            for movie in ranked
        ]


def create_predictor(
    model_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    allow_fallback: bool = True,
    regularization: float = 3.0,
) -> BasePredictor:
    """Load the configured ModelPredictor or explicitly surface fallback/error state."""
    resolved_model = Path(
        model_path
        if model_path is not None
        else os.getenv("MODEL_ARTIFACT_PATH", DEFAULT_MODEL_ARTIFACT_PATH)
    )
    resolved_catalog = Path(
        catalog_path
        if catalog_path is not None
        else os.getenv("CATALOG_PATH", DEFAULT_CATALOG_PATH)
    )
    if not resolved_catalog.is_file() and resolved_catalog.name == "movie-catalog.csv":
        b2_catalog = resolved_catalog.with_name("movies.csv")
        if b2_catalog.is_file():
            resolved_catalog = b2_catalog

    try:
        return ModelPredictor(
            model_path=resolved_model,
            catalog_path=resolved_catalog,
            regularization=regularization,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if not allow_fallback:
            raise
        logger.warning(
            "Failed to load model artifact from %s (%s); using FallbackRecommender.",
            resolved_model,
            reason,
        )
        return FallbackRecommender(fallback_reason=reason)
