from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi import Path as FastPath
from fastapi.responses import FileResponse

from api.schemas import (
    CatalogGenresResponse,
    CatalogMovie,
    CatalogOptionsResponse,
    HealthResponse,
    MovieRecommendation,
    NewUserRecommendationRequest,
    NewUserRecommendationResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from src.serving.predictor import BasePredictor, ModelPredictor, create_predictor

DEFAULT_POSTERS_DIR = "artifacts/posters"

_default_predictor: BasePredictor | None = None


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _build_predictor_from_env() -> BasePredictor:
    return create_predictor(
        model_path=os.getenv("MODEL_ARTIFACT_PATH"),
        catalog_path=os.getenv("CATALOG_PATH"),
        allow_fallback=_parse_bool_env("ALLOW_FALLBACK_PREDICTOR", True),
    )


def _posters_dir() -> Path:
    return Path(os.getenv("POSTERS_DIR", DEFAULT_POSTERS_DIR))


def _has_poster(movie_id: int) -> bool:
    return (_posters_dir() / f"{movie_id}.jpg").is_file()


def _to_movie_recommendation(rec: dict) -> MovieRecommendation:
    movie_id = int(rec["movie_id"])
    return MovieRecommendation(
        movie_id=movie_id,
        title=str(rec["title"]),
        score=float(rec["score"]),
        genres=list(rec.get("genres", [])),
        release_year=rec.get("release_year"),
        has_poster=_has_poster(movie_id),
    )


def _require_catalog_predictor(predictor: BasePredictor) -> ModelPredictor:
    if not isinstance(predictor, ModelPredictor) or not predictor.catalog_loaded:
        detail = getattr(predictor, "catalog_error", None) or (
            "Model artifact and movie catalog are not loaded."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _default_predictor
    _default_predictor = _build_predictor_from_env()
    app.state.predictor = _default_predictor
    yield
    _default_predictor = None


app = FastAPI(
    title="Movie Recommender API",
    description="API serving movie recommendations for users (US6.2).",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


def get_predictor() -> BasePredictor:
    """Dependency injection provider for the recommendation model/predictor."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = _build_predictor_from_env()
    return _default_predictor


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check endpoint",
)
def health_check(predictor: BasePredictor = Depends(get_predictor)):
    using_fallback = bool(getattr(predictor, "using_fallback", False))
    return HealthResponse(
        status="degraded" if using_fallback else "healthy",
        model_version=getattr(predictor, "model_version", "unknown"),
        using_fallback=using_fallback,
        artifact_loaded=bool(getattr(predictor, "artifact_loaded", False)),
        catalog_loaded=bool(getattr(predictor, "catalog_loaded", False)),
        fallback_reason=getattr(predictor, "fallback_reason", None),
    )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get movie recommendations for a user",
    description="Accepts a user ID and top_k limit, returning structured movie recommendations.",
    status_code=status.HTTP_200_OK,
)
def recommend(
    request: RecommendationRequest,
    predictor: BasePredictor = Depends(get_predictor),
):
    try:
        if request.genres:
            try:
                recommendations_raw = predictor.predict(
                    user_id=request.user_id,
                    top_k=request.top_k,
                    genres=request.genres,
                )
            except TypeError:
                recommendations_raw = predictor.predict(
                    user_id=request.user_id,
                    top_k=request.top_k,
                )
        else:
            recommendations_raw = predictor.predict(
                user_id=request.user_id,
                top_k=request.top_k,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    recommendations = [_to_movie_recommendation(rec) for rec in recommendations_raw]
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_version=getattr(predictor, "model_version", "fallback"),
        using_fallback=bool(getattr(predictor, "using_fallback", False)),
    )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get movie recommendations via GET",
    description="Convenience GET endpoint to fetch recommendations by user_id path parameter and top_k query parameter.",
    status_code=status.HTTP_200_OK,
)
def recommend_get(
    user_id: int = FastPath(
        ..., gt=0, description="The user ID requesting recommendations (positive integer)"
    ),
    top_k: int = Query(10, ge=1, le=100, description="Number of recommendations to return (1-100)"),
    genres: list[str] | None = Query(default=None, description="Optional genre filter"),
    predictor: BasePredictor = Depends(get_predictor),
):
    request = RecommendationRequest(user_id=user_id, top_k=top_k, genres=genres)
    return recommend(request=request, predictor=predictor)


@app.get(
    "/catalog/genres",
    response_model=CatalogGenresResponse,
    tags=["Catalog"],
    summary="List available movie genres",
)
def catalog_genres(predictor: BasePredictor = Depends(get_predictor)):
    model_predictor = _require_catalog_predictor(predictor)
    return CatalogGenresResponse(
        genres=model_predictor.genres,
        model_version=model_predictor.model_version,
        catalog_loaded=True,
    )


@app.get(
    "/catalog/options",
    response_model=CatalogOptionsResponse,
    tags=["Catalog"],
    summary="List popular rateable movies for selected genres",
)
def catalog_options(
    genres: list[str] = Query(..., min_length=1, description="Selected genres"),
    per_genre: int = Query(25, ge=1, le=5000, description="Maximum movies per genre"),
    predictor: BasePredictor = Depends(get_predictor),
):
    model_predictor = _require_catalog_predictor(predictor)
    try:
        movies = model_predictor.rating_options(genres=genres, per_genre=per_genre)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return CatalogOptionsResponse(
        movies=[CatalogMovie(**movie) for movie in movies],
        model_version=model_predictor.model_version,
    )


@app.post(
    "/recommend/new-user",
    response_model=NewUserRecommendationResponse,
    tags=["Recommendations"],
    summary="Get personalized recommendations for a cold-start user",
)
def recommend_new_user(
    request: NewUserRecommendationRequest,
    predictor: BasePredictor = Depends(get_predictor),
):
    model_predictor = _require_catalog_predictor(predictor)
    try:
        recommendations_raw = model_predictor.recommend_new_user(
            genres=request.genres,
            ratings=request.ratings,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return NewUserRecommendationResponse(
        recommendations=[_to_movie_recommendation(rec) for rec in recommendations_raw],
        model_version=model_predictor.model_version,
        using_fallback=False,
    )


@app.get(
    "/posters/{movie_id}",
    tags=["Catalog"],
    summary="Fetch poster image for a movie",
)
def get_poster(
    movie_id: int = FastPath(..., gt=0, description="Movie ID"),
):
    poster_path = _posters_dir() / f"{movie_id}.jpg"
    if not poster_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Poster for movie {movie_id} not found.",
        )
    return FileResponse(poster_path, media_type="image/jpeg")
