from fastapi import Depends, FastAPI, Query, status
from fastapi import Path as FastPath

from api.schemas import (
    HealthResponse,
    MovieRecommendation,
    RecommendationRequest,
    RecommendationResponse,
)
from src.serving.predictor import BasePredictor, FallbackRecommender

app = FastAPI(
    title="Movie Recommender API",
    description="API serving movie recommendations for users (US6.2).",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Global or injectable predictor instance
_default_predictor = FallbackRecommender()


def get_predictor() -> BasePredictor:
    """Dependency injection provider for the recommendation model/predictor."""
    return _default_predictor


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check endpoint",
)
def health_check(predictor: BasePredictor = Depends(get_predictor)):
    return {
        "status": "healthy",
        "model_version": getattr(predictor, "model_version", "unknown"),
    }


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
    recommendations_raw = predictor.predict(user_id=request.user_id, top_k=request.top_k)
    recommendations = [
        MovieRecommendation(
            movie_id=rec["movie_id"],
            title=rec["title"],
            score=rec["score"],
        )
        for rec in recommendations_raw
    ]
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_version=getattr(predictor, "model_version", "fallback"),
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
    predictor: BasePredictor = Depends(get_predictor),
):
    request = RecommendationRequest(user_id=user_id, top_k=top_k)
    return recommend(request=request, predictor=predictor)
