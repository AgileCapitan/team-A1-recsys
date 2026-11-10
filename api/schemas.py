from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(
        ...,
        gt=0,
        description="The ID of the user requesting recommendations (must be a positive integer).",
        examples=[42],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return (between 1 and 100).",
        examples=[5],
    )


class MovieRecommendation(BaseModel):
    movie_id: int = Field(..., description="Unique identifier of the recommended movie.")
    title: str = Field(..., description="Title of the movie.")
    score: float = Field(..., description="Predicted relevance or rating score.")


class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="The user ID for whom recommendations were generated.")
    recommendations: list[MovieRecommendation] = Field(
        ..., description="List of recommended movies ordered by score descending."
    )
    model_version: str = Field(..., description="Identifier/version of the serving model.")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_version: str = Field(..., examples=["fallback-v0.1"])
