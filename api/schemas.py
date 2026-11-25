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
    genres: list[str] | None = Field(
        default=None,
        description="Optional genre filter applied when catalog metadata is loaded.",
        examples=[["Action", "Sci-Fi"]],
    )


class MovieRecommendation(BaseModel):
    movie_id: int = Field(..., description="Unique identifier of the recommended movie.")
    title: str = Field(..., description="Title of the movie.")
    score: float = Field(..., description="Predicted relevance or rating score.")
    genres: list[str] = Field(default_factory=list, description="Movie genres.")
    release_year: int | None = Field(default=None, description="Movie release year.")
    has_poster: bool = Field(default=False, description="Whether a poster image is available.")


class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="The user ID for whom recommendations were generated.")
    recommendations: list[MovieRecommendation] = Field(
        ..., description="List of recommended movies ordered by score descending."
    )
    model_version: str = Field(..., description="Identifier/version of the serving model.")
    using_fallback: bool = Field(
        default=False,
        description="True when serving fallback recommendations instead of a trained model.",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_version: str = Field(..., examples=["movielens-v1"])
    using_fallback: bool = Field(default=False)
    artifact_loaded: bool = Field(default=False)
    catalog_loaded: bool = Field(default=False)
    fallback_reason: str | None = Field(default=None)


class CatalogGenresResponse(BaseModel):
    genres: list[str] = Field(..., description="Available catalog genres.")
    model_version: str = Field(..., description="Identifier/version of the serving model.")
    catalog_loaded: bool = Field(default=True)


class CatalogMovie(BaseModel):
    movie_id: int = Field(..., description="Unique identifier of the movie.")
    title: str = Field(..., description="Title of the movie.")
    genres: list[str] = Field(default_factory=list, description="Movie genres.")
    release_year: int | None = Field(default=None, description="Movie release year.")


class CatalogOptionsResponse(BaseModel):
    movies: list[CatalogMovie] = Field(
        ..., description="Popular movies matching the selected genres for onboarding ratings."
    )
    model_version: str = Field(..., description="Identifier/version of the serving model.")


class NewUserRecommendationRequest(BaseModel):
    genres: list[str] = Field(
        ...,
        min_length=1,
        description="Favorite genres selected by the new user.",
        examples=[["Action", "Sci-Fi"]],
    )
    ratings: dict[int, int] = Field(
        ...,
        description="Mapping from movie_id to integer rating (1-5), 3 to 10 movies.",
        examples=[{1: 5, 260: 4, 2571: 5}],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return (between 1 and 20).",
        examples=[5],
    )


class NewUserRecommendationResponse(BaseModel):
    recommendations: list[MovieRecommendation] = Field(
        ..., description="List of recommended movies ordered by score descending."
    )
    model_version: str = Field(..., description="Identifier/version of the serving model.")
    using_fallback: bool = Field(default=False)
