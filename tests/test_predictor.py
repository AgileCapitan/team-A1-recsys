import csv
import gzip
from pathlib import Path

import joblib
import pytest

from src.models.batch_recommend import train_and_save
from src.serving.new_user import NewUserRecommender, build_movie_catalog
from src.serving.predictor import FallbackRecommender, ModelPredictor, create_predictor


def _write_fixture_artifacts(
    tmp_path: Path, *, catalog_filename: str = "movie-catalog.csv", include_year: bool = True
) -> tuple[Path, Path]:
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

    catalog_path = tmp_path / catalog_filename
    rows = [
        ("10", "Toy Action (1995)", "Action|Adventure", "1995"),
        ("20", "Comedy Night (1998)", "Comedy", "1998"),
        ("30", "Space Quest (2001)", "Action|Sci-Fi", "2001"),
        ("40", "Action Hero (1999)", "Action", "1999"),
        ("50", "Funny Sci-Fi (2002)", "Comedy|Sci-Fi", "2002"),
        ("60", "Drama Story (1994)", "Drama", "1994"),
    ]
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if include_year:
            writer.writerow(["movie_id", "title", "genres", "release_year"])
            writer.writerows(rows)
        else:
            writer.writerow(["movie_id", "title", "genres"])
            writer.writerows([(mid, title, genres) for mid, title, genres, _ in rows])

    return model_path, catalog_path


def test_model_predictor_loads_and_excludes_seen_movies(tmp_path):
    model_path, catalog_path = _write_fixture_artifacts(tmp_path)
    predictor = ModelPredictor(model_path=model_path, catalog_path=catalog_path)

    assert predictor.model_version == "movielens-v1"
    assert predictor.artifact_loaded is True
    assert predictor.catalog_loaded is True
    assert predictor.using_fallback is False
    assert predictor.genres == ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi"]

    known_recs = predictor.predict(user_id=1, top_k=4)
    assert len(known_recs) == 4
    known_ids = {rec["movie_id"] for rec in known_recs}
    assert known_ids.isdisjoint({10, 20})
    assert all("genres" in rec and "release_year" in rec for rec in known_recs)
    scores = [rec["score"] for rec in known_recs]
    assert scores == sorted(scores, reverse=True)

    # Unknown user / empty preferences falls back to popularity ranking from bundle.
    unknown_recs = predictor.predict(user_id=9999, top_k=3)
    assert len(unknown_recs) == 3
    assert [rec["movie_id"] for rec in unknown_recs] == [10, 20, 30]


def test_model_predictor_genre_filter_and_b2_movies_csv_year_parsing(tmp_path):
    model_path, catalog_path = _write_fixture_artifacts(
        tmp_path, catalog_filename="movies.csv", include_year=False
    )
    predictor = ModelPredictor(model_path=model_path, catalog_path=catalog_path)

    assert predictor.catalog_loaded is True
    filtered = predictor.predict(user_id=1, top_k=5, genres=["Action"])
    assert filtered
    assert all("Action" in rec["genres"] for rec in filtered)
    assert all(rec["movie_id"] not in {10, 20} for rec in filtered)
    assert all(isinstance(rec["release_year"], int) for rec in filtered)

    with pytest.raises(ValueError, match="available genre"):
        predictor.predict(user_id=1, top_k=5, genres=["NonExistentGenre"])


def test_model_predictor_new_user_scoring_and_validation(tmp_path):
    model_path, catalog_path = _write_fixture_artifacts(tmp_path)
    predictor = ModelPredictor(model_path=model_path, catalog_path=catalog_path)

    options = predictor.rating_options(genres=["Action"], per_genre=10)
    assert options
    assert all("Action" in item["genres"] for item in options)

    rated = {10: 5, 30: 4, 40: 2}
    # Add Drama so unrated Action/Drama movies exist; test unseen exclusion & genre match.
    recs = predictor.recommend_new_user(
        genres=["Action", "Comedy"],
        ratings=rated,
        top_k=5,
    )
    assert recs
    assert all(rec["movie_id"] not in rated for rec in recs)
    assert all(set(rec["genres"]) & {"Action", "Comedy"} for rec in recs)

    # Empty / insufficient ratings raise ValueError explicitly.
    with pytest.raises(ValueError, match="3 and 10"):
        predictor.recommend_new_user(genres=["Action"], ratings={}, top_k=5)

    with pytest.raises(ValueError, match="available genre"):
        predictor.recommend_new_user(genres=[], ratings=rated, top_k=5)

    with pytest.raises(ValueError, match="selected genres"):
        predictor.recommend_new_user(genres=["Drama"], ratings={10: 5, 30: 4, 40: 5}, top_k=5)


def test_model_predictor_without_catalog_still_serves_user_predictions(tmp_path):
    model_path, _ = _write_fixture_artifacts(tmp_path)
    predictor = ModelPredictor(model_path=model_path, catalog_path=tmp_path / "missing.csv")

    assert predictor.artifact_loaded is True
    assert predictor.catalog_loaded is False
    assert "missing.csv" in (predictor.catalog_error or "")

    recs = predictor.predict(user_id=1, top_k=3)
    assert len(recs) == 3
    assert all(rec["movie_id"] not in {10, 20} for rec in recs)

    with pytest.raises(RuntimeError):
        _ = predictor.genres

    with pytest.raises(RuntimeError):
        predictor.rating_options(genres=["Action"])

    with pytest.raises(RuntimeError):
        predictor.recommend_new_user(genres=["Action"], ratings={10: 5, 30: 4, 40: 3})

    with pytest.raises(RuntimeError):
        predictor.predict(user_id=1, top_k=3, genres=["Action"])


def test_create_predictor_fallback_and_fail_fast(tmp_path):
    missing_model = tmp_path / "missing.joblib"
    fallback = create_predictor(model_path=missing_model, allow_fallback=True)
    assert isinstance(fallback, FallbackRecommender)
    assert fallback.using_fallback is True
    assert fallback.artifact_loaded is False
    assert fallback.fallback_reason is not None
    assert len(fallback.predict(user_id=1, top_k=3)) == 3

    with pytest.raises(FileNotFoundError):
        create_predictor(model_path=missing_model, allow_fallback=False)

    corrupt_model = tmp_path / "corrupt.joblib"
    joblib.dump({"not": "a bundle"}, corrupt_model)
    corrupt_fallback = create_predictor(model_path=corrupt_model, allow_fallback=True)
    assert isinstance(corrupt_fallback, FallbackRecommender)
    assert "ValueError" in (corrupt_fallback.fallback_reason or "")

    with pytest.raises(ValueError):
        create_predictor(model_path=corrupt_model, allow_fallback=False)


def test_create_predictor_discovers_movies_csv_when_catalog_missing(tmp_path):
    model_path, _ = _write_fixture_artifacts(
        tmp_path, catalog_filename="movies.csv", include_year=False
    )
    predictor = create_predictor(
        model_path=model_path,
        catalog_path=tmp_path / "movie-catalog.csv",
        allow_fallback=False,
    )
    assert isinstance(predictor, ModelPredictor)
    assert predictor.catalog_loaded is True


def test_build_movie_catalog_from_gzip_and_new_user_from_path(tmp_path):
    source_gz = tmp_path / "compressed_data.csv.gz"
    with gzip.open(source_gz, "wt", encoding="utf-8", newline="") as handle:
        handle.write(
            "user_id,movie_id,rating,title,genres,release_year\n"
            "1,10,5,Toy Action (1995),Action|Adventure,1995\n"
            "2,10,4,Toy Action (1995),Action|Adventure,1995\n"
            "1,20,4,Comedy Night (1998),Comedy,\n"
        )

    output_csv = tmp_path / "movie-catalog.csv"
    count = build_movie_catalog(source_gz, output_csv)
    assert count == 2

    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "movie_id": "10",
            "title": "Toy Action (1995)",
            "genres": "Action|Adventure",
            "release_year": "1995",
        },
        {
            "movie_id": "20",
            "title": "Comedy Night (1998)",
            "genres": "Comedy",
            "release_year": "1998",
        },
    ]

    model_path, catalog_path = _write_fixture_artifacts(tmp_path)
    standalone = NewUserRecommender(model_path=model_path, catalog_path=catalog_path)
    assert standalone.model_version == "movielens-v1"
