import csv
import math
from unittest.mock import Mock, patch

import pytest

from src.models.batch_recommend import load_training_data, train_and_save, write_recommendations


def test_load_training_data_from_rustfs(monkeypatch):
    body = Mock()
    body.read.return_value = b"user_id,movie_id,rating,title\n1,10,5,Ten\n"
    client = Mock()
    client.get_object.return_value = {"Body": body}
    monkeypatch.setenv("RUSTFS_ACCESS_KEY", "access")
    monkeypatch.setenv("RUSTFS_SECRET_KEY", "secret")

    with patch("boto3.client", return_value=client) as create_client:
        ratings = load_training_data("s3://recsys/processed/movielens-1m/v1/ratings.csv")

    create_client.assert_called_once()
    client.get_object.assert_called_once_with(
        Bucket="recsys", Key="processed/movielens-1m/v1/ratings.csv"
    )
    assert ratings.to_dict("records") == [
        {"user_id": 1, "movie_id": 10, "rating": 5, "title": "Ten"}
    ]


def test_load_training_data_requires_rustfs_credentials(monkeypatch):
    monkeypatch.delenv("RUSTFS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("RUSTFS_SECRET_KEY", raising=False)

    with pytest.raises(ValueError, match="RUSTFS_ACCESS_KEY"):
        load_training_data("s3://recsys/ratings.csv")


def test_saved_model_generates_ranked_unseen_movies(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(
        "user_id,movie_id,rating,title\n"
        "11,10,5,Ten\n21,20,4,Twenty\n31,30,3,Thirty\n31,40,5,Forty\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "model.joblib"
    train_and_save(ratings, artifact)

    seen = {11: {10}, 21: {20}, 31: {30, 40}}
    for method in ("popularity", "mf"):
        output = tmp_path / f"{method}.csv"
        write_recommendations(artifact, output, method, n=2)
        with output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == ["user_id", "rank", "movie_id", "title", "score"]
            rows = list(reader)

        assert [int(row["user_id"]) for row in rows] == [11, 11, 21, 21, 31, 31]
        for user_id, rated_movies in seen.items():
            group = [row for row in rows if int(row["user_id"]) == user_id]
            movies = {int(row["movie_id"]) for row in group}
            scores = [float(row["score"]) for row in group]
            assert [int(row["rank"]) for row in group] == [1, 2]
            assert len(movies) == 2 and movies.isdisjoint(rated_movies)
            assert all(math.isfinite(score) for score in scores)
            assert scores == sorted(scores, reverse=True)

        if method == "popularity":
            assert [int(row["movie_id"]) for row in rows[:2]] == [20, 30]
