import json
from unittest.mock import Mock

from src.models import train_from_b2


def test_latest_manifest_uses_highest_published_version():
    client = Mock()
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "processed/movielens-1m/v2/manifest.json"}]},
        {
            "Contents": [
                {"Key": "processed/movielens-1m/v11/ratings.csv"},
                {"Key": "processed/movielens-1m/v10/manifest.json"},
                {"Key": "processed/movielens-1m/v99/debug/manifest.json"},
            ]
        },
    ]

    assert train_from_b2.latest_manifest_key(client, "test") == (
        "processed/movielens-1m/v10/manifest.json"
    )


def test_trains_only_without_matching_finished_run(monkeypatch, tmp_path):
    env = {
        "B2_BUCKET": "test",
        "B2_ENDPOINT": "https://s3.example.test",
        "B2_REGION": "us-east-005",
        "B2_KEY_ID": "id",
        "B2_APPLICATION_KEY": "key",
        "MLFLOW_TRACKING_URI": "https://tracking.example.test/mlflow",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    ratings_key = "processed/movielens-1m/v1/ratings.csv"
    client = Mock()
    client.get_object.return_value = {
        "Body": Mock(
            read=Mock(
                return_value=json.dumps(
                    {"outputs": [{"key": ratings_key, "sha256": "a" * 64}]}
                ).encode()
            )
        )
    }
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "processed/movielens-1m/v1/manifest.json"}]}
    ]
    monkeypatch.setattr(train_from_b2, "make_client", Mock(return_value=client))
    tracking = Mock()
    tracking.search_runs.side_effect = [[], [Mock()]]
    monkeypatch.setattr(train_from_b2, "MlflowClient", Mock(return_value=tracking))
    trainer = Mock()
    monkeypatch.setattr(train_from_b2, "train_and_save", trainer)

    artifact = tmp_path / "model.joblib"
    train_from_b2.train_from_b2(artifact, "digest", "commit")
    train_from_b2.train_from_b2(artifact, "digest", "commit")

    trainer.assert_called_once()
    assert trainer.call_args.args == (f"s3://test/{ratings_key}", artifact)
    assert trainer.call_args.kwargs["expected_sha256"] == "a" * 64
    identity = trainer.call_args.kwargs["run_tags"]["training_input"]
    assert tracking.search_runs.call_args.args[1] == (
        f"tags.training_input = '{identity}' AND attributes.status = 'FINISHED'"
    )
