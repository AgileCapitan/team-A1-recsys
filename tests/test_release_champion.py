"""Release integration tests use only tiny bundles and in-memory external services."""

import copy
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion, ModelVersionTag

from src.data.storage import sha256_hex
from src.models.recommender import MatrixFactorization, rating_history
from src.serving import release_champion as release
from src.serving import stage_artifacts


@pytest.fixture
def champion(tmp_path, s3, monkeypatch):
    bucket = "fixture-bucket"
    run_id = "f" * 32
    evaluation_id = "e" * 32
    name = "fixture-recommender"
    model_artifact = "model/movielens-v1.joblib"
    ratings = pd.DataFrame(
        [
            (1, 10, 5, "Ten (2000)"),
            (1, 20, 4, "Twenty (2001)"),
            (2, 10, 4, "Ten (2000)"),
            (2, 30, 5, "Thirty (2002)"),
            (3, 20, 5, "Twenty (2001)"),
            (3, 40, 4, "Forty (2003)"),
        ],
        columns=["user_id", "movie_id", "rating", "title"],
    )
    bundle = {
        "model_version": "movielens-v1",
        "model": MatrixFactorization().fit(ratings, factors=2, epochs=1),
        "history": rating_history(ratings),
        "popularity": ratings.groupby("movie_id").size(),
        "titles": ratings.drop_duplicates("movie_id").set_index("movie_id")["title"].to_dict(),
    }
    model_path = tmp_path / "fixture.joblib"
    joblib.dump(bundle, model_path, compress=3)
    prefix = "processed/movielens-1m/v7/"
    catalog_key = prefix + "movies.csv"
    catalog = b"movie_id,title,genres\n10,Ten (2000),Action\n20,Twenty (2001),Comedy\n30,Thirty (2002),Action\n40,Forty (2003),Comedy\n"
    ratings_bytes = ratings.to_csv(index=False).encode()
    manifest = release.encoded(
        {
            "outputs": [
                {
                    "key": prefix + "ratings.csv",
                    "sha256": sha256_hex(ratings_bytes),
                    "size_bytes": len(ratings_bytes),
                },
                {"key": catalog_key, "sha256": sha256_hex(catalog), "size_bytes": len(catalog)},
            ]
        }
    )
    s3.objects[(bucket, catalog_key)] = catalog
    s3.objects[(bucket, prefix + "manifest.json")] = manifest
    tags = {
        "dataset_uri": f"s3://{bucket}/{prefix}ratings.csv",
        "dataset_sha256": sha256_hex(ratings_bytes),
        "dataset_manifest_sha256": sha256_hex(manifest),
        "model_version": "movielens-v1",
    }
    source = SimpleNamespace(
        info=SimpleNamespace(
            status="FINISHED", artifact_uri="mlflow-artifacts:/1/fixture/artifacts"
        ),
        data=SimpleNamespace(tags=tags),
    )
    evaluation = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(metrics={"fixture_metric": 0.5}),
    )
    decision = {
        "schema_version": 1,
        "decision": "promoted",
        "registered_model": name,
        "registry_version": "8",
        "source_run_id": run_id,
        "model_artifact": model_artifact,
        "model_sha256": sha256_hex(model_path.read_bytes()),
        "bundle_version": "movielens-v1",
        "evaluation": {
            "primary_metric": "fixture_metric",
            "k": 3,
            "retention_rule": "fixture rule; not a project decision",
            "heldout_sha256": "a" * 64,
            "candidate_run_ids": [run_id],
        },
    }
    version = ModelVersion(
        name=name,
        version="8",
        creation_timestamp=1,
        status="READY",
        source=f"runs:/{run_id}/{model_artifact}",
        run_id=run_id,
        tags=[ModelVersionTag(release.EVALUATION_TAG, evaluation_id)],
    )
    tracking = Mock(spec_set=MlflowClient)
    tracking.get_model_version_by_alias.return_value = version
    tracking.get_run.side_effect = lambda rid: source if rid == run_id else evaluation

    def download_artifact(rid, path, dst):
        if rid == evaluation_id and path == release.DECISION_PATH:
            decision_path = tmp_path / "decision.json"
            decision_path.write_bytes(release.encoded(decision))
            return str(decision_path)
        assert (rid, path) == (run_id, model_artifact)
        return str(model_path)

    tracking.download_artifacts.side_effect = download_artifact
    deployments = Mock()
    deployments.find.return_value = None
    deployments.wait.return_value = {"id": 123, "html_url": "https://example.test/runs/123"}
    dispatch = Mock()
    monkeypatch.setattr(stage_artifacts, "_dispatch_deployment", dispatch)
    monkeypatch.setattr(stage_artifacts.shutil, "which", lambda _: "/fixture/gh")
    monkeypatch.setattr(
        stage_artifacts,
        "StorageSettings",
        SimpleNamespace(from_env=lambda **_: SimpleNamespace(bucket=bucket)),
    )
    monkeypatch.setattr(stage_artifacts, "make_client", lambda _: s3)

    def invoke():
        return release.release_champion(
            tracking, s3, bucket, name, tmp_path, deployments=deployments
        )

    return SimpleNamespace(**locals())


def test_releases_exact_bundle_and_catalog_then_skips_unchanged(champion):
    c = champion
    result = c.invoke()
    assert result["status"] == "dispatched"
    assert result["registry_version"] == "8"
    assert result["mlflow_run_id"] == c.run_id
    assert result["model_version"] == "movielens-v1"
    assert f"/v8/{c.run_id}/" in result["model_key"]
    assert c.s3.objects[(c.bucket, result["model_key"])] == c.model_path.read_bytes()
    assert result["catalog_key"] == c.catalog_key
    assert result["catalog_sha256"] == sha256_hex(c.catalog)
    assert c.catalog_key not in c.s3.put_order
    assert result["deployment_run"]["id"] == 123
    summary, run_id = c.dispatch.call_args.args
    assert run_id == c.run_id
    assert summary["model"]["key"] == result["model_key"]
    assert summary["catalog"]["key"] == c.catalog_key
    assert summary["model"]["sha256"] == c.decision["model_sha256"]
    puts = c.s3.put_order.copy()
    downloads = c.tracking.download_artifacts.call_count
    assert c.invoke()["status"] == "unchanged"
    assert c.s3.put_order == puts
    assert c.tracking.download_artifacts.call_count == downloads + 1  # decision only
    c.dispatch.assert_called_once()
    c.tracking.set_registered_model_alias.assert_not_called()
    c.tracking.set_model_version_tag.assert_not_called()


def test_retaining_champion_skips_all_publication(champion):
    c = champion
    c.decision["decision"] = "retained"
    assert c.invoke()["status"] == "retained"
    assert c.s3.put_order == []
    c.dispatch.assert_not_called()


@pytest.mark.parametrize(
    "invalid",
    [
        "training_only",
        "same_run",
        "unfinished_evaluation",
        "unfinished_source",
        "missing_metric",
        "wrong_run",
        "wrong_version",
        "wrong_source",
        "wrong_bundle_version",
        "unevaluated",
        "bad_k",
        "no_rule",
        "model_checksum",
        "manifest_checksum",
        "catalog_checksum",
    ],
)
def test_invalid_provenance_never_publishes_or_dispatches(champion, invalid):
    c = champion
    if invalid == "training_only":
        c.version.tags.clear()
    elif invalid == "same_run":
        c.version.tags[release.EVALUATION_TAG] = c.run_id
    elif invalid == "unfinished_evaluation":
        c.evaluation.info.status = "RUNNING"
    elif invalid == "unfinished_source":
        c.source.info.status = "RUNNING"
    elif invalid == "missing_metric":
        c.evaluation.data.metrics.clear()
    elif invalid == "wrong_run":
        c.decision["source_run_id"] = "0" * 32
    elif invalid == "wrong_version":
        c.decision["registry_version"] = "9"
    elif invalid == "wrong_source":
        c.version._source = "runs:/other/model.joblib"
    elif invalid == "wrong_bundle_version":
        c.decision["bundle_version"] = "other"
    elif invalid == "unevaluated":
        c.decision["evaluation"]["candidate_run_ids"] = []
    elif invalid == "bad_k":
        c.decision["evaluation"]["k"] = 0
    elif invalid == "no_rule":
        c.decision["evaluation"]["retention_rule"] = ""
    elif invalid == "model_checksum":
        c.decision["model_sha256"] = "0" * 64
    elif invalid == "manifest_checksum":
        c.tags["dataset_manifest_sha256"] = "0" * 64
    elif invalid == "catalog_checksum":
        c.s3.objects[(c.bucket, c.catalog_key)] = b"corrupt"
    with pytest.raises(ValueError):
        c.invoke()
    assert c.s3.put_order == []
    c.dispatch.assert_not_called()


def test_upload_success_dispatch_failure_resumes_without_overwrite(champion):
    c = champion
    c.dispatch.side_effect = subprocess.CalledProcessError(1, "gh")
    with pytest.raises(SystemExit):
        c.invoke()
    original_puts = c.s3.put_order.copy()
    assert len(original_puts) == 2  # release plan and model
    c.dispatch.side_effect = None
    result = c.invoke()
    assert result["status"] == "dispatched"
    assert c.s3.put_order[:-1] == original_puts
    assert c.s3.put_order[-1].endswith("deployment.json")


def test_corrupted_uploaded_model_prevents_retry_dispatch(champion):
    c = champion
    c.dispatch.side_effect = subprocess.CalledProcessError(1, "gh")
    with pytest.raises(SystemExit):
        c.invoke()
    model_key = next(key for key in c.s3.put_order if key.endswith("model.joblib"))
    c.s3.objects[(c.bucket, model_key)] = b"corrupt"
    c.dispatch.reset_mock(side_effect=True)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        c.invoke()
    c.dispatch.assert_not_called()


def test_dispatch_accepted_but_receipt_missing_is_reconciled(champion):
    c = champion
    c.deployments.wait.side_effect = RuntimeError("not visible yet")
    with pytest.raises(RuntimeError, match="not visible"):
        c.invoke()
    c.deployments.find.return_value = {"id": 321, "html_url": "https://example.test/runs/321"}
    result = c.invoke()
    assert result["deployment_run"]["id"] == 321
    c.dispatch.assert_called_once()


def test_changed_alias_during_download_prevents_handoff(champion):
    c = champion
    newer = copy.copy(c.version)
    newer._version = "9"
    c.tracking.get_model_version_by_alias.side_effect = [c.version, newer]
    with pytest.raises(ValueError, match="Champion changed"):
        c.invoke()
    c.dispatch.assert_not_called()
    assert not any(key.endswith("model.joblib") for key in c.s3.put_order)


def test_mutated_decision_after_release_is_rejected(champion):
    c = champion
    c.invoke()
    c.decision["evaluation"]["retention_rule"] = "changed"
    with pytest.raises(ValueError, match="provenance changed"):
        c.invoke()
    c.dispatch.assert_called_once()


def test_deployment_lookup_parses_paginated_json_and_correlates_exact_title(monkeypatch):
    runs = [
        {"id": 1, "display_title": "Oracle staging abc", "html_url": "url1"},
        {"id": 2, "display_title": "Oracle staging abcd", "html_url": "url2"},
        {"id": 3, "display_title": "Oracle staging abc", "html_url": "url3"},
    ]
    run = Mock(
        return_value=SimpleNamespace(
            stdout=json.dumps([{"workflow_runs": runs[:1]}, {"workflow_runs": runs[1:]}], indent=2)
        )
    )
    monkeypatch.setattr(release.subprocess, "run", run)
    assert release.Deployments().find("abc")["id"] == 3
    assert "--paginate" in run.call_args.args[0]
    assert "--slurp" in run.call_args.args[0]
    assert release.Deployments().find("absent") is None


def test_real_local_mlflow_registry_and_artifact_contract(champion, tmp_path):
    """Exercise MLflow's real alias/version/download APIs against a temporary SQLite store."""
    c = champion
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.create_experiment(
        "release-fixture", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    source = client.create_run(experiment, tags=c.tags)
    exact_model = tmp_path / "movielens-v1.joblib"
    exact_model.write_bytes(c.model_path.read_bytes())
    client.log_artifact(source.info.run_id, str(exact_model), "model")
    client.set_terminated(source.info.run_id)
    client.create_registered_model(c.name)
    version = client.create_model_version(
        c.name, source=f"{source.info.artifact_uri}/{c.model_artifact}", run_id=source.info.run_id
    )
    evaluation = client.create_run(experiment)
    decision = copy.deepcopy(c.decision)
    decision["source_run_id"] = source.info.run_id
    decision["registry_version"] = str(version.version)
    decision["evaluation"]["candidate_run_ids"] = [source.info.run_id]
    decision_file = tmp_path / "decision.json"
    decision_file.write_bytes(release.encoded(decision))
    client.log_artifact(evaluation.info.run_id, str(decision_file), "release")
    client.log_metric(evaluation.info.run_id, "fixture_metric", 0.5)
    client.set_terminated(evaluation.info.run_id)
    client.set_model_version_tag(
        c.name, version.version, release.EVALUATION_TAG, evaluation.info.run_id
    )
    client.set_registered_model_alias(c.name, "champion", version.version)
    work = tmp_path / "release-work"
    work.mkdir()
    result = release.release_champion(
        client, c.s3, c.bucket, c.name, work, deployments=c.deployments
    )
    assert result["registry_version"] == "1"
    assert result["mlflow_run_id"] == source.info.run_id
    assert result["evaluation_run_id"] == evaluation.info.run_id
    assert c.s3.objects[(c.bucket, result["model_key"])] == exact_model.read_bytes()


def test_fixture_bundle_serves_known_user_with_expected_version(champion):
    from fastapi.testclient import TestClient

    from api.main import app, get_predictor
    from src.serving.predictor import ModelPredictor

    c = champion
    c.invoke()
    predictor = ModelPredictor(c.model_path, c.tmp_path / "movies.csv")
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        with TestClient(app) as client:
            health = client.get("/health").json()
            assert health["model_version"] == "movielens-v1"
            assert health["using_fallback"] is False
            assert health["artifact_loaded"] is True
            response = client.get("/recommend/1?top_k=3")
            assert response.status_code == 200
            result = response.json()
            assert result["user_id"] == 1
            assert result["model_version"] == "movielens-v1"
            assert result["using_fallback"] is False
            assert result["recommendations"]
            assert {r["movie_id"] for r in result["recommendations"]}.isdisjoint({10, 20})
    finally:
        app.dependency_overrides.pop(get_predictor, None)


def test_new_registry_version_releases_even_with_same_bundle_version(champion):
    c = champion
    first = c.invoke()
    c.version._version = "9"
    c.decision["registry_version"] = "9"
    c.deployments.wait.return_value = {"id": 124, "html_url": "https://example.test/runs/124"}
    second = c.invoke()
    assert first["model_version"] == second["model_version"]
    assert first["model_key"] != second["model_key"]
    assert first["release_id"] != second["release_id"]
    assert second["deployment_run"]["id"] == 124
    assert c.dispatch.call_count == 2


def test_partial_model_and_catalog_upload_resumes_without_overwriting(champion, monkeypatch):
    c = champion
    c.invoke()  # materialize the tiny matching catalog locally
    model_key = "models/fixture/partial.joblib"
    catalog_key = "models/fixture/partial.csv"
    original_put = c.s3.put_object

    def fail_catalog(**kwargs):
        if kwargs["Key"] == catalog_key:
            raise RuntimeError("catalog upload interrupted")
        return original_put(**kwargs)

    monkeypatch.setattr(c.s3, "put_object", fail_catalog)
    with pytest.raises(RuntimeError, match="catalog upload interrupted"):
        stage_artifacts.upload_serving_artifacts(
            c.model_path,
            model_key,
            catalog_path=c.tmp_path / "movies.csv",
            catalog_key=catalog_key,
            client=c.s3,
            bucket=c.bucket,
            reuse_verified=True,
        )
    monkeypatch.setattr(c.s3, "put_object", original_put)
    result = stage_artifacts.upload_serving_artifacts(
        c.model_path,
        model_key,
        catalog_path=c.tmp_path / "movies.csv",
        catalog_key=catalog_key,
        client=c.s3,
        bucket=c.bucket,
        reuse_verified=True,
    )
    assert c.s3.put_order.count(model_key) == 1
    assert c.s3.put_order.count(catalog_key) == 1
    assert result["catalog"]["sha256"] == sha256_hex(c.catalog)


def test_failed_upload_readback_prevents_dispatch_and_can_retry(champion, monkeypatch):
    import io

    c = champion
    original_get = c.s3.get_object

    def corrupt_readback(**kwargs):
        if kwargs["Key"].endswith("model.joblib"):
            return {"Body": io.BytesIO(b"bad readback")}
        return original_get(**kwargs)

    monkeypatch.setattr(c.s3, "get_object", corrupt_readback)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        c.invoke()
    c.dispatch.assert_not_called()
    monkeypatch.setattr(c.s3, "get_object", original_get)
    assert c.invoke()["status"] == "dispatched"
    assert sum(key.endswith("model.joblib") for key in c.s3.put_order) == 1
    c.dispatch.assert_called_once()


def test_mutated_source_dataset_after_release_is_rejected(champion):
    c = champion
    c.invoke()
    c.tags["dataset_uri"] = "s3://fixture-bucket/processed/other/ratings.csv"
    with pytest.raises(ValueError, match="provenance changed"):
        c.invoke()
    c.dispatch.assert_called_once()
