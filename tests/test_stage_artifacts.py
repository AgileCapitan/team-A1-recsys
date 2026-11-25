import csv
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import joblib
import pytest

from src.models.batch_recommend import train_and_save
from src.serving import stage_artifacts
from src.serving.stage_artifacts import upload_serving_artifacts, validate_and_write_manifest


def _write_fixture_serving_files(
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
        "3,40,4,Action Hero (1999)\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "movielens-v1.joblib"
    train_and_save(ratings_csv, model_path)

    catalog_path = tmp_path / catalog_filename
    rows = [
        ("10", "Toy Action (1995)", "Action|Adventure", "1995"),
        ("20", "Comedy Night (1998)", "Comedy", "1998"),
        ("30", "Space Quest (2001)", "Action|Sci-Fi", "2001"),
        ("40", "Action Hero (1999)", "Action", "1999"),
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


@pytest.mark.parametrize(
    "catalog_filename,include_year",
    [
        ("movie-catalog.csv", True),
        ("movies.csv", False),
    ],
)
def test_validate_and_write_manifest_succeeds(tmp_path, catalog_filename, include_year):
    model_path, catalog_path = _write_fixture_serving_files(
        tmp_path, catalog_filename=catalog_filename, include_year=include_year
    )
    manifest_path = tmp_path / "staged-version.json"

    manifest = validate_and_write_manifest(
        model_path,
        catalog_path,
        model_key="models/movielens-v1/movielens-v1.joblib",
        catalog_key=f"processed/movielens-1m/v1/{catalog_filename}",
        manifest_path=manifest_path,
        expected_version="movielens-v1",
        mlflow_run_id="run-123",
        git_sha="abc1234",
    )

    assert manifest_path.is_file()
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written == manifest
    assert written["model_version"] == "movielens-v1"
    assert written["model_size_bytes"] == model_path.stat().st_size
    assert written["catalog_size_bytes"] == catalog_path.stat().st_size
    assert written["mlflow_run_id"] == "run-123"
    assert written["git_sha"] == "abc1234"
    assert len(written["model_sha256"]) == 64
    assert len(written["catalog_sha256"]) == 64


def test_validate_and_write_manifest_rejects_version_mismatch_and_bad_files(tmp_path):
    model_path, catalog_path = _write_fixture_serving_files(tmp_path)
    manifest_path = tmp_path / "staged-version.json"

    with pytest.raises(ValueError, match="Model version mismatch"):
        validate_and_write_manifest(
            model_path,
            catalog_path,
            model_key="models/movielens-v1/movielens-v1.joblib",
            catalog_key="processed/movielens-1m/v1/movies.csv",
            manifest_path=manifest_path,
            expected_version="movielens-v99",
        )
    assert not manifest_path.exists()

    with pytest.raises(FileNotFoundError):
        validate_and_write_manifest(
            tmp_path / "missing.joblib",
            catalog_path,
            model_key="models/missing.joblib",
            catalog_key="processed/movielens-1m/v1/movies.csv",
            manifest_path=manifest_path,
        )
    assert not manifest_path.exists()

    corrupt_path = tmp_path / "corrupt.joblib"
    joblib.dump({"bad": "bundle"}, corrupt_path)
    with pytest.raises(ValueError, match="missing keys"):
        validate_and_write_manifest(
            corrupt_path,
            catalog_path,
            model_key="models/corrupt.joblib",
            catalog_key="processed/movielens-1m/v1/movies.csv",
            manifest_path=manifest_path,
        )
    assert not manifest_path.exists()


def test_upload_serving_artifacts_verifies_and_refuses_overwrite(tmp_path, s3):
    model_path, catalog_path = _write_fixture_serving_files(tmp_path)

    result = upload_serving_artifacts(
        model_path,
        "models/movielens-v1/movielens-v1.joblib",
        catalog_path=catalog_path,
        catalog_key="models/movielens-v1/movie-catalog.csv",
        client=s3,
        bucket="team-a1-recsys",
    )

    assert result["model_version"] == "movielens-v1"
    assert result["model"]["size_bytes"] == model_path.stat().st_size
    assert result["catalog"]["size_bytes"] == catalog_path.stat().st_size
    assert ("team-a1-recsys", "models/movielens-v1/movielens-v1.joblib") in s3.objects
    assert ("team-a1-recsys", "models/movielens-v1/movie-catalog.csv") in s3.objects

    with pytest.raises(FileExistsError):
        upload_serving_artifacts(
            model_path,
            "models/movielens-v1/movielens-v1.joblib",
            client=s3,
            bucket="team-a1-recsys",
        )


@pytest.mark.parametrize("deploy", [False, True])
def test_upload_command_dispatches_the_verified_keys_only_when_requested(monkeypatch, deploy):
    upload = Mock(
        return_value={
            "model_version": "movielens-v2",
            "model": {"key": "models/v2/verified.joblib"},
            "catalog": {"key": "models/v2/verified.csv"},
        }
    )
    run = Mock()
    monkeypatch.setattr(stage_artifacts, "upload_serving_artifacts", upload)
    monkeypatch.setattr(stage_artifacts.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(stage_artifacts.subprocess, "run", run)
    args = ["upload", "--model", "model.joblib", "--model-key", "models/v2/requested.joblib"]
    if deploy:
        args += ["--deploy", "--mlflow-run-id", "training-run"]

    stage_artifacts.main(args)

    upload.assert_called_once()
    if deploy:
        run.assert_called_once()
        assert run.call_args.args[0] == [
            "gh",
            "workflow",
            "run",
            "deploy.yml",
            "--repo",
            "AgileCapitan/team-A1-recsys",
            "--ref",
            "main",
            "--json",
        ]
        assert json.loads(run.call_args.kwargs["input"]) == {
            "model_key": "models/v2/verified.joblib",
            "catalog_key": "models/v2/verified.csv",
            "expected_version": "movielens-v2",
            "mlflow_run_id": "training-run",
        }
    else:
        run.assert_not_called()


def test_upload_command_does_not_deploy_after_upload_failure(monkeypatch):
    run = Mock()
    monkeypatch.setattr(
        stage_artifacts,
        "upload_serving_artifacts",
        Mock(side_effect=ValueError("checksum mismatch")),
    )
    monkeypatch.setattr(stage_artifacts.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(stage_artifacts.subprocess, "run", run)

    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_artifacts.main(
            ["upload", "--model", "model.joblib", "--model-key", "key", "--deploy"]
        )

    run.assert_not_called()


def test_upload_command_reports_published_key_after_dispatch_failure(monkeypatch, capsys):
    summary = {"model_version": "movielens-v2", "model": {"key": "models/v2/model.joblib"}}
    monkeypatch.setattr(stage_artifacts, "upload_serving_artifacts", Mock(return_value=summary))
    monkeypatch.setattr(stage_artifacts.shutil, "which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(
        stage_artifacts.subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, "gh"))
    )

    with pytest.raises(SystemExit) as result:
        stage_artifacts.main(
            ["upload", "--model", "model.joblib", "--model-key", "key", "--deploy"]
        )

    assert result.value.code == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == summary
    assert "B2 upload verified" in output.err
