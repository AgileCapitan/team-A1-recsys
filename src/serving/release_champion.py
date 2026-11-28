"""Release an evaluated registry champion; never select or promote one."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from mlflow import MlflowClient

from src.data.storage import (
    StorageSettings,
    download,
    head,
    make_client,
    sha256_hex,
    upload_verified,
    verify_object,
)
from src.serving import stage_artifacts
from src.serving.predictor import ModelPredictor

REPOSITORY = "AgileCapitan/team-A1-recsys"
DECISION_PATH = "release/decision.json"
EVALUATION_TAG = "release.evaluation_run_id"


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "Invalid SHA-256")
    return value


def artifact_path(value):
    require(isinstance(value, str) and value and "\\" not in value, "Invalid artifact path")
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and ".." not in path.parts and str(path) == value,
        "Artifact path must be relative and canonical",
    )
    return value


class Deployments:
    """Find the accepted dispatch even if the publisher died before saving its receipt."""

    def find(self, release_id):
        result = subprocess.run(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{REPOSITORY}/actions/workflows/deploy.yml/runs?event=workflow_dispatch&branch=main&per_page=100",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        pages = json.loads(result.stdout)
        matches = [
            {key: run[key] for key in ("id", "display_title", "html_url")}
            for page in pages
            for run in page["workflow_runs"]
            if run["display_title"] == f"Oracle staging {release_id}"
        ]
        return max(matches, key=lambda run: run["id"]) if matches else None

    def wait(self, release_id):
        for _ in range(6):
            if run := self.find(release_id):
                return run
            time.sleep(2)
        raise RuntimeError(
            "Dispatch not visible yet; retry to reconcile it before sending another."
        )


def resolve_champion(tracking, name, work):
    version = tracking.get_model_version_by_alias(name, "champion")
    require(
        version.name == name and re.fullmatch(r"[1-9][0-9]*", str(version.version)),
        "Invalid registry identity",
    )
    require(re.fullmatch(r"[a-f0-9]{32}", version.run_id or ""), "Invalid source run ID")
    require(version.status == "READY", "Champion registry version is not READY")
    source = tracking.get_run(version.run_id)
    require(source.info.status == "FINISHED", "Champion source run is not FINISHED")
    evaluation_id = version.tags.get(EVALUATION_TAG)
    require(
        evaluation_id and evaluation_id != version.run_id,
        "Champion needs a separate evaluation run; training alone cannot release",
    )
    evaluation = tracking.get_run(evaluation_id)
    require(evaluation.info.status == "FINISHED", "Evaluation run is not FINISHED")
    decision_file = tracking.download_artifacts(evaluation_id, DECISION_PATH, str(work))
    decision = json.loads(Path(decision_file).read_bytes())
    require(decision.get("schema_version") == 1, "Unsupported evaluation decision schema")
    require(
        decision.get("registered_model") == name
        and str(decision.get("registry_version")) == str(version.version)
        and decision.get("source_run_id") == version.run_id,
        "Evaluation decision does not identify the resolved champion",
    )
    require(decision.get("decision") in ("promoted", "retained"), "Missing promotion decision")
    if decision["decision"] == "retained":
        return version, source, decision, evaluation_id

    evidence = decision["evaluation"]
    metric = evidence["primary_metric"]
    score = evaluation.data.metrics.get(metric)
    require(
        isinstance(score, (float, int)) and math.isfinite(score),
        "Evaluation run lacks the primary metric",
    )
    require(type(evidence["k"]) is int and evidence["k"] > 0, "Evaluation K must be positive")
    require(
        isinstance(evidence["retention_rule"], str) and evidence["retention_rule"].strip(),
        "Evaluation retention rule is required",
    )
    digest(evidence["heldout_sha256"])
    require(version.run_id in evidence["candidate_run_ids"], "Source run was not evaluated")
    path = artifact_path(decision["model_artifact"])
    require(path.endswith(".joblib"), "Registry must identify the exact serving .joblib")
    require(
        version.source
        in (f"runs:/{version.run_id}/{path}", f"{source.info.artifact_uri.rstrip('/')}/{path}"),
        "Registry source does not identify the evaluated serving artifact",
    )
    require(
        decision["bundle_version"] == source.data.tags.get("model_version"),
        "Bundle version does not match the source run",
    )
    digest(decision["model_sha256"])
    return version, source, decision, evaluation_id


def matching_catalog(storage, bucket, source, work):
    """Derive the catalog from the exact dataset manifest used by the source run."""
    tags = source.data.tags
    uri = urlparse(tags["dataset_uri"])
    key = uri.path.lstrip("/")
    require(
        uri.scheme == "s3"
        and uri.netloc == bucket
        and key.endswith("/ratings.csv")
        and not uri.query
        and not uri.fragment,
        "Source dataset must be in the project bucket",
    )
    prefix = key.removesuffix("ratings.csv")
    manifest_key = prefix + "manifest.json"
    content = download(storage, bucket, manifest_key)
    require(
        sha256_hex(content) == digest(tags["dataset_manifest_sha256"]),
        "Source dataset manifest SHA-256 mismatch",
    )
    outputs = {entry["key"]: entry for entry in json.loads(content)["outputs"]}
    require(
        outputs[key]["sha256"] == digest(tags["dataset_sha256"]),
        "Source ratings checksum differs from manifest",
    )
    catalog_key = prefix + "movies.csv"
    catalog = outputs[catalog_key]
    verify_object(storage, bucket, catalog_key, digest(catalog["sha256"]), catalog["size_bytes"])
    path = work / "movies.csv"
    path.write_bytes(download(storage, bucket, catalog_key))
    require(sha256_hex(path.read_bytes()) == catalog["sha256"], "Catalog changed during download")
    return path, catalog_key, manifest_key


def save_immutable(storage, bucket, key, value):
    content = encoded(value)
    try:
        return upload_verified(storage, bucket, key, content, "application/json")
    except FileExistsError:
        return verify_object(storage, bucket, key, sha256_hex(content), len(content))


def release_champion(tracking, storage, bucket, name, work, *, deployments=None):
    """Resolve once, validate provenance, resume verified uploads, and dispatch existing staging."""
    deployments = deployments or Deployments()
    version, source, decision, evaluation_id = resolve_champion(tracking, name, work)
    if decision["decision"] == "retained":
        return {"status": "retained", "registry_version": str(version.version)}

    identity = {
        "registered_model": name,
        "registry_version": str(version.version),
        "source_run_id": version.run_id,
    }
    release_id = sha256_hex(encoded(identity))
    prefix = f"models/champions/{release_id}/v{version.version}/{version.run_id}/"
    receipt_key = prefix + "deployment.json"
    decision_digest = sha256_hex(encoded(decision))
    if head(storage, bucket, receipt_key):
        receipt = json.loads(download(storage, bucket, receipt_key))
        require(
            receipt["decision_sha256"] == decision_digest
            and receipt["evaluation_run_id"] == evaluation_id
            and receipt["registry_source"] == version.source
            and all(
                receipt[field] == source.data.tags.get(field)
                for field in ("dataset_uri", "dataset_sha256", "dataset_manifest_sha256")
            ),
            "Released champion provenance changed",
        )
        return {"status": "unchanged", **receipt}

    model_path = Path(
        tracking.download_artifacts(version.run_id, decision["model_artifact"], str(work))
    )
    require(
        sha256_hex(model_path.read_bytes()) == decision["model_sha256"],
        "Evaluated model SHA-256 mismatch",
    )
    catalog_path, catalog_key, manifest_key = matching_catalog(storage, bucket, source, work)
    predictor = ModelPredictor(model_path, catalog_path)
    require(1 in predictor.history, "Release smoke user 1 must be a known training user")
    model_key = prefix + "model.joblib"
    manifest = stage_artifacts.validate_and_write_manifest(
        model_path,
        catalog_path,
        model_key=model_key,
        catalog_key=catalog_key,
        manifest_path=work / "validated.json",
        expected_version=decision["bundle_version"],
        mlflow_run_id=version.run_id,
    )
    # No timestamps: repeated attempts must produce the same immutable release record.
    manifest.pop("staged_at_utc")
    record = {
        **identity,
        **manifest,
        "release_id": release_id,
        "bucket": bucket,
        "registry_source": version.source,
        "evaluation_run_id": evaluation_id,
        "decision_sha256": decision_digest,
        "decision": decision,
        "dataset_manifest_key": manifest_key,
        "dataset_manifest_sha256": source.data.tags["dataset_manifest_sha256"],
        "dataset_uri": source.data.tags["dataset_uri"],
        "dataset_sha256": source.data.tags["dataset_sha256"],
    }
    save_immutable(storage, bucket, prefix + "release.json", record)

    if deployed := deployments.find(release_id):
        # A previous attempt dispatched but could not persist its receipt.
        verify_object(
            storage, bucket, model_key, manifest["model_sha256"], manifest["model_size_bytes"]
        )
    else:
        # Detect alias moves before the external handoff; do not release a stale snapshot.
        current = tracking.get_model_version_by_alias(name, "champion")
        require(
            str(current.version) == str(version.version) and current.run_id == version.run_id,
            "Champion changed during release; retry the current champion",
        )
        stage_artifacts.main(
            [
                "upload",
                "--model",
                str(model_path),
                "--model-key",
                model_key,
                "--catalog",
                str(catalog_path),
                "--catalog-key",
                catalog_key,
                "--reuse-verified",
                "--deploy",
                "--mlflow-run-id",
                version.run_id,
                "--release-id",
                release_id,
            ]
        )
        deployed = deployments.wait(release_id)
    receipt = {**record, "deployment_run": deployed}
    save_immutable(storage, bucket, receipt_key, receipt)
    return {"status": "dispatched", **receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-model", required=True)
    args = parser.parse_args()
    settings = StorageSettings.from_env()
    with tempfile.TemporaryDirectory() as directory:
        result = release_champion(
            MlflowClient(),
            make_client(settings),
            settings.bucket,
            args.registered_model,
            Path(directory),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
