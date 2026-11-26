from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.storage import (
    StorageSettings,
    make_client,
    sha256_hex,
    upload_verified,
    verify_object,
)
from src.serving.predictor import ModelPredictor


def _require_non_empty_file(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    data = path.read_bytes()
    if not data:
        raise ValueError(f"{label} file is empty: {path}")
    return data


def validate_and_write_manifest(
    model_path: Path,
    catalog_path: Path,
    *,
    model_key: str,
    catalog_key: str,
    manifest_path: Path,
    expected_version: str | None = None,
    mlflow_run_id: str | None = None,
    git_sha: str | None = None,
    expected_model_sha256: str | None = None,
    expected_catalog_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate staged model + catalog artifacts and write staged-version.json."""
    model_file = Path(model_path)
    catalog_file = Path(catalog_path)
    model_bytes = _require_non_empty_file(model_file, "Model artifact")
    catalog_bytes = _require_non_empty_file(catalog_file, "Movie catalog")

    for label, content, expected in (
        ("Model", model_bytes, expected_model_sha256),
        ("Catalog", catalog_bytes, expected_catalog_sha256),
    ):
        if expected and sha256_hex(content) != expected:
            raise ValueError(f"{label} SHA-256 mismatch.")

    predictor = ModelPredictor(model_path=model_file, catalog_path=catalog_file)
    if not predictor.artifact_loaded or predictor.using_fallback:
        raise ValueError("Model artifact did not load into ModelPredictor.")
    if not predictor.catalog_loaded:
        raise ValueError(predictor.catalog_error or "Movie catalog did not load.")

    clean_expected = expected_version.strip() if expected_version else ""
    if clean_expected and predictor.model_version != clean_expected:
        raise ValueError(
            f"Model version mismatch: expected {clean_expected!r}, got {predictor.model_version!r}."
        )

    recommendations = predictor.predict(user_id=1, top_k=3)
    if not recommendations:
        raise ValueError("Smoke prediction returned no recommendations.")
    if not all(math.isfinite(float(rec["score"])) for rec in recommendations):
        raise ValueError("Smoke prediction returned non-finite recommendation scores.")
    if not predictor.genres:
        raise ValueError("Loaded catalog contains no genres.")

    manifest = {
        "model_version": predictor.model_version,
        "model_key": model_key,
        "model_sha256": sha256_hex(model_bytes),
        "model_size_bytes": len(model_bytes),
        "catalog_key": catalog_key,
        "catalog_sha256": sha256_hex(catalog_bytes),
        "catalog_size_bytes": len(catalog_bytes),
        "mlflow_run_id": mlflow_run_id.strip() if mlflow_run_id and mlflow_run_id.strip() else None,
        "git_sha": git_sha.strip() if git_sha and git_sha.strip() else None,
        "staged_at_utc": datetime.now(UTC).isoformat(),
    }

    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def upload_serving_artifacts(
    model_path: Path,
    model_key: str,
    *,
    catalog_path: Path | None = None,
    catalog_key: str | None = None,
    client: Any = None,
    bucket: str | None = None,
    reuse_verified: bool = False,
) -> dict[str, Any]:
    """Validate and upload immutable serving artifacts to Backblaze B2."""
    if bool(catalog_path) != bool(catalog_key):
        raise ValueError("Provide both --catalog and --catalog-key together, or neither.")

    model_file = Path(model_path)
    model_bytes = _require_non_empty_file(model_file, "Model artifact")
    catalog_file = Path(catalog_path) if catalog_path is not None else None
    catalog_bytes = (
        _require_non_empty_file(catalog_file, "Movie catalog") if catalog_file is not None else None
    )

    predictor = ModelPredictor(model_path=model_file, catalog_path=catalog_file)
    if catalog_file is not None and not predictor.catalog_loaded:
        raise ValueError(predictor.catalog_error or "Movie catalog did not load.")

    if client is None or bucket is None:
        settings = StorageSettings.from_env(bucket=bucket)
        client = client or make_client(settings)
        bucket = bucket or settings.bucket

    def publish(client, bucket, key, data, content_type):
        try:
            return upload_verified(client, bucket, key, data, content_type=content_type)
        except FileExistsError:
            if not reuse_verified:
                raise
            return verify_object(client, bucket, key, sha256_hex(data), len(data))

    model_result = publish(
        client,
        bucket,
        model_key,
        model_bytes,
        content_type="application/octet-stream",
    )

    summary: dict[str, Any] = {
        "model_version": predictor.model_version,
        "bucket": bucket,
        "model": model_result,
    }

    if catalog_bytes is not None and catalog_key is not None:
        summary["catalog"] = publish(
            client,
            bucket,
            catalog_key,
            catalog_bytes,
            content_type="text/csv",
        )

    return summary


def _dispatch_deployment(summary: dict[str, Any], mlflow_run_id: str | None) -> None:
    inputs = {
        "model_key": summary["model"]["key"],
        "expected_version": summary["model_version"],
    }
    if "catalog" in summary:
        inputs["catalog_key"] = summary["catalog"]["key"]
    for field, artifact in (("model_sha256", "model"), ("catalog_sha256", "catalog")):
        if summary.get(artifact, {}).get("sha256"):
            inputs[field] = summary[artifact]["sha256"]
    if summary.get("release_id"):
        inputs["release_id"] = summary["release_id"]
    if mlflow_run_id:
        inputs["mlflow_run_id"] = mlflow_run_id

    subprocess.run(
        [
            "gh",
            "workflow",
            "run",
            "deploy.yml",
            "--repo",
            "AgileCapitan/team-A1-recsys",
            "--ref",
            "main",
            "--json",
        ],
        input=json.dumps(inputs),
        text=True,
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate staged serving artifacts and upload versioned bundles to B2."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate_cmd = commands.add_parser(
        "validate",
        help="Validate downloaded serving files and write staged-version.json.",
    )
    validate_cmd.add_argument("--model", type=Path, required=True)
    validate_cmd.add_argument("--catalog", type=Path, required=True)
    validate_cmd.add_argument("--model-key", required=True)
    validate_cmd.add_argument("--catalog-key", required=True)
    validate_cmd.add_argument("--manifest", type=Path, required=True)
    validate_cmd.add_argument("--expected-version", default=None)
    validate_cmd.add_argument("--mlflow-run-id", default=None)
    validate_cmd.add_argument("--git-sha", default=None)
    validate_cmd.add_argument("--expected-model-sha256", default=None)
    validate_cmd.add_argument("--expected-catalog-sha256", default=None)

    upload_cmd = commands.add_parser(
        "upload",
        help="Validate and immutably upload serving artifact(s) to Backblaze B2.",
    )
    upload_cmd.add_argument("--model", type=Path, required=True)
    upload_cmd.add_argument("--model-key", required=True)
    upload_cmd.add_argument("--catalog", type=Path, default=None)
    upload_cmd.add_argument("--catalog-key", default=None)
    upload_cmd.add_argument(
        "--deploy",
        action="store_true",
        help="Start the Oracle deployment after the upload is verified (requires gh).",
    )
    upload_cmd.add_argument(
        "--reuse-verified",
        action="store_true",
        help="Reuse existing objects only after full size/checksum verification.",
    )
    upload_cmd.add_argument("--release-id", default=None)
    upload_cmd.add_argument("--mlflow-run-id", help="MLflow run ID to record during deployment.")

    args = parser.parse_args(argv)
    if args.command == "validate":
        manifest = validate_and_write_manifest(
            args.model,
            args.catalog,
            model_key=args.model_key,
            catalog_key=args.catalog_key,
            manifest_path=args.manifest,
            expected_version=args.expected_version,
            mlflow_run_id=args.mlflow_run_id,
            git_sha=args.git_sha,
            expected_model_sha256=args.expected_model_sha256,
            expected_catalog_sha256=args.expected_catalog_sha256,
        )
        print(json.dumps(manifest, indent=2))
    else:
        if args.deploy and shutil.which("gh") is None:
            parser.error("--deploy requires the GitHub CLI (gh).")
        result = upload_serving_artifacts(
            args.model,
            args.model_key,
            catalog_path=args.catalog,
            catalog_key=args.catalog_key,
            reuse_verified=args.reuse_verified,
        )
        if args.release_id:
            result["release_id"] = args.release_id
        print(json.dumps(result, indent=2), flush=args.deploy)
        if args.deploy:
            try:
                _dispatch_deployment(result, args.mlflow_run_id)
            except (OSError, subprocess.CalledProcessError):
                parser.exit(
                    1,
                    "B2 upload verified, but the deploy request failed. "
                    "Retry Deploy to Cloud VM using the uploaded keys above.\n",
                )


if __name__ == "__main__":
    main()
