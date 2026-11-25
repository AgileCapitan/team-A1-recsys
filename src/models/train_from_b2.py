"""Train once per published dataset and trainer version for GitHub Actions."""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from mlflow import MlflowClient

from src.data.storage import PROCESSED_PREFIX, StorageSettings, download, make_client, sha256_hex
from src.models.batch_recommend import train_and_save
from src.models.mlflow_config import load_params

DATASET_PREFIX = PROCESSED_PREFIX.rsplit("/", 2)[0] + "/"
MANIFEST_KEY = re.compile(rf"{re.escape(DATASET_PREFIX)}v([1-9][0-9]*)/manifest\.json")


def latest_manifest_key(client, bucket):
    manifests = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=DATASET_PREFIX
    ):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            if match := MANIFEST_KEY.fullmatch(key):
                manifests.append((int(match[1]), key))
    if not manifests:
        raise ValueError(f"No published dataset manifest under s3://{bucket}/{DATASET_PREFIX}")
    return max(manifests)[1]


def train_from_b2(artifact: Path, trainer_digest: str, commit: str):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if urlparse(tracking_uri).scheme != "https":
        raise ValueError("The workflow requires a shared HTTPS MLFLOW_TRACKING_URI.")

    settings = StorageSettings.from_env()
    storage = make_client(settings)
    manifest_key = latest_manifest_key(storage, settings.bucket)
    manifest_bytes = download(storage, settings.bucket, manifest_key)
    ratings_key = manifest_key.removesuffix("manifest.json") + "ratings.csv"
    manifest = json.loads(manifest_bytes)
    ratings = next((entry for entry in manifest["outputs"] if entry["key"] == ratings_key), None)
    if ratings is None or not re.fullmatch(r"[a-f0-9]{64}", ratings.get("sha256", "")):
        raise ValueError(f"Missing ratings.csv checksum in {manifest_key}")

    ratings_uri = f"s3://{settings.bucket}/{ratings_key}"
    manifest_digest = sha256_hex(manifest_bytes)
    identity = sha256_hex(f"{ratings_uri}:{manifest_digest}:{trainer_digest}".encode())
    tracking = MlflowClient(tracking_uri=tracking_uri)
    experiment = tracking.get_experiment_by_name(load_params()["mlflow"]["experiment_name"])
    if experiment and tracking.search_runs(
        [experiment.experiment_id],
        f"tags.training_input = '{identity}' AND attributes.status = 'FINISHED'",
        max_results=1,
    ):
        print(f"Training already completed for {ratings_uri} with this trainer version.")
        return

    train_and_save(
        ratings_uri,
        artifact,
        expected_sha256=ratings["sha256"],
        run_tags={
            "training_input": identity,
            "dataset_uri": ratings_uri,
            "dataset_sha256": ratings["sha256"],
            "dataset_manifest_sha256": manifest_digest,
            "trainer_commit": commit,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trainer-digest", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    train_from_b2(args.artifact, args.trainer_digest, args.commit)
