import argparse
import csv
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import pandas as pd

from src.models.mlflow_config import configure_mlflow, load_params
from src.models.recommender import MatrixFactorization, rating_history, recommend


def load_training_data(data_path):
    """Load ratings from a local CSV or an S3-compatible B2 object."""
    data_uri = str(data_path)
    if not data_uri.startswith("s3://"):
        return pd.read_csv(data_uri, usecols=["user_id", "movie_id", "rating", "title"])

    parsed = urlparse(data_uri)
    if not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("S3 training data must use the form s3://bucket/object.csv")

    access_key = os.getenv("B2_KEY_ID")
    secret_key = os.getenv("B2_APPLICATION_KEY")
    if not access_key or not secret_key:
        raise ValueError("B2_KEY_ID and B2_APPLICATION_KEY are required for S3 input")

    import boto3

    endpoint = os.getenv("B2_ENDPOINT")
    region = os.getenv("B2_REGION")
    if not endpoint or not region:
        raise ValueError("B2_ENDPOINT and B2_REGION are required for S3 input")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=boto3.session.Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )
    response = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    return pd.read_csv(
        BytesIO(response["Body"].read()),
        usecols=["user_id", "movie_id", "rating", "title"],
    )


def train_and_save(data_path, artifact_path):
    ratings = load_training_data(data_path)
    params = load_params()
    model_params = params.get("model", {})
    if not isinstance(model_params, dict):
        raise ValueError("The 'model' section is required in params.yaml")

    seed = params.get("seed", 42)
    factors = model_params.get("factors", 24)
    epochs = model_params.get("epochs", 12)
    regularization = model_params.get("regularization", 3.0)
    mlflow_params = {
        "seed": seed,
        "factors": factors,
        "epochs": epochs,
        "regularization": regularization,
        "model_name": model_params.get("name", "baseline"),
    }

    configure_mlflow(params)
    with mlflow.start_run() as run:
        model = MatrixFactorization().fit(
            ratings,
            factors=factors,
            epochs=epochs,
            regularization=regularization,
            seed=seed,
        )
        bundle = {
            "model_version": "movielens-v1",
            "model": model,
            "history": rating_history(ratings),
            "popularity": ratings.groupby("movie_id", sort=True).size(),
            "titles": ratings.drop_duplicates("movie_id").set_index("movie_id")["title"].to_dict(),
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, artifact_path, compress=3)

        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(
            {
                "training_ratings": len(ratings),
                "training_users": ratings["user_id"].nunique(),
                "training_movies": ratings["movie_id"].nunique(),
                "rating_mean": ratings["rating"].mean(),
            }
        )
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
        mlflow.set_tag("model_version", bundle["model_version"])
        print(
            f"Saved {bundle['model_version']} trained on {len(ratings):,} ratings "
            f"to {artifact_path} (run {run.info.run_id})"
        )


def write_recommendations(artifact_path, output_path, method, n):
    bundle = joblib.load(artifact_path)
    history = bundle["history"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0

    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "rank", "movie_id", "title", "score"])
        for user_id in sorted(history):
            ranked = recommend(user_id, n, bundle["model"], bundle["popularity"], history, method)
            for rank, (movie_id, score) in enumerate(ranked.items(), start=1):
                writer.writerow([user_id, rank, movie_id, bundle["titles"][movie_id], score])
            rows += len(ranked)

    print(f"Wrote {rows:,} {method} recommendations for {len(history):,} users to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and run the MovieLens batch recommender.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Fit both recommenders and save one artifact.")
    train.add_argument(
        "--data",
        required=True,
        help="Local CSV path or s3://bucket/object.csv B2 URI.",
    )
    train.add_argument("--artifact", type=Path, required=True)

    batch = commands.add_parser("batch", help="Load an artifact and write top-N recommendations.")
    batch.add_argument("--artifact", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--method", choices=["popularity", "mf"], default="popularity")
    batch.add_argument("--n", type=int, default=10)

    args = parser.parse_args()
    if args.command == "train":
        train_and_save(args.data, args.artifact)
    else:
        if args.n < 1:
            batch.error("--n must be at least 1")
        write_recommendations(args.artifact, args.output, args.method, args.n)


if __name__ == "__main__":
    main()
