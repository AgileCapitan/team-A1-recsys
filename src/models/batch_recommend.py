import argparse
import csv
from pathlib import Path

import joblib
import pandas as pd

from src.models.recommender import MatrixFactorization, rating_history, recommend


def train_and_save(data_path, artifact_path):
    ratings = pd.read_csv(data_path, usecols=["user_id", "movie_id", "rating", "title"])
    bundle = {
        "model_version": "movielens-v1",
        "model": MatrixFactorization().fit(ratings),
        "history": rating_history(ratings),
        "popularity": ratings.groupby("movie_id", sort=True).size(),
        "titles": ratings.drop_duplicates("movie_id").set_index("movie_id")["title"].to_dict(),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path, compress=3)
    print(f"Saved {bundle['model_version']} trained on {len(ratings):,} ratings to {artifact_path}")


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
    train.add_argument("--data", type=Path, required=True)
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
