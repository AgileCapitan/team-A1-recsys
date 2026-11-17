"""Chronological train/test split of the cleaned ratings.

Same rule as the evaluation in notebooks/movieRecommender.ipynb: the cutoff is the
timestamp at the (1 - test_fraction) quantile of all ratings; ratings strictly before
it go to train, the rest to test. Splitting by time (not at random) mimics production:
the model is trained on the past and evaluated on the future.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def chronological_split(
    ratings: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    timestamps = pd.to_numeric(ratings["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Every rating needs a timestamp for a chronological split.")
    position = int((1 - test_fraction) * len(ratings))
    cutoff = int(np.sort(timestamps.to_numpy())[position])
    train = ratings.loc[timestamps < cutoff]
    test = ratings.loc[timestamps >= cutoff]
    return train, test, cutoff


def split_file(input_path: Path, output_dir: Path, test_fraction: float) -> dict:
    ratings = pd.read_csv(input_path, dtype="string", keep_default_na=False)
    train, test, cutoff = chronological_split(ratings, test_fraction)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("train", train), ("test", test)):
        table.to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")

    train_users = set(train["user_id"])
    train_movies = set(train["movie_id"])
    report = {
        "test_fraction": test_fraction,
        "cutoff_timestamp": cutoff,
        "cutoff_utc": pd.to_datetime(cutoff, unit="s", utc=True).isoformat(),
        "train_ratings": len(train),
        "test_ratings": len(test),
        "test_users": int(test["user_id"].nunique()),
        "cold_start_test_users": int((~test["user_id"].drop_duplicates().isin(train_users)).sum()),
        "cold_start_test_movies": int(
            (~test["movie_id"].drop_duplicates().isin(train_movies)).sum()
        ),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Chronological train/test split.")
    parser.add_argument(
        "--input", type=Path, default=root / "data" / "processed" / "ratings_clean.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed" / "split")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    print(json.dumps(split_file(args.input, args.output_dir, args.test_fraction), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
