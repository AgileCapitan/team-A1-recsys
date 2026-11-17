"""Build model-ready feature tables from the chronological train/test split.

Inputs:  data/processed/split/{train,test}.csv (written by ``src.data.split``)
Outputs: data/processed/features/
    - ratings_train.csv / ratings_test.csv  one row per rating
    - users.csv   one row per user (demographics + train statistics + genre affinity)
    - movies.csv  one row per movie (metadata + train statistics)
    - report.json shapes and missing-value handling summary

Leakage rules:
    * every statistic (means, std, popularity, affinity, indices) is fitted on TRAIN only;
      test rows are transformed with those train values (cold-start -> fallbacks);
    * sequential features (rating rank, days since first rating) only look at the past.
Every transformation is deterministic: same inputs -> byte-identical outputs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

RATING_MIN = 1
RATING_MAX = 5
SECONDS_PER_DAY = 86_400
# MovieLens 1M age codes -> readable bucket used for one-hot columns.
AGE_BUCKETS = {
    1: "under_18",
    18: "18_24",
    25: "25_34",
    35: "35_44",
    45: "45_49",
    50: "50_55",
    56: "56_plus",
}
UNKNOWN = "unknown"
# Pseudo-ratings used to shrink per-genre affinity toward 0 (see _genre_affinity).
AFFINITY_PRIOR = 5
USER_COLUMNS = ("gender", "age_code", "occupation_code", "zip_code")
MOVIE_COLUMNS = ("title", "genres", "release_year")


def load_ratings(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"zip_code": "string"}, keep_default_na=False, na_values=[""])
    missing = {"user_id", "movie_id", "rating"} - set(frame.columns)
    if missing:
        raise ValueError(f"Ratings are missing columns: {', '.join(sorted(missing))}")
    return frame


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _log1p(series: pd.Series) -> pd.Series:
    return pd.Series(np.log1p(series.to_numpy(dtype=float)), index=series.index)


def _min_max(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if pd.isna(low) or high == low:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


def _contiguous_index(values: pd.Series) -> pd.Series:
    """Map sorted unique train ids to 0..n-1 so models can use them as matrix rows."""
    unique = pd.Index(sorted(values.unique()))
    return pd.Series(range(len(unique)), index=unique)


def _genre_dummies(genres: pd.Series) -> pd.DataFrame:
    dummies = genres.fillna("").str.get_dummies(sep="|")
    dummies = dummies[sorted(dummies.columns)]
    dummies.columns = [_slug(column) for column in dummies.columns]
    return dummies


# --------------------------------------------------------------------------- movies
def build_movies_table(train: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Metadata for every known movie; rating statistics from train only."""
    meta_columns = [c for c in MOVIE_COLUMNS if c in catalog]
    if meta_columns:
        movies = catalog.groupby("movie_id")[meta_columns].first()
    else:
        movies = pd.DataFrame(index=pd.Index(sorted(catalog["movie_id"].unique()), name="movie_id"))

    stats = train.groupby("movie_id")["rating"].agg(
        n_ratings="size", mean_rating="mean", rating_std="std"
    )
    movies = movies.join(stats)
    movies["n_ratings"] = movies["n_ratings"].fillna(0).astype(int)
    movies["rating_std"] = movies["rating_std"].fillna(0.0)

    # Bayesian average: shrinks the mean of rarely-rated movies toward the global mean.
    global_mean = train["rating"].mean()
    prior_weight = float(stats["n_ratings"].median())
    movies["mean_rating"] = movies["mean_rating"].fillna(global_mean)
    movies["bayesian_rating"] = (
        movies["mean_rating"] * movies["n_ratings"] + global_mean * prior_weight
    ) / (movies["n_ratings"] + prior_weight)
    movies["popularity_norm"] = _min_max(_log1p(movies["n_ratings"]))

    imputed_years = 0
    if "release_year" in movies:
        missing_year = movies["release_year"].isna()
        imputed_years = int(missing_year.sum())
        median_year = movies["release_year"].median()
        fill_year = int(median_year) if pd.notna(median_year) else 0
        movies["release_year_missing"] = missing_year.astype(int)
        movies["release_year"] = movies["release_year"].fillna(fill_year).astype(int)
        movies["release_year_norm"] = _min_max(movies["release_year"].astype(float))
        movies["release_decade"] = (movies["release_year"] // 10 * 10).astype(int)

    if "genres" in movies:
        movies["genres"] = movies["genres"].fillna("")
        movies = movies.join(_genre_dummies(movies["genres"]).add_prefix("genre_"))

    train_index = _contiguous_index(train["movie_id"])
    movies.insert(0, "movie_idx", movies.index.map(train_index).fillna(-1).astype(int))
    # Movies only seen in test: their statistics are fallbacks, not observed values.
    movies.insert(1, "is_cold_start", movies["movie_idx"].lt(0).astype(int))
    return movies.reset_index(), imputed_years


# --------------------------------------------------------------------------- users
def build_users_table(train: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Demographics for every known user; statistics and genre affinity from train only."""
    profile_columns = [c for c in USER_COLUMNS if c in catalog]
    if profile_columns:
        users = catalog.groupby("user_id")[profile_columns].first()
    else:
        users = pd.DataFrame(index=pd.Index(sorted(catalog["user_id"].unique()), name="user_id"))

    stats = train.groupby("user_id")["rating"].agg(
        n_ratings="size", mean_rating="mean", rating_std="std"
    )
    users = users.join(stats)
    users["n_ratings"] = users["n_ratings"].fillna(0).astype(int)
    users["mean_rating"] = users["mean_rating"].fillna(train["rating"].mean())
    users["rating_std"] = users["rating_std"].fillna(0.0)
    users["n_ratings_log_norm"] = _min_max(_log1p(users["n_ratings"]))

    if "timestamp" in train:
        span = train.groupby("user_id")["timestamp"].agg(lambda ts: ts.max() - ts.min())
        users["active_days"] = (users.index.map(span).fillna(0) / SECONDS_PER_DAY).round(3)

    if "gender" in users:
        users["gender"] = users["gender"].fillna(UNKNOWN)
        users["is_female"] = users["gender"].eq("F").astype(int)
    if "age_code" in users:
        users["age_bucket"] = users["age_code"].map(AGE_BUCKETS).fillna(UNKNOWN)
        age_dummies = pd.get_dummies(users["age_bucket"], prefix="age", dtype=int)
        buckets = [*AGE_BUCKETS.values(), UNKNOWN]
        users = users.join(age_dummies[[f"age_{b}" for b in buckets if f"age_{b}" in age_dummies]])
    if "occupation_code" in users:
        users["occupation_code"] = users["occupation_code"].fillna(-1).astype(int)
    if "zip_code" in users:
        # First digit = US region; finer granularity mostly adds noise.
        users["zip_region"] = users["zip_code"].fillna("").str[:1].replace("", UNKNOWN)

    if "genres" in train:
        users = users.join(_genre_affinity(train), how="left")
        affinity_columns = [c for c in users if c.startswith(("share_", "affinity_"))]
        users[affinity_columns] = users[affinity_columns].fillna(0.0)

    train_index = _contiguous_index(train["user_id"])
    users.insert(0, "user_idx", users.index.map(train_index).fillna(-1).astype(int))
    # Users only seen in test: their share_/affinity_ zeros mean "no history", not "dislike".
    users.insert(1, "is_cold_start", users["user_idx"].lt(0).astype(int))
    return users.reset_index()


def _genre_affinity(train: pd.DataFrame) -> pd.DataFrame:
    """Per user and genre: share of their ratings, and mean rating relative to their own mean.

    affinity_<genre> > 0 means the user rates that genre above their personal average.
    The mean is shrunk toward 0 with AFFINITY_PRIOR pseudo-ratings, so a genre rated
    once or twice cannot produce an extreme affinity (mean * n / (n + prior)).
    """
    genres = _genre_dummies(train["genres"])
    centered = train["rating"] - train.groupby("user_id")["rating"].transform("mean")
    counts = genres.groupby(train["user_id"]).sum()
    share = counts.div(train.groupby("user_id").size(), axis=0)
    centered_sum = genres.mul(centered, axis=0).groupby(train["user_id"]).sum()
    affinity = centered_sum / (counts + AFFINITY_PRIOR)
    return share.add_prefix("share_").join(affinity.add_prefix("affinity_"))


# --------------------------------------------------------------------------- ratings
def build_ratings_tables(
    train: pd.DataFrame, test: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_lookup = users.set_index("user_id")
    movie_lookup = movies.set_index("movie_id")
    columns = [c for c in ("user_id", "movie_id", "rating", "timestamp") if c in train]
    combined = pd.concat(
        [train[columns].assign(split="train"), test[columns].assign(split="test")],
        ignore_index=True,
    )

    combined["user_idx"] = combined["user_id"].map(user_lookup["user_idx"]).fillna(-1).astype(int)
    combined["movie_idx"] = (
        combined["movie_id"].map(movie_lookup["movie_idx"]).fillna(-1).astype(int)
    )
    combined["rating_norm"] = (combined["rating"] - RATING_MIN) / (RATING_MAX - RATING_MIN)
    # Centered on the TRAIN user mean (cold-start users fall back to the global train mean).
    combined["rating_centered"] = combined["rating"] - combined["user_id"].map(
        user_lookup["mean_rating"]
    )

    if "timestamp" in combined:
        rated_at = pd.to_datetime(combined["timestamp"], unit="s", utc=True)
        combined["rated_year"] = rated_at.dt.year
        combined["rated_month"] = rated_at.dt.month
        combined["rated_weekday"] = rated_at.dt.weekday
        if "release_year" in movie_lookup:
            release = combined["movie_id"].map(movie_lookup["release_year"])
            combined["movie_age_at_rating"] = (combined["rated_year"] - release).clip(lower=0)

        # Sequential, past-only features: computed in time order over train then test.
        ordered = combined.sort_values(["timestamp", "user_id", "movie_id"], kind="stable")
        by_user = ordered.groupby("user_id")
        combined["user_rating_rank"] = by_user.cumcount().reindex(combined.index) + 1
        first_seen = by_user["timestamp"].transform("min").reindex(combined.index)
        combined["days_since_user_first_rating"] = (
            (combined["timestamp"] - first_seen) / SECONDS_PER_DAY
        ).round(3)

    tables = []
    for name in ("train", "test"):
        part = combined.loc[combined["split"].eq(name)].drop(columns="split")
        tables.append(part.sort_values(["user_id", "movie_id"], kind="stable"))
    return tables[0].reset_index(drop=True), tables[1].reset_index(drop=True)


# --------------------------------------------------------------------------- pipeline
def build_features(train_path: Path, test_path: Path, output_dir: Path) -> dict:
    train = load_ratings(train_path)
    test = load_ratings(test_path)
    if train.empty:
        raise ValueError("Train split is empty.")
    # Catalog = metadata only (no ratings are used from it): safe to read from both splits.
    catalog = pd.concat([train, test], ignore_index=True)

    movies, imputed_years = build_movies_table(train, catalog)
    users = build_users_table(train, catalog)
    ratings_train, ratings_test = build_ratings_tables(train, test, users, movies)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "ratings_train": ratings_train,
        "ratings_test": ratings_test,
        "users": users,
        "movies": movies,
    }
    for name, table in tables.items():
        table.to_csv(
            output_dir / f"{name}.csv", index=False, float_format="%.6f", lineterminator="\n"
        )

    n_train_users = int(users["user_idx"].ge(0).sum())
    n_train_movies = int(movies["movie_idx"].ge(0).sum())
    report = {
        "n_train_ratings": len(ratings_train),
        "n_test_ratings": len(ratings_test),
        "n_users": len(users),
        "n_movies": len(movies),
        "train_density": round(len(ratings_train) / (n_train_users * n_train_movies), 6),
        "train_rating_mean": round(float(train["rating"].mean()), 6),
        "cold_start_test_ratings": int(
            (ratings_test["user_idx"].lt(0) | ratings_test["movie_idx"].lt(0)).sum()
        ),
        "release_year_imputed_with_median": imputed_years,
        "columns": {name: list(table.columns) for name, table in tables.items()},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    split_dir = root / "data" / "processed" / "split"
    parser = argparse.ArgumentParser(description="Build feature tables from the train/test split.")
    parser.add_argument("--train", type=Path, default=split_dir / "train.csv")
    parser.add_argument("--test", type=Path, default=split_dir / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed" / "features")
    args = parser.parse_args(argv)

    report = build_features(args.train, args.test, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "columns"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
