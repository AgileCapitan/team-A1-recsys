from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix

from src.models.recommender import update_factors

_YEAR_SUFFIX_RE = re.compile(r"\((\d{4})\)\s*$")


def _parse_release_year(row: dict[str, str], title: str) -> int | None:
    """Extract release year from row metadata or trailing '(YYYY)' in title."""
    raw_year = (row.get("release_year") or "").strip()
    if raw_year:
        return int(float(raw_year))
    match = _YEAR_SUFFIX_RE.search(title)
    return int(match.group(1)) if match else None


class NewUserRecommender:
    """Cold-start onboarding recommender using selected genres and user ratings."""

    def __init__(
        self,
        model_path: Path,
        catalog_path: Path,
        regularization: float = 3.0,
    ) -> None:
        bundle = joblib.load(model_path)
        self._init_from_bundle(bundle, catalog_path, regularization)

    @classmethod
    def from_bundle(
        cls,
        bundle: dict[str, Any],
        catalog_path: Path,
        regularization: float = 3.0,
    ) -> NewUserRecommender:
        """Create a NewUserRecommender from an already-loaded model bundle."""
        instance = cls.__new__(cls)
        instance._init_from_bundle(bundle, catalog_path, regularization)
        return instance

    def _init_from_bundle(
        self,
        bundle: dict[str, Any],
        catalog_path: Path,
        regularization: float = 3.0,
    ) -> None:
        self.model = bundle["model"]
        self.model_version = bundle["model_version"]
        self.regularization = regularization

        self.catalog: dict[int, dict[str, Any]] = {}
        with Path(catalog_path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                movie_id = int(row["movie_id"])
                if movie_id not in self.model.movie_ids:
                    continue

                title = str(bundle["titles"].get(movie_id) or row.get("title") or "")
                genres = tuple(
                    genre.strip() for genre in (row.get("genres") or "").split("|") if genre.strip()
                )
                self.catalog[movie_id] = {
                    "movie_id": movie_id,
                    "title": title,
                    "genres": genres,
                    "release_year": _parse_release_year(row, title),
                }

        missing = set(self.model.movie_ids) - self.catalog.keys()
        if missing:
            raise ValueError(f"Catalog is missing {len(missing)} trained movies.")

        self.genres = sorted(
            {genre for movie in self.catalog.values() for genre in movie["genres"]}
        )
        self.popular_ids = [
            int(movie_id)
            for movie_id in bundle["popularity"].sort_values(ascending=False, kind="stable").index
        ]

    def rating_options(self, genres: list[str], per_genre: int = 25) -> list[int]:
        """Offer popular, recognizable films from each selected genre."""
        options: dict[int, None] = {}
        for genre in genres:
            matches = [
                movie_id
                for movie_id in self.popular_ids
                if genre in self.catalog[movie_id]["genres"]
            ]
            for movie_id in matches[:per_genre]:
                options[movie_id] = None
        return list(options)

    def recommend(
        self,
        genres: list[str],
        ratings: dict[int, int],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        selected_genres = set(genres)
        if not selected_genres or selected_genres.difference(self.genres):
            raise ValueError("Choose at least one available genre.")
        if not 3 <= len(ratings) <= 10:
            raise ValueError("Rate between 3 and 10 different movies.")
        if any(type(rating) is not int or not 1 <= rating <= 5 for rating in ratings.values()):
            raise ValueError("Each rating must be a whole number from 1 to 5.")
        if not 1 <= top_k <= 20:
            raise ValueError("Choose between 1 and 20 recommendations.")

        for movie_id in ratings:
            movie = self.catalog.get(movie_id)
            if movie is None or selected_genres.isdisjoint(movie["genres"]):
                raise ValueError("Rate movies from your selected genres.")

        model = self.model
        rated_columns = model.movie_ids.get_indexer(list(ratings))
        observations = csr_matrix(
            (
                np.array(list(ratings.values()), dtype=float),
                (np.zeros(len(ratings), dtype=int), rated_columns),
            ),
            shape=(1, len(model.movie_ids)),
        )
        user = update_factors(observations, model.movies, self.regularization, model.mean)[0]

        scores = model.mean + user[0] + model.movies[:, 0] + model.movies[:, 1:] @ user[1:]
        candidates = [
            (int(movie_id), float(score))
            for movie_id, score in zip(model.movie_ids, scores, strict=True)
            if movie_id not in ratings
            and not selected_genres.isdisjoint(self.catalog[movie_id]["genres"])
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        return [
            {**self.catalog[movie_id], "score": score} for movie_id, score in candidates[:top_k]
        ]


def build_movie_catalog(source_path: Path, output_path: Path) -> int:
    """Build a deduplicated movie catalog CSV from a ratings or movies CSV/CSV.GZ file."""
    source = Path(source_path)
    output = Path(output_path)

    opener = gzip.open if source.suffix == ".gz" else Path.open
    movies: dict[int, dict[str, Any]] = {}

    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            movie_id = int(row["movie_id"])
            if movie_id in movies:
                continue
            title = (row.get("title") or "").strip()
            genres = (row.get("genres") or "").strip()
            release_year = _parse_release_year(row, title)
            movies[movie_id] = {
                "movie_id": movie_id,
                "title": title,
                "genres": genres,
                "release_year": "" if release_year is None else str(release_year),
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["movie_id", "title", "genres", "release_year"],
        )
        writer.writeheader()
        for movie_id in sorted(movies):
            writer.writerow(movies[movie_id])

    return len(movies)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a movie-catalog.csv file for NewUserRecommender."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input ratings/movies CSV or .csv.gz path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination movie-catalog.csv path.",
    )
    args = parser.parse_args()
    count = build_movie_catalog(args.source, args.output)
    print(f"Wrote {count:,} movies to {args.output}")


if __name__ == "__main__":
    main()
