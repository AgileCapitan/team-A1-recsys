import csv
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix

from src.models.recommender import update_factors


class NewUserRecommender:
    def __init__(
        self,
        model_path: Path,
        catalog_path: Path,
        regularization: float = 3.0,
    ):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.model_version = bundle["model_version"]
        self.regularization = regularization

        self.catalog = {}
        with catalog_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                movie_id = int(row["movie_id"])
                if movie_id not in self.model.movie_ids:
                    continue

                self.catalog[movie_id] = {
                    "movie_id": movie_id,
                    "title": bundle["titles"][movie_id],
                    "genres": tuple(
                        genre for genre in row["genres"].split("|") if genre
                    ),
                    "release_year": (
                        int(row["release_year"]) if row["release_year"] else None
                    ),
                }

        missing = set(self.model.movie_ids) - self.catalog.keys()
        if missing:
            raise ValueError(f"Catalog is missing {len(missing)} trained movies.")

        self.genres = sorted(
            {genre for movie in self.catalog.values() for genre in movie["genres"]}
        )
        self.popular_ids = [
            int(movie_id)
            for movie_id in bundle["popularity"]
            .sort_values(ascending=False, kind="stable")
            .index
        ]

    def rating_options(self, genres: list[str], per_genre: int = 25) -> list[int]:
        """Offer popular, recognizable films from each selected genre."""
        options = {}
        for genre in genres:
            matches = [
                movie_id for movie_id in self.popular_ids
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
    ) -> list[dict]:
        selected_genres = set(genres)
        if not selected_genres or selected_genres.difference(self.genres):
            raise ValueError("Choose at least one available genre.")
        if not 3 <= len(ratings) <= 10:
            raise ValueError("Rate between 3 and 10 different movies.")
        if any(type(rating) is not int or not 1 <= rating <= 5
               for rating in ratings.values()):
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
        user = update_factors(
            observations, model.movies, self.regularization, model.mean
        )[0]

        scores = (
            model.mean
            + user[0]
            + model.movies[:, 0]
            + model.movies[:, 1:] @ user[1:]
        )
        candidates = [
            (int(movie_id), float(score))
            for movie_id, score in zip(model.movie_ids, scores)
            if movie_id not in ratings
            and not selected_genres.isdisjoint(self.catalog[movie_id]["genres"])
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        return [
            {**self.catalog[movie_id], "score": score}
            for movie_id, score in candidates[:top_k]
        ]