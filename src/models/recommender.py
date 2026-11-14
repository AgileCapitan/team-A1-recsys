import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def rating_history(ratings):
    return ratings.groupby("user_id")["movie_id"].agg(set).to_dict()


def update_factors(observed, fixed, regularization, mean):
    # The first column is the bias. The remaining columns are latent factors.
    features = np.column_stack([np.ones(len(fixed)), fixed[:, 1:]])
    penalty = regularization * np.eye(features.shape[1])
    updated = np.empty((observed.shape[0], features.shape[1]))

    for row in range(observed.shape[0]):
        start, stop = observed.indptr[row : row + 2]
        columns = observed.indices[start:stop]
        design = features[columns]
        target = observed.data[start:stop] - mean - fixed[columns, 0]
        updated[row] = np.linalg.solve(design.T @ design + penalty, design.T @ target)

    return updated


class MatrixFactorization:
    def fit(self, ratings, factors=24, epochs=12, regularization=3.0, seed=42):
        user_ids = pd.Index(np.sort(ratings["user_id"].unique()))
        self.movie_ids = pd.Index(np.sort(ratings["movie_id"].unique()), name="movie_id")
        self.user_rows = {user_id: row for row, user_id in enumerate(user_ids)}
        self.mean = ratings["rating"].mean()

        user_rows = user_ids.get_indexer(ratings["user_id"])
        movie_rows = self.movie_ids.get_indexer(ratings["movie_id"])
        observations = csr_matrix(
            (ratings["rating"].to_numpy(dtype=float), (user_rows, movie_rows)),
            shape=(len(user_ids), len(self.movie_ids)),
        )
        transposed = observations.T.tocsr()

        rng = np.random.default_rng(seed)
        self.movies = rng.normal(0, 0.1, size=(len(self.movie_ids), factors + 1))
        self.movies[:, 0] = 0.0
        for _ in range(epochs):
            self.users = update_factors(observations, self.movies, regularization, self.mean)
            self.movies = update_factors(transposed, self.users, regularization, self.mean)
        return self

    def score(self, user_id):
        user = self.users[self.user_rows[user_id]]
        scores = self.mean + user[0] + self.movies[:, 0] + self.movies[:, 1:] @ user[1:]
        return pd.Series(scores, index=self.movie_ids, name="predicted_score")


def recommend(user_id, n, model, popularity, history, method):
    scores = popularity
    if method == "mf" and user_id in model.user_rows:
        scores = model.score(user_id)

    unseen = scores.loc[~scores.index.isin(history.get(user_id, ()))]
    # Both scoring indexes start in movie ID order, so ties keep the lower ID first.
    return unseen.sort_values(ascending=False, kind="stable").head(n)