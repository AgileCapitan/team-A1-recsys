import json

import pandas as pd
import pytest

from src.data.features import build_features
from src.data.split import chronological_split

HEADER = "user_id,movie_id,rating,timestamp,title,genres,release_year,gender,age_code,occupation_code,zip_code"
TRAIN_ROWS = [
    "1,10,5,978300760,Film A (1995),Drama|Comedy,1995,F,1,10,48067",
    "1,20,3,978302109,Film B (2000),Action,2000,F,1,10,48067",
    "2,10,4,978298413,Film A (1995),Drama|Comedy,1995,M,56,16,70072",
    "2,30,1,978298000,Film C,Drama,,M,56,16,70072",
]
TEST_ROWS = [
    "1,30,2,978400000,Film C,Drama,,F,1,10,48067",
    "3,40,5,978500000,Film D (1990),Horror,1990,M,25,1,10001",  # cold-start user and movie
]


def _write(path, rows):
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def built(tmp_path):
    train = _write(tmp_path / "train.csv", TRAIN_ROWS)
    test = _write(tmp_path / "test.csv", TEST_ROWS)
    out = tmp_path / "features"
    report = build_features(train, test, out)
    names = ("ratings_train", "ratings_test", "users", "movies")
    tables = {name: pd.read_csv(out / f"{name}.csv") for name in names}
    return report, tables, out, (train, test)


def test_report_shapes(built):
    report, tables, out, _ = built
    assert (report["n_train_ratings"], report["n_test_ratings"]) == (4, 2)
    assert (report["n_users"], report["n_movies"]) == (3, 4)
    assert report["cold_start_test_ratings"] == 1
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report


def test_statistics_use_train_only(tmp_path, built):
    """Changing a test rating must not change any user/movie statistic (no leakage)."""
    _, tables, _, (train, _) = built
    altered = _write(
        tmp_path / "test_altered.csv", [TEST_ROWS[0].replace(",2,", ",5,"), TEST_ROWS[1]]
    )
    build_features(train, altered, tmp_path / "altered")
    for name in ("users", "movies"):
        pd.testing.assert_frame_equal(
            tables[name], pd.read_csv(tmp_path / "altered" / f"{name}.csv")
        )

    users = tables["users"].set_index("user_id")
    assert users.loc[1, "n_ratings"] == 2 and users.loc[1, "mean_rating"] == pytest.approx(4.0)
    movies = tables["movies"].set_index("movie_id")
    assert movies.loc[30, "n_ratings"] == 1  # the test rating of movie 30 is not counted


def test_cold_start_entities(built):
    _, tables, _, _ = built
    users = tables["users"].set_index("user_id")
    movies = tables["movies"].set_index("movie_id")
    assert users.loc[3, "user_idx"] == -1 and users.loc[3, "n_ratings"] == 0
    assert movies.loc[40, "movie_idx"] == -1 and movies.loc[40, "genre_horror"] == 1
    assert sorted(users["user_idx"]) == [-1, 0, 1]
    assert users["is_cold_start"].to_dict() == {1: 0, 2: 0, 3: 1}
    assert movies.loc[40, "is_cold_start"] == 1 and movies.loc[10, "is_cold_start"] == 0
    test = tables["ratings_test"].set_index("user_id")
    assert test.loc[3, "user_idx"] == -1 and test.loc[1, "movie_idx"] >= 0


def test_rating_features(built):
    _, tables, _, _ = built
    train, test = tables["ratings_train"], tables["ratings_test"]
    assert train["rating_norm"].between(0, 1).all()
    # Train ratings centered on the train user mean sum to zero per user.
    assert train.groupby("user_id")["rating_centered"].sum().abs().max() == pytest.approx(0)
    # Test rating of user 1 is centered on the TRAIN mean (4.0).
    assert test.set_index("user_id").loc[1, "rating_centered"] == pytest.approx(-2.0)
    assert set(train["rated_year"]) == {2000}  # 978300760 = 2000-12-31 UTC
    film_a = train.query("user_id == 1 and movie_id == 10").iloc[0]
    assert film_a["movie_age_at_rating"] == 5


def test_sequential_features_only_look_back(built):
    _, tables, _, _ = built
    user1 = pd.concat([tables["ratings_train"], tables["ratings_test"]]).query("user_id == 1")
    user1 = user1.sort_values("timestamp")
    assert user1["user_rating_rank"].tolist() == [1, 2, 3]
    assert user1["days_since_user_first_rating"].iloc[0] == 0


def test_user_and_movie_encodings(built):
    _, tables, _, _ = built
    users = tables["users"].set_index("user_id")
    assert users.loc[1, "is_female"] == 1 and users.loc[2, "is_female"] == 0
    assert users.loc[1, "age_under_18"] == 1 and users.loc[2, "age_56_plus"] == 1
    assert users.loc[1, "zip_region"] == 4
    # User 1: Drama+Comedy rated 5 (mean 4) -> positive affinity; Action rated 3 -> negative.
    assert users.loc[1, "share_drama"] == pytest.approx(0.5)
    assert users.loc[1, "affinity_drama"] > 0 > users.loc[1, "affinity_action"]
    # One Drama rating at +1 vs the user mean, shrunk with 5 pseudo-ratings: 1 / (1 + 5).
    assert users.loc[1, "affinity_drama"] == pytest.approx(1 / 6, abs=1e-6)

    movies = tables["movies"].set_index("movie_id")
    assert movies.loc[10, ["genre_drama", "genre_comedy", "genre_action"]].tolist() == [1, 1, 0]
    assert movies.loc[30, "release_year_missing"] == 1
    assert movies.loc[30, "release_year"] == 1995  # median of 1990, 1995, 2000
    assert movies.loc[10, "release_decade"] == 1990
    assert movies["popularity_norm"].between(0, 1).all()


def test_is_deterministic(tmp_path, built):
    _, _, out, (train, test) = built
    build_features(train, test, tmp_path / "second")
    for path in out.iterdir():
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()


def test_missing_required_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("movie_id,rating\n1,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="user_id"):
        build_features(bad, bad, tmp_path / "out")


def test_chronological_split_matches_notebook_rule():
    ratings = pd.DataFrame(
        {
            "user_id": [1] * 10,
            "movie_id": range(10),
            "rating": [4] * 10,
            "timestamp": range(100, 110),
        }
    )
    train, test, cutoff = chronological_split(ratings, test_fraction=0.2)
    assert cutoff == 108
    assert train["timestamp"].max() < cutoff <= test["timestamp"].min()
    assert (len(train), len(test)) == (8, 2)
    with pytest.raises(ValueError):
        chronological_split(ratings, test_fraction=1.5)
