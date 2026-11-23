import hashlib
import json
from pathlib import Path

import streamlit as st

from src.serving.new_user import NewUserRecommender

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
RATINGS_KEY = "onboarding_saved_ratings"
USERS_FILE = ARTIFACT_DIR / "users.json"

st.set_page_config(page_title="Movie Recommender", layout="wide")


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _sync_current_user() -> None:
    """Persist the active user's genres and ratings whenever the page reruns."""
    username = st.session_state.get("auth_username")
    if not username:
        return
    users = _load_users()
    if username not in users:
        return

    raw_ratings = st.session_state.get(RATINGS_KEY, {})
    users[username]["saved_ratings"] = {str(k): v for k, v in raw_ratings.items()}
    if "onboarding_genres" in st.session_state:
        users[username]["genres"] = list(st.session_state["onboarding_genres"])
    _save_users(users)


def render_auth_page() -> bool:
    """Render Login / Create Account screen before entering the recommender."""
    username = st.session_state.get("auth_username")
    if username:
        _sync_current_user()
        with st.sidebar:
            st.success(f"Signed in as **{username}**")
            if st.button("Log Out", use_container_width=True):
                reset_session()
                del st.session_state["auth_username"]
                st.rerun()
        return True

    st.title("Movie Recommender")
    st.write("Sign in or create an account to continue.")

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        tab_login, tab_register = st.tabs(["Log In", "Create Account"])

        with tab_login:
            with st.form("user_login_form"):
                login_user = st.text_input("Username")
                login_pass = st.text_input("Password", type="password")
                login_submit = st.form_submit_button(
                    "Log In", type="primary", use_container_width=True
                )

            if login_submit:
                users = _load_users()
                record = users.get(login_user.strip())
                if not record or record["password"] != _hash_password(login_pass):
                    st.error("Invalid username or password.")
                else:
                    st.session_state["auth_username"] = login_user.strip()
                    st.session_state[RATINGS_KEY] = {
                        int(k): v for k, v in record.get("saved_ratings", {}).items()
                    }
                    if record.get("genres"):
                        st.session_state["onboarding_genres"] = record["genres"]
                    st.rerun()

        with tab_register:
            with st.form("user_register_form"):
                new_user = st.text_input("Choose a Username")
                new_pass = st.text_input("Choose a Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                reg_submit = st.form_submit_button(
                    "Create Account", type="primary", use_container_width=True
                )

            if reg_submit:
                clean_user = new_user.strip()
                users = _load_users()
                if len(clean_user) < 2:
                    st.warning("Username must be at least 2 characters.")
                elif len(new_pass) < 3:
                    st.warning("Password must be at least 3 characters.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif clean_user in users:
                    st.error("Username already exists. Please log in.")
                else:
                    users[clean_user] = {
                        "password": _hash_password(new_pass),
                        "genres": [],
                        "saved_ratings": {},
                    }
                    _save_users(users)
                    st.session_state["auth_username"] = clean_user
                    st.session_state[RATINGS_KEY] = {}
                    st.rerun()

    return False


@st.cache_resource
def load_recommender() -> NewUserRecommender:
    return NewUserRecommender(
        model_path=ARTIFACT_DIR / "movielens-v1.joblib",
        catalog_path=ARTIFACT_DIR / "movie-catalog.csv",
    )


def reset_session() -> None:
    for key in list(st.session_state):
        if key.startswith("onboarding_"):
            del st.session_state[key]


def render_movie(rank: int, movie: dict) -> None:
    poster = ARTIFACT_DIR / "posters" / f"{movie['movie_id']}.jpg"
    if poster.is_file():
        image_column, details = st.columns([1, 4])
        image_column.image(str(poster), width=110)
    else:
        details = st.container()

    with details:
        st.subheader(f"{rank}. {movie['title']}")
        metadata = list(movie["genres"])
        if movie["release_year"]:
            metadata.insert(0, str(movie["release_year"]))
        if metadata:
            st.caption(" · ".join(metadata))
        st.write(f"Predicted rating score · {movie['score']:.2f}")


def main() -> None:
    st.title("Movie Recommender")
    st.write("Choose genres you enjoy and rate movies you’ve seen.")

    try:
        recommender = load_recommender()
    except FileNotFoundError:
        st.error("Place movielens-v1.joblib and movie-catalog.csv in the artifacts directory.")
        return
    except (OSError, ValueError) as error:
        st.error(f"Could not load the recommender. {error}")
        return

    st.button("Start over", on_click=reset_session)
    saved_ratings = st.session_state.setdefault(RATINGS_KEY, {})

    genres = st.multiselect(
        "Choose your favorite genres",
        options=recommender.genres,
        key="onboarding_genres",
    )
    st.caption(f"Movies rated this session · {len(saved_ratings)}")

    if saved_ratings:
        with st.expander("Your ratings"):
            for movie_id, rating in saved_ratings.items():
                title = recommender.catalog[movie_id]["title"]
                st.write(f"{title} · {rating}/5")

    if not genres:
        return

    options = [
        movie_id
        for movie_id in recommender.rating_options(genres, per_genre=len(recommender.catalog))
        if movie_id not in saved_ratings
    ]

    if options:
        batch_key = f"{len(saved_ratings)}:" + "|".join(sorted(genres))
        movie_ids = st.multiselect(
            "Choose up to 10 movies to rate",
            options=options,
            format_func=lambda movie_id: recommender.catalog[movie_id]["title"],
            max_selections=10,
            key=f"onboarding_movies:{batch_key}",
        )

        if movie_ids:
            with st.form(f"onboarding_form:{batch_key}"):
                st.caption("1 means you disliked it. 5 means you loved it.")
                new_ratings = {}

                for movie_id in movie_ids:
                    rating = st.selectbox(
                        recommender.catalog[movie_id]["title"],
                        options=[1, 2, 3, 4, 5],
                        index=None,
                        placeholder="Choose a rating",
                        key=f"onboarding_rating:{batch_key}:{movie_id}",
                    )
                    if rating is not None:
                        new_ratings[movie_id] = rating

                submitted = st.form_submit_button("Save ratings", type="primary")

            if submitted:
                if len(new_ratings) != len(movie_ids):
                    st.warning("Give each selected movie a rating from 1 to 5.")
                else:
                    st.session_state[RATINGS_KEY] = {**saved_ratings, **new_ratings}
                    st.rerun()
    else:
        st.info("You’ve rated every available movie in these genres.")

    if len(saved_ratings) < 3:
        remaining = 3 - len(saved_ratings)
        label = "movie" if remaining == 1 else "movies"
        st.info(f"Rate {remaining} more {label} to unlock recommendations.")
        return

    top_k = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=5,
        key="onboarding_top_k",
    )

    try:
        results = recommender.recommend(
            genres=genres,
            ratings=saved_ratings,
            top_k=top_k,
        )
    except ValueError as error:
        st.warning(str(error))
        return

    if not results:
        st.info("No unrated movies match your selected genres.")
        return

    st.header("Recommended for you")
    st.caption(f"Model · {recommender.model_version}")
    for rank, movie in enumerate(results, start=1):
        render_movie(rank, movie)


if __name__ == "__main__":
    if render_auth_page():
        main()
