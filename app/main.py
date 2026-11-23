"""
Streamlit Frontend Application for Movie Recommendation System (US7.1 & US7.2).
Connects to the FastAPI recommendation backend service via HTTP.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

try:
    from app.api_client import APIError, RecommenderAPIClient
except ModuleNotFoundError:  # pragma: no cover - when launched directly from app/
    from api_client import APIError, RecommenderAPIClient

BACKEND_URL = os.getenv("API_URL", "http://localhost:8000")
RATINGS_KEY = "onboarding_saved_ratings"
MAX_ONBOARDING_RATINGS = 10
MIN_ONBOARDING_RATINGS = 3


def reset_onboarding_session() -> None:
    """Clear all onboarding session state keys."""
    for key in list(st.session_state):
        if key.startswith("onboarding_"):
            del st.session_state[key]


def render_movie_card(
    rank: int,
    movie: dict[str, Any],
    client: RecommenderAPIClient,
) -> None:
    """Render a recommendation card with metadata, score, and optional poster."""
    poster_bytes = None
    if movie.get("has_poster") and movie.get("movie_id"):
        poster_bytes = client.get_poster_bytes(int(movie["movie_id"]))

    if poster_bytes:
        image_column, details = st.columns([1, 4])
        image_column.image(poster_bytes, width=110)
    else:
        details = st.container()

    with details:
        st.subheader(f"{rank}. {movie.get('title', 'Unknown Title')}")
        metadata: list[str] = []
        if movie.get("release_year"):
            metadata.append(str(movie["release_year"]))
        genres = movie.get("genres") or []
        metadata.extend(genres)
        if metadata:
            st.caption(" · ".join(metadata))
        st.write(
            f"Movie ID: `{movie.get('movie_id')}` · "
            f"Predicted score: ⭐ `{float(movie.get('score', 0.0)):.2f}`"
        )


def render_sidebar(default_url: str) -> tuple[RecommenderAPIClient, dict[str, Any] | None]:
    """Render sidebar configuration and live backend health status."""
    with st.sidebar:
        st.header("🚀 Deployment Status")
        st.info(
            "Environment: **Oracle Cloud Production**\n\nDeployment: **Automated via GitHub Actions**"
        )

        st.markdown("---")
        st.header("⚙️ Settings")
        api_url = st.text_input("Backend API URL", value=default_url)
        client = RecommenderAPIClient(api_url)

        st.markdown("---")
        st.header("🔌 Service Health")
        health_data: dict[str, Any] | None = None
        try:
            health_data = client.get_health()
            status_label = health_data.get("status", "unknown")
            model_version = health_data.get("model_version", "unknown")
            using_fallback = bool(health_data.get("using_fallback", False))
            catalog_loaded = bool(health_data.get("catalog_loaded", False))

            if using_fallback:
                st.warning(f"Backend: Connected ({status_label} — fallback mode)")
                reason = health_data.get("fallback_reason")
                if reason:
                    st.caption(f"Fallback reason: `{reason}`")
            else:
                st.success(f"Backend: Connected ({status_label})")

            st.caption(f"Model Version: `{model_version}`")
            st.caption(f"Catalog Loaded: `{catalog_loaded}`")
        except APIError as exc:
            st.error("Backend: Disconnected")
            st.caption(exc.detail)

    return client, health_data


def render_new_user_tab(
    client: RecommenderAPIClient,
    available_genres: list[str],
    catalog_error: str | None,
) -> None:
    """Render the interactive cold-start onboarding flow backed by FastAPI."""
    st.subheader("Rate a few movies for personalized recommendations")
    st.write("Choose genres you enjoy and rate 3 to 10 movies you’ve seen.")

    if not available_genres:
        st.warning(
            catalog_error
            or "Movie catalog is not currently available on the backend. "
            "Use the Existing User ID tab in the meantime."
        )
        return

    st.button("Start over", on_click=reset_onboarding_session)
    saved_ratings: dict[int, dict[str, Any]] = st.session_state.setdefault(RATINGS_KEY, {})

    genres = st.multiselect(
        "Choose your favorite genres",
        options=available_genres,
        key="onboarding_genres",
    )
    st.caption(f"Movies rated this session · {len(saved_ratings)}/{MAX_ONBOARDING_RATINGS}")

    if saved_ratings:
        with st.expander("Your ratings"):
            for entry in saved_ratings.values():
                st.write(f"{entry['title']} · {entry['rating']}/5")

    if not genres:
        return

    remaining_slots = MAX_ONBOARDING_RATINGS - len(saved_ratings)
    if remaining_slots > 0:
        try:
            options_resp = client.get_rating_options(genres=genres, per_genre=200)
            all_options = options_resp.get("movies", [])
        except APIError as exc:
            st.error(f"Could not load movie options: {exc.detail}")
            return

        catalog_by_id = {int(movie["movie_id"]): movie for movie in all_options}
        unrated_ids = [
            int(movie["movie_id"])
            for movie in all_options
            if int(movie["movie_id"]) not in saved_ratings
        ]

        if unrated_ids:
            batch_key = f"{len(saved_ratings)}:" + "|".join(sorted(genres))
            selected_ids = st.multiselect(
                f"Choose up to {remaining_slots} movie(s) to rate",
                options=unrated_ids,
                format_func=lambda mid: catalog_by_id[mid]["title"],
                max_selections=remaining_slots,
                key=f"onboarding_movies:{batch_key}",
            )

            if selected_ids:
                with st.form(f"onboarding_form:{batch_key}"):
                    st.caption("1 means you disliked it. 5 means you loved it.")
                    new_ratings: dict[int, dict[str, Any]] = {}

                    for mid in selected_ids:
                        title = catalog_by_id[mid]["title"]
                        rating = st.selectbox(
                            title,
                            options=[1, 2, 3, 4, 5],
                            index=None,
                            placeholder="Choose a rating",
                            key=f"onboarding_rating:{batch_key}:{mid}",
                        )
                        if rating is not None:
                            new_ratings[mid] = {"title": title, "rating": int(rating)}

                    submitted = st.form_submit_button("Save ratings", type="primary")

                if submitted:
                    if len(new_ratings) != len(selected_ids):
                        st.warning("Give each selected movie a rating from 1 to 5.")
                    else:
                        st.session_state[RATINGS_KEY] = {**saved_ratings, **new_ratings}
                        st.rerun()
        else:
            st.info("You’ve rated every available movie in these genres.")
    else:
        st.info(f"Maximum of {MAX_ONBOARDING_RATINGS} onboarding ratings reached.")

    if len(saved_ratings) < MIN_ONBOARDING_RATINGS:
        remaining = MIN_ONBOARDING_RATINGS - len(saved_ratings)
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

    ratings_payload = {
        int(movie_id): int(entry["rating"]) for movie_id, entry in saved_ratings.items()
    }
    try:
        rec_resp = client.recommend_new_user(
            genres=genres,
            ratings=ratings_payload,
            top_k=int(top_k),
        )
    except APIError as exc:
        st.warning(exc.detail)
        return

    results = rec_resp.get("recommendations", [])
    if not results:
        st.info("No unrated movies match your selected genres.")
        return

    st.header("Recommended for you")
    st.caption(f"Model · `{rec_resp.get('model_version', 'unknown')}`")
    for rank, movie in enumerate(results, start=1):
        render_movie_card(rank, movie, client)


def render_existing_user_tab(
    client: RecommenderAPIClient,
    available_genres: list[str],
) -> None:
    """Render the existing-user recommendation flow backed by POST /recommend."""
    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.subheader("User Preferences")
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=100000,
            value=1,
            step=1,
            help="Enter a numeric user ID for personalized or popularity-fallback recommendations",
        )

        top_k = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="existing_user_top_k",
        )

        preferred_genres = st.multiselect(
            "Filter by Genre (Optional)",
            options=available_genres,
            default=[],
            disabled=not available_genres,
            key="existing_user_genres",
        )

        get_recommendations = st.button(
            "✨ Get Recommendations", type="primary", use_container_width=True
        )

    with col_results:
        st.subheader("Recommended Movies")
        if get_recommendations:
            with st.spinner("Fetching recommendations..."):
                try:
                    data = client.recommend_for_user(
                        user_id=int(user_id),
                        top_k=int(top_k),
                        genres=preferred_genres or None,
                    )
                except APIError as exc:
                    st.error(f"API Error: {exc.detail}")
                    return

                recommendations = data.get("recommendations", [])
                model_version = data.get("model_version", "unknown")
                using_fallback = bool(data.get("using_fallback", False))

                if using_fallback:
                    st.warning(
                        f"Generated {len(recommendations)} fallback recommendations "
                        f"(Model: `{model_version}`)"
                    )
                else:
                    st.success(
                        f"Generated {len(recommendations)} recommendations (Model: `{model_version}`)"
                    )

                for rank, movie in enumerate(recommendations, start=1):
                    render_movie_card(rank, movie, client)
                    st.divider()
        else:
            st.info(
                "Select a user ID and click **Get Recommendations** to view movie picks from the API."
            )


def main() -> None:
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 Movie Recommendation System")
    st.caption("AI-powered personalized movie recommendations • FastAPI + Streamlit")

    client, _ = render_sidebar(BACKEND_URL)

    available_genres: list[str] = []
    catalog_error: str | None = None
    try:
        genre_data = client.get_genres()
        available_genres = genre_data.get("genres", [])
    except APIError as exc:
        catalog_error = exc.detail

    tab_new_user, tab_existing_user = st.tabs(["🌟 New User Onboarding", "👤 Existing User ID"])

    with tab_new_user:
        render_new_user_tab(client, available_genres, catalog_error)

    with tab_existing_user:
        render_existing_user_tab(client, available_genres)


if __name__ == "__main__":
    main()
