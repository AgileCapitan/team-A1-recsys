"""
Streamlit Frontend Application for Movie Recommendation System (US7.1 & US7.2).
Connects to the FastAPI recommendation backend service.
"""

import os

import requests
import streamlit as st

# Backend API URL configuration (can be overridden via environment variable)
BACKEND_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Movie Recommendation System")
st.caption("AI-powered personalized movie recommendations • 🚀 Auto-Deployed via GitHub Actions")
st.success("✨ Live Deployment Verified: CI/CD Pipeline Operational on Oracle Cloud VM!")

# Sidebar: Configuration and Service Status
with st.sidebar:
    st.header("🚀 CI/CD Status")
    st.info(
        "Environment: **Oracle Cloud Production**\n\nDeployment: **Automated via GitHub Actions**"
    )

    st.markdown("---")
    st.header("⚙️ Settings")
    api_url = st.text_input("Backend API URL", value=BACKEND_URL)

    st.markdown("---")
    st.header("🔌 Service Health")
    try:
        health_resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success(f"Backend: Connected ({health_data.get('status', 'ok')})")
            st.caption(f"Model Version: `{health_data.get('model_version', 'unknown')}`")
        else:
            st.warning(f"Backend responded with status {health_resp.status_code}")
    except Exception:
        st.error("Backend: Disconnected")
        st.caption(f"Could not reach {api_url}")

# Main Layout
col_input, col_results = st.columns([1, 2])

with col_input:
    st.subheader("Your Preferences")
    user_id = st.number_input(
        "User ID",
        min_value=1,
        max_value=100000,
        value=1,
        step=1,
        help="Enter the numeric user ID for personalized recommendations",
    )

    top_k = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    # Genre & Mood filters (US7.1 baseline)
    preferred_genres = st.multiselect(
        "Filter by Genre (Optional)",
        options=[
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Drama",
            "Sci-Fi",
            "Thriller",
        ],
        default=[],
    )

    get_recommendations = st.button(
        "✨ Get Recommendations", type="primary", use_container_width=True
    )

with col_results:
    st.subheader("Recommended Movies")
    if get_recommendations:
        with st.spinner("Fetching recommendations..."):
            try:
                payload = {"user_id": int(user_id), "top_k": int(top_k)}
                endpoint = f"{api_url.rstrip('/')}/recommend"
                response = requests.post(endpoint, json=payload, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    model_version = data.get("model_version", "unknown")

                    st.success(
                        f"Generated {len(recommendations)} recommendations (Model: `{model_version}`)"
                    )

                    for idx, movie in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(
                                f"### {idx}. {movie.get('title', 'Unknown Title')}\n"
                                f"- **Movie ID**: `{movie.get('movie_id')}`\n"
                                f"- **Score**: ⭐ `{movie.get('score', 0.0):.2f}`"
                            )
                            st.divider()
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Unable to connect to backend at `{api_url}`. "
                    "Please verify the backend container is running and healthy."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.info(
            "Select your user ID and click **Get Recommendations** to view personalized movie picks."
        )
