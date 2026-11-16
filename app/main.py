"""
Streamlit Frontend Application for Movie Recommendation System (US7.1 & US7.2).
Connects to the FastAPI recommendation backend service.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
from collections import Counter
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

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
# --- Data Profiling Dashboard Section (US1.1-T2) ---


@st.cache_data(show_spinner=False)
def load_profiling_summary() -> dict | None:
    """Compute lightweight schema, distribution, and drift metrics from local MovieLens data."""
    root_dir = Path(__file__).resolve().parent.parent
    interim_csv = root_dir / "data" / "interim" / "movielens_1m_clean.csv"
    gz_csv = root_dir / "compressed_data.csv.gz"

    if interim_csv.exists():
        source_name = "data/interim/movielens_1m_clean.csv"
        handle = interim_csv.open("r", encoding="utf-8", newline="")
    elif gz_csv.exists():
        source_name = "compressed_data.csv.gz"
        handle = gzip.open(gz_csv, "rt", encoding="utf-8", newline="")
    else:
        return None

    rating_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    users: set[str] = set()
    movies: set[str] = set()
    rows: list[tuple[int, str]] = []

    with handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        col_map = {c.strip().lower(): c for c in columns}
        u_col = col_map.get("user_id")
        m_col = col_map.get("movie_id")
        r_col = col_map.get("rating")
        g_col = col_map.get("genres") or col_map.get("genre")
        t_col = col_map.get("timestamp")

        for idx, row in enumerate(reader):
            if u_col and row.get(u_col):
                users.add(row[u_col])
            if m_col and row.get(m_col):
                movies.add(row[m_col])
            rating_val = row.get(r_col, "") if r_col else ""
            if rating_val:
                rating_counts[rating_val] += 1
            if g_col and row.get(g_col):
                for genre in row[g_col].split("|"):
                    if genre:
                        genre_counts[genre] += 1
            ts_val = int(row[t_col]) if (t_col and row.get(t_col, "").isdigit()) else idx
            rows.append((ts_val, rating_val))

    rows.sort(key=lambda item: item[0])
    mid = len(rows) // 2
    ref_counts = Counter(r for _, r in rows[:mid] if r)
    cur_counts = Counter(r for _, r in rows[mid:] if r)
    ref_total = max(sum(ref_counts.values()), 1)
    cur_total = max(sum(cur_counts.values()), 1)

    psi = 0.0
    for key in set(ref_counts) | set(cur_counts):
        ref_p = max(ref_counts[key] / ref_total, 1e-6)
        cur_p = max(cur_counts[key] / cur_total, 1e-6)
        psi += (cur_p - ref_p) * math.log(cur_p / ref_p)

    return {
        "source_name": source_name,
        "total_rows": len(rows),
        "unique_users": len(users),
        "unique_movies": len(movies),
        "rating_counts": dict(sorted(rating_counts.items())),
        "top_genres": dict(genre_counts.most_common(10)),
        "psi": psi,
    }


def render_data_profiling_section() -> None:
    """Render dataset quality metrics, distributions, drift alert, and HTML report."""
    st.markdown("---")
    st.header("📊 Data Profiling & Quality Dashboard (US1)")

    summary = load_profiling_summary()
    if summary:
        st.caption(f"Active Dataset Source: `{summary['source_name']}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Interactions", f"{summary['total_rows']:,}")
        m2.metric("Unique Users", f"{summary['unique_users']:,}")
        m3.metric("Unique Movies", f"{summary['unique_movies']:,}")

        psi_val = summary["psi"]
        drift_label = "No Drift" if psi_val < 0.1 else ("Moderate" if psi_val < 0.25 else "Severe")
        m4.metric("Rating Drift (PSI)", f"{psi_val:.4f}", drift_label)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Rating Distribution (1–5 Scale)**")
            st.bar_chart(summary["rating_counts"])
        with c2:
            st.markdown("**Top 10 Genres by Interaction Count**")
            st.bar_chart(summary["top_genres"])
    else:
        st.info(
            "Local dataset not found. Place `compressed_data.csv.gz` in the project root "
            "or `movielens_1m_clean.csv` in `data/interim/` to display live profiling charts."
        )

    root_dir = Path(__file__).resolve().parent.parent
    report_candidates = [
        root_dir / "reports" / "report.html",
        root_dir / "reports" / "data_profile_report.html",
    ]
    report_file = next((p for p in report_candidates if p.exists()), None)
    if report_file:
        st.subheader("📑 Embedded HTML Profiling Report")
        html_content = report_file.read_text(encoding="utf-8")
        components.html(html_content, height=650, scrolling=True)


render_data_profiling_section()
