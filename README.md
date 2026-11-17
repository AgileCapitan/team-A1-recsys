# team-A1-recsys

ML-based movie recommender system (Agile + MLOps).

## 🌐 Live Cloud Deployment (US7.2)

The application is deployed on an Oracle Cloud Infrastructure VM instance and orchestrated via Docker Compose with automated GitHub Actions CI/CD.

| Service | Access URL | Description |
| :--- | :--- | :--- |
| **Streamlit Web UI** | [http://92.5.11.34:8501](http://92.5.11.34:8501) | Interactive frontend for movie recommendations |
| **FastAPI Swagger Docs** | [http://92.5.11.34:8000/docs](http://92.5.11.34:8000/docs) | Interactive API documentation & OpenAPI schema |
| **Backend Health Check** | [http://92.5.11.34:8000/health](http://92.5.11.34:8000/health) | Live service status & loaded model version |

### Deployment Architecture
- **Host Provider**: Oracle Cloud Infrastructure (Always Free VM)
- **Public IP**: `92.5.11.34`
- **Orchestration**: Docker Compose (`backend` + `frontend` multi-container stack)
- **CI/CD Pipeline**: GitHub Actions auto-deployment on merge to `main` (`.github/workflows/deploy.yml`)

---

## Structure

- data/ : raw, interim, processed (DVC)
- notebooks/ : exploration
- src/ : data, models, serving
- app/ : Streamlit frontend
- api/ : FastAPI endpoints
- mlruns/ : MLflow local tracking

---

## Local Multi-Container Setup (Docker Compose)

The backend (FastAPI) and frontend (Streamlit) services can be orchestrated using Docker Compose.

### Quickstart

1. (Optional) Copy `.env.example` to `.env` to customize ports:
   ```bash
   cp .env.example .env
   ```

2. Build and start both services:
   ```bash
   docker compose up -d --build
   ```

3. Access the services locally:
   - **Frontend (Streamlit)**: [http://localhost:8501](http://localhost:8501)
   - **Backend API (FastAPI)**: [http://localhost:8000](http://localhost:8000)
   - **API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Checks**:
     - Backend: `curl http://localhost:8000/health`
     - Frontend: `curl http://localhost:8501/_stcore/health`

4. Stop the services:
   ```bash
   docker compose down
   ```

### RustFS object storage

The Compose stack includes RustFS, an S3-compatible object store for processed
training data. Copy the example environment file before starting the stack so
credentials are supplied externally:

```bash
cp .env.example .env
docker compose up -d
```

RustFS is available at `http://localhost:9000` (console:
`http://localhost:9001`). The `rustfs-init` service creates the `${RUSTFS_BUCKET}`
bucket idempotently. Objects are stored under these versioned keys:

| Object | Key |
| :--- | :--- |
| Ratings CSV | `processed/movielens-1m/v1/ratings.csv` |
| Manifest | `processed/movielens-1m/v1/manifest.json` |

The CSV schema is `user_id,movie_id,rating,title`; the manifest records the
object key, SHA-256 checksum, byte size, row count, and schema. Publish the
manifest last so consumers never treat an incomplete upload as a release.

Run training directly from RustFS with:

```bash
python -m src.models.batch_recommend train \
  --data s3://recsys/processed/movielens-1m/v1/ratings.csv \
  --artifact artifacts/movielens-v1.joblib
```

For a host-side training run, set `RUSTFS_ENDPOINT=http://localhost:9000`,
`RUSTFS_ACCESS_KEY`, `RUSTFS_SECRET_KEY`, and `RUSTFS_REGION` in `.env` or the
shell environment. The local `--data compressed_data.csv.gz` path remains
supported.

To verify persistence, upload a small object, restart only RustFS, and read it
back:

```bash
docker compose run --rm --entrypoint /bin/sh rustfs-init -c \
  'printf persistence-check > /tmp/persistence-check.txt &&
   aws s3 cp /tmp/persistence-check.txt
   s3://$RUSTFS_BUCKET/persistence-check.txt --endpoint-url "$RUSTFS_ENDPOINT"'
docker compose restart rustfs
docker compose run --rm --entrypoint /bin/sh rustfs-init -c \
  'aws s3 cp s3://$RUSTFS_BUCKET/persistence-check.txt -
   --endpoint-url "$RUSTFS_ENDPOINT"'
```

The named `rustfs-data` volume is intentionally not removed by
`docker compose down`; use `docker compose down -v` only when resetting local
object storage.

## Data preparation pipeline (US3.2, DVC)

The raw dataset `compressed_data.csv.gz` is cleaned, split and turned into
model-ready feature tables by a three-stage DVC pipeline defined in `dvc.yaml`:

| Stage | Command | Output |
| :--- | :--- | :--- |
| `clean` | `python -m src.data.cleaner` | `data/processed/ratings_clean.csv` + `cleaning_report.json` |
| `split` | `python -m src.data.split` | `data/processed/split/{train,test}.csv` + `report.json` |
| `features` | `python -m src.data.features` | `data/processed/features/{ratings_train,ratings_test,users,movies}.csv` + `report.json` |

```bash
python -m pip install -r requirements.txt
dvc repro          # runs only the stages whose inputs changed
dvc metrics show   # cleaning report (rows read/written, invalid rows...)
dvc push           # store outputs in the local remote (../dvc-storage)
```

**Cleaning** keeps rows with positive user/movie IDs and an integer rating from
1 to 5, clears malformed or implausible optional integers (e.g. negative or
future timestamps), trims and normalizes text and genres, removes exact
duplicates, keeps only the most recent rating when a user re-rates a movie, and
harmonizes conflicting user/movie metadata with their latest row. The input
file is never modified.

**Split**: chronological, same rule as the evaluation notebook — the cutoff is
the timestamp at the 80% quantile; earlier ratings are train, later ones test.

**Features** — every statistic is fitted on **train only** (no test leakage);
cold-start users/movies get `*_idx = -1` and global-mean fallbacks:
- `ratings_{train,test}.csv`: contiguous `user_idx`/`movie_idx` (matrix rows for
  MF), `rating_norm` (0–1), `rating_centered` (minus the train user mean),
  `rated_year`/`rated_month`/`rated_weekday`, `movie_age_at_rating`, and
  past-only sequential features `user_rating_rank`, `days_since_user_first_rating`.
- `users.csv`: rating count/mean/std, `active_days`, `is_female`, one-hot age
  buckets, occupation, `zip_region` (first ZIP digit), log-normalized activity,
  and per-genre `share_<genre>` (fraction of the user's ratings) and
  `affinity_<genre>` (mean rating minus the user's own mean).
- `movies.csv`: multi-hot `genre_*`, release year (median-imputed with a
  `release_year_missing` flag), `release_year_norm`, `release_decade`, rating
  count/mean/std, `bayesian_rating` (mean shrunk toward the global mean),
  log-normalized popularity.

Outputs are deterministic and ignored by Git (tracked by `dvc.lock`).
Tests: `pytest tests/test_cleaner.py tests/test_features.py`.

## Train and export batch recommendations

```bash
python -m pip install -r requirements.txt

python -m src.models.batch_recommend train \
  --data compressed_data.csv.gz \
  --artifact artifacts/movielens-v1.joblib

python -m src.models.batch_recommend batch \
  --artifact artifacts/movielens-v1.joblib \
  --output data/processed/recommendations_popularity.csv \
  --method popularity --n 10

python -m src.models.batch_recommend batch \
  --artifact artifacts/movielens-v1.joblib \
  --output data/processed/recommendations_mf.csv \
  --method mf --n 10
```

Each CSV contains `user_id,rank,movie_id,title,score`, with up to N ranked movies
per user. The export preserves original MovieLens IDs and excludes movies the
user has already rated. Popularity scores are rating counts. Matrix factorization
scores are raw predicted scores and may fall outside the 1–5 rating scale.

On the bundled dataset, each export at N=10 contains 60,400 rows for 6,040 users.
The chronological holdout in `movieRecommender.ipynb` reports NDCG@10 of 0.2144
for popularity and 0.1394 for matrix factorization. The batch command defaults
to popularity based on that comparison.

## Model configuration

Model parameters are stored in `src/models/params.yaml`. After installing the
dependencies in `requirements.txt`, configure local MLflow tracking with:

```python
from src.models.mlflow_config import configure_mlflow

configure_mlflow()
```

This uses the repository-level `mlruns/` directory and the
`movie-recommender` experiment.
## Data Profiling & Exploratory Analysis

Data schema validation, missing-value checks, demographic–genre analysis, and temporal drift monitoring are documented in [`notebooks/dataProfiling.ipynb`](notebooks/dataProfiling.ipynb).

- **Dataset & Structure:** Profiles the MovieLens-1M dataset (`6,040` users, zero duplicate `(user_id, movie_id)` pairs) using the cleaned interim table at `data/interim/movielens_1m_clean.csv` alongside `compressed_data.csv.gz`.
- **Key Analytical Findings:**
  - **Rating & Activity Skew:** Ratings skew toward positive scores (`3` and `4`), while user activity and movie popularity follow long-tail distributions.
  - **Demographic Segmentation:** Age brackets and gender exhibit distinct genre consumption and rating patterns (e.g., higher `Action`/`Sci-Fi` shares in younger cohorts vs. higher `Drama`/`Romance` shares in older cohorts).
  - **Temporal Stability:** Chronological split comparison shows low rating drift (`PSI < 0.10`) between early and recent interactions.
- **Batch Output Verification:** Confirms that `data/processed/recommendations_popularity.csv` and `data/processed/recommendations_mf.csv` each contain `60,400` rows (`6,040` users $\times$ `10` items) with zero seen-movie leakage.
- **Interactive Dashboard View:** Run `streamlit run app/main.py` and scroll to the **Data Profiling & Quality Dashboard (US1)** section to view live dataset metrics, rating/genre charts, PSI drift status, and any exported HTML profile report (`reports/report.html`).
