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

### Backblaze B2 object storage

Backblaze B2 is the shared S3-compatible object store. FastAPI and Streamlit
remain on the Oracle VM; the Compose stack does not run an object-storage
service. Copy the example environment file for local configuration:

```bash
cp .env.example .env
docker compose up -d
```

Shared objects use these prefixes:

| Object | Key |
| :--- | :--- |
| Fixtures | `test/noe/` |
| Raw MovieLens files | `raw/movielens-1m/` |
| Ratings CSV | `processed/movielens-1m/v1/ratings.csv` |
| Manifest | `processed/movielens-1m/v1/manifest.json` |
| Model artifacts | `models/` |

The `ratings.csv` schema is
`user_id,movie_id,rating,timestamp,title,genres,gender,age_code,occupation_code,zip_code`
(training reads `user_id,movie_id,rating,title`); `users.csv` and `movies.csv`
are published alongside it. The manifest records the source and output object
keys, SHA-256 checksums, byte sizes, row counts, columns, pipeline commit, and
UTC creation time. The manifest is published last so consumers never treat an
incomplete upload as a release.

GitHub Actions has the manually triggered **Check B2 Access** workflow. It uses
the repository's configured B2 secrets and variables to perform a read-only
bucket check, so teammates can verify access without personal keys. It does
not publish data or run training.

Run training directly from B2 on a laptop with:

```bash
python -m src.models.batch_recommend train \
  --data "s3://${B2_BUCKET}/processed/movielens-1m/v1/ratings.csv" \
  --artifact artifacts/movielens-v1.joblib
```

Laptop training requires local `B2_ENDPOINT`, `B2_REGION`, `B2_KEY_ID`, and
`B2_APPLICATION_KEY` credentials. The local `--data compressed_data.csv.gz`
path remains supported. Publishing and training through GitHub need separate
workflow steps with explicitly scoped B2 permissions; the access-check workflow
is intentionally read-only.

The legacy `rustfs-data` named volume is retained in Compose for data
preservation during the cutover, but RustFS is no longer deployed or used.

### Publishing the MovieLens 1M dataset (US3.3)

The conversion lives in `src/data/` and needs `B2_ENDPOINT`, `B2_REGION`,
`B2_BUCKET`, `B2_KEY_ID`, and `B2_APPLICATION_KEY` in the environment, with a
key that can write to the bucket. Put the original `users.dat`, `movies.dat`
and `ratings.dat` in `data/raw/ml-1m/`, then:

```bash
# 1. Write and read back a test object under test/noe/
python -m src.data.check_b2 roundtrip

# 2. Upload the raw files to raw/movielens-1m/ (existing objects are verified, never overwritten)
python -m src.data.upload_raw --source-dir data/raw/ml-1m

# 3. Dry run: read the raw objects, validate, and write CSVs + manifest locally
python -m src.data.prepare_movielens --output-dir data/processed/movielens-1m/v1

# 4. Publish to processed/movielens-1m/v1/ (manifest last), then download and verify
python -m src.data.prepare_movielens --output-dir data/processed/movielens-1m/v1 --publish

# 5. Re-verify the published objects against the manifest at any time
python -m src.data.check_b2 verify-dataset --prefix processed/movielens-1m/v1/
```

Validation fails the run, so nothing is uploaded, when a row has the wrong
number of `::` fields, an ID is not a positive integer, a rating is not an
integer from 1 to 5, a timestamp is missing, invalid, or in the future, a
user/movie ID or `(user_id, movie_id)` pair is duplicated, or a rating
references an unknown user or movie. Original MovieLens IDs and leading zeros
in ZIP codes are preserved, and the output is sorted, so the CSVs are
byte-for-byte reproducible. Publishing refuses to write into a prefix that
already holds a release: changed output goes to a new prefix such as
`processed/movielens-1m/v2/`. For a local run without storage access, use
`--source-dir data/raw/ml-1m`; it cannot be combined with `--publish`.

The tests in `tests/test_prepare_movielens.py` use an in-memory S3 client and
need no credentials.

### Staging serving artifacts to Oracle VM (US6.1-T3)

Versioned model artifacts are stored in B2 under `models/` and staged onto the
Oracle VM by GitHub Actions (`.github/workflows/deploy.yml`). The FastAPI
container reads only local files mounted read-only from `./artifacts` into
`/app/artifacts/` — B2 credentials never live on the VM.

1. Upload a validated model bundle (and optional catalog) to an immutable B2 key:

   ```bash
   python -m src.serving.stage_artifacts upload \
     --model artifacts/movielens-v1.joblib \
     --model-key models/movielens-v1/movielens-v1.joblib \
     --catalog artifacts/movie-catalog.csv \
     --catalog-key models/movielens-v1/movie-catalog.csv
   ```

2. Configure repository variables (`SERVING_MODEL_KEY`, optional
   `SERVING_CATALOG_KEY` defaulting to `processed/movielens-1m/v1/movies.csv`,
   optional `SERVING_MODEL_VERSION`, optional `SERVING_MLFLOW_RUN_ID`) or pass
   the equivalent inputs when running **Deploy to Cloud VM** via
   `workflow_dispatch`.
3. During deployment, the runner downloads and validates the files with
   `python -m src.serving.stage_artifacts validate`, writes `staged-version.json`
   with SHA-256 digests and version metadata, transfers them to the VM over SSH,
   atomically promotes them into `~/team-A1-recsys/artifacts/`, restarts Docker
   Compose, and verifies `/health` reports `artifact_loaded: true` and
   `using_fallback: false`.

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

By default this uses the repository-level `mlruns/` directory and the
`movie-recommender` experiment. When set, `MLFLOW_TRACKING_URI` overrides the
YAML `mlflow.tracking_uri`; otherwise the YAML local default is used. To use
the persistent local MLflow server instead, start its optional Compose service:

```bash
docker compose --profile mlflow up -d mlflow
```

Open the MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000). To point
the existing trainer at the server, set the tracking URI in the same shell and
run the usual training command:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m src.models.batch_recommend train \
  --data compressed_data.csv.gz \
  --artifact artifacts/movielens-v1.joblib
```

The Compose service binds only to localhost. Its `mlflow-data` named volume
stores both the SQLite tracking/registry database and model artifacts; Compose
keeps this volume when the service container is recreated or when `docker
compose down` is run. Do not use `docker compose down -v` if you want to retain
the MLflow history.

### Shared Oracle HTTPS routes

The Oracle deployment will use one HTTPS hostname with these routes:

| URL path | Service |
| :--- | :--- |
| `/` | Streamlit |
| `/api/` | FastAPI (`/api/docs` for Swagger) |
| `/mlflow/` | MLflow UI and tracking API (HTTP Basic Auth required) |

Caddy terminates HTTPS and routes traffic internally. Generate the bcrypt
value with the same Caddy image used by Compose:

```bash
docker run --rm -it caddy:2.10.2-alpine caddy hash-password
```

Save the generated hash as the repository Actions secret
`MLFLOW_AUTH_PASSWORD_HASH`. The deploy workflow writes the hash,
`API_ROOT_PATH=/api`, and `MLFLOW_STATIC_PREFIX=/mlflow` to the VM's ignored
`.env` before starting the `mlflow` Compose profile. The default login is
`mlflow`. Share the original password with teammates through the team's
password manager.
Training clients use the authenticated HTTPS tracking URI and credentials:

```bash
export MLFLOW_TRACKING_URI=https://team-a1-recsys.duckdns.org/mlflow
export MLFLOW_TRACKING_USERNAME=mlflow
export MLFLOW_TRACKING_PASSWORD='<team password>'
```


## Data Profiling & Exploratory Analysis

Data schema validation, missing-value checks, demographic–genre analysis, and temporal drift monitoring are documented in [`notebooks/dataProfiling.ipynb`](notebooks/dataProfiling.ipynb).

- **Dataset & Structure:** Profiles the MovieLens-1M dataset (`6,040` users, zero duplicate `(user_id, movie_id)` pairs) using the cleaned interim table at `data/interim/movielens_1m_clean.csv` alongside `compressed_data.csv.gz`.
- **Key Analytical Findings:**
  - **Rating & Activity Skew:** Ratings skew toward positive scores (`3` and `4`), while user activity and movie popularity follow long-tail distributions.
  - **Demographic Segmentation:** Age brackets and gender exhibit distinct genre consumption and rating patterns (e.g., higher `Action`/`Sci-Fi` shares in younger cohorts vs. higher `Drama`/`Romance` shares in older cohorts).
  - **Temporal Stability:** Chronological split comparison shows low rating drift (`PSI < 0.10`) between early and recent interactions.
- **Batch Output Verification:** Confirms that `data/processed/recommendations_popularity.csv` and `data/processed/recommendations_mf.csv` each contain `60,400` rows (`6,040` users $\times$ `10` items) with zero seen-movie leakage.
- **Interactive Dashboard View:** Run `streamlit run app/main.py` and scroll to the **Data Profiling & Quality Dashboard (US1)** section to view live dataset metrics, rating/genre charts, PSI drift status, and any exported HTML profile report (`reports/report.html`).
