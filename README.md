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
