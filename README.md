# team-A1-recsys

ML-based movie recommender system (Agile + MLOps).

## Structure

- data/ : raw, interim, processed (DVC)
- notebooks/ : exploration
- src/ : data, models, serving
- app/ : Streamlit frontend
- api/ : FastAPI endpoints
- mlruns/ : MLflow local tracking

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

3. Access the services:
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

