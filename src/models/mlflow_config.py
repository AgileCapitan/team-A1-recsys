"""MLflow configuration for local, reproducible experiments."""

import os
from pathlib import Path
from typing import Any

import mlflow
import yaml

MODEL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_DIR.parents[1]
PARAMS_PATH = MODEL_DIR / "params.yaml"


def load_params(params_path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load model parameters from a YAML file."""
    with params_path.open(encoding="utf-8") as params_file:
        params = yaml.safe_load(params_file)

    if not isinstance(params, dict):
        raise ValueError(f"Expected a mapping in {params_path}")

    return params


def configure_mlflow(params: dict[str, Any] | None = None) -> str:
    """Configure the local MLflow tracking URI and experiment."""
    params = params or load_params()
    mlflow_params = params.get("mlflow")

    if not isinstance(mlflow_params, dict):
        raise ValueError("The 'mlflow' section is required in params.yaml")

    tracking_uri = mlflow_params.get("tracking_uri")
    experiment_name = mlflow_params.get("experiment_name")
    if not isinstance(tracking_uri, str) or not tracking_uri:
        raise ValueError("The MLflow tracking URI must be a non-empty string")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("The MLflow experiment name must be a non-empty string")

    if tracking_uri.startswith("file:./"):
        tracking_path = REPOSITORY_ROOT / tracking_uri.removeprefix("file:./")
        tracking_uri = tracking_path.as_uri()
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri
