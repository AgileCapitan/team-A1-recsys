from unittest.mock import patch

from src.models import mlflow_config

PARAMS = {
    "mlflow": {
        "experiment_name": "test-experiment",
        "tracking_uri": "file:./mlruns",
    }
}


def test_configure_mlflow_prefers_tracking_uri_environment_variable(monkeypatch):
    tracking_uri = "http://localhost:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    with (
        patch.object(mlflow_config.mlflow, "set_tracking_uri") as set_tracking_uri,
        patch.object(mlflow_config.mlflow, "set_experiment") as set_experiment,
    ):
        configured_uri = mlflow_config.configure_mlflow(PARAMS)

    assert configured_uri == tracking_uri
    set_tracking_uri.assert_called_once_with(tracking_uri)
    set_experiment.assert_called_once_with("test-experiment")


def test_configure_mlflow_uses_yaml_tracking_uri_by_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    expected_uri = (mlflow_config.REPOSITORY_ROOT / "mlruns").as_uri()

    with (
        patch.object(mlflow_config.mlflow, "set_tracking_uri") as set_tracking_uri,
        patch.object(mlflow_config.mlflow, "set_experiment") as set_experiment,
    ):
        configured_uri = mlflow_config.configure_mlflow(PARAMS)

    assert configured_uri == expected_uri
    set_tracking_uri.assert_called_once_with(expected_uri)
    set_experiment.assert_called_once_with("test-experiment")
