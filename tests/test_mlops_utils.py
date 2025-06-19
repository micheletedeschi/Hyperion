import pytest

pytest.importorskip("numpy")
import importlib

mlops = importlib.import_module("hyperion3.utils.mlops")
validate_metrics = mlops.validate_metrics


def test_validate_metrics_pass():
    metrics = {"sharpe_ratio": 1.0, "drawdown": -0.05}
    thresholds = {"sharpe_ratio": 0.5, "drawdown": -0.1}
    assert validate_metrics(metrics, thresholds)


def test_validate_metrics_fail():
    metrics = {"sharpe_ratio": 0.2, "drawdown": -0.2}
    thresholds = {"sharpe_ratio": 0.5, "drawdown": -0.1}
    assert not validate_metrics(metrics, thresholds)
