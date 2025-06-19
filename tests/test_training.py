import pytest

pytest.importorskip("pandas")
import pandas as pd

pytest.importorskip("yaml")
from config.base_config import HyperionV2Config

pytest.importorskip("torch", reason="ModelTrainer requires torch")

from training.trainer import ModelTrainer


def test_mock_backtest_returns_metrics():
    trainer = ModelTrainer(HyperionV2Config())
    data = pd.DataFrame(
        {
            "open": [1, 2, 3, 4],
            "high": [1, 2, 3, 4],
            "low": [1, 2, 3, 4],
            "close": [1, 2, 3, 4],
            "volume": [1, 1, 1, 1],
        }
    )
    metrics = trainer._mock_backtest(None, data)
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
