import pytest

pytest.importorskip("pandas")
import pandas as pd

pytest.importorskip("torch")
from hyperion3.models.rl_agents.sac import TradingEnvironmentSAC


def test_environment_metrics():
    df = pd.DataFrame(
        {
            "close": [1, 2, 3, 4, 5],
            "feat": [0, 0, 0, 0, 0],
        }
    )
    env = TradingEnvironmentSAC(df, feature_columns=["feat"], lookback_window=1)
    env.reset()
    for _ in range(len(df) - 1):
        env.step([0.0])
    metrics = env.get_performance_metrics()
    assert "sharpe_ratio" in metrics
