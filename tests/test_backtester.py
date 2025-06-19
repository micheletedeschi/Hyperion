import pytest

pytest.importorskip("pandas")
import pandas as pd

pytest.importorskip("yaml")
from config.base_config import HyperionV2Config
from evaluations.backtester import AdvancedBacktester


@pytest.mark.asyncio
async def test_backtester_run():
    df = pd.DataFrame(
        {"close": [1, 2, 3, 4]}, index=pd.date_range("2020-01-01", periods=4, freq="H")
    )
    backtester = AdvancedBacktester(HyperionV2Config())
    results = await backtester.run(None, df)
    assert "equity_curve" in results
    assert len(results["equity_curve"]) == 3


@pytest.mark.asyncio
async def test_backtester_trending_positive():
    df = pd.DataFrame(
        {"close": [1, 2, 3, 4, 5]},
        index=pd.date_range("2020-01-01", periods=5, freq="H"),
    )
    backtester = AdvancedBacktester(HyperionV2Config())
    results = await backtester.run(None, df)
    assert results["equity_curve"][-1] > 1.0


@pytest.mark.asyncio
async def test_backtester_returns_metrics():
    df = pd.DataFrame(
        {"close": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="H")
    )
    backtester = AdvancedBacktester(HyperionV2Config())
    results = await backtester.run(None, df)
    assert "metrics" in results
    assert "sharpe_ratio" in results["metrics"]
