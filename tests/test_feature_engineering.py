import pytest

pytest.importorskip("pandas")
import pandas as pd
from data.feature_engineering import (
    add_moving_average,
    add_rsi,
    add_atr,
    add_market_regime,
)


def test_indicators_create_columns():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
        }
    )
    assert "ma_2" in add_moving_average(df, window=2).columns
    assert "rsi_2" in add_rsi(df, window=2).columns
    assert "atr_2" in add_atr(df, window=2).columns
    assert "regime_2" in add_market_regime(df, window=2).columns
