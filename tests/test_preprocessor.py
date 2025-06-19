import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
import numpy as np
import pandas as pd

pytest.importorskip("yaml")
from config.base_config import HyperionV2Config
from data.preprocessor import DataPreprocessor


def test_preprocessor_scaling_uses_training_data():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6]})
    config = HyperionV2Config()
    pre = DataPreprocessor(config)
    train = df.iloc[:3]
    test = df.iloc[3:]
    pre.fit(train)
    transformed = pre.transform(test)
    mean = train["close"].mean()
    std = train["close"].std()
    expected = (test["close"] - mean) / std
    assert np.allclose(transformed["close_norm"], expected)


def test_preprocessor_retains_ohlcv_columns():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10, 20, 30],
        }
    )
    config = HyperionV2Config()
    pre = DataPreprocessor(config)
    out = pre.fit_transform(df)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in out.columns
