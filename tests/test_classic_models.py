import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("catboost")
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from hyperion3.optimization.flaml_optimizer import ModelWrapper


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df


def test_lightgbm_model(sample_data):
    X = sample_data.drop(columns=["target"])
    y = sample_data["target"]
    model = ModelWrapper(
        LGBMRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, verbose=-1),
        "lgbm",
        {},
    )
    model.model.fit(X, y)
    preds = model.model.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_xgboost_model(sample_data):
    X = sample_data.drop(columns=["target"])
    y = sample_data["target"]
    model = ModelWrapper(
        XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, verbosity=0),
        "xgboost",
        {},
    )
    model.model.fit(X, y)
    preds = model.model.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_catboost_model(sample_data):
    X = sample_data.drop(columns=["target"])
    y = sample_data["target"]
    model = ModelWrapper(
        CatBoostRegressor(iterations=50, learning_rate=0.1, depth=3, verbose=False),
        "catboost",
        {},
    )
    model.model.fit(X, y)
    preds = model.model.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()
