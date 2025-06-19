import pytest

pytest.importorskip("pydantic")
from hyperion3.config.base_config import DataConfig


def test_data_config_fields():
    cfg = DataConfig()
    assert cfg.data_dir == "data"
    assert isinstance(cfg.symbols, list)
    assert cfg.lookback_window > 0
    assert 0 < cfg.train_split < 1
    assert 0 < cfg.val_split < 1
    assert 0 < cfg.test_split < 1
