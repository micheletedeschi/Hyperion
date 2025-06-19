import os, sys
import pytest

pytest.importorskip("yaml")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tempfile
from config.base_config import HyperionV2Config
from config.yaml_utils import save_config, load_config


def test_save_and_load_config(tmp_path):
    cfg = HyperionV2Config()
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(str(path))
    assert loaded.model_type == cfg.model_type
    assert loaded.data.symbols == cfg.data.symbols
