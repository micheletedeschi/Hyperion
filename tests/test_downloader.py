import pytest

pytest.importorskip("pandas")
import pandas as pd

pytest.importorskip("yaml")
from config.base_config import HyperionV2Config

pytest.importorskip("ccxt", reason="DataDownloader requires ccxt")

from data.downloader import DataDownloader


def test_validate_data():
    config = HyperionV2Config()
    downloader = DataDownloader(config)
    df = pd.read_csv("data/SOL_USDT_20250610.csv")
    assert downloader.validate_data(df)
