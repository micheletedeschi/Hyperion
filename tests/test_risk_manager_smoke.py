import pytest

pytest.importorskip("pandas")
import pandas as pd

pytest.importorskip("websockets")
from deployment.risk_manager import RiskManager


def test_risk_manager_position_size():
    cfg = {}
    rm = RiskManager(cfg)
    data = pd.DataFrame({"close": [100, 101, 102, 103, 104], "volume": [10] * 5})
    result = rm.calculate_position_size(
        signal_strength=0.5,
        current_price=100,
        account_balance=1000,
        market_volatility=0.01,
        existing_positions=[],
        market_data=data,
    )
    assert result.position_size >= 0
