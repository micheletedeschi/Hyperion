import numpy as np
import pandas as pd
from typing import Any, Dict
import logging
from .metrics import FinancialMetrics

logger = logging.getLogger(__name__)


class AdvancedBacktester:
    """Minimal backtester used for evaluation tests."""

    def __init__(self, config: Any):
        self.config = config
        self.metrics = FinancialMetrics()

    async def run(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute a simple backtest and return performance data.

        Parameters
        ----------
        model : Any
            Model used for predictions. Currently unused.
        data : pd.DataFrame
            Price series containing either 'close' or 'close_norm' column.

        Returns
        -------
        Dict[str, Any]
            Dictionary with equity curve, returns and timestamps.
        """

        # Verificar si tenemos datos crudos o preprocesados
        if "close" in data.columns:
            price_col = "close"
        elif "close_norm" in data.columns:
            price_col = "close_norm"
            logger.warning("Using normalized close prices for backtesting")
        else:
            raise ValueError("Data must contain either 'close' or 'close_norm' column")

        returns = data[price_col].pct_change().dropna().values
        equity_curve = np.cumprod(1 + returns)
        results = {
            "equity_curve": equity_curve,
            "returns": returns,
            "timestamps": data.index[1:].tolist(),
            "trades": [],
            "data": data,
        }

        # Calculate performance metrics
        try:
            results["metrics"] = self.metrics.calculate_all(results)
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            results["metrics"] = {}

        return results


__all__ = ["AdvancedBacktester"]
