"""
Financial Metrics Calculator for Hyperion V2
Comprehensive metrics for cryptocurrency trading evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

try:
    from scipy import stats  # heavy optional
except Exception:  # pragma: no cover
    stats = None
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class FinancialMetrics:
    """Calculate comprehensive financial metrics for trading evaluation"""

    def calculate_all(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all metrics from backtest results"""
        equity_curve = backtest_results["equity_curve"]
        returns = backtest_results["returns"]
        trades = backtest_results["trades"]

        metrics = {}

        # Returns metrics
        metrics["total_return"] = self._total_return(equity_curve)
        metrics["annual_return"] = self._annual_return(
            returns, backtest_results["timestamps"]
        )
        metrics["monthly_return"] = self._monthly_return(returns)

        # Risk metrics
        metrics["volatility"] = self._volatility(returns)
        metrics["sharpe_ratio"] = self._sharpe_ratio(returns)
        metrics["sortino_ratio"] = self._sortino_ratio(returns)
        metrics["calmar_ratio"] = self._calmar_ratio(returns, equity_curve)

        # Drawdown metrics
        dd_stats = self._drawdown_analysis(equity_curve)
        metrics.update(dd_stats)

        # Trade metrics
        trade_stats = self._trade_analysis(trades)
        metrics.update(trade_stats)

        # Risk-adjusted metrics
        metrics["information_ratio"] = self._information_ratio(returns)
        metrics["omega_ratio"] = self._omega_ratio(returns)

        # Additional metrics
        metrics["profit_factor"] = self._profit_factor(trades)
        metrics["expectancy"] = self._expectancy(trades)
        metrics["kelly_criterion"] = self._kelly_criterion(trades)

        # Crypto-specific metrics
        metrics["hodl_comparison"] = self._hodl_comparison(backtest_results)
        metrics["risk_adjusted_return"] = self._risk_adjusted_return(returns)

        return metrics

    def _total_return(self, equity_curve: np.ndarray) -> float:
        """Calculate total return"""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve[-1] / equity_curve[0]) - 1

    def _annual_return(self, returns: np.ndarray, timestamps: List) -> float:
        """Calculate annualized return"""
        if len(returns) < 2:
            return 0.0

        delta = timestamps[-1] - timestamps[0]
        if hasattr(delta, "days"):
            days = delta.days
        else:  # allow numeric indexes for light testing
            days = len(timestamps) - 1
        years = days / 365.25

        if years <= 0:
            return 0.0

        total_return = (1 + returns).prod() - 1
        return (1 + total_return) ** (1 / years) - 1

    def _monthly_return(self, returns: np.ndarray) -> float:
        """Calculate average monthly return"""
        if len(returns) < 21:
            return np.mean(returns) * 21
        # Assuming daily returns
        monthly_return = (1 + returns).prod() ** (21 / len(returns)) - 1
        return monthly_return

    def _volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)

    def _sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        std_dev = np.std(excess_returns)

        if std_dev == 0:
            return 0.0

        return np.sqrt(252) * np.mean(excess_returns) / std_dev

    def _sortino_ratio(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0

        downside_returns = returns[returns < target_return]

        if len(downside_returns) < 2:
            return 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        return np.sqrt(252) * (np.mean(returns) - target_return) / downside_std

    def _calmar_ratio(self, returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        annual_return = self._annual_return(returns, list(range(len(returns))))
        max_dd = self._max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        return -annual_return / max_dd

    def _drawdown_analysis(self, equity_curve: np.ndarray) -> Dict[str, float]:
        """Comprehensive drawdown analysis"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak

        return {
            "max_drawdown": np.min(drawdown),
            "avg_drawdown": np.mean(drawdown[drawdown < 0]) if any(drawdown < 0) else 0,
            "max_drawdown_duration": self._max_drawdown_duration(drawdown),
            "recovery_factor": self._recovery_factor(equity_curve, drawdown),
        }

    def _max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)

    def _max_drawdown_duration(self, drawdown: np.ndarray) -> int:
        """Calculate maximum drawdown duration in periods"""
        if not any(drawdown < 0):
            return 0

        in_drawdown = drawdown < 0
        starts = np.where(np.diff(np.concatenate(([False], in_drawdown))) == 1)[0]
        ends = np.where(np.diff(np.concatenate((in_drawdown, [False]))) == -1)[0]

        if len(starts) == 0:
            return 0

        durations = ends - starts
        return int(np.max(durations)) if len(durations) > 0 else 0

    def _recovery_factor(self, equity_curve: np.ndarray, drawdown: np.ndarray) -> float:
        """Calculate recovery factor"""
        total_return = self._total_return(equity_curve)
        max_dd = np.min(drawdown)

        if max_dd == 0:
            return 0.0

        return -total_return / max_dd

    def _trade_analysis(self, trades: List[Any]) -> Dict[str, float]:
        """Analyze trade statistics"""
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_trade_return": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_trade_duration": 0.0,
                "trades_per_day": 0.0,
            }

        pnls = [trade.pnl for trade in trades if trade.pnl is not None]

        if not pnls:
            return self._empty_trade_stats()

        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]

        # Calculate durations
        durations = []
        for trade in trades:
            if trade.exit_time and trade.entry_time:
                duration = (
                    trade.exit_time - trade.entry_time
                ).total_seconds() / 3600  # hours
                durations.append(duration)

        return {
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "avg_win": np.mean(wins) if wins else 0.0,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "avg_trade_return": np.mean(pnls) if pnls else 0.0,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
            "avg_trade_duration": np.mean(durations) if durations else 0.0,
            "trades_per_day": self._trades_per_day(trades),
        }

    def _empty_trade_stats(self) -> Dict[str, float]:
        """Return empty trade statistics"""
        return {
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_trade_return": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_trade_duration": 0.0,
            "trades_per_day": 0.0,
        }

    def _trades_per_day(self, trades: List[Any]) -> float:
        """Calculate average trades per day"""
        if not trades:
            return 0.0

        first_trade = min(trade.entry_time for trade in trades)
        last_trade = max(trade.exit_time or trade.entry_time for trade in trades)

        days = (last_trade - first_trade).days
        return len(trades) / days if days > 0 else 0.0

    def _information_ratio(
        self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None
    ) -> float:
        """Calculate Information Ratio"""
        if benchmark_returns is None:
            # Use 0 as benchmark (absolute returns)
            active_returns = returns
        else:
            active_returns = returns - benchmark_returns

        if len(active_returns) < 2:
            return 0.0

        tracking_error = np.std(active_returns)

        if tracking_error == 0:
            return 0.0

        return np.sqrt(252) * np.mean(active_returns) / tracking_error

    def _omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio"""
        if len(returns) < 2:
            return 0.0

        positive_returns = returns[returns > threshold] - threshold
        negative_returns = threshold - returns[returns < threshold]

        if len(negative_returns) == 0 or np.sum(negative_returns) == 0:
            return np.inf if len(positive_returns) > 0 else 0.0

        return np.sum(positive_returns) / np.sum(negative_returns)

    def _profit_factor(self, trades: List[Any]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0.0

        gross_profits = sum(
            trade.pnl for trade in trades if trade.pnl is not None and trade.pnl > 0
        )
        gross_losses = abs(
            sum(
                trade.pnl for trade in trades if trade.pnl is not None and trade.pnl < 0
            )
        )

        if gross_losses == 0:
            return np.inf if gross_profits > 0 else 0.0

        return gross_profits / gross_losses

    def _expectancy(self, trades: List[Any]) -> float:
        """Calculate trade expectancy"""
        if not trades:
            return 0.0

        pnls = [trade.pnl for trade in trades if trade.pnl is not None]

        if not pnls:
            return 0.0

        win_rate = len([pnl for pnl in pnls if pnl > 0]) / len(pnls)
        avg_win = (
            np.mean([pnl for pnl in pnls if pnl > 0])
            if any(pnl > 0 for pnl in pnls)
            else 0
        )
        avg_loss = (
            abs(np.mean([pnl for pnl in pnls if pnl < 0]))
            if any(pnl < 0 for pnl in pnls)
            else 0
        )

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _kelly_criterion(self, trades: List[Any]) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if not trades:
            return 0.0

        pnls = [trade.pnl for trade in trades if trade.pnl is not None]

        if not pnls:
            return 0.0

        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [abs(pnl) for pnl in pnls if pnl < 0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss

        # Kelly formula: f = p - q/b
        # where p = win rate, q = loss rate, b = win/loss ratio
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Cap Kelly at 25% for safety
        return min(max(kelly, 0), 0.25)

    def _hodl_comparison(self, backtest_results: Dict[str, Any]) -> float:
        """Compare strategy performance vs buy-and-hold"""
        if "data" not in backtest_results:
            return 0.0

        data = backtest_results["data"]

        # Verificar si tenemos datos crudos o preprocesados
        if "close" in data.columns:
            price_col = "close"
        elif "close_norm" in data.columns:
            price_col = "close_norm"
            logger.warning("Using normalized close prices for HODL comparison")
        else:
            logger.warning("No price column found for HODL comparison")
            return 0.0

        initial_price = data.iloc[0][price_col]
        final_price = data.iloc[-1][price_col]

        hodl_return = (final_price / initial_price) - 1
        strategy_return = self._total_return(backtest_results["equity_curve"])

        return strategy_return - hodl_return

    def _risk_adjusted_return(self, returns: np.ndarray) -> float:
        """Calculate risk-adjusted return (return per unit of risk)"""
        if len(returns) < 2:
            return 0.0

        total_return = (1 + returns).prod() - 1
        volatility = np.std(returns)

        if volatility == 0:
            return 0.0

        return total_return / volatility

    def calculate_rolling_metrics(
        self, equity_curve: np.ndarray, window: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling metrics for performance analysis"""
        if len(equity_curve) < window:
            return pd.DataFrame()

        returns = np.diff(equity_curve) / equity_curve[:-1]

        rolling_data = []
        for i in range(window, len(returns)):
            window_returns = returns[i - window : i]
            window_equity = equity_curve[i - window : i + 1]

            rolling_data.append(
                {
                    "index": i,
                    "rolling_return": (window_equity[-1] / window_equity[0]) - 1,
                    "rolling_volatility": np.std(window_returns) * np.sqrt(252),
                    "rolling_sharpe": self._sharpe_ratio(window_returns),
                    "rolling_max_dd": self._max_drawdown(window_equity),
                }
            )

        return pd.DataFrame(rolling_data)
