"""
Advanced Risk Management System for Hyperion V2
Implements sophisticated risk controls for cryptocurrency trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class RiskLevel(Enum):
    """Risk levels for position sizing"""

    ULTRA_LOW = 0.001
    LOW = 0.005
    MEDIUM = 0.01
    HIGH = 0.02
    AGGRESSIVE = 0.05


@dataclass
class RiskMetrics:
    """Container for risk metrics"""

    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    exposure: float
    leverage: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""

    position_size: float
    risk_amount: float
    stop_loss: float
    take_profit: Optional[float]
    leverage: float
    margin_required: float
    risk_reward_ratio: float


class RiskManager:
    """
    Advanced Risk Management System

    Features:
    - Dynamic position sizing
    - Portfolio-level risk controls
    - Correlation-based adjustments
    - Liquidity considerations
    - Real-time risk monitoring
    """

    def __init__(self, config: Dict):
        self.config = config
        self.max_portfolio_risk = config.get(
            "max_portfolio_risk", 0.06
        )  # 6% max portfolio risk
        self.max_position_risk = config.get(
            "max_position_risk", 0.02
        )  # 2% max per position
        self.max_correlation = config.get("max_correlation", 0.7)
        self.max_leverage = config.get("max_leverage", 3.0)
        self.min_liquidity_ratio = config.get("min_liquidity_ratio", 0.1)

        # Risk buffers
        self.emergency_stop = config.get("emergency_stop", 0.1)  # 10% portfolio loss
        self.risk_reduction_threshold = config.get("risk_reduction_threshold", 0.05)

        # Historical data for calculations
        self.returns_history = []
        self.positions_history = []
        self.risk_events = []

    def calculate_position_size(
        self,
        signal_strength: float,
        current_price: float,
        account_balance: float,
        market_volatility: float,
        existing_positions: List[Dict],
        market_data: pd.DataFrame,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size considering multiple factors

        Args:
            signal_strength: Signal strength [0, 1]
            current_price: Current asset price
            account_balance: Total account balance
            market_volatility: Current market volatility
            existing_positions: List of current positions
            market_data: Recent market data

        Returns:
            PositionSizeResult with sizing details
        """
        # Base position size using Kelly Criterion
        kelly_size = self._kelly_criterion(signal_strength, market_data)

        # Adjust for volatility
        volatility_adjusted_size = self._volatility_adjustment(
            kelly_size, market_volatility
        )

        # Adjust for correlation with existing positions
        correlation_adjusted_size = self._correlation_adjustment(
            volatility_adjusted_size, existing_positions, market_data
        )

        # Adjust for liquidity
        liquidity_adjusted_size = self._liquidity_adjustment(
            correlation_adjusted_size, market_data
        )

        # Apply portfolio-level constraints
        final_size = self._apply_portfolio_constraints(
            liquidity_adjusted_size, account_balance, existing_positions
        )

        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_exits(
            current_price, market_volatility, signal_strength
        )

        # Calculate required margin and risk metrics
        position_value = final_size * current_price
        leverage = position_value / (position_value / self.max_leverage)
        margin_required = position_value / leverage
        risk_amount = position_value * self.max_position_risk

        # Risk-reward ratio
        risk_reward_ratio = (
            (take_profit - current_price) / (current_price - stop_loss)
            if take_profit
            else 2.0
        )

        return PositionSizeResult(
            position_size=final_size,
            risk_amount=risk_amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            margin_required=margin_required,
            risk_reward_ratio=risk_reward_ratio,
        )

    def _kelly_criterion(
        self, signal_strength: float, market_data: pd.DataFrame
    ) -> float:
        """Calculate position size using Kelly Criterion"""
        if len(market_data) < 100:
            return 0.01  # Default small size

        returns = market_data["close"].pct_change().dropna()

        # Estimate win probability and win/loss ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.01

        win_probability = len(positive_returns) / len(returns)
        avg_win = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())

        # Adjust by signal strength
        adjusted_win_prob = win_probability * (0.5 + 0.5 * signal_strength)

        # Kelly formula: f = p - q/b
        # where p = win probability, q = loss probability, b = win/loss ratio
        if avg_loss == 0:
            return 0.01

        kelly_fraction = adjusted_win_prob - (1 - adjusted_win_prob) / (
            avg_win / avg_loss
        )

        # Apply Kelly fraction with safety factor (25% of full Kelly)
        return max(0, min(kelly_fraction * 0.25, 0.25))

    def _volatility_adjustment(
        self, base_size: float, market_volatility: float
    ) -> float:
        """Adjust position size based on market volatility"""
        # Target volatility approach
        target_volatility = 0.15  # 15% annual target
        current_volatility = market_volatility * np.sqrt(252)  # Annualize

        if current_volatility == 0:
            return base_size

        volatility_scalar = min(target_volatility / current_volatility, 2.0)

        return base_size * volatility_scalar

    def _correlation_adjustment(
        self,
        base_size: float,
        existing_positions: List[Dict],
        market_data: pd.DataFrame,
    ) -> float:
        """Adjust for correlation with existing positions"""
        if not existing_positions:
            return base_size

        # Calculate correlation matrix
        correlation_penalty = 0

        for position in existing_positions:
            # Simplified: assume some correlation
            # In practice, calculate actual correlation
            correlation = 0.5  # Placeholder
            position_weight = position["value"] / sum(
                p["value"] for p in existing_positions
            )

            correlation_penalty += correlation * position_weight

        # Reduce size based on correlation
        adjustment_factor = 1 - (correlation_penalty * 0.5)  # Max 50% reduction

        return base_size * max(adjustment_factor, 0.5)

    def _liquidity_adjustment(
        self, base_size: float, market_data: pd.DataFrame
    ) -> float:
        """Adjust position size based on liquidity"""
        if "volume" not in market_data.columns:
            return base_size

        # Calculate average volume
        avg_volume = market_data["volume"].rolling(24).mean().iloc[-1]
        recent_volume = market_data["volume"].iloc[-1]

        # Liquidity score
        liquidity_score = (
            min(recent_volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
        )

        # Adjust size (reduce in low liquidity)
        if liquidity_score < 0.5:
            return base_size * 0.5
        elif liquidity_score < 0.8:
            return base_size * 0.8
        else:
            return base_size

    def _apply_portfolio_constraints(
        self, base_size: float, account_balance: float, existing_positions: List[Dict]
    ) -> float:
        """Apply portfolio-level risk constraints"""
        # Calculate current portfolio exposure
        current_exposure = sum(p["value"] for p in existing_positions)

        # Maximum position size based on account balance
        max_position_value = account_balance * self.max_position_risk

        # Maximum based on total portfolio exposure
        remaining_exposure = (
            account_balance * self.max_portfolio_risk
        ) - current_exposure

        # Take the minimum of constraints
        max_allowed_value = min(
            max_position_value, remaining_exposure, account_balance * 0.3
        )

        # Convert to position size (assuming we have price)
        # This is simplified - in practice, we'd use the actual price
        final_size = min(base_size, max_allowed_value / account_balance)

        return max(final_size, 0)

    def _calculate_exits(
        self, current_price: float, volatility: float, signal_strength: float
    ) -> Tuple[float, Optional[float]]:
        """Calculate stop loss and take profit levels"""
        # ATR-based stop loss
        atr_multiplier = 2.0 - (
            signal_strength * 0.5
        )  # Tighter stops for stronger signals
        stop_distance = current_price * volatility * atr_multiplier
        stop_loss = current_price - stop_distance

        # Take profit based on risk-reward ratio
        min_risk_reward = 1.5
        target_risk_reward = min_risk_reward + (signal_strength * 2.5)  # 1.5 to 4.0

        take_profit = current_price + (stop_distance * target_risk_reward)

        return stop_loss, take_profit

    def calculate_portfolio_risk(
        self,
        positions: List[Dict],
        market_data: Dict[str, pd.DataFrame],
        lookback_days: int = 30,
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics

        Args:
            positions: Current positions
            market_data: Market data for each asset
            lookback_days: Days to look back for calculations

        Returns:
            RiskMetrics object
        """
        if not positions:
            return self._empty_risk_metrics()

        # Aggregate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(
            positions, market_data, lookback_days
        )

        if len(portfolio_returns) < 2:
            return self._empty_risk_metrics()

        # Calculate VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        # Calculate drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]

        # Performance ratios
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        calmar_ratio = self._calculate_calmar_ratio(portfolio_returns, max_drawdown)

        # Risk concentrations
        total_value = sum(p["value"] for p in positions)
        exposure = total_value / (total_value + sum(p["margin"] for p in positions))
        leverage = total_value / sum(p["margin"] for p in positions)

        # Correlation risk
        correlation_risk = self._calculate_correlation_risk(positions, market_data)

        # Concentration risk (Herfindahl-Hirschman Index)
        position_weights = [p["value"] / total_value for p in positions]
        concentration_risk = sum(w**2 for w in position_weights)

        # Liquidity risk
        liquidity_risk = self._calculate_liquidity_risk(positions, market_data)

        return RiskMetrics(
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            exposure=exposure,
            leverage=leverage,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
        )

    def _calculate_portfolio_returns(
        self,
        positions: List[Dict],
        market_data: Dict[str, pd.DataFrame],
        lookback_days: int,
    ) -> pd.Series:
        """Calculate historical portfolio returns"""
        portfolio_returns = pd.Series(dtype=float)

        for position in positions:
            symbol = position["symbol"]
            weight = position["value"] / sum(p["value"] for p in positions)

            if symbol in market_data:
                asset_returns = market_data[symbol]["close"].pct_change().dropna()
                asset_returns
