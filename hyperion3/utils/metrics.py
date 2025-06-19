"""
Módulo de métricas para Hyperion V2
Implementa métricas financieras y de rendimiento para evaluación de modelos
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from scipy import stats


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    predictions: Optional[Union[pd.Series, np.ndarray]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calcula el Ratio de Sharpe para una serie de retornos

    Args:
        returns: Retornos reales o diferencias entre reales y predicciones
        predictions: Predicciones del modelo (opcional)
        risk_free_rate: Tasa libre de riesgo anual (default: 0.0)
        periods_per_year: Número de períodos por año (default: 252 para trading diario)

    Returns:
        Ratio de Sharpe anualizado
    """
    # Si se proporcionan predicciones, calcular retornos
    if predictions is not None:
        returns = returns - predictions

    # Convertir a numpy si es pandas
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Calcular retorno y desviación estándar
    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)

    # Evitar división por cero
    if std_return == 0:
        return 0.0

    # Calcular Sharpe Ratio y anualizar
    sharpe = mean_return / std_return
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    predictions: Optional[Union[pd.Series, np.ndarray]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Calcula el Ratio de Sortino para una serie de retornos

    Args:
        returns: Retornos reales o diferencias entre reales y predicciones
        predictions: Predicciones del modelo (opcional)
        risk_free_rate: Tasa libre de riesgo anual (default: 0.0)
        periods_per_year: Número de períodos por año (default: 252)
        target_return: Retorno objetivo (default: 0.0)

    Returns:
        Ratio de Sortino anualizado
    """
    if predictions is not None:
        returns = returns - predictions

    if isinstance(returns, pd.Series):
        returns = returns.values

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)

    # Calcular desviación a la baja
    downside_returns = excess_returns[excess_returns < target_return]
    if len(downside_returns) == 0:
        return 0.0

    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0:
        return 0.0

    sortino = mean_return / downside_std
    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(
    returns: Union[pd.Series, np.ndarray],
    predictions: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Tuple[float, int, int]:
    """
    Calcula el máximo drawdown y sus puntos de inicio/fin

    Args:
        returns: Retornos reales o diferencias entre reales y predicciones
        predictions: Predicciones del modelo (opcional)

    Returns:
        Tuple con (máximo drawdown, índice de inicio, índice de fin)
    """
    if predictions is not None:
        returns = returns - predictions

    if isinstance(returns, pd.Series):
        returns = returns.values

    # Calcular retornos acumulados
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max

    # Encontrar máximo drawdown
    max_dd = np.min(drawdowns)
    end_idx = np.argmin(drawdowns)
    start_idx = np.argmax(cum_returns[:end_idx])

    return max_dd, start_idx, end_idx


def calculate_win_rate(
    returns: Union[pd.Series, np.ndarray],
    predictions: Optional[Union[pd.Series, np.ndarray]] = None,
    threshold: float = 0.0,
) -> float:
    """
    Calcula la tasa de aciertos (win rate)

    Args:
        returns: Retornos reales o diferencias entre reales y predicciones
        predictions: Predicciones del modelo (opcional)
        threshold: Umbral para considerar una predicción como ganadora

    Returns:
        Tasa de aciertos (0-1)
    """
    if predictions is not None:
        returns = returns - predictions

    if isinstance(returns, pd.Series):
        returns = returns.values

    wins = np.sum(returns > threshold)
    total = len(returns)

    return wins / total if total > 0 else 0.0


def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    predictions: Optional[Union[pd.Series, np.ndarray]] = None,
    periods_per_year: int = 252,
) -> float:
    """
    Calcula el Ratio de Calmar (retorno anualizado / máximo drawdown)

    Args:
        returns: Retornos reales o diferencias entre reales y predicciones
        predictions: Predicciones del modelo (opcional)
        periods_per_year: Número de períodos por año (default: 252)

    Returns:
        Ratio de Calmar
    """
    if predictions is not None:
        returns = returns - predictions

    if isinstance(returns, pd.Series):
        returns = returns.values

    # Calcular retorno anualizado
    total_return = np.prod(1 + returns) - 1
    years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1

    # Calcular máximo drawdown
    max_dd, _, _ = calculate_max_drawdown(returns)

    # Evitar división por cero
    if max_dd == 0:
        return 0.0

    return annual_return / abs(max_dd)


def calculate_information_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    """
    Calcula el Information Ratio

    Args:
        returns: Retornos del modelo
        benchmark: Retornos del benchmark
        periods_per_year: Número de períodos por año (default: 252)

    Returns:
        Information Ratio anualizado
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(benchmark, pd.Series):
        benchmark = benchmark.values

    # Calcular retornos activos
    active_returns = returns - benchmark

    # Calcular ratio
    mean_active = np.mean(active_returns)
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        return 0.0

    return (mean_active / tracking_error) * np.sqrt(periods_per_year)


def calculate_all_metrics(
    returns: Union[pd.Series, np.ndarray],
    predictions: Optional[Union[pd.Series, np.ndarray]] = None,
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Calcula todas las métricas disponibles

    Args:
        returns: Retornos reales
        predictions: Predicciones del modelo (opcional)
        benchmark: Retornos del benchmark (opcional)
        risk_free_rate: Tasa libre de riesgo anual
        periods_per_year: Número de períodos por año

    Returns:
        Diccionario con todas las métricas
    """
    metrics = {
        "sharpe_ratio": calculate_sharpe_ratio(
            returns, predictions, risk_free_rate, periods_per_year
        ),
        "sortino_ratio": calculate_sortino_ratio(
            returns, predictions, risk_free_rate, periods_per_year
        ),
        "max_drawdown": calculate_max_drawdown(returns, predictions)[0],
        "win_rate": calculate_win_rate(returns, predictions),
        "calmar_ratio": calculate_calmar_ratio(returns, predictions, periods_per_year),
    }

    if benchmark is not None:
        metrics["information_ratio"] = calculate_information_ratio(
            returns, benchmark, periods_per_year
        )

    return metrics


# Alias para compatibilidad
calculate_metrics = calculate_all_metrics

__all__ = [
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_win_rate",
    "calculate_calmar_ratio",
    "calculate_information_ratio",
    "calculate_all_metrics",
    "calculate_metrics",
]


class FinancialMetrics:
    """
    Wrapper para métricas financieras y de rendimiento.
    Permite usar las funciones como métodos de instancia o estáticos.
    """

    @staticmethod
    def sharpe_ratio(
        returns, predictions=None, risk_free_rate=0.0, periods_per_year=252
    ):
        from .metrics import calculate_sharpe_ratio

        return calculate_sharpe_ratio(
            returns, predictions, risk_free_rate, periods_per_year
        )

    @staticmethod
    def sortino_ratio(
        returns,
        predictions=None,
        risk_free_rate=0.0,
        periods_per_year=252,
        target_return=0.0,
    ):
        from .metrics import calculate_sortino_ratio

        return calculate_sortino_ratio(
            returns, predictions, risk_free_rate, periods_per_year, target_return
        )

    @staticmethod
    def max_drawdown(returns, predictions=None):
        from .metrics import calculate_max_drawdown

        return calculate_max_drawdown(returns, predictions)

    @staticmethod
    def win_rate(returns, predictions=None, threshold=0.0):
        from .metrics import calculate_win_rate

        return calculate_win_rate(returns, predictions, threshold)

    @staticmethod
    def calmar_ratio(returns, predictions=None, periods_per_year=252):
        from .metrics import calculate_calmar_ratio

        return calculate_calmar_ratio(returns, predictions, periods_per_year)

    @staticmethod
    def information_ratio(returns, benchmark, periods_per_year=252):
        from .metrics import calculate_information_ratio

        return calculate_information_ratio(returns, benchmark, periods_per_year)

    @staticmethod
    def all_metrics(
        returns,
        predictions=None,
        benchmark=None,
        risk_free_rate=0.0,
        periods_per_year=252,
    ):
        from .metrics import calculate_all_metrics

        return calculate_all_metrics(
            returns, predictions, benchmark, risk_free_rate, periods_per_year
        )
