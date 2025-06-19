"""Cross validation helpers for time series backtesting."""

import numpy as np
from typing import Iterable, Tuple

try:
    from sklearn.model_selection import TimeSeriesSplit
except Exception:  # pragma: no cover - optional in light environments
    TimeSeriesSplit = None


class PurgedKFoldCV:
    """Simple purged k-fold cross validator."""

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X: Iterable) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        X = list(X)
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        for i in range(self.n_splits):
            start = i * fold_size
            stop = start + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate(
                [
                    indices[: max(0, start - self.embargo)],
                    indices[stop + self.embargo :],
                ]
            )
            yield train_idx, test_idx


class TimeSeriesCV:
    """Wrapper around scikit-learn ``TimeSeriesSplit`` for convenience."""

    def __init__(self, n_splits: int = 5):
        if TimeSeriesSplit is None:
            raise ImportError("scikit-learn is required for TimeSeriesCV")
        self.ts = TimeSeriesSplit(n_splits=n_splits)

    def split(self, X: Iterable) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series cross-validation splits."""
        X = list(X)
        indices = np.arange(len(X))
        for train_idx, test_idx in self.ts.split(indices):
            yield indices[train_idx], indices[test_idx]


__all__ = ["PurgedKFoldCV", "TimeSeriesCV"]
