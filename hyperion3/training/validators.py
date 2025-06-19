from typing import List, Tuple
import numpy as np


def purged_kfold(
    n_samples: int, n_splits: int = 5, embargo: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged k-fold indices."""
    indices = np.arange(n_samples)
    fold_sizes = n_samples // n_splits
    splits = []
    for i in range(n_splits):
        start = i * fold_sizes
        stop = start + fold_sizes
        test_idx = indices[start:stop]
        train_idx = np.concatenate(
            [
                indices[: max(0, start - embargo)],
                indices[stop + embargo :],
            ]
        )
        splits.append((train_idx, test_idx))
    return splits


__all__ = ["purged_kfold"]
