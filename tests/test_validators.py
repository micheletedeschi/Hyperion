import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
import numpy as np
from hyperion3.evaluations.validators import PurgedKFoldCV, TimeSeriesCV


def test_purged_kfold_split():
    cv = PurgedKFoldCV(n_splits=3, embargo=1)
    splits = list(cv.split(range(9)))
    assert len(splits) == 3
    train_idx, test_idx = splits[0]
    assert len(test_idx) > 0
    assert set(train_idx).isdisjoint(set(test_idx))


def test_time_series_cv():
    cv = TimeSeriesCV(n_splits=3)
    splits = list(cv.split(range(9)))
    assert len(splits) == 3
    # Ensure each split respects time order
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx)
