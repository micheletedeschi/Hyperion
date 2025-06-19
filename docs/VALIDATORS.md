# PurgedKFoldCV

`PurgedKFoldCV` implements a lightweight purged k-fold split generator. It can
be used in backtesting scenarios where data leakage between training and test
splits must be avoided. The `embargo` parameter removes a configurable number of
observations around each test fold to reduce look-ahead bias.

Example:

```python
from evaluations.validators import PurgedKFoldCV

cv = PurgedKFoldCV(n_splits=5, embargo=2)
for train_idx, test_idx in cv.split(range(100)):
    print(train_idx, test_idx)
```

The package also provides `TimeSeriesCV`, a convenience wrapper around
`TimeSeriesSplit` from scikit-learn for standard time-ordered cross validation.

splits must be avoided by applying an optional embargo period.

## Example

```python
from evaluations.validators import PurgedKFoldCV
import pandas as pd

data = pd.read_csv('dataset.csv')
cv = PurgedKFoldCV(n_splits=5, embargo=24)
for train_idx, test_idx in cv.split(data):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    # train model here
```

Using a purged split helps prevent look‑ahead bias when validating time series models.


