# AdvancedBacktester

`AdvancedBacktester` is a minimal asynchronous backtesting engine used to

evaluate trading models. The component accepts a pandas data frame with price
history and simulates a buy-and-hold strategy in order to produce baseline
results.

The main steps are:

1. Calculate percentage returns from the input prices.
2. Build an equity curve by compounding those returns.
3. Collect trades (empty by default) and timestamps.
4. Optionally compute financial metrics using `FinancialMetrics`.

Example usage:

```python
from evaluations.backtester import AdvancedBacktester
from config.base_config import HyperionV2Config

backtester = AdvancedBacktester(HyperionV2Config())
results = await backtester.run(model, price_data)
print(results["metrics"]["sharpe_ratio"])
```

The returned dictionary contains the equity curve, raw returns, timestamps and a
`metrics` key with standard performance statistics such as Sharpe ratio,
drawdown and profit factor.

evaluate models. It computes simple returns from the provided price series and
produces an equity curve that can be analysed with the metrics module.

## Usage

Instantiate the backtester with a configuration object and call `run` with a model and a DataFrame of OHLCV data. The method returns a dictionary containing the equity curve.

```python
import pandas as pd
from config.base_config import HyperionV2Config
from evaluations.backtester import AdvancedBacktester

config = HyperionV2Config()
data = pd.read_csv('data/BTCUSDT.csv', parse_dates=True, index_col=0)

backtester = AdvancedBacktester(config)
results = await backtester.run(model, data)
```

The resulting `results['equity_curve']` can then be passed to `hyperion3.utils.metrics.calculate_all_metrics` for performance statistics.


