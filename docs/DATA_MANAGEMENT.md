# Data Management

This document describes the datasets expected by Hyperion V2 and how to
process them.

## Available data sources

- **OHLCV**: historical price candles stored in `data/datasets/`.
- **Sentiment**: optional sentiment scores collected from social media.
- **On-chain metrics**: blockchain statistics such as active addresses.
- **Orderbook snapshots**: depth of market information for each symbol.

All these sources can be enabled or disabled via `DataConfig` in
`config/base_config.py`.

## Processing workflow

1. Download the raw data using the scripts in `data/downloader.py`.
2. Run `data/preprocessor.py` to clean missing values and add technical
   indicators.
3. Apply `data/augmentation.py` if additional synthetic series are desired.
4. Features are saved under `data/processed/` and can then be loaded by the
   training pipeline.

Splits between train, validation and test sets are defined in the
configuration (`train_split`, `val_split`, `test_split`). Adjust the
`lookback_window` and `prediction_horizon` according to your modelling needs.
