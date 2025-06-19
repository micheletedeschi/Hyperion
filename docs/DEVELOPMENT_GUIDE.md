# Development Guide

This project uses `pytest` for testing and requires the packages listed in
`requirements.txt`.

For detailed setup instructions see [INSTALLATION.md](INSTALLATION.md).

## Running Tests

```bash
pip install -r requirements.txt
pytest
```

## Datasets
Datasets should be placed in `data/datasets/`. Use `validate_ohlcv` from
`data.validators` to check the integrity of downloaded data.


## YAML Configuration

Configurations can be edited in YAML format. See `config/sample_config.yaml` for
an example. Load and save configs via `config.load_config` and `config.save_config`.


