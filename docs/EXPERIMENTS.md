# Running Experiments

This project includes a helper script in `scripts/run_experiment.py` to train a model and run a backtest using a YAML configuration.

```bash
python scripts/run_experiment.py config/sample_config.yaml --backtest
```

Use `--no-automl` to skip the hyperparameter optimization stage.

Metrics are logged to MLflow if available. Results are printed to the console otherwise.
