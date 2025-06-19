#!/usr/bin/env python
"""Run a training experiment using a YAML configuration."""

import argparse
import asyncio
from config.yaml_utils import load_config
from training.trainer import ModelTrainer
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Hyperion V2 experiment")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--no-automl", action="store_true", help="Disable hyperparameter optimization"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run a backtest after training"
    )
    return parser.parse_args()


async def run_experiment(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    trainer = ModelTrainer(config)
    symbol = config.data.symbols[0]

    train_data = (
        pd.DataFrame()
    )  # (o bien, se carga desde un archivo o se genera con datos de prueba)

    await trainer.train_with_automl(
        train_data, symbol=symbol, optimize_hyperparams=not args.no_automl
    )

    if args.backtest:
        try:
            from evaluation.backtester import AdvancedBacktester
        except ImportError:
            # Fallback to mock backtest from main module
            from main import HyperionV2System

            system = HyperionV2System()
            system.config = config
            await system._mock_backtest(symbol)
            return

        backtester = AdvancedBacktester(config)
        test_data = await trainer._prepare_data(symbol, None, None)
        results = await backtester.run(model=trainer, data=test_data)
        print(results)


def main() -> None:
    args = parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
