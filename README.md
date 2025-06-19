# Hyperion3 - Advanced Cryptocurrency Trading System

![Version](https://img.shields.io/badge/version-3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Quick Start

```bash
# 1. Start the professional system (RECOMMENDED)
python main.py

# 2. Or access directly the professional interface
python main_professional.py

# 3. Validate the modular structure
python test_modular_structure.py
```

## ✨ NEW: Professional System Interface

Hyperion3 now features a **complete professional interface** with:

### 🎯 **Main Menu Features**
- **🤖 MODELS**: Train individual models by category (sklearn, ensemble, pytorch, automl)
- **🎯 HYPERPARAMETERS**: Automatic and manual hyperparameter optimization
- **🎭 ENSEMBLES**: Create and manage ensembles (voting, weighted, stacking, bagging)
- **📊 ANALYSIS**: Complete analysis of results and performance metrics
- **⚙️ CONFIGURATION**: System configuration management
- **📈 MONITORING**: Real-time system monitoring

### 🔧 **Modular Training Options**
```bash
# Train specific models
s1, s2, s3...    # sklearn models (Random Forest, Gradient Boosting, etc.)
e1, e2, e3...    # ensemble models (XGBoost, LightGBM, CatBoost)
p1, p2, p3...    # pytorch models (MLP, LSTM, Transformer)
a1, a2...        # automl models (FLAML, Optuna)

# Train by category
sklearn, ensemble, pytorch, automl

# Train all models
all
```

## 🎯 Key Features

### ✨ Professional Architecture
- **🎨 Rich UI**: Beautiful console interface with Rich library
- **🔧 Modular Design**: Clean separation across utils/ modules
- **⚡ Performance**: Optimized for Apple Silicon (MPS) and CUDA
- **💾 Auto-Save**: Automatic saving of models, results, and configurations
- **📊 Analytics**: Built-in analysis and comparison tools

### 🤖 Advanced Models
- **Ensemble Learning**: XGBoost, LightGBM, CatBoost, RandomForest
- **Reinforcement Learning**: SAC, TD3, Rainbow DQN, Ensemble Agents
- **Transformers**: PatchTST, TFT (Temporal Fusion Transformer)
- **AutoML**: FLAML integration for automated model selection

### 📈 Professional Features
- **Real-time data** processing with Binance API
- **Advanced backtesting** with multiple strategies
- **Risk management** and portfolio optimization
- **MLOps integration** with experiment tracking
- **Multi-timeframe** analysis and prediction

## Installation

The quickest way to install the dependencies is:

```bash
pip install -r requirements.txt
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed instructions and a helper script for Apple Silicon users. A lightweight requirements file is also available under `requirements-test.txt` for continuous integration environments.

### Requirements

* Python 3.8 or newer
* Unix-like OS (Linux or macOS recommended)
* Optional: GPU with CUDA for deep learning models

## Architecture Overview

The code base is organised into modular packages:

- `models/` – transformers and RL agents.
- `training/` – training loops and callbacks.
- `evaluations/` – backtester and financial metrics.
- `optimization/` – AutoML utilities with FLAML and Ray Tune.
- `data/` – downloaders, preprocessing and feature engineering tools.
- `deployment/` – live trading engine and monitoring utilities.
- `scripts/` – helper commands for training and testing.
- `docs/` – additional documentation for backtesting and data handling.

A detailed Spanish overview is available in [`estructura del proyecto.txt`](estructura%20del%20proyecto.txt).

## Dataset Management

Raw datasets reside in `data/`. Use the provided preprocessing scripts to generate features and augmentations. The `DataConfig` class controls symbols, lookback windows and additional data sources such as sentiment, orderbook or on‑chain metrics. See [`docs/DATA_MANAGEMENT.md`](docs/DATA_MANAGEMENT.md) for a walkthrough.


## Documentation

Further guides are available under the `docs/` directory:

- [`BACKTESTER.md`](docs/BACKTESTER.md) – advanced backtesting engine.
- [`EXPERIMENTS.md`](docs/EXPERIMENTS.md) – running configurable experiments.
- [`VALIDATORS.md`](docs/VALIDATORS.md) – cross‑validation helpers.


## Contributing

Contributions are welcome! Open issues or pull requests on GitHub. Please ensure all tests pass before submitting.

## License

This project is released under the terms of the Apache 2.0 License. See [`LICENSE`](LICENSE) for details.

