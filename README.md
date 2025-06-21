# Hyperion: An Algorithmic Trading Framework Forged in Obsession

![Version](https://img.shields.io/badge/version-3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Trading](https://img.shields.io/badge/trading-algorithmic-gold.svg)
![AI](https://img.shields.io/badge/AI-enabled-purple.svg)
![MLOps](https://img.shields.io/badge/MLOps-integrated-orange.svg)

> "Hyperion is not just code. It's the answer to a challenge that completely captivated me. It's the tool I wish I had when I started."

(This is still a beta version - not all parts of the bot are working perfectly yet, but as more days pass, I realize it will never be as perfect as I dream, so I decided to show what's been developed. In the coming days, I'll fix the remaining errors and implement all the features. Thanks for reading, I hope to be up to the challenge)

## 🚨 CRITICAL WARNING - READ THIS FIRST

<table>
<tr>
<td>
<h3>⚠️ EXTREME FINANCIAL RISK</h3>

**Hyperion is a RESEARCH and LEARNING tool, NOT a money-making machine.**

- 🔮 **Backtests do NOT guarantee future returns**
- 🌪️ **The real market is chaotic and unpredictable**
- 💰 **NEVER invest money you can't afford to lose**
- ⚖️ **Use it under your COMPLETE responsibility**
- 🧪 **ALWAYS practice in paper trading first**

**Algorithmic trading requires deep knowledge, risk management, and understanding that losses are part of the game.**
</td>
</tr>
</table>

## 💫 The Story Behind Hyperion

Hello, I'm the creator of Hyperion. Giovanni, but I prefer to be called Ganador (Winner). Let me tell you a story.

I always dreamed of creating a trading bot that operated autonomously, an intelligent machine capable of navigating the turbulent seas of the market. When I started, my naivety made me think it would be an easy task. I believed that in a couple of weekends I would have something working.

**How wrong I was.**

I soon came face to face with reality: building a cutting-edge bot, starting from scratch and alone, is a colossal challenge. I immersed myself in an ocean of research papers, model architectures, and preprocessing techniques, and each new layer of knowledge revealed ten more that I didn't know.

When looking for help, I realized that the landscape of public and free trading bots was bleak. Most were too simple, black boxes without flexibility, or directly useless. I felt immense frustration. How could someone, with more desire than experience, start in this world?

It was in that moment of frustration and challenge that **Hyperion was born**. I decided that if the tool I needed didn't exist, I would build it myself.

This project is the result of countless hours of work, trial and error, small failures and great victories. My most sincere hope is that Hyperion saves you part of that difficult path and gives you the power so that you too can transform your ideas into real strategies.

## 🚀 Quick Start

### 🎯 Complete Setup (5 minutes)
```bash
# 1. Start the professional system (RECOMMENDED)
pip install -r requirements.txt
python main.py

# 2. Access the professional interface directly
python main_professional.py

# 3. Validate the modular structure
python test_modular_structure.py
```

## 🏗️ The Hyperion Pipeline: Anatomy of an Idea

Hyperion is designed as a **modular and automated pipeline** that transforms raw market data into robust and validated trading strategies. The entire process is controlled from a single configuration file (`config.json`), simplifying a workflow that would otherwise be extremely complex.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Config    │ -> │    Data     │ -> │Preprocessor │ -> │   Model     │
│ (config.json)│    │ Downloader  │    │& Features   │    │  Trainer    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                  │
                                                                  v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    MLOps    │ <- │ Backtester  │ <- │  Ensemble   │ <- │ Hyperopt    │
│  Tracking   │    │& Validation │    │ Creation    │    │ (FLAML)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 🧠 Stage 1: Configuration (config.json)
**The Brain of the Operation**: Here you define everything: the cryptocurrency pair (e.g., BTC/USDT), the time interval (1h, 4h, 1d), start and end dates, and most importantly: the list of models you want to test and the optimizer configuration.

### 📊 Stage 2: Data Acquisition (hyperion3/data/downloader.py)
Hyperion connects to data sources and downloads the OHLCV (Open, High, Low, Close, Volume) price history for the asset you specified. Data is saved locally for quick access and reuse.

### ⚙️ Stage 3: Preprocessing and Feature Engineering
**Cleaning and Preparation**: Raw data is cleaned of missing values and prepared for analysis.

**Intelligence Creation**: Here's where the magic happens! Hyperion doesn't just use price. It generates an arsenal of **more than 100 features** to give models deep market context:

- **📈 Momentum Indicators**: RSI, Stochastic, MACD, Williams %R
- **📊 Trend Indicators**: Moving Averages (SMA, EMA), Bollinger Bands, ADX, Ichimoku Cloud, Vortex
- **💨 Volatility Indicators**: ATR (Average True Range), Keltner Channels
- **📊 Volume Analysis**: On-Balance Volume (OBV)
- **🕯️ Japanese Candlestick Patterns**: Doji, Engulfing, Hammer, etc.
- **🎭 Data Augmentation**: To avoid overfitting, synthetic variations of the data are created

### 🔬 Stage 4: Advanced Hyperparameter Optimization
**The Search for Perfection**: Hyperion features a sophisticated multi-engine optimization system that automatically discovers optimal parameters for **50+ models** across multiple categories.

#### 🎯 **Optimization Engines**:
- **🔥 Optuna**: Bayesian optimization with TPE (Tree-structured Parzen Estimator) sampler for intelligent parameter space exploration
- **⚡ FLAML**: Microsoft's AutoML framework for rapid optimization with resource constraints
- **🧠 Scikit-learn**: Traditional GridSearch/RandomSearch for comprehensive parameter coverage
- **🎲 TPOT**: Genetic programming for automated pipeline optimization

#### 📊 **Supported Model Categories** (50+ models):

**🌳 Scikit-learn Models (35+ algorithms)**:
- *Tree-based*: RandomForest, GradientBoosting, ExtraTrees, AdaBoost, Bagging, HistGradientBoosting
- *Linear Models*: Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor
- *SVM Variants*: SVR, NuSVR, LinearSVR with RBF/Polynomial/Linear kernels
- *Neural Networks*: MLPRegressor with customizable architectures
- *Advanced*: GaussianProcess, KernelRidge, QuantileRegressor, TweedieRegressor, PoissonRegressor, GammaRegressor

**⚡ Gradient Boosting Libraries**:
- **XGBoost**: Full parameter space optimization with GPU acceleration support
- **LightGBM**: High-performance boosting with memory-efficient optimization
- **CatBoost**: Categorical feature handling with automatic GPU detection

**🧠 Deep Learning Models**:
- **PyTorch**: SimpleMLP, DeepMLP, LSTM networks with architecture optimization
- **Transformers**: Temporal Fusion Transformer (TFT), PatchTST with attention mechanism tuning
- **Custom**: Neural architecture search for optimal layer configurations

**🎮 Reinforcement Learning Agents**:
- **SAC** (Soft Actor-Critic): Continuous action spaces with entropy regularization
- **TD3** (Twin Delayed DDPG): Policy gradient methods with noise injection
- **Rainbow DQN**: Multi-improvement DQN with distributional learning

#### 🚀 **Advanced Optimization Features**:
```python
# Comprehensive optimization example
from utils.hyperopt import HyperparameterOptimizer

# Initialize with GPU configuration
optimizer = HyperparameterOptimizer(
    console=console,
    gpu_config={
        'xgboost_params': {'tree_method': 'gpu_hist'},
        'lightgbm_params': {'device': 'gpu'},
        'catboost_params': {'task_type': 'GPU'}
    }
)

# Optimize all 50+ models
results = optimizer.optimize_all_models(
    X_train, y_train, X_val, y_val, 
    n_trials=100  # Bayesian optimization trials
)

# Category-specific optimization
sklearn_best = optimizer.optimize_sklearn_models(X_train, y_train, X_val, y_val, n_trials=50)
rl_best = optimizer.optimize_rl_agents(X_train, y_train, X_val, y_val, n_trials=30)
```

#### 🎨 **Smart Parameter Spaces**:
- **Dynamic ranges**: Parameters automatically adjusted based on dataset characteristics
- **Model-specific constraints**: Each algorithm has tailored parameter boundaries
- **Multi-objective**: Simultaneous optimization for accuracy, speed, and memory usage
- **Early stopping**: Intelligent trial pruning with Optuna's median stopping
- **Cross-validation**: Integrated CV for robust hyperparameter validation

### 🤝 Stage 5: Ensemble Creation
**The Wisdom of the Crowd**: Instead of relying on a single "genius", Hyperion can combine the predictions of your best models in an ensemble. This often leads to more stable and robust decisions.

### 🧪 Stage 6: Rigorous Backtesting
**The Trial by Fire**: The Backtester simulates how your strategy would have performed in the past, trade by trade. It provides you with critical metrics like Total Return, Sharpe Ratio, Maximum Drawdown, and hit rate.

### 📈 Stage 7: Analysis and MLOps
**Reproducibility and Transparency**: Every detail of your experiment is automatically recorded with MLflow. This allows you to compare different approaches and return to any point in your research without getting lost.

## 🤖 The Model Arsenal: A Complete Spectrum of Intelligence

Hyperion integrates an exceptionally diverse model library, allowing you to approach the problem from multiple angles. All models are instantiated through `hyperion3/models/model_factory.py`.

### 📊 1. Classical and Statistical Models
- **Prophet**: Developed by Facebook, excellent for capturing seasonalities and trends robustly

### 🌳 2. Machine Learning Models (Tree-Based)
The backbone of modern data science. They are fast, interpretable, and very powerful:

- **🚀 LightGBM**: The fastest option. Uses extremely efficient leaf-wise growth
- **🏆 XGBoost**: The gold standard. Famous for its performance and anti-overfitting regularization
- **🎯 CatBoost**: Specially designed to handle data efficiently, very robust
- **🌲 RandomForest and ExtraTrees**: Ensembles of multiple trees to improve robustness

### 🧠 3. Deep Learning Models for Time Series
Specifically designed to capture complex temporal dependencies:

- **📈 N-BEATS**: Decomposes the time series into interpretable components
- **⚡ N-HITS**: Evolution of N-BEATS with better efficiency and frequency spectrum
- **🔥 TFT (Temporal Fusion Transformer)**: Fuses different types of data with attention mechanisms
- **💎 PatchTST (Transformer)**: The crown jewel! Based on Google's Transformer architecture, processes the time series in "patches" to capture short and long-term relationships

### 🎮 4. Reinforcement Learning (RL)
**The most radical paradigm shift**. Instead of predicting the future, agents learn to act to maximize rewards:

- **🎭 SAC (Soft Actor-Critic)**: Modern, efficient and very stable algorithm
- **🎯 TD3 (Twin Delayed DDPG)**: Robust, designed to mitigate value overestimation
- **🌈 Rainbow DQN**: Improvement of the classic DQN that combines multiple techniques

**How does RL work?** The agent is the "trader". It observes the market and decides actions (buy/sell/hold). If it wins, it receives positive reward. After thousands of simulations, it learns a policy to maximize profits. It's the closest thing to teaching an AI to "think" like a trader.

## ✨ Professional Interface

Hyperion3 features a **complete professional interface**:

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

## 🚀 Installation

### Quick Installation
```bash
pip install -r requirements.txt
```

### 🍎 For Apple Silicon users
```bash
./install_mac.sh
```

### 📋 Requirements
- **Python 3.8+**
- **Unix OS** (Linux or macOS recommended)
- **Optional**: GPU with CUDA for deep learning models

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed instructions.

## 🏗️ Project Architecture

The code is organized in modular packages:

- **`hyperion3/models/`** – transformers and RL agents
- **`hyperion3/training/`** – training loops and callbacks
- **`hyperion3/evaluations/`** – backtester and financial metrics
- **`hyperion3/optimization/`** – AutoML utilities with FLAML
- **`hyperion3/data/`** – downloaders, preprocessing and feature engineering
- **`scripts/deployment/`** – live trading engine and monitoring
- **`scripts/`** – auxiliary commands for training and testing
- **`docs/`** – additional documentation

### 🎨 Professional Features
- **🎨 Rich UI**: Beautiful console interface with Rich library
- **🔧 Modular Design**: Clean separation in utils/ modules
- **⚡ Performance**: Optimized for Apple Silicon (MPS) and CUDA
- **💾 Auto-Save**: Automatic saving of models, results and configurations
- **📊 Analytics**: Integrated analysis and comparison tools

### 📈 Advanced Features
- **📊 Real-time data** with Binance API
- **🧪 Advanced backtesting** with multiple strategies
- **⚠️ Risk management** and portfolio optimization
- **🔬 MLOps integration** with experiment tracking
- **⏰ Multi-timeframe analysis** and prediction

## 📊 Dataset Management

Raw datasets reside in `data/`. Use the provided preprocessing scripts to generate features and augmentations. The `DataConfig` class controls symbols, lookback windows, and additional data sources like sentiment, orderbook, or on-chain metrics.

See [`docs/DATA_MANAGEMENT.md`](docs/DATA_MANAGEMENT.md) for a complete tutorial.

## 📚 Documentation

Additional guides available in the `docs/` directory:

- [`BACKTESTER.md`](docs/BACKTESTER.md) – advanced backtesting engine
- [`EXPERIMENTS.md`](docs/EXPERIMENTS.md) – running configurable experiments
- [`VALIDATORS.md`](docs/VALIDATORS.md) – cross-validation helpers
- [`INSTALLATION.md`](docs/INSTALLATION.md) – detailed installation instructions
- [`DEVELOPMENT_GUIDE.md`](docs/DEVELOPMENT_GUIDE.md) – development guide

## 🤝 Join the Journey

Hyperion is a **living and constantly evolving project**. If you're passionate about this world, your help is welcome. You can contribute:

- 🐛 **Reporting bugs** via Issues
- 💡 **Suggesting new features** 
- 🔧 **Adding your own code** via Pull Requests
- 📖 **Improving documentation**
- 🧪 **Sharing experiment results**

The project structure is modular, which facilitates adding new models, metrics, or data processors.

### 🛠️ How to Contribute
1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Make sure all tests pass before submitting.

## 📜 License

This project is published under the terms of the Apache 2.0 License. See [`LICENSE`](LICENSE) for more details.

---

## 🌟 Acknowledgments

Thanks to the open source community and all the researchers whose work has made Hyperion possible.

**✨ Ready to transform your ideas into real strategies? Start your journey with Hyperion today!**

```bash
git clone https://github.com/your-username/hyperion.git
cd hyperion
pip install -r requirements.txt
python main.py
```

*Thanks for reading to the end. This is my first project and I hope this and the following ones I work on can be useful.*

---

**📖 Documentation available in multiple languages:**
- 🇺🇸 [English](README.md) (current)
- 🇪🇸 [Español](README_es.md)
