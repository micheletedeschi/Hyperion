# 🎯 HYPERION HYPERPARAMETER OPTIMIZATION SYSTEM

## Overview

The Hyperion Hyperparameter Optimization System is a comprehensive, production-ready solution for optimizing machine learning models across multiple categories including traditional ML, deep learning, advanced time series models, and reinforcement learning agents specifically designed for trading applications.

## 🚀 Key Features

### **46+ Optimizable Models**
- **Traditional ML**: 36 sklearn models (Random Forest, Gradient Boosting, SVR, etc.)
- **Ensemble Methods**: XGBoost, LightGBM, CatBoost
- **Deep Learning**: MLP, LSTM, CNN architectures
- **Advanced Time Series**: Temporal Fusion Transformer (TFT), PatchTST
- **Reinforcement Learning**: SAC, TD3, RainbowDQN (Trading-optimized)
- **AutoML**: FLAML automated machine learning

### **Trading-Specific RL Optimization**
- Custom trading environment simulator
- Real trading metrics: Sharpe ratio, returns, drawdown
- Optimized for trading strategies (not generic regression)
- Support for continuous and discrete action spaces

### **Advanced Optimization Engine**
- **Optuna-powered**: State-of-the-art Bayesian optimization
- **TPE Sampler**: Tree-structured Parzen Estimator for efficient search
- **Model-specific parameter spaces**: Tailored hyperparameter ranges
- **Robust error handling**: Graceful degradation for missing dependencies

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYPERION OPTIMIZATION SYSTEM                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   TRADITIONAL   │  │   DEEP LEARNING │  │   REINFORCEMENT │  │
│  │      ML         │  │                 │  │    LEARNING     │  │
│  │                 │  │                 │  │                 │  │
│  │ • Sklearn (36)  │  │ • PyTorch       │  │ • SAC Trading   │  │
│  │ • XGBoost       │  │ • LSTM/CNN      │  │ • TD3 Trading   │  │
│  │ • LightGBM      │  │ • TFT           │  │ • Rainbow DQN   │  │
│  │ • CatBoost      │  │ • PatchTST      │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     OPTIMIZATION ENGINE                        │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │     OPTUNA      │  │   PARAMETER     │  │   EVALUATION    │  │
│  │   TPE SAMPLER   │  │    SPACES       │  │    METRICS      │  │
│  │                 │  │                 │  │                 │  │
│  │ • Bayesian Opt  │  │ • Model-specific│  │ • R² Score      │  │
│  │ • Multi-trial   │  │ • Trading-aware │  │ • Sharpe Ratio  │  │
│  │ • Parallel      │  │ • Robust ranges │  │ • Trading PnL   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠 Installation & Setup

### Prerequisites

```bash
# Core dependencies
pip install optuna scikit-learn xgboost lightgbm catboost flaml

# Deep learning (optional)
pip install torch torchvision transformers

# RL dependencies (optional)
pip install stable-baselines3 gym

# Visualization
pip install rich matplotlib seaborn
```

### Quick Start

```python
from hyperion3.main_professional import run_professional_mode

# Launch the professional interface
run_professional_mode()
```

## 🎯 Usage Guide

### Interactive Menu System

The system provides an intuitive menu-driven interface:

```
╔══════════════════════════════════════════════════════════════╗
║                 HYPERPARAMETER OPTIMIZATION                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 Available Commands:                                      ║
║                                                              ║
║  🔧 Model Categories:                                        ║
║     sklearn      - Optimize 36 sklearn models                ║
║     ensemble     - XGBoost, LightGBM, CatBoost               ║
║     pytorch      - MLP, LSTM, CNN models                     ║
║     advanced     - TFT, PatchTST, Transformers               ║
║     rl_agents    - SAC, TD3, RainbowDQN (Trading)            ║
║     automl       - FLAML AutoML                              ║
║                                                              ║
║  🎯 Specific Models:                                         ║
║     xgboost      - XGBoost optimization                      ║
║     tft          - Temporal Fusion Transformer               ║
║     sac          - SAC Trading Agent                         ║
║                                                              ║
║  📋 Utilities:                                               ║
║     capabilities - Show all available models                 ║
║     all          - Optimize all models                       ║
║     compare      - Compare optimization results              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Command Examples

#### 1. Optimize by Category
```bash
# Optimize all sklearn models
> sklearn

# Optimize reinforcement learning agents for trading
> rl_agents

# Optimize deep learning models
> pytorch
```

#### 2. Optimize Specific Models
```bash
# Optimize XGBoost specifically
> xgboost

# Optimize Temporal Fusion Transformer
> tft

# Optimize SAC trading agent
> sac
```

#### 3. View System Capabilities
```bash
# Show all available models
> capabilities

# Output:
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Category               ┃ Status             ┃ Available Models            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Sklearn                │ ✅ 36/36          │ RF, GB, Ridge, Lasso, SVR... │
│ Ensemble               │ ✅ 3/3            │ XGBoost, LightGBM, CatBoost  │
│ Deep_Learning          │ ✅ 3/3            │ SimpleMLP, DeepMLP, LSTM     │
│ Advanced_Models        │ ✅ 3/3            │ TFT, PatchTST, Transformers  │
│ Reinforcement_Learning │ ✅ 3/3 TRADING    │ SAC, TD3, RainbowDQN         │
│ AutoML                 │ ✅ 1/3            │ FLAML                        │
└────────────────────────┴────────────────────┴─────────────────────────────┘
```

## 🤖 Reinforcement Learning Trading Optimization

### Overview

The system includes specialized optimization for RL agents designed specifically for trading applications, moving beyond generic regression to real trading scenarios.

### Supported RL Agents

#### 1. **SAC (Soft Actor-Critic) Trading**
```python
# Optimized parameters:
{
    'hidden_dims': [[128, 128], [256, 256], [512, 256], [256, 256, 128]],
    'activation': ['relu', 'tanh', 'elu'],
    'gamma': (0.95, 0.999),           # Discount factor
    'tau': (0.001, 0.02),             # Soft update rate
    'alpha': (0.05, 0.3),             # Entropy coefficient
    'batch_size': [64, 128, 256],
    'learning_rate': (1e-5, 1e-2),
    'risk_factor': (0.01, 0.1),       # Trading-specific
    'reward_scaling': (0.1, 2.0),     # Trading-specific
    'exploration_noise': (0.05, 0.3)  # Trading-specific
}
```

#### 2. **TD3 (Twin Delayed DDPG) Trading**
```python
# Optimized parameters:
{
    'actor_hidden': [[256, 256], [400, 300], [512, 256]],
    'critic_hidden': [[256, 256], [400, 300], [512, 256]],
    'gamma': (0.95, 0.999),
    'tau': (0.001, 0.01),
    'policy_noise': (0.1, 0.3),       # TD3-specific
    'noise_clip': (0.3, 0.7),         # TD3-specific
    'policy_delay': (1, 4),           # TD3-specific
    'max_action': (0.5, 2.0),         # Trading action scale
    'exploration_noise': (0.05, 0.2),
    'action_noise': (0.1, 0.3)
}
```

#### 3. **Rainbow DQN Trading**
```python
# Optimized parameters:
{
    'hidden_size': [256, 512, 1024],
    'num_layers': (2, 4),
    'double_dqn': [True, False],       # Rainbow component
    'dueling': [True, False],          # Rainbow component
    'noisy': [True, False],            # Rainbow component
    'n_steps': (1, 5),                 # Multi-step learning
    'num_atoms': [51, 101],            # Distributional RL
    'v_min': (-10, -1),                # Value distribution
    'v_max': (1, 10),                  # Value distribution
    'epsilon_decay': (500, 2000)       # Exploration decay
}
```

### Trading Environment Simulator

The system includes a sophisticated trading environment simulator:

#### Features:
- **Synthetic Market Data**: Realistic OHLCV data generation
- **Technical Indicators**: SMA, RSI, MACD, Bollinger Bands
- **Transaction Costs**: Realistic trading fees and slippage
- **Position Management**: Long/short positions with risk limits
- **Performance Metrics**: Trading-specific evaluation

#### Trading Metrics:
```python
{
    'sharpe_ratio': 0.847,      # Risk-adjusted returns
    'total_return': 0.234,      # Total portfolio return
    'max_drawdown': -0.156,     # Maximum loss from peak
    'win_rate': 0.621,          # Percentage of profitable trades
    'calmar_ratio': 1.501,      # Return/Max Drawdown
    'volatility': 0.198,        # Return volatility
    'num_trades': 142           # Total number of trades
}
```

### Usage Example: RL Trading Optimization

```python
from utils.hyperopt import HyperparameterOptimizer
from utils.trading_rl_optimizer import create_synthetic_trading_data

# Create trading data
trading_data = create_synthetic_trading_data(n_points=500)

# Initialize optimizer
optimizer = HyperparameterOptimizer()

# Optimize RL agents for trading
results = optimizer.optimize_rl_agents(
    X_train=training_features,
    y_train=training_targets,
    X_val=validation_features,
    y_val=validation_targets,
    n_trials=50  # Number of optimization trials
)

# Results example:
{
    'SAC_Trading': {
        'params': {'gamma': 0.987, 'hidden_dims': [256, 256], ...},
        'score': 0.834,  # Sharpe ratio
        'model_type': 'reinforcement_learning_trading',
        'metric': 'sharpe_ratio'
    },
    'TD3_Trading': {
        'params': {'actor_hidden': [400, 300], 'gamma': 0.991, ...},
        'score': 0.762,
        'model_type': 'reinforcement_learning_trading',
        'metric': 'sharpe_ratio'
    }
}
```

## 📈 Advanced Models

### Temporal Fusion Transformer (TFT)

Optimized for complex time series forecasting:

```python
# TFT hyperparameters:
{
    'hidden_size': [32, 64, 128, 256],
    'num_attention_heads': [1, 2, 4, 8],
    'dropout': (0.1, 0.5),
    'learning_rate': (1e-5, 1e-2),
    'batch_size': [16, 32, 64],
    'num_layers': (1, 4),
    'context_length': [24, 48, 96],
    'prediction_length': [1, 12, 24]
}
```

### PatchTST (Patch Time Series Transformer)

State-of-the-art time series model:

```python
# PatchTST hyperparameters:
{
    'patch_len': [12, 16, 24],
    'stride': [6, 8, 12],
    'd_model': [64, 128, 256],
    'n_heads': [4, 8, 16],
    'e_layers': [2, 3, 4],
    'dropout': (0.1, 0.3),
    'learning_rate': (1e-5, 1e-2)
}
```

## 🔧 Configuration & Customization

### Custom Parameter Spaces

You can extend the system with custom parameter spaces:

```python
# Add custom model optimization
def optimize_custom_model(trial):
    params = {
        'custom_param1': trial.suggest_float('custom_param1', 0.1, 1.0),
        'custom_param2': trial.suggest_int('custom_param2', 10, 100),
        'custom_param3': trial.suggest_categorical('custom_param3', ['a', 'b', 'c'])
    }
    
    # Your model training and evaluation logic
    model = CustomModel(**params)
    score = evaluate_model(model, X_train, y_train, X_val, y_val)
    
    return score
```

### Environment Variables

```bash
# Configure optimization behavior
export OPTUNA_TRIALS=100           # Number of trials per model
export TRADING_EPISODES=5          # Episodes for RL evaluation
export RANDOM_SEED=42              # Reproducibility seed
export OPTIMIZATION_TIMEOUT=3600   # Max optimization time (seconds)
```

## 📊 Results Analysis

### Output Format

Optimization results are returned in a structured format:

```python
{
    'model_name': {
        'params': {...},              # Best hyperparameters found
        'score': 0.845,              # Best validation score
        'model_type': 'sklearn',      # Model category
        'metric': 'r2_score',        # Optimization metric
        'trials': 50,                # Number of trials completed
        'optimization_time': 120.5,  # Time taken (seconds)
        'best_trial_number': 23      # Trial number of best result
    }
}
```

### Comparison Tools

```python
# Compare multiple optimization results
from utils.hyperopt import compare_optimization_results

results = {
    'XGBoost': {'score': 0.892, 'params': {...}},
    'LightGBM': {'score': 0.887, 'params': {...}},
    'CatBoost': {'score': 0.901, 'params': {...}}
}

comparison = compare_optimization_results(results)
print(comparison.get_summary())
```

## 🚀 Performance Tips

### 1. **Parallel Optimization**
```python
# Enable parallel trials
optimizer = HyperparameterOptimizer(n_jobs=-1)  # Use all CPU cores
```

### 2. **Early Stopping**
```python
# Stop optimization early if no improvement
optimizer.optimize_with_early_stopping(
    patience=10,        # Trials without improvement
    min_improvement=0.001  # Minimum improvement threshold
)
```

### 3. **Memory Management**
```python
# For large datasets, use data sampling
X_sample = X_train.sample(n=10000)  # Sample for faster optimization
y_sample = y_train.sample(n=10000)
```

### 4. **Custom Metrics**
```python
# Define custom evaluation metrics
def custom_trading_metric(y_true, y_pred):
    # Your custom metric logic
    return custom_score

optimizer.set_custom_metric(custom_trading_metric)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. **Missing Dependencies**
```bash
# Error: Module 'optuna' not found
pip install optuna

# Error: RL agents not available
pip install stable-baselines3 gym torch
```

#### 2. **Memory Issues**
```python
# Reduce batch size for large models
params['batch_size'] = 32  # Instead of 256

# Use data sampling
X_train_sample = X_train[:10000]
```

#### 3. **Optimization Timeout**
```python
# Set shorter optimization time
optimizer.optimize_sklearn(n_trials=10)  # Instead of 50
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

optimizer = HyperparameterOptimizer(debug=True)
```

## 📚 API Reference

### HyperparameterOptimizer Class

```python
class HyperparameterOptimizer:
    def __init__(self, 
                 console=None,           # Rich console for output
                 random_seed=42,         # Reproducibility seed
                 n_jobs=1):              # Parallel jobs
        
    def optimize_sklearn(self, X_train, y_train, X_val, y_val, n_trials=20)
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, n_trials=30)
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=30)
    def optimize_catboost(self, X_train, y_train, X_val, y_val, n_trials=30)
    def optimize_pytorch(self, X_train, y_train, X_val, y_val, n_trials=25)
    def optimize_tft(self, X_train, y_train, X_val, y_val, n_trials=25)
    def optimize_patchtst(self, X_train, y_train, X_val, y_val, n_trials=25)
    def optimize_rl_agents(self, X_train, y_train, X_val, y_val, n_trials=25)
    def optimize_specific_models(self, model_types, X_train, y_train, X_val, y_val, n_trials=20)
```

### TradingEnvironmentSimulator Class

```python
class TradingEnvironmentSimulator:
    def __init__(self, 
                 data,                   # Trading data DataFrame
                 initial_balance=10000,  # Starting capital
                 transaction_cost=0.001, # Trading fees
                 max_position=1.0):      # Position size limit
        
    def simulate_trading(self, agent, num_episodes=1)
    def get_state(self, step)
    def step(self, action)
    def reset()
```

## 🎯 Future Enhancements

### Planned Features

1. **Multi-Objective Optimization**: Optimize for multiple metrics simultaneously
2. **Distributed Computing**: Scale across multiple machines
3. **Real-Time Optimization**: Continuous learning during live trading
4. **Model Ensemble**: Automatic ensemble creation from optimized models
5. **A/B Testing Framework**: Compare strategies in production
6. **Risk Management**: Advanced position sizing and risk controls

### Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*For more information, visit our [GitHub repository](https://github.com/hyperion-project) or contact the development team.*

## 🎉 **LATEST UPDATES - JUNE 2025**

### ✅ **Complete RL Trading Agent System**
All reinforcement learning trading agents are now **fully operational** with real trading simulations:

- **SAC (Soft Actor-Critic)**: ⭐ **Best performing agent** - Fixed continuous action issues, now produces positive Sharpe ratios
- **TD3 (Twin Delayed DDPG)**: Robust continuous control agent for trading
- **RainbowDQN**: Advanced discrete action agent with distributional learning

### ✅ **Real Trading Metrics**
- **Sharpe Ratio**: Risk-adjusted returns calculation
- **Portfolio Returns**: Actual profit/loss from trading simulation
- **Maximum Drawdown**: Risk management metrics
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Return vs maximum drawdown

### ✅ **Fixed Critical Issues**
- **SAC Action Dimension**: Fixed mismatch between continuous (1D) and discrete (3D) actions
- **State Space Handling**: Proper interface between optimization framework and RL agents
- **Trading Environment**: Robust simulator with realistic market conditions
- **Error Handling**: Graceful failure recovery for all model types
