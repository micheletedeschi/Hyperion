# 📚 HYPERION API REFERENCE

## Core Classes and Functions

This document provides a comprehensive API reference for the Hyperion Hyperparameter Optimization System.

## 🎯 HyperparameterOptimizer Class

The main optimization engine for all model types.

```python
class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization system supporting 46+ models
    across traditional ML, deep learning, and reinforcement learning.
    """
```

### Constructor

```python
def __init__(self, 
             console=None,           # Rich console for styled output
             random_seed=42,         # Reproducibility seed
             n_jobs=1,              # Parallel jobs (-1 for all cores)
             debug=False):          # Enable debug logging
```

**Parameters:**
- `console` (rich.Console, optional): Rich console for formatted output
- `random_seed` (int): Random seed for reproducible results
- `n_jobs` (int): Number of parallel jobs for optimization
- `debug` (bool): Enable detailed debug logging

**Example:**
```python
from rich.console import Console
from utils.hyperopt import HyperparameterOptimizer

console = Console()
optimizer = HyperparameterOptimizer(console=console, n_jobs=-1, debug=True)
```

### Core Optimization Methods

#### `optimize_sklearn()`

Optimize traditional sklearn models.

```python
def optimize_sklearn(self, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray, 
                    X_val: np.ndarray, 
                    y_val: np.ndarray, 
                    n_trials: int = 20) -> Dict[str, Dict]:
```

**Parameters:**
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training targets  
- `X_val` (np.ndarray): Validation features
- `y_val` (np.ndarray): Validation targets
- `n_trials` (int): Number of optimization trials

**Returns:**
- `Dict[str, Dict]`: Optimization results for each model

**Models Optimized (36 total):**
- **Ensemble**: RandomForest, GradientBoosting, ExtraTrees, AdaBoost, Bagging, HistGradientBoosting
- **Linear**: Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression, HuberRegressor
- **SVM**: SVR, NuSVR, LinearSVR
- **Neighbors**: KNeighborsRegressor, RadiusNeighborsRegressor
- **Neural**: MLPRegressor
- **Trees**: DecisionTree, ExtraTree
- **Gaussian**: GaussianProcessRegressor
- **Other**: KernelRidge, PLSRegression, and more

**Example:**
```python
results = optimizer.optimize_sklearn(X_train, y_train, X_val, y_val, n_trials=30)

# Access results
best_rf_params = results['RandomForestRegressor']['params']
best_rf_score = results['RandomForestRegressor']['score']
```

#### `optimize_xgboost()`

Optimize XGBoost with extensive hyperparameter search.

```python
def optimize_xgboost(self, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray, 
                    X_val: np.ndarray, 
                    y_val: np.ndarray, 
                    n_trials: int = 30) -> Dict[str, Dict]:
```

**Optimized Parameters:**
```python
{
    'n_estimators': (50, 1000),
    'max_depth': (3, 12),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0, 10),
    'reg_lambda': (1, 10),
    'gamma': (0, 5),
    'min_child_weight': (1, 10)
}
```

#### `optimize_lightgbm()`

Optimize LightGBM with gradient boosting specific parameters.

```python
def optimize_lightgbm(self, 
                     X_train: np.ndarray, 
                     y_train: np.ndarray, 
                     X_val: np.ndarray, 
                     y_val: np.ndarray, 
                     n_trials: int = 30) -> Dict[str, Dict]:
```

**Optimized Parameters:**
```python
{
    'n_estimators': (50, 1000),
    'max_depth': (3, 15),
    'learning_rate': (0.01, 0.3),
    'num_leaves': (10, 300),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0, 10),
    'reg_lambda': (0, 10),
    'min_child_samples': (5, 100)
}
```

#### `optimize_catboost()`

Optimize CatBoost with categorical-aware features.

```python
def optimize_catboost(self, 
                     X_train: np.ndarray, 
                     y_train: np.ndarray, 
                     X_val: np.ndarray, 
                     y_val: np.ndarray, 
                     n_trials: int = 30) -> Dict[str, Dict]:
```

**Optimized Parameters:**
```python
{
    'iterations': (100, 1000),
    'depth': (4, 10),
    'learning_rate': (0.01, 0.3),
    'l2_leaf_reg': (1, 10),
    'border_count': (32, 255),
    'bagging_temperature': (0, 1),
    'random_strength': (0, 10),
    'subsample': (0.6, 1.0)
}
```

#### `optimize_pytorch()`

Optimize PyTorch neural network models.

```python
def optimize_pytorch(self, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray, 
                    X_val: np.ndarray, 
                    y_val: np.ndarray, 
                    n_trials: int = 25) -> Dict[str, Dict]:
```

**Models:**
- **SimpleMLP**: Basic multi-layer perceptron
- **DeepMLP**: Deep neural network with dropout
- **LSTMModel**: LSTM for sequence modeling

**Optimized Parameters:**
```python
{
    'hidden_size': [64, 128, 256, 512],
    'num_layers': (1, 5),
    'dropout': (0.1, 0.5),
    'learning_rate': (1e-5, 1e-2),
    'batch_size': [16, 32, 64, 128],
    'weight_decay': (1e-6, 1e-2),
    'epochs': (50, 200)
}
```

#### `optimize_tft()`

Optimize Temporal Fusion Transformer for time series.

```python
def optimize_tft(self, 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                X_val: np.ndarray, 
                y_val: np.ndarray, 
                n_trials: int = 25) -> Dict[str, Dict]:
```

**Optimized Parameters:**
```python
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

#### `optimize_patchtst()`

Optimize PatchTST transformer for time series.

```python
def optimize_patchtst(self, 
                     X_train: np.ndarray, 
                     y_train: np.ndarray, 
                     X_val: np.ndarray, 
                     y_val: np.ndarray, 
                     n_trials: int = 25) -> Dict[str, Dict]:
```

**Optimized Parameters:**
```python
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

#### `optimize_rl_agents()`

Optimize reinforcement learning agents for trading.

```python
def optimize_rl_agents(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray, 
                      X_val: np.ndarray, 
                      y_val: np.ndarray, 
                      n_trials: int = 25) -> Dict[str, Dict]:
```

**Agents:**
- **SAC_Trading**: Soft Actor-Critic for continuous trading
- **TD3_Trading**: Twin Delayed DDPG for robust control  
- **RainbowDQN_Trading**: Advanced DQN for discrete trading

**Returns:**
Results with Sharpe ratio as primary metric instead of R² score.

#### `optimize_specific_models()`

Optimize specific model types or individual models.

```python
def optimize_specific_models(self, 
                           model_types: List[str], 
                           X_train: np.ndarray, 
                           y_train: np.ndarray, 
                           X_val: np.ndarray, 
                           y_val: np.ndarray, 
                           n_trials: int = 20) -> Dict[str, Dict]:
```

**Parameters:**
- `model_types` (List[str]): Model categories or specific models

**Supported Values:**
```python
model_types = [
    'sklearn',      # All sklearn models
    'ensemble',     # XGBoost, LightGBM, CatBoost  
    'pytorch',      # Neural networks
    'advanced',     # TFT, PatchTST, Transformers
    'rl_agents',    # RL trading agents
    'automl',       # FLAML AutoML
    'xgboost',      # XGBoost only
    'lightgbm',     # LightGBM only
    'catboost',     # CatBoost only
    'tft',          # TFT only
    'patchtst',     # PatchTST only
    'sac',          # SAC agent only
    'td3',          # TD3 agent only
    'rainbow'       # Rainbow DQN only
]
```

### Utility Methods

#### `show_model_capabilities()`

Display available models and their status.

```python
def show_model_capabilities(self) -> None:
```

**Output Format:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Category               ┃ Status             ┃ Available Models            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Sklearn                │ ✅ 36/36          │ RF, GB, Ridge, Lasso, SVR...│
│ Ensemble               │ ✅ 3/3            │ XGBoost, LightGBM, CatBoost │
│ Deep_Learning          │ ✅ 3/3            │ SimpleMLP, DeepMLP, LSTM    │
│ Advanced_Models        │ ✅ 3/3            │ TFT, PatchTST, Transformers │
│ Reinforcement_Learning │ ✅ 3/3 TRADING    │ SAC, TD3, RainbowDQN        │
│ AutoML                 │ ✅ 1/3            │ FLAML                       │
└────────────────────────┴────────────────────┴─────────────────────────────┘
```

## 🎮 TradingEnvironmentSimulator Class

Advanced trading environment for RL agent evaluation.

```python
class TradingEnvironmentSimulator:
    """
    Sophisticated trading environment simulator for evaluating RL agents
    with realistic market conditions and trading mechanics.
    """
```

### Constructor

```python
def __init__(self, 
             data: pd.DataFrame,              # Trading data (OHLCV)
             initial_balance: float = 10000,  # Starting capital
             transaction_cost: float = 0.001, # Trading fees (0.1%)
             max_position: float = 1.0):      # Maximum position size
```

**Parameters:**
- `data` (pd.DataFrame): Market data with columns ['open', 'high', 'low', 'close', 'volume']
- `initial_balance` (float): Starting trading capital
- `transaction_cost` (float): Transaction cost as fraction of trade value
- `max_position` (float): Maximum position size as fraction of capital

### Methods

#### `simulate_trading()`

Run complete trading simulation with an RL agent.

```python
def simulate_trading(self, 
                    agent,                    # RL agent with act() method
                    num_episodes: int = 1     # Number of episodes
                    ) -> Dict[str, float]:
```

**Returns:**
```python
{
    'total_return': 0.234,      # Cumulative return (23.4%)
    'sharpe_ratio': 0.847,      # Sharpe ratio (annualized)
    'max_drawdown': -0.156,     # Maximum drawdown (-15.6%)
    'win_rate': 0.621,          # Win rate (62.1%)
    'volatility': 0.198,        # Return volatility (19.8%)
    'calmar_ratio': 1.501,      # Calmar ratio
    'final_balance': 12340.0,   # Final account balance
    'num_trades': 142,          # Total number of trades
    'avg_trade_pnl': 0.0023     # Average P&L per trade
}
```

#### `get_state()`

Get market state features for a given time step.

```python
def get_state(self, step: int) -> np.ndarray:
```

**Returns:**
Array with normalized features:
```python
[
    normalized_price,      # Close price / 100
    normalized_volume,     # Volume / 1e7
    returns,              # Price returns
    sma_5,                # 5-period SMA
    sma_20,               # 20-period SMA  
    rsi,                  # RSI indicator
    price_momentum,       # Price momentum
    volume_ratio,         # Volume ratio
    trend_strength,       # Trend strength
    current_position,     # Current position
    normalized_balance    # Balance / initial_balance
]
```

#### `step()`

Execute one trading step.

```python
def step(self, action: float) -> Tuple[np.ndarray, float, bool]:
```

**Parameters:**
- `action` (float): Trading action (-1 to 1)

**Returns:**
- `next_state` (np.ndarray): Next market state
- `reward` (float): Trading reward
- `done` (bool): Episode completion flag

#### `reset()`

Reset simulator to initial state.

```python
def reset(self) -> None:
```

## 🔧 Utility Functions

### `create_synthetic_trading_data()`

Generate synthetic market data for testing and optimization.

```python
def create_synthetic_trading_data(n_points: int = 300,
                                 volatility: float = 0.02,
                                 trend: float = 0.0001,
                                 random_seed: int = 42) -> pd.DataFrame:
```

**Parameters:**
- `n_points` (int): Number of data points
- `volatility` (float): Daily volatility (default 2%)
- `trend` (float): Daily trend (default 0.01%)
- `random_seed` (int): Random seed

**Returns:**
DataFrame with columns:
```python
['open', 'high', 'low', 'close', 'volume', 'returns']
```

### `evaluate_rl_agent_trading()`

Evaluate RL agent performance on trading task.

```python
def evaluate_rl_agent_trading(agent,
                             data: pd.DataFrame,
                             num_episodes: int = 3) -> Dict[str, float]:
```

**Parameters:**
- `agent`: RL agent with act() method
- `data` (pd.DataFrame): Trading data
- `num_episodes` (int): Number of evaluation episodes

**Returns:**
Trading performance metrics including adjusted Sharpe ratio.

## 📊 Data Structures

### Optimization Result Format

```python
OptimizationResult = {
    'model_name': {
        'params': Dict[str, Any],        # Best hyperparameters
        'score': float,                  # Best validation score
        'model_type': str,               # Model category
        'metric': str,                   # Optimization metric
        'trials': int,                   # Number of trials
        'optimization_time': float,      # Time taken (seconds)
        'best_trial_number': int,        # Best trial index
        'validation_scores': List[float], # All trial scores
        'parameter_importance': Dict[str, float]  # Parameter importance
    }
}
```

### Trading Metrics Format

```python
TradingMetrics = {
    'total_return': float,          # Cumulative return
    'sharpe_ratio': float,          # Risk-adjusted return
    'max_drawdown': float,          # Maximum loss from peak
    'win_rate': float,              # Profitable trades ratio
    'volatility': float,            # Return volatility
    'calmar_ratio': float,          # Return / Max Drawdown
    'final_balance': float,         # Final account value
    'num_trades': int,              # Total trades executed
    'avg_trade_pnl': float,         # Average trade profit
    'avg_trade_duration': float,    # Average holding period
    'profit_factor': float,         # Gross profit / Gross loss
    'sortino_ratio': float          # Downside risk-adjusted return
}
```

## 🎯 Configuration Constants

### Available Models

```python
# Sklearn models (36 total)
SKLEARN_MODELS = [
    'RandomForestRegressor', 'GradientBoostingRegressor', 'ExtraTreesRegressor',
    'AdaBoostRegressor', 'BaggingRegressor', 'HistGradientBoostingRegressor',
    'Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge', 'ARDRegression',
    'HuberRegressor', 'TheilSenRegressor', 'RANSACRegressor',
    'SVR', 'NuSVR', 'LinearSVR', 'KNeighborsRegressor',
    'MLPRegressor', 'DecisionTreeRegressor', 'ExtraTreeRegressor',
    'GaussianProcessRegressor', 'KernelRidge', 'PLSRegression',
    # ... and more
]

# Ensemble models
ENSEMBLE_MODELS = ['XGBoost', 'LightGBM', 'CatBoost']

# Deep learning models  
PYTORCH_MODELS = ['SimpleMLP', 'DeepMLP', 'LSTM']

# Advanced models
ADVANCED_MODELS = ['TFT', 'PatchTST', 'Transformers']

# RL agents
RL_AGENTS = ['SAC_Trading', 'TD3_Trading', 'RainbowDQN_Trading']

# AutoML
AUTOML_MODELS = ['FLAML']
```

### Default Hyperparameter Ranges

```python
DEFAULT_RANGES = {
    'learning_rate': (1e-5, 1e-2),      # Log scale
    'batch_size': [16, 32, 64, 128, 256],
    'hidden_size': [64, 128, 256, 512, 1024],
    'dropout': (0.1, 0.5),
    'weight_decay': (1e-6, 1e-2),
    'gamma': (0.95, 0.999),              # RL discount factor
    'tau': (0.001, 0.02),                # RL soft update
    'n_estimators': (50, 1000),          # Tree models
    'max_depth': (3, 15)                 # Tree depth
}
```

## 🚨 Error Handling

### Exception Types

```python
class HyperionOptimizationError(Exception):
    """Base exception for optimization errors"""
    pass

class ModelNotAvailableError(HyperionOptimizationError):
    """Raised when a model is not available due to missing dependencies"""
    pass

class DataValidationError(HyperionOptimizationError):
    """Raised when input data is invalid"""
    pass

class OptimizationTimeoutError(HyperionOptimizationError):
    """Raised when optimization exceeds time limit"""
    pass
```

### Error Handling Example

```python
try:
    results = optimizer.optimize_sklearn(X_train, y_train, X_val, y_val)
except ModelNotAvailableError as e:
    print(f"Model not available: {e}")
    # Install missing dependencies
except DataValidationError as e:
    print(f"Invalid data: {e}")
    # Fix data issues
except OptimizationTimeoutError as e:
    print(f"Optimization timeout: {e}")
    # Reduce n_trials or complexity
```

## 🔧 Advanced Configuration

### Custom Parameter Spaces

```python
# Define custom parameter space
def custom_parameter_space(trial):
    return {
        'custom_param1': trial.suggest_float('custom_param1', 0.1, 1.0),
        'custom_param2': trial.suggest_int('custom_param2', 10, 100),
        'custom_param3': trial.suggest_categorical('custom_param3', ['a', 'b'])
    }

# Add to optimizer
optimizer.add_custom_space('MyModel', custom_parameter_space)
```

### Custom Metrics

```python
# Define custom evaluation metric
def custom_metric(y_true, y_pred):
    # Your custom logic here
    return custom_score

# Set custom metric
optimizer.set_custom_metric(custom_metric)
```

### Parallel Configuration

```python
# Configure parallel optimization
optimizer = HyperparameterOptimizer(
    n_jobs=-1,                    # Use all CPU cores
    parallel_backend='threading'   # or 'multiprocessing'
)
```

## 📚 Examples

### Complete Optimization Pipeline

```python
from utils.hyperopt import HyperparameterOptimizer
from utils.trading_rl_optimizer import create_synthetic_trading_data
import numpy as np

# Generate synthetic data
trading_data = create_synthetic_trading_data(1000)
X_train = np.random.randn(800, 10)
y_train = np.random.randn(800)
X_val = np.random.randn(200, 10)  
y_val = np.random.randn(200)

# Initialize optimizer
optimizer = HyperparameterOptimizer(n_jobs=-1)

# Optimize all model types
all_results = {}

# Traditional ML
sklearn_results = optimizer.optimize_sklearn(X_train, y_train, X_val, y_val)
all_results.update(sklearn_results)

# Ensemble methods
ensemble_results = optimizer.optimize_specific_models(
    ['xgboost', 'lightgbm', 'catboost'], 
    X_train, y_train, X_val, y_val
)
all_results.update(ensemble_results)

# RL agents for trading
rl_results = optimizer.optimize_rl_agents(X_train, y_train, X_val, y_val)
all_results.update(rl_results)

# Find best overall model
best_model = max(all_results.items(), key=lambda x: x[1]['score'])
print(f"Best model: {best_model[0]} with score: {best_model[1]['score']:.4f}")
```

---

**📖 This API reference covers all major components of the Hyperion optimization system. For more examples, see the documentation and example notebooks.**
