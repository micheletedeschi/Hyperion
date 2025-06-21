# 🏗️ Hyperion Architecture Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Module Structure](#module-structure)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Model Integration](#model-integration)
5. [Training Pipeline](#training-pipeline)
6. [MLOps Infrastructure](#mlops-infrastructure)
7. [Configuration Management](#configuration-management)
8. [Extension Points](#extension-points)

## System Overview

Hyperion is built as a modular, extensible framework for algorithmic trading using machine learning and reinforcement learning. The architecture follows clean separation of concerns with distinct layers for data processing, model training, evaluation, and deployment.

### Core Design Principles

- **Modularity**: Each component can be used independently
- **Extensibility**: Easy to add new models, data sources, or strategies
- **Configurability**: Everything controlled through configuration files
- **Reproducibility**: Complete experiment tracking and versioning
- **Performance**: Optimized for both CPU and GPU environments

## Module Structure

```
hyperion3/
├── config/              # Configuration management
│   ├── base_config.py   # Core configuration classes
│   ├── model_configs.py # Model-specific configurations
│   └── __init__.py
├── data/                # Data processing pipeline
│   ├── downloader.py    # Market data acquisition
│   ├── preprocessor.py  # Data cleaning and preparation
│   ├── feature_engineering.py # Feature creation
│   └── augmentation.py  # Data augmentation
├── models/              # Model implementations
│   ├── transformers/    # Transformer-based models
│   │   ├── patchtst.py  # PatchTST implementation
│   │   └── tft.py       # Temporal Fusion Transformer
│   ├── rl_agents/       # Reinforcement Learning agents
│   │   ├── sac.py       # Soft Actor-Critic
│   │   ├── td3.py       # Twin Delayed DDPG
│   │   └── rainbow_dqn.py # Rainbow DQN
│   ├── ensemble/        # Ensemble methods
│   └── base.py          # Base model classes
├── training/            # Training infrastructure
│   ├── trainer.py       # Main training orchestrator
│   ├── validators.py    # Cross-validation utilities
│   └── callbacks.py     # Training callbacks
├── optimization/        # Hyperparameter optimization
│   ├── flaml_optimizer.py # FLAML integration
│   └── meta_learning.py # Meta-learning algorithms
├── evaluations/         # Model evaluation
│   ├── backtester.py    # Backtesting engine
│   ├── metrics.py       # Financial metrics
│   └── validators.py    # Model validation
└── utils/               # Utility functions
    ├── data_utils.py    # Data manipulation utilities
    ├── model_utils.py   # Model saving/loading
    ├── metrics.py       # Metric calculations
    └── mlops.py         # MLOps utilities
```

## Data Flow Architecture

### 1. Data Acquisition Layer
```python
# Market data flows from exchanges through the downloader
Exchange API → Downloader → Raw Data Storage → Preprocessor
```

**Components:**
- **Downloader**: Fetches OHLCV data from multiple exchanges (Binance, etc.)
- **Storage**: Local caching for efficient data access
- **Validation**: Data quality checks and missing value handling

### 2. Feature Engineering Pipeline
```python
Raw OHLCV → Technical Indicators → Pattern Recognition → Augmentation → ML-Ready Features
```

**Feature Categories:**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Price Patterns**: Candlestick patterns, support/resistance levels
- **Volume Analysis**: OBV, Volume Profile, VWAP
- **Statistical Features**: Rolling statistics, volatility measures
- **Temporal Features**: Time-based cyclical features

### 3. Model Training Flow
```python
Features → Data Split → Hyperparameter Optimization → Model Training → Validation → Model Storage
```

## Model Integration

### Model Factory Pattern
All models are created through the `ModelFactory` class, providing a unified interface:

```python
from hyperion3.models import ModelFactory, ModelType

factory = ModelFactory(config)
model = factory.create_model(ModelType.PATCHTST)
```

### Supported Model Types

#### Classical ML Models
- **Sklearn Models**: RandomForest, GradientBoosting, SVM, etc.
- **Ensemble Models**: XGBoost, LightGBM, CatBoost
- **Statistical Models**: Prophet, ARIMA

#### Deep Learning Models
- **Transformers**: PatchTST, TFT (Temporal Fusion Transformer)
- **RNNs**: LSTM, GRU
- **CNNs**: 1D Convolutional networks for time series

#### Reinforcement Learning Agents
- **SAC**: Soft Actor-Critic for continuous action spaces
- **TD3**: Twin Delayed DDPG with improved stability
- **Rainbow DQN**: Enhanced Deep Q-Network with multiple improvements

### Model Interface
All models implement a common interface:

```python
class BaseModel:
    def train(self, train_data, val_data=None, **kwargs):
        """Train the model"""
        pass
    
    def predict(self, data):
        """Make predictions"""
        pass
    
    def evaluate(self, test_data):
        """Evaluate model performance"""
        pass
    
    def save(self, path):
        """Save model to disk"""
        pass
    
    def load(self, path):
        """Load model from disk"""
        pass
```

## Training Pipeline

### 1. Data Preparation
```python
# Automatic data splitting with temporal validation
train_data, val_data, test_data = prepare_temporal_split(data, config)
```

### 2. Hyperparameter Optimization
```python
# FLAML-powered optimization
optimizer = FLAMLOptimizer(config)
best_params = optimizer.optimize(model_type, train_data, val_data)
```

### 3. Model Training
```python
# Unified training interface
trainer = Trainer(config)
model = await trainer.train(train_data, val_data, model_type)
```

### 4. Evaluation and Backtesting
```python
# Comprehensive evaluation
backtester = Backtester(config)
results = backtester.evaluate(model, test_data)
```

## MLOps Infrastructure

### Experiment Tracking
- **MLflow Integration**: Automatic experiment logging
- **Metrics Tracking**: Performance metrics, model parameters
- **Artifact Storage**: Models, plots, configurations
- **Version Control**: Model versioning and lineage

### Model Management
```python
# Automatic model versioning and storage
mlops = MLOpsManager(config)
experiment_id = mlops.start_experiment(model_name, params)
mlops.log_metrics(metrics)
mlops.save_model(model, experiment_id)
```

### Monitoring and Alerts
- **Performance Monitoring**: Real-time metric tracking
- **Drift Detection**: Data and model drift detection
- **Alert System**: Configurable thresholds and notifications

## Configuration Management

### Hierarchical Configuration
The system uses a hierarchical configuration system:

```python
@dataclass
class HyperionV2Config:
    # Sub-configurations
    api: APIConfig
    data: DataConfig
    transformer: TransformerConfig
    rl: RLConfig
    optimization: OptimizationConfig
    trading: TradingConfig
    mlops: MLOpsConfig
```

### Configuration Sources
1. **Default Values**: Sensible defaults for all parameters
2. **Config Files**: JSON/YAML configuration files
3. **Environment Variables**: Runtime overrides
4. **CLI Arguments**: Command-line parameter overrides

### Example Configuration
```json
{
  "data": {
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "lookback_window": 168,
    "augmentation_techniques": ["noise", "scaling"]
  },
  "models": {
    "enabled_models": ["lightgbm", "xgboost", "catboost", "patchtst", "sac"]
  },
  "optimization": {
    "metric": "sharpe_ratio",
    "time_budget": 3600,
    "method": "flaml"
  }
}
```

## Extension Points

### Adding New Models
1. **Implement Base Interface**: Extend `BaseModel` or `BaseTradingModel`
2. **Register in Factory**: Add to `ModelFactory`
3. **Add Configuration**: Define model-specific config class
4. **Add Tests**: Implement unit and integration tests

### Adding New Data Sources
1. **Implement Downloader**: Extend `BaseDownloader`
2. **Add Configuration**: Define data source config
3. **Register in Pipeline**: Add to data processing pipeline

### Adding New Metrics
1. **Implement Metric**: Add to `metrics.py`
2. **Add to Evaluation**: Include in backtesting and evaluation
3. **Add to Reporting**: Include in experiment reports

### Custom Strategies
```python
class CustomStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
    
    def generate_signals(self, data):
        # Custom signal generation logic
        pass
    
    def manage_risk(self, position, market_data):
        # Custom risk management
        pass
```

## Performance Considerations

### Optimization Strategies
- **GPU Acceleration**: Automatic GPU detection and usage
- **Parallel Processing**: Multi-core optimization for CPU-bound tasks
- **Memory Management**: Efficient data loading and caching
- **Apple Silicon**: Optimized for M1/M2/M3/m4 processors

### Scaling Guidelines
- **Data Volume**: Efficient handling of large datasets
- **Model Complexity**: Memory-efficient model implementations
- **Training Time**: Optimized training loops and early stopping
- **Inference Speed**: Fast prediction for real-time trading

## Security and Safety

### Risk Management
- **Position Sizing**: Configurable position limits
- **Stop Loss**: Automatic stop-loss implementation
- **Drawdown Limits**: Maximum drawdown protection
- **Paper Trading**: Safe testing environment

### Data Security
- **API Key Management**: Secure credential storage
- **Data Encryption**: Encrypted data storage options
- **Access Control**: User permission management

## Future Architecture Plans

### Planned Enhancements
1. **Distributed Training**: Multi-node training support
2. **Real-time Streaming**: Live data processing pipeline
3. **Multi-Asset**: Cross-asset strategy development
4. **Advanced Ensembles**: Hierarchical ensemble methods
5. **AutoML Enhancements**: Automated feature engineering
6. **Cloud Integration**: Cloud deployment options

This architecture provides a solid foundation for building, testing, and deploying sophisticated trading strategies while maintaining flexibility for future enhancements and customizations.
