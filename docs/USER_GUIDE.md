# 📚 Hyperion3 User Guide

A comprehensive guide to understanding and using all features of Hyperion3.

## Table of Contents
1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Model Types and Categories](#model-types-and-categories)
4. [Training Workflows](#training-workflows)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Ensemble Methods](#ensemble-methods)
7. [Analysis and Evaluation](#analysis-and-evaluation)
8. [Configuration Management](#configuration-management)
9. [MLOps and Tracking](#mlops-and-tracking)
10. [Advanced Features](#advanced-features)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

## System Overview

Hyperion3 is a comprehensive algorithmic trading framework that combines:
- **Machine Learning**: Traditional ML models (sklearn, ensemble methods)
- **Deep Learning**: Neural networks and transformer models
- **Reinforcement Learning**: Trading agents that learn from market interactions
- **AutoML**: Automated hyperparameter optimization
- **MLOps**: Experiment tracking and model management

### Core Philosophy
- **Modularity**: Each component can be used independently
- **Reproducibility**: Complete experiment tracking
- **Flexibility**: Easy to extend and customize
- **Performance**: Optimized for both research and production

## Getting Started

### First Launch
```bash
# Start the professional interface
python main_professional.py
```

You'll see the main menu with these options:
- 🤖 **MODELS**: Train individual models
- 🎯 **HYPERPARAMETERS**: Optimize model parameters
- 🎭 **ENSEMBLES**: Combine multiple models
- 📊 **ANALYSIS**: View results and metrics
- ⚙️ **CONFIGURATION**: System settings
- 📈 **MONITORING**: Real-time tracking

### Your First Model Training Session

1. **Select Models Menu** (`🤖 MODELS`)
2. **Choose a Category** (start with `ensemble`)
3. **Train Your First Model** (`e1` for XGBoost)
4. **View Results** in the analysis section

## Model Types and Categories

### 📊 Sklearn Models (s1-s7)
Traditional machine learning models from scikit-learn:

| Code | Model | Best For | Training Time |
|------|--------|----------|---------------|
| `s1` | Random Forest | Stability, interpretability | 30-60s |
| `s2` | Gradient Boosting | Balanced performance | 45-90s |
| `s3` | Extra Trees | Fast training, good baseline | 20-40s |
| `s4` | Ridge Regression | Linear relationships | 5-15s |
| `s5` | Lasso Regression | Feature selection | 5-15s |
| `s6` | Elastic Net | Regularized linear models | 5-15s |
| `s7` | MLP (sklearn) | Simple neural networks | 60-120s |

### 🌟 Ensemble Models (e1-e4)
High-performance gradient boosting models:

| Code | Model | Strengths | Training Time |
|------|--------|-----------|---------------|
| `e1` | XGBoost | Industry standard, reliable | 30-90s |
| `e2` | LightGBM | Fastest training, efficient | 20-60s |
| `e3` | CatBoost | Robust, handles categories well | 60-120s |
| `e4` | Voting Ensemble | Combines multiple models | 90-180s |

### 🧠 PyTorch Models (p1-p4)
Neural networks implemented in PyTorch:

| Code | Model | Description | Training Time |
|------|--------|-------------|---------------|
| `p1` | Simple MLP | Basic feedforward network | 2-5min |
| `p2` | Deep MLP | Multi-layer neural network | 3-8min |
| `p3` | LSTM | Recurrent network for sequences | 5-15min |
| `p4` | 1D CNN | Convolutional network for time series | 3-10min |

### 🤖 AutoML Models (a1-a2)
Automated machine learning with optimization:

| Code | Model | Description | Training Time |
|------|--------|-------------|---------------|
| `a1` | FLAML Auto | Microsoft's AutoML framework | 10-60min |
| `a2` | Optuna Ensemble | Hyperparameter optimization | 15-90min |

### 🎯 Advanced Models (adv1-adv5)
State-of-the-art models for expert users:

| Code | Model | Type | Description | Training Time |
|------|--------|------|-------------|---------------|
| `adv1` | TFT | Transformer | Temporal Fusion Transformer | 15-45min |
| `adv2` | PatchTST | Transformer | Patch-based time series transformer | 10-30min |
| `adv3` | SAC | RL Agent | Soft Actor-Critic for trading | 20-60min |
| `adv4` | TD3 | RL Agent | Twin Delayed DDPG | 25-75min |
| `adv5` | Rainbow DQN | RL Agent | Enhanced Deep Q-Network | 30-90min |

## Training Workflows

### Beginner Workflow: Start Simple
```
Day 1: Learn the Basics
├── Train s1 (Random Forest) → Understand R² scores
├── Train e1 (XGBoost) → Compare performance
└── Analysis → Review results

Day 2: Explore Ensemble
├── Train e1, e2, e3 → Multiple models
├── Ensembles → Create voting ensemble
└── Compare individual vs ensemble performance

Day 3: Optimization
├── Hyperparameters → Auto optimization
├── Train optimized models
└── Compare before/after optimization
```

### Intermediate Workflow: Systematic Testing
```
Week 1: Foundation
├── Train all ensemble models (e1-e4)
├── Train all sklearn models (s1-s7)
├── Create voting ensembles
└── Document best performers

Week 2: Deep Learning
├── Train PyTorch models (p1-p4)
├── Compare with traditional ML
├── Experiment with different architectures
└── Optimize neural network hyperparameters

Week 3: AutoML
├── Run FLAML optimization (a1)
├── Compare AutoML vs manual tuning
├── Create optimized ensembles
└── Validate on different time periods
```

### Advanced Workflow: Research and Production
```
Month 1: Model Development
├── Train all model categories
├── Comprehensive hyperparameter optimization
├── Advanced ensemble methods (stacking, blending)
├── Cross-validation and robustness testing
└── Feature engineering experiments

Month 2: Strategy Development
├── Multi-timeframe analysis
├── Risk management integration
├── Portfolio optimization
├── Walk-forward validation
└── Paper trading implementation

Month 3: Production
├── Real-time data integration
├── Model monitoring and retraining
├── Performance tracking
├── Risk monitoring
└── Continuous improvement
```

## Hyperparameter Optimization

### Understanding Hyperparameter Optimization
Hyperparameters are settings that control how a model learns. Examples:
- **Learning rate**: How fast the model learns
- **Regularization**: How much to prevent overfitting
- **Tree depth**: Complexity of tree-based models
- **Batch size**: Number of samples processed together

### Optimization Methods in Hyperion

#### Auto Optimization (Recommended for Beginners)
```
Hyperparameters Menu → Auto Optimization
```
- Automatically finds best parameters
- Uses intelligent search algorithms
- Typically runs 50-200 trials
- Takes 30-120 minutes depending on model

#### Manual Optimization (Advanced Users)
```
Hyperparameters Menu → Manual Optimization
```
- Choose specific models to optimize
- Control number of trials
- Set custom parameter ranges
- More control but requires expertise

#### Category Optimization
```
Hyperparameters Menu → Optimize Category
```
- Optimize all models in a category
- Useful for comparing model families
- Can run overnight for comprehensive results

### Best Practices for Optimization

1. **Start with Fast Models**: Optimize ensemble models first
2. **Use Sufficient Trials**: 50+ for simple models, 200+ for complex ones
3. **Monitor Progress**: Check intermediate results
4. **Validate Results**: Ensure optimized models generalize well
5. **Save Results**: Hyperion automatically saves optimized parameters

## Ensemble Methods

### Why Use Ensembles?
Ensembles combine multiple models to:
- **Reduce Overfitting**: Multiple models cancel out individual errors
- **Improve Robustness**: Less sensitive to market changes
- **Increase Accuracy**: Often outperform individual models
- **Provide Confidence**: Multiple predictions increase reliability

### Ensemble Types in Hyperion

#### 1. Voting Ensemble (Simple)
```
Ensembles Menu → Auto Ensemble → Voting
```
- Takes average of model predictions
- Equal weight to each model
- Simple and effective
- Good starting point

#### 2. Weighted Ensemble (Intermediate)
```
Ensembles Menu → Specific Ensemble → Weighted
```
- Different weights based on model performance
- Better models get higher weights
- Automatically calculated from validation scores
- Usually outperforms simple voting

#### 3. Stacking Ensemble (Advanced)
```
Ensembles Menu → Specific Ensemble → Stacking
```
- Uses a meta-model to combine predictions
- Learns optimal combination strategy
- Most sophisticated method
- Often achieves best performance

#### 4. Bagging Ensemble (Expert)
```
Ensembles Menu → Specific Ensemble → Bagging
```
- Trains multiple versions of same model
- Uses different data subsets
- Reduces variance and overfitting
- Good for unstable models

### Ensemble Best Practices

1. **Diverse Models**: Combine different types (tree + neural network + RL)
2. **Quality Base Models**: Start with well-performing individual models
3. **Avoid Correlation**: Don't combine too similar models
4. **Validate Thoroughly**: Test on unseen data
5. **Monitor Complexity**: More models ≠ always better

## Analysis and Evaluation

### Understanding Model Metrics

#### Regression Metrics
- **R² Score**: Proportion of variance explained (0-1, higher better)
  - 0.7+ = Excellent
  - 0.5-0.7 = Good
  - 0.3-0.5 = Acceptable
  - <0.3 = Poor

- **MSE (Mean Squared Error)**: Average squared prediction error (lower better)
- **MAE (Mean Absolute Error)**: Average absolute prediction error (lower better)
- **RMSE**: Square root of MSE, same units as target (lower better)

#### Trading Metrics
- **Sharpe Ratio**: Risk-adjusted returns (higher better)
  - 2.0+ = Excellent
  - 1.0-2.0 = Good
  - 0.5-1.0 = Acceptable
  - <0.5 = Poor

- **Max Drawdown**: Worst loss period (closer to 0 better)
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Analysis Workflows

#### Quick Analysis
```
Analysis Menu → Models Analysis
```
- View top 5 performing models
- Compare R² scores and metrics
- Identify best performers quickly

#### Comprehensive Analysis
```
Analysis Menu → Compare Models
```
- Detailed comparison tables
- Performance rankings
- Statistical significance tests
- Export results for further analysis

#### Visual Analysis
```
Analysis Menu → Generate Plots
```
- Performance charts
- Prediction vs actual plots
- Feature importance visualizations
- Learning curves

## Configuration Management

### Config File Structure
The `config.json` file controls all system behavior:

```json
{
  "data": {
    "symbols": ["BTC/USDT"],
    "timeframe": "1h",
    "start_date": "2023-01-01",
    "end_date": "2024-06-01",
    "lookback_window": 168,
    "prediction_horizon": 24
  },
  "models": {
    "enabled_models": ["lightgbm", "xgboost", "catboost"],
    "ensemble": {
      "enabled": true,
      "method": "voting",
      "top_n": 3
    }
  },
  "optimization": {
    "metric": "sharpe_ratio",
    "time_budget": 3600,
    "n_splits": 5
  },
  "trading": {
    "initial_balance": 10000.0,
    "strategy": "long_short",
    "risk_factor": 0.2
  }
}
```

### Key Configuration Sections

#### Data Configuration
- **symbols**: Trading pairs to analyze
- **timeframe**: Data granularity (1m, 5m, 15m, 1h, 4h, 1d)
- **date_range**: Historical period for training
- **lookback_window**: Number of periods to look back
- **features**: Technical indicators to generate

#### Model Configuration
- **enabled_models**: Which models to train
- **hyperparameters**: Model-specific settings
- **ensemble_settings**: Ensemble configuration
- **training_params**: Training hyperparameters

#### Trading Configuration
- **initial_balance**: Starting capital for backtesting
- **strategy**: Trading approach (long_only, short_only, long_short)
- **risk_management**: Position sizing and stop-loss settings
- **transaction_costs**: Fees and slippage assumptions

## MLOps and Tracking

### Experiment Tracking
Hyperion automatically tracks:
- **Model Parameters**: All hyperparameters and settings
- **Training Metrics**: Performance during training
- **Validation Results**: Out-of-sample performance
- **Model Artifacts**: Trained model files
- **Data Lineage**: What data was used
- **Code Version**: Exact code that produced results

### MLOps Features

#### Automatic Logging
```python
# Every model training automatically logs:
- Hyperparameters
- Training metrics
- Validation metrics
- Model artifacts
- Training time
- System info
```

#### Experiment Management
```
Configuration Menu → MLOps Status
```
- View all experiments
- Compare experiment results
- Restore previous configurations
- Export experiment data

#### Model Versioning
- Automatic model versioning
- Model lineage tracking
- Easy model rollback
- Performance history

### Integration with MLflow
```bash
# Start MLflow server (optional)
mlflow server --host 0.0.0.0 --port 5000

# View experiments in browser
open http://localhost:5000
```

## Advanced Features

### Multi-Timeframe Analysis
Train models on different timeframes:
```json
{
  "data": {
    "timeframes": {
      "primary": "1h",
      "secondary": "4h",
      "tertiary": "1d"
    }
  }
}
```

### Custom Feature Engineering
Add your own technical indicators:
```python
# In hyperion3/data/feature_engineering.py
def custom_indicator(data):
    # Your custom indicator logic
    return custom_values
```

### Risk Management Integration
```json
{
  "trading": {
    "max_position_size": 0.25,
    "stop_loss": 0.02,
    "take_profit": 0.06,
    "max_daily_loss": 0.05
  }
}
```

### Real-time Data Integration
```python
# Enable live data feeds
{
  "data": {
    "live_data": true,
    "update_frequency": "1m"
  }
}
```

## Best Practices

### For Beginners
1. **Start Simple**: Begin with ensemble models (e1, e2, e3)
2. **Learn Gradually**: Master one category before moving to next
3. **Validate Results**: Always check if results make sense
4. **Paper Trade First**: Never use real money without extensive testing
5. **Read Documentation**: Understand what each metric means

### For Intermediate Users
1. **Systematic Approach**: Train models in categories
2. **Optimize Hyperparameters**: Use auto-optimization features
3. **Create Ensembles**: Combine your best models
4. **Monitor Overfitting**: Check validation vs training performance
5. **Document Experiments**: Keep track of what works

### For Advanced Users
1. **Multi-Asset Analysis**: Test across different cryptocurrencies
2. **Custom Features**: Develop domain-specific indicators
3. **Production Pipeline**: Build robust deployment systems
4. **Continuous Learning**: Implement online learning systems
5. **Risk Management**: Sophisticated position sizing and hedging

### General Best Practices
1. **Data Quality**: Ensure clean, reliable data
2. **Feature Engineering**: Create meaningful predictive features
3. **Cross-Validation**: Use proper validation techniques
4. **Risk Management**: Never risk more than you can afford to lose
5. **Continuous Monitoring**: Watch for model degradation
6. **Regular Retraining**: Update models with new data
7. **Documentation**: Keep detailed records of experiments
8. **Version Control**: Track code and configuration changes

## Troubleshooting

### Common Issues and Solutions

#### Training Errors
**Problem**: Model training fails with error
**Solutions**:
1. Check data availability and quality
2. Reduce model complexity (batch size, layers)
3. Try simpler models first (s1, e1)
4. Check system resources (RAM, disk space)

#### Poor Performance
**Problem**: All models show low R² scores
**Solutions**:
1. Check data period (ensure sufficient history)
2. Verify symbol format (BTC/USDT not BTCUSDT)
3. Try different timeframes
4. Increase feature engineering
5. Check for data leakage

#### Memory Issues
**Problem**: Out of memory during training
**Solutions**:
1. Reduce batch size in config
2. Use smaller lookback windows
3. Reduce number of features
4. Close other applications
5. Use cloud computing for large models

#### Slow Training
**Problem**: Training takes too long
**Solutions**:
1. Start with faster models (LightGBM, sklearn)
2. Reduce number of optimization trials
3. Use GPU if available
4. Reduce data size for initial experiments

#### Inconsistent Results
**Problem**: Results vary between runs
**Solutions**:
1. Set random seeds in configuration
2. Use larger datasets
3. Increase cross-validation folds
4. Check for data randomization issues

### Getting Help
1. **Check Logs**: Look at console output for error details
2. **Read Documentation**: Most issues are covered in docs
3. **Search Issues**: Check GitHub issues for similar problems
4. **Ask Community**: Use discussions for strategy questions
5. **Report Bugs**: Create issues for reproducible bugs

### System Requirements Troubleshooting
- **Python Version**: Ensure Python 3.8+
- **Dependencies**: Update all packages to latest versions
- **Platform**: Some features work better on Linux/macOS
- **Hardware**: Minimum 8GB RAM, 16GB+ recommended

## Conclusion

Hyperion3 is a powerful framework for algorithmic trading research and development. Start with simple models, gradually explore advanced features, and always prioritize risk management. The key to success is systematic experimentation, thorough validation, and continuous learning.

Remember: **Past performance does not guarantee future results.** Always test thoroughly in paper trading before risking real capital.

For additional help:
- 📖 [Architecture Guide](ARCHITECTURE.md)
- 🔧 [Installation Guide](INSTALLATION.md)
- 📊 [API Reference](API_REFERENCE.md)
- 🏃 [Quick Start](QUICK_START.md)

---

**Happy Trading!** 🚀
