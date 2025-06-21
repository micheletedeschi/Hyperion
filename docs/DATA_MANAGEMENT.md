# 📊 Data Management Guide - Hyperion3

Complete guide to data acquisition, processing, and feature engineering in Hyperion3.

## Table of Contents
1. [Data Sources](#data-sources)
2. [Data Download and Storage](#data-download-and-storage)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Data Validation](#data-validation)
6. [Custom Data Sources](#custom-data-sources)
7. [Data Configuration](#data-configuration)
8. [Troubleshooting](#troubleshooting)

## Data Sources

### Supported Exchanges
Hyperion3 supports multiple cryptocurrency exchanges through the CCXT library:

#### Primary Exchanges
- **Binance**: Most liquid, comprehensive data
- **Coinbase Pro**: High-quality US market data
- **Kraken**: European market leader
- **Bitfinex**: Professional trading features

#### Secondary Exchanges
- **Huobi**: Asian markets
- **OKX**: Derivatives and spot
- **KuCoin**: Altcoin variety
- **Gate.io**: Emerging tokens

### Data Types Available

#### OHLCV Data (Core)
- **Open**: Opening price for the period
- **High**: Highest price during the period
- **Low**: Lowest price during the period
- **Close**: Closing price for the period
- **Volume**: Trading volume during the period

#### Additional Data (Advanced)
- **Order Book**: Bid/ask depth data
- **Trades**: Individual transaction data
- **Funding Rates**: For perpetual futures
- **Open Interest**: Derivatives data

## Data Download and Storage

### Automatic Data Download
Hyperion3 automatically downloads required data when you start training:

```python
# Configuration in config.json
{
  "data": {
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframe": "1h",
    "start_date": "2023-01-01",
    "end_date": "2024-06-01"
  }
}
```

### Manual Data Download
```bash
# Download specific data
python -c "
from hyperion3.data.downloader import DataDownloader
downloader = DataDownloader()
downloader.download_symbol('BTC/USDT', '1h', '2023-01-01', '2024-01-01')
"
```

### Data Storage Structure
```
data/
├── BTC_USDT_1h_20250521.csv      # Raw OHLCV data
├── BTC_USDT_4h_20250521.csv      # Different timeframes
├── BTC_USDT_1d_20250521.csv      # Daily data
└── processed/                     # Processed data with features
    ├── BTC_USDT_1h_features.csv
    └── BTC_USDT_1h_targets.csv
```

### Supported Timeframes
- **1m, 5m, 15m, 30m**: High frequency (scalping)
- **1h, 2h, 4h, 6h, 12h**: Medium frequency (swing trading)
- **1d, 3d, 1w**: Low frequency (position trading)

## Data Preprocessing

### Automatic Preprocessing Pipeline
```python
# Executed automatically during training
1. Data Quality Checks
   ├── Missing value detection
   ├── Outlier identification
   ├── Data consistency validation
   └── Timestamp verification

2. Data Cleaning
   ├── Fill missing values (forward fill, interpolation)
   ├── Remove outliers (IQR, Z-score methods)
   ├── Handle gaps in data
   └── Normalize timestamps

3. Data Transformation
   ├── Log returns calculation
   ├── Price normalization
   ├── Volume normalization
   └── Volatility adjustment
```

### Data Quality Metrics
Hyperion3 automatically calculates:
- **Completeness**: Percentage of non-missing values
- **Consistency**: Data format and range validation
- **Timeliness**: Timestamp gaps and irregularities
- **Accuracy**: Cross-validation with multiple sources

### Handling Missing Data
```python
# Methods available in preprocessor
missing_data_strategies = {
    'forward_fill': 'Use last known value',
    'backward_fill': 'Use next known value',
    'linear_interpolation': 'Linear interpolation between points',
    'spline_interpolation': 'Smooth curve interpolation',
    'drop': 'Remove rows with missing data'
}
```

## Feature Engineering

### Technical Indicators (100+ Available)

#### Momentum Indicators
```python
# Automatically generated features
momentum_indicators = {
    'RSI': 'Relative Strength Index (14, 21, 50 periods)',
    'MACD': 'Moving Average Convergence Divergence',
    'Stochastic': 'Stochastic Oscillator (%K, %D)',
    'Williams_R': 'Williams %R',
    'ROC': 'Rate of Change',
    'CMO': 'Chande Momentum Oscillator',
    'TSI': 'True Strength Index',
    'UO': 'Ultimate Oscillator'
}
```

#### Trend Indicators
```python
trend_indicators = {
    'SMA': 'Simple Moving Average (5, 10, 20, 50, 100, 200)',
    'EMA': 'Exponential Moving Average (12, 26, 50, 100)',
    'WMA': 'Weighted Moving Average',
    'TEMA': 'Triple Exponential Moving Average',
    'DEMA': 'Double Exponential Moving Average',
    'KAMA': 'Kaufman Adaptive Moving Average',
    'MAMA': 'MESA Adaptive Moving Average',
    'T3': 'Tillson T3 Moving Average',
    'Bollinger_Bands': 'Upper, Middle, Lower bands',
    'Donchian_Channel': 'Highest/Lowest over N periods',
    'Keltner_Channel': 'ATR-based channel',
    'PSAR': 'Parabolic SAR',
    'ADX': 'Average Directional Index',
    'Aroon': 'Aroon Up/Down',
    'Ichimoku': 'Complete Ichimoku system',
    'VORTEX': 'Vortex Indicator'
}
```

#### Volatility Indicators
```python
volatility_indicators = {
    'ATR': 'Average True Range (14, 21 periods)',
    'NATR': 'Normalized ATR',
    'TRANGE': 'True Range',
    'Bollinger_Width': 'Band width percentage',
    'Chaikin_Volatility': 'Price volatility based on H-L',
    'Standard_Deviation': 'Rolling standard deviation',
    'Variance': 'Rolling variance'
}
```

#### Volume Indicators
```python
volume_indicators = {
    'OBV': 'On Balance Volume',
    'AD': 'Accumulation/Distribution Line',
    'ADOSC': 'Chaikin A/D Oscillator',
    'MFI': 'Money Flow Index',
    'CMF': 'Chaikin Money Flow',
    'FI': 'Force Index',
    'EMV': 'Ease of Movement',
    'VPT': 'Volume Price Trend',
    'NVI': 'Negative Volume Index',
    'PVI': 'Positive Volume Index',
    'VWAP': 'Volume Weighted Average Price'
}
```

#### Pattern Recognition
```python
candlestick_patterns = {
    'Doji': 'Indecision patterns',
    'Engulfing': 'Bullish/Bearish engulfing',
    'Hammer': 'Reversal hammer patterns',
    'Shooting_Star': 'Bearish reversal',
    'Morning_Star': 'Bullish reversal',
    'Evening_Star': 'Bearish reversal',
    'Three_White_Soldiers': 'Strong bullish pattern',
    'Three_Black_Crows': 'Strong bearish pattern',
    'Piercing_Pattern': 'Bullish reversal',
    'Dark_Cloud_Cover': 'Bearish reversal'
}
```

### Custom Feature Engineering

#### Adding Custom Indicators
```python
# In hyperion3/data/feature_engineering.py
class CustomFeatureEngineer(FeatureEngineer):
    def add_custom_features(self, data):
        """Add your custom features here"""
        
        # Example: Custom momentum indicator
        data['custom_momentum'] = (
            data['close'].pct_change(5) + 
            data['close'].pct_change(10)
        ) / 2
        
        # Example: Volume-price relationship
        data['volume_price_ratio'] = (
            data['volume'] / data['close'].rolling(20).mean()
        )
        
        # Example: Volatility-adjusted returns
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        data['vol_adjusted_returns'] = returns / volatility
        
        return data
```

#### Feature Selection
```python
# Automatic feature selection methods
feature_selection_methods = {
    'correlation_filter': 'Remove highly correlated features',
    'variance_filter': 'Remove low-variance features',
    'mutual_information': 'Select based on mutual information',
    'recursive_feature_elimination': 'Recursive feature elimination',
    'lasso_selection': 'L1 regularization feature selection',
    'tree_importance': 'Tree-based feature importance'
}
```

### Advanced Feature Engineering

#### Multi-Timeframe Features
```python
# Combine features from different timeframes
multi_timeframe_config = {
    "timeframes": {
        "primary": "1h",      # Main trading timeframe
        "context_1": "4h",    # Higher timeframe context
        "context_2": "1d"     # Long-term context
    }
}
```

#### Regime Detection Features
```python
# Market regime classification features
regime_features = {
    'volatility_regime': 'High/Low volatility periods',
    'trend_regime': 'Trending/Ranging markets',
    'momentum_regime': 'Strong/Weak momentum periods',
    'volume_regime': 'High/Low volume periods'
}
```

#### Cross-Asset Features
```python
# Features from related assets
cross_asset_features = {
    'btc_dominance': 'Bitcoin market dominance',
    'correlation_matrix': 'Inter-asset correlations',
    'relative_strength': 'Performance vs benchmark',
    'sector_rotation': 'Crypto sector performance'
}
```

## Data Validation

### Automatic Validation Checks
```python
validation_checks = {
    'data_quality': {
        'missing_values': 'Check for gaps in data',
        'outliers': 'Detect extreme values',
        'consistency': 'Verify OHLC relationships',
        'timestamp_gaps': 'Find missing time periods'
    },
    'feature_quality': {
        'infinite_values': 'Check for inf/-inf values',
        'nan_propagation': 'Ensure no NaN in features',
        'feature_distribution': 'Verify reasonable ranges',
        'correlation_check': 'Identify redundant features'
    },
    'target_leakage': {
        'future_data': 'Prevent look-ahead bias',
        'target_correlation': 'Check for data leakage',
        'temporal_consistency': 'Ensure proper time ordering'
    }
}
```

### Data Quality Reports
```python
# Generated automatically
quality_report = {
    'completeness_score': 0.98,  # 98% complete data
    'consistency_score': 0.95,   # 95% consistent
    'outlier_percentage': 0.02,  # 2% outliers detected
    'feature_count': 127,        # Number of features generated
    'warning_flags': []          # Any issues found
}
```

## Custom Data Sources

### Adding New Exchanges
```python
# In hyperion3/data/custom_downloaders.py
class CustomExchangeDownloader(BaseDownloader):
    def __init__(self, exchange_id):
        super().__init__(exchange_id)
        
    def download_data(self, symbol, timeframe, start_date, end_date):
        """Implement custom download logic"""
        # Your custom implementation
        pass
```

### Alternative Data Sources
```python
# Examples of additional data sources
alternative_data = {
    'social_sentiment': {
        'twitter_sentiment': 'Twitter sentiment analysis',
        'reddit_mentions': 'Reddit discussion volume',
        'news_sentiment': 'Financial news sentiment'
    },
    'on_chain_data': {
        'transaction_volume': 'Blockchain transaction data',
        'active_addresses': 'Network activity metrics',
        'whale_movements': 'Large holder activity'
    },
    'macro_data': {
        'interest_rates': 'Central bank rates',
        'inflation_data': 'Economic indicators',
        'currency_strength': 'Dollar index, etc.'
    }
}
```

### Data Fusion
```python
# Combine multiple data sources
data_fusion_config = {
    "primary_source": "binance",
    "secondary_sources": ["coinbase", "kraken"],
    "fusion_method": "weighted_average",
    "quality_weights": {
        "binance": 0.5,
        "coinbase": 0.3,
        "kraken": 0.2
    }
}
```

## Data Configuration

### Complete Data Configuration Example
```json
{
  "data": {
    "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
    "start_date": "2023-01-01",
    "end_date": "2024-06-01",
    "timeframe": "1h",
    "exchange": "binance",
    
    "preprocessing": {
      "missing_data_strategy": "forward_fill",
      "outlier_method": "iqr",
      "outlier_threshold": 3.0,
      "normalization": "minmax"
    },
    
    "features": {
      "technical_indicators": true,
      "candlestick_patterns": true,
      "volume_features": true,
      "custom_features": true,
      "feature_selection": true,
      "max_features": 100
    },
    
    "augmentation": {
      "enabled": true,
      "techniques": ["noise", "scaling", "jittering"],
      "augmentation_factor": 2
    },
    
    "validation": {
      "quality_checks": true,
      "leakage_detection": true,
      "distribution_analysis": true
    }
  }
}
```

### Environment-Specific Configuration
```bash
# Development environment
export HYPERION_DATA_PATH="/dev/data"
export HYPERION_CACHE_SIZE="1GB"

# Production environment
export HYPERION_DATA_PATH="/prod/data"
export HYPERION_CACHE_SIZE="10GB"
export HYPERION_DATA_BACKUP="enabled"
```

## Troubleshooting

### Common Data Issues

#### Missing Data
```python
# Diagnosis
missing_data_report = {
    'total_missing': '5%',
    'missing_periods': ['2023-03-15 14:00', '2023-04-02 08:00'],
    'cause': 'Exchange maintenance',
    'solution': 'Use forward fill or interpolation'
}
```

#### Data Quality Issues
```python
# Common problems and solutions
data_issues = {
    'extreme_outliers': {
        'problem': 'Prices showing impossible values',
        'solution': 'Apply outlier filtering (IQR method)',
        'prevention': 'Add real-time data validation'
    },
    'timestamp_gaps': {
        'problem': 'Missing time periods in data',
        'solution': 'Use interpolation or multiple exchanges',
        'prevention': 'Implement redundant data sources'
    },
    'inconsistent_ohlc': {
        'problem': 'Open/High/Low/Close relationships invalid',
        'solution': 'Use data validation and cleaning',
        'prevention': 'Real-time data quality monitoring'
    }
}
```

#### Performance Issues
```python
# Data processing performance optimization
optimization_techniques = {
    'chunked_processing': 'Process data in smaller chunks',
    'parallel_download': 'Download multiple symbols simultaneously',
    'caching': 'Cache processed features',
    'vectorization': 'Use numpy/pandas vectorized operations',
    'feature_reduction': 'Reduce number of features'
}
```

### Debugging Data Pipeline
```python
# Debug mode for data processing
debug_config = {
    "data": {
        "debug_mode": true,
        "verbose_logging": true,
        "save_intermediate": true,
        "validation_reports": true
    }
}
```

### Data Recovery
```python
# Automatic data recovery strategies
recovery_strategies = {
    'backup_exchanges': 'Use alternative data sources',
    'interpolation': 'Fill gaps with mathematical interpolation',
    'historical_patterns': 'Use historical data patterns',
    'manual_override': 'Manual data correction tools'
}
```

## Best Practices

### Data Management Best Practices
1. **Multiple Sources**: Always have backup data sources
2. **Quality Monitoring**: Implement continuous data quality checks
3. **Version Control**: Track data versions and changes
4. **Documentation**: Document all data transformations
5. **Testing**: Test data pipeline with known good data

### Performance Optimization
1. **Efficient Storage**: Use appropriate data formats (Parquet, HDF5)
2. **Lazy Loading**: Load data only when needed
3. **Parallel Processing**: Utilize multiple CPU cores
4. **Memory Management**: Monitor and optimize memory usage
5. **Caching**: Cache frequently used datasets

### Data Security
1. **API Key Security**: Secure storage of exchange credentials
2. **Data Encryption**: Encrypt sensitive data at rest
3. **Access Control**: Implement proper access controls
4. **Audit Trail**: Log all data access and modifications
5. **Backup Strategy**: Regular backups with encryption

---

This comprehensive data management guide ensures you can effectively handle all aspects of data in Hyperion3, from acquisition to feature engineering to validation. Proper data management is crucial for successful algorithmic trading strategies.
