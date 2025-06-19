"""
Hyperion V2 - Advanced Configuration System
Modular configuration for multiple models, strategies, and cryptocurrencies
"""

from dataclasses import field
from pydantic.dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
from enum import Enum
import os
from datetime import datetime, timedelta


class ModelType(str, Enum):
    """Supported model types"""

    PATCHTST = "patchtst"
    SAC = "sac"
    TD3 = "td3"
    RAINBOW_DQN = "rainbow_dqn"
    TFT = "tft"
    DIFFUSION = "diffusion"
    ENSEMBLE = "ensemble"


class TradingStrategy(str, Enum):
    """Supported trading strategies"""

    LONG_SHORT = "long_short"
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    MARKET_NEUTRAL = "market_neutral"
    ADAPTIVE = "adaptive"  # Agregada para compatibilidad temporal


class DataAugmentation(Enum):
    """Data augmentation techniques"""

    TIME_WARP = "time_warp"
    MAGNITUDE_SCALE = "magnitude_scale"
    WAVELET = "wavelet"
    MIXUP = "mixup"
    ALL = "all"


@dataclass
class APIConfig:
    """Exchange API configuration"""

    exchange_id: str = "binance"
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_SECRET", ""))
    is_testnet: bool = True
    rate_limit: bool = True
    max_retries: int = 3
    timeout: int = 30


@dataclass
class DataConfig:
    """Data configuration"""

    data_dir: str = "data"
    symbols: List[str] = field(
        default_factory=lambda: ["SOL/USDT", "BTC/USDT", "ETH/USDT"]
    )
    lookback_window: int = 312  # Ajustado a la longitud real de los datos
    prediction_horizon: int = 24  # 1 day ahead
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    target_column: str = "close"
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "log_return",
            "volatility",
            "rsi",
            "sma_20",
            "std_20",
            "upper_band",
            "lower_band",
            "macd",
            "macd_signal",
            "volume_norm",
            "momentum",
            "atr",
            "obv",
            "hurst",
        ]
    )
    augmentation_techniques: List[str] = field(
        default_factory=lambda: ["noise", "scaling", "time_warping"]
    )

    # Advanced features
    use_sentiment: bool = True
    use_onchain: bool = True
    use_orderbook: bool = True
    use_social_metrics: bool = True

    # Download settings
    start_date: str = field(
        default_factory=lambda: (datetime.now() - timedelta(days=2190)).strftime(
            "%Y-%m-%d"
        )
    )  # 6 años atrás
    end_date: Optional[str] = None  # None means current date
    update_frequency: str = "1h"  # How often to update data


@dataclass
class TransformerConfig:
    """Transformer model configurations"""

    # PatchTST settings
    enabled: bool = True  # Enable PatchTST
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_layers: int = 3  # Alias para e_layers para compatibilidad

    # TFT settings
    tft_enabled: bool = True  # Enable TFT
    hidden_size: int = 160
    lstm_layers: int = 2
    attention_heads: int = 4
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Multimodal settings
    fusion_method: str = "attention"  # attention, concat, gating
    modality_dropout: float = 0.2


@dataclass
class RLConfig:
    """Reinforcement Learning configurations"""

    # Common settings
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005

    # SAC specific
    sac_alpha: float = 0.2  # Entropy regularization
    sac_automatic_entropy_tuning: bool = True

    # TD3 specific
    td3_policy_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_policy_freq: int = 2

    # Rainbow DQN specific
    rainbow_n_steps: int = 3
    rainbow_num_atoms: int = 51
    rainbow_v_min: float = -10.0
    rainbow_v_max: float = 10.0

    # Training settings
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10

    # Advanced features
    use_her: bool = True  # Hindsight Experience Replay
    use_per: bool = True  # Prioritized Experience Replay


@dataclass
class OptimizationConfig:
    """Optimization configuration"""

    metric: str = "sharpe_ratio"  # rmse, mae, sharpe_ratio
    time_budget: int = 21600  # 6 hours
    n_splits: int = 5
    flaml_time_budget: int = 21600  # 6 hours
    flaml_metric: str = "sharpe_ratio"
    flaml_estimator_list: List[str] = field(
        default_factory=lambda: ["lgbm", "xgboost", "catboost", "tft", "patchtst"]
    )
    flaml_num_samples: int = 100
    flaml_max_concurrent_trials: int = 4
    flaml_scheduler: str = "bayesian"  # cfo, bayesian, blendsearch


@dataclass
class TradingConfig:
    """Trading configuration"""

    initial_balance: float = 10000.0  # Balance inicial en USD
    strategy: TradingStrategy = TradingStrategy.LONG_SHORT
    position_sizing: str = "kelly"  # fixed, kelly, volatility
    risk_factor: float = 0.2
    max_position_size: float = 1.0
    stop_loss: float = 0.02
    take_profit: float = 0.04
    commission: float = 0.001
    slippage: float = 0.0005


@dataclass
class MLOpsConfig:
    """MLOps configuration"""

    # Experiment tracking
    use_mlflow: bool = True
    use_wandb: bool = False
    tracking_uri: str = "mlflow"
    experiment_name: str = "hyperion3"
    log_dir: str = "logs"
    save_plots: bool = True
    save_predictions: bool = True

    # Infrastructure
    use_gpu: bool = True
    use_mlx: bool = True  # Apple Silicon optimization
    distributed_training: bool = False
    num_workers: int = 4

    # Monitoring
    monitor_metrics: List[str] = field(
        default_factory=lambda: ["returns", "sharpe", "drawdown", "win_rate"]
    )
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "drawdown": -0.1,
            "daily_loss": -0.05,
            "correlation_breakdown": 0.3,
        }
    )


@dataclass
class HyperionV2Config:
    """Main configuration class"""

    # Sub-configurations
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    mlops: MLOpsConfig = field(default_factory=MLOpsConfig)

    # Model selection
    model_type: ModelType = ModelType.ENSEMBLE
    ensemble_models: List[ModelType] = field(
        default_factory=lambda: [
            ModelType.PATCHTST,
            ModelType.TFT,
            ModelType.SAC,
            ModelType.TD3,
            ModelType.RAINBOW_DQN,
        ]
    )

    hyperparam_opt: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "time_budget_per_model": 3600,
            "metric_to_optimize": "sharpe_ratio",
            "framework": "flaml",
            "flaml_estimator_list": ["lgbm", "xgboost", "catboost"],
        }
    )

    # Strategy selection
    trading_strategy: TradingStrategy = TradingStrategy.ADAPTIVE

    # System settings
    project_name: str = "HyperionV2"
    debug_mode: bool = False
    random_seed: int = 42

    # Paths
    data_dir: str = "./data"
    model_dir: str = "./models"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "api": self.api.__dict__,
            "data": self.data.__dict__,
            "transformer": self.transformer.__dict__,
            "rl": self.rl.__dict__,
            "optimization": self.optimization.__dict__,
            "trading": self.trading.__dict__,
            "mlops": self.mlops.__dict__,
            "model_type": self.model_type.value,
            "trading_strategy": self.trading_strategy.value,
            "project_name": self.project_name,
            "hyperparam_opt": self.hyperparam_opt,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key, maintaining compatibility with dict.get()"""
        try:
            # First try to get from sub-configs
            for subconfig in [
                self.api,
                self.data,
                self.transformer,
                self.rl,
                self.optimization,
                self.trading,
                self.mlops,
            ]:
                if hasattr(subconfig, key):
                    return getattr(subconfig, key)

            # Then try direct attributes
            if hasattr(self, key):
                value = getattr(self, key)
                if isinstance(value, (ModelType, TradingStrategy)):
                    return value.value
                return value

            return default
        except Exception:
            return default

    @classmethod
    def from_dict(cls, data: dict) -> "HyperionV2Config":
        """Create ``HyperionV2Config`` from a dictionary."""
        cfg = cls()

        if "api" in data:
            cfg.api = APIConfig(**data["api"])
        if "data" in data:
            cfg.data = DataConfig(**data["data"])
        if "transformer" in data:
            cfg.transformer = TransformerConfig(**data["transformer"])
        if "rl" in data:
            cfg.rl = RLConfig(**data["rl"])
        if "optimization" in data:
            cfg.optimization = OptimizationConfig(**data["optimization"])
        if "trading" in data:
            cfg.trading = TradingConfig(**data["trading"])
        if "mlops" in data:
            cfg.mlops = MLOpsConfig(**data["mlops"])

        cfg.model_type = ModelType(data.get("model_type", cfg.model_type.value))
        cfg.trading_strategy = TradingStrategy(
            data.get("trading_strategy", cfg.trading_strategy.value)
        )

        cfg.ensemble_models = [
            ModelType(m)
            for m in data.get(
                "ensemble_models",
                [m.value for m in cfg.ensemble_models],
            )
        ]

        cfg.project_name = data.get("project_name", cfg.project_name)

        if "models" in data:
            cfg.models = data["models"]
            ensemble_cfg = data["models"].get("ensemble", {})
            if "models_to_train" in ensemble_cfg:
                cfg.ensemble_models = [ModelType(m) for m in ensemble_cfg["models_to_train"]]

        if "hyperparam_opt" in data:
            cfg.hyperparam_opt = data["hyperparam_opt"]

        # Attach any additional unknown fields
        for key, value in data.items():
            if not hasattr(cfg, key):
                setattr(cfg, key, value)

        return cfg

    @classmethod
    def from_strategy(cls, strategy: TradingStrategy) -> "HyperionV2Config":
        """Create config optimized for specific strategy"""
        config = cls()

        if strategy == TradingStrategy.SCALPING:
            config.data.timeframes = {"primary": "1m"}
            config.data.lookback_window = 100
            config.trading.stop_loss = 0.01
            config.trading.take_profit = 0.02
            config.rl.batch_size = 512
            config.model_type = ModelType.SAC

        elif strategy == TradingStrategy.SWING:
            config.data.timeframes = {"primary": "15m"}
            config.data.lookback_window = 336
            config.trading.stop_loss = 0.03
            config.trading.take_profit = 0.08
            config.rl.batch_size = 256
            config.model_type = ModelType.ENSEMBLE

        elif strategy == TradingStrategy.POSITION:
            config.data.timeframes = {"primary": "4h"}
            config.data.lookback_window = 720
            config.trading.stop_loss = 0.05
            config.trading.take_profit = 0.15
            config.rl.batch_size = 128
            config.model_type = ModelType.PATCHTST

        config.trading_strategy = strategy
        return config


# Preset configurations for quick start
PRESET_CONFIGS = {
    "conservative": HyperionV2Config(
        model_type=ModelType.TD3,
        trading=TradingConfig(max_position_size=0.3, stop_loss=0.015, take_profit=0.03),
    ),
    "aggressive": HyperionV2Config(
        model_type=ModelType.SAC,
        trading=TradingConfig(
            max_position_size=0.7, leverage=3, stop_loss=0.03, take_profit=0.10
        ),
    ),
    "balanced": HyperionV2Config(
        model_type=ModelType.ENSEMBLE,
        trading_strategy=TradingStrategy.ADAPTIVE,
        ensemble_models=[
            ModelType.PATCHTST,
            ModelType.TFT,
            ModelType.SAC,
            ModelType.TD3,
            ModelType.RAINBOW_DQN,
        ],
        hyperparam_opt={"enabled": True},
    ),
}
