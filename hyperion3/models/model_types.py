"""Model types enum for Hyperion."""

from enum import Enum
from typing import Dict, Any


class ModelType(Enum):
    """Supported model types in Hyperion."""
    
    # Traditional ML
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    
    # Linear models
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    
    # Neural networks
    MLP = "mlp"
    LSTM = "lstm"
    
    # Transformers
    TFT = "tft"
    PATCHTST = "patchtst"
    
    # Reinforcement Learning
    SAC = "sac"
    TD3 = "td3"
    RAINBOW_DQN = "rainbow_dqn"
    
    # Ensembles
    VOTING = "voting"
    STACKING = "stacking"
    
    @classmethod
    def get_model_config(cls, model_type: 'ModelType') -> Dict[str, Any]:
        """Get default configuration for model type."""
        configs = {
            cls.TFT: {
                "hidden_size": 256,
                "attention_heads": 8,
                "lstm_layers": 2,
                "dropout": 0.2,
                "prediction_length": 24,
                "quantiles": [0.1, 0.5, 0.9]
            },
            cls.PATCHTST: {
                "patch_len": 16,
                "stride": 8,
                "d_model": 512,
                "n_heads": 8,
                "e_layers": 3,
                "dropout": 0.1
            },
            cls.SAC: {
                "hidden_sizes": [256, 256],
                "learning_rate": 3e-4,
                "buffer_size": 1000000,
                "batch_size": 256
            }
        }
        return configs.get(model_type, {})
