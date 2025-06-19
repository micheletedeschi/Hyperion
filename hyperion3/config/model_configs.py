"""Model configurations."""

from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from pydantic import BaseModel as PydanticBaseModel, Field
import numpy as np

if TYPE_CHECKING:
    from hyperion3.models import ModelFactory
    from hyperion3.config.base_config import ModelType, HyperionV2Config


@dataclass
class PatchTSTConfig:
    seq_len: int = 312
    pred_len: int = 24
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    dropout: float = 0.1


@dataclass
class TFTConfig:
    hidden_size: int = 160
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1


@dataclass
class RLAgentConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 100000


@dataclass
class BaseModel:
    """Base model configuration"""

    type: str
    params: Dict[str, Any]


class LGBMConfig(PydanticBaseModel):
    """Configuración para el modelo LightGBM."""

    # Parámetros básicos
    n_estimators: int = Field(1000, ge=1)
    learning_rate: float = Field(0.01, ge=0.0, le=1.0)
    num_leaves: int = Field(31, ge=1)
    max_depth: int = Field(-1, ge=-1)
    min_child_samples: int = Field(20, ge=1)
    subsample: float = Field(1.0, ge=0.0, le=1.0)
    colsample_bytree: float = Field(1.0, ge=0.0, le=1.0)
    reg_alpha: float = Field(0.0, ge=0.0)
    reg_lambda: float = Field(0.0, ge=0.0)
    random_state: Optional[int] = Field(None)

    # Parámetros de entrenamiento
    early_stopping_rounds: int = Field(50, ge=1)
    verbose: int = Field(-1)

    # Parámetros de validación
    validation_split: float = Field(0.2, ge=0.0, lt=1.0)

    class Config:
        """Configuración de Pydantic"""

        arbitrary_types_allowed = True
        validate_assignment = True


class XGBoostConfig(PydanticBaseModel):
    """Configuración para el modelo XGBoost"""

    # Parámetros básicos
    n_estimators: int = Field(1000, ge=1)
    learning_rate: float = Field(0.01, ge=0.0, le=1.0)
    max_depth: int = Field(6, ge=1)
    min_child_weight: int = Field(1, ge=0)
    subsample: float = Field(1.0, ge=0.0, le=1.0)
    colsample_bytree: float = Field(1.0, ge=0.0, le=1.0)
    gamma: float = Field(0.0, ge=0.0)
    reg_alpha: float = Field(0.0, ge=0.0)
    reg_lambda: float = Field(1.0, ge=0.0)
    random_state: Optional[int] = Field(None)

    # Parámetros de entrenamiento
    early_stopping_rounds: int = Field(50, ge=1)
    verbose: int = Field(-1)

    # Parámetros de validación
    validation_split: float = Field(0.2, ge=0.0, lt=1.0)

    class Config:
        """Configuración de Pydantic"""

        arbitrary_types_allowed = True
        validate_assignment = True


class CatBoostConfig(PydanticBaseModel):
    """Configuración para el modelo CatBoost"""

    # Parámetros básicos
    iterations: int = Field(1000, ge=1)
    learning_rate: float = Field(0.01, ge=0.0, le=1.0)
    depth: int = Field(6, ge=1)
    l2_leaf_reg: float = Field(3.0, ge=0.0)
    bootstrap_type: str = Field("Bernoulli")
    subsample: float = Field(0.8, ge=0.0, le=1.0)
    random_strength: float = Field(1.0, ge=0.0)
    random_state: Optional[int] = Field(None)

    # Parámetros de entrenamiento
    early_stopping_rounds: int = Field(50, ge=1)
    verbose: bool = Field(False)

    # Parámetros de validación
    validation_split: float = Field(0.2, ge=0.0, lt=1.0)

    class Config:
        """Configuración de Pydantic"""

        arbitrary_types_allowed = True
        validate_assignment = True


# Diccionario de configuraciones por defecto
DEFAULT_CONFIGS = {
    "lgbm": LGBMConfig(),
    "xgboost": XGBoostConfig(),
    "catboost": CatBoostConfig(),
}


def get_model_config(model_type: str) -> BaseModel:
    """Get model configuration

    Args:
        model_type: Type of model to configure

    Returns:
        Model configuration object
    """
    # Import here to avoid circular import
    from hyperion3.models import ModelFactory
    from hyperion3.config.base_config import ModelType, HyperionV2Config

    # Create base config
    config = HyperionV2Config()

    # Create model factory
    factory = ModelFactory(config)

    # Get model type
    model_type_enum = ModelType(model_type.lower())

    # Create model instance
    model = factory.create_model(model_type_enum)

    # Return configuration
    return BaseModel(
        type=model_type_enum.value,
        params=model.get_config() if hasattr(model, "get_config") else {},
    )


__all__ = [
    "PatchTSTConfig",
    "TFTConfig",
    "RLAgentConfig",
    "LGBMConfig",
    "XGBoostConfig",
    "CatBoostConfig",
    "DEFAULT_CONFIGS",
    "get_model_config",
]
