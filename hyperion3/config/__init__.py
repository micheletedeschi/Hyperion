"""Convenience imports for configuration package."""

import json
from .base_config import (
    HyperionV2Config,
    ModelType,
    TradingStrategy,
    PRESET_CONFIGS,
    DataAugmentation,
)

from .model_configs import PatchTSTConfig, TFTConfig, RLAgentConfig
from .strategies import ScalpingStrategy, SwingStrategy, PositionStrategy
from .yaml_utils import load_config, save_config


def get_config(config_or_dict=None):
    """
    Obtiene la configuración del sistema.

    Args:
        config_or_dict: Puede ser:
            - None: Retorna configuración por defecto
            - Un objeto con método to_dict()
            - Un diccionario con la configuración
            - Una ruta a un archivo JSON/YAML con la configuración

    Returns:
        HyperionV2Config: La configuración del sistema
    """
    if config_or_dict is None:
        return HyperionV2Config()  # Retorna configuración por defecto

    if hasattr(config_or_dict, "to_dict"):
        config_dict = config_or_dict.to_dict()
    elif isinstance(config_or_dict, dict):
        config_dict = config_or_dict
    else:
        # Se espera un str, bytes o os.PathLike (por ejemplo, un archivo de configuración)
        with open(config_or_dict, "r") as f:
            config_dict = json.load(f)

    cfg = HyperionV2Config.from_dict(config_dict)

    # Attach extra keys so they are accessible as attributes
    for key, value in config_dict.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg


__all__ = [
    "HyperionV2Config",
    "ModelType",
    "TradingStrategy",
    "PRESET_CONFIGS",
    "DataAugmentation",
    "PatchTSTConfig",
    "TFTConfig",
    "RLAgentConfig",
    "ScalpingStrategy",
    "SwingStrategy",
    "PositionStrategy",
    "load_config",
    "save_config",
    "get_config",
]
