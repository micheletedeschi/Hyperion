"""Hyperion model collection."""

from typing import Dict, Any, Optional, Union
from .base import BaseModel, BaseTradingModel
from .model_factory import ModelFactory
from .model_types import ModelType

# Importaciones condicionales para evitar errores
__all__ = [
    # Base classes
    "BaseModel",
    "BaseTradingModel",
    # Factory
    "ModelFactory",
    "ModelType",
]

# Transformers - importación condicional
try:
    from .transformers import PatchTST, PatchTSTPredictor, PatchTSTTrainer
    __all__.extend(["PatchTST", "PatchTSTPredictor", "PatchTSTTrainer"])
except ImportError:
    pass

try:
    from .transformers.tft import TFTCryptoPredictor
    __all__.append("TFTCryptoPredictor")
except ImportError:
    pass

# RL Agents - importación condicional
try:
    from .rl_agents import SACTradingAgent, TD3TradingAgent
    __all__.extend(["SACTradingAgent", "TD3TradingAgent"])
except ImportError:
    pass
