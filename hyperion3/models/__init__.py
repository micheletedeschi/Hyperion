"""Hyperion model collection."""

from typing import Dict, Any, Optional, Union
from .model_factory import ModelFactory
from .model_types import ModelType
from .transformers import PatchTST, PatchTSTPredictor, PatchTSTTrainer
from .transformers.tft import TFTCryptoPredictor
from .rl_agents import SACTradingAgent, TD3TradingAgent, EnsembleAgent

__all__ = [
    # Factory
    "ModelFactory",
    "ModelType",
    # Transformers
    "PatchTST",
    "PatchTSTPredictor",
    "PatchTSTTrainer",
    "TFTCryptoPredictor",
    # RL Agents
    "SACTradingAgent",
    "TD3TradingAgent",
    # Ensembles
    "EnsembleAgent",
]
