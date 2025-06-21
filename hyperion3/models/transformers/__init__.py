"""Transformer-based models."""

# Importaciones condicionales para evitar errores
__all__ = []

try:
    from .patchtst import PatchTST, PatchTSTPredictor, PatchTSTTrainer
    __all__.extend(["PatchTST", "PatchTSTPredictor", "PatchTSTTrainer"])
except ImportError:
    pass

try:
    from .tft import TFTCryptoPredictor
    __all__.append("TFTCryptoPredictor")
except ImportError:
    pass
