"""Utilidades para detección de dispositivo en entornos con PyTorch."""

import torch


def get_device() -> str:
    """Return the best available device (CUDA, MPS or CPU)."""
    if torch.cuda.is_available():
        return "cuda"

    """Obtiene el dispositivo preferido para ejecutar modelos."""

    if torch.backends.mps.is_available():
        return "mps"  # Metal Performance Shaders para Apple Silicon
    return "cpu"


__all__ = ["get_device"]
