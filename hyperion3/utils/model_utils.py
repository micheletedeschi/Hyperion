"""Funciones para guardar y cargar modelos de PyTorch de forma segura."""

import os
import torch
import logging
from torch.serialization import add_safe_globals
from typing import Any, Optional

# Añadir numpy._core.multiarray._reconstruct a los globals seguros
add_safe_globals(["numpy._core.multiarray._reconstruct"])

logger = logging.getLogger(__name__)


def load_model(model_path: str, model: Any, weights_only: bool = False) -> bool:
    """Cargar modelo desde archivo

    Args:
        model_path: Ruta al archivo del modelo
        model: Instancia del modelo a cargar
        weights_only: Si True, solo carga los pesos del modelo

    Returns:
        bool: True si se cargó exitosamente
    """
    try:
        if os.path.exists(model_path):
            logger.info(f"Encontrado modelo guardado en: {model_path}")
            if hasattr(model, "load"):
                model.load(model_path)
                logger.info(
                    f"Modelo cargado exitosamente con weights_only={weights_only}"
                )
                return True
            else:
                # Cargar directamente con torch.load si el modelo no tiene método load
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=weights_only
                )
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(
                    f"Modelo cargado exitosamente con weights_only={weights_only}"
                )
                return True
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return False
    return False


def save_model(model: Any, model_path: str, weights_only: bool = False) -> bool:
    """Guardar modelo a archivo

    Args:
        model: Instancia del modelo a guardar
        model_path: Ruta donde guardar el modelo
        weights_only: Si True, solo guarda los pesos del modelo

    Returns:
        bool: True si se guardó exitosamente
    """
    try:
        if hasattr(model, "save"):
            model.save(model_path)
        else:
            # Guardar directamente con torch.save si el modelo no tiene método save
            torch.save(
                model.state_dict(), model_path, _use_new_zipfile_serialization=True
            )
        logger.info(f"Modelo guardado exitosamente en {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error guardando modelo: {str(e)}")
        return False
