"""Pequeñas utilidades para manejar loggers."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Obtiene un :class:`logging.Logger` con configuración básica.

    Si el logger no tiene handlers asociados se añadirá uno de consola con un
    formato estándar.

    Args:
        name: Nombre del logger a obtener.

    Returns:
        Instancia de :class:`logging.Logger` lista para usar.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


__all__ = ["get_logger"]
