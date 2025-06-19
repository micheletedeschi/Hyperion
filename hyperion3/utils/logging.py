"""
Módulo de logging para Hyperion V2
"""

import logging
import os
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Configura un logger con formato personalizado y salida a archivo

    Args:
        name: Nombre del logger
        log_file: Nombre del archivo de log (opcional)
        level: Nivel de logging (default: INFO)
        format_str: Formato personalizado (opcional)

    Returns:
        Logger configurado
    """
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar handlers duplicados
    if logger.handlers:
        return logger

    # Formato por defecto
    if format_str is None:
        format_str = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    formatter = logging.Formatter(format_str)

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo si se especifica
    if log_file:
        # Crear directorio de logs si no existe
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Añadir timestamp al nombre del archivo si no tiene extensión
        if not os.path.splitext(log_file)[1]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{log_file}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
