"""Utilidades relacionadas con MLOps para el proyecto Hyperion."""

import logging

logger = logging.getLogger(__name__)


def setup_mlops(config, experiment_name: str = "hyperion_experiment") -> None:
    """Inicializa herramientas de seguimiento de experimentos.

    De manera condicional y según la configuración, se intentará activar mlflow
    y/o wandb. Las importaciones se realizan de forma perezosa para no requerir
    estas dependencias en tiempo de ejecución si no se van a utilizar.

    Args:
        config: Configuración del proyecto con opciones ``mlops``.
        experiment_name: Nombre del experimento a registrar.
    """

    use_mlflow = config.mlops.use_mlflow
    use_wandb = config.mlops.use_wandb

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow experimento configurado: %s", experiment_name)
        except ImportError:
            logger.warning("MLflow no está instalado. Se omitirá su inicialización.")
    if use_wandb:
        try:
            import wandb
            wandb.init(project=experiment_name, config=config.to_dict() if hasattr(config, "to_dict") else dict(config))
            logger.info("Wandb inicializado con experimento: %s", experiment_name)
        except ImportError:
            logger.warning("Wandb no está instalado. Se omitirá su inicialización.")


def validate_metrics(metrics: dict, thresholds: dict) -> bool:
    """Validate performance metrics against minimum thresholds."""
    for name, limit in thresholds.items():
        value = metrics.get(name)
        if value is None:
            logger.warning("Métrica %s no encontrada", name)
            return False
        if value < limit:
            logger.warning("Métrica %s=%.4f por debajo del umbral %.4f", name, value, limit)
            return False
    return True


__all__ = ["setup_mlops", "validate_metrics"]

