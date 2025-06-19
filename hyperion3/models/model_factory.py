"""Model factory for creating different types of models."""

from typing import Dict, Any, Optional, Union, TYPE_CHECKING
import logging

from .transformers.patchtst import PatchTST
from .transformers.tft import TFTCryptoPredictor
from .rl_agents.sac import SACTradingAgent
from .rl_agents.td3 import TD3TradingAgent
from .rl_agents.rainbow_dqn import RainbowTradingAgent
from .rl_agents.ensemble_agent import EnsembleAgent
from .base import BaseModel
from .model_types import ModelType

if TYPE_CHECKING:
    from ..config.base_config import HyperionV2Config

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory para crear diferentes tipos de modelos."""

    def __init__(self, config: Optional[Any] = None):
        """Inicializa el factory.

        Args:
            config: Configuración opcional para los modelos
        """
        self.config = config
        self.model_classes = {
            ModelType.PATCHTST.value: PatchTST,
            ModelType.TFT.value: TFTCryptoPredictor,
            ModelType.SAC.value: SACTradingAgent,
            ModelType.TD3.value: TD3TradingAgent,
            ModelType.RAINBOW_DQN.value: RainbowTradingAgent,
            ModelType.ENSEMBLE.value: EnsembleAgent,
        }

        # Registrar tipos de modelos soportados
        self.supported_types = set(self.model_classes.keys())
        logger.info(f"Tipos de modelos soportados: {', '.join(self.supported_types)}")

    def create_model(
        self, model_type: Union[str, ModelType], config: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """
        Crea una instancia de modelo basada en el tipo y configuración.

        Args:
            model_type: Tipo de modelo a crear (string o ModelType enum)
            config: Diccionario de configuración del modelo (opcional)

        Returns:
            Instancia del modelo

        Raises:
            ValueError: Si el tipo de modelo no está soportado
        """
        try:
            # Convertir a string si es ModelType
            if isinstance(model_type, ModelType):
                model_type = model_type.value

            # Normalizar el tipo de modelo
            model_type = model_type.lower().replace("-", "_")

            # Verificar si el tipo de modelo está soportado
            if model_type not in self.supported_types:
                supported = ", ".join(sorted(self.supported_types))
                raise ValueError(
                    f"Tipo de modelo no soportado: {model_type}. "
                    f"Tipos soportados: {supported}"
                )

            # Combinar configuraciones
            model_config = {}
            if self.config:
                if hasattr(self.config, "to_dict"):
                    model_config.update(self.config.to_dict())
                elif isinstance(self.config, dict):
                    model_config.update(self.config)
                else:
                    model_config.update(vars(self.config))
            if config:
                model_config.update(config)

            # Crear instancia del modelo
            model_class = self.model_classes[model_type]
            model = model_class(model_config)

            logger.info(f"Modelo {model_type} creado exitosamente")
            return model

        except Exception as e:
            logger.error(f"Error creando modelo {model_type}: {str(e)}")
            raise
