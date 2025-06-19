"""Ensemble agent implementation with safe async training."""

import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
import logging
import torch

from ..base import BaseModel
from ..model_types import ModelType
from ..transformers.patchtst import PatchTST
from ..rl_agents.sac import SACTradingAgent, TradingEnvironmentSAC

from ...data import DataPreprocessor

from ..rl_agents.td3 import TD3TradingAgent

# Asumiendo que existe un entorno para TD3, si no, se debe crear o adaptar.
# from ..rl_agents.td3 import TradingEnvironmentTD3
from ..rl_agents.rainbow_dqn import RainbowTradingAgent

# Asumiendo que existe un entorno para Rainbow, si no, se debe crear o adaptar.
# from ..rl_agents.rainbow_dqn import TradingEnvironmentRainbow

logger = logging.getLogger(__name__)


# Clase de entorno genérica o marcador de posición si los específicos no existen
class BaseTradingEnvironment:
    def __init__(self, market_data, feature_columns, **kwargs):
        self.market_data = market_data
        self.feature_columns = feature_columns

        # Placeholder for gym.Space
        class ActionSpace:
            def __init__(self, shape=None, n=None):
                self.shape = shape
                self.n = n

        if kwargs.get("action_type") == "continuous":
            self.action_space = ActionSpace(shape=(kwargs.get("action_dim", 1),))
        else:
            self.action_space = ActionSpace(n=kwargs.get("action_dim", 3))

        self.state_dim = len(feature_columns) * kwargs.get("lookback_window", 30) + 4


class EnsembleAgent(BaseModel):
    """
    Un ensemble de modelos que pueden ser entrenados y utilizados para predicción.
    Implementa entrenamiento asíncrono seguro usando copias aisladas de los datos.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el EnsembleAgent.

        Args:
            config: Diccionario de configuración general de la aplicación.
        """
        super().__init__(config)
        self.config = config  # Guardar la configuración completa
        dp_conf = (
            config.get("data", {})
            if isinstance(config, dict)
            else getattr(config, "data", {})
        )
        self.preprocessor = DataPreprocessor(dp_conf)
        if isinstance(dp_conf, dict):
            self.lookback_window = dp_conf.get("lookback_window", 30)
        else:
            self.lookback_window = getattr(dp_conf, "lookback_window", 30)

        # Definir las columnas de características para cada tipo de modelo
        self.feature_columns = {
            "patchtst": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "macd",
                "macd_signal",
                "macd_hist",
                "bb_upper",
                "bb_middle",
                "bb_lower",
            ],
            "sac": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "macd",
                "macd_signal",
                "macd_hist",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "returns",
                "volatility",
                "spread",
            ],
            "td3": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "macd",
                "macd_signal",
                "macd_hist",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "returns",
                "volatility",
                "spread",
            ],
            "rainbow_dqn": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "rsi",
                "macd",
                "macd_signal",
                "macd_hist",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "returns",
                "volatility",
                "spread",
            ],
        }

        # Obtener configuración de modelos del ensemble
        self.model_configs = {}
        # Lógica para procesar la configuración desde un objeto o un diccionario
        if hasattr(config, "models"):
            models_cfg = getattr(config, "models")
            ensemble_cfg = models_cfg.get("ensemble", {}) if isinstance(models_cfg, dict) else {}
            if "models_to_train" in ensemble_cfg:
                config.ensemble_models = [ModelType(m) for m in ensemble_cfg["models_to_train"]]
            for model_name in ["patchtst", "tft", "sac", "td3", "rainbow_dqn"]:
                if f"{model_name}_params" in models_cfg:
                    self.model_configs[model_name] = models_cfg[f"{model_name}_params"]

        if hasattr(config, "ensemble_models") and hasattr(config, "rl"):
            for model_type in config.ensemble_models:
                if model_type == ModelType.PATCHTST:
                    params = config.transformer.__dict__.copy()
                    params.update(self.model_configs.get("patchtst", {}))
                    self.model_configs["patchtst"] = params
                elif model_type == ModelType.TFT:
                    params = config.transformer.__dict__.copy()
                    params.update(self.model_configs.get("tft", {}))
                    self.model_configs["tft"] = params
                elif model_type in [
                    ModelType.SAC,
                    ModelType.TD3,
                    ModelType.RAINBOW_DQN,
                ]:
                    rl_config = config.rl.__dict__.copy()
                    model_name = model_type.name.lower()

                    if "hidden_dims" in rl_config and not isinstance(
                        rl_config["hidden_dims"], (list, tuple)
                    ):
                        rl_config["hidden_dims"] = (
                            rl_config["hidden_dims"],
                            rl_config["hidden_dims"],
                        )

                    rl_config.update(self.model_configs.get(model_name, {}))
                    rl_config.setdefault(
                        "lookback_window",
                        getattr(config.data, "lookback_window", self.lookback_window),
                    )
                    rl_config.setdefault(
                        "state_dim",
                        len(self.feature_columns.get(model_name, []))
                        * int(rl_config.get("lookback_window", self.lookback_window))
                        + 4,
                    )
                    rl_config.setdefault(
                        "action_dim", 3 if model_name != "rainbow_dqn" else 8
                    )
                    self.model_configs[model_name] = rl_config

        elif isinstance(config, dict):
            self.model_configs = config.get("models", {})
            for model_type in ["sac", "td3", "rainbow_dqn"]:
                if model_type in self.model_configs:
                    model_config = self.model_configs[model_type]
                    if "hidden_dims" in model_config and not isinstance(
                        model_config["hidden_dims"], (list, tuple)
                    ):
                        model_config["hidden_dims"] = (
                            model_config["hidden_dims"],
                            model_config["hidden_dims"],
                        )

                    model_config.setdefault(
                        "lookback_window",
                        config.get("data", {}).get(
                            "lookback_window", self.lookback_window
                        ),
                    )
                    model_config.setdefault(
                        "state_dim",
                        len(self.feature_columns.get(model_type, []))
                        * int(model_config["lookback_window"])
                        + 4,
                    )
                    model_config.setdefault(
                        "action_dim", 3 if model_type != "rainbow_dqn" else 8
                    )

        if not self.model_configs:
            logger.info("Usando configuración por defecto para modelos del ensemble")
            self.model_configs = self._get_default_configs()

        # Inicializar modelos
        self.models = {}
        self._initialize_models()
        self.trained = False

        if not self.models:
            logger.warning("No se pudieron inicializar modelos para el ensemble")
        else:
            logger.info(
                f"Ensemble inicializado con {len(self.models)} modelos: {', '.join(self.models.keys())}"
            )

    def _get_default_configs(self):
        """Retorna las configuraciones por defecto para los modelos."""
        return {
            "patchtst": {
                "n_vars": len(self.feature_columns["patchtst"]),
                "seq_len": self.lookback_window,
                "pred_len": 24,
                "patch_len": 16,
                "stride": 8,
                "d_model": 128,
                "n_heads": 8,
                "e_layers": 3,
                "dropout": 0.1,
            },
            "tft": {
                "hidden_size": 160,
                "lstm_layers": 2,
                "attention_heads": 4,
                "dropout": 0.1,
            },
            "sac": {
                "state_dim": len(self.feature_columns["sac"]) * self.lookback_window
                + 4,
                "lookback_window": self.lookback_window,
                "action_dim": 3,
                "hidden_dims": (256, 256),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "alpha": 0.2,
                "batch_size": 256,
                "buffer_size": 1000000,
                "gradient_steps": 1,
                "train_freq": 1,
                "start_timesteps": 10000,
                "policy_delay": 2,
                "target_update_interval": 1,
                "target_noise": 0.2,
                "noise_clip": 0.5,
                "exploration_noise": 0.1,
            },
            "td3": {
                "state_dim": len(self.feature_columns["td3"]) * self.lookback_window
                + 4,
                "lookback_window": self.lookback_window,
                "action_dim": 1,
                "hidden_dims": (256, 256),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 256,
                "buffer_size": 1000000,
            },
            "rainbow_dqn": {
                "state_dim": len(self.feature_columns["rainbow_dqn"]) * self.lookback_window
                + 4,
                "lookback_window": self.lookback_window,
                "action_dim": 8,
                "hidden_dims": (256, 256),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "batch_size": 256,
                "buffer_size": 100000,
            },
        }

    def _initialize_models(self):
        """Inicializa los modelos del ensemble según la configuración."""
        try:
            model_map = {
                "patchtst": (PatchTST, {}),
                "tft": (PatchTST, {}),  # TODO: replace with real TFT model
                "sac": (SACTradingAgent, {}),
                "td3": (TD3TradingAgent, {}),
                "rainbow_dqn": (RainbowTradingAgent, {}),
            }

            for name, (model_class, _) in model_map.items():
                if name in self.model_configs:
                    model_config = self.model_configs[name].copy()
                    model_config["feature_columns"] = self.feature_columns.get(name, [])
                    model_config.setdefault("lookback_window", self.lookback_window)
                    model_config.setdefault(
                        "state_dim",
                        len(model_config["feature_columns"])
                        * int(model_config["lookback_window"])
                        + 4,
                    )

                    if "hidden_dims" in model_config:
                        dims = model_config["hidden_dims"]
                        if isinstance(dims, dict):
                            model_config["hidden_dims"] = tuple(
                                int(v) for v in dims.values()
                            )
                        elif isinstance(dims, (list, tuple)):
                            model_config["hidden_dims"] = tuple(int(d) for d in dims)
                        else:
                            model_config["hidden_dims"] = (int(dims), int(dims))

                    if name in ["sac", "td3", "rainbow_dqn"]:
                        # Para agentes RL, pasar la config como argumento de palabra clave
                        logger.info(
                            f"Inicializando {name.upper()} con config: {model_config}"
                        )
                        # El constructor de SACTradingAgent espera `config` como kwarg
                        self.models[name] = model_class(config=model_config)

                    elif name == "patchtst":
                        # PatchTST espera la config directamente
                        model_config.setdefault(
                            "n_vars", len(model_config["feature_columns"])
                        )
                        self.models[name] = model_class(model_config)

                    logger.info(f"Modelo {name.upper()} inicializado en el ensemble")

        except Exception as e:
            logger.error(
                f"Error inicializando modelos del ensemble: {str(e)}", exc_info=True
            )
            raise

    async def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Entrena los modelos del ensemble de forma asíncrona.
        """
        loop = asyncio.get_running_loop()
        tasks = []
        executor = ThreadPoolExecutor()

        for model_name, model in self.models.items():
            try:
                model_train_data = train_data.copy()
                model_val_data = val_data.copy() if val_data is not None else None

                # Seleccionar columnas de características específicas
                feature_cols = self.feature_columns.get(model_name, [])
                if not all(col in model_train_data.columns for col in feature_cols):
                    logger.info(
                        f"Aplicando preprocesamiento para {model_name} por columnas faltantes"
                    )
                    processed = self.preprocessor.fit_transform(model_train_data)
                    model_train_data = processed[feature_cols]
                    if model_val_data is not None:
                        model_val_data = self.preprocessor.transform(model_val_data)[
                            feature_cols
                        ]
                else:
                    model_train_data = model_train_data[feature_cols].copy()
                    if model_val_data is not None:
                        model_val_data = model_val_data[feature_cols].copy()

                logger.info(
                    f"Preparando entrenamiento para {model_name} con {len(model_train_data)} muestras y {len(feature_cols)} características."
                )

                task_func = (
                    self._train_patchtst
                    if model_name == "patchtst"
                    else self._train_rl_agent
                )

                # Para modelos que no son async, envolver en run_in_executor
                if not asyncio.iscoroutinefunction(
                    model.fit if hasattr(model, "fit") else model.train
                ):
                    task = loop.run_in_executor(
                        executor,
                        task_func,
                        model_name,
                        model,
                        model_train_data,
                        model_val_data,
                        kwargs,
                    )
                else:
                    task = task_func(
                        model_name, model, model_train_data, model_val_data, kwargs
                    )

                tasks.append((model_name, task))

            except Exception as e:
                logger.error(
                    f"Error preparando entrenamiento para {model_name}: {str(e)}",
                    exc_info=True,
                )
                raise

        if not tasks:
            logger.warning("No hay modelos configurados para entrenar")
            return {}

        logger.info("Iniciando entrenamiento asíncrono de modelos...")
        results = {}
        for model_name, task in tasks:
            try:
                result = await task
                results[model_name] = result
                logger.info(f"Entrenamiento de {model_name} completado exitosamente")
            except Exception as e:
                logger.error(
                    f"Error en entrenamiento de {model_name}: {str(e)}", exc_info=True
                )
                raise

        executor.shutdown(wait=True)
        self.trained = True
        return results

    def _train_patchtst(
        self,
        model_name: str,
        model: PatchTST,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Entrena modelo PatchTST de forma síncrona."""
        try:
            logger.info(f"Entrenando PatchTST con {train_data.shape[0]} registros.")
            feature_cols = self.feature_columns[model_name]
            # Validar y reordenar columnas
            if isinstance(train_data, pd.DataFrame):
                if list(train_data[feature_cols].columns) != feature_cols:
                    train_data = train_data[feature_cols]
            elif isinstance(train_data, dict):
                if "market_data" in train_data and "feature_columns" in train_data:
                    df = train_data["market_data"]
                    cols = train_data["feature_columns"]
                    if list(df[cols].columns) != cols:
                        train_data["market_data"] = df[cols]
            if val_data is not None:
                if isinstance(val_data, pd.DataFrame):
                    if list(val_data[feature_cols].columns) != feature_cols:
                        val_data = val_data[feature_cols]
                elif isinstance(val_data, dict):
                    if "market_data" in val_data and "feature_columns" in val_data:
                        df = val_data["market_data"]
                        cols = val_data["feature_columns"]
                        if list(df[cols].columns) != cols:
                            val_data["market_data"] = df[cols]
            train_dict = {"market_data": train_data, "feature_columns": feature_cols}
            val_dict = (
                {"market_data": val_data, "feature_columns": feature_cols}
                if val_data is not None
                else None
            )
            return model.fit(train_dict, val_dict, **kwargs)
        except Exception as e:
            logger.error(f"Error en entrenamiento de PatchTST: {str(e)}", exc_info=True)
            raise

    def _train_rl_agent(
        self,
        model_name: str,
        model: Any,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Entrena un agente de RL."""
        try:
            logger.info(
                f"Entrenando agente RL '{model_name}' con {train_data.shape[0]} registros."
            )

            if model_name == "sac":
                # El agente SAC espera recibir los datos de mercado y las columnas
                # de características por separado.
                return model.fit(
                    train_data,
                    self.feature_columns[model_name],
                    **kwargs,
                )
            else:
                # Otros agentes de RL utilizan el formato tradicional
                # train_data y opcionalmente val_data.
                return model.fit(train_data, val_data, **kwargs)
        except Exception as e:
            logger.error(
                f"Error en entrenamiento de {model_name}: {str(e)}", exc_info=True
            )
            raise

    def predict(self, data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Genera predicciones usando el ensemble de modelos.
        """
        predictions = {}
        for model_name, model in self.models.items():
            try:
                feature_cols = self.feature_columns.get(model_name, [])
                if not all(col in data.columns for col in feature_cols):
                    logger.info(
                        f"Aplicando preprocesamiento para predicción de {model_name}"
                    )
                    processed = self.preprocessor.transform(data)
                    model_data = processed[feature_cols]
                else:
                    model_data = data[feature_cols].copy()
                pred = model.predict(model_data, **kwargs)
                predictions[model_name] = pred

            except Exception as e:
                logger.error(
                    f"Error en predicción de {model_name}: {str(e)}", exc_info=True
                )
                raise

        return predictions

    def state_dict(self) -> Dict[str, Any]:
        """Retorna el estado del ensemble."""
        return {
            "config": self.config,
            "model_configs": self.model_configs,
            "feature_columns": self.feature_columns,
            "trained": self.trained,
            "models": {
                name: model.state_dict()
                for name, model in self.models.items()
                if hasattr(model, "state_dict")
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Carga el estado del ensemble."""
        self.config = state_dict.get("config", self.config)
        self.model_configs = state_dict.get("model_configs", self.model_configs)
        self.feature_columns = state_dict.get("feature_columns", self.feature_columns)
        self.trained = state_dict.get("trained", self.trained)

        if not self.models:
            self._initialize_models()

        for name, model_state in state_dict.get("models", {}).items():
            if name in self.models and hasattr(self.models[name], "load_state_dict"):
                try:
                    self.models[name].load_state_dict(model_state)
                except Exception as e:
                    logger.warning(
                        f"No se pudo cargar el estado del modelo {name}: {str(e)}"
                    )

    def save(self, path: str):
        """Guarda los modelos del ensemble."""
        if not path:
            return self.state_dict()

        for model_name, model in self.models.items():
            try:
                model_path = f"{path}/{model_name}.pth"
                if hasattr(
                    model, "save"
                ):  # Para modelos con su propia lógica de guardado
                    model.save(f"{path}/{model_name}")
                elif hasattr(model, "state_dict"):
                    torch.save(model.state_dict(), model_path)
                logger.info(f"Modelo {model_name} guardado en {model_path}")
            except Exception as e:
                logger.error(
                    f"Error guardando modelo {model_name}: {str(e)}", exc_info=True
                )

    def load(self, path: str):
        """Carga los modelos del ensemble."""
        for model_name, model in self.models.items():
            try:
                model_path_pth = f"{path}/{model_name}.pth"
                if hasattr(model, "load") and not hasattr(
                    model, "load_state_dict"
                ):  # Modelos con su propia lógica
                    model.load(f"{path}/{model_name}")
                elif hasattr(model, "load_state_dict"):
                    state = torch.load(
                        model_path_pth, map_location=lambda storage, loc: storage
                    )
                    model.load_state_dict(state)
                logger.info(f"Modelo {model_name} cargado desde {path}")
            except FileNotFoundError:
                logger.warning(
                    f"No se encontró el archivo para el modelo {model_name} en {path}, se omitió la carga."
                )
            except Exception as e:
                logger.error(
                    f"Error cargando modelo {model_name}: {str(e)}", exc_info=True
                )

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Entrena los modelos del ensemble de forma síncrona.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Si ya hay un bucle corriendo, no se puede usar run_until_complete
                # Esto es común en notebooks. Se puede manejar de otra forma,
                # pero por simplicidad, lanzamos un error o advertencia.
                logger.warning(
                    "fit() llamado en un bucle de eventos ya en ejecución. El comportamiento puede ser inesperado."
                )
                # Una opción es crear una nueva tarea y esperar
                task = loop.create_task(self.train(train_data, val_data, **kwargs))
                # Esta parte requeriría una gestión más compleja del bucle.
                # Por ahora, mantendremos la llamada a run_until_complete.
        except RuntimeError:  # No running event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        history = loop.run_until_complete(self.train(train_data, val_data, **kwargs))
        self.is_trained = True
        return history
