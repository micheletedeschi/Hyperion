"""
Trainer module for Hyperion V2
Handles model training, validation and optimization
"""

import logging
import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from tqdm import tqdm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from pathlib import Path
import glob
import asyncio

from hyperion3.models import ModelFactory
from hyperion3.config import get_config
from hyperion3.config.base_config import ModelType, HyperionV2Config
from hyperion3.optimization.flaml_optimizer import FLAMLOptimizer, ModelWrapper
from hyperion3.utils.metrics import calculate_metrics
from hyperion3.utils.data_utils import prepare_data_for_training
from hyperion3.data import DataPreprocessor
from hyperion3.data.augmentation import TimeSeriesAugmentor
from hyperion3.utils.metrics import FinancialMetrics
from hyperion3.utils.mlops import setup_mlops, validate_metrics
from hyperion3.models.rl_agents.sac import SACTradingAgent, TradingEnvironmentSAC
from hyperion3.models.rl_agents.td3 import TD3TradingAgent
from hyperion3.models.rl_agents.rainbow_dqn import RainbowTradingAgent
from hyperion3.models.rl_agents.ensemble import WeightedVotingEnsemble
from hyperion3.models.transformers.patchtst import PatchTST
from hyperion3.models.rl_agents.ensemble_agent import EnsembleAgent
from hyperion3.utils.model_utils import save_model, load_model

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for Hyperion V2"""

    def __init__(self, config: HyperionV2Config):
        """Initialize trainer

        Args:
            config: Configuration object
        """
        self.config = config
        self.model_factory = ModelFactory(config)
        self.metrics = FinancialMetrics()
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Initialize components
        self.model = None
        self.optimizer = None

        # Training state
        self.best_val_score = float("-inf")
        self.training_history = []

        # Setup directories
        self._setup_directories()

        # Setup MLOps tracking
        self._setup_mlops()

        # Data preprocessor and augmentor
        self.preprocessor = DataPreprocessor(self.config)
        self.augmentor = TimeSeriesAugmentor(self.config)

    def check_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate metrics against configured thresholds."""
        thresholds = self.config.mlops.alert_thresholds
        return validate_metrics(metrics, thresholds)

    def _setup_directories(self):
        """Create necessary directories"""
        dirs = [
            getattr(self.config, "data_dir", "data"),
            getattr(self.config, "model_dir", "models"),
            getattr(self.config, "log_dir", "logs"),
            getattr(self.config, "checkpoint_dir", "checkpoints"),
            "optimization_results",
            "backtest_results",
            "plots",
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _setup_mlops(self):
        """Setup MLOps tracking"""
        self.use_mlflow = self.config.mlops.use_mlflow
        self.use_wandb = self.config.mlops.use_wandb

        if self.use_mlflow:
            import mlflow

            mlflow.set_experiment(self.config.mlops.experiment_name)

        if self.use_wandb:
            import wandb

            wandb.init(
                project=self.config.mlops.experiment_name,
                config=(
                    self.config.to_dict()
                    if hasattr(self.config, "to_dict")
                    else dict(self.config)
                ),
            )

    async def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        model_type: Optional[ModelType] = None,
    ) -> Union[EnsembleAgent, Any]:
        """Train model

        Args:
            train_data: Training data
            val_data: Optional validation data
            model_type: Optional model type override

        Returns:
            Trained model
        """
        model_type = model_type or self.config.model_type

        if model_type == ModelType.ENSEMBLE:
            return await self._train_ensemble(train_data, val_data)

        # Create model instance
        self.model = self.model_factory.create_model(model_type)

        # Train model
        if model_type == ModelType.PATCHTST:
            await self._train_patchtst(train_data, val_data)
        elif model_type in [ModelType.SAC, ModelType.TD3]:
            await self._train_rl_model(train_data, val_data)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return self.model

    async def _train_ensemble(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None
    ) -> EnsembleAgent:
        """Train ensemble model

        Args:
            train_data: Training data
            val_data: Optional validation data

        Returns:
            Trained ensemble agent
        """
        # Configure base models
        base_models = {
            ModelType.PATCHTST: {
                "type": ModelType.PATCHTST,
                "config": self.config.patchtst_config,
            },
            ModelType.SAC: {"type": ModelType.SAC, "config": self.config.sac_config},
        }

        # Create ensemble agent
        ensemble = EnsembleAgent(model_configs=base_models, config=self.config)

        # Train ensemble
        try:
            await ensemble.train(train_data, val_data)
            logger.info("Ensemble training completed successfully")
            return ensemble
        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            raise

    async def _train_patchtst(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None
    ):
        """Train PatchTST model

        Args:
            train_data: Training data
            val_data: Optional validation data
        """
        try:
            # Prepare data
            X_train = train_data[self.config.patchtst_config.feature_columns].values
            y_train = train_data[self.config.patchtst_config.target_column].values

            if val_data is not None:
                X_val = val_data[self.config.patchtst_config.feature_columns].values
                y_val = val_data[self.config.patchtst_config.target_column].values
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )

            # Train model
            self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.patchtst_config.epochs,
                batch_size=self.config.patchtst_config.batch_size,
                callbacks=self.config.patchtst_config.callbacks,
            )

            logger.info("PatchTST training completed successfully")

        except Exception as e:
            logger.error(f"Error training PatchTST: {str(e)}")
            raise

    async def _train_rl_model(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None
    ):
        """Train RL model

        Args:
            train_data: Training data
            val_data: Optional validation data
        """
        try:
            # Train model
            await self.model.train(
                train_data,
                validation_data=val_data,
                epochs=self.config.rl_config.epochs,
                batch_size=self.config.rl_config.batch_size,
            )

            # Test on validation if available
            if val_data is not None:
                await self._test_rl_model(
                    self.model,
                    val_data,
                    self.config.model_type.value,
                    self.config.data.symbols[0],
                )

            logger.info(
                f"{self.config.model_type.value} training completed successfully"
            )

        except Exception as e:
            logger.error(f"Error training {self.config.model_type.value}: {str(e)}")
            raise

    def _create_dataloader(
        self, data: pd.DataFrame, batch_size: int
    ) -> torch.utils.data.DataLoader:
        """
        Crea un DataLoader para los datos de entrenamiento/validación

        Args:
            data: DataFrame con los datos
            batch_size: Tamaño del batch

        Returns:
            DataLoader configurado
        """
        # Convertir a numpy array
        data_array = data.values

        # Crear secuencias
        seq_len = self.config.data.lookback_window
        pred_len = self.config.data.prediction_horizon

        # Calcular número de muestras
        n_samples = len(data_array) - seq_len - pred_len + 1

        # Crear arrays para inputs y targets
        inputs = np.zeros((n_samples, seq_len, data_array.shape[1]))

        # Índice de la columna objetivo
        target_idx = data.columns.get_loc(self.config.data.target_column)

        # Targets solo con la columna objetivo
        targets = np.zeros((n_samples, pred_len, 1))

        # Llenar arrays
        for i in range(n_samples):
            inputs[i] = data_array[i : i + seq_len]
            targets[i, :, 0] = data_array[
                i + seq_len : i + seq_len + pred_len, target_idx
            ]

        # Convertir a tensores
        inputs_tensor = torch.FloatTensor(inputs)
        targets_tensor = torch.FloatTensor(targets)

        # Crear dataset y dataloader
        dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        return dataloader

    def _save_results(self, results: Dict[str, Any]):
        """Save training results"""
        os.makedirs("results", exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/training_{self.config.model_type}_{timestamp}.json"

        # Convertir resultados a formato serializable
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                serializable_results[key] = value
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, ModelWrapper):
                serializable_results[key] = {
                    "type": value.model_type,
                    "params": value.params,
                }
            else:
                try:
                    serializable_results[key] = str(value)
                except:
                    serializable_results[key] = f"<{type(value).__name__}>"

        try:
            with open(filename, "w") as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error guardando resultados: {str(e)}")
            # Intentar guardar una versión simplificada
            try:
                simplified_results = {
                    "error": str(e),
                    "timestamp": timestamp,
                    "model_type": self.config.model_type,
                }
                with open(f"results/error_{timestamp}.json", "w") as f:
                    json.dump(simplified_results, f, indent=2)
                logger.info(f"Error log saved to results/error_{timestamp}.json")
            except:
                logger.error("Failed to save error log")

    async def _train_sac(
        self, env: TradingEnvironmentSAC, agent: SACTradingAgent, params: Dict
    ) -> None:
        """Train SAC agent"""
        try:
            # Configurar parámetros de entrenamiento
            n_episodes = params.get("n_episodes", 1000)
            max_steps = params.get("max_steps", 1000)
            batch_size = params.get("batch_size", 256)
            update_interval = params.get("update_interval", 1)

            # Entrenar agente
            for episode in range(n_episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                step = 0

                while not done and step < max_steps:
                    # Seleccionar acción
                    action = agent.select_action(state)

                    # Ejecutar acción
                    next_state, reward, done, info = env.step(action)

                    # Guardar experiencia
                    agent.memory.push(state, action, reward, next_state, done)

                    # Actualizar estado y recompensa
                    state = next_state
                    episode_reward += reward
                    step += 1

                    # Actualizar red si hay suficientes muestras
                    if len(agent.memory) > batch_size and step % update_interval == 0:
                        agent.update_parameters(batch_size)

                # Logging
                if episode % 10 == 0:
                    logger.info(
                        f"Episodio {episode}: Recompensa = {episode_reward:.2f}"
                    )

        except Exception as e:
            logger.error(f"Error en entrenamiento de SAC: {str(e)}")
            raise

    async def _train_td3(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, params: Dict
    ) -> Any:
        """Train TD3 model"""
        try:
            from hyperion3.models.rl_agents.td3 import TD3TradingAgent

            logger.info("Iniciando entrenamiento de TD3...")

            # Crear y entrenar el agente TD3
            agent = TD3TradingAgent(config=self.config, device=self.device)

            # Entrenar el modelo
            await agent.train(train_data=train_data, val_data=val_data, **params)

            return agent

        except Exception as e:
            logger.error(f"Error en entrenamiento de TD3: {str(e)}")
            raise

    async def _train_rainbow(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, params: Dict
    ) -> Any:
        """Train Rainbow DQN model"""
        try:
            from hyperion3.models.rl_agents.rainbow_dqn import RainbowTradingAgent

            logger.info("Iniciando entrenamiento de Rainbow DQN...")

            # Crear y entrenar el agente Rainbow
            agent = RainbowTradingAgent(config=self.config, device=self.device)

            # Entrenar el modelo
            await agent.train(train_data=train_data, val_data=val_data, **params)

            return agent

        except Exception as e:
            logger.error(f"Error en entrenamiento de Rainbow DQN: {str(e)}")
            raise

    def _save_checkpoint(self, model: Any, model_type: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        try:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }

            checkpoint_path = checkpoint_dir / f"{model_type}_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint guardado en epoch {epoch}")

        except Exception as e:
            logger.error(f"Error guardando checkpoint: {str(e)}")

    def _mock_backtest(self, model: Any, data: pd.DataFrame) -> Dict[str, float]:
        """Return fake backtest metrics for testing purposes."""
        import random

        metrics = {
            "total_return": random.uniform(-0.1, 0.8),
            "annual_return": random.uniform(0.1, 0.5),
            "sharpe_ratio": random.uniform(0.5, 2.5),
            "sortino_ratio": random.uniform(0.7, 3.0),
            "max_drawdown": random.uniform(-0.25, -0.05),
            "win_rate": random.uniform(0.45, 0.65),
            "profit_factor": random.uniform(1.1, 2.5),
            "total_trades": random.randint(100, 1000),
        }
        self.check_metrics(metrics)
        return metrics


class ModelTrainer(Trainer):
    """Backward compatible alias for :class:`Trainer`."""

    pass
