"""
Advanced Ensemble Methods for Hyperion V2
Combines multiple models for superior performance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import optuna
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from .sac import TradingEnvironmentSAC, SACTradingAgent
from ..transformers.patchtst import PatchTST

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    """Base class for ensemble methods"""

    def __init__(self, models: Dict[str, Any], config: Dict):
        self.models = models
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loop = asyncio.get_event_loop()
        self.trained = False

    @abstractmethod
    def fit(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Train the ensemble synchronously"""
        pass

    @abstractmethod
    async def fit_async(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Train the ensemble asynchronously"""
        pass

    @abstractmethod
    def predict(self, x: Any) -> np.ndarray:
        """
        Make predictions.

        Args:
            x: Input data

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Get ensemble weights"""
        pass

    def save(self, filepath: str):
        """Save ensemble configuration"""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "model_names": list(self.models.keys()),
                    "weights": self.get_weights(),
                },
                f,
            )

    def load(self, path: str) -> None:
        """
        Load ensemble configuration.

        Args:
            path: Path to load from
        """
        config = torch.load(path)
        self.config = config["config"]
        # Los modelos deben ser cargados por separado

    async def train(self, data: Dict[str, np.ndarray], **kwargs) -> None:
        """
        Train all models in the ensemble

        Args:
            data: Dictionary containing training data for each model
            **kwargs: Additional training parameters
        """
        try:
            # Crear tareas para entrenar cada modelo
            tasks = []
            for i, model in enumerate(self.models):
                if hasattr(model, "train"):
                    if asyncio.iscoroutinefunction(model.train):
                        # Si el modelo tiene un método train asíncrono
                        task = self.loop.create_task(
                            model.train(data.get(f"model_{i}", data), **kwargs)
                        )
                    else:
                        # Si el modelo tiene un método train síncrono
                        task = self.loop.create_task(
                            self._run_sync_train(
                                model, data.get(f"model_{i}", data), **kwargs
                            )
                        )
                    tasks.append(task)

            # Esperar a que todos los modelos terminen de entrenar
            await asyncio.gather(*tasks, loop=self.loop)
            self.trained = True

        except Exception as e:
            logger.error(f"Error en entrenamiento del ensemble: {str(e)}")
            raise
        finally:
            # Limpiar el loop de eventos
            self.loop.close()

    async def _run_sync_train(
        self, model: Any, data: Dict[str, np.ndarray], **kwargs
    ) -> None:
        """Ejecutar entrenamiento síncrono en un hilo separado"""

        def train_sync():
            try:
                model.train(data, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error entrenando modelo {model.__class__.__name__}: {str(e)}"
                )
                raise

        # Ejecutar entrenamiento síncrono en un hilo separado usando el loop correcto
        await self.loop.run_in_executor(None, train_sync)


class WeightedVotingEnsemble(BaseEnsemble):
    """Ensemble with weighted voting"""

    def __init__(self, models: Dict[str, Any], config: Dict):
        super().__init__(models, config)
        self.weight_method = config.get("weight_method", "equal")
        self.weights = {name: 1.0 / len(models) for name in models.keys()}
        self.model_metrics = {}

        # Definir columnas de características específicas para cada modelo
        self.feature_columns = {
            "patchtst": [
                "log_return",
                "volatility",
                "rsi",
                "sma_20",
                "std_20",
                "upper_band",
                "lower_band",
                "macd",
                "macd_signal",
                "volume_norm",
                "momentum",
                "atr",
            ],
            "sac": [
                "log_return",
                "volatility",
                "rsi",
                "sma_20",
                "std_20",
                "upper_band",
                "lower_band",
                "macd",
                "macd_signal",
                "volume_norm",
                "momentum",
                "atr",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        }

    def fit(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Train the ensemble synchronously"""
        try:
            self.loop.run_until_complete(self.fit_async(train_data, val_data))
        except Exception as e:
            logger.error(f"Error en entrenamiento del ensemble: {str(e)}")
            raise

    async def fit_async(
        self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Train the ensemble asynchronously"""
        try:
            # Crear tareas de entrenamiento para cada modelo
            training_tasks = []
            for name, model in self.models.items():
                task = asyncio.create_task(
                    self._train_model(name, model, train_data, val_data)
                )
                training_tasks.append(task)

            # Ejecutar entrenamiento en paralelo
            await asyncio.gather(*training_tasks)

            # Actualizar pesos basados en métricas
            if val_data is not None:
                self._update_weights()

            logger.info("Ensemble entrenado exitosamente")

        except Exception as e:
            logger.error(f"Error en entrenamiento asíncrono del ensemble: {str(e)}")
            raise

    def _update_weights(self) -> None:
        """Update model weights based on validation metrics"""
        if self.weight_method == "equal":
            # Pesos iguales
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}

        elif self.weight_method == "performance":
            # Pesos basados en rendimiento
            total_performance = sum(
                metrics.get("sharpe_ratio", 0)
                for metrics in self.model_metrics.values()
            )
            if total_performance > 0:
                self.weights = {
                    name: metrics.get("sharpe_ratio", 0) / total_performance
                    for name, metrics in self.model_metrics.items()
                }
            else:
                self.weights = {
                    name: 1.0 / len(self.models) for name in self.models.keys()
                }

        logger.info(f"Pesos actualizados: {self.weights}")

    async def _train_model(
        self,
        name: str,
        model: Any,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Entrena un modelo base (PatchTST o SAC) con los datos correctos.
        """
        # Definir columnas para PatchTST y SAC
        patchtst_cols = [
            "log_return",
            "volatility",
            "rsi",
            "sma_20",
            "std_20",
            "upper_band",
            "lower_band",
            "macd",
            "macd_signal",
            "volume_norm",
            "momentum",
            "atr",
        ]
        sac_cols = patchtst_cols + ["open", "high", "low", "close", "volume"]

        try:
            if name == "patchtst":
                # Verificar que todas las columnas estén presentes y en el orden correcto
                missing = [
                    col for col in patchtst_cols if col not in train_data.columns
                ]
                logging.error(
                    f"PatchTST columnas presentes: {train_data.columns.tolist()}"
                )
                if missing:
                    logging.error(f"Faltan columnas para PatchTST: {missing}")
                    raise ValueError(f"Faltan columnas para PatchTST: {missing}")

                # Asegurar que las columnas estén en el orden correcto
                df_patch = train_data[patchtst_cols].copy()
                if len(df_patch.columns) != 12:
                    raise ValueError(
                        f"PatchTST espera 12 columnas, pero recibió {len(df_patch.columns)}: {df_patch.columns.tolist()}"
                    )

                # Preparar datos de entrenamiento
                train_dict = {"market_data": df_patch, "feature_columns": patchtst_cols}

                # Preparar datos de validación si existen
                val_dict = None
                if val_data is not None:
                    missing_val = [
                        col for col in patchtst_cols if col not in val_data.columns
                    ]
                    logging.error(
                        f"PatchTST VAL columnas presentes: {val_data.columns.tolist()}"
                    )
                    if missing_val:
                        logging.error(
                            f"Faltan columnas para PatchTST VAL: {missing_val}"
                        )
                        raise ValueError(
                            f"Faltan columnas para PatchTST VAL: {missing_val}"
                        )

                    # Asegurar que las columnas estén en el orden correcto
                    df_val_patch = val_data[patchtst_cols].copy()
                    if len(df_val_patch.columns) != 12:
                        raise ValueError(
                            f"PatchTST VAL espera 12 columnas, pero recibió {len(df_val_patch.columns)}: {df_val_patch.columns.tolist()}"
                        )

                    val_dict = {
                        "market_data": df_val_patch,
                        "feature_columns": patchtst_cols,
                    }

                # Configurar y entrenar modelo
                model.n_vars = len(patchtst_cols)
                if not hasattr(model, "fit"):
                    raise RuntimeError("El modelo PatchTST no tiene método fit.")
                model.fit(train_dict, val_dict)
                logger.info("PatchTST entrenado correctamente.")

            elif name == "sac":
                # Verificar columnas para SAC
                missing = [col for col in sac_cols if col not in train_data.columns]
                logging.error(f"SAC columnas presentes: {train_data.columns.tolist()}")
                if missing:
                    logging.error(f"Faltan columnas para SAC: {missing}")
                    raise ValueError(f"Faltan columnas para SAC: {missing}")

                # Asegurar que las columnas estén en el orden correcto
                df_sac = train_data[sac_cols].copy()
                if len(df_sac.columns) != 17:
                    raise ValueError(
                        f"SAC espera 17 columnas, pero recibió {len(df_sac.columns)}: {df_sac.columns.tolist()}"
                    )

                # Crear entorno de trading
                env = TradingEnvironmentSAC(
                    market_data=df_sac,
                    feature_columns=sac_cols,
                    lookback_window=30,
                    transaction_cost=0.001,
                    max_position_size=1.0,
                    reward_scaling=1.0,
                    max_steps=1000,
                )

                # Configurar dimensiones del modelo
                model.state_dim = env.state_dim
                model.action_dim = 3

                # Inicializar redes si es necesario
                if hasattr(model, "_initialize_networks"):
                    model._initialize_networks()

                # Entrenar modelo
                if not hasattr(model, "train"):
                    raise RuntimeError("El modelo SAC no tiene método train.")
                model.train(env, val_env=None)
                logger.info("SAC entrenado correctamente.")

            else:
                raise ValueError(f"Modelo no soportado: {name}")

        except Exception as e:
            logger.error(f"Error entrenando modelo {name}: {str(e)}")
            raise

    def _evaluate_model(self, model: Any, val_data: Any) -> Dict[str, float]:
        """Evaluar modelo en datos de validación"""
        try:
            if hasattr(model, "evaluate"):
                return model.evaluate(val_data)
            elif hasattr(model, "predict"):
                predictions = model.predict(val_data)
                if isinstance(val_data, tuple):
                    _, y_true = val_data
                else:
                    y_true = val_data
                return {
                    "mse": np.mean((predictions - y_true) ** 2),
                    "rmse": np.sqrt(np.mean((predictions - y_true) ** 2)),
                }
            else:
                return {"error": "No se puede evaluar el modelo"}
        except Exception as e:
            return {"error": str(e)}

    def predict(self, x: Any) -> np.ndarray:
        """Make weighted ensemble prediction"""
        predictions = []

        for i, (name, model) in enumerate(self.models.items()):
            # Get prediction from each model
            if hasattr(model, "predict"):
                pred = model.predict(x)
            elif hasattr(model, "forward"):
                # PyTorch model
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x).to(self.device)
                    pred = model(x_tensor).cpu().numpy()
            else:
                # Assume callable
                pred = model(x)

            predictions.append(pred * self.weights[name])

        # Weighted average
        return np.sum(predictions, axis=0)

    def get_weights(self) -> np.ndarray:
        """Get current weights"""
        return np.array(list(self.weights.values()))

    def update_weights(self, new_weights: np.ndarray):
        """Update ensemble weights"""
        assert len(new_weights) == len(self.models)
        self.weights = {name: w for name, w in zip(self.models.keys(), new_weights)}


class StackingEnsemble(BaseEnsemble):
    """Stacking ensemble with meta-learner"""

    def __init__(self, models: Dict[str, Any], config: Dict):
        super().__init__(models, config)
        self.meta_learner_type = config.get("meta_learner_type", "linear")
        self.meta_learner = self._create_meta_learner()
        self.is_trained = False

    def _create_meta_learner(self):
        """Create meta-learner based on type"""
        if self.meta_learner_type == "linear":
            return LinearRegression()

        elif self.meta_learner_type == "neural":
            # Neural network meta-learner
            n_models = len(self.models)
            return nn.Sequential(
                nn.Linear(n_models, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            ).to(self.device)

        elif self.meta_learner_type == "xgboost":
            import xgboost as xgb

            return xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)

        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

    def _get_base_predictions(self, x: np.ndarray) -> np.ndarray:
        """Get predictions from all base models"""
        predictions = []

        for name, model in self.models.items():
            if hasattr(model, "predict"):
                pred = model.predict(x)
            elif hasattr(model, "forward"):
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x).to(self.device)
                    pred = model(x_tensor).cpu().numpy()
            else:
                pred = model(x)

            # Ensure 2D array
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)

            predictions.append(pred)

        # Stack predictions
        return np.hstack(predictions)

    def fit(self, train_data: Any, val_data: Optional[Any] = None) -> None:
        """Train the stacking ensemble"""
        try:
            # First train all base models
            for name, model in self.models.items():
                logger.info(f"Entrenando modelo base {name}...")
                if hasattr(model, "fit"):
                    model.fit(train_data, val_data)
                elif hasattr(model, "train"):
                    model.train(train_data, val_data)
                else:
                    logger.warning(f"Modelo {name} no tiene método fit o train")

            # Then train meta-learner
            self.train_meta_learner(train_data, val_data)

        except Exception as e:
            logger.error(f"Error entrenando ensemble: {str(e)}")
            raise

    def train_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """Train the meta-learner"""
        # Get base model predictions
        train_meta_features = self._get_base_predictions(X_train)

        if self.meta_learner_type == "neural":
            # Neural network training
            self._train_neural_meta_learner(train_meta_features, y_train, X_val, y_val)
        else:
            # Sklearn model training
            self.meta_learner.fit(train_meta_features, y_train)

        self.is_trained = True

    def _train_neural_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        epochs: int = 100,
    ):
        """Train neural network meta-learner"""
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)

        if X_val is not None:
            val_meta_features = self._get_base_predictions(X_val)
            X_val_t = torch.FloatTensor(val_meta_features).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Training loop
        for epoch in range(epochs):
            self.meta_learner.train()

            # Forward pass
            outputs = self.meta_learner(X_train_t).squeeze()
            loss = criterion(outputs, y_train_t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            if X_val is not None and epoch % 10 == 0:
                self.meta_learner.eval()
                with torch.no_grad():
                    val_outputs = self.meta_learner(X_val_t).squeeze()
                    val_loss = criterion(val_outputs, y_val_t)
                print(
                    f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make stacking ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained before prediction")

        # Get base predictions
        meta_features = self._get_base_predictions(x)

        # Meta-learner prediction
        if self.meta_learner_type == "neural":
            self.meta_learner.eval()
            with torch.no_grad():
                meta_features_t = torch.FloatTensor(meta_features).to(self.device)
                predictions = self.meta_learner(meta_features_t).cpu().numpy()
        else:
            predictions = self.meta_learner.predict(meta_features)

        return predictions.squeeze()

    def get_weights(self) -> np.ndarray:
        """Get feature importances from meta-learner"""
        if self.meta_learner_type == "linear":
            return self.meta_learner.coef_
        elif self.meta_learner_type == "xgboost":
            return self.meta_learner.feature_importances_
        else:
            # For neural networks, return equal weights
            return np.ones(len(self.models)) / len(self.models)


class BlendingEnsemble(BaseEnsemble):
    """Blending ensemble with validation set optimization"""

    def __init__(self, models: Dict[str, Any], config: Dict):
        super().__init__(models, config)
        self.optimization_method = config.get("optimization_method", "optuna")
        self.optimization_metric = config.get("optimization_metric", "sharpe_ratio")
        self.blend_weights = None

    def fit(self, train_data: Any, val_data: Optional[Any] = None) -> None:
        """Train the blending ensemble"""
        try:
            # First train all base models
            for name, model in self.models.items():
                logger.info(f"Entrenando modelo base {name}...")
                if hasattr(model, "fit"):
                    model.fit(train_data, val_data)
                elif hasattr(model, "train"):
                    model.train(train_data, val_data)
                else:
                    logger.warning(f"Modelo {name} no tiene método fit o train")

            # Then optimize weights if validation data is available
            if val_data is not None:
                if self.optimization_method == "optuna":
                    self._optimize_with_optuna(val_data, n_trials=100)
                else:
                    self._optimize_with_scipy(val_data)

        except Exception as e:
            logger.error(f"Error entrenando ensemble: {str(e)}")
            raise

    def _optimize_with_optuna(
        self, X_val: np.ndarray, y_val: np.ndarray, n_trials: int
    ):
        """Optimize weights using Optuna"""

        def objective(trial):
            # Sample weights
            weights = {}
            for name in self.models.keys():
                weights[name] = trial.suggest_float(f"weight_{name}", 0.0, 1.0)

            weights = {name: w / sum(weights.values()) for name, w in weights.items()}

            # Get predictions
            predictions = self._get_weighted_predictions(X_val, weights)

            # Calculate metric
            if self.optimization_metric == "sharpe_ratio":
                returns = np.diff(predictions) / predictions[:-1]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
                return sharpe
            elif self.optimization_metric == "mse":
                return -np.mean((predictions - y_val) ** 2)
            else:
                return np.corrcoef(predictions, y_val)[0, 1]

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Get best weights
        best_weights = {}
        for name in self.models.keys():
            best_weights[name] = study.best_params[f"weight_{name}"]

        self.blend_weights = best_weights

    def _optimize_with_scipy(self, X_val: np.ndarray, y_val: np.ndarray):
        """Optimize weights using scipy"""
        from scipy.optimize import minimize

        def objective(weights):
            # Normalize weights
            weights = {name: w / sum(weights.values()) for name, w in weights.items()}

            # Get predictions
            predictions = self._get_weighted_predictions(X_val, weights)

            # Calculate negative Sharpe ratio (to minimize)
            returns = np.diff(predictions) / predictions[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)

            return -sharpe

        # Initial weights
        n_models = len(self.models)
        initial_weights = {name: 1.0 / n_models for name in self.models.keys()}

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: sum(w.values()) - 1},  # Sum to 1
            {"type": "ineq", "fun": lambda w: w},  # Non-negative
        ]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n_models)],
        )

        self.blend_weights = {
            name: result.x[i] for i, name in enumerate(self.models.keys())
        }

    def _get_weighted_predictions(
        self, x: np.ndarray, weights: Dict[str, float]
    ) -> np.ndarray:
        """Get weighted predictions from models"""
        predictions = []

        for name, model in self.models.items():
            if hasattr(model, "predict"):
                pred = model.predict(x)
            elif hasattr(model, "forward"):
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x).to(self.device)
                    pred = model(x_tensor).cpu().numpy()
            else:
                pred = model(x)

            predictions.append(pred * weights[name])

        return np.sum(predictions, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make blending ensemble prediction"""
        if self.blend_weights is None:
            # Use equal weights if not optimized
            self.blend_weights = {
                name: 1.0 / len(self.models) for name in self.models.keys()
            }

        return self._get_weighted_predictions(x, self.blend_weights)

    def get_weights(self) -> np.ndarray:
        """Get blending weights"""
        if self.blend_weights is None:
            return np.ones(len(self.models)) / len(self.models)
        return np.array(list(self.blend_weights.values()))


class DynamicEnsemble(BaseEnsemble):
    """Dynamic ensemble that adapts to market regimes"""

    def __init__(self, models: Dict[str, Any], config: Dict):
        super().__init__(models, config)
        self.n_regimes = config.get("n_regimes", 3)
        self.adaptation_rate = config.get("adaptation_rate", 0.1)
        self.regime_weights = {
            i: {name: 1.0 / len(models) for name in models.keys()}
            for i in range(self.n_regimes)
        }

    def fit(self, train_data: Any, val_data: Optional[Any] = None) -> None:
        """Train the dynamic ensemble"""
        try:
            # First train all base models
            for name, model in self.models.items():
                logger.info(f"Entrenando modelo base {name}...")
                if hasattr(model, "fit"):
                    model.fit(train_data, val_data)
                elif hasattr(model, "train"):
                    model.train(train_data, val_data)
                else:
                    logger.warning(f"Modelo {name} no tiene método fit o train")

            # If validation data is available, initialize regime weights
            if val_data is not None:
                # Simple regime detection based on volatility
                returns = (
                    np.diff(val_data["target"].values) / val_data["target"].values[:-1]
                )
                volatility = np.std(returns)
                regimes = np.zeros(len(val_data))
                regimes[returns > volatility] = 1  # High volatility regime
                regimes[returns < -volatility] = 2  # Low volatility regime

                # Calculate performance in each regime
                for regime in range(self.n_regimes):
                    regime_mask = regimes == regime
                    if np.any(regime_mask):
                        regime_X = val_data[regime_mask]
                        regime_y = val_data["target"].values[regime_mask]
                        performances = {}
                        for name, model in self.models.items():
                            try:
                                pred = self.predict(regime_X)
                                performance = np.corrcoef(pred, regime_y)[0, 1]
                                performances[name] = max(0, performance)
                            except Exception as e:
                                logger.error(
                                    f"Error evaluando modelo {name} en régimen {regime}: {str(e)}"
                                )
                                performances[name] = 0.0

                        if performances:
                            self.update_weights(performances, regime)

        except Exception as e:
            logger.error(f"Error entrenando ensemble: {str(e)}")
            raise

    def detect_regime(self, market_features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Detect current market regime"""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(market_features).to(self.device)
            regime_logits = self.regime_detector(features_tensor)
            regime_probs = torch.softmax(regime_logits, dim=-1)
            regime = regime_probs.argmax().item()

        return regime, regime_probs.cpu().numpy()

    def predict(
        self, x: np.ndarray, market_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make dynamic ensemble prediction"""
        # Detect market regime if features provided
        if market_features is not None:
            regime, regime_probs = self.detect_regime(market_features)

            # Get regime-specific weights
            base_weights = self.regime_weights[regime]

            # Smooth transition between regimes
            weights = np.zeros_like(base_weights)
            for r, prob in enumerate(regime_probs):
                weights += prob * self.regime_weights[r]
        else:
            # Use equal weights if no regime detection
            weights = np.ones(len(self.models)) / len(self.models)

        # Get weighted predictions
        predictions = []
        for i, (name, model) in enumerate(self.models.items()):
            if hasattr(model, "predict"):
                pred = model.predict(x)
            elif hasattr(model, "forward"):
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x).to(self.device)
                    pred = model(x_tensor).cpu().numpy()
            else:
                pred = model(x)

            predictions.append(pred * weights[i])

        return np.sum(predictions, axis=0)

    def update_weights(self, performance_metrics: Dict[str, float], regime: int):
        """Update regime weights based on performance"""
        # Calculate relative performance
        performances = np.array(
            [performance_metrics.get(name, 0.0) for name in self.models.keys()]
        )

        # Normalize to [0, 1]
        if performances.max() > performances.min():
            normalized_perf = (performances - performances.min()) / (
                performances.max() - performances.min()
            )
        else:
            normalized_perf = np.ones_like(performances) * 0.5

        # Update weights with momentum
        old_weights = self.regime_weights[regime]
        new_weights = (
            1 - self.adaptation_rate
        ) * old_weights + self.adaptation_rate * normalized_perf

        # Normalize
        self.regime_weights[regime] = {
            name: w for name, w in zip(self.models.keys(), new_weights)
        }

    def get_weights(self) -> np.ndarray:
        """Get average weights across all regimes"""
        all_weights = np.array(list(self.regime_weights.values()))
        return all_weights.mean(axis=0)


class HierarchicalEnsemble(BaseEnsemble):
    """Hierarchical ensemble with multiple levels"""

    def __init__(self, models: Dict[str, Any], config: Dict):
        super().__init__(models, config)
        self.hierarchy = self._build_hierarchy()

    def fit(self, train_data: Any, val_data: Optional[Any] = None) -> None:
        """Train the hierarchical ensemble"""
        try:
            # Train each sub-ensemble
            for group_name, ensemble in self.hierarchy.items():
                logger.info(f"Entrenando sub-ensemble {group_name}...")
                ensemble.fit(train_data, val_data)

        except Exception as e:
            logger.error(f"Error entrenando sub-ensemble {group_name}: {str(e)}")
            raise

    def _build_hierarchy(self) -> Dict:
        """Build hierarchical structure"""
        # Group models by type
        model_groups = {"transformers": [], "rl_agents": [], "traditional": []}

        for name, model in self.models.items():
            if "transformer" in name.lower() or "tft" in name.lower():
                model_groups["transformers"].append(name)
            elif any(rl in name.lower() for rl in ["sac", "td3", "dqn", "ppo"]):
                model_groups["rl_agents"].append(name)
            else:
                model_groups["traditional"].append(name)

        # Create sub-ensembles for each group
        hierarchy = {}
        for group_name, model_names in model_groups.items():
            if model_names:
                group_models = {name: self.models[name] for name in model_names}
                hierarchy[group_name] = WeightedVotingEnsemble(
                    group_models, {"weight_method": "equal"}
                )

        return hierarchy

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make hierarchical prediction"""
        # Get predictions from each group
        group_predictions = {}
        for group_name, ensemble in self.hierarchy.items():
            group_predictions[group_name] = ensemble.predict(x)

        # Combine group predictions
        # Could use another ensemble method here
        final_predictions = []
        group_weights = self.config.get("group_weights", {})

        for group_name, pred in group_predictions.items():
            weight = group_weights.get(group_name, 1.0 / len(group_predictions))
            final_predictions.append(pred * weight)

        return np.sum(final_predictions, axis=0)

    def get_weights(self) -> np.ndarray:
        """Get flattened weights"""
        all_weights = []
        for group_name, ensemble in self.hierarchy.items():
            group_weight = self.config.get("group_weights", {}).get(
                group_name, 1.0 / len(self.hierarchy)
            )
            for w in ensemble.get_weights():
                all_weights.append(w * group_weight)

        return np.array(all_weights)


class AdaptiveEnsemble(BaseEnsemble):
    """Adaptive ensemble that selects models based on performance"""

    def __init__(self, models: Dict[str, Any], config: Dict):
        super().__init__(models, config)
        self.performance_window = config.get("performance_window", 100)
        self.min_models = config.get("min_models", 2)
        self.performance_history = {name: [] for name in models.keys()}
        self.active_models = list(models.keys())

    def fit(self, train_data: Any, val_data: Optional[Any] = None) -> None:
        """Train the adaptive ensemble"""
        try:
            # First train all base models
            for name, model in self.models.items():
                logger.info(f"Entrenando modelo base {name}...")
                if hasattr(model, "fit"):
                    model.fit(train_data, val_data)
                elif hasattr(model, "train"):
                    model.train(train_data, val_data)
                else:
                    logger.warning(f"Modelo {name} no tiene método fit o train")

            # If validation data is available, initialize active models
            if val_data is not None:
                performances = {}
                for name, model in self.models.items():
                    try:
                        pred = self.predict(val_data)
                        if len(pred.shape) > 1 and pred.shape[1] > 1:
                            pred = pred.argmax(axis=1)
                        performance = np.corrcoef(pred, val_data["target"].values)[0, 1]
                        performances[name] = max(0, performance)
                    except Exception as e:
                        logger.error(f"Error evaluando modelo {name}: {str(e)}")
                        performances[name] = 0.0

                if performances:
                    # Select top performing models
                    sorted_models = sorted(
                        performances.items(), key=lambda x: x[1], reverse=True
                    )
                    self.active_models = [
                        name
                        for name, _ in sorted_models[
                            : max(self.min_models, len(sorted_models) // 2)
                        ]
                    ]

        except Exception as e:
            logger.error(f"Error entrenando ensemble: {str(e)}")
            raise

    def update_performance(self, model_name: str, metric: float):
        """Update model performance history"""
        self.performance_history[model_name].append(metric)

        # Keep only recent history
        if len(self.performance_history[model_name]) > self.performance_window:
            self.performance_history[model_name].pop(0)

    def select_active_models(self):
        """Select models based on recent performance"""
        if all(len(hist) > 0 for hist in self.performance_history.values()):
            # Calculate average recent performance
            avg_performances = {
                name: np.mean(hist[-self.performance_window :])
                for name, hist in self.performance_history.items()
                if hist
            }

            # Select top performers
            sorted_models = sorted(
                avg_performances.items(), key=lambda x: x[1], reverse=True
            )

            # Keep models above threshold
            threshold = sorted_models[0][1] * self.selection_threshold
            self.active_models = [
                name for name, perf in sorted_models if perf >= threshold
            ]

            # Ensure at least 2 models are active
            if len(self.active_models) < 2:
                self.active_models = [name for name, _ in sorted_models[:2]]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make prediction using active models"""
        active_ensemble = WeightedVotingEnsemble(
            {name: self.models[name] for name in self.active_models},
            {"weight_method": "equal"},
        )

        return active_ensemble.predict(x)

    def get_weights(self) -> np.ndarray:
        """Get weights for all models (0 for inactive)"""
        weights = np.zeros(len(self.models))
        active_weight = 1.0 / len(self.active_models)

        for i, name in enumerate(self.models.keys()):
            if name in self.active_models:
                weights[i] = active_weight

        return weights


def create_ensemble(
    models: Dict[str, Any], ensemble_type: str, config: Dict
) -> BaseEnsemble:
    """Factory function to create ensemble"""

    ensemble_classes = {
        "voting": WeightedVotingEnsemble,
        "stacking": StackingEnsemble,
        "blending": BlendingEnsemble,
        "dynamic": DynamicEnsemble,
        "hierarchical": HierarchicalEnsemble,
        "adaptive": AdaptiveEnsemble,
    }

    if ensemble_type not in ensemble_classes:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    return ensemble_classes[ensemble_type](models, config)
