"""
FLAML Optimizer for Hyperion V2
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
import flaml
from flaml import AutoML
import logging
from flaml.automl.model import LGBMEstimator
from hyperion3.config.model_configs import LGBMConfig, XGBoostConfig, CatBoostConfig
from hyperion3.utils.metrics import calculate_sharpe_ratio
from hyperion3.utils.logging import setup_logger
from hyperion3.config.base_config import HyperionV2Config
from sklearn.metrics import mean_squared_error
from hyperion3.models.rl_agents.ensemble import WeightedVotingEnsemble
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from ray import tune
from sklearn.model_selection import TimeSeriesSplit
import json
import pickle
import os

logger = setup_logger(__name__, "flaml_optimization.log")


@dataclass
class OptimizationResult:
    """Resultado de la optimización"""

    best_model: Any
    best_params: Dict[str, Any]
    metric: float
    sharpe_ratio: Optional[float]
    automl: AutoML


class ModelWrapper:
    """Wrapper para modelos que permite serialización"""

    def __init__(self, model, model_type: str, params: Dict[str, Any]):
        self.model = model
        self.model_type = model_type
        self.params = params

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.params

    def save(self, path: str):
        """Guarda el modelo usando pickle"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str, model_type: str, params: Dict[str, Any]):
        """Carga un modelo guardado"""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return cls(model, model_type, params)


class FLAMLOptimizer:
    """Optimizador de hiperparámetros usando validación cruzada"""

    def __init__(self, config):
        """
        Inicializa el optimizador

        Args:
            config: Configuración del optimizador
        """
        self.config = config
        self.best_score = float("inf")
        self.best_params = None
        self.best_model = None

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Optimiza los hiperparámetros usando validación cruzada

        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)

        Returns:
            Diccionario con resultados de la optimización
        """
        try:
            logger.info("Iniciando optimización de hiperparámetros...")

            # Definir espacios de búsqueda para cada modelo
            lgbm_params = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.3],
                "num_leaves": [31, 63, 127],
                "max_depth": [3, 5, 7],
                "min_child_samples": [10, 20, 50],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [0.0, 0.1, 1.0],
            }

            xgb_params = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0.0, 0.1, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [0.0, 0.1, 1.0],
            }

            cat_params = {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.3],
                "depth": [3, 5, 7],
                "l2_leaf_reg": [1.0, 3.0, 5.0],
                "bootstrap_type": ["Bernoulli", "Bayesian"],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bylevel": [0.6, 0.8, 1.0],
                "min_data_in_leaf": [1, 10, 20],
                "random_strength": [0.1, 1.0, 10.0],
            }

            # Configurar validación cruzada
            tscv = TimeSeriesSplit(n_splits=3)

            # Optimizar cada modelo
            best_models = {}
            best_scores = {}
            best_params = {}

            # LightGBM
            logger.info("Optimizando LightGBM...")
            best_lgbm_score = float("inf")
            best_lgbm_params = None
            best_lgbm_model = None

            for n_estimators in lgbm_params["n_estimators"]:
                for learning_rate in lgbm_params["learning_rate"]:
                    for num_leaves in lgbm_params["num_leaves"]:
                        for max_depth in lgbm_params["max_depth"]:
                            params = {
                                "n_estimators": n_estimators,
                                "learning_rate": learning_rate,
                                "num_leaves": num_leaves,
                                "max_depth": max_depth,
                                "verbose": -1,
                            }
                            model = LGBMRegressor(**params)

                            # Validación cruzada
                            scores = []
                            for train_idx, val_idx in tscv.split(X_train):
                                X_fold_train = X_train.iloc[train_idx]
                                y_fold_train = y_train.iloc[train_idx]
                                X_fold_val = X_train.iloc[val_idx]
                                y_fold_val = y_train.iloc[val_idx]

                                model.fit(X_fold_train, y_fold_train)
                                y_pred = model.predict(X_fold_val)
                                score = mean_squared_error(y_fold_val, y_pred)
                                scores.append(score)

                            avg_score = np.mean(scores)
                            if avg_score < best_lgbm_score:
                                best_lgbm_score = avg_score
                                best_lgbm_params = params.copy()
                                best_lgbm_model = ModelWrapper(model, "lgbm", params)

            best_models["lgbm"] = best_lgbm_model
            best_scores["lgbm"] = best_lgbm_score
            best_params["lgbm"] = best_lgbm_params

            # XGBoost
            logger.info("Optimizando XGBoost...")
            best_xgb_score = float("inf")
            best_xgb_params = None
            best_xgb_model = None

            for n_estimators in xgb_params["n_estimators"]:
                for learning_rate in xgb_params["learning_rate"]:
                    for max_depth in xgb_params["max_depth"]:
                        params = {
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "verbosity": 0,  # Usar verbosity en lugar de verbose
                        }
                        model = XGBRegressor(**params)

                        # Validación cruzada
                        scores = []
                        for train_idx, val_idx in tscv.split(X_train):
                            X_fold_train = X_train.iloc[train_idx]
                            y_fold_train = y_train.iloc[train_idx]
                            X_fold_val = X_train.iloc[val_idx]
                            y_fold_val = y_train.iloc[val_idx]

                            model.fit(X_fold_train, y_fold_train)
                            y_pred = model.predict(X_fold_val)
                            score = mean_squared_error(y_fold_val, y_pred)
                            scores.append(score)

                        avg_score = np.mean(scores)
                        if avg_score < best_xgb_score:
                            best_xgb_score = avg_score
                            best_xgb_params = params.copy()
                            best_xgb_model = ModelWrapper(model, "xgboost", params)

            best_models["xgboost"] = best_xgb_model
            best_scores["xgboost"] = best_xgb_score
            best_params["xgboost"] = best_xgb_params

            # CatBoost
            logger.info("Optimizando CatBoost...")
            best_cat_score = float("inf")
            best_cat_params = None
            best_cat_model = None

            for iterations in cat_params["iterations"]:
                for learning_rate in cat_params["learning_rate"]:
                    for depth in cat_params["depth"]:
                        params = {
                            "iterations": iterations,
                            "learning_rate": learning_rate,
                            "depth": depth,
                            "verbose": False,
                        }
                        model = CatBoostRegressor(**params)

                        # Validación cruzada
                        scores = []
                        for train_idx, val_idx in tscv.split(X_train):
                            X_fold_train = X_train.iloc[train_idx]
                            y_fold_train = y_train.iloc[train_idx]
                            X_fold_val = X_train.iloc[val_idx]
                            y_fold_val = y_train.iloc[val_idx]

                            model.fit(X_fold_train, y_fold_train)
                            y_pred = model.predict(X_fold_val)
                            score = mean_squared_error(y_fold_val, y_pred)
                            scores.append(score)

                        avg_score = np.mean(scores)
                        if avg_score < best_cat_score:
                            best_cat_score = avg_score
                            best_cat_params = params.copy()
                            best_cat_model = ModelWrapper(model, "catboost", params)

            best_models["catboost"] = best_cat_model
            best_scores["catboost"] = best_cat_score
            best_params["catboost"] = best_cat_params

            # Guardar modelos
            os.makedirs("models", exist_ok=True)
            for name, model in best_models.items():
                model.save(f"models/{name}_model.pkl")

            # Crear diccionario de resultados
            results = {
                "base_models": {
                    name: {
                        "type": model.model_type,
                        "params": model.params,
                        "path": f"models/{name}_model.pkl",
                    }
                    for name, model in best_models.items()
                },
                "best_scores": best_scores,
                "best_params": best_params,
            }

            return results

        except Exception as e:
            logger.error(f"Error en optimización: {str(e)}")
            raise


def sharpe_ratio_metric(y_true, y_pred, *args, **kwargs):
    """
    Métrica personalizada de Sharpe Ratio para FLAML

    Args:
        y_true: Valores reales
        y_pred: Predicciones
        *args: Argumentos posicionales adicionales (ignorados)
        **kwargs: Argumentos con nombre adicionales (ignorados)

    Returns:
        float: Valor del sharpe ratio negado (para minimización)
    """
    try:
        # Asegurarse de que los inputs son arrays numpy
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # Verificar y corregir dimensiones
        if y_pred.ndim > 1:
            # Si y_pred es 2D, tomar la primera columna
            if y_pred.shape[1] > 1:
                logger.warning(
                    f"y_pred tiene forma {y_pred.shape}, usando primera columna"
                )
            y_pred = y_pred[:, 0]

        if y_true.ndim > 1:
            # Si y_true es 2D, tomar la primera columna
            if y_true.shape[1] > 1:
                logger.warning(
                    f"y_true tiene forma {y_true.shape}, usando primera columna"
                )
            y_true = y_true[:, 0]

        # Asegurar que tienen la misma forma
        if y_true.shape != y_pred.shape:
            logger.warning(
                f"Dimensiones incompatibles: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
            # Intentar ajustar dimensiones
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        # Calcular diferencias
        diff = y_pred - y_true

        # Calcular estadísticas
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        # Evitar división por cero
        if std_diff == 0:
            logger.warning("Desviación estándar es cero, retornando 0.0")
            return 0.0

        # Calcular Sharpe ratio
        sharpe = mean_diff / std_diff

        # Verificar valores inválidos
        if not np.isfinite(sharpe):
            logger.warning(f"Sharpe ratio no es finito: {sharpe}, retornando 0.0")
            return 0.0

        # Retornar negativo para minimización
        return float(-sharpe)

    except Exception as e:
        logger.error(f"Error calculando Sharpe ratio: {str(e)}")
        return 0.0  # Valor por defecto en caso de error
