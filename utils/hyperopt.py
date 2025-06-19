#!/usr/bin/env python3
"""
🎯 OPTIMIZACIÓN DE HIPERPARÁMETROS PARA HYPERION3
Optimización centralizada usando Optuna para todos los modelos
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Imports para optimización
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False

# Imports sklearn
try:
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    )
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Imports opcionales
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

class HyperparameterOptimizer:
    """Optimizador de hiperparámetros para todos los modelos incluyendo Hyperion3"""
    
    def __init__(self, console=None, gpu_config: Optional[Dict] = None):
        self.console = console
        self.optimization_results = {}
        self.gpu_config = gpu_config or {
            'xgboost_params': {'tree_method': 'hist', 'n_jobs': 4},
            'lightgbm_params': {'device': 'cpu', 'num_threads': 4},
            'catboost_params': {'task_type': 'CPU', 'thread_count': 4}
        }
    
    def optimize_all_models(self, X_train, y_train, X_val, y_val, 
                           n_trials: int = 20) -> Dict[str, Dict]:
        """
        Optimizar hiperparámetros para todos los modelos disponibles
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            n_trials: Número de trials por modelo
            
        Returns:
            Dict con resultados de optimización
        """
        
        if not OPTUNA_AVAILABLE:
            if self.console:
                self.console.print("❌ Optuna no disponible - usando parámetros por defecto")
            return self._get_default_parameters()
        
        if self.console:
            self.console.print("🎯 Iniciando optimización completa de hiperparámetros...")
        
        optimization_results = {}
        
        # Lista de modelos a optimizar
        models_to_optimize = [
            ('sklearn', self.optimize_sklearn_models),
            ('xgboost', self.optimize_xgboost),
            ('lightgbm', self.optimize_lightgbm),
            ('catboost', self.optimize_catboost),
            ('pytorch', self.optimize_pytorch_models)
        ]
        
        # Optimizar cada tipo de modelo
        for model_type, optimizer_func in models_to_optimize:
            try:
                if self.console:
                    self.console.print(f"🔍 Optimizando modelos {model_type}...")
                
                results = optimizer_func(X_train, y_train, X_val, y_val, n_trials)
                if results:
                    optimization_results.update(results)
                    
            except Exception as e:
                if self.console:
                    self.console.print(f"❌ Error optimizando {model_type}: {e}")
                continue
        
        # Guardar resultados
        self.optimization_results = optimization_results
        
        if self.console:
            self.console.print(f"🎯 Optimización completada: {len(optimization_results)} modelos optimizados")
        
        # Mostrar resumen
        if optimization_results and self.console:
            self._show_optimization_summary(optimization_results)
        
        return optimization_results
    
    def optimize_sklearn_models(self, X_train, y_train, X_val, y_val, 
                               n_trials: int = 15) -> Dict[str, Dict]:
        """Optimizar hiperparámetros de modelos sklearn con Optuna"""
        
        if not (OPTUNA_AVAILABLE and SKLEARN_AVAILABLE):
            return {}
        
        results = {}
        
        models_to_optimize = {
            "RandomForest": (RandomForestRegressor, {
                'n_estimators': ('int', 50, 300), 
                'max_depth': ('int', 5, 30),
                'min_samples_split': ('int', 2, 10), 
                'min_samples_leaf': ('int', 1, 5),
                'max_features': ('categorical', ['sqrt', 'log2', None])
            }),
            "GradientBoosting": (GradientBoostingRegressor, {
                'n_estimators': ('int', 50, 200), 
                'learning_rate': ('float', 0.01, 0.2),
                'max_depth': ('int', 3, 8),
                'subsample': ('float', 0.6, 1.0)
            }),
            "SVR": (SVR, {
                'C': ('float', 0.01, 10.0),
                'kernel': ('categorical', ['linear', 'rbf']),
                'epsilon': ('float', 0.01, 0.2),
                'gamma': ('categorical', ['scale', 'auto'])
            }),
            "Ridge": (Ridge, {
                'alpha': ('float', 0.1, 10.0)
            }),
            "Lasso": (Lasso, {
                'alpha': ('float', 0.01, 1.0)
            }),
            "ElasticNet": (ElasticNet, {
                'alpha': ('float', 0.1, 10.0), 
                'l1_ratio': ('float', 0.1, 0.9)
            }),
            "ExtraTreesRegressor": (ExtraTreesRegressor, {
                'n_estimators': ('int', 50, 300), 
                'max_depth': ('int', 5, 30),
                'min_samples_split': ('int', 2, 10)
            }),
            "KNeighborsRegressor": (KNeighborsRegressor, {
                'n_neighbors': ('int', 3, 20), 
                'weights': ('categorical', ['uniform', 'distance']),
                'p': ('int', 1, 2)  # 1=manhattan, 2=euclidean
            })
        }
        
        for model_name, (model_class, param_space) in models_to_optimize.items():
            try:
                if self.console:
                    self.console.print(f"   Optimizando {model_name}...")
                
                def objective(trial):
                    params = {}
                    for param_name, param_config in param_space.items():
                        if param_config[0] == 'int':
                            params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                        elif param_config[0] == 'float':
                            params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
                        elif param_config[0] == 'categorical':
                            params[param_name] = trial.suggest_categorical(param_name, param_config[1])
                    
                    # Añadir parámetros fijos
                    if model_name == "RandomForest":
                        params.update({'random_state': 42, 'n_jobs': 4})
                    elif model_name == "GradientBoosting":
                        params.update({'random_state': 42})
                    elif model_name == "ExtraTreesRegressor":
                        params.update({'random_state': 42, 'n_jobs': 4})
                    
                    try:
                        model = model_class(**params)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        return r2_score(y_val, y_pred)
                    except Exception as e:
                        if self.console:
                            self.console.print(f"❌ Error en {model_name}: {str(e)}")
                        return -999  # Penalizar parámetros inválidos
                
                study = optuna.create_study(
                    direction='maximize',
                    sampler=TPESampler(seed=42)
                )
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                
                results[model_name] = {
                    'params': study.best_params,
                    'score': study.best_value
                }
                
            except Exception as e:
                if self.console:
                    self.console.print(f"❌ Error optimizando {model_name}: {e}")
                continue
        
        return results
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, 
                        n_trials: int = 30) -> Dict[str, Dict]:
        """Optimizar hiperparámetros de XGBoost"""
        
        if not (OPTUNA_AVAILABLE and XGBOOST_AVAILABLE):
            return {}
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': 42
            }
            
            # Añadir parámetros de GPU config
            params.update(self.gpu_config.get('xgboost_params', {}))
            
            try:
                # Para XGBoost 3.x, usar early_stopping_rounds en el constructor
                params['early_stopping_rounds'] = 10
                model = xgb.XGBRegressor(**params)
                
                # En XGBoost 3.x, fit() es más simple
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                return r2_score(y_val, y_pred)
            except Exception as e:
                if self.console:
                    self.console.print(f"❌ Error en XGBoost trial: {str(e)}")
                return -999
        
        try:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            return {
                'XGBoost': {
                    'params': study.best_params,
                    'score': study.best_value
                }
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error optimizando XGBoost: {e}")
            return {}
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, 
                         n_trials: int = 30) -> Dict[str, Dict]:
        """Optimizar hiperparámetros de LightGBM"""
        
        if not (OPTUNA_AVAILABLE and LIGHTGBM_AVAILABLE):
            return {}
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': 42
            }
            
            # Añadir parámetros de GPU config
            params.update(self.gpu_config.get('lightgbm_params', {}))
            
            try:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(0)])
                y_pred = model.predict(X_val)
                return r2_score(y_val, y_pred)
            except Exception as e:
                if self.console:
                    self.console.print(f"❌ Error en LightGBM trial: {str(e)}")
                return -999
        
        try:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            return {
                'LightGBM': {
                    'params': study.best_params,
                    'score': study.best_value
                }
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error optimizando LightGBM: {e}")
            return {}
    
    def optimize_catboost(self, X_train, y_train, X_val, y_val, 
                         n_trials: int = 30) -> Dict[str, Dict]:
        """Optimizar hiperparámetros de CatBoost"""
        
        if not (OPTUNA_AVAILABLE and CATBOOST_AVAILABLE):
            return {}
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
                'random_seed': 42,
                'verbose': False
            }
            
            # Añadir parámetros de GPU config
            params.update(self.gpu_config.get('catboost_params', {}))
            
            try:
                model = CatBoostRegressor(**params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                         early_stopping_rounds=10, verbose=False)
                y_pred = model.predict(X_val)
                return r2_score(y_val, y_pred)
            except Exception:
                return -999
        
        try:
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            return {
                'CatBoost': {
                    'params': study.best_params,
                    'score': study.best_value
                }
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error optimizando CatBoost: {e}")
            return {}
    
    def optimize_pytorch_models(self, X_train, y_train, X_val, y_val, 
                               n_trials: int = 15) -> Dict[str, Dict]:
        """Optimizar hiperparámetros de todos los modelos PyTorch"""
        
        if not OPTUNA_AVAILABLE:
            return {}
        
        try:
            import torch
            import torch.nn as nn
            PYTORCH_AVAILABLE = True
        except ImportError:
            return {}
        
        results = {}
        input_size = X_train.shape[1]
        
        # Crear función objetivo genérica para PyTorch
        def create_objective(model_name):
            def objective(trial):
                # Parámetros comunes
                lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
                epochs = trial.suggest_int('epochs', 20, 50)
                
                # Parámetros específicos del modelo
                if model_name == "SimpleMLP":
                    hidden_size = trial.suggest_int('hidden_size', 64, 256)
                    model_params = {'hidden_size': hidden_size}
                elif model_name == "DeepMLP":
                    hidden_size = trial.suggest_int('hidden_size', 128, 512)
                    num_layers = trial.suggest_int('num_layers', 2, 5)
                    dropout = trial.suggest_float('dropout', 0.1, 0.5)
                    model_params = {'hidden_size': hidden_size, 'num_layers': num_layers, 'dropout': dropout}
                elif model_name == "LSTM":
                    hidden_size = trial.suggest_int('hidden_size', 64, 256)
                    num_layers = trial.suggest_int('num_layers', 1, 3)
                    model_params = {'hidden_size': hidden_size, 'num_layers': num_layers}
                else:
                    model_params = {}
                
                try:
                    # Importar el trainer de modelos PyTorch desde el módulo local
                    from .models import train_pytorch_model, create_pytorch_models
                    
                    # Crear modelo
                    models = create_pytorch_models(input_size)
                    if model_name not in models:
                        return -999
                    
                    model = models[model_name]
                    
                    # Entrenar con parámetros optimizados
                    train_params = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'weight_decay': weight_decay
                    }
                    train_params.update(model_params)
                    
                    result = train_pytorch_model(
                        model, X_train, y_train, X_val, y_val,
                        model_name, epochs=epochs, optimized_params=train_params
                    )
                    
                    if 'error' in result:
                        return -999
                    
                    return result.get('val_r2', -999)
                    
                except Exception:
                    return -999
            
            return objective
        
        # Optimizar cada modelo PyTorch
        pytorch_models = ["SimpleMLP", "DeepMLP", "LSTM"]
        
        for model_name in pytorch_models:
            try:
                if self.console:
                    self.console.print(f"   Optimizando {model_name}...")
                
                study = optuna.create_study(
                    direction='maximize',
                    sampler=TPESampler(seed=42)
                )
                study.optimize(create_objective(model_name), n_trials=n_trials, show_progress_bar=False)
                
                results[model_name] = {
                    'params': study.best_params,
                    'score': study.best_value
                }
                
            except Exception as e:
                if self.console:
                    self.console.print(f"❌ Error optimizando {model_name}: {e}")
                continue
        
        return results
    
    def _get_default_parameters(self) -> Dict[str, Dict]:
        """Obtener parámetros por defecto cuando Optuna no está disponible"""
        
        return {
            'RandomForest': {
                'params': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': 4
                },
                'score': 0.0
            },
            'XGBoost': {
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'score': 0.0
            },
            'LightGBM': {
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbose': -1
                },
                'score': 0.0
            },
            'SVR': {
                'params': {
                    'kernel': 'linear',
                    'C': 0.1,
                    'epsilon': 0.1
                },
                'score': 0.0
            }
        }
    
    def _show_optimization_summary(self, results: Dict[str, Dict]) -> None:
        """Mostrar resumen de la optimización"""
        
        if not self.console:
            return
        
        from rich.table import Table
        
        table = Table(title="🎯 Resumen de Optimización de Hiperparámetros")
        table.add_column("Modelo", style="cyan", no_wrap=True)
        table.add_column("Mejor Score", style="magenta", justify="right")
        table.add_column("Parámetros Clave", style="green")
        
        for model_name, result in results.items():
            score = result.get('score', 0)
            params = result.get('params', {})
            
            # Mostrar solo algunos parámetros clave
            key_params = []
            for key, value in list(params.items())[:3]:
                if isinstance(value, float):
                    key_params.append(f"{key}={value:.3f}")
                else:
                    key_params.append(f"{key}={value}")
            
            params_str = ", ".join(key_params)
            if len(params) > 3:
                params_str += "..."
            
            table.add_row(model_name, f"{score:.4f}", params_str)
        
        self.console.print(table)

def optimize_hyperparameters(models: Dict, X_train, y_train, X_val, y_val, 
                           console=None, n_trials: int = 20) -> Dict:
    """
    Función wrapper para optimizar hiperparámetros de modelos
    
    Args:
        models: Diccionario de modelos a optimizar
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        console: Rich console para mostrar progreso (opcional)
        n_trials: Número de trials para optimización
        
    Returns:
        Diccionario con mejores parámetros para cada modelo
    """
    optimizer = HyperparameterOptimizer(console=console)
    
    return optimizer.optimize_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        models=models,
        n_trials=n_trials
    )

def quick_optimize_hyperparameters(model_type: str, X_train, y_train, X_val, y_val,
                                 console=None, n_trials: int = 10) -> Dict:
    """
    Optimización rápida para un tipo específico de modelo
    
    Args:
        model_type: Tipo de modelo ('xgboost', 'lightgbm', 'catboost', 'sklearn')
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        console: Rich console para mostrar progreso (opcional)
        n_trials: Número de trials para optimización
        
    Returns:
        Diccionario con mejores parámetros
    """
    optimizer = HyperparameterOptimizer(console=console)
    
    if model_type == 'xgboost':
        return optimizer.optimize_xgboost(X_train, y_train, X_val, y_val, n_trials)
    elif model_type == 'lightgbm':
        return optimizer.optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials)
    elif model_type == 'catboost':
        return optimizer.optimize_catboost(X_train, y_train, X_val, y_val, n_trials)
    elif model_type == 'sklearn':
        return optimizer.optimize_sklearn_models(X_train, y_train, X_val, y_val, n_trials)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
