#!/usr/bin/env python3
"""
🚀 ENTRENADOR PRINCIPAL PARA HYPERION3
Clase principal para entrenar el ensemble ultra completo
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Rich imports para UI
try:
    from rich.console import Console
    from utils.safe_progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ML imports
try:
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
        AdaBoostRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
    )
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, 
        TheilSenRegressor, RANSACRegressor, BayesianRidge
    )
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR, NuSVR
    from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler, RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Imports de nuestros módulos
try:
    from .env_config import (
        initialize_environment, get_optimal_sklearn_params, 
        configure_multiprocessing_for_mac, get_financial_baseline_params
    )
    from .features import create_clean_features, verify_no_leakage, select_top_features
    from .models import create_pytorch_models, train_pytorch_model
    from .hyperopt import HyperparameterOptimizer
except ImportError as e:
    print(f"❌ Error importando módulos locales: {e}")
    # Fallback imports si los módulos no están disponibles
    sys.exit(1)

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

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

class UltraCompleteEnsembleTrainer:
    """Entrenador de ensemble ultra completo con todas las optimizaciones"""
    
    def __init__(self, optimize_hyperparams: bool = True):
        """
        Inicializar el entrenador
        
        Args:
            optimize_hyperparams: Si optimizar hiperparámetros con Optuna
        """
        self.console = Console() if RICH_AVAILABLE else None
        self.models = {}
        self.ensembles = {}
        self.results = {}
        self.optimize_hyperparams = optimize_hyperparams
        self.best_hyperparams = {}
        
        # Inicializar entorno
        if self.console:
            self.console.print("🔍 Inicializando entorno Hyperion3...")
        
        try:
            dependency_status, gpu_config, validation_issues = initialize_environment()
            self.dependency_status = dependency_status
            self.gpu_config = gpu_config
            self.validation_issues = validation_issues
        except Exception as e:
            print(f"❌ Error inicializando entorno: {e}")
            self.gpu_config = {'pytorch_device': 'cpu'}
            self.validation_issues = [f"Error de inicialización: {e}"]
        
        # Configuraciones adicionales
        self.sklearn_params = get_optimal_sklearn_params()
        self.mac_config = configure_multiprocessing_for_mac()
        
        # Inicializar optimizador de hiperparámetros
        if optimize_hyperparams:
            self.hyperopt = HyperparameterOptimizer(self.console, self.gpu_config)
        else:
            self.hyperopt = None
        
        if self.console:
            device_str = self.gpu_config.get('pytorch_device', 'CPU')
            cores_str = self.mac_config['total_cores']
            self.console.print(f"🔧 Configuración: {cores_str} cores, GPU: {device_str}")
    
    def train_ultra_complete_ensemble(self, symbol: str = 'SOL/USDT') -> List[Dict]:
        """
        Entrenar ensemble ultra completo
        
        Args:
            symbol: Símbolo de trading a analizar
            
        Returns:
            Lista con resultados de todos los modelos
        """
        
        # Panel inicial
        if self.console:
            panel = Panel.fit(
                f"🚀 HYPERION3 - ENSEMBLE ULTRA COMPLETO\n"
                f"📊 Símbolo: {symbol}\n"
                f"🎯 Todos los modelos: sklearn, XGB, LGB, CatBoost, PyTorch, etc.\n"
                f"✅ Sin data leakage garantizado",
                title="Ultra Complete Ensemble Training",
                border_style="green"
            )
            self.console.print(panel)
        
        try:
            # 1. Cargar datos
            if self.console:
                self.console.print("\n📊 Cargando datos...")
            
            filename = f"./data/{symbol.replace('/', '_')}_20250613.csv"
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Archivo de datos no encontrado: {filename}")
            
            data = pd.read_csv(filename)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            if self.console:
                self.console.print(f"📊 Total filas: {len(data):,}")
            
            # 2. Crear features limpias
            feature_df = create_clean_features(data, self.console)
            
            if self.console:
                self.console.print(f"📊 Filas después de features: {len(feature_df):,}")
                self.console.print(f"🎯 Total features: {len(feature_df.columns)-2}")
            
            # 3. Verificar data leakage
            if not verify_no_leakage(feature_df, self.console):
                raise ValueError("❌ Data leakage detectado - deteniendo entrenamiento")
            
            # 4. Split temporal
            split_idx = int(0.7 * len(feature_df))
            val_split_idx = int(0.85 * len(feature_df))
            
            train_data = feature_df.iloc[:split_idx]
            val_data = feature_df.iloc[split_idx:val_split_idx]
            test_data = feature_df.iloc[val_split_idx:]
            
            feature_cols = [col for col in feature_df.columns if col not in ['target', 'timestamp']]
            
            # Separar X y y antes de normalizar
            X_train_raw = train_data[feature_cols].fillna(0)
            y_train = train_data['target']
            X_val_raw = val_data[feature_cols].fillna(0)
            y_val = val_data['target']
            X_test_raw = test_data[feature_cols].fillna(0)
            y_test = test_data['target']
            
            # 5. Normalización robusta SIN DATA LEAKAGE
            if self.console:
                self.console.print("🔧 Aplicando normalización robusta...")
            
            scaler = RobustScaler()
            
            # Verificar datos antes de normalizar
            for col in X_train_raw.columns:
                if X_train_raw[col].isna().all():
                    if self.console:
                        self.console.print(f"⚠️ Columna {col} está completamente vacía")
                    continue
            
            try:
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train_raw), 
                    columns=X_train_raw.columns,
                    index=X_train_raw.index
                )
                X_val = pd.DataFrame(
                    scaler.transform(X_val_raw), 
                    columns=X_val_raw.columns,
                    index=X_val_raw.index
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test_raw), 
                    columns=X_test_raw.columns,
                    index=X_test_raw.index
                )
            except Exception as e:
                if self.console:
                    self.console.print(f"❌ Error en normalización: {e}")
                # Usar datos sin normalizar como fallback
                X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw
            
            if self.console:
                self.console.print(f"\n📅 SPLITS:")
                self.console.print(f"  Train: {len(train_data):,} filas")
                self.console.print(f"  Val:   {len(val_data):,} filas")
                self.console.print(f"  Test:  {len(test_data):,} filas")
            
            # 6. Feature selection
            if self.console:
                self.console.print("\n🎯 Aplicando feature selection...")
            
            try:
                # Crear DataFrame temporal para feature selection
                temp_df = X_train.copy()
                temp_df['target'] = y_train
                
                selected_features = select_top_features(temp_df, n_features=30, method='correlation')
                
                if selected_features:
                    X_train = X_train[selected_features]
                    X_val = X_val[selected_features]
                    X_test = X_test[selected_features]
                    
                    if self.console:
                        self.console.print(f"✅ Seleccionadas {len(selected_features)} features principales")
                else:
                    if self.console:
                        self.console.print("⚠️ Feature selection falló, usando todas las features")
                        
            except Exception as e:
                if self.console:
                    self.console.print(f"⚠️ Error en feature selection: {e}")
            
            # 7. Entrenar modelos
            if self.console:
                self.console.print("\n🤖 Entrenando modelos...")
            
            # Optimizar hiperparámetros si está habilitado
            if self.optimize_hyperparams and self.hyperopt:
                if self.console:
                    self.console.print("🎯 Optimizando hiperparámetros...")
                
                optimization_results = self.hyperopt.optimize_all_models(
                    X_train, y_train, X_val, y_val, n_trials=15
                )
                self.best_hyperparams = optimization_results
            else:
                self.best_hyperparams = get_financial_baseline_params()
            
            # Entrenar modelos base
            base_models = self.train_all_models(X_train, y_train, X_val, y_val)
            
            # 8. Entrenar modelos FLAML AutoML
            flaml_models = self.train_flaml_models(X_train, y_train, X_val, y_val, time_budget=120)
            
            # Combinar modelos
            if flaml_models:
                base_models.update(flaml_models)
            
            # 9. Crear ensembles
            ensembles = self.create_ensembles(base_models, X_train, y_train, X_val, y_val)
            
            # 10. Evaluación final en test set
            if self.console:
                self.console.print(f"\n📊 EVALUACIÓN FINAL EN TEST SET:")
                self.console.print("=" * 60)
            
            all_results = []
            
            # Evaluar modelos base y ensembles
            all_models = {**base_models, **ensembles}
            
            if self.console and RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    console=self.console
                ) as progress:
                    
                    task = progress.add_task("[green]Evaluando modelos...", total=len(all_models))
                    
                    for model_name, model_info in all_models.items():
                        try:
                            result = self.realistic_backtest(
                                model_info, X_test, y_test, model_name, 
                                model_info.get('type', 'sklearn')
                            )
                            result['model_name'] = model_name
                            all_results.append(result)
                        except Exception as e:
                            all_results.append({
                                'model_name': model_name,
                                'error': str(e)
                            })
                        
                        progress.advance(task)
            else:
                # Versión sin progress bar
                for model_name, model_info in all_models.items():
                    try:
                        result = self.realistic_backtest(
                            model_info, X_test, y_test, model_name, 
                            model_info.get('type', 'sklearn')
                        )
                        result['model_name'] = model_name
                        all_results.append(result)
                    except Exception as e:
                        all_results.append({
                            'model_name': model_name,
                            'error': str(e)
                        })
            
            # 11. Mostrar resultados
            self.display_results(all_results)
            
            # 12. Encontrar mejor modelo
            valid_results = [r for r in all_results if 'error' not in r]
            if valid_results:
                best_result = max(valid_results, key=lambda x: x.get('trading_metrics', {}).get('win_rate', 0))
                if self.console:
                    self.console.print(f"\n🏆 MEJOR MODELO: {best_result['model_name']}")
                    self.console.print(f"   Win Rate: {best_result['trading_metrics']['win_rate']:.2%}")
                    self.console.print(f"   R²: {best_result['r2']:.4f}")
            
            # 13. Guardar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"./results/ultra_complete_ensemble_{symbol.replace('/', '_').lower()}_{timestamp}.json"
            
            # Crear directorio si no existe
            os.makedirs('./results', exist_ok=True)
            
            import json
            with open(results_filename, 'w') as f:
                # Convertir numpy types a tipos serializables
                serializable_results = []
                for result in all_results:
                    clean_result = {}
                    for k, v in result.items():
                        if isinstance(v, (np.integer, np.floating)):
                            clean_result[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            clean_result[k] = v.tolist()
                        elif hasattr(v, 'item'):  # numpy scalar
                            clean_result[k] = v.item()
                        else:
                            clean_result[k] = v
                    serializable_results.append(clean_result)
                
                json.dump(serializable_results, f, indent=2, default=str)
            
            if self.console:
                self.console.print(f"\n✅ RESULTADOS GUARDADOS: {results_filename}")
            
            return all_results
            
        except Exception as e:
            if self.console:
                self.console.print(f"❌ ERROR CRÍTICO: {e}")
            else:
                print(f"❌ ERROR CRÍTICO: {e}")
            
            # Intentar modelo de emergencia
            try:
                from .env_config import emergency_fallback_models
                return emergency_fallback_models(X_train, y_train, X_val, y_val)
            except:
                return [{'error': f'Fallo completo del sistema: {e}'}]
    
    def train_all_models(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Entrenar todos los modelos disponibles"""
        
        models = {}
        opt_params = self.best_hyperparams if hasattr(self, 'best_hyperparams') else {}
        
        if self.console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                
                # Lista de modelos a entrenar
                model_list = [
                    ('RandomForest', lambda: self._train_random_forest(X_train, y_train, X_val, y_val, opt_params)),
                    ('GradientBoosting', lambda: self._train_gradient_boosting(X_train, y_train, X_val, y_val, opt_params)),
                    ('SVR', lambda: self._train_svr(X_train, y_train, X_val, y_val, opt_params)),
                    ('Ridge', lambda: self._train_ridge(X_train, y_train, X_val, y_val, opt_params)),
                    ('XGBoost', lambda: self._train_xgboost(X_train, y_train, X_val, y_val, opt_params)),
                    ('LightGBM', lambda: self._train_lightgbm(X_train, y_train, X_val, y_val, opt_params)),
                    ('CatBoost', lambda: self._train_catboost(X_train, y_train, X_val, y_val, opt_params)),
                    ('PyTorch', lambda: self._train_pytorch_models(X_train, y_train, X_val, y_val, opt_params))
                ]
                
                task = progress.add_task("[cyan]Entrenando modelos...", total=len(model_list))
                
                for model_name, train_func in model_list:
                    progress.update(task, description=f"[cyan]Entrenando {model_name}...")
                    
                    try:
                        result = train_func()
                        if result:
                            if isinstance(result, dict) and 'model' in result:
                                models[model_name] = result
                            else:
                                models.update(result)
                    except Exception as e:
                        if self.console:
                            self.console.print(f"❌ Error entrenando {model_name}: {e}")
                    
                    progress.advance(task)
        else:
            # Versión sin progress bar
            model_trainers = {
                'RandomForest': lambda: self._train_random_forest(X_train, y_train, X_val, y_val, opt_params),
                'GradientBoosting': lambda: self._train_gradient_boosting(X_train, y_train, X_val, y_val, opt_params),
                'SVR': lambda: self._train_svr(X_train, y_train, X_val, y_val, opt_params),
                'Ridge': lambda: self._train_ridge(X_train, y_train, X_val, y_val, opt_params),
                'XGBoost': lambda: self._train_xgboost(X_train, y_train, X_val, y_val, opt_params),
                'LightGBM': lambda: self._train_lightgbm(X_train, y_train, X_val, y_val, opt_params),
                'CatBoost': lambda: self._train_catboost(X_train, y_train, X_val, y_val, opt_params),
                'PyTorch': lambda: self._train_pytorch_models(X_train, y_train, X_val, y_val, opt_params)
            }
            
            for model_name, train_func in model_trainers.items():
                try:
                    result = train_func()
                    if result:
                        if isinstance(result, dict) and 'model' in result:
                            models[model_name] = result
                        else:
                            models.update(result)
                except Exception as e:
                    print(f"❌ Error entrenando {model_name}: {e}")
        
        if self.console:
            self.console.print(f"✅ Entrenamiento completado: {len(models)} modelos exitosos")
        
        return models
    
    def _train_random_forest(self, X_train, y_train, X_val, y_val, opt_params) -> Dict:
        """Entrenar Random Forest"""
        if not SKLEARN_AVAILABLE:
            return {}
        
        params = opt_params.get('RandomForest', {}).get('params', {})
        params.setdefault('n_estimators', 100)
        params.setdefault('random_state', 42)
        params.setdefault('n_jobs', 4)
        
        try:
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            return {
                'model': model,
                'r2': r2,
                'predictions': y_pred,
                'type': 'sklearn'
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en RandomForest: {e}")
            return {}
    
    def _train_gradient_boosting(self, X_train, y_train, X_val, y_val, opt_params) -> Dict:
        """Entrenar Gradient Boosting"""
        if not SKLEARN_AVAILABLE:
            return {}
        
        params = opt_params.get('GradientBoosting', {}).get('params', {})
        params.setdefault('n_estimators', 100)
        params.setdefault('random_state', 42)
        
        try:
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            return {
                'model': model,
                'r2': r2,
                'predictions': y_pred,
                'type': 'sklearn'
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en GradientBoosting: {e}")
            return {}
    
    def _train_svr(self, X_train, y_train, X_val, y_val, opt_params) -> Dict:
        """Entrenar SVR"""
        if not SKLEARN_AVAILABLE:
            return {}
        
        params = opt_params.get('SVR', {}).get('params', {
            'kernel': 'linear', 'C': 0.1, 'epsilon': 0.1
        })
        
        try:
            model = SVR(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            return {
                'model': model,
                'r2': r2,
                'predictions': y_pred,
                'type': 'sklearn'
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en SVR: {e}")
            return {}
    
    def _train_ridge(self, X_train, y_train, X_val, y_val, opt_params) -> Dict:
        """Entrenar Ridge"""
        if not SKLEARN_AVAILABLE:
            return {}
        
        params = opt_params.get('Ridge', {}).get('params', {'alpha': 1.0})
        
        try:
            model = Ridge(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            return {
                'model': model,
                'r2': r2,
                'predictions': y_pred,
                'type': 'sklearn'
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en Ridge: {e}")
            return {}
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, opt_params) -> Dict:
        """Entrenar XGBoost"""
        if not XGBOOST_AVAILABLE:
            return {}
        
        params = opt_params.get('XGBoost', {}).get('params', {})
        params.setdefault('n_estimators', 100)
        params.setdefault('random_state', 42)
        params.update(self.gpu_config.get('xgboost_params', {}))
        
        try:
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     verbose=False, early_stopping_rounds=10)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            return {
                'model': model,
                'r2': r2,
                'predictions': y_pred,
                'type': 'sklearn'
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en XGBoost: {e}")
            return {}
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, opt_params) -> Dict:
        """Entrenar LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            return {}
        
        params = opt_params.get('LightGBM', {}).get('params', {})
        params.setdefault('n_estimators', 100)
        params.setdefault('random_state', 42)
        params.update(self.gpu_config.get('lightgbm_params', {}))
        
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            return {
                'model': model,
                'r2': r2,
                'predictions': y_pred,
                'type': 'sklearn'
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en LightGBM: {e}")
            return {}
    
    def _train_catboost(self, X_train, y_train, X_val, y_val, opt_params) -> Dict:
        """Entrenar CatBoost"""
        if not CATBOOST_AVAILABLE:
            return {}
        
        params = opt_params.get('CatBoost', {}).get('params', {})
        params.setdefault('iterations', 100)
        params.setdefault('random_seed', 42)
        params.setdefault('verbose', False)
        params.update(self.gpu_config.get('catboost_params', {}))
        
        try:
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                     early_stopping_rounds=10, verbose=False)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            return {
                'model': model,
                'r2': r2,
                'predictions': y_pred,
                'type': 'sklearn'
            }
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en CatBoost: {e}")
            return {}
    
    def _train_pytorch_models(self, X_train, y_train, X_val, y_val, opt_params) -> Dict:
        """Entrenar modelos PyTorch"""
        try:
            pytorch_models = create_pytorch_models(X_train.shape[1])
            if not pytorch_models:
                return {}
            
            results = {}
            device = self.gpu_config.get('pytorch_device')
            
            for model_name, model in pytorch_models.items():
                try:
                    optimized_params = opt_params.get(model_name, {}).get('params', {})
                    
                    result = train_pytorch_model(
                        model, X_train, y_train, X_val, y_val,
                        model_name, epochs=30, optimized_params=optimized_params,
                        device=device
                    )
                    
                    if 'error' not in result:
                        results[model_name] = result
                        
                except Exception as e:
                    if self.console:
                        self.console.print(f"❌ Error en {model_name}: {e}")
            
            return results
            
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en modelos PyTorch: {e}")
            return {}
    
    def train_flaml_models(self, X_train, y_train, X_val, y_val, time_budget: int = 300) -> Dict:
        """Entrenar modelos usando FLAML AutoML"""
        if not FLAML_AVAILABLE:
            return {}
        
        if self.console:
            self.console.print(f"🔥 Entrenando modelos con FLAML AutoML (budget: {time_budget}s)...")
        
        try:
            automl = AutoML()
            automl.fit(
                X_train, y_train,
                task='regression',
                time_budget=time_budget,
                eval_method='holdout',
                split_ratio=0.2,
                verbose=0,
                ensemble=True
            )
            
            # Evaluar en validation set
            y_pred = automl.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            
            return {
                'FLAML_AutoML': {
                    'model': automl,
                    'r2': r2,
                    'predictions': y_pred,
                    'type': 'flaml'
                }
            }
            
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error en FLAML: {e}")
            return {}
    
    def create_ensembles(self, base_models: Dict, X_train, y_train, X_val, y_val) -> Dict:
        """Crear ensembles de los modelos base"""
        if len(base_models) < 2:
            return {}
        
        ensembles = {}
        
        # Filtrar solo modelos sklearn que tienen el método fit
        sklearn_models = []
        sklearn_names = []
        
        for name, model_info in base_models.items():
            if (model_info.get('type') == 'sklearn' and 
                hasattr(model_info.get('model'), 'fit') and
                hasattr(model_info.get('model'), 'predict')):
                sklearn_models.append((name, model_info['model']))
                sklearn_names.append(name)
        
        if len(sklearn_models) >= 2:
            try:
                # Voting Ensemble
                voting_regressor = VotingRegressor(sklearn_models)
                voting_regressor.fit(X_train, y_train)
                voting_pred = voting_regressor.predict(X_val)
                voting_r2 = r2_score(y_val, voting_pred)
                
                ensembles['VotingEnsemble'] = {
                    'model': voting_regressor,
                    'r2': voting_r2,
                    'predictions': voting_pred,
                    'type': 'sklearn'
                }
                
                # Stacking Ensemble
                if len(sklearn_models) >= 3:
                    stacking_regressor = StackingRegressor(
                        sklearn_models[:3],  # Usar solo los primeros 3
                        final_estimator=LinearRegression(),
                        cv=3
                    )
                    stacking_regressor.fit(X_train, y_train)
                    stacking_pred = stacking_regressor.predict(X_val)
                    stacking_r2 = r2_score(y_val, stacking_pred)
                    
                    ensembles['StackingEnsemble'] = {
                        'model': stacking_regressor,
                        'r2': stacking_r2,
                        'predictions': stacking_pred,
                        'type': 'sklearn'
                    }
                
            except Exception as e:
                if self.console:
                    self.console.print(f"❌ Error creando ensembles: {e}")
        
        return ensembles
    
    def realistic_backtest(self, model_info: Dict, X_test, y_test, model_name: str, 
                          model_type: str = 'sklearn') -> Dict:
        """Hacer backtest realista del modelo"""
        try:
            model = model_info.get('model')
            if model is None:
                return {'error': 'Modelo no encontrado'}
            
            # Hacer predicciones
            if model_type == 'pytorch':
                # Para modelos PyTorch
                model.eval()
                try:
                    import torch
                    device = model_info.get('device', 'cpu')
                    X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
                    if device != 'cpu':
                        X_test_tensor = X_test_tensor.to(device)
                    
                    with torch.no_grad():
                        predictions = model(X_test_tensor).cpu().numpy().flatten()
                except Exception as e:
                    return {'error': f'Error en predicción PyTorch: {e}'}
                    
            elif model_type == 'flaml':
                predictions = model.predict(X_test)
            else:
                # Para modelos sklearn y similares
                predictions = model.predict(X_test)
            
            # Calcular métricas básicas
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            # Simular trading básico
            trading_metrics = self._simulate_trading(predictions, y_test)
            
            return {
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'trading_metrics': trading_metrics,
                'model_type': model_type
            }
            
        except Exception as e:
            return {'error': f'Error en backtest: {str(e)}'}
    
    def _simulate_trading(self, predictions, actual_returns) -> Dict:
        """Simular trading básico"""
        try:
            # Estrategia simple: comprar si predicción > umbral
            threshold = np.percentile(predictions, 60)  # Top 40% de predicciones
            signals = (predictions > threshold).astype(int)
            
            # Calcular returns del trading
            strategy_returns = signals * actual_returns
            total_return = np.sum(strategy_returns)
            
            # Métricas básicas
            num_trades = np.sum(signals)
            winning_trades = np.sum((signals == 1) & (actual_returns > 0))
            win_rate = winning_trades / num_trades if num_trades > 0 else 0
            
            # Retorno anualizado (asumiendo datos horarios)
            annual_return = total_return * (365 * 24) / len(actual_returns)
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'num_trades': int(num_trades),
                'win_rate': float(win_rate),
                'avg_return_per_trade': float(total_return / num_trades) if num_trades > 0 else 0
            }
            
        except Exception as e:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0,
                'error': str(e)
            }
    
    def display_results(self, results: List[Dict]) -> None:
        """Mostrar resultados en tabla bonita"""
        
        # Filtrar resultados válidos
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            if self.console:
                self.console.print("❌ No hay resultados válidos para mostrar")
            return
        
        if self.console and RICH_AVAILABLE:
            # Crear tabla
            table = Table(title="📊 Resultados del Ultra Complete Ensemble")
            
            table.add_column("Modelo", style="cyan", no_wrap=True)
            table.add_column("R²", style="magenta", justify="right")
            table.add_column("Win Rate", style="green", justify="right")
            table.add_column("Annual Return", style="blue", justify="right")
            table.add_column("Trades", style="yellow", justify="right")
            
            # Ordenar por win rate
            valid_results.sort(key=lambda x: x.get('trading_metrics', {}).get('win_rate', 0), reverse=True)
            
            for result in valid_results[:15]:  # Mostrar top 15
                model_name = result.get('model_name', 'Unknown')
                r2 = result.get('r2', 0)
                trading_metrics = result.get('trading_metrics', {})
                win_rate = trading_metrics.get('win_rate', 0)
                annual_return = trading_metrics.get('annual_return', 0)
                num_trades = trading_metrics.get('num_trades', 0)
                
                table.add_row(
                    model_name[:20],  # Truncar nombre si es muy largo
                    f"{r2:.4f}",
                    f"{win_rate:.2%}",
                    f"{annual_return:.2%}",
                    str(num_trades)
                )
            
            self.console.print(table)
        else:
            # Versión sin rich
            print("\n📊 Resultados del Ultra Complete Ensemble")
            print("=" * 60)
            print(f"{'Modelo':<20} {'R²':<8} {'Win Rate':<10} {'Annual Return':<15} {'Trades':<8}")
            print("-" * 60)
            
            valid_results.sort(key=lambda x: x.get('trading_metrics', {}).get('win_rate', 0), reverse=True)
            
            for result in valid_results[:15]:
                model_name = result.get('model_name', 'Unknown')[:20]
                r2 = result.get('r2', 0)
                trading_metrics = result.get('trading_metrics', {})
                win_rate = trading_metrics.get('win_rate', 0)
                annual_return = trading_metrics.get('annual_return', 0)
                num_trades = trading_metrics.get('num_trades', 0)
                
                print(f"{model_name:<20} {r2:<8.4f} {win_rate:<10.2%} {annual_return:<15.2%} {num_trades:<8}")

# Alias para compatibilidad
HyperionTrainer = UltraCompleteEnsembleTrainer

class SimpleHyperionTrainer:
    """
    Versión simplificada del trainer para testing y desarrollo rápido
    """
    
    def __init__(self, device="cpu", dtype=None, console=None):
        """
        Args:
            device: Dispositivo de cómputo (cpu, cuda, mps)
            dtype: Tipo de datos (opcional)
            console: Rich console (opcional)
        """
        self.device = device
        self.dtype = dtype
        self.console = console or (Console() if RICH_AVAILABLE else None)
        
        # Configurar entorno
        dependency_status, gpu_config, validation_issues = initialize_environment()
        self.dependency_status = dependency_status
        self.gpu_config = gpu_config
        self.validation_issues = validation_issues
        
        if self.console:
            # Obtener número de cores disponibles
            import os
            total_cores = os.cpu_count() or 8
            self.console.print(f"🔧 Configuración: {total_cores} cores, GPU: {self.device}")
    
    def train_ensemble(self, symbol: str = "BTC/USDT", quick_mode: bool = True):
        """
        Entrenar ensemble de modelos de forma simplificada
        
        Args:
            symbol: Par de trading
            quick_mode: Modo rápido con menos modelos
            
        Returns:
            Resultados del entrenamiento
        """
        if self.console:
            self.console.print(f"🚀 Entrenamiento simplificado para {symbol}")
        
        # Crear datos sintéticos para testing
        from utils.features import engineer_features
        import pandas as pd
        import numpy as np
        
        # Datos sintéticos mínimos
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.random(1000) * 100 + 50000,
            'high': np.random.random(1000) * 100 + 50000,
            'low': np.random.random(1000) * 100 + 50000,
            'close': np.random.random(1000) * 100 + 50000,
            'volume': np.random.random(1000) * 1000,
        })
        
        # Crear features
        features_df = engineer_features(data, self.console)
        
        if self.console:
            self.console.print(f"✅ Features creadas: {features_df.shape}")
        
        return {
            "status": "completed",
            "symbol": symbol,
            "features_shape": features_df.shape,
            "device": self.device,
            "models_trained": ["synthetic_test"]
        }

def main():
    """Función principal para ejecutar el entrenamiento"""
    try:
        trainer = UltraCompleteEnsembleTrainer(optimize_hyperparams=True)
        results = trainer.train_ultra_complete_ensemble('SOL/USDT')
        return results
    except Exception as e:
        print(f"❌ Error en ejecución principal: {e}")
        return []

if __name__ == "__main__":
    main()
