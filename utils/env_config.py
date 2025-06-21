#!/usr/bin/env python3
"""
🔧 CONFIGURACIÓN DE ENTORNO Y DEPENDENCIAS
Configuración centralizada para Mac M4 con Metal/MPS y validación de dependencias
"""

import os
import sys
import warnings
import logging
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Tuple

# Configuración básica para Mac ANTES que cualquier otra cosa
warnings.filterwarnings('ignore')

# Configurar variables de entorno para optimización en Mac
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'spawn'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Allow unlimited MPS memory

# Setup logger
logger = logging.getLogger(__name__)

# Global flags for available libraries
PYTORCH_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False
OPTUNA_AVAILABLE = False
FLAML_AVAILABLE = False
RAY_AVAILABLE = False
HYPERION_MODELS_AVAILABLE = False
DIGA_AVAILABLE = False
SKLEARN_SEARCH_AVAILABLE = False
FINANCIAL_HYPEROPT_AVAILABLE = False

# Device configuration
DEVICE = None

def check_and_import_dependencies() -> Dict[str, bool]:
    """
    Verificar e importar todas las dependencias del proyecto
    Returns: Dict con el estado de cada librería
    """
    global PYTORCH_AVAILABLE, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
    global CATBOOST_AVAILABLE, OPTUNA_AVAILABLE, FLAML_AVAILABLE
    global RAY_AVAILABLE, HYPERION_MODELS_AVAILABLE, DIGA_AVAILABLE
    global SKLEARN_SEARCH_AVAILABLE, FINANCIAL_HYPEROPT_AVAILABLE, DEVICE
    
    dependency_status = {}
    
    # PyTorch imports
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        PYTORCH_AVAILABLE = True
        dependency_status['pytorch'] = True
        
        # Configurar device para Mac M4 con MPS
        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
            print("🚀 PyTorch: USANDO Metal Performance Shaders (MPS) - Mac M4 OPTIMIZADO!")
            # Verificar que MPS funciona correctamente
            test_tensor = torch.tensor([1.0, 2.0]).to(torch.device("mps"))
            print(f"✅ MPS Test exitoso: {test_tensor}")
        elif torch.cuda.is_available():
            DEVICE = torch.device("cuda")
            print("🚀 PyTorch: Usando CUDA")
        else:
            DEVICE = torch.device("cpu")
            print("ℹ️ PyTorch: Usando CPU")
        
        print(f"✅ PyTorch cargado correctamente - Device: {DEVICE}")
        
    except ImportError as e:
        PYTORCH_AVAILABLE = False
        DEVICE = None
        dependency_status['pytorch'] = False
        print(f"❌ PyTorch no disponible: {e}")
    
    # XGBoost imports
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
        dependency_status['xgboost'] = True
        print("✅ XGBoost cargado correctamente")
    except ImportError as e:
        XGBOOST_AVAILABLE = False
        dependency_status['xgboost'] = False
        print(f"❌ XGBoost no disponible: {e}")
    
    # LightGBM imports
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
        dependency_status['lightgbm'] = True
        print("✅ LightGBM cargado correctamente")
    except ImportError as e:
        LIGHTGBM_AVAILABLE = False
        dependency_status['lightgbm'] = False
        print(f"❌ LightGBM no disponible: {e}")
    
    # CatBoost imports
    try:
        from catboost import CatBoostRegressor
        CATBOOST_AVAILABLE = True
        dependency_status['catboost'] = True
        print("✅ CatBoost cargado correctamente")
    except ImportError as e:
        CATBOOST_AVAILABLE = False
        dependency_status['catboost'] = False
        print(f"❌ CatBoost no disponible: {e}")
    
    # Optuna imports
    try:
        import optuna
        from optuna.samplers import TPESampler
        OPTUNA_AVAILABLE = True
        dependency_status['optuna'] = True
        print("✅ Optuna cargado correctamente")
    except ImportError as e:
        OPTUNA_AVAILABLE = False
        dependency_status['optuna'] = False
        print(f"❌ Optuna no disponible: {e}")
    
    # FLAML imports
    try:
        from flaml import AutoML
        FLAML_AVAILABLE = True
        dependency_status['flaml'] = True
        print("✅ FLAML cargado correctamente")
    except ImportError as e:
        FLAML_AVAILABLE = False
        dependency_status['flaml'] = False
        print(f"❌ FLAML no disponible: {e}")
    
    # Ray imports
    try:
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        RAY_AVAILABLE = True
        dependency_status['ray'] = True
        print("✅ Ray cargado correctamente")
    except ImportError as e:
        RAY_AVAILABLE = False
        dependency_status['ray'] = False
        print(f"❌ Ray no disponible: {e}")
    
    # Sklearn search imports
    try:
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        from scipy.stats import randint, uniform
        SKLEARN_SEARCH_AVAILABLE = True
        dependency_status['sklearn_search'] = True
        print("✅ Sklearn search cargado correctamente")
    except ImportError as e:
        SKLEARN_SEARCH_AVAILABLE = False
        dependency_status['sklearn_search'] = False
        print(f"❌ Sklearn search no disponible: {e}")
    
    # Hyperion3 models imports - Verificar cada uno individualmente
    hyperion_models_status = {}
    hyperion_models_count = 0
    
    # PatchTST
    try:
        from hyperion3.models.transformers.patchtst import PatchTST
        hyperion_models_status['patchtst'] = True
        hyperion_models_count += 1
        print("✅ PatchTST cargado correctamente")
    except ImportError as e:
        hyperion_models_status['patchtst'] = False
        print(f"⚠️ PatchTST no disponible: {e}")
    
    # TFT
    try:
        from hyperion3.models.transformers.tft import TFTCryptoPredictor
        hyperion_models_status['tft'] = True
        hyperion_models_count += 1
        print("✅ TFT cargado correctamente")
    except ImportError as e:
        hyperion_models_status['tft'] = False
        print(f"⚠️ TFT no disponible: {e}")
    
    # Rainbow DQN
    try:
        from hyperion3.models.rl_agents.rainbow_dqn import RainbowTradingAgent
        hyperion_models_status['rainbow_dqn'] = True
        hyperion_models_count += 1
        print("✅ RainbowDQN cargado correctamente")
    except ImportError as e:
        hyperion_models_status['rainbow_dqn'] = False
        print(f"⚠️ RainbowDQN no disponible: {e}")
    
    # SAC
    try:
        from hyperion3.models.rl_agents.sac import SACTradingAgent
        hyperion_models_status['sac'] = True
        hyperion_models_count += 1
        print("✅ SAC cargado correctamente")
    except ImportError as e:
        hyperion_models_status['sac'] = False
        print(f"⚠️ SAC no disponible: {e}")
    
    # TD3
    try:
        from hyperion3.models.rl_agents.td3 import TD3TradingAgent
        hyperion_models_status['td3'] = True
        hyperion_models_count += 1
        print("✅ TD3 cargado correctamente")
    except ImportError as e:
        hyperion_models_status['td3'] = False
        print(f"⚠️ TD3 no disponible: {e}")
    
    # Ensemble Agent - Re-enabled with robust import handling
    try:
        from hyperion3.models.rl_agents.ensemble_agent import EnsembleAgent
        hyperion_models_status['ensemble_agent'] = True
        hyperion_models_count += 1
        print("✅ EnsembleAgent disponible")
    except ImportError as e:
        hyperion_models_status['ensemble_agent'] = False
        print(f"⚠️ EnsembleAgent no disponible: {e}")
    
    # Determinar estado general
    HYPERION_MODELS_AVAILABLE = hyperion_models_count > 0
    dependency_status['hyperion_models'] = HYPERION_MODELS_AVAILABLE
    dependency_status['hyperion_models_detail'] = hyperion_models_status
    
    if HYPERION_MODELS_AVAILABLE:
        print(f"🚀 Modelos avanzados Hyperion3: {hyperion_models_count}/6 cargados exitosamente")
    else:
        print("⚠️ Ningún modelo avanzado Hyperion3 disponible")
    
    # DiGA imports
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        advanced_models_path = os.path.join(script_dir, '..', 'advanced models')
        
        if advanced_models_path not in sys.path:
            sys.path.insert(0, advanced_models_path)
        
        from diga import DiffusionGuidedAgent
        DIGA_AVAILABLE = True
        dependency_status['diga'] = True
        print(f"🌟 DiGA (Diffusion Guided Agent) cargado exitosamente from {advanced_models_path}!")
    except ImportError as e:
        DIGA_AVAILABLE = False
        dependency_status['diga'] = False
        print(f"⚠️ DiGA no disponible: {e}")
    
    # Financial hyperopt imports
    try:
        from financial_hyperopt import FinancialHyperparameterOptimizer, get_financial_baseline_params
        FINANCIAL_HYPEROPT_AVAILABLE = True
        dependency_status['financial_hyperopt'] = True
        print("🎯 Optimizador financiero especializado cargado exitosamente!")
    except ImportError as e:
        FINANCIAL_HYPEROPT_AVAILABLE = False
        dependency_status['financial_hyperopt'] = False
        print(f"⚠️ Optimizador financiero no disponible: {e}")
    
    print("🔧 Configuración Mac aplicada")
    return dependency_status

def configure_gpu_for_models() -> Dict[str, Any]:
    """
    Configurar GPU/Metal para diferentes librerías
    Returns: Dict con configuración optimizada para cada framework
    """
    gpu_config = {
        'pytorch_device': DEVICE,
        'xgboost_params': {},
        'lightgbm_params': {},
        'catboost_params': {}
    }
    
    # XGBoost: usar todos los cores para Mac M4
    gpu_config['xgboost_params'] = {
        'tree_method': 'hist',  # Más rápido en Mac
        'n_jobs': 8,  # Usar todos los cores del M4
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # LightGBM: optimizado para Mac M4
    gpu_config['lightgbm_params'] = {
        'device': 'cpu',
        'num_threads': 8,  # Todos los cores
        'force_row_wise': True,  # Mejor para Mac
        'verbose': -1
    }
    
    # CatBoost: máximo rendimiento en Mac M4  
    gpu_config['catboost_params'] = {
        'task_type': 'CPU',
        'thread_count': 8,  # Todos los cores
        'bootstrap_type': 'Bernoulli',  # Más rápido
        'od_type': 'Iter',
        'od_wait': 20
    }
    
    print(f"🔧 GPU Config: PyTorch device = {gpu_config['pytorch_device']}")
    print(f"🔧 Usando 8 cores para XGBoost, LightGBM y CatBoost")
    
    return gpu_config

def get_optimal_sklearn_params() -> Dict[str, Any]:
    """
    Parámetros optimizados para sklearn - MENOS conservadores para mejor R²
    Returns: Dict con parámetros optimizados
    """
    return {
        'n_jobs': 4, 
        'random_state': 42,
        # Parámetros menos conservadores para Random Forest
        'rf_n_estimators': 100,    # Incrementado
        'rf_max_depth': 12,        # Menos limitado
        'rf_min_samples_split': 5,  # Reducido
        'rf_min_samples_leaf': 2,   # Reducido
        # Parámetros más conservadores para XGBoost/LightGBM  
        'xgb_n_estimators': 50,    # Reducido para evitar overfitting
        'xgb_max_depth': 4,        # Más shallow
        'xgb_learning_rate': 0.05, # Más conservador
        'xgb_subsample': 0.7,      # Más restrictivo para regularización
        'xgb_colsample_bytree': 0.7,
        # Regularización más fuerte
        'xgb_reg_alpha': 0.1,      # Aumentada
        'xgb_reg_lambda': 0.1      # Aumentada
    }

def configure_multiprocessing_for_mac() -> Dict[str, int]:
    """
    Configurar multiprocessing para Mac
    Returns: Dict con configuración de cores y procesos
    """
    cpu_count = mp.cpu_count() if mp.cpu_count() else 8
    return {
        'total_cores': cpu_count,
        'physical_cores': cpu_count // 2,
        'sklearn_jobs': 4,
        'search_cv_jobs': 2,
        'optuna_jobs': 1,
        'preprocessing_jobs': 4
    }

def validate_environment_and_dependencies() -> List[str]:
    """
    Validar entorno y dependencias antes del entrenamiento
    Returns: Lista de problemas encontrados
    """
    validation_issues = []
    
    # Verificar PyTorch y MPS
    if PYTORCH_AVAILABLE:
        try:
            import torch
            if not torch.backends.mps.is_available():
                validation_issues.append("⚠️ MPS no disponible - usando CPU para PyTorch")
            else:
                print("✅ PyTorch con MPS disponible")
        except Exception as e:
            validation_issues.append(f"❌ Error con PyTorch: {e}")
    else:
        validation_issues.append("❌ PyTorch no está instalado")
    
    # Verificar librerías críticas
    critical_libs = {
        'sklearn': 'scikit-learn',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'rich': 'rich'
    }
    
    for lib, install_name in critical_libs.items():
        try:
            __import__(lib)
            print(f"✅ {install_name} disponible")
        except ImportError:
            validation_issues.append(f"❌ {install_name} no está instalado")
    
    # Verificar librerías opcionales
    optional_libs = {
        'xgboost': XGBOOST_AVAILABLE,
        'lightgbm': LIGHTGBM_AVAILABLE, 
        'catboost': CATBOOST_AVAILABLE,
        'flaml': FLAML_AVAILABLE,
        'optuna': OPTUNA_AVAILABLE
    }
    
    for lib, available in optional_libs.items():
        if available:
            print(f"✅ {lib} disponible")
        else:
            print(f"⚠️ {lib} no disponible (opcional)")
    
    # Verificar archivos de datos
    data_file = "./data/SOL_USDT_20250613.csv"
    if os.path.exists(data_file):
        print(f"✅ Archivo de datos encontrado: {data_file}")
    else:
        validation_issues.append(f"❌ Archivo de datos no encontrado: {data_file}")
    
    # Verificar directorios
    dirs_to_check = ['./results', './logs', './checkpoints']
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✅ Directorio {dir_path} existe")
        else:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Directorio {dir_path} creado")
            except Exception as e:
                validation_issues.append(f"❌ No se pudo crear directorio {dir_path}: {e}")
    
    return validation_issues

def get_financial_baseline_params() -> Dict[str, Dict[str, Any]]:
    """
    Función de fallback para parámetros baseline financieros
    Returns: Dict con parámetros optimizados para modelos financieros
    """
    return {
        'SVR': {
            'params': {
                'kernel': 'rbf',
                'C': 2.0,
                'epsilon': 0.01,
                'gamma': 'scale'
            }
        },
        'RandomForest': {
            'params': {
                'n_estimators': 150,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42
            }
        },
        'XGBoost': {
            'params': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
        },
        'LightGBM': {
            'params': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbose': -1
            }
        },
        'MLPRegressor': {
            'params': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'alpha': 0.01,
                'learning_rate_init': 0.01,
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 20,
                'random_state': 42
            }
        }
    }

def emergency_fallback_models(X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    """
    Modelos de emergencia si todo falla
    Args:
        X_train, y_train, X_val, y_val: Datos de entrenamiento y validación
    Returns: Dict con modelos de emergencia entrenados
    """
    fallback_models = {}
    
    try:
        # Modelo más simple posible
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        pred = linear_model.predict(X_val)
        r2 = r2_score(y_val, pred)
        
        fallback_models['EmergencyLinear'] = {
            'model': linear_model,
            'r2': r2,
            'type': 'sklearn'
        }
        print(f"✅ Modelo de emergencia LinearRegression: R² = {r2:.4f}")
        
    except Exception as e:
        print(f"❌ Incluso el modelo de emergencia falló: {e}")
    
    return fallback_models

def check_hyperion_models() -> Dict[str, bool]:
    """
    Verificar el estado de los modelos Hyperion3 individualmente
    Returns: Dict con el estado de cada modelo
    """
    model_status = {}
    
    # PatchTST
    try:
        from hyperion3.models.transformers.patchtst import PatchTST
        model_status['patchtst'] = True
    except ImportError:
        model_status['patchtst'] = False
    
    # TFT
    try:
        from hyperion3.models.transformers.tft import TFTCryptoPredictor
        model_status['tft'] = True
    except ImportError:
        model_status['tft'] = False
    
    # Rainbow DQN
    try:
        from hyperion3.models.rl_agents.rainbow_dqn import RainbowTradingAgent
        model_status['rainbow_dqn'] = True
    except ImportError:
        model_status['rainbow_dqn'] = False
    
    # SAC
    try:
        from hyperion3.models.rl_agents.sac import SACTradingAgent
        model_status['sac'] = True
    except ImportError:
        model_status['sac'] = False
    
    # TD3
    try:
        from hyperion3.models.rl_agents.td3 import TD3TradingAgent
        model_status['td3'] = True
    except ImportError:
        model_status['td3'] = False
    
    # Ensemble Agent
    try:
        from hyperion3.models.rl_agents.ensemble_agent import EnsembleAgent
        model_status['ensemble_agent'] = True
    except ImportError:
        model_status['ensemble_agent'] = False
    
    return model_status

def get_available_hyperion_models() -> List[str]:
    """
    Obtener lista de modelos Hyperion3 disponibles
    Returns: Lista de nombres de modelos disponibles
    """
    status = check_hyperion_models()
    return [model for model, available in status.items() if available]

def print_hyperion_models_status():
    """Imprimir estado detallado de modelos Hyperion3"""
    status = check_hyperion_models()
    available_count = sum(status.values())
    total_count = len(status)
    
    print(f"\n🤖 Estado de Modelos Hyperion3: {available_count}/{total_count}")
    print("=" * 50)
    
    for model, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"{status_icon} {model.upper().replace('_', ' ')}")
    
    print("=" * 50)
    
    if available_count == total_count:
        print("🎉 Todos los modelos Hyperion3 están disponibles!")
    elif available_count > 0:
        print(f"⚠️ {available_count} de {total_count} modelos disponibles")
    else:
        print("❌ Ningún modelo Hyperion3 disponible")
    
    return status

def initialize_environment() -> Tuple[Dict[str, bool], Dict[str, Any], List[str]]:
    """
    Inicializar completamente el entorno
    Returns: Tuple con (dependency_status, gpu_config, validation_issues)
    """
    print("🔍 Inicializando entorno Hyperion3...")
    
    # 1. Verificar e importar dependencias
    dependency_status = check_and_import_dependencies()
    
    # 2. Configurar GPU y hardware
    gpu_config = configure_gpu_for_models()
    
    # 3. Validar entorno
    validation_issues = validate_environment_and_dependencies()
    
    if validation_issues:
        print("⚠️ Se encontraron problemas en el entorno:")
        for issue in validation_issues:
            print(f"  {issue}")
        print("⚠️ Continuando con funcionalidad limitada...")
    else:
        print("✅ Entorno validado correctamente")
    
    return dependency_status, gpu_config, validation_issues
