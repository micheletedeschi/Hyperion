#!/usr/bin/env python3
"""
Test script para diagnóstico de errores en hiperparámetros
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Probar imports de librerías"""
    print("🔍 Probando imports...")
    
    try:
        import optuna
        print("✅ Optuna disponible")
    except ImportError as e:
        print(f"❌ Optuna no disponible: {e}")
        
    try:
        import xgboost as xgb
        print("✅ XGBoost disponible")
    except ImportError as e:
        print(f"❌ XGBoost no disponible: {e}")
        
    try:
        import lightgbm as lgb
        print("✅ LightGBM disponible")
    except ImportError as e:
        print(f"❌ LightGBM no disponible: {e}")

def test_basic_model():
    """Probar modelo básico"""
    print("\n🧪 Probando modelo básico...")
    
    # Crear datos sintéticos
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        # Probar Random Forest
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        print(f"✅ Random Forest R2: {score:.4f}")
        
    except Exception as e:
        print(f"❌ Error en Random Forest: {e}")

def test_xgboost():
    """Probar XGBoost específicamente"""
    print("\n🧪 Probando XGBoost...")
    
    try:
        import xgboost as xgb
        
        # Crear datos sintéticos
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Probar con parámetros básicos
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        print(f"✅ XGBoost R2: {score:.4f}")
        
    except Exception as e:
        print(f"❌ Error en XGBoost: {e}")
        import traceback
        traceback.print_exc()

def test_lightgbm():
    """Probar LightGBM específicamente"""
    print("\n🧪 Probando LightGBM...")
    
    try:
        import lightgbm as lgb
        
        # Crear datos sintéticos
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Probar con parámetros básicos
        model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            device='cpu',
            num_threads=1,
            verbose=-1
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        print(f"✅ LightGBM R2: {score:.4f}")
        
    except Exception as e:
        print(f"❌ Error en LightGBM: {e}")
        import traceback
        traceback.print_exc()

def test_optuna_simple():
    """Probar Optuna simple"""
    print("\n🧪 Probando Optuna simple...")
    
    try:
        import optuna
        from optuna.samplers import TPESampler
        
        # Crear datos sintéticos
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return r2_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=5)
        
        print(f"✅ Optuna Best score: {study.best_value:.4f}")
        print(f"✅ Optuna Best params: {study.best_params}")
        
    except Exception as e:
        print(f"❌ Error en Optuna: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 DIAGNÓSTICO DE HIPERPARÁMETROS")
    print("=" * 50)
    
    test_imports()
    test_basic_model()
    test_xgboost()
    test_lightgbm()
    test_optuna_simple()
    
    print("\n🏁 Diagnóstico completado")
