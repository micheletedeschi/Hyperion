#!/usr/bin/env python3
"""
Optimización simplificada de hiperparámetros para diagnóstico
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def create_simple_data(n_samples=1000):
    """Crear datos sintéticos simples para pruebas"""
    # Usar make_regression directamente
    X, y = make_regression(
        n_samples=n_samples, 
        n_features=10, 
        noise=0.1, 
        random_state=42
    )
    
    # Convertir a DataFrame para compatibilidad
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y

def test_single_model():
    """Probar un modelo simple"""
    print("🧪 Probando modelo simple...")
    
    # Crear datos
    X, y = create_simple_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Probar Random Forest simple
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = r2_score(y_val, y_pred)
    
    print(f"✅ Random Forest R2: {score:.4f}")
    return score > 0.5  # Score debe ser decente

def test_optuna_simple():
    """Probar Optuna con datos simples"""
    print("🧪 Probando Optuna con datos simples...")
    
    try:
        import optuna
        from optuna.samplers import TPESampler
        
        # Crear datos simples
        X, y = create_simple_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(trial):
            # Parámetros simples
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            
            try:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                print(f"  Trial: n_est={n_estimators}, depth={max_depth}, score={score:.4f}")
                return score
            except Exception as e:
                print(f"  ❌ Error en trial: {e}")
                return -999
        
        # Crear estudio
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=5, show_progress_bar=False)
        
        print(f"✅ Mejor score: {study.best_value:.4f}")
        print(f"✅ Mejores parámetros: {study.best_params}")
        
        return study.best_value > 0.5
        
    except Exception as e:
        print(f"❌ Error en Optuna: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xgboost_simple():
    """Probar XGBoost con Optuna"""
    print("🧪 Probando XGBoost con Optuna...")
    
    try:
        import optuna
        import xgboost as xgb
        from optuna.samplers import TPESampler
        
        # Crear datos simples
        X, y = create_simple_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(trial):
            try:
                # Para XGBoost 3.x, usar early_stopping_rounds en el constructor  
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'early_stopping_rounds': 10,  # Mover aquí para XGBoost 3.x
                    'random_state': 42,
                    'tree_method': 'hist',
                    'n_jobs': 1
                }
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                print(f"  XGBoost trial: score={score:.4f}")
                return score
            except Exception as e:
                print(f"  ❌ Error en XGBoost trial: {e}")
                return -999
        
        # Crear estudio
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=3, show_progress_bar=False)
        
        print(f"✅ XGBoost mejor score: {study.best_value:.4f}")
        print(f"✅ XGBoost mejores parámetros: {study.best_params}")
        
        return study.best_value > 0.5
        
    except Exception as e:
        print(f"❌ Error en XGBoost: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 DIAGNÓSTICO SIMPLIFICADO DE HIPERPARÁMETROS")
    print("=" * 60)
    
    success = 0
    total = 0
    
    # Test 1: Modelo simple
    total += 1
    if test_single_model():
        success += 1
    
    print()
    
    # Test 2: Optuna simple
    total += 1
    if test_optuna_simple():
        success += 1
    
    print()
    
    # Test 3: XGBoost con Optuna
    total += 1
    if test_xgboost_simple():
        success += 1
    
    print()
    print(f"🏁 Resultado: {success}/{total} tests exitosos")
    
    if success == total:
        print("✅ Todo funciona correctamente - el problema está en otro lugar")
    else:
        print("❌ Hay problemas en la configuración básica")
