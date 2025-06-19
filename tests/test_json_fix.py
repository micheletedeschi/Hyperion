#!/usr/bin/env python3
"""
Test para verificar que las correcciones de JSON funcionan
"""

import sys
import os
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

from hyperion_mlops import make_json_serializable, clean_metrics_for_json
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def test_json_serialization():
    """Probar la serialización JSON con objetos de modelo"""
    print("🧪 Probando serialización JSON...")
    
    # Crear un modelo
    model = RandomForestRegressor(n_estimators=10)
    X = np.random.random((100, 5))
    y = np.random.random(100)
    model.fit(X, y)
    
    # Crear métricas que incluyen el modelo (problemático antes)
    metrics = {
        'r2_score': 0.85,
        'mse': 0.15,
        'model_object': model,  # Esto causaba el error
        'feature_importance': {'feature_1': 0.3, 'feature_2': 0.7},
        'config': {'n_estimators': 10, 'random_state': 42}
    }
    
    # Probar función de limpieza
    try:
        cleaned = clean_metrics_for_json(metrics)
        print(f"✅ Métricas limpiadas exitosamente")
        print(f"   - Keys originales: {list(metrics.keys())}")
        print(f"   - Keys limpiadas: {list(cleaned.keys())}")
        print(f"   - model_object removido: {'model_object' not in cleaned}")
        
        # Intentar serializar a JSON
        import json
        json_str = json.dumps(cleaned)
        print(f"✅ JSON serializado exitosamente (length: {len(json_str)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TEST DE CORRECCIONES JSON")
    print("=" * 40)
    
    success = test_json_serialization()
    
    if success:
        print("\n✅ Las correcciones JSON funcionan correctamente")
    else:
        print("\n❌ Hay problemas con las correcciones JSON")
