#!/usr/bin/env python3
"""
Script de prueba para verificar que las importaciones de TFT funcionen
"""

import sys
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

def test_tft_import():
    """Probar importación de TFT"""
    print("🧪 Probando importación de TFT...")
    
    try:
        # Probar importación directa
        from hyperion3.models.transformers.tft import TemporalFusionTransformer
        print("✅ TemporalFusionTransformer importado correctamente")
        
        # Probar creación del modelo
        config = {
            "hidden_size": 128,
            "attention_heads": 4,
            "lstm_layers": 2,
            "dropout": 0.1,
            "num_inputs": 5,
            "num_time_features": 2,
            "num_static_features": 0,
            "prediction_length": 12,
            "quantiles": [0.1, 0.5, 0.9]
        }
        
        model = TemporalFusionTransformer(config)
        print("✅ Modelo TFT creado correctamente")
        
        # Probar optimizador
        from utils.hyperopt import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer()
        capabilities = optimizer.get_optimization_capabilities()
        
        tft_available = capabilities['model_categories']['advanced_models']['tft']
        print(f"🔍 TFT disponible para optimización: {tft_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba TFT: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tft_import()
    sys.exit(0 if success else 1)
