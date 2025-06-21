#!/usr/bin/env python3
"""
Script para verificar que las importaciones de modelos avanzados funcionan
"""

import sys
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

def test_advanced_models():
    """Probar importación de modelos avanzados"""
    print("🧪 Probando importaciones de modelos avanzados...")
    
    try:
        # Test TFT
        from hyperion3.models.transformers.tft import TemporalFusionTransformer
        print("✅ TemporalFusionTransformer importado correctamente")
        
        # Test config básico
        config = {
            'hidden_size': 128,
            'attention_heads': 4,
            'dropout': 0.1,
            'lstm_layers': 2,
            'num_inputs': 10,
            'num_time_features': 4,
            'prediction_length': 24,
            'quantiles': [0.1, 0.5, 0.9]
        }
        tft_model = TemporalFusionTransformer(config=config)
        print("✅ TFT model creado correctamente")
        
    except Exception as e:
        print(f"❌ Error con TFT: {e}")
    
    try:
        # Test PatchTST
        from hyperion3.models.transformers.patchtst import PatchTST
        print("✅ PatchTST importado correctamente")
        
        # Test config básico
        config = {
            'n_vars': 10,
            'lookback_window': 96,
            'pred_len': 24,
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.1,
            'patch_size': 16
        }
        patchtst_model = PatchTST(config=config)
        print("✅ PatchTST model creado correctamente")
        
    except Exception as e:
        print(f"❌ Error con PatchTST: {e}")
    
    try:
        # Test optimización
        from utils.hyperopt import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer()
        capabilities = optimizer.get_optimization_capabilities()
        print(f"✅ Optimizador funcionando con {len(capabilities['model_categories'])} categorías")
        
        # Mostrar modelos avanzados disponibles
        adv_models = capabilities['model_categories'].get('advanced_models', {})
        print(f"🔍 Modelos avanzados: TFT={adv_models.get('tft', False)}, PatchTST={adv_models.get('patchtst', False)}")
        
    except Exception as e:
        print(f"❌ Error con optimizador: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_models()
