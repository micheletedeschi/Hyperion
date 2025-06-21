#!/usr/bin/env python3
"""
Script de prueba para verificar el menú de hiperparámetros actualizado
"""

import sys
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

def test_hyperparameters_menu():
    """Probar el menú de hiperparámetros"""
    print("🧪 Probando capacidades de optimización...")
    
    try:
        # Verificar que el optimizador está disponible
        from utils.hyperopt import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer()
        
        print("✅ Optimizador inicializado correctamente")
        
        # Mostrar capacidades usando la función incorporada
        print("\n� Mostrando capacidades completas:")
        optimizer.print_optimization_summary()
        
        print("\n✅ Test completado exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hyperparameters_menu()
    sys.exit(0 if success else 1)
