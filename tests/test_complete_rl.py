#!/usr/bin/env python3
"""
Test para verificar el estado de RL agents desde el sistema completo
"""

import sys
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

def test_complete_system():
    """Probar el estado completo desde el sistema principal"""
    print("🧪 Probando estado completo del sistema...")
    
    try:
        # Importar y crear el sistema principal
        from main_professional import HyperionMainSystem
        system = HyperionMainSystem()
        print("✅ Sistema principal creado")
        
        # Importar el optimizador
        from utils.hyperopt import HyperparameterOptimizer
        optimizer = HyperparameterOptimizer(console=system.console)
        print("✅ Optimizador creado")
        
        # Obtener capacidades
        capabilities = optimizer.get_optimization_capabilities()
        print("✅ Capacidades obtenidas")
        
        # Verificar RL agents específicamente
        rl_info = capabilities['model_categories']['reinforcement_learning']
        print(f"🔍 RL Info completa: {rl_info}")
        print(f"🎯 Agentes disponibles: {rl_info.get('agents', [])}")
        print(f"📊 Disponibilidad general: {rl_info.get('available', False)}")
        
        # Verificar también las variables individuales dentro del optimizador
        from utils.hyperopt import RL_SAC_AVAILABLE, RL_TD3_AVAILABLE, RL_RAINBOW_AVAILABLE
        print(f"🔧 Variables de disponibilidad:")
        print(f"   - RL_SAC_AVAILABLE: {RL_SAC_AVAILABLE}")
        print(f"   - RL_TD3_AVAILABLE: {RL_TD3_AVAILABLE}")
        print(f"   - RL_RAINBOW_AVAILABLE: {RL_RAINBOW_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test completo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)
