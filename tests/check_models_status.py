#!/usr/bin/env python3
"""
Verificar estado de todos los modelos
"""

import sys
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

def check_all_models():
    print("🔍 Verificando estado de todos los modelos...")
    
    try:
        from utils.hyperopt import HyperparameterOptimizer
        
        optimizer = HyperparameterOptimizer()
        capabilities = optimizer.get_optimization_capabilities()
        
        print("\n📊 Estado actual:")
        for category, info in capabilities['model_categories'].items():
            print(f"  📁 {category}:")
            if category == 'sklearn' and info.get('available'):
                print(f"     ✅ {len(info['models'])} modelos disponibles")
            elif category == 'ensemble':
                available = [k for k, v in info.items() if v and k != 'available']
                print(f"     ✅ {len(available)}/3: {', '.join(available)}")
            elif category == 'deep_learning':
                if info.get('pytorch'):
                    print(f"     ✅ PyTorch: {', '.join(info['models'])}")
                else:
                    print("     ❌ PyTorch no disponible")
            elif category == 'advanced_models':
                available = [k for k, v in info.items() if v and k != 'available']
                print(f"     {'✅' if available else '❌'} {len(available)}/3: {', '.join(available) if available else 'Ninguno'}")
            elif category == 'reinforcement_learning':
                if info.get('available'):
                    print(f"     ✅ RL: {', '.join(info['agents'])}")
                else:
                    print("     ❌ RL no disponible")
        
        print(f"\n🎯 Total modelos optimizables: {sum(capabilities['hyperparameter_spaces'].values())}")
        
        # Verificar específicamente qué falta
        print("\n⚠️ Modelos faltantes:")
        if not capabilities['model_categories']['advanced_models'].get('tft'):
            print("  - TFT (Temporal Fusion Transformer)")
        if not capabilities['model_categories']['advanced_models'].get('patchtst'):
            print("  - PatchTST")
        if not capabilities['model_categories']['reinforcement_learning'].get('available'):
            print("  - Agentes RL (SAC, TD3, Rainbow DQN)")
            
        return capabilities
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    check_all_models()
