#!/usr/bin/env python3
"""
🔍 DEBUG: Optuna RL Optimization
"""

import sys
import os
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

import warnings
warnings.filterwarnings('ignore')

def test_optuna_rl_optimization():
    """Test de la optimización de Optuna directamente"""
    print("🔍 DEBUG: Optuna RL Optimization")
    print("=" * 50)
    
    try:
        from utils.hyperopt import HyperparameterOptimizer
        from utils.trading_rl_optimizer import create_synthetic_trading_data, TradingEnvironmentSimulator
        import numpy as np
        
        print("✅ Importaciones exitosas")
        
        # Crear datos sintéticos
        X_train = np.random.randn(200, 10)
        y_train = np.random.randn(200)
        X_val = np.random.randn(100, 10)
        y_val = np.random.randn(100)
        
        print("📊 Datos de entrenamiento creados")
        
        # Crear optimizador
        optimizer = HyperparameterOptimizer(console=None)
        
        print("🔧 Optimizador creado")
        
        # Probar solo 1 trial para debug
        print("\n🎯 Ejecutando optimización de RL agents (1 trial)...")
        results = optimizer.optimize_rl_agents(
            X_train, y_train, X_val, y_val, n_trials=1
        )
        
        print(f"\n📊 Resultados recibidos:")
        for agent_name, result in results.items():
            print(f"   {agent_name}:")
            print(f"      Score: {result['score']}")
            print(f"      Params: {list(result['params'].keys())[:3]}...")
            
        # Test manual del agente SAC
        print("\n🔍 Test manual de agente SAC...")
        try:
            from hyperion3.models.rl_agents.sac import SACAgent
            trading_data = create_synthetic_trading_data(100)
            simulator = TradingEnvironmentSimulator(trading_data)
            
            # Crear agente con parámetros específicos
            agent = SACAgent(
                state_dim=trading_data.shape[1],
                action_dim=3,
                hidden_dims=[256, 256],
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                batch_size=128,
                replay_buffer_size=50000
            )
            
            # Evaluar agente
            metrics = simulator.simulate_trading(agent, num_episodes=1)
            print(f"   📈 Sharpe ratio manual: {metrics['sharpe_ratio']:.6f}")
            print(f"   📊 Retorno total: {metrics['total_return']:.6f}")
            
        except Exception as e:
            print(f"   ❌ Error en test manual: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optuna_rl_optimization()
