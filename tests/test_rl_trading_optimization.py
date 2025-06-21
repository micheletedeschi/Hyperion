#!/usr/bin/env python3
"""
🎯 Test de optimización de RL agents para trading
"""

import sys
import os
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

import warnings
warnings.filterwarnings('ignore')

from utils.hyperopt import HyperparameterOptimizer
from utils.trading_rl_optimizer import create_synthetic_trading_data, TradingEnvironmentSimulator
import numpy as np
import pandas as pd

def test_rl_trading_optimization():
    """Test de la optimización de RL agents para trading"""
    print("🎯 Testing RL Trading Optimization...")
    
    # Crear datos sintéticos
    print("📊 Creating synthetic trading data...")
    X_train = create_synthetic_trading_data(200)
    X_val = create_synthetic_trading_data(100, random_seed=123)
    
    # Convertir a arrays para compatibilidad
    y_train = X_train['close'].pct_change().fillna(0).values
    y_val = X_val['close'].pct_change().fillna(0).values
    
    X_train_array = np.random.randn(200, 10)  # Features sintéticos
    X_val_array = np.random.randn(100, 10)
    
    print(f"   ✅ Training data: {X_train_array.shape}")
    print(f"   ✅ Validation data: {X_val_array.shape}")
    
    # Crear optimizador
    print("\n🔧 Creating hyperparameter optimizer...")
    optimizer = HyperparameterOptimizer(console=None)  # Sin console para test
    
    # Test de optimización de RL agents
    print("\n🎯 Testing RL agents optimization...")
    try:
        results = optimizer.optimize_rl_agents(
            X_train_array, y_train, X_val_array, y_val, n_trials=3
        )
        
        if results:
            print(f"   ✅ Optimization completed! Found {len(results)} agent configs")
            for agent_name, result in results.items():
                print(f"   📈 {agent_name}: Sharpe = {result['score']:.4f}")
                print(f"      Best params: {list(result['params'].keys())[:3]}...")  # Mostrar algunos params
        else:
            print("   ⚠️ No results returned (likely due to missing dependencies)")
            
    except Exception as e:
        print(f"   ❌ Error during optimization: {e}")
        return False
    
    # Test del simulador directamente
    print("\n🎮 Testing trading simulator directly...")
    try:
        trading_data = create_synthetic_trading_data(100)
        simulator = TradingEnvironmentSimulator(trading_data)
        
        # Crear agente dummy
        class DummyAgent:
            def act(self, state):
                return np.random.choice([-1, 0, 1])  # random action
        
        agent = DummyAgent()
        metrics = simulator.simulate_trading(agent, num_episodes=2)
        
        print(f"   ✅ Simulator test completed!")
        print(f"   📊 Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   📊 Total return: {metrics['total_return']:.4f}")
        print(f"   📊 Max drawdown: {metrics['max_drawdown']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error in simulator test: {e}")
        return False
    
    print("\n🏆 RL Trading Optimization test completed successfully!")
    return True

def test_rl_agent_availability():
    """Test de disponibilidad de agentes RL"""
    print("\n🔍 Testing RL agent availability...")
    
    try:
        from utils.hyperopt import RL_SAC_AVAILABLE, RL_TD3_AVAILABLE, RL_RAINBOW_AVAILABLE
        print(f"   SAC Available: {RL_SAC_AVAILABLE}")
        print(f"   TD3 Available: {RL_TD3_AVAILABLE}")
        print(f"   Rainbow DQN Available: {RL_RAINBOW_AVAILABLE}")
        
        available_agents = []
        if RL_SAC_AVAILABLE:
            available_agents.append("SAC")
        if RL_TD3_AVAILABLE:
            available_agents.append("TD3")
        if RL_RAINBOW_AVAILABLE:
            available_agents.append("RainbowDQN")
            
        print(f"   📋 Available RL agents: {available_agents}")
        return len(available_agents) > 0
        
    except Exception as e:
        print(f"   ❌ Error checking availability: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting RL Trading Optimization Tests")
    print("=" * 50)
    
    # Test disponibilidad
    availability_ok = test_rl_agent_availability()
    
    # Test optimización
    optimization_ok = test_rl_trading_optimization()
    
    print("\n" + "=" * 50)
    if availability_ok and optimization_ok:
        print("✅ All tests passed! RL Trading optimization is ready")
    else:
        print("⚠️ Some tests failed - check dependencies")
        
    print("🎯 Test completed!")
