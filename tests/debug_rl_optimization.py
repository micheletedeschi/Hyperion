#!/usr/bin/env python3
"""
Script para debuggear la optimización de agentes RL
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

# Importaciones
try:
    from utils.trading_rl_optimizer import create_synthetic_trading_data, TradingEnvironmentSimulator
    from hyperion3.models.rl_agents.sac import SACAgent
    print("✅ Importaciones exitosas")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

def test_trading_data():
    """Probar creación de datos sintéticos"""
    print("\n🔍 Probando datos sintéticos...")
    
    data = create_synthetic_trading_data(100)
    print(f"   📊 Datos creados: {data.shape}")
    print(f"   📈 Columnas: {list(data.columns)}")
    print(f"   💰 Precio inicial: ${data['close'].iloc[0]:.2f}")
    print(f"   💰 Precio final: ${data['close'].iloc[-1]:.2f}")
    print(f"   📊 Retorno total: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    return data

def test_simulator_setup(data):
    """Probar configuración del simulador"""
    print("\n🎮 Probando simulador...")
    
    simulator = TradingEnvironmentSimulator(data, initial_balance=10000)
    print(f"   ✅ Simulador creado")
    print(f"   📊 Features shape: {simulator.features.shape}")
    print(f"   💰 Precios shape: {simulator.prices.shape}")
    print(f"   📈 Balance inicial: ${simulator.initial_balance}")
    
    # Verificar que no hay NaN en features
    nan_count = np.isnan(simulator.features).sum()
    print(f"   🔍 NaN en features: {nan_count}")
    
    if nan_count > 0:
        print("   ⚠️ WARNING: Hay NaN en los features!")
        
    return simulator

def test_agent_creation():
    """Probar creación de agente SAC"""
    print("\n🤖 Probando creación de agente SAC...")
    
    # Parámetros de prueba
    params = {
        'hidden_dims': [256, 256],
        'gamma': 0.98,
        'tau': 0.005,
        'alpha': 0.2,
        'batch_size': 128,
        'replay_buffer_size': 50000
    }
    
    try:
        agent = SACAgent(
            state_dim=9,  # Número de features del simulador
            action_dim=3,  # Buy, Hold, Sell
            hidden_dims=params['hidden_dims'],
            gamma=params['gamma'],
            tau=params['tau'],
            alpha=params['alpha'],
            batch_size=params['batch_size'],
            replay_buffer_size=params['replay_buffer_size']
        )
        print(f"   ✅ Agente SAC creado exitosamente")
        print(f"   🧠 Parámetros: {params}")
        
        return agent
    except Exception as e:
        print(f"   ❌ Error creando agente: {e}")
        return None

def test_agent_action(agent, simulator):
    """Probar obtención de acciones del agente"""
    print("\n🎯 Probando acciones del agente...")
    
    if agent is None:
        print("   ❌ No hay agente para probar")
        return
    
    # Probar con un estado
    state = simulator.features[0]
    print(f"   📊 Estado shape: {state.shape}")
    print(f"   📊 Estado: {state[:5]}...")  # Primeros 5 valores
    
    try:
        action = agent.act(state)
        print(f"   🎯 Acción original: {action}")
        print(f"   🎯 Tipo de acción: {type(action)}")
        print(f"   🎯 Shape de acción: {action.shape if hasattr(action, 'shape') else 'N/A'}")
        
        # Probar conversión a acción discreta
        if isinstance(action, (list, np.ndarray)):
            action_val = action[0] if len(action) > 0 else 0
        else:
            action_val = action
            
        if action_val > 0.3:
            discrete_action = 1  # buy
        elif action_val < -0.3:
            discrete_action = -1  # sell
        else:
            discrete_action = 0  # hold
            
        print(f"   🎯 Acción discreta: {discrete_action}")
        
        return discrete_action
        
    except Exception as e:
        print(f"   ❌ Error obteniendo acción: {e}")
        return 0

def test_trading_simulation(agent, simulator):
    """Probar simulación completa de trading"""
    print("\n📈 Probando simulación completa...")
    
    if agent is None:
        print("   ❌ No hay agente para probar")
        return {}
    
    try:
        metrics = simulator.simulate_trading(agent, num_episodes=1)
        print(f"   ✅ Simulación exitosa")
        print(f"   📊 Métricas:")
        for key, value in metrics.items():
            print(f"      {key}: {value:.6f}")
        
        return metrics
        
    except Exception as e:
        print(f"   ❌ Error en simulación: {e}")
        return {}

def test_manual_trading_loop(agent, simulator):
    """Probar loop de trading manual paso a paso"""
    print("\n🔄 Probando loop manual de trading...")
    
    if agent is None:
        print("   ❌ No hay agente para probar")
        return
    
    returns = []
    position = 0
    
    # Probar solo los primeros 10 pasos
    num_steps = min(10, len(simulator.features) - 1)
    
    for i in range(num_steps):
        current_state = simulator.features[i]
        current_price = simulator.prices[i]
        next_price = simulator.prices[i + 1]
        
        # Obtener acción
        try:
            action = agent.act(current_state)
            
            # Convertir a acción discreta
            if isinstance(action, (list, np.ndarray)):
                action_val = action[0] if len(action) > 0 else 0
            else:
                action_val = action
                
            if action_val > 0.3:
                discrete_action = 1  # buy
            elif action_val < -0.3:
                discrete_action = -1  # sell
            else:
                discrete_action = 0  # hold
                
            # Simular trade simple
            trade_return = 0.0
            if discrete_action == 1 and position <= 0:  # Buy
                trade_return = (next_price - current_price) / current_price
                position = 1
            elif discrete_action == -1 and position >= 0:  # Sell
                trade_return = (current_price - next_price) / current_price
                position = -1
            elif position == 1:  # Hold long
                trade_return = (next_price - current_price) / current_price
            elif position == -1:  # Hold short
                trade_return = (current_price - next_price) / current_price
            
            returns.append(trade_return)
            
            print(f"   Paso {i}: Precio {current_price:.2f} -> {next_price:.2f}, "
                  f"Acción {discrete_action}, Return {trade_return:.6f}")
                  
        except Exception as e:
            print(f"   ❌ Error en paso {i}: {e}")
            returns.append(0.0)
    
    # Calcular métricas simples
    if returns:
        total_return = np.sum(returns)
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        print(f"   📊 Resumen de {len(returns)} trades:")
        print(f"      Total Return: {total_return:.6f}")
        print(f"      Mean Return: {mean_return:.6f}")
        print(f"      Volatility: {volatility:.6f}")
        print(f"      Sharpe Ratio: {sharpe_ratio:.6f}")

def main():
    """Función principal de debug"""
    print("🔍 DEBUG: Optimización de Agentes RL")
    print("=" * 50)
    
    # 1. Probar datos sintéticos
    data = test_trading_data()
    
    # 2. Probar simulador
    simulator = test_simulator_setup(data)
    
    # 3. Probar creación de agente
    agent = test_agent_creation()
    
    # 4. Probar acción del agente
    test_agent_action(agent, simulator)
    
    # 5. Probar simulación completa
    metrics = test_trading_simulation(agent, simulator)
    
    # 6. Probar loop manual
    test_manual_trading_loop(agent, simulator)
    
    print("\n" + "=" * 50)
    print("🎯 DEBUG completado")

if __name__ == "__main__":
    main()
