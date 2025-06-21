#!/usr/bin/env python3
"""
Debug script to isolate the Optuna RL optimization issue
"""

import sys
import os
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    print("✅ Optuna disponible")
except ImportError as e:
    OPTUNA_AVAILABLE = False
    print(f"❌ Optuna no disponible: {e}")

# Try to import our RL stuff
try:
    from utils.trading_rl_optimizer import TradingRLSimulator
    RL_TRADING_SIMULATOR_AVAILABLE = True
    print("✅ Trading simulator disponible")
except ImportError as e:
    RL_TRADING_SIMULATOR_AVAILABLE = False
    print(f"❌ Trading simulator no disponible: {e}")

try:
    from hyperion3.models.rl_agents.sac_agent import SAC
    RL_SAC_AVAILABLE = True
    print("✅ SAC Agent disponible")
except ImportError as e:
    RL_SAC_AVAILABLE = False
    print(f"❌ SAC Agent no disponible: {e}")

def test_direct_simulator():
    """Test the simulator directly without Optuna"""
    print("\n🔍 Testing simulator directly...")
    
    if not RL_TRADING_SIMULATOR_AVAILABLE:
        print("❌ No simulator available")
        return
    
    # Create simulator
    np.random.seed(42)
    trading_data = np.random.randn(1000, 10)  # 1000 timesteps, 10 features
    
    simulator = TradingRLSimulator(
        data=trading_data,
        initial_balance=10000,
        transaction_cost=0.001
    )
    
    if RL_SAC_AVAILABLE:
        # Create a simple SAC agent
        agent = SAC(
            state_dim=10,
            action_dim=1,
            max_action=1.0,
            config={'gamma': 0.99, 'alpha': 0.2, 'learning_rate': 0.0003}
        )
        
        # Test simulation
        metrics = simulator.simulate_trading(agent, num_episodes=1)
        print(f"📊 Direct simulation metrics: {metrics}")
        print(f"📈 Direct Sharpe ratio: {metrics.get('sharpe_ratio', 'N/A')}")
        return metrics.get('sharpe_ratio', 0.0)
    else:
        print("❌ No SAC agent available for testing")
        return 0.0

def test_optuna_simple():
    """Test simple Optuna optimization without RL complexity"""
    print("\n🔍 Testing simple Optuna optimization...")
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna not available")
        return
    
    def simple_objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        result = -(x**2 + y**2)  # Should maximize at (0,0) with value 0
        print(f"   Trial {trial.number}: x={x:.3f}, y={y:.3f}, result={result:.3f}")
        return result
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(simple_objective, n_trials=5, show_progress_bar=False)
    
    print(f"📈 Best value: {study.best_value}")
    print(f"📊 Best params: {study.best_params}")

def test_optuna_with_simulator():
    """Test Optuna with our simulator to isolate the issue"""
    print("\n🔍 Testing Optuna with trading simulator...")
    
    if not (OPTUNA_AVAILABLE and RL_TRADING_SIMULATOR_AVAILABLE and RL_SAC_AVAILABLE):
        print("❌ Missing dependencies")
        return
    
    # Create synthetic trading data
    np.random.seed(42)
    trading_data = np.random.randn(1000, 10)
    
    simulator = TradingRLSimulator(
        data=trading_data,
        initial_balance=10000,
        transaction_cost=0.001
    )
    
    def rl_objective(trial):
        # Simple parameter space
        gamma = trial.suggest_float('gamma', 0.95, 0.99)
        alpha = trial.suggest_float('alpha', 0.1, 0.3)
        lr = trial.suggest_float('learning_rate', 0.0001, 0.001, log=True)
        
        try:
            agent = SAC(
                state_dim=10,
                action_dim=1,
                max_action=1.0,
                config={'gamma': gamma, 'alpha': alpha, 'learning_rate': lr}
            )
            
            # Simulate trading
            metrics = simulator.simulate_trading(agent, num_episodes=1)
            sharpe = metrics.get('sharpe_ratio', -999)
            
            print(f"   🎯 Trial {trial.number}:")
            print(f"      📊 Gamma: {gamma:.3f}, Alpha: {alpha:.3f}, LR: {lr:.6f}")
            print(f"      📈 Sharpe ratio: {sharpe}")
            print(f"      📋 All metrics: {metrics}")
            
            # Return Sharpe ratio
            return sharpe
            
        except Exception as e:
            print(f"   ❌ Trial {trial.number} failed: {e}")
            return -999
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(rl_objective, n_trials=3, show_progress_bar=False)
    
    print(f"\n📈 Optuna Best Value: {study.best_value}")
    print(f"📊 Optuna Best Params: {study.best_params}")
    
    # Also test a trial manually
    print(f"\n🔍 Manual trial test:")
    trial = study.trials[0] if study.trials else None
    if trial:
        print(f"   Trial 0 value: {trial.value}")
        print(f"   Trial 0 params: {trial.params}")

if __name__ == "__main__":
    print("🚀 Debugging Optuna RL Optimization Issue")
    print("=" * 50)
    
    # Test 1: Direct simulator
    direct_sharpe = test_direct_simulator()
    
    # Test 2: Simple Optuna
    test_optuna_simple()
    
    # Test 3: Optuna with simulator
    test_optuna_with_simulator()
    
    print("\n" + "=" * 50)
    print("🏁 Debug complete")
