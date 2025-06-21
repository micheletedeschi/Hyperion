#!/usr/bin/env python3
"""
Debug script to trace the exact flow of RL optimization 
and find where the Sharpe ratio gets lost
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.trading_rl_optimizer import create_synthetic_trading_data, TradingEnvironmentSimulator

# Try to import RL availability flags
try:
    from utils.env_config import RL_SAC_AVAILABLE, RL_TD3_AVAILABLE, RL_RAINBOW_AVAILABLE
except ImportError:
    RL_SAC_AVAILABLE = False
    RL_TD3_AVAILABLE = False
    RL_RAINBOW_AVAILABLE = False

def debug_sharpe_ratio_flow():
    """Debug the complete flow from simulator to optimization result"""
    
    print("🔍 DEBUG: Testing Sharpe Ratio Flow in RL Optimization")
    print("=" * 60)
    
    # 1. Test data creation
    print("\n1. 📊 Creating synthetic trading data...")
    trading_data = create_synthetic_trading_data(100)
    print(f"   ✅ Data shape: {trading_data.shape}")
    print(f"   ✅ Columns: {list(trading_data.columns)}")
    print(f"   ✅ Price range: {trading_data['close'].min():.2f} - {trading_data['close'].max():.2f}")
    
    # 2. Test simulator creation
    print("\n2. 🎮 Creating trading simulator...")
    simulator = TradingEnvironmentSimulator(trading_data)
    print(f"   ✅ Features shape: {simulator.features.shape}")
    print(f"   ✅ Initial balance: {simulator.initial_balance}")
    
    # 3. Create a simple mock agent for testing
    class MockAgent:
        """Simple mock agent for testing"""
        def __init__(self):
            self.action_count = 0
            
        def act(self, state):
            # Rotate between buy, hold, sell
            self.action_count += 1
            actions = [1, 0, -1]  # buy, hold, sell
            return actions[self.action_count % 3]
    
    print("\n3. 🤖 Testing with mock agent...")
    mock_agent = MockAgent()
    
    # 4. Test direct simulation
    print("\n4. 📈 Testing direct simulation...")
    metrics = simulator.simulate_trading(mock_agent, num_episodes=1)
    print(f"   ✅ Simulation complete!")
    print(f"   ✅ Returned metrics: {list(metrics.keys())}")
    for key, value in metrics.items():
        print(f"      📊 {key}: {value}")
    
    # 5. Test the specific Sharpe ratio value
    sharpe_ratio = metrics.get('sharpe_ratio', 'NOT_FOUND')
    print(f"\n5. 🎯 Sharpe Ratio Analysis:")
    print(f"   ✅ Sharpe ratio: {sharpe_ratio}")
    print(f"   ✅ Type: {type(sharpe_ratio)}")
    print(f"   ✅ Is zero: {sharpe_ratio == 0.0}")
    print(f"   ✅ Is nan: {np.isnan(sharpe_ratio) if isinstance(sharpe_ratio, (int, float)) else 'N/A'}")
    
    # 6. Test returns and volatility calculation manually
    print("\n6. 🔢 Manual calculation verification:")
    # Since we can't access private variables, let's run simulation again with detailed analysis
    
    # 7. Test with real RL agents if available
    print("\n7. 🧠 Testing with real RL agents...")
    
    # Test SAC agent if available
    if RL_SAC_AVAILABLE:
        print("\n   🎭 Testing SAC Agent...")
        try:
            from hyperion3.models.rl_agents.sac import SACAgent
            sac_agent = SACAgent(
                state_dim=simulator.features.shape[1],
                action_dim=3,
                hidden_dims=(128, 128)  # Use tuple instead of list
            )
            sac_metrics = simulator.simulate_trading(sac_agent, num_episodes=1)
            print(f"      ✅ SAC Sharpe ratio: {sac_metrics.get('sharpe_ratio', 'ERROR')}")
        except Exception as e:
            print(f"      ❌ SAC Error: {e}")
    
    # Test TD3 agent if available  
    if RL_TD3_AVAILABLE:
        print("\n   🎯 Testing TD3 Agent...")
        try:
            from hyperion3.models.rl_agents.td3 import TD3
            td3_config = {
                'gamma': 0.99,
                'tau': 0.005,
                'learning_rate': 0.001
            }
            td3_agent = TD3(
                state_dim=simulator.features.shape[1],
                action_dim=1,
                max_action=1.0,
                config=td3_config
            )
            td3_metrics = simulator.simulate_trading(td3_agent, num_episodes=1)
            print(f"      ✅ TD3 Sharpe ratio: {td3_metrics.get('sharpe_ratio', 'ERROR')}")
        except Exception as e:
            print(f"      ❌ TD3 Error: {e}")
    
    # Test Rainbow DQN agent if available
    if RL_RAINBOW_AVAILABLE:
        print("\n   🌈 Testing Rainbow DQN Agent...")
        try:
            from hyperion3.models.rl_agents.rainbow_dqn import RainbowDQN
            rainbow_config = {
                'batch_size': 64,
                'gamma': 0.99,
                'learning_rate': 0.001
            }
            rainbow_agent = RainbowDQN(
                state_dim=simulator.features.shape[1],
                action_dim=5,
                config=rainbow_config
            )
            rainbow_metrics = simulator.simulate_trading(rainbow_agent, num_episodes=1)
            print(f"      ✅ Rainbow Sharpe ratio: {rainbow_metrics.get('sharpe_ratio', 'ERROR')}")
        except Exception as e:
            print(f"      ❌ Rainbow Error: {e}")
    
    print("\n8. 🔍 Next Steps:")
    if sharpe_ratio == 0.0:
        print("   ⚠️  Sharpe ratio is exactly 0.0 - investigating cause...")
        print("   💡 Possible issues:")
        print("      - Mean return is 0 (no profit/loss)")
        print("      - Volatility calculation issue")
        print("      - Agent not making meaningful trades")
        print("      - Returns array is empty or all zeros")
    else:
        print("   ✅ Sharpe ratio calculation looks good!")
        print("   ✅ The issue might be in the Optuna optimization flow")
    
    return metrics

if __name__ == "__main__":
    metrics = debug_sharpe_ratio_flow()
