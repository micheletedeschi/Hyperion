#!/usr/bin/env python3
"""
Debug SAC Agent Action Selection
Focused debugging to understand why SAC returns 0.0 trading metrics
"""

import os
import sys
import numpy as np
import pandas as pd
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 100) -> pd.DataFrame:
    """Create synthetic market data for testing"""
    np.random.seed(42)
    
    # Create price series with trend and volatility
    returns = np.random.normal(0.001, 0.02, n_samples)  # 0.1% mean, 2% std daily returns
    prices = [100.0]  # Starting price
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = prices[1:]  # Remove initial price
    
    # Create feature columns
    data = {
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'sma_20': [np.mean(prices[max(0, i-19):i+1]) for i in range(n_samples)],
        'volatility': [np.std(prices[max(0, i-19):i+1]) if i >= 19 else 0 for i in range(n_samples)]
    }
    
    return pd.DataFrame(data)

def test_sac_action_selection():
    """Test SAC agent action selection in isolation"""
    logger.info("🔍 Testing SAC Agent Action Selection")
    
    try:
        from hyperion3.models.rl_agents.sac import SACAgent, TradingEnvironmentSAC
        
        # Create test data
        market_data = create_test_data(200)
        feature_columns = ['close', 'volume', 'rsi', 'sma_20', 'volatility']
        
        # Create environment
        env = TradingEnvironmentSAC(
            market_data=market_data,
            feature_columns=feature_columns,
            window_size=20
        )
        
        # Initialize agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=(256, 256),
            max_action=1.0
        )
        
        logger.info(f"Environment state dim: {state_dim}, action dim: {action_dim}")
        
        # Test action selection with different states
        state, _ = env.reset()
        logger.info(f"Initial state shape: {state.shape}")
        logger.info(f"Initial state sample: {state[:5]}...")
        
        # Test multiple action selections
        actions = []
        for i in range(10):
            action = agent.act(state, deterministic=False)
            actions.append(action[0])
            logger.info(f"Action {i+1}: {action[0]:.6f}")
            
            # Take a step to get new state
            if i < 9:  # Don't step on last iteration
                state, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    state, _ = env.reset()
        
        logger.info("\nAction statistics:")
        logger.info(f"Mean: {np.mean(actions):.6f}")
        logger.info(f"Std: {np.std(actions):.6f}")
        logger.info(f"Min: {np.min(actions):.6f}")
        logger.info(f"Max: {np.max(actions):.6f}")
        logger.info(f"All same?: {len(set(np.round(actions, 6))) == 1}")
        
        # Test deterministic vs stochastic
        logger.info("\n🎯 Testing deterministic vs stochastic actions:")
        state, _ = env.reset()
        
        det_actions = []
        stoch_actions = []
        
        for i in range(5):
            det_action = agent.act(state, deterministic=True)
            stoch_action = agent.act(state, deterministic=False)
            
            det_actions.append(det_action[0])
            stoch_actions.append(stoch_action[0])
            
            logger.info(f"Step {i+1} - Deterministic: {det_action[0]:.6f}, Stochastic: {stoch_action[0]:.6f}")
        
        logger.info(f"Deterministic actions std: {np.std(det_actions):.6f}")
        logger.info(f"Stochastic actions std: {np.std(stoch_actions):.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SAC action selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_td3_action_selection():
    """Test TD3 agent action selection for comparison"""
    logger.info("\n🔍 Testing TD3 Agent Action Selection (for comparison)")
    
    try:
        from hyperion3.models.rl_agents.td3 import TD3TradingAgent
        
        # Create test data
        market_data = create_test_data(200)
        feature_columns = ['close', 'volume', 'rsi', 'sma_20', 'volatility']
        
        # Initialize agent
        config = {
            'window_size': 20,
            'gamma': 0.99,
            'tau': 0.005,
            'exploration_noise': 0.1,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_freq': 2
        }
        
        agent = TD3TradingAgent(
            market_data=market_data,
            feature_columns=feature_columns,
            config=config
        )
        
        # Test action selection
        state = agent.env.reset()[0]
        logger.info(f"TD3 state shape: {state.shape}")
        
        # Test multiple action selections
        actions = []
        for i in range(10):
            action = agent.td3.select_action(state, add_noise=True)
            actions.append(action[0])
            logger.info(f"TD3 Action {i+1}: {action[0]:.6f}")
            
            # Take a step
            if i < 9:
                state, reward, done, truncated, info = agent.env.step(action)
                if done or truncated:
                    state = agent.env.reset()[0]
        
        logger.info("\nTD3 Action statistics:")
        logger.info(f"Mean: {np.mean(actions):.6f}")
        logger.info(f"Std: {np.std(actions):.6f}")
        logger.info(f"Min: {np.min(actions):.6f}")
        logger.info(f"Max: {np.max(actions):.6f}")
        logger.info(f"All same?: {len(set(np.round(actions, 6))) == 1}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TD3 action selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_trading_simulation():
    """Test manual trading simulation with SAC to see where issues occur"""
    logger.info("\n🔍 Testing Manual Trading Simulation with SAC")
    
    try:
        from hyperion3.models.rl_agents.sac import SACAgent, TradingEnvironmentSAC
        
        # Create test data
        market_data = create_test_data(100)
        feature_columns = ['close', 'volume', 'rsi', 'sma_20', 'volatility']
        
        # Create environment
        env = TradingEnvironmentSAC(
            market_data=market_data,
            feature_columns=feature_columns,
            window_size=20
        )
        
        # Initialize agent
        agent = SACAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dims=(256, 256)
        )
        
        # Run simulation
        state, info = env.reset()
        logger.info(f"Initial portfolio: balance={env.balance:.2f}, position={env.position:.3f}, value={env.total_value:.2f}")
        
        total_reward = 0
        trades_made = 0
        positions = []
        values = []
        
        for step in range(50):  # Run for 50 steps
            action = agent.act(state, deterministic=False)
            prev_position = env.position
            
            state, reward, done, truncated, info = env.step(action)
            
            if abs(action[0] - prev_position) > 0.01:  # Trade was made
                trades_made += 1
                logger.info(f"Step {step}: Trade made - action={action[0]:.3f}, prev_pos={prev_position:.3f}, new_pos={env.position:.3f}")
            
            positions.append(env.position)
            values.append(env.total_value)
            total_reward += reward
            
            if step % 10 == 0:
                logger.info(f"Step {step}: action={action[0]:.3f}, pos={env.position:.3f}, value={env.total_value:.2f}, reward={reward:.6f}")
            
            if done or truncated:
                break
        
        # Calculate final metrics
        returns = [(values[i] - values[0]) / values[0] for i in range(len(values))]
        final_return = (env.total_value - env.initial_balance) / env.initial_balance
        
        logger.info("\n📊 Trading Simulation Results:")
        logger.info(f"Total steps: {step + 1}")
        logger.info(f"Trades made: {trades_made}")
        logger.info(f"Final portfolio value: {env.total_value:.2f}")
        logger.info(f"Total return: {final_return:.6f}")
        logger.info(f"Total reward: {total_reward:.6f}")
        logger.info(f"Position range: [{np.min(positions):.3f}, {np.max(positions):.3f}]")
        logger.info(f"Position std: {np.std(positions):.6f}")
        
        # Calculate Sharpe ratio manually
        if len(returns) > 1:
            returns_array = np.array(returns[1:])  # Skip first return (always 0)
            returns_diff = np.diff(returns_array) if len(returns_array) > 1 else returns_array
            
            if len(returns_diff) > 0 and np.std(returns_diff) > 0:
                sharpe = np.mean(returns_diff) / np.std(returns_diff)
                logger.info(f"Manual Sharpe ratio: {sharpe:.6f}")
            else:
                logger.info("Manual Sharpe ratio: 0.0 (no variance)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Manual trading simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all SAC debugging tests"""
    logger.info("🚀 Starting SAC Agent Debugging")
    
    success = True
    
    # Test 1: SAC action selection
    if not test_sac_action_selection():
        success = False
    
    # Test 2: TD3 action selection for comparison
    if not test_td3_action_selection():
        success = False
    
    # Test 3: Manual trading simulation
    if not test_manual_trading_simulation():
        success = False
    
    if success:
        logger.info("\n✅ All SAC debugging tests completed successfully")
    else:
        logger.error("\n❌ Some SAC debugging tests failed")
    
    return success

if __name__ == "__main__":
    main()
