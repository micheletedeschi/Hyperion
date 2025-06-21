#!/usr/bin/env python3
"""
Test SAC Optimization Fix
Quick test to verify that SAC optimization now works correctly
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

def test_sac_optimization_fix():
    """Test the fixed SAC optimization"""
    logger.info("🔍 Testing Fixed SAC Optimization")
    
    try:
        # Import necessary modules
        from utils.trading_rl_optimizer import create_synthetic_trading_data, TradingEnvironmentSimulator
        from utils.sac_trading_wrapper import create_sac_trading_agent
        
        # Create synthetic trading data
        trading_data = create_synthetic_trading_data(100)
        logger.info(f"Created trading data with shape: {trading_data.shape}")
        
        # Create trading simulator
        simulator = TradingEnvironmentSimulator(trading_data)
        logger.info("Created trading simulator")
        
        # Create SAC agent configuration
        sac_config = {
            'state_dim': trading_data.shape[1],  # Match features from trading data
            'hidden_dims': (256, 256),
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 256,
            'replay_buffer_size': 100000
        }
        
        logger.info(f"SAC config: {sac_config}")
        
        # Create SAC agent using wrapper
        agent = create_sac_trading_agent(sac_config)
        
        if agent is None:
            logger.error("❌ Failed to create SAC agent")
            return False
        
        logger.info("✅ Created SAC agent successfully")
        
        # Test trading simulation
        logger.info("🎯 Running trading simulation...")
        metrics = simulator.simulate_trading(agent, num_episodes=1)
        
        logger.info("📊 Trading Simulation Results:")
        for key, value in metrics.items():
            logger.info(f"   {key}: {value}")
        
        # Check if we got meaningful results
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        total_return = metrics.get('total_return', 0)
        total_trades = metrics.get('total_trades', 0)
        
        logger.info(f"\n🎯 Key Metrics:")
        logger.info(f"   Sharpe Ratio: {sharpe_ratio:.6f}")
        logger.info(f"   Total Return: {total_return:.6f}")
        logger.info(f"   Total Trades: {total_trades}")
        
        # Check if SAC is now producing meaningful results
        if sharpe_ratio != 0.0 or total_return != 0.0 or total_trades > 0:
            logger.info("✅ SAC agent produced meaningful trading results!")
            return True
        else:
            logger.warning("⚠️ SAC agent still producing zero results")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_sac_action():
    """Test SAC agent action selection directly"""
    logger.info("\n🔍 Testing Direct SAC Action Selection")
    
    try:
        from utils.sac_trading_wrapper import create_sac_trading_agent
        
        # Create SAC agent
        sac_config = {
            'state_dim': 10,  # Simple state dimension
            'hidden_dims': (128, 128),
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 64,
            'replay_buffer_size': 10000
        }
        
        agent = create_sac_trading_agent(sac_config)
        
        if agent is None:
            logger.error("❌ Failed to create SAC agent")
            return False
        
        # Test action selection
        state = np.random.randn(10)  # Random state
        logger.info(f"Test state: {state[:5]}...")
        
        actions = []
        for i in range(5):
            action = agent.act(state, deterministic=False)
            actions.append(action[0])
            logger.info(f"Action {i+1}: {action[0]:.6f}")
        
        logger.info(f"Action stats: mean={np.mean(actions):.6f}, std={np.std(actions):.6f}")
        
        if np.std(actions) > 0:
            logger.info("✅ SAC agent producing varied actions")
            return True
        else:
            logger.warning("⚠️ SAC agent actions have no variance")
            return False
        
    except Exception as e:
        logger.error(f"❌ Direct action test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run SAC optimization fix tests"""
    logger.info("🚀 Starting SAC Optimization Fix Tests")
    
    success = True
    
    # Test 1: Direct SAC action selection
    if not test_direct_sac_action():
        success = False
    
    # Test 2: SAC optimization with trading simulator
    if not test_sac_optimization_fix():
        success = False
    
    if success:
        logger.info("\n✅ All SAC optimization fix tests passed!")
    else:
        logger.error("\n❌ Some SAC optimization fix tests failed")
    
    return success

if __name__ == "__main__":
    main()
