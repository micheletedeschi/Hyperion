#!/usr/bin/env python3
"""
Debug script for tracing the exact flow during Optuna RL optimization
This will help identify where the 0.0 Sharpe ratios are coming from
"""

import sys
import os
import traceback
import logging

# Add project root to path
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

def setup_logging():
    """Setup detailed logging to trace the flow"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('debug_optuna_flow.log')
        ]
    )
    return logging.getLogger(__name__)

def test_individual_components():
    """Test each component individually"""
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("🔍 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 80)
    
    # Test 1: Import and create trading simulator
    try:
        from utils.trading_rl_optimizer import create_synthetic_trading_data, TradingEnvironmentSimulator
        print("✅ Trading simulator imports successful")
        
        trading_data = create_synthetic_trading_data(50)
        simulator = TradingEnvironmentSimulator(trading_data)
        print(f"✅ Trading simulator created with data shape: {trading_data.shape}")
        
    except Exception as e:
        print(f"❌ Trading simulator error: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Import and create RL agents
    agents_created = {}
    
    # Test SAC
    try:
        from hyperion3.models.rl_agents.sac import SACAgent
        state_dim = trading_data.shape[1]
        
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=3,
            hidden_dims=[128, 128],
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            batch_size=64,
            replay_buffer_size=10000
        )
        agents_created['SAC'] = agent
        print(f"✅ SAC agent created successfully")
        
    except Exception as e:
        print(f"❌ SAC agent creation error: {e}")
        traceback.print_exc()
    
    # Test TD3
    try:
        from hyperion3.models.rl_agents.td3 import TD3TradingAgent as TD3
        
        config = {
            'gamma': 0.99,
            'tau': 0.005,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_delay': 2,
            'batch_size': 64,
            'learning_rate': 0.001
        }
        
        agent = TD3(
            state_dim=state_dim,
            action_dim=1,
            max_action=1.0,
            config=config
        )
        agents_created['TD3'] = agent
        print(f"✅ TD3 agent created successfully")
        
    except Exception as e:
        print(f"❌ TD3 agent creation error: {e}")
        traceback.print_exc()
    
    # Test RainbowDQN
    try:
        from hyperion3.models.rl_agents.rainbow_dqn import RainbowTradingAgent as RainbowDQN
        
        config = {
            'batch_size': 64,
            'gamma': 0.99,
            'learning_rate': 0.001,
            'tau': 0.005,
            'hidden_size': 256,
            'num_atoms': 51,
            'v_min': -10,
            'v_max': 10
        }
        
        agent = RainbowDQN(
            state_dim=state_dim,
            action_dim=5,
            config=config
        )
        agents_created['RainbowDQN'] = agent
        print(f"✅ RainbowDQN agent created successfully")
        
    except Exception as e:
        print(f"❌ RainbowDQN agent creation error: {e}")
        traceback.print_exc()
    
    # Test 3: Test direct simulation with each agent
    print("\n🏦 TESTING DIRECT SIMULATION")
    print("-" * 40)
    
    for agent_name, agent in agents_created.items():
        try:
            print(f"\n🎯 Testing {agent_name} simulation...")
            
            # Direct simulation call
            metrics = simulator.simulate_trading(agent, num_episodes=1)
            sharpe = metrics.get('sharpe_ratio', -999)
            
            print(f"   📊 Raw metrics: {metrics}")
            print(f"   📈 Sharpe ratio: {sharpe}")
            print(f"   📊 Metrics type: {type(metrics)}")
            print(f"   📈 Sharpe type: {type(sharpe)}")
            
            if sharpe != -999 and sharpe != 0.0:
                print(f"   ✅ {agent_name}: Working - got Sharpe {sharpe}")
            else:
                print(f"   ❌ {agent_name}: Problem - got Sharpe {sharpe}")
                
        except Exception as e:
            print(f"   ❌ {agent_name} simulation error: {e}")
            traceback.print_exc()
    
    return len(agents_created) > 0, simulator, agents_created

def test_optuna_objective_functions():
    """Test the individual Optuna objective functions"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 80)
    print("🎯 TESTING OPTUNA OBJECTIVE FUNCTIONS")
    print("=" * 80)
    
    # Import Optuna
    try:
        import optuna
        print("✅ Optuna imported successfully")
    except ImportError:
        print("❌ Optuna not available")
        return False
    
    # Get trading data and simulator
    from utils.trading_rl_optimizer import create_synthetic_trading_data, TradingEnvironmentSimulator
    trading_data = create_synthetic_trading_data(50)
    simulator = TradingEnvironmentSimulator(trading_data)
    state_dim = trading_data.shape[1]
    
    # Create a mock trial for testing
    def create_mock_trial():
        """Create a mock trial object for testing"""
        class MockTrial:
            def __init__(self):
                self.number = 0
                
            def suggest_categorical(self, name, choices):
                if name == 'hidden_dims':
                    return [128, 128]
                elif name == 'activation':
                    return 'relu'
                elif name == 'batch_size':
                    return 64
                elif name == 'actor_hidden':
                    return [256, 256]
                elif name == 'critic_hidden':
                    return [256, 256]
                else:
                    return choices[0]
                    
            def suggest_float(self, name, low, high, log=False):
                if name == 'gamma':
                    return 0.99
                elif name == 'tau':
                    return 0.005
                elif name == 'alpha':
                    return 0.2
                elif name == 'learning_rate':
                    return 0.001
                elif name == 'risk_factor':
                    return 0.05
                elif name == 'reward_scaling':
                    return 1.0
                elif name == 'exploration_noise':
                    return 0.1
                elif name == 'policy_noise':
                    return 0.2
                elif name == 'noise_clip':
                    return 0.5
                elif name == 'action_noise':
                    return 0.2
                elif name == 'max_action':
                    return 1.0
                elif name == 'epsilon_start':
                    return 1.0
                elif name == 'epsilon_end':
                    return 0.01
                else:
                    return (low + high) / 2
                    
            def suggest_int(self, name, low, high):
                if name == 'replay_buffer_size':
                    return 10000
                elif name == 'buffer_size':
                    return 10000
                elif name == 'policy_delay':
                    return 2
                elif name == 'num_layers':
                    return 2
                elif name == 'n_steps':
                    return 1
                elif name == 'epsilon_decay':
                    return 1000
                else:
                    return (low + high) // 2
                    
        return MockTrial()
    
    # Test SAC objective function
    print("\n🎭 Testing SAC objective function...")
    try:
        from hyperion3.models.rl_agents.sac import SACAgent
        
        def test_sac_objective(trial):
            params = {
                'hidden_dims': trial.suggest_categorical('hidden_dims', [[128, 128], [256, 256]]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'tau': trial.suggest_float('tau', 0.001, 0.02),
                'alpha': trial.suggest_float('alpha', 0.05, 0.3),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
                'replay_buffer_size': trial.suggest_int('replay_buffer_size', 10000, 100000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'risk_factor': trial.suggest_float('risk_factor', 0.01, 0.1),
                'reward_scaling': trial.suggest_float('reward_scaling', 0.1, 2.0),
                'exploration_noise': trial.suggest_float('exploration_noise', 0.05, 0.3)
            }
            
            print(f"   📊 SAC params: {params}")
            
            # Create agent
            agent = SACAgent(
                state_dim=state_dim,
                action_dim=3,
                hidden_dims=params['hidden_dims'],
                gamma=params['gamma'],
                tau=params['tau'],
                alpha=params['alpha'],
                batch_size=params['batch_size'],
                replay_buffer_size=params['replay_buffer_size']
            )
            
            print(f"   ✅ SAC agent created")
            
            # Simulate
            metrics = simulator.simulate_trading(agent, num_episodes=1)
            sharpe_value = metrics.get('sharpe_ratio', -999)
            
            print(f"   📊 SAC Simulation metrics: {metrics}")
            print(f"   📈 SAC Sharpe value: {sharpe_value}")
            print(f"   📈 SAC Sharpe type: {type(sharpe_value)}")
            
            return sharpe_value
        
        mock_trial = create_mock_trial()
        sac_result = test_sac_objective(mock_trial)
        print(f"   📈 SAC objective result: {sac_result}")
        
        if sac_result != 0.0 and sac_result != -999:
            print(f"   ✅ SAC objective function working - returned {sac_result}")
        else:
            print(f"   ❌ SAC objective function problem - returned {sac_result}")
            
    except Exception as e:
        print(f"   ❌ SAC objective function error: {e}")
        traceback.print_exc()
    
    # Test TD3 objective function
    print("\n🎯 Testing TD3 objective function...")
    try:
        from hyperion3.models.rl_agents.td3 import TD3TradingAgent as TD3
        
        def test_td3_objective(trial):
            params = {
                'actor_hidden': trial.suggest_categorical('actor_hidden', [[256, 256], [400, 300]]),
                'critic_hidden': trial.suggest_categorical('critic_hidden', [[256, 256], [400, 300]]),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'tau': trial.suggest_float('tau', 0.001, 0.01),
                'policy_noise': trial.suggest_float('policy_noise', 0.1, 0.3),
                'noise_clip': trial.suggest_float('noise_clip', 0.3, 0.7),
                'policy_delay': trial.suggest_int('policy_delay', 1, 4),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'buffer_size': trial.suggest_int('buffer_size', 10000, 100000),
                'exploration_noise': trial.suggest_float('exploration_noise', 0.05, 0.2),
                'action_noise': trial.suggest_float('action_noise', 0.1, 0.3),
                'max_action': trial.suggest_float('max_action', 0.5, 2.0)
            }
            
            print(f"   📊 TD3 params: {params}")
            
            config = {
                'gamma': params['gamma'],
                'tau': params['tau'],
                'policy_noise': params['policy_noise'],
                'noise_clip': params['noise_clip'],
                'policy_delay': params['policy_delay'],
                'batch_size': params['batch_size'],
                'learning_rate': params['learning_rate']
            }
            
            agent = TD3(
                state_dim=state_dim,
                action_dim=1,
                max_action=params['max_action'],
                config=config
            )
            
            print(f"   ✅ TD3 agent created")
            
            metrics = simulator.simulate_trading(agent, num_episodes=1)
            sharpe_value = metrics.get('sharpe_ratio', -999)
            
            print(f"   📊 TD3 Simulation metrics: {metrics}")
            print(f"   📈 TD3 Sharpe value: {sharpe_value}")
            
            return sharpe_value
        
        mock_trial = create_mock_trial()
        td3_result = test_td3_objective(mock_trial)
        print(f"   📈 TD3 objective result: {td3_result}")
        
        if td3_result != 0.0 and td3_result != -999:
            print(f"   ✅ TD3 objective function working - returned {td3_result}")
        else:
            print(f"   ❌ TD3 objective function problem - returned {td3_result}")
            
    except Exception as e:
        print(f"   ❌ TD3 objective function error: {e}")
        traceback.print_exc()

def test_actual_optuna_optimization():
    """Test the actual Optuna optimization with minimal trials"""
    print("\n" + "=" * 80)
    print("🔬 TESTING ACTUAL OPTUNA OPTIMIZATION")
    print("=" * 80)
    
    try:
        import optuna
        from optuna.samplers import TPESampler
        from utils.trading_rl_optimizer import create_synthetic_trading_data, TradingEnvironmentSimulator
        from hyperion3.models.rl_agents.sac import SACAgent
        
        # Create trading environment
        trading_data = create_synthetic_trading_data(50)
        simulator = TradingEnvironmentSimulator(trading_data)
        state_dim = trading_data.shape[1]
        
        def sac_objective(trial):
            print(f"\n   🔄 Optuna Trial {trial.number}")
            
            params = {
                'hidden_dims': trial.suggest_categorical('hidden_dims', [[128, 128], [256, 256]]),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'tau': trial.suggest_float('tau', 0.001, 0.02),
                'alpha': trial.suggest_float('alpha', 0.05, 0.3),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
                'replay_buffer_size': trial.suggest_int('replay_buffer_size', 10000, 50000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            }
            
            print(f"   📊 Trial params: {params}")
            
            try:
                # Create agent
                agent = SACAgent(
                    state_dim=state_dim,
                    action_dim=3,
                    hidden_dims=params['hidden_dims'],
                    gamma=params['gamma'],
                    tau=params['tau'],
                    alpha=params['alpha'],
                    batch_size=params['batch_size'],
                    replay_buffer_size=params['replay_buffer_size']
                )
                
                print(f"   ✅ Agent created successfully")
                
                # Simulate trading
                metrics = simulator.simulate_trading(agent, num_episodes=1)
                sharpe_value = metrics.get('sharpe_ratio', -999)
                
                print(f"   📊 Metrics: {metrics}")
                print(f"   📈 Sharpe: {sharpe_value} (type: {type(sharpe_value)})")
                
                # Make sure we return a valid number
                if sharpe_value == -999 or sharpe_value is None:
                    print(f"   ⚠️ Invalid Sharpe, returning -1")
                    return -1.0
                
                result = float(sharpe_value)
                print(f"   📤 Returning: {result}")
                return result
                
            except Exception as e:
                print(f"   ❌ Trial error: {e}")
                traceback.print_exc()
                return -999.0
        
        print("\n🚀 Running Optuna optimization with 3 trials...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(sac_objective, n_trials=3, show_progress_bar=False)
        
        print(f"\n📊 OPTUNA RESULTS:")
        print(f"   🏆 Best value: {study.best_value}")
        print(f"   📋 Best params: {study.best_params}")
        print(f"   📊 All trials:")
        
        for trial in study.trials:
            print(f"      Trial {trial.number}: value={trial.value}, state={trial.state}")
        
        # Check if all values are 0.0
        all_values = [trial.value for trial in study.trials if trial.value is not None]
        if all(v == 0.0 for v in all_values):
            print(f"   ❌ All trial values are 0.0 - this is the bug!")
        elif all(v == -999.0 for v in all_values):
            print(f"   ❌ All trial values are -999.0 - error in objective function")
        else:
            print(f"   ✅ Got varying trial values: {all_values}")
            
    except Exception as e:
        print(f"❌ Optuna optimization test error: {e}")
        traceback.print_exc()

def main():
    """Main debug function"""
    logger = setup_logging()
    
    print("🔍 HYPERION RL OPTIMIZATION DEBUG")
    print("=" * 80)
    print("This script will trace the exact flow during Optuna RL optimization")
    print("to identify where the 0.0 Sharpe ratios are coming from.")
    print("=" * 80)
    
    # Test 1: Individual components
    success, simulator, agents = test_individual_components()
    if not success:
        print("❌ Individual components failed - aborting")
        return
    
    # Test 2: Objective functions
    test_optuna_objective_functions()
    
    # Test 3: Actual Optuna optimization
    test_actual_optuna_optimization()
    
    print("\n" + "=" * 80)
    print("🔍 DEBUG COMPLETE")
    print("=" * 80)
    print("Check the output above to identify where the 0.0 values are coming from.")

if __name__ == "__main__":
    main()
