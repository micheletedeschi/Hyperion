#!/usr/bin/env python3
"""Test script to verify EnsembleAgent import and availability."""

import sys
import os

def test_ensemble_agent_import():
    """Test importing EnsembleAgent without runtime errors."""
    print("Testing EnsembleAgent import...")
    
    try:
        from hyperion3.models.rl_agents.ensemble_agent import EnsembleAgent
        print("✅ EnsembleAgent imported successfully!")
        
        # Check if the class can be inspected
        print(f"   Class name: {EnsembleAgent.__name__}")
        print(f"   Module: {EnsembleAgent.__module__}")
        
        # Check available methods
        methods = [method for method in dir(EnsembleAgent) if not method.startswith('_')]
        print(f"   Available public methods: {len(methods)}")
        print(f"   Methods: {', '.join(methods[:5])}{'...' if len(methods) > 5 else ''}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import EnsembleAgent: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing EnsembleAgent: {e}")
        return False

def test_env_config():
    """Test importing through env_config."""
    print("\nTesting env_config import status...")
    
    try:
        from utils.env_config import check_hyperion_models
        status = check_hyperion_models()
        
        if 'ensemble_agent' in status:
            if status['ensemble_agent']:
                print("✅ EnsembleAgent available through env_config")
            else:
                print("⚠️ EnsembleAgent marked as unavailable in env_config")
        else:
            print("⚠️ EnsembleAgent not found in env_config status")
            
        return status
        
    except Exception as e:
        print(f"❌ Error checking env_config: {e}")
        return {}

if __name__ == "__main__":
    print("=== EnsembleAgent Import Test ===")
    
    # Test direct import
    import_success = test_ensemble_agent_import()
    
    # Test through env_config
    config_status = test_env_config()
    
    print(f"\n=== Summary ===")
    print(f"Direct import: {'✅ Success' if import_success else '❌ Failed'}")
    print(f"Config status: {'✅ Available' if config_status.get('ensemble_agent', False) else '❌ Unavailable'}")
    
    if import_success:
        print("\n🎉 EnsembleAgent is ready to use!")
        sys.exit(0)
    else:
        print("\n❌ EnsembleAgent has import issues")
        sys.exit(1)
