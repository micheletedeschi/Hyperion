#!/usr/bin/env python3
"""
Test para verificar los valores de las variables RL
"""

import sys
sys.path.append('/Users/giovanniarangio/carpeta sin título 2/Hyperion')

# Importar las variables directamente desde utils.hyperopt
from utils.hyperopt import (
    RL_SAC_AVAILABLE, RL_TD3_AVAILABLE, RL_RAINBOW_AVAILABLE, RL_AGENTS_AVAILABLE
)

print("🔍 Valores de las variables RL:")
print(f"  RL_SAC_AVAILABLE: {RL_SAC_AVAILABLE}")
print(f"  RL_TD3_AVAILABLE: {RL_TD3_AVAILABLE}")
print(f"  RL_RAINBOW_AVAILABLE: {RL_RAINBOW_AVAILABLE}")
print(f"  RL_AGENTS_AVAILABLE: {RL_AGENTS_AVAILABLE}")

# Probar la construcción de la lista
agents_list = [
    agent for agent, available in [
        ('SAC', RL_SAC_AVAILABLE),
        ('TD3', RL_TD3_AVAILABLE), 
        ('RainbowDQN', RL_RAINBOW_AVAILABLE)
    ] if available
]

print(f"\n🎯 Lista de agentes construida: {agents_list}")

# Probar las importaciones individuales
try:
    from hyperion3.models.rl_agents.sac import SACAgent
    print("✅ SACAgent se puede importar")
except Exception as e:
    print(f"❌ Error importando SACAgent: {e}")

try:
    from hyperion3.models.rl_agents.td3 import TD3TradingAgent
    print("✅ TD3TradingAgent se puede importar")
except Exception as e:
    print(f"❌ Error importando TD3TradingAgent: {e}")

try:
    from hyperion3.models.rl_agents.rainbow_dqn import RainbowTradingAgent
    print("✅ RainbowTradingAgent se puede importar")
except Exception as e:
    print(f"❌ Error importando RainbowTradingAgent: {e}")
