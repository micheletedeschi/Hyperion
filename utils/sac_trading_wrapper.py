#!/usr/bin/env python3
"""
SAC Agent Wrapper for Hyperparameter Optimization
Bridges the gap between the optimization framework and SAC agent's native environment
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class SACTradingWrapper:
    """Wrapper que permite usar SAC Agent en el framework de optimización"""
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dims: tuple,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 batch_size: int = 256,
                 replay_buffer_size: int = 100000):
        """
        Inicializar wrapper SAC para trading
        
        Args:
            state_dim: Dimensión del estado (será ajustada internamente)
            hidden_dims: Dimensiones de las capas ocultas
            gamma: Factor de descuento
            tau: Factor de actualización suave
            alpha: Coeficiente de entropía
            batch_size: Tamaño del batch
            replay_buffer_size: Tamaño del buffer de replay
        """
        try:
            from hyperion3.models.rl_agents.sac import SACAgent
            
            # Crear agente SAC con dimensiones apropiadas
            # SAC espera action_dim=1 para acciones continuas
            self.agent = SACAgent(
                state_dim=state_dim,
                action_dim=1,  # Acción continua para posición
                hidden_dims=hidden_dims,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                batch_size=batch_size,
                replay_buffer_size=replay_buffer_size
            )
            
            self.initialized = True
            
        except Exception as e:
            print(f"❌ Error inicializando SAC Agent: {e}")
            self.agent = None
            self.initialized = False
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Seleccionar acción usando el agente SAC
        
        Args:
            state: Estado del ambiente
            deterministic: Si usar selección determinística
            
        Returns:
            Acción continua entre -1 y 1
        """
        if not self.initialized:
            return np.array([0.0])  # Acción neutral por defecto
        
        try:
            # Asegurar que el estado tenga la forma correcta
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            # Si el estado es demasiado pequeño, rellenarlo
            expected_dim = self.agent.state_dim
            if state.shape[1] < expected_dim:
                # Pad con ceros
                padding = np.zeros((state.shape[0], expected_dim - state.shape[1]))
                state = np.concatenate([state, padding], axis=1)
            elif state.shape[1] > expected_dim:
                # Truncar
                state = state[:, :expected_dim]
            
            # Usar el agente SAC para obtener acción
            action = self.agent.act(state.flatten(), deterministic=deterministic)
            
            return action
            
        except Exception as e:
            print(f"⚠️ Warning en SACTradingWrapper.act: {e}")
            return np.array([0.0])  # Acción neutral por defecto
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Método compatible con interfaz de otros agentes"""
        return self.act(state, deterministic=not add_noise)
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Método compatible con interfaz de otros agentes"""
        return self.act(state, deterministic=False)


def create_sac_trading_agent(config: Dict[str, Any]) -> Optional[SACTradingWrapper]:
    """
    Crear agente SAC configurado para trading
    
    Args:
        config: Diccionario con configuración del agente
        
    Returns:
        SACTradingWrapper configurado o None si hay error
    """
    try:
        # Valores por defecto para configuración
        default_config = {
            'state_dim': 50,  # Dimensión por defecto
            'hidden_dims': (256, 256),
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'batch_size': 256,
            'replay_buffer_size': 100000
        }
        
        # Combinar con configuración proporcionada
        final_config = {**default_config, **config}
        
        # Crear wrapper
        wrapper = SACTradingWrapper(**final_config)
        
        if wrapper.initialized:
            return wrapper
        else:
            print("❌ Failed to initialize SAC trading wrapper")
            return None
            
    except Exception as e:
        print(f"❌ Error creating SAC trading agent: {e}")
        return None
