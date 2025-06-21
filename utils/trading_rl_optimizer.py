"""
Trading environment simulator for RL agent hyperparameter optimization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class TradingEnvironmentSimulator:
    """Simulador de ambiente de trading para optimización de hiperparámetros de RL agents"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        """
        Args:
            data: DataFrame con columnas ['open', 'high', 'low', 'close', 'volume']
            initial_balance: Balance inicial para trading
        """
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for neutral
        self.trades = []
        self.returns = []
        
        # Preparar features técnicos
        self._prepare_features()
        
    def _prepare_features(self):
        """Preparar features técnicos para el trading"""
        data = self.data.copy()
        
        # Indicators técnicos básicos
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['rsi'] = self._calculate_rsi(data['close'])
        data['macd'] = self._calculate_macd(data['close'])
        data['bb_upper'], data['bb_lower'] = self._calculate_bollinger_bands(data['close'])
        
        # Price features
        data['price_change'] = data['close'].pct_change()
        data['volatility'] = data['price_change'].rolling(20).std()
        data['volume_ma'] = data['volume'].rolling(20).mean()
        
        # Features normalizados
        scaler = StandardScaler()
        feature_cols = ['sma_20', 'sma_50', 'rsi', 'macd', 'bb_upper', 'bb_lower', 
                       'price_change', 'volatility', 'volume_ma']
        
        # Llenar NaN con forward fill y backward fill
        data[feature_cols] = data[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Normalizar features
        data[feature_cols] = scaler.fit_transform(data[feature_cols])
        
        self.features = data[feature_cols].values
        self.prices = data['close'].values
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calcular MACD"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calcular Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def simulate_trading(self, agent, num_episodes: int = 1) -> Dict[str, float]:
        """
        Simular trading con un agente RL de manera robusta
        
        Args:
            agent: Agente RL (puede ser de diferentes tipos)
            num_episodes: Número de episodios de trading
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        total_returns = []
        all_trades = []
        
        for episode in range(num_episodes):
            episode_returns = self._run_single_episode_robust(agent)
            total_returns.extend(episode_returns)
            all_trades.extend(self.trades)
            
        # Calcular métricas
        metrics = self._calculate_trading_metrics(total_returns, all_trades)
        return metrics
    
    def _run_single_episode_robust(self, agent) -> List[float]:
        """Ejecutar un episodio de trading de manera robusta"""
        self.current_balance = self.initial_balance
        self.position = 0
        self.trades = []
        episode_returns = []
        
        # Simular secuencia de trading
        for i in range(len(self.features) - 1):
            current_state = self.features[i]
            current_price = self.prices[i]
            next_price = self.prices[i + 1]
            
            # Obtener acción del agente de manera robusta
            action = self._get_agent_action(agent, current_state, i)
            
            # Ejecutar trade
            trade_return = self._execute_trade(action, current_price, next_price)
            episode_returns.append(trade_return)
            
        return episode_returns
    
    def _get_agent_action(self, agent, state, step_idx: int):
        """Obtener acción del agente de manera robusta para diferentes tipos"""
        try:
            # Intentar diferentes métodos de acción comunes
            if hasattr(agent, 'act'):
                action = agent.act(state)
            elif hasattr(agent, 'select_action'):
                action = agent.select_action(state)
            elif hasattr(agent, 'get_action'):
                action = agent.get_action(state)
            elif hasattr(agent, 'predict'):
                action = agent.predict(state.reshape(1, -1))[0]
            elif hasattr(agent, 'forward'):
                # Para modelos PyTorch
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = agent.forward(state_tensor).cpu().numpy()[0]
            else:
                # Fallback: estrategia simple basada en features
                action = self._simple_strategy_action(state)
                
            # Normalizar acción según el tipo
            if isinstance(action, (list, np.ndarray)):
                action = action[0] if len(action) > 0 else 0
                
            # Manejo especial para agentes SAC (acciones continuas)
            from hyperion3.models.rl_agents.sac import SACAgent
            if isinstance(agent, SACAgent):
                # SAC produce acciones continuas entre -1 y 1
                # Mapear directamente a acciones de trading
                if isinstance(action, (int, np.integer)):
                    action = float(action)
                
                # Convertir acción continua SAC a acción discreta de trading
                if action > 0.1:  # Umbral más bajo para más sensibilidad
                    return 1  # buy
                elif action < -0.1:
                    return -1  # sell
                else:
                    return 0  # hold
            else:
                # Para otros agentes, convertir a acción discreta
                if isinstance(action, (int, np.integer)):
                    return np.clip(action, -1, 1)
                else:
                    # Convertir acción continua a discreta
                    if action > 0.3:
                        return 1  # buy
                    elif action < -0.3:
                        return -1  # sell
                    else:
                        return 0  # hold
                    
        except Exception as e:
            # En caso de error, usar acción conservadora
            print(f"Warning: Error getting agent action: {e}")
            return 0  # hold
    
    def _simple_strategy_action(self, state) -> int:
        """Estrategia simple como fallback"""
        try:
            # Usar algunos features para decisión simple
            if len(state) >= 3:
                price_momentum = state[0]  # Asumiendo que el primer feature es momentum
                sma_signal = state[1] if len(state) > 1 else 0
                
                if price_momentum > 0.01 and sma_signal > 0:
                    return 1  # buy
                elif price_momentum < -0.01 and sma_signal < 0:
                    return -1  # sell
                else:
                    return 0  # hold
            else:
                return 0
        except Exception:
            return 0
    
    def _execute_trade(self, action: int, current_price: float, next_price: float) -> float:
        """Ejecutar una operación de trading"""
        trade_return = 0.0
        
        # Simplificado: action = 1 (buy), -1 (sell), 0 (hold)
        if action == 1 and self.position <= 0:  # Buy signal
            if self.position == -1:  # Close short position
                trade_return = (current_price - next_price) / current_price
            self.position = 1
            trade_return += (next_price - current_price) / current_price
            
        elif action == -1 and self.position >= 0:  # Sell signal
            if self.position == 1:  # Close long position
                trade_return = (next_price - current_price) / current_price
            self.position = -1
            trade_return += (current_price - next_price) / current_price
            
        # Hold (action == 0) mantiene la posición actual
        elif self.position == 1:  # Mantener long
            trade_return = (next_price - current_price) / current_price
        elif self.position == -1:  # Mantener short
            trade_return = (current_price - next_price) / current_price
            
        self.trades.append({
            'action': action,
            'price': current_price,
            'return': trade_return,
            'position': self.position
        })
        
        return trade_return
    
    def _calculate_trading_metrics(self, returns: List[float], trades: List[Dict]) -> Dict[str, float]:
        """Calcular métricas de trading"""
        returns_array = np.array(returns)
        
        # Métricas básicas
        total_return = np.sum(returns_array)
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array)
        
        # Sharpe ratio (asumiendo risk-free rate = 0)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_trades = len([r for r in returns if r > 0])
        total_trades = len([t for t in trades if t['action'] != 0])
        win_rate = positive_trades / total_trades if total_trades > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': float(total_return),
            'mean_return': float(mean_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'calmar_ratio': float(calmar_ratio),
            'total_trades': int(total_trades),
            'score': float(sharpe_ratio)  # Métrica principal para optimización
        }


def create_synthetic_trading_data(n_points: int = 300, 
                                 volatility: float = 0.02,
                                 trend: float = 0.0001,
                                 random_seed: int = 42) -> pd.DataFrame:
    """
    Crear datos sintéticos de trading para optimización rápida
    
    Args:
        n_points: Número de puntos de datos
        volatility: Volatilidad del precio
        trend: Tendencia base
        random_seed: Semilla para reproducibilidad
        
    Returns:
        DataFrame con datos OHLCV sintéticos y features técnicos
    """
    np.random.seed(random_seed)
    
    # Generar precios con random walk + trend
    returns = np.random.normal(trend, volatility, n_points)
    prices = 100 * np.exp(np.cumsum(returns))  # Precio base 100
    
    # Generar OHLC realista
    data = []
    for i, close in enumerate(prices):
        # High y Low alrededor del close
        high = close * (1 + np.random.uniform(0, 0.02))
        low = close * (1 - np.random.uniform(0, 0.02))
        
        # Open como precio anterior + ruido
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.uniform(-0.01, 0.01))
        
        # Volume sintético
        volume = np.random.uniform(1000000, 10000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'returns': returns[i]
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
    
    return df


def evaluate_rl_agent_for_trading(agent, config: Dict[str, Any]) -> float:
    """
    Evaluar un agente RL para trading y retornar score para optimización
    
    Args:
        agent: Agente RL instanciado
        config: Configuración del agente
        
    Returns:
        Score de trading (Sharpe ratio)
    """
    try:
        # Crear datos sintéticos de trading
        trading_data = create_synthetic_trading_data(500)  # 500 puntos para optimización
        
        # Crear simulador
        simulator = TradingEnvironmentSimulator(trading_data)
        
        # Simular trading
        metrics = simulator.simulate_trading(agent, num_episodes=1)
        
        # Retornar Sharpe ratio como métrica principal
        return metrics['sharpe_ratio']
        
    except Exception as e:
        print(f"Error evaluating RL agent: {e}")
        return -999.0  # Score muy bajo para indicar fallo
