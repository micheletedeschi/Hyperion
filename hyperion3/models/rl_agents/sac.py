"""
Soft Actor-Critic (SAC) Implementation for Hyperion V2
State-of-the-art off-policy RL algorithm for continuous control
Superior to PPO for cryptocurrency trading
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import TypeVar

import os
import torch
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, TypeVar
from dataclasses import dataclass
from collections import namedtuple, defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import gymnasium as gym
from torch.distributions import Normal
from sklearn.preprocessing import StandardScaler

# Permitir fallback si MPS no está disponible
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Configure logging
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select execution device preferring Metal on macOS."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Usando dispositivo MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("Usando dispositivo CUDA")
        return torch.device("cuda")
    logger.warning("MPS/CUDA no disponibles, usando CPU")
    return torch.device("cpu")


# Initialize device
DEVICE = get_device()
logger.info(f"🚀 Usando dispositivo: {DEVICE}")

# Añadir tipos de retorno para metrics
T = TypeVar("T", bound=Union[float, List[float]])

# Named tuple for transitions
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


@dataclass
class TradingConfig:
    """Configuration for the trading environment"""

    window_size: int = 50
    reward_scaling: float = 1.0
    transaction_fee: float = 0.001
    initial_balance: float = 10000.0
    """
    Trading environment for SAC with continuous actions.
    - State: Market data + portfolio state
    - Action: Continuous position between -1 (full short) and 1 (full long)
    - Reward: Portfolio returns with risk penalties
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        feature_columns: List[str],
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        window_size: int = 50,
        reward_scaling: float = 1.0,
    ):
        """Initialize the trading environment

        Args:
            market_data: DataFrame with market data including 'close' prices
            feature_columns: List of feature column names to use
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction of trade value
            window_size: Number of past time steps to include in state
            reward_scaling: Scaling factor for rewards
        """
        self.market_data = market_data
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.reward_scaling = reward_scaling

        # Dimensions
        self.num_features = len(feature_columns)
        self.state_dim = self.num_features * window_size + 4  # +4 for portfolio state

        # Action space: continuous position between -1 and 1
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State space
        low = np.full(self.state_dim, -np.inf, dtype=np.float32)
        high = np.full(self.state_dim, np.inf, dtype=np.float32)

        # Observation space bounds for portfolio features
        low[-4:] = np.array(
            [0, -1, 0, 0]
        )  # [balance_ratio, position, price, position_value]
        high[-4:] = np.array([np.inf, 1, np.inf, np.inf])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Feature scaling
        self.scaler = StandardScaler()
        market_features = self.market_data[feature_columns].values
        self.scaler.fit(market_features)

        # Trading state
        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state

        Args:
            seed: The seed that is used to initialize the environment's RNG
            options: Additional options used to customize the environment's behavior

        Returns:
            Tuple of initial observation and an info dictionary
        """
        super().reset(seed=seed)

        # Trading state
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_value = self.initial_balance

        # History
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        self.performance_metrics = {}

        state = self._get_state()
        info = {
            "portfolio_value": self.total_value,
            "position": self.position,
            "balance": self.balance,
        }

        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment

        Args:
            action: Array with position between -1 and 1

        Returns:
            State, reward, terminated, truncated and info
        """
        # Get target position from action
        target_position = float(action[0])  # Ensure float

        # Current price and portfolio value
        current_price = self.market_data.iloc[self.current_step]["close"]
        prev_value = self.total_value

        # Execute trade
        position_change = target_position - self.position
        transaction_cost = 0.0

        if abs(position_change) > 0.01:  # 1% minimum trade
            # Transaction costs
            cost = abs(position_change * current_price * self.initial_balance)
            transaction_cost = cost * self.transaction_fee
            self.balance -= transaction_cost

            # Update position and calculate costs/proceeds
            trade_value = position_change * current_price * self.initial_balance

            if position_change > 0 and trade_value <= self.balance:
                # Buy
                self.balance -= trade_value
                self.position = target_position
                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "buy",
                        "price": current_price,
                        "size": position_change,
                        "value": trade_value,
                        "cost": transaction_cost,
                    }
                )
            elif position_change < 0:
                # Sell
                self.balance -= trade_value  # Negative trade_value adds to balance
                self.position = target_position
                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "sell",
                        "price": current_price,
                        "size": position_change,
                        "value": -trade_value,
                        "cost": transaction_cost,
                    }
                )

        # Move to next step
        self.current_step += 1

        # Calculate new portfolio value
        new_price = self.market_data.iloc[self.current_step]["close"]
        position_value = self.position * new_price * self.initial_balance
        self.total_value = self.balance + position_value
        self.portfolio_values.append(self.total_value)

        # Calculate reward
        returns = (self.total_value - prev_value) / prev_value
        reward = returns * self.reward_scaling

        # Penalties
        reward -= 0.01 * abs(self.position)  # Holdings cost
        reward -= 0.1 * (abs(position_change) > 0.01)  # Trading cost
        if transaction_cost > 0:
            reward -= (
                transaction_cost / self.initial_balance
            )  # Transaction cost penalty

        # Done conditions
        done = False
        truncated = False

        # Episode ends if:
        # 1. No more data
        # 2. Portfolio value too low
        # 3. Max steps reached
        if self.current_step >= len(self.market_data) - 1:
            done = True
        elif self.total_value <= self.initial_balance * 0.5:  # 50% max drawdown
            done = True
            reward -= 1.0  # Additional penalty for busting account
        elif len(self.portfolio_values) >= 1000:  # Max episode length
            truncated = True

        state = self._get_state()
        info = {
            "portfolio_value": self.total_value,
            "position": self.position,
            "balance": self.balance,
            "trades": len(self.trades),
            "returns": returns,
        }

        if done or truncated:
            self.performance_metrics = self.calculate_metrics()
            info.update(self.performance_metrics)

        return state, reward, done, truncated, info

    def _get_state(self) -> np.ndarray:
        """Get current state observation"""
        # Market features (scaled)
        market_window = self.market_data.iloc[
            self.current_step - self.window_size : self.current_step
        ][self.feature_columns].values

        market_features = self.scaler.transform(market_window).flatten()
        if np.isnan(market_features).any():
            logger.warning(
                "Market features contain NaN values. Replacing with 0..."
            )
            market_features = np.nan_to_num(market_features)

        # Portfolio features
        current_price = self.market_data.iloc[self.current_step]["close"]
        position_value = self.position * current_price * self.initial_balance
        portfolio_features = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                self.position,  # Current position
                current_price,  # Current price
                position_value / self.initial_balance,  # Normalized position value
            ]
        )
        if np.isnan(portfolio_features).any():
            logger.warning(
                "Portfolio features contain NaN values. Replacing with 0..."
            )
            portfolio_features = np.nan_to_num(portfolio_features)

        state = np.concatenate([market_features, portfolio_features])
        if np.isnan(state).any():
            logger.warning("State contains NaN values. Replacing with 0...")
            state = np.nan_to_num(state)

        return state

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics at end of episode"""
        returns = (
            np.array(self.portfolio_values[1:]) / np.array(self.portfolio_values[:-1])
            - 1
        )

        metrics = {}
        metrics["final_value"] = self.total_value
        metrics["total_return"] = (self.total_value / self.initial_balance - 1) * 100
        metrics["max_drawdown"] = self._calculate_max_drawdown()

        if len(returns) > 1:
            metrics["volatility"] = float(np.std(returns) * np.sqrt(252))  # Annualized
            metrics["sharpe_ratio"] = float(
                np.mean(returns) / np.std(returns) * np.sqrt(252)
            )

        if len(self.trades) > 0:
            metrics["num_trades"] = len(self.trades)
            metrics["avg_trade_size"] = float(
                np.mean([abs(t["size"]) for t in self.trades])
            )

        return metrics

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        return float(abs(min(drawdowns)))

    def render(self, mode="human"):
        """Display current state"""
        current_price = self.market_data.iloc[self.current_step]["close"]
        position_value = self.position * current_price * self.initial_balance

        print(f"\nStep: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Position: {self.position:.4f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position Value: ${position_value:.2f}")
        print(f"Total Value: ${self.total_value:.2f}")
        print(f"Return: {(self.total_value/self.initial_balance - 1)*100:.2f}%")
        print(f"Trades: {len(self.trades)}")


class ReplayBuffer:
    """Simple replay buffer for storing and sampling transitions"""

    def __init__(self, max_size: int):
        """Initialize replay buffer

        Args:
            max_size: Maximum number of transitions to store
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Buffers
        self.states = np.zeros((max_size, 0))  # Will resize on first add
        self.actions = np.zeros((max_size, 0))  # Will resize on first add
        self.rewards = np.zeros(max_size)
        self.next_states = np.zeros((max_size, 0))  # Will resize on first add
        self.dones = np.zeros(max_size, dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer"""
        # Resize buffers if needed
        if self.states.shape[1] == 0:
            self.states = np.zeros((self.max_size, state.shape[0]))
            self.next_states = np.zeros((self.max_size, state.shape[0]))
            self.actions = np.zeros((self.max_size, action.shape[0]))

        # Store transition
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Update pointer
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of transitions"""
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...],
        max_action: float = 1.0,
    ):
        super().__init__()

        self.device = DEVICE
        layers = []
        prev_dim = state_dim

        # Construir capas ocultas con normalización y ReLU
        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            )
            prev_dim = hidden_dim

        # Capa de salida con Tanh para acciones acotadas
        layers.extend([nn.Linear(prev_dim, action_dim), nn.Tanh()])

        self.network = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.max_action = torch.tensor(max_action).to(self.device)

        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mover al dispositivo correcto si es necesario
        if state.device != self.device:
            state = state.to(self.device)

        mean = self.network(state)
        std = self.log_std.exp()
        return mean, std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mover al dispositivo correcto si es necesario
        if state.device != self.device:
            state = state.to(self.device)

        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x = normal.rsample()  # Reparametrization trick
        log_prob = normal.log_prob(x).sum(dim=-1, keepdim=True)
        action = self.max_action * torch.tanh(x)
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()

        self.device = DEVICE
        layers = []
        prev_dim = state_dim + action_dim

        # Construir capas ocultas
        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Mover al dispositivo correcto si es necesario
        if state.device != self.device:
            state = state.to(self.device)
        if action.device != self.device:
            action = action.to(self.device)

        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...],
        max_action: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 256,
        replay_buffer_size: int = 1_000_000,
    ):
        # Parámetros
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size

        # Dispositivo
        self.device = DEVICE

        # Inicializar Replay Buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Redes
        self.policy = Actor(state_dim, action_dim, hidden_dims, max_action).to(
            self.device
        )
        self.critic1 = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)

        # Copiar pesos a las redes objetivo
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizadores
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

    def get_state_dict(self) -> Dict[str, Any]:
        """Return state dict for saving"""
        return {
            "policy": self.policy.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
        }

    def set_state_dict(self, state_dict: Dict[str, Any]):
        """Load from state dict"""
        self.policy.load_state_dict(state_dict["policy"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self.critic1_target.load_state_dict(state_dict["critic1_target"])
        self.critic2_target.load_state_dict(state_dict["critic2_target"])
        self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
        self.critic1_optimizer.load_state_dict(state_dict["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state_dict["critic2_optimizer"])

    def train(
        self, env: "TradingEnvironmentSAC", episodes: int = 1000
    ) -> Dict[str, float]:
        """Train the agent

        Args:
            env: Trading environment
            episodes: Number of episodes to train for

        Returns:
            Dict of training metrics
        """
        metrics = defaultdict(list)
        total_steps = 0

        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Get action from agent
                with torch.no_grad():
                    state_tensor = (
                        torch.as_tensor(state, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    action, _, _ = self.policy.sample(state_tensor)
                    action = action.squeeze(0).cpu().numpy()

                # Take step in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Update agent if enough samples
                if len(self.replay_buffer) > self.batch_size:
                    update_info = self.update()
                    for k, v in update_info.items():
                        metrics[k].append(v)

                state = next_state
                episode_reward += reward
                total_steps += 1

            metrics["episode_reward"].append(episode_reward)
            metrics["portfolio_value"].append(env.total_value)

            if episode % 10 == 0:
                mean_reward = float(np.mean(metrics["episode_reward"][-10:]))
                mean_value = float(np.mean(metrics["portfolio_value"][-10:]))
                logger.info(
                    f"Episode {episode}: Mean Reward = {mean_reward:.2f}, Mean Value = {mean_value:.2f}"
                )

        # Convert lists to means for final metrics
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for given state"""
        # Convertir estado a tensor en MPS
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if deterministic:
                # Modo determinístico: usar la media directamente
                action, _, _ = self.policy.sample(state_tensor)
            else:
                # Modo estocástico: muestrear de la distribución
                action, _, _ = self.policy.sample(state_tensor)

            # Mantener en MPS hasta el último momento
            # Solo convertir a CPU/numpy cuando sea absolutamente necesario
            return action.squeeze(0).cpu().numpy()

    def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Update the agent's networks

        Args:
            batch_size: Batch size to use for update. If None, uses self.batch_size

        Returns:
            Dict of training metrics
        """
        metrics = {}
        batch_size = batch_size or self.batch_size

        if len(self.replay_buffer) > batch_size:
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                batch_size
            )

            # Convert to tensors on MPS
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            next_states = torch.as_tensor(
                next_states, dtype=torch.float32, device=self.device
            )
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            rewards = torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
            dones = torch.as_tensor(
                dones, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)

            # Get target Q values
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(
                    next_states
                )
                target_q1 = self.critic1_target(next_states, next_state_action)
                target_q2 = self.critic2_target(next_states, next_state_action)
                target_q = (
                    torch.min(target_q1, target_q2) - self.alpha * next_state_log_pi
                )
                target_q = rewards + (1 - dones) * self.gamma * target_q

            # Update critics
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)

            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            metrics["critic1_loss"] = float(critic1_loss.item())
            metrics["critic2_loss"] = float(critic2_loss.item())

            # Update policy
            pi, log_pi, _ = self.policy.sample(states)
            q1_pi = self.critic1(states, pi)
            q2_pi = self.critic2(states, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)

            policy_loss = (self.alpha * log_pi - min_q_pi).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            metrics["policy_loss"] = float(policy_loss.item())

            # Update targets
            self._soft_update(self.critic1_target, self.critic1, self.tau)
            self._soft_update(self.critic2_target, self.critic2, self.tau)

        return metrics

    def _soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float):
        """
        Soft update of target network parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            target_net: Target network to update
            source_net: Source network to copy from
            tau: Interpolation parameter
        """
        for target_param, source_param in zip(
            target_net.parameters(), source_net.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )


class SACTradingAgent:
    """
    Trading agent that uses Soft Actor-Critic (SAC) for continuous control.
    Designed for compatibility with the trading ensemble.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SAC Trading Agent from a configuration dictionary

        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config

        feature_cols = []
        window_size = None
        if isinstance(config, dict):
            feature_cols = config.get("feature_columns") or config.get("data", {}).get(
                "feature_columns", []
            )
            window_size = (
                config.get("window_size")
                or config.get("lookback_window")
                or config.get("data", {}).get("lookback_window")
            )
        else:
            data_cfg = getattr(config, "data", None)
            if data_cfg is not None:
                feature_cols = getattr(data_cfg, "feature_columns", [])
                window_size = getattr(data_cfg, "lookback_window", None)
        if feature_cols and window_size is not None:
            self.state_dim = len(feature_cols) * int(window_size) + 4
        else:
            self.state_dim = config.get("state_dim", 15)  # Fallback

        self.action_dim = config.get("action_dim", 1)  # Default 1 for continuous action
        hidden_dims = config.get("hidden_dims", (256, 256))
        if not isinstance(hidden_dims, (list, tuple)):
            hidden_dims = (hidden_dims, hidden_dims)

        # Initialize base SAC agent
        self.agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=tuple(hidden_dims),
            max_action=config.get("max_action", 1.0),
            gamma=config.get("gamma", 0.99),
            tau=config.get("tau", 0.005),
            alpha=config.get("alpha", 0.2),
            batch_size=config.get("batch_size", 256),
            replay_buffer_size=config.get("replay_buffer_size", 1_000_000),
        )

    def fit(
        self, market_data: pd.DataFrame, feature_columns: List[str], **kwargs
    ) -> Dict[str, float]:
        """
        Train the agent on market data

        Args:
            market_data: DataFrame with market data
            feature_columns: List of column names to use as features
            **kwargs: Additional arguments

        Returns:
            Dictionary with training metrics
        """
        # Create environment
        env = TradingEnvironmentSAC(
            market_data=market_data,
            feature_columns=feature_columns,
            initial_balance=self.config.get("initial_balance", 10000.0),
            transaction_fee=self.config.get("transaction_fee", 0.001),
            window_size=(
                self.config.get("window_size")
                or self.config.get("lookback_window", 100)
            ),
            reward_scaling=self.config.get("reward_scaling", 1.0),
        )

        # Train the agent
        return self.agent.train(env=env, episodes=self.config.get("episodes", 1000))

    def predict(
        self, market_data: pd.DataFrame, feature_columns: List[str], **kwargs
    ) -> np.ndarray:
        """
        Generate trading actions for market data

        Args:
            market_data: DataFrame with market data
            feature_columns: List of column names to use as features
            **kwargs: Additional arguments

        Returns:
            Array of trading actions
        """
        env = TradingEnvironmentSAC(
            market_data=market_data,
            feature_columns=feature_columns,
            initial_balance=self.config.get("initial_balance", 10000.0),
            transaction_fee=self.config.get("transaction_fee", 0.001),
            window_size=(
                self.config.get("window_size")
                or self.config.get("lookback_window", 100)
            ),
            reward_scaling=self.config.get("reward_scaling", 1.0),
        )

        state, _ = env.reset()
        actions = []
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = (
                    torch.as_tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.agent.device)
                )
                action, _, _ = self.agent.policy.sample(state_tensor)
                action = action.squeeze(0).cpu().numpy()

            actions.append(action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        return np.array(actions)

    def save(self, path: str):
        """Save the agent's state"""
        torch.save(
            {"config": self.config, "agent_state": self.agent.get_state_dict()}, path
        )

    def load(self, path: str):
        """Load the agent's state"""
        checkpoint = torch.load(path)
        self.config = checkpoint["config"]
        self.agent.set_state_dict(checkpoint["agent_state"])


class TradingEnvironmentSAC(gym.Env):
    """
    Trading environment for SAC with continuous actions.
    - State: Market data + portfolio state
    - Action: Continuous position between -1 (full short) and 1 (full long)
    - Reward: Portfolio returns with risk penalties
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        feature_columns: List[str],
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        window_size: int = 50,
        reward_scaling: float = 1.0,
    ):
        """Initialize the trading environment

        Args:
            market_data: DataFrame with market data including 'close' prices
            feature_columns: List of feature column names to use
            initial_balance: Initial portfolio balance
            transaction_fee: Transaction fee as a fraction of trade value
            window_size: Number of past time steps to include in state
            reward_scaling: Scaling factor for rewards
        """
        self.market_data = market_data
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.reward_scaling = reward_scaling

        # Dimensions
        self.num_features = len(feature_columns)
        self.state_dim = self.num_features * window_size + 4  # +4 for portfolio state

        # Action space: continuous position between -1 and 1
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State space
        low = np.full(self.state_dim, -np.inf, dtype=np.float32)
        high = np.full(self.state_dim, np.inf, dtype=np.float32)

        # Observation space bounds for portfolio features
        low[-4:] = np.array(
            [0, -1, 0, 0]
        )  # [balance_ratio, position, price, position_value]
        high[-4:] = np.array([np.inf, 1, np.inf, np.inf])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Feature scaling
        self.scaler = StandardScaler()
        market_features = self.market_data[feature_columns].values
        self.scaler.fit(market_features)

        # Trading state
        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state

        Args:
            seed: The seed that is used to initialize the environment's RNG
            options: Additional options used to customize the environment's behavior

        Returns:
            Tuple of initial observation and an info dictionary
        """
        super().reset(seed=seed)

        # Trading state
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_value = self.initial_balance

        # History
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        self.performance_metrics = {}

        state = self._get_state()
        info = {
            "portfolio_value": self.total_value,
            "position": self.position,
            "balance": self.balance,
        }

        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment

        Args:
            action: Array with position between -1 and 1

        Returns:
            State, reward, terminated, truncated and info
        """
        # Get target position from action
        target_position = float(action[0])  # Ensure float

        # Current price and portfolio value
        current_price = self.market_data.iloc[self.current_step]["close"]
        prev_value = self.total_value

        # Execute trade
        position_change = target_position - self.position
        transaction_cost = 0.0

        if abs(position_change) > 0.01:  # 1% minimum trade
            # Transaction costs
            cost = abs(position_change * current_price * self.initial_balance)
            transaction_cost = cost * self.transaction_fee
            self.balance -= transaction_cost

            # Update position and calculate costs/proceeds
            trade_value = position_change * current_price * self.initial_balance

            if position_change > 0 and trade_value <= self.balance:
                # Buy
                self.balance -= trade_value
                self.position = target_position
                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "buy",
                        "price": current_price,
                        "size": position_change,
                        "value": trade_value,
                        "cost": transaction_cost,
                    }
                )
            elif position_change < 0:
                # Sell
                self.balance -= trade_value  # Negative trade_value adds to balance
                self.position = target_position
                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "sell",
                        "price": current_price,
                        "size": position_change,
                        "value": -trade_value,
                        "cost": transaction_cost,
                    }
                )

        # Move to next step
        self.current_step += 1

        # Calculate new portfolio value
        new_price = self.market_data.iloc[self.current_step]["close"]
        position_value = self.position * new_price * self.initial_balance
        self.total_value = self.balance + position_value
        self.portfolio_values.append(self.total_value)

        # Calculate reward
        returns = (self.total_value - prev_value) / prev_value
        reward = returns * self.reward_scaling

        # Penalties
        reward -= 0.01 * abs(self.position)  # Holdings cost
        reward -= 0.1 * (abs(position_change) > 0.01)  # Trading cost
        if transaction_cost > 0:
            reward -= (
                transaction_cost / self.initial_balance
            )  # Transaction cost penalty

        # Done conditions
        done = False
        truncated = False

        # Episode ends if:
        # 1. No more data
        # 2. Portfolio value too low
        # 3. Max steps reached
        if self.current_step >= len(self.market_data) - 1:
            done = True
        elif self.total_value <= self.initial_balance * 0.5:  # 50% max drawdown
            done = True
            reward -= 1.0  # Additional penalty for busting account
        elif len(self.portfolio_values) >= 1000:  # Max episode length
            truncated = True

        state = self._get_state()
        info = {
            "portfolio_value": self.total_value,
            "position": self.position,
            "balance": self.balance,
            "trades": len(self.trades),
            "returns": returns,
        }

        if done or truncated:
            self.performance_metrics = self.calculate_metrics()
            info.update(self.performance_metrics)

        return state, reward, done, truncated, info

    def _get_state(self) -> np.ndarray:
        """Get current state observation"""
        # Market features (scaled)
        market_window = self.market_data.iloc[
            self.current_step - self.window_size : self.current_step
        ][self.feature_columns].values

        market_features = self.scaler.transform(market_window).flatten()
        if np.isnan(market_features).any():
            logger.warning(
                "Market features contain NaN values. Replacing with 0..."
            )
            market_features = np.nan_to_num(market_features)

        # Portfolio features
        current_price = self.market_data.iloc[self.current_step]["close"]
        position_value = self.position * current_price * self.initial_balance
        portfolio_features = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                self.position,  # Current position
                current_price,  # Current price
                position_value / self.initial_balance,  # Normalized position value
            ]
        )
        if np.isnan(portfolio_features).any():
            logger.warning(
                "Portfolio features contain NaN values. Replacing with 0..."
            )
            portfolio_features = np.nan_to_num(portfolio_features)

        state = np.concatenate([market_features, portfolio_features])
        if np.isnan(state).any():
            logger.warning("State contains NaN values. Replacing with 0...")
            state = np.nan_to_num(state)

        return state

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics at end of episode"""
        returns = (
            np.array(self.portfolio_values[1:]) / np.array(self.portfolio_values[:-1])
            - 1
        )

        metrics = {}
        metrics["final_value"] = self.total_value
        metrics["total_return"] = (self.total_value / self.initial_balance - 1) * 100
        metrics["max_drawdown"] = self._calculate_max_drawdown()

        if len(returns) > 1:
            metrics["volatility"] = float(np.std(returns) * np.sqrt(252))  # Annualized
            metrics["sharpe_ratio"] = float(
                np.mean(returns) / np.std(returns) * np.sqrt(252)
            )

        if len(self.trades) > 0:
            metrics["num_trades"] = len(self.trades)
            metrics["avg_trade_size"] = float(
                np.mean([abs(t["size"]) for t in self.trades])
            )

        return metrics

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        return float(abs(min(drawdowns)))

    def render(self, mode="human"):
        """Display current state"""
        current_price = self.market_data.iloc[self.current_step]["close"]
        position_value = self.position * current_price * self.initial_balance

        print(f"\nStep: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Position: {self.position:.4f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position Value: ${position_value:.2f}")
        print(f"Total Value: ${self.total_value:.2f}")
        print(f"Return: {(self.total_value/self.initial_balance - 1)*100:.2f}%")
        print(f"Trades: {len(self.trades)}")
