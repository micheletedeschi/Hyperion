"""
Twin Delayed DDPG (TD3) Implementation for Hyperion V2
State-of-the-art continuous control with improved stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from collections import deque, namedtuple
import random
import copy
from gymnasium import spaces
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Named tuple for transitions
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class Actor(nn.Module):
    """Actor network for TD3"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_action: float = 1.0,
    ):
        super().__init__()

        self.max_action = max_action

        # Network architecture
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, action_dim)

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.l1(state)))
        x = F.relu(self.ln2(self.l2(x)))
        x = F.relu(self.ln3(self.l3(x)))
        x = self.max_action * torch.tanh(self.l4(x))
        return x


class Critic(nn.Module):
    """Twin Q-networks for TD3"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1 network
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l4 = nn.Linear(hidden_dim, 1)

        # Q2 network (twin)
        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l4 = nn.Linear(hidden_dim, 1)

        # Layer normalization
        self.ln1_q1 = nn.LayerNorm(hidden_dim)
        self.ln2_q1 = nn.LayerNorm(hidden_dim)
        self.ln3_q1 = nn.LayerNorm(hidden_dim)

        self.ln1_q2 = nn.LayerNorm(hidden_dim)
        self.ln2_q2 = nn.LayerNorm(hidden_dim)
        self.ln3_q2 = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        # Q1 forward pass
        q1 = F.relu(self.ln1_q1(self.q1_l1(sa)))
        q1 = F.relu(self.ln2_q1(self.q1_l2(q1)))
        q1 = F.relu(self.ln3_q1(self.q1_l3(q1)))
        q1 = self.q1_l4(q1)

        # Q2 forward pass
        q2 = F.relu(self.ln1_q2(self.q2_l1(sa)))
        q2 = F.relu(self.ln2_q2(self.q2_l2(q2)))
        q2 = F.relu(self.ln3_q2(self.q2_l3(q2)))
        q2 = self.q2_l4(q2)

        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q1 value only (for policy update)"""
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.ln1_q1(self.q1_l1(sa)))
        q1 = F.relu(self.ln2_q1(self.q1_l2(q1)))
        q1 = F.relu(self.ln3_q1(self.q1_l3(q1)))
        q1 = self.q1_l4(q1)

        return q1


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient

    Key improvements over DDPG:
    1. Twin Q-networks to reduce overestimation
    2. Delayed policy updates
    3. Target policy smoothing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        config: Dict,
        device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    ):
        self.device = device
        self.config = config

        # Hyperparameters
        self.max_action = max_action
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_freq = config.get("policy_freq", 2)
        self.batch_size = config.get("batch_size", 256)
        self.lr_actor = config.get("lr_actor", 3e-4)
        self.lr_critic = config.get("lr_critic", 3e-4)

        # Networks
        self.actor = Actor(
            state_dim, action_dim, config.get("hidden_dim", 256), max_action
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor
        )

        self.critic = Critic(state_dim, action_dim, config.get("hidden_dim", 256)).to(
            device
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic
        )

        # Replay buffer
        self.memory = ReplayBuffer(config.get("buffer_size", 1000000))

        # Training tracking
        self.total_it = 0

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action from policy"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
            # Add exploration noise
            noise = np.random.normal(
                0,
                self.max_action * self.config.get("exploration_noise", 0.1),
                size=action.shape,
            )
            action = (action + noise).clip(-self.max_action, self.max_action)

        return action

    def train(self, replay_buffer, iterations: int = 1):
        """Train the TD3 agent"""

        for it in range(iterations):
            self.total_it += 1

            # Sample batch from replay buffer
            state, action, reward, next_state, done = replay_buffer.sample(
                self.batch_size
            )

            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).reshape(-1, 1).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).reshape(-1, 1).to(self.device)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )

                next_action = (self.actor_target(next_state) + noise).clamp(
                    -self.max_action, self.max_action
                )

                # Compute target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * self.gamma * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": (
                actor_loss.item() if self.total_it % self.policy_freq == 0 else 0
            ),
        }

    def save(self, filename: str):
        """Save model checkpoint"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
            },
            filename,
        )

    def load(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_it = checkpoint["total_it"]


class ReplayBuffer:
    """Experience replay buffer for TD3"""

    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)


class TD3TradingAgent:
    """
    Complete TD3 agent for cryptocurrency trading
    """

    def __init__(
        self,
        config: Dict,
        device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    ):
        self.config = config
        self.device = device

        # Obtener las dimensiones de los datos de entrada
        num_features = len(
            config.get(
                "feature_columns",
                [
                    "log_return",
                    "volatility",
                    "rsi",
                    "sma_20",
                    "std_20",
                    "upper_band",
                    "lower_band",
                    "macd",
                    "macd_signal",
                    "volume_norm",
                    "momentum",
                    "atr",
                ],
            )
        )

        # Calcular la dimensión del estado
        lookback_window = config.get("lookback_window", 50)
        self.state_dim = num_features * lookback_window + 4  # +4 for portfolio state

        # Environment configuration
        self.action_dim = 1  # Position size
        self.max_action = config.get("max_position_size", 1.0)

        # Initialize TD3
        self.td3 = TD3(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            config=config,
            device=device,
        )

        # Trading specific parameters
        self.initial_balance = config.get("initial_balance", 10000)
        self.transaction_fee = config.get("transaction_fee", 0.001)

        # Training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "sharpe_ratios": [],
            "actor_losses": [],
            "critic_losses": [],
        }

    def fit(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None):
        """Train the agent (compatible with ensemble interface)"""
        # Convertir DataFrame a numpy array si es necesario
        if isinstance(train_data, pd.DataFrame):
            # Asegurarse de que tenemos las columnas necesarias
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in train_data.columns for col in required_cols):
                raise ValueError(
                    f"DataFrame debe contener las columnas: {required_cols}"
                )

            # Convertir a numpy array manteniendo el orden de las columnas
            train_data = train_data[required_cols].values

        if val_data is not None and isinstance(val_data, pd.DataFrame):
            val_data = val_data[required_cols].values

        return self.train(train_data, val_data)

    async def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        num_episodes: int = 1000,
        save_freq: int = 100,
    ):
        """Train the TD3 agent"""
        from environment import TradingEnvironmentTD3

        env = TradingEnvironmentTD3(train_data, self.config)

        # Warm up replay buffer
        state = env.reset()
        for _ in range(self.config.get("warm_up_steps", 1000)):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.td3.memory.push(state, action, reward, next_state, done)
            state = env.reset() if done else next_state

        # Training loop
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0

            while True:
                # Select action
                if episode < self.config.get("pure_exploration_episodes", 10):
                    action = np.random.uniform(-self.max_action, self.max_action)
                else:
                    action = self.td3.select_action(state)

                # Take step
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Store transition
                self.td3.memory.push(
                    state, np.array([action]), reward, next_state, done
                )

                # Train agent
                if len(self.td3.memory) > self.td3.batch_size:
                    losses = self.td3.train(self.td3.memory)

                    if episode_steps % 100 == 0:
                        self.training_metrics["actor_losses"].append(
                            losses["actor_loss"]
                        )
                        self.training_metrics["critic_losses"].append(
                            losses["critic_loss"]
                        )

                state = next_state

                if done:
                    break

            # Record metrics
            self.training_metrics["episode_rewards"].append(episode_reward)
            self.training_metrics["portfolio_values"].append(info["portfolio_value"])

            # Calculate Sharpe ratio
            if len(env.portfolio_values) > 1:
                returns = np.diff(env.portfolio_values) / env.portfolio_values[:-1]
                sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-6)
                self.training_metrics["sharpe_ratios"].append(sharpe)

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_metrics["episode_rewards"][-100:])
                portfolio_return = (
                    info["portfolio_value"] / env.initial_balance - 1
                ) * 100

                print(f"Episode {episode}/{num_episodes}")
                print(f"Average Reward: {avg_reward:.3f}")
                print(f"Portfolio Return: {portfolio_return:.2f}%")
                print(f"Sharpe Ratio: {sharpe:.3f}")
                print("-" * 50)

            # Validation
            if val_data is not None and episode % 50 == 0:
                val_metrics = await self.evaluate(val_data)
                print(
                    f"Validation - Return: {val_metrics['total_return']:.2f}%, "
                    f"Sharpe: {val_metrics['sharpe_ratio']:.3f}"
                )

            # Save model
            if episode % save_freq == 0:
                self.save(f"td3_model_episode_{episode}.pth")

    async def evaluate(
        self, test_data: np.ndarray, render: bool = False
    ) -> Dict[str, float]:
        """Evaluate the agent on test data"""
        from environment import TradingEnvironmentTD3

        env = TradingEnvironmentTD3(test_data, self.config, training=False)

        state = env.reset()
        done = False

        while not done:
            action = self.td3.select_action(state, add_noise=False)
            state, reward, done, info = env.step(action)

            if render:
                env.render()

        # Calculate metrics
        total_return = (info["portfolio_value"] / env.initial_balance - 1) * 100

        returns = np.diff(env.portfolio_values) / env.portfolio_values[:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-6)

        # Maximum drawdown
        peak = np.maximum.accumulate(env.portfolio_values)
        drawdown = (env.portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": len(env.trades),
            "final_portfolio_value": info["portfolio_value"],
        }

    def save(self, filepath: str):
        """Save the model"""
        self.td3.save(filepath)

        # Save training metrics
        import pickle

        metrics_path = filepath.replace(".pth", "_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(self.training_metrics, f)

    def load(self, filepath: str):
        """Load the model"""
        self.td3.load(filepath)

        # Load training metrics if available
        import pickle

        metrics_path = filepath.replace(".pth", "_metrics.pkl")
        try:
            with open(metrics_path, "rb") as f:
                self.training_metrics = pickle.load(f)
        except FileNotFoundError:
            pass


class TradingEnvironmentTD3:
    """Trading environment for TD3"""

    def __init__(self, data: np.ndarray, config: Dict, training: bool = True):
        self.data = data
        self.config = config
        self.training = training

        # Environment parameters
        self.initial_balance = config.get("initial_balance", 10000)
        self.transaction_fee = config.get("transaction_fee", 0.001)
        self.lookback_window = config.get("lookback_window", 50)

        # Action space
        self.action_space = spaces.Box(
            low=-config.get("max_position_size", 1.0),
            high=config.get("max_position_size", 1.0),
            shape=(1,),
            dtype=np.float32,
        )

        # State tracking
        self.current_step = 0
        self.max_steps = len(data) - self.lookback_window - 1

        # Portfolio tracking
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_values = [self.initial_balance]
        self.trades = []

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_values = [self.initial_balance]
        self.trades = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state"""
        # Market data
        market_data = self.data[
            self.current_step - self.lookback_window : self.current_step
        ].flatten()

        # Portfolio features
        current_price = self.data[self.current_step, 3]
        position_value = self.position * current_price
        total_value = self.balance + position_value

        portfolio_features = np.array(
            [
                self.balance / self.initial_balance,
                self.position,
                total_value / self.initial_balance,
            ]
        )

        return np.concatenate([market_data, portfolio_features])

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action"""
        # Get target position from action
        target_position = np.clip(
            action,
            -self.config.get("max_position_size", 1.0),
            self.config.get("max_position_size", 1.0),
        )

        # Current price
        current_price = self.data[self.current_step, 3]

        # Execute trade
        position_change = target_position - self.position

        if abs(position_change) > 0.01:
            # Transaction cost
            transaction_cost = (
                abs(position_change) * current_price * self.transaction_fee
            )
            self.balance -= transaction_cost

            # Update position
            cost = position_change * current_price
            if position_change > 0 and cost <= self.balance:
                # Buy
                self.balance -= cost
                self.position = target_position
                self.trades.append(
                    {
                        "step": self.current_step,
                        "action": "buy",
                        "price": current_price,
                        "size": position_change,
                    }
                )
            elif position_change < 0:
                # Sell
                revenue = -position_change * current_price
                self.balance += revenue
                self.position = target_position
                self.trades.append(
                    {
                        "step": self.current_step,
                        "action": "sell",
                        "price": current_price,
                        "size": position_change,
                    }
                )

        # Move to next step
        self.current_step += 1
        new_price = self.data[self.current_step, 3]

        # Calculate portfolio value
        position_value = self.position * new_price
        total_value = self.balance + position_value
        self.portfolio_values.append(total_value)

        # Calculate reward
        returns = (total_value - self.portfolio_values[-2]) / self.portfolio_values[-2]

        # TD3 works better with dense rewards
        reward = returns * 100  # Scale up

        # Penalize large positions
        risk_penalty = -0.01 * abs(self.position)
        reward += risk_penalty

        # Done?
        done = (
            self.current_step >= self.max_steps - 1
            or total_value <= self.initial_balance * 0.5
        )

        info = {
            "portfolio_value": total_value,
            "position": self.position,
            "balance": self.balance,
            "returns": returns,
        }

        return self._get_state(), reward, done, info

    def render(self):
        """Render current state"""
        current_price = self.data[self.current_step, 3]
        position_value = self.position * current_price
        total_value = self.balance + position_value

        print(f"Step: {self.current_step}")
        print(f"Price: {current_price:.2f}")
        print(f"Position: {self.position:.3f}")
        print(f"Portfolio Value: ${total_value:.2f}")
        print(f"Return: {(total_value/self.initial_balance - 1)*100:.2f}%")
