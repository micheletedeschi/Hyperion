"""
Rainbow DQN Implementation for Hyperion V2
Combines all major DQN improvements for discrete action trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import math
import random
from collections import namedtuple, deque
from gymnasium import spaces
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Named tuple for transitions
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration"""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorised noise
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise for factorised Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Reset noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DuelingNetwork(nn.Module):
    """Dueling network architecture for Rainbow DQN"""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 512,
        atom_size: int = 51,
        support: torch.Tensor = None,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.atom_size = atom_size
        self.support = support

        # Feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Advantage stream (using noisy layers)
        self.advantage_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.advantage = NoisyLinear(hidden_dim, num_actions * atom_size)

        # Value stream (using noisy layers)
        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.value = NoisyLinear(hidden_dim, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning distribution over atoms"""
        batch_size = x.size(0)

        # Feature extraction
        features = self.feature_layer(x)

        # Advantage stream
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage(advantage).view(
            batch_size, self.num_actions, self.atom_size
        )

        # Value stream
        value = F.relu(self.value_hidden(features))
        value = self.value(value).view(batch_size, 1, self.atom_size)

        # Combine streams
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to get distribution
        q_dist = F.softmax(q_atoms, dim=-1)
        q_dist = q_dist.clamp(min=1e-3)  # For numerical stability

        return q_dist

    def reset_noise(self):
        """Reset noise in noisy layers"""
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        self.value_hidden.reset_noise()
        self.value.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, transition: Transition):
        """Add transition with maximum priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        N = len(self.buffer)

        # Calculate sampling probabilities
        priorities = self.priorities[:N]
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6

    def __len__(self):
        return len(self.buffer)


class RainbowDQN:
    """
    Rainbow DQN: Combining all DQN improvements
    - Double DQN
    - Prioritized Experience Replay
    - Dueling Networks
    - Multi-step Learning
    - Distributional RL
    - Noisy Networks
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict,
        device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    ):
        self.device = device
        self.config = config

        # Hyperparameters
        self.action_dim = action_dim
        self.batch_size = config.get("batch_size", 32)
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 6.25e-5)
        self.tau = config.get("tau", 1e-3)

        # Multi-step learning
        self.n_steps = config.get("n_steps", 3)
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # Distributional RL
        self.atom_size = config.get("atom_size", 51)
        self.v_min = config.get("v_min", -10.0)
        self.v_max = config.get("v_max", 10.0)
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)
        self.delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        # Networks
        self.online_net = DuelingNetwork(
            state_dim,
            action_dim,
            config.get("hidden_dim", 512),
            self.atom_size,
            self.support,
        ).to(device)

        self.target_net = DuelingNetwork(
            state_dim,
            action_dim,
            config.get("hidden_dim", 512),
            self.atom_size,
            self.support,
        ).to(device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)

        # Replay buffer
        self.memory = PrioritizedReplayBuffer(
            config.get("buffer_size", 1000000),
            config.get("alpha", 0.6),
            config.get("beta", 0.4),
        )

        # Training variables
        self.training_steps = 0
        self.target_update_freq = config.get("target_update_freq", 1000)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using the online network"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if not evaluate:
                self.online_net.reset_noise()

            q_dist = self.online_net(state)
            q_values = (q_dist * self.support).sum(dim=-1)
            action = q_values.argmax(dim=-1).item()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in n-step buffer"""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Only store when we have n_steps transitions
        if len(self.n_step_buffer) == self.n_steps:
            # Calculate n-step return
            n_step_return = 0
            for i in range(self.n_steps):
                n_step_return += (self.gamma**i) * self.n_step_buffer[i][2]

            # Get first and last states
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_next_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]

            # Store in replay buffer
            transition = Transition(
                first_state, first_action, n_step_return, last_next_state, last_done
            )
            self.memory.push(transition)

    def train(self) -> Dict[str, float]:
        """Train the Rainbow DQN"""
        if len(self.memory) < self.batch_size:
            return {}

        # Sample from replay buffer
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors
        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q distribution
        current_dist = self.online_net(states)
        current_dist = current_dist.gather(
            1, actions.unsqueeze(-1).expand(-1, -1, self.atom_size)
        )
        current_dist = current_dist.squeeze(1)

        # Calculate target distribution
        with torch.no_grad():
            # Double DQN action selection
            next_dist = self.online_net(next_states)
            next_q_values = (next_dist * self.support).sum(dim=-1)
            next_actions = next_q_values.argmax(dim=-1)

            # Target network evaluation
            target_dist = self.target_net(next_states)
            target_dist = target_dist.gather(
                1, next_actions.unsqueeze(-1).expand(-1, -1, self.atom_size)
            )
            target_dist = target_dist.squeeze(1)

            # Compute Tz (Bellman operator)
            t_z = (
                rewards.unsqueeze(-1)
                + (1 - dones.unsqueeze(-1)) * (self.gamma**self.n_steps) * self.support
            )
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            # Compute projection of Tz onto support
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Fix out of bound
            l[(l == u) & (l > 0)] -= 1
            u[(u == 0)] = 1

            # Distribute probability
            m = states.new_zeros(self.batch_size, self.atom_size)
            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            m.view(-1).index_add_(
                0, (l + offset).view(-1), (target_dist * (u.float() - b)).view(-1)
            )
            m.view(-1).index_add_(
                0, (u + offset).view(-1), (target_dist * (b - l.float())).view(-1)
            )

        # Cross-entropy loss
        loss = -(m * current_dist.clamp(min=1e-3).log()).sum(dim=-1)

        # Apply importance sampling weights
        loss = (loss * weights).mean()

        # Update priorities
        priorities = loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self._update_target_network()

        # Reset noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        return {"loss": loss.item()}

    def _update_target_network(self):
        """Soft update of target network"""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath: str):
        """Save model"""
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint["training_steps"]


class RainbowTradingAgent:
    """
    Complete Rainbow DQN agent for cryptocurrency trading
    Uses discrete actions for position management
    """

    def __init__(
        self, config: Dict, device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = config
        self.device = device

        # Trading actions
        # 0: Hold, 1: Buy 25%, 2: Buy 50%, 3: Buy 100%
        # 4: Sell 25%, 5: Sell 50%, 6: Sell 100%, 7: Close position
        self.num_actions = 8
        self.action_meanings = {
            0: "hold",
            1: "buy_25",
            2: "buy_50",
            3: "buy_100",
            4: "sell_25",
            5: "sell_50",
            6: "sell_100",
            7: "close",
        }

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

        # Initialize Rainbow DQN
        self.rainbow = RainbowDQN(
            state_dim=self.state_dim,
            action_dim=self.num_actions,
            config=config,
            device=device,
        )

        # Training metrics
        self.training_metrics = {
            "episode_rewards": [],
            "portfolio_values": [],
            "sharpe_ratios": [],
            "losses": [],
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
        """Train the Rainbow DQN agent"""
        env = TradingEnvironmentRainbow(train_data, self.config)

        # Training loop
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            losses = []

            while True:
                # Select action
                action = self.rainbow.select_action(state)

                # Take step
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Store transition
                self.rainbow.store_transition(state, action, reward, next_state, done)

                # Train
                if len(self.rainbow.memory) > self.config.get("learning_starts", 1000):
                    if episode_steps % self.config.get("train_freq", 4) == 0:
                        loss_dict = self.rainbow.train()
                        if loss_dict:
                            losses.append(loss_dict["loss"])

                state = next_state

                if done:
                    break

            # Record metrics
            self.training_metrics["episode_rewards"].append(episode_reward)
            self.training_metrics["portfolio_values"].append(info["portfolio_value"])

            if losses:
                self.training_metrics["losses"].extend(losses)

            # Calculate Sharpe ratio
            if len(env.portfolio_values) > 20:
                returns = (
                    np.diff(env.portfolio_values[-20:]) / env.portfolio_values[-21:-1]
                )
                sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-6)
                self.training_metrics["sharpe_ratios"].append(sharpe)

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_metrics["episode_rewards"][-100:])
                portfolio_return = (
                    info["portfolio_value"] / env.initial_balance - 1
                ) * 100
                avg_loss = (
                    np.mean(self.training_metrics["losses"][-100:])
                    if self.training_metrics["losses"]
                    else 0
                )

                print(f"Episode {episode}/{num_episodes}")
                print(f"Average Reward: {avg_reward:.3f}")
                print(f"Portfolio Return: {portfolio_return:.2f}%")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Actions taken: {env.action_counts}")
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
                self.save(f"rainbow_model_episode_{episode}.pth")

    async def evaluate(
        self, test_data: np.ndarray, render: bool = False
    ) -> Dict[str, float]:
        """Evaluate the agent"""
        env = TradingEnvironmentRainbow(test_data, self.config, training=False)

        state = env.reset()
        done = False

        while not done:
            action = self.rainbow.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)

            if render:
                env.render()

        # Calculate metrics
        total_return = (info["portfolio_value"] / env.initial_balance - 1) * 100

        if len(env.portfolio_values) > 1:
            returns = np.diff(env.portfolio_values) / env.portfolio_values[:-1]
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-6)
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        peak = np.maximum.accumulate(env.portfolio_values)
        drawdown = (env.portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        # Win rate
        winning_trades = sum(1 for t in env.trades if t["pnl"] > 0)
        win_rate = winning_trades / len(env.trades) * 100 if env.trades else 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(env.trades),
            "action_distribution": env.action_counts,
            "final_portfolio_value": info["portfolio_value"],
        }

    def save(self, filepath: str):
        """Save model and metrics"""
        self.rainbow.save(filepath)

        # Save metrics
        import pickle

        metrics_path = filepath.replace(".pth", "_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pickle.dump(self.training_metrics, f)

    def load(self, filepath: str):
        """Load model and metrics"""
        self.rainbow.load(filepath)

        # Load metrics
        import pickle

        metrics_path = filepath.replace(".pth", "_metrics.pkl")
        try:
            with open(metrics_path, "rb") as f:
                self.training_metrics = pickle.load(f)
        except FileNotFoundError:
            pass


class TradingEnvironmentRainbow:
    """Trading environment for Rainbow DQN"""

    def __init__(self, data: np.ndarray, config: Dict, training: bool = True):
        self.data = data
        self.config = config
        self.training = training

        # Environment parameters
        self.initial_balance = config.get("initial_balance", 10000)
        self.transaction_fee = config.get("transaction_fee", 0.001)
        self.lookback_window = config.get("lookback_window", 50)

        # Action space (discrete)
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

        # Action tracking
        self.action_counts = {i: 0 for i in range(8)}

        # State tracking
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = self.lookback_window
        self.max_steps = len(self.data) - 1

        self.balance = self.initial_balance
        self.position = 0.0
        self.avg_entry_price = 0.0

        self.portfolio_values = [self.initial_balance]
        self.trades = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state"""
        # Market data
        market_data = self.data[
            self.current_step - self.lookback_window : self.current_step
        ].flatten()

        # Portfolio state
        current_price = self.data[self.current_step, 3]
        position_value = self.position * current_price
        total_value = self.balance + position_value

        # Profit/loss
        if self.position != 0 and self.avg_entry_price != 0:
            pnl = (current_price - self.avg_entry_price) / self.avg_entry_price
        else:
            pnl = 0

        portfolio_features = np.array(
            [
                self.balance / self.initial_balance,
                self.position,
                total_value / self.initial_balance,
                pnl,
            ]
        )

        return np.concatenate([market_data, portfolio_features])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute discrete action"""
        self.action_counts[action] += 1

        current_price = self.data[self.current_step, 3]
        prev_value = self.balance + self.position * current_price

        # Execute action
        self._execute_action(action, current_price)

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False

        new_price = self.data[self.current_step, 3] if not done else current_price

        # Calculate new portfolio value
        position_value = self.position * new_price
        total_value = self.balance + position_value
        self.portfolio_values.append(total_value)

        # Calculate reward
        reward = (total_value - prev_value) / prev_value * 100

        # Risk penalty for holding large positions
        if abs(self.position) > 0.8:
            reward -= 0.1

        # Check if portfolio is wiped out
        if total_value <= self.initial_balance * 0.5:
            done = True
            reward -= 10  # Large penalty for major loss

        info = {
            "portfolio_value": total_value,
            "position": self.position,
            "balance": self.balance,
            "action": action,
        }

        return self._get_state(), reward, done, info

    def _execute_action(self, action: int, current_price: float):
        """Execute the discrete action"""
        transaction_cost = 0

        if action == 0:  # Hold
            return

        elif action in [1, 2, 3]:  # Buy actions
            # Determine buy amount
            if action == 1:
                buy_fraction = 0.25
            elif action == 2:
                buy_fraction = 0.5
            else:  # action == 3
                buy_fraction = 1.0

            # Calculate maximum we can buy
            max_position = self.config.get("max_position_size", 1.0)
            available_balance = self.balance * buy_fraction

            if self.position < max_position and available_balance > 0:
                # Calculate position size
                position_to_buy = min(
                    available_balance / current_price, max_position - self.position
                )

                # Update position and balance
                cost = position_to_buy * current_price
                transaction_cost = cost * self.transaction_fee

                if cost + transaction_cost <= self.balance:
                    # Update average entry price
                    if self.position > 0:
                        total_cost = self.position * self.avg_entry_price + cost
                        self.position += position_to_buy
                        self.avg_entry_price = total_cost / self.position
                    else:
                        self.position = position_to_buy
                        self.avg_entry_price = current_price

                    self.balance -= cost + transaction_cost

                    self.trades.append(
                        {
                            "step": self.current_step,
                            "action": "buy",
                            "price": current_price,
                            "size": position_to_buy,
                            "cost": cost + transaction_cost,
                        }
                    )

        elif action in [4, 5, 6]:  # Sell actions
            if self.position > 0:
                # Determine sell amount
                if action == 4:
                    sell_fraction = 0.25
                elif action == 5:
                    sell_fraction = 0.5
                else:  # action == 6
                    sell_fraction = 1.0

                position_to_sell = self.position * sell_fraction

                # Calculate proceeds
                proceeds = position_to_sell * current_price
                transaction_cost = proceeds * self.transaction_fee

                # Update position and balance
                self.position -= position_to_sell
                self.balance += proceeds - transaction_cost

                # Calculate PnL for this trade
                pnl = (
                    current_price - self.avg_entry_price
                ) * position_to_sell - transaction_cost

                self.trades.append(
                    {
                        "step": self.current_step,
                        "action": "sell",
                        "price": current_price,
                        "size": position_to_sell,
                        "proceeds": proceeds - transaction_cost,
                        "pnl": pnl,
                    }
                )

                # Reset avg entry price if position closed
                if self.position < 0.001:
                    self.position = 0
                    self.avg_entry_price = 0

        elif action == 7:  # Close position
            if self.position != 0:
                proceeds = abs(self.position) * current_price
                transaction_cost = proceeds * self.transaction_fee

                if self.position > 0:
                    # Selling to close long
                    self.balance += proceeds - transaction_cost
                    pnl = (
                        current_price - self.avg_entry_price
                    ) * self.position - transaction_cost
                else:
                    # Buying to close short
                    self.balance -= proceeds + transaction_cost
                    pnl = (self.avg_entry_price - current_price) * abs(
                        self.position
                    ) - transaction_cost

                self.trades.append(
                    {
                        "step": self.current_step,
                        "action": "close",
                        "price": current_price,
                        "size": abs(self.position),
                        "pnl": pnl,
                    }
                )

                self.position = 0
                self.avg_entry_price = 0

    def render(self):
        """Display current state"""
        current_price = self.data[self.current_step, 3]
        position_value = self.position * current_price
        total_value = self.balance + position_value

        print(f"\nStep: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Position: {self.position:.4f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position Value: ${position_value:.2f}")
        print(f"Total Portfolio: ${total_value:.2f}")
        print(f"Return: {(total_value/self.initial_balance - 1)*100:.2f}%")
        print(f"Action Counts: {self.action_counts}")
