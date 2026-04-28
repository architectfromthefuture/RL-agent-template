"""Agent: model, target model, replay memory, optimizer, epsilon-greedy exploration."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from minimal_rl_agent.memory import ReplayMemory
from minimal_rl_agent.model import DQN


class Agent:
    """
    Deep Q-Network (DQN) agent for discrete actions.

    *Think + decide* happens in `act`: run the state through the model, pick the
    best action or explore randomly.
    *Learn* happens in `learn`: one gradient step from a replay batch, using the
    target network for stable bootstrap targets.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        memory_size: int = 10_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_every: int = 10,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_every = target_update_every
        self.epsilon = epsilon_start
        self._learn_steps = 0

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy: random action with probability `epsilon`, else best Q."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.model(x)
        return int(q.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        """Call once per env step (or episode, if you prefer) to anneal exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self) -> None:
        """If enough experiences exist, sample a batch and apply one DQN update."""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_states_t).max(dim=1).values
            targets = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = F.mse_loss(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_every == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        """Copy policy weights into the target network (slow-moving target for TD targets)."""
        self.target_model.load_state_dict(self.model.state_dict())
