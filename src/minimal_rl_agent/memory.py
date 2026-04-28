"""Replay memory: store past (s, a, r, s', done) tuples for batch learning."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(frozen=True)
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayMemory:
    """Fixed-size FIFO buffer; `sample` draws random minibatches."""

    def __init__(self, capacity: int) -> None:
        self._storage: Deque[Experience] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._storage.append(
            Experience(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        if batch_size > len(self._storage):
            raise ValueError("batch_size is larger than stored experiences")
        batch = random.sample(self._storage, batch_size)
        states = np.stack([e.state for e in batch])
        actions = np.array([e.action for e in batch], dtype=np.int64)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.stack([e.next_state for e in batch])
        dones = np.array([e.done for e in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones
