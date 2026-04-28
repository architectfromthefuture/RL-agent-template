"""Tests for replay memory."""

from __future__ import annotations

import numpy as np
import pytest

from minimal_rl_agent.memory import ReplayMemory


def _experience(i: int = 0) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
    state = np.asarray([float(i), 0.0, 0.0, 0.0], dtype=np.float32)
    next_state = np.asarray([float(i + 1), 0.0, 0.0, 0.0], dtype=np.float32)
    return state, 0, 1.0, next_state, False


def test_memory_stores_and_reports_length() -> None:
    mem = ReplayMemory(capacity=100)
    assert len(mem) == 0
    mem.push(*_experience(0))
    mem.push(*_experience(1))
    assert len(mem) == 2


def test_memory_sample_batch_size() -> None:
    mem = ReplayMemory(capacity=100)
    for i in range(20):
        mem.push(*_experience(i))
    states, actions, rewards, next_states, dones = mem.sample(8)
    assert states.shape == (8, 4)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, 4)
    assert dones.shape == (8,)


def test_memory_sample_requires_enough_data() -> None:
    mem = ReplayMemory(capacity=50)
    mem.push(*_experience())
    with pytest.raises(ValueError):
        mem.sample(4)
