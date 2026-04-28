"""Training loop for CartPole-v1."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch

from minimal_rl_agent.agent import Agent

# Default checkpoint location (script may create `models/` first).
MODEL_SAVE_PATH = "models/cartpole_dqn.pt"


def train_cartpole(episodes: int = 300) -> list[float]:
    """
    Run DQN on CartPole-v1.

    Loop: observe → act → step env → remember → learn → decay exploration.

    Saves policy weights to ``models/cartpole_dqn.pt`` (parent dirs created if needed).
    Returns total reward per episode.
    """
    env = gym.make("CartPole-v1")
    state_dim = int(env.observation_space.shape[0])  # type: ignore[union-attr]
    n_actions = int(env.action_space.n)  # type: ignore[union-attr]
    agent = Agent(state_dim, n_actions)

    episode_rewards: list[float] = []

    for episode in range(episodes):
        state, _info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, float(reward), next_state, done)
            agent.learn()
            agent.decay_epsilon()

            state = next_state
            ep_reward += float(reward)

        episode_rewards.append(ep_reward)

        if (episode + 1) % 10 == 0:
            print(f"episode {episode + 1}/{episodes}  return={ep_reward:.1f}  epsilon={agent.epsilon:.4f}")

    env.close()

    save_path = Path(MODEL_SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": agent.model.state_dict()}, save_path)

    return episode_rewards
