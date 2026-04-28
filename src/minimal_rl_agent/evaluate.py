"""Evaluate a trained checkpoint on CartPole-v1 (no exploration)."""

from __future__ import annotations

import torch
import gymnasium as gym

from minimal_rl_agent.model import DQN


def _load_state(path: str, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def evaluate_cartpole(
    model_path: str,
    episodes: int = 5,
    render: bool = False,
) -> list[float]:
    """
    Load weights and run the greedy policy (always pick best predicted action).

    If ``render`` is True, opens a window (slows runs down; needs a display).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    state_dim = int(env.observation_space.shape[0])  # type: ignore[union-attr]
    n_actions = int(env.action_space.n)  # type: ignore[union-attr]

    model = DQN(state_dim, n_actions).to(device)
    ckpt = _load_state(model_path, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    returns: list[float] = []
    for _ in range(episodes):
        state, _info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(model(x).argmax(dim=1).item())
            state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        returns.append(ep_ret)

    env.close()
    return returns
