#!/usr/bin/env python3
"""Train DQN on CartPole-v1 and save weights to ``models/cartpole_dqn.pt``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from minimal_rl_agent.train import MODEL_SAVE_PATH, train_cartpole


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CartPole DQN")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    args = parser.parse_args()

    Path("models").mkdir(parents=True, exist_ok=True)
    train_cartpole(episodes=args.episodes)
    print(f"Wrote weights to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
