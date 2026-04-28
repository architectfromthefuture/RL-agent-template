#!/usr/bin/env python3
"""Evaluate a saved CartPole DQN checkpoint (greedy policy)."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from minimal_rl_agent.evaluate import evaluate_cartpole

DEFAULT_MODEL = "models/cartpole_dqn.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CartPole DQN")
    parser.add_argument("--render", action="store_true", help="Open Gymnasium window")
    args = parser.parse_args()

    returns = evaluate_cartpole(DEFAULT_MODEL, episodes=5, render=args.render)
    mean_r = statistics.mean(returns)
    print(f"returns={returns}")
    print(f"mean return (5 episodes): {mean_r:.2f}")


if __name__ == "__main__":
    main()
