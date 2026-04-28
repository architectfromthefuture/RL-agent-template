# Minimal RL Agent

A small reinforcement learning agent template built with **Gymnasium** and **PyTorch**. It is intentionally **minimal**: one environment (`CartPole-v1`), one algorithm (DQN-style updates), and plain code you can read in an afternoon. This is a **learning template**, not a production RL stack.

## What this repo teaches

An RL agent is a loop:

**Observe → Think → Decide → Act → Learn → Repeat**

- **Observe:** read the state from the environment.
- **Think / decide:** use the network to score actions (and optionally explore).
- **Act:** apply an action and step the environment.
- **Learn:** improve the network from past experience stored in replay memory.

The same basic pieces appear in larger RL systems; they are just split across more files and abstractions there.

## The six core pieces

| Concept | Code |
|---------|------|
| Environment | Gymnasium `env` in `train.py` / `evaluate.py` |
| State | Observation from `env.reset()` / `env.step()` |
| Policy / brain | `DQN` in `model.py` |
| Memory | `ReplayMemory` in `memory.py` |
| Learning | `Agent.learn()` — loss, backprop, optimizer |
| Exploration | Epsilon-greedy in `Agent.act()` |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training and evaluation scripts add `src/` to `sys.path` so you do not need an editable install for a quick start.

## Train

```bash
python scripts/train_cartpole.py
```

Training uses **300 episodes** by default. To try a shorter run:

```bash
python scripts/train_cartpole.py --episodes 50
```

Weights are written to **`models/cartpole_dqn.pt`** (the `models/` directory is created if needed).

## Evaluate

```bash
python scripts/evaluate_cartpole.py
```

Optional windowed rollout:

```bash
python scripts/evaluate_cartpole.py --render
```

## Project structure

```text
src/minimal_rl_agent/
  model.py      # DQN (state → action values)
  memory.py     # replay buffer
  agent.py      # act, remember, learn, target sync
  train.py      # train_cartpole()
  evaluate.py   # evaluate_cartpole()
scripts/
  train_cartpole.py
  evaluate_cartpole.py
tests/
  test_memory.py
```

## How the loop works

1. **`train_cartpole`** builds `CartPole-v1` and an `Agent`.
2. Each step: **`act(state)`** picks an action (random with probability `epsilon`, else best Q-value).
3. The transition is stored with **`remember(...)`**, then **`learn()`** may perform one gradient step if the buffer has enough samples.
4. **`decay_epsilon()`** slowly reduces random exploration.
5. The **target network** is refreshed every few updates so the TD target stays stable.

## What to change next

Reasonable first edits (still one algorithm):

- Tweak **`Agent`** hyperparameters (`gamma`, `epsilon_decay`, `batch_size`, …).
- Change hidden width in **`DQN`** (still 64×64 by default).
- Plot `episode_rewards` returned from **`train_cartpole`** in a notebook *outside* this repo if you want curves—this template does not add plotting dependencies.

Run tests:

```bash
pytest
```

---

**Honest scope:** This code is for **understanding and classroom-style experiments**. It does not claim to be production-grade control software, a general “AI,” or comparable to large industrial RL platforms.
