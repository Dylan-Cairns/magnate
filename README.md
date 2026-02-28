# Magnate (Web) + Bot Training

Single-player Magnate with a deterministic TypeScript engine, browser UI, and Python training stack.

## Project Direction

- TypeScript engine is the canonical rules implementation.
- Python training calls the TS engine through a Node bridge.
- Bridge contract stays intentionally small and versioned.
- Do not duplicate full rules logic in Python unless throughput proves to be a hard bottleneck.

## Current Status

- Deterministic engine loop is implemented (`setup`, `legalActions`, `applyAction`, `advanceToDecision`, scoring/terminal).
- Browser game shell is playable with policy-driven bots.
- Browser PPO champion profile is enabled and currently default.
- Browser rollout-eval search profile is available for direct in-UI play/testing (explicit failure surface, no hidden fallback).
- Bridge runtime is implemented (`metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`).
- Trainer supports:
  - random/heuristic evaluation
  - BC warm-start
  - REINFORCE fine-tuning
  - PPO training
  - additive search and MCTS policy evaluation/benchmarking

## Local Commands

- Install JS deps: `yarn`
- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`

## Python Setup

From repo root:

1. Create/update env (PowerShell): `.\scripts\setup_python_env.ps1`
2. Activate env (PowerShell): `.\.venv\Scripts\Activate.ps1`

macOS/Linux equivalent:

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`

## Common Training Commands

With `.venv` active:

- Smoke check: `python -m scripts.smoke_trainer`
- Eval matchup: `python -m scripts.eval --games 50 --player-a-policy heuristic --player-b-policy random`
- Eval MCTS vs heuristic: `python -m scripts.eval --games 200 --player-a-policy mcts --player-b-policy heuristic`
- Canonical benchmark: `python -m scripts.benchmark --candidate-policy ppo --candidate-checkpoint artifacts/ppo_checkpoint.pt`
- BC warm-start: `python -m scripts.train --games 50`
- RL fine-tune from BC: `python -m scripts.finetune --checkpoint-in artifacts/bc_checkpoint.json --checkpoint-out artifacts/rl_checkpoint.json --episodes 300`
- PPO training: `python -m scripts.train_ppo --checkpoint-out artifacts/ppo_checkpoint.pt --episodes 1024 --episodes-per-update 32`
- Teacher-data generation (for distillation): `python -m scripts.generate_teacher_data --games 200 --teacher-policy search --teacher-players both --out artifacts/teacher_data/teacher_search.jsonl`
- Train search guidance checkpoint from teacher data: `python -m scripts.train_search_guidance --samples-in artifacts/teacher_data/teacher_search.jsonl --checkpoint-out artifacts/search_guidance_checkpoint.pt`
- Run full unattended guidance A/B pipeline: `python -m scripts.run_guidance_ab_pipeline --run-label guidance-pilot --games 200`

Use `--help` on each script for full options.

Guidance-enabled search and MCTS:

- `scripts.eval`, `scripts.benchmark`, and `scripts.generate_teacher_data` now support:
  - `--search-guidance-checkpoint`
  - `--mcts-guidance-checkpoint`
  - `--guidance-temperature`
- Guidance checkpoints use PPO checkpoint format (`magnate_ppo_policy_v1`) and can be trained with `scripts.train_search_guidance`.

## Source-of-Truth Docs

- Agent manifest: `AGENTS.md`
- Memory workflow: `docs/AGENT_GUIDE.md`
- Training handoff/restart context: `docs/TRAINING_HANDOFF.md`
- Memory Bank: `memoryBank/`
- Rules reference: `memoryBank/magnateRules.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
- Training encoding: `docs/TRAINING_ENCODING.md`
