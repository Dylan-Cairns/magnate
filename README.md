# Magnate (Web) + TD/Keldon Training Pivot

Single-player Magnate with a deterministic TypeScript engine, browser UI, and Python training stack.

## Project Direction

- TypeScript engine is the canonical rules implementation.
- Python training/eval calls TS through a Node bridge.
- Current mission is singular:
  - build a TD-Gammon / Keldon-like training pipeline,
  - keep rollout search as warm-start signal only,
  - deploy a stronger learned bot in this web app.

## Current Status

- Browser game is playable; default web bot is rollout-eval search.
- Bridge runtime is implemented (`metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`).
- Python side now keeps only `random`, `heuristic`, and `search` policy paths.
- PPO, MCTS, and guidance/distillation codepaths were intentionally removed.

## Local Commands

- Install JS deps: `yarn`
- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`

## Python Setup

From repo root:

1. `./scripts/setup_python_env.ps1`
2. `./.venv/Scripts/Activate.ps1`

macOS/Linux equivalent:

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`

## Python Commands (Current)

With `.venv` active:

- Smoke: `python -m scripts.smoke_trainer`
- Canonical side-swapped eval: `python -m scripts.eval_suite --games-per-side 200 --workers 2 --candidate-policy search --opponent-policy heuristic`
- Search sweep: `python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse`
- Teacher data generation (warm-start labels): `python -m scripts.generate_teacher_data --games 200 --teacher-policy search --teacher-players both --out artifacts/teacher_data/teacher_search.jsonl`

Use `--help` on each script for full options.

## Source-of-Truth Docs

- Agent manifest: `AGENTS.md`
- Memory workflow: `docs/AGENT_GUIDE.md`
- Training handoff/restart: `docs/TRAINING_HANDOFF.md`
- Training plan: `docs/TRAINING_PLAN_SEARCH_FIRST.md`
- Command cookbook: `docs/TRAINING_COMMANDS.md`
- Encoding contract: `docs/TRAINING_ENCODING.md`
- Memory Bank: `memoryBank/`
- Rules reference: `memoryBank/magnateRules.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
