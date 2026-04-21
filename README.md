# Magnate

Single-player Magnate with a deterministic TypeScript engine, browser UI, and Python training stack.

## At A Glance

- Browser game is playable; default bot is `TD Search Fast`.
- Browser bot profiles currently include `TD Search Fast`, `TD Search`, `Rollout Search`, and `Random legal`.
- TypeScript engine is the canonical rules implementation.
- Python training and evaluation call the engine through the Node bridge.
- Training progression is bootstrap or recalibration with `scripts.run_td_loop`, then self-play-focused iteration with `scripts.run_td_loop_selfplay`.

## Quickstart

1. Install Node `22.12.0` and Yarn classic.
2. `yarn install`
3. `yarn dev`
4. `yarn test`
5. Set up Python with `.\scripts\setup_python_env.ps1` on Windows, or create `.venv` manually on macOS or Linux.

Use [memoryBank/techContext.md](memoryBank/techContext.md) for the full setup, training, evaluation, wrapper, and recovery runbooks.

## Common Commands

- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`
- Python test: `.\.venv\Scripts\python -m pytest`
- Python lint: `python -m ruff check scripts trainer trainer_tests`

## Source-of-Truth Docs

- Operational runbook: [memoryBank/techContext.md](memoryBank/techContext.md)
- Current project context: [memoryBank/activeContext.md](memoryBank/activeContext.md)
- Rules reference: [memoryBank/magnateRules.md](memoryBank/magnateRules.md)
- Bridge contract: [memoryBank/bridgeInterfaceContract.md](memoryBank/bridgeInterfaceContract.md)
