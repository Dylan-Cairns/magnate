# Magnate

Single-player Magnate with a deterministic TypeScript engine, browser UI, and Python training stack.

## At A Glance

- Browser game is playable with selectable bot profiles.
- TypeScript engine is the canonical rules implementation.
- Python training and evaluation call the engine through the Node bridge.
- Training progression is bootstrap or recalibration with `scripts.run_td_loop`, then self-play-focused iteration with `scripts.run_td_loop_selfplay`.
- TD promoted checkpoints use `models/td_checkpoints/manifest.json` as the checked-in warm-start and opponent-pool registry.
- Self-play training uses checkpoint selection, generator gating, and small replay windows.
- TD value training defaults to sequence-aware `td-lambda` targets (`--train-value-target-mode td-lambda`).

## Quickstart

1. Install Node `22.23.1` (the version pinned in `.nvmrc`) and Yarn classic.
2. `yarn install`
3. `yarn dev`
4. `yarn test`
5. Set up Python with `.\scripts\setup_python_env.ps1` on Windows, or create `.venv` manually on macOS or Linux.

Use [memoryBank/techContext.md](memoryBank/techContext.md) for tooling context and links to runbooks.

## Common Commands

- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`
- TypeScript browser-bot head-to-head eval: `yarn bot:eval head-to-head --config configs/bot-eval/head-to-head.example.json`
- TypeScript rollout-search sweep: `yarn bot:eval rollout-search-sweep --config configs/bot-eval/rollout-search-width-sweep.example.json`
- TypeScript rollout-search TD replay export: `yarn bot:eval collect-td-replay --config configs/bot-eval/collect-td-replay.rollout-search.example.json`
- Strategic-position characterization: `yarn bot:eval strategic-positions --repetitions 1`
- Override timed heartbeat cadence for supported bot-eval commands: append `--progress-interval-seconds 10` (`0` disables heartbeats); strategic-position characterization reports per decision instead.
- Replay one recorded TypeScript bot game: `yarn bot:eval replay --artifact artifacts/ts-bot-evals/<run>/matchup.json --game-id pair-0001-candidate-as-a`
- Python test: `.\.venv\Scripts\python -m pytest`
- Python lint: `python -m ruff check scripts trainer trainer_tests`
- Register promoted TD checkpoint: `.\.venv\Scripts\python -m scripts.promote_td_checkpoint --help`

## Source-of-Truth Docs

- Tooling context: [memoryBank/techContext.md](memoryBank/techContext.md)
- Current project context: [memoryBank/activeContext.md](memoryBank/activeContext.md)
- Rules reference: [memoryBank/magnateRules.md](memoryBank/magnateRules.md)
- Bridge contract: [memoryBank/bridgeInterfaceContract.md](memoryBank/bridgeInterfaceContract.md)
- Strategic-state design: [docs/design/strategic-state-summary-v0.md](docs/design/strategic-state-summary-v0.md)
- Runbooks: [docs/runbooks/](docs/runbooks/)
