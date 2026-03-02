# Magnate

Single-player Magnate with a deterministic TypeScript engine, browser UI, and Python training stack.

## At A Glance

- Browser game is playable; default bot is `TD Search`.
- Browser bot profiles currently include `TD Search`, `TD Search Fast`, `Heuristic`, `Rollout Search`, and `Random legal`.
- TypeScript engine is the canonical rules implementation.
- Python training and evaluation call the engine through the Node bridge.
- Training progression is bootstrap or recalibration with `scripts.run_td_loop`, then self-play-focused iteration with `scripts.run_td_loop_selfplay`.
- TD promoted checkpoints use `models/td_checkpoints/manifest.json` as the checked-in warm-start and opponent-pool registry.
- Self-play chooses among saved training checkpoints with a cheap incumbent eval before each chunk contributes a learner candidate; the final training step is not assumed best.
- Self-play generator updates are gated at configurable block boundaries (`--generator-update-chunks`); the best candidate from the block must pass a resumable sequential incumbent test before it can drive future collection.
- Windows self-play wrapper defaults use recent replay history and 3-chunk generator blocks.
- Self-play training uses a small replay window by default (`--train-replay-window-chunks 3`), tracked as referenced chunk replays rather than duplicated merged window files. The window source can be `accepted` for gate-passing chunks or `recent` for cumulative learner training across failed generator gates.
- TD value training defaults to sequence-aware `td-lambda` targets (`--train-value-target-mode td-lambda`).

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
- Register promoted TD checkpoint: `.\.venv\Scripts\python -m scripts.promote_td_checkpoint --help`

## Source-of-Truth Docs

- Operational runbook: [memoryBank/techContext.md](memoryBank/techContext.md)
- Current project context: [memoryBank/activeContext.md](memoryBank/activeContext.md)
- Rules reference: [memoryBank/magnateRules.md](memoryBank/magnateRules.md)
- Bridge contract: [memoryBank/bridgeInterfaceContract.md](memoryBank/bridgeInterfaceContract.md)
