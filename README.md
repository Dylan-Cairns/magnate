# Magnate

Single-player Magnate with a deterministic TypeScript engine, browser UI, and Python training stack.

## At A Glance

- Browser game is playable with selectable bot profiles: Easy, Medium, and Hard
  (rollout search with heuristic v2), plus an Experimental trained TD bot that
  supports the standard ruleset only.
- New games support the standard ruleset or the extended ruleset, which adds the
  four Court property cards.
- Action and submenu hovers preview affected cards, resources, and destinations;
  placement ghosts distinguish incomplete deeds from completed properties.
- Games autosave locally at human decision windows and restore to a settled
  board; New Game replaces the save. Saves do not provide offline app loading.
- TypeScript engine is the canonical rules implementation; Python training and
  evaluation call it through the Node bridge.
- Training progression is bootstrap or recalibration with
  `scripts.run_td_loop`, then self-play iteration with
  `scripts.run_td_loop_selfplay`.
- Promoted checkpoints are registered in
  `models/td_checkpoints/manifest.json`. Self-play uses checkpoint selection,
  generator gating, and replay windows, with `td-lambda` value targets by
  default.

## Quickstart

1. Install [fnm](https://github.com/Schniz/fnm) and enable its shell integration.
2. From the repo root, run `fnm install` and `fnm use`; `.nvmrc` pins Node `22.23.1`.
3. Run `corepack enable` and `corepack install` to activate the `package.json` Yarn `4.15.0` pin.
4. `yarn install`
5. `yarn dev`
6. `yarn test`
7. Set up Python with `.\scripts\setup_python_env.ps1` on Windows, or create `.venv` manually on macOS or Linux.

Use [memoryBank/techContext.md](memoryBank/techContext.md) for tooling context and links to runbooks.

## Common Commands

- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`
- TypeScript browser-bot head-to-head eval: `yarn bot:eval head-to-head --config configs/bot-eval/head-to-head.example.json`
- TypeScript rollout-search sweep: `yarn bot:eval rollout-search-sweep --config configs/bot-eval/rollout-search-width-sweep.example.json`
- Sharded TD replay export: `yarn bot:eval collect-td-replay-sharded --config configs/bot-eval/collect-td-replay.v2-hard.json --workers 8 --shard-games 1`
- Strategic-position characterization: `yarn bot:eval strategic-positions --repetitions 1`
- Replay one recorded TypeScript bot game: `yarn bot:eval replay --artifact artifacts/ts-bot-evals/<run>/matchup.json --game-id pair-0001-candidate-as-a`
- Python test: `.\.venv\Scripts\python -m pytest`
- Python lint: `.\.venv\Scripts\python -m ruff check scripts trainer trainer_tests`
- Python typecheck: `.\.venv\Scripts\python -m pyright -p .`
- Register promoted TD checkpoint: `.\.venv\Scripts\python -m scripts.promote_td_checkpoint --help`

## Credits & License

Magnate is an unofficial, noncommercial fan implementation of the Decktet game
[Magnate](https://decktet.wikidot.com/game:magnate), designed by Cristyn Magnus
with additional development by P.D. Magnus. The Decktet is created by P.D.
Magnus; its card art and game material are used under a
[Creative Commons Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/4.0/)
license. The same credits appear in the in-app info menu.

This project is licensed under CC BY-NC-SA 4.0; see [LICENSE](LICENSE). Bundled
third-party software notices are listed in
[public/third-party-notices.txt](public/third-party-notices.txt).

## Source-of-Truth Docs

- Agent workflow: [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md)
- Tooling context: [memoryBank/techContext.md](memoryBank/techContext.md)
- Current project context: [memoryBank/activeContext.md](memoryBank/activeContext.md)
- Rules reference: [memoryBank/magnateRules.md](memoryBank/magnateRules.md)
- Bridge contract: [memoryBank/bridgeInterfaceContract.md](memoryBank/bridgeInterfaceContract.md)
- Strategic-state design: [docs/design/strategic-state-summary-v0.md](docs/design/strategic-state-summary-v0.md)
- District-symmetry design: [docs/design/district-symmetry.md](docs/design/district-symmetry.md)
- Runbooks: [docs/runbooks/](docs/runbooks/)
