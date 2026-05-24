# Tech Context

## Stack

- Node.js 20.19.0+
- TypeScript strict
- React + Vite
- Vitest
- Python 3.12+ through the project `.venv`
- Pytest
- Ruff
- Pyright
- PyTorch + NumPy
- ESLint + Prettier

## Layout

- Engine + browser app: `src/`
- Browser policies: `src/policies/`
- TypeScript bot evaluation: `src/botEval/`
- Bridge runtime: `src/bridge/`
- Python trainer/tooling: `trainer/`, `scripts/`
- Trainer tests: `trainer_tests/`
- Bridge contract: `contracts/`
- Operational runbooks: `docs/runbooks/`

## Tooling Notes

- Package manager: Yarn classic.
- JS scripts: `dev`, `build`, `bridge`, `bot:eval`, `test`, `lint`, `typecheck`, `format`.
- GitHub Pages deploy: `.github/workflows/deploy_pages.yml` gates deployment on `yarn test`, `yarn lint`, then `yarn build` with Node `20.19.0`.
- VS Code workspace pins `${workspaceFolder}\\.venv\\Scripts\\python.exe`.
- Checked-in pyright scope covers `trainer/` plus trainer-side tests in `trainer_tests/`; some `scripts/` orchestration remains outside checked-in pyright scope.
- TypeScript bridge output is canonical. Python models the consumed subset in `trainer/bridge_payloads.py`.

## Core Commands

- Install JS deps: `yarn install`
- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`
- Python test: `.\.venv\Scripts\python -m pytest`
- Python targeted test: `.\.venv\Scripts\python -m pytest trainer_tests/<test_file>.py`
- Python lint: `python -m ruff check scripts trainer trainer_tests`
- Python lint autofix: `python -m ruff check --fix scripts trainer trainer_tests`
- Python typecheck: `.\.venv\Scripts\python -m pyright -p .`
- Promote/register checkpoint pair: `.\.venv\Scripts\python -m scripts.promote_td_checkpoint --help`

## Python Workflow

- Use the project `.venv` for any Python command in this repo.
- When changing Python code, run targeted pytest tests for touched behavior plus Ruff and Pyright before handoff.
- If the change touches Python code outside checked-in pyright scope, note that explicitly in handoff.

## Checkpoint Manifest

- `models/td_checkpoints/manifest.json` is the canonical checked-in registry for TD checkpoint warm starts and opponent-pool entries.
- Manifest schema v2 uses `defaultWarmStart`, `opponentPool`, and `checkpoints.<key>.value` / `.opponent`.
- Referenced checkpoint files under `models/td_checkpoints/<key>/` should be committed when the manifest changes.
- Successful promotions in TD loop scripts copy accepted checkpoint pairs into `models/td_checkpoints/<key>/` and update the manifest unless `--disable-manifest-promotion` is set.

## Runbooks

- Windows local setup and laptop wrappers: `docs/runbooks/windows-local.md`
- RunPod/Linux CPU setup: `docs/runbooks/runpod-linux.md`
- Python training and evaluation loops: `docs/runbooks/training-loop.md`
- TypeScript browser-bot evaluation: `docs/runbooks/bot-eval.md`

## Constraints

- Static deployment target; no gameplay backend.
- Deterministic gameplay is required for replay, eval, and training.
- Rule semantics stay in TS unless explicitly re-approved.
- Python training scripts are fail-fast and expect the active project virtualenv.
- `scripts.train_td` enforces Python 3.12+ and active `.venv` at startup.

## Known Gaps

- `td-root-search` needs full TD guidance through the shared rollout-search core.
- Search baseline promotion thresholds still need repeated confirmation.
- Browser TD deployment needs the new canonical TD-root rollout model pack wired in; legacy checked-in browser model artifacts have been removed.
- Direct TypeScript bot evaluation needs Node-local model-pack loading for serialized TD-root rollout specs.

_Updated: 2026-06-18._
