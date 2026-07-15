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
- Strategic-position diagnostics support `--positions`, `--variants`, and
  `--start-repetition` for targeted seed extensions. The command does not
  resume or merge prior output; targeted extensions should use a separate
  output directory because reusing one overwrites its files.
- Opt-in strategic variant `td-root-search-v2-800-visits` clones TD V2 Medium
  and changes only sampled worlds from 10 to 50. It provides 800 root visits
  without joining the default variant set.
- Opt-in suffixes `-heuristic-root`, `-heuristic-rollout`, and
  `-heuristic-root-rollout` select heuristic-v2 guidance for the named hooks at
  the same 800-visit budget. Leaf remains TD-guided; when terminal rate is 1 it
  is not invoked.
- `strategic-forced-rollouts` accepts explicit position, repetition-ID, and
  action-local scenario-ID lists. It forces preserve/overwrite through one
  shared hidden-world/seed schedule under TD and heuristic-v2 rollout play,
  requires terminal completion, and writes detailed JSON plus an aggregate
  Markdown summary under ignored bot-eval artifacts.

## Core Commands

- Install JS deps: `yarn install`
- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`
- Strategic position smoke check: `yarn bot:eval strategic-positions --repetitions 1`
- Strategic position stability screen: `yarn bot:eval strategic-positions --repetitions 8`
- Strategic forced-rollout trace: `yarn bot:eval strategic-forced-rollouts --positions known-hand-optionality-holdout-original,known-hand-optionality-holdout-mirror --repetitions 0,1`
- TD district-symmetry audit: `yarn bot:eval td-symmetry --replay-dir <replay-run> --sample-size <n> --sampling-seed <seed>`
- Python test: `.\.venv\Scripts\python -m pytest`
- Python targeted test: `.\.venv\Scripts\python -m pytest trainer_tests/<test_file>.py`
- Python lint: `python -m ruff check scripts trainer trainer_tests`
- Python lint autofix: `python -m ruff check --fix scripts trainer trainer_tests`
- Python typecheck: `.\.venv\Scripts\python -m pyright -p .`
- Promote/register checkpoint pair: `.\.venv\Scripts\python -m scripts.promote_td_checkpoint --help`
- Export browser TD-root model pack: `.\.venv\Scripts\python -m scripts.export_browser_td_root_pack --value-checkpoint <value.pt> --opponent-checkpoint <opponent.pt> --set-default`
- Reconstruct optimizer-free trainer checkpoints from a browser TD-root pack: `.\.venv\Scripts\python -m scripts.reconstruct_browser_td_root_checkpoints --manifest <pack-manifest.json> --output-dir <directory>`

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

- Search baseline promotion thresholds still need repeated confirmation.
- Browser TD deployment needs a current exported `td-root-search-v1` model pack committed under `public/model-packs/`; legacy checked-in browser model artifacts have been removed.
- Direct TypeScript TD-root matchup throughput can still improve: bot-eval can load local model packs in Node and child-process workers, but each individual Node search decision remains synchronous.

_Updated: 2026-07-14._
