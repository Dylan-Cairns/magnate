# Tech Context

## Stack

- Node.js 20+
- TypeScript (strict)
- React + Vite
- Vitest
- ESLint + Prettier
- Python 3.11+ (`.venv`)
- PyTorch + NumPy

## Layout

- Engine + browser app: `src/`
- Bridge runtime: `src/bridge/`
- Python trainer/tooling: `trainer/`, `scripts/`
- Bridge contract: `contracts/`

## Tooling Notes

- Package manager: Yarn
- JS scripts: `dev`, `build`, `bridge`, `test`, `lint`, `typecheck`, `format`
- Python bootstrap:
  - `scripts/setup_python_env.ps1`
  - `./.venv/Scripts/Activate.ps1`
- Active Python entrypoints:
  - `python -m scripts.eval`
  - `python -m scripts.eval_suite` (`--workers` for deterministic parallel sharding)
  - `python -m scripts.search_teacher_sweep` (`--jobs` preset parallelism, forwards `--workers`)
  - `python -m scripts.generate_teacher_data`
  - `python -m scripts.run_td_loop` (collect -> train -> eval orchestration; `--collect-workers` shards replay collection across CPUs; `--cloud` applies fixed 8 vCPU worker profile; `--eval-first-last-checkpoints` emits begin/end improvement summary; `--train-value-target-mode td-lambda` enables TD(lambda) path)
  - `python -m scripts.smoke_trainer`

## Constraints

- Static deployment target (no gameplay backend).
- Deterministic gameplay required for replay/eval/training.
- Rule semantics stay in TS unless explicitly re-approved.
- Python training scripts are fail-fast and expect an active project virtualenv.

## Known Gaps

- TD replay/train/eval orchestration exists, but automated online replay refresh is not wired yet.
- `td-search` exists, but current form is still rollout-guided and needs stronger search/value coupling and throughput optimization.
- Search baseline promotion thresholds still need repeated confirmation.
- Browser deployment path for learned TD models is not implemented yet.

_Updated: 2026-03-01._
