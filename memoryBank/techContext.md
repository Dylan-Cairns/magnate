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
  - `python -m scripts.eval_suite` (loop default is fixed-size certify flow; `--workers` for deterministic parallel sharding; per-worker thread caps via `--worker-torch-threads`, `--worker-torch-interop-threads`, `--worker-blas-threads`)
  - `python -m scripts.search_teacher_sweep` (`--jobs` preset parallelism, forwards `--workers`; default `--python-bin` resolves cross-platform via `sys.executable` then `.venv` paths)
  - `python -m scripts.generate_teacher_data` (teacher policy must support root action probabilities for label generation)
  - `python -m scripts.export_browser_model_pack` (export TD value checkpoint to static browser model-pack artifacts in `public/model-packs/`)
  - `python -m scripts.export_browser_td_search_pack` (export TD value+opponent checkpoints to static browser td-search model-pack artifacts in `public/model-packs/`)
  - `python -m scripts.run_td_loop` (chunked collect/train -> single promotion eval orchestration; `--chunks-per-loop`, `--collect-workers`, `--eval-workers`; `--cloud --cloud-vcpus 8|16|32` applies preset worker/thread profile; `--progress-heartbeat-minutes` uses minute-based stage heartbeats; `--train-value-target-mode td-lambda` enables TD(lambda) path; default `--promotion-min-ci-low` is `0.5`; replay regime currently `chunk-local`)
  - `python -m scripts.smoke_trainer`

## Constraints

- Static deployment target (no gameplay backend).
- Deterministic gameplay required for replay/eval/training.
- Rule semantics stay in TS unless explicitly re-approved.
- Python training scripts are fail-fast and expect an active project virtualenv.
- `scripts.train_td` fail-fast enforces Python 3.11+ and active `.venv` at startup.

## Known Gaps

- TD replay/train/eval orchestration exists, but automated online replay refresh is not wired yet.
- `td-search` exists, but current form is still rollout-guided and needs stronger search/value coupling and throughput optimization.
- Search baseline promotion thresholds still need repeated confirmation.
- Browser `td-value` and `td-search` deployment paths exist via static model-pack export/loading; remaining gap is browser runtime performance tuning for `td-search`.

_Updated: 2026-03-05._
