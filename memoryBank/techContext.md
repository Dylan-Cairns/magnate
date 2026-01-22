# Tech Context

## Stack

- Node.js 22.12.0+
- TypeScript (strict)
- React + Vite
- Vitest
- ESLint + Prettier
- Ruff
- Pyright
- Python 3.12+ (`.venv`)
- PyTorch + NumPy

## Layout

- Engine + browser app: `src/`
- Bridge runtime: `src/bridge/`
- Python trainer/tooling: `trainer/`, `scripts/`
- Bridge contract: `contracts/`

## Tooling Notes

- Package manager: Yarn
- JS scripts: `dev`, `build`, `bridge`, `test`, `lint`, `typecheck`, `format`
- GitHub Pages deploy: `.github/workflows/deploy_pages.yml` runs on pushes to `main` (and manual dispatch), then gates deployment on `yarn test` + `yarn lint` before `yarn build` using Node `22.12.0`.
- VS Code workspace pins `${workspaceFolder}\\.venv\\Scripts\\python.exe`.
- Checked-in static analysis runs from the repo venv: `.\.venv\Scripts\python -m pyright -p .`.
- Checked-in pyright scope currently covers `trainer/` plus trainer-side tests in `trainer_tests/`; `scripts/` and `trainer_tests/test_eval_suite*.py` remain excluded.
- TypeScript bridge output remains canonical. Python models only the consumed subset in `trainer/bridge_payloads.py`, validates it once at ingress in `trainer/bridge_parsing.py`, and then passes typed payloads through the trainer stack.

## Local Commands

- Install JS deps: `yarn install`
- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`
- Python lint: `python -m ruff check scripts trainer trainer_tests`
- Python lint autofix: `python -m ruff check --fix scripts trainer trainer_tests`
- Python typecheck: `.\.venv\Scripts\python -m pyright -p .`

## Python Workflow

- Use the project `.venv` for any Python command in this repo.
- When changing Python code, run targeted tests for the touched behavior plus the Python Ruff and Pyright commands listed in `Local Commands`.
- If the change touches Python code outside the checked-in pyright scope, note that explicitly in handoff.

## Windows Local Setup

From repo root:

1. Install or use Node `22.12.0` with `nvm`:
   `nvm install 22.12.0`
   `nvm use 22.12.0`
2. Install Yarn classic if needed:
   `npm install -g yarn`
3. Install JS deps:
   `yarn install`
4. Install Python deps:
   `.\scripts\setup_python_env.ps1`
5. Activate the venv when you want an interactive Python shell:
   `.\.venv\Scripts\Activate.ps1`

`setup_python_env.ps1` is the recommended Windows path for local training. It installs `requirements-dev.txt` including Ruff and Pyright, installs CPU-only PyTorch from the official CPU wheel index, and routes temp and cache files into repo-local `.tmp/`, `.pip-cache/`, `.npm-cache/`, and `.yarn-cache/` folders.

## macOS/Linux Python Setup

Manual equivalent:

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements-dev.txt`

## Windows Laptop Training Runbook

Use the PowerShell wrappers so laptop-safe worker and thread settings stay separate from the Linux and RunPod flows.

- Bootstrap or recalibration loop: `.\scripts\run_td_loop_bootstrap_laptop.ps1`
- Self-play loop: `.\scripts\run_td_loop_selfplay_laptop.ps1`
- Self-play resume: `.\scripts\resume_td_loop_selfplay_laptop.ps1 -RunId <interrupted-selfplay-run-id>`

Wrapper behavior:

- requires Node `22.12.0+`, `yarn install`, and a populated `.venv`
- sets repo-local temp and cache dirs plus BLAS and OpenMP thread caps
- auto-sizes the CPU budget from logical core count
- defaults to `-CpuTargetPercent 60 -ReserveLogicalCores 2`
- maps that budget into `collect-workers`, `eval-workers`, `incumbent-eval-workers`, `train-num-threads`, and `train-num-interop-threads`
- keeps search-cost tuning explicit through loop args such as `--collect-search-worlds` and `--collect-search-depth`
- streams child collect, train, and eval output into parent logs under `artifacts/logs/`

Useful wrapper invocations:

- Inspect the resolved self-play command without running it:
  `.\scripts\run_td_loop_selfplay_laptop.ps1 -DryRun`
- Increase CPU budget while keeping headroom:
  `.\scripts\run_td_loop_selfplay_laptop.ps1 -CpuTargetPercent 70 -DryRun`
- Override loop args:
  `.\scripts\run_td_loop_selfplay_laptop.ps1 -LoopArgs @('--run-label', 'td-loop-selfplay-laptop-test', '--collect-games', '300')`

## RunPod/Linux CPU Setup

Use this flow for Linux CPU pods with a persistent `/workspace` volume.

One-time setup on the persistent volume:

```bash
cd /workspace
git clone <your-repo-url> magnate
cd /workspace/magnate

apt-get update
apt-get install -y curl ca-certificates gnupg
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs python3.12 python3.12-venv
npm install -g yarn
npm install -g npm@11.11.0

yarn install

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

export TMPDIR=/workspace/tmp
mkdir -p /workspace/tmp

python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements.txt
```

Each new pod session on the same persistent volume:

```bash
cd /workspace/magnate
git pull --ff-only
source .venv/bin/activate
export TMPDIR=/workspace/tmp
npm install -g yarn
yarn install
```

For long runs, prefer `tmux`.

## Python Entry Points

With `.venv` active:

- `python -m scripts.smoke_trainer`
  Quick trainer smoke test.
- `python -m scripts.eval`
  Simple evaluation entrypoint.
- `python -m scripts.eval_suite --mode certify --games-per-side 200 --workers 2 --candidate-policy search --opponent-policy heuristic`
  Canonical side-swapped evaluation. Supports `--mode gate|certify`, deterministic worker sharding, worker thread caps, and separate td-search checkpoints per side.
- `python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse`
  Search profile sweep for teacher-data and search tuning work.
- `python -m scripts.generate_teacher_data --games 200 --teacher-policy search --teacher-players both --out artifacts/teacher_data/teacher_search.jsonl`
  Teacher-label generation; teacher policy must emit root action probabilities.
- `python -m scripts.collect_td_self_play --games 200 --player-a-policy search --player-b-policy search --out-dir artifacts/td_replay --run-label td-replay-search`
  Replay generation for TD training primitives.
- `python -m scripts.train_td --value-replay artifacts/td_replay/<run>.value.jsonl --opponent-replay artifacts/td_replay/<run>.opponent.jsonl --steps 2000 --run-label td-v1`
  TD training primitive over replay files.
- `python -m scripts.run_td_loop --run-label td-loop-r1 --chunks-per-loop 3 --collect-games 1200 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search --promotion-min-ci-low 0.5`
  Bootstrap or recalibration loop. Supports `--collect-workers`, `--eval-workers`, `--eval-seed-start-indices`, `--train-value-target-mode td-lambda`, and cloud presets via `--cloud --cloud-vcpus 8|16|32`.
- `python -m scripts.run_td_loop_selfplay --cloud --cloud-vcpus 16 --run-label td-loop-selfplay-r1 --chunks-per-loop 12 --collect-games 600 --train-steps 10000 --eval-games-per-side 200 --incumbent-eval-games-per-side 200 --progress-heartbeat-minutes 30 --eval-progress-log-minutes 30`
  Primary post-bootstrap self-play loop. Uses mixed td-search-heavy collection, promoted opponent-pool sampling, and dual promotion gates versus fixed `search` plus incumbent `td-search`.
- `python -m scripts.resume_td_loop_selfplay --run-id <interrupted-selfplay-run-id>`
  Resume an interrupted self-play loop from the latest fully completed chunk, rerun the next partial chunk from scratch, then finish the remaining chunks plus dual promotion evals.
- `python -m scripts.resume_td_loop_run`
  Legacy bootstrap recovery helper for the interrupted chunk-003 training and promotion path.
- `python -m scripts.benchmark_collect_search_profiles --workers 4 --games 8`
  Benchmark laptop-friendly td-search collect throughput across a small `search-worlds` and `search-depth` matrix.
- `python -m scripts.benchmark_selfplay_collect_setup`
  Compare single-process versus sharded self-play collection and recommend a `--collect-workers` setting for the current machine.

Use `--help` on each script for the full option surface.

## Constraints

- Static deployment target (no gameplay backend).
- Deterministic gameplay required for replay/eval/training.
- Rule semantics stay in TS unless explicitly re-approved.
- Python training scripts are fail-fast and expect the active project virtualenv.
- `scripts.train_td` fail-fast enforces Python 3.12+ and active `.venv` at startup.

## Known Gaps

- TD replay/train/eval orchestration exists, but automated online replay refresh is not wired yet.
- `td-search` exists, but current form is still rollout-guided and needs stronger search/value coupling and throughput optimization.
- Search baseline promotion thresholds still need repeated confirmation.
- Browser `td-value` and `td-search` deployment paths exist via static model-pack export/loading; remaining gap is browser runtime performance tuning for `td-search`.

_Updated: 2026-04-20._
