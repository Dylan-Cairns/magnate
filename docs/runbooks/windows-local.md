# Windows Local Runbook

## Local Setup

From repo root:

1. Install `fnm` and configure its PowerShell integration. Use recursive
   version-file lookup if you work from directories below the repo root.
2. Install and select the Node version pinned in `.nvmrc`:
   `fnm install`
   `fnm use`
3. Enable Corepack and install the Yarn version pinned in `package.json`:
   `corepack enable`
   `corepack install`
4. Verify the resolved toolchain:
   `node --version`
   `yarn --version`
5. Install JS deps:
   `yarn install`
6. Install Python deps:
   `.\scripts\setup_python_env.ps1`
7. Activate the venv when you want an interactive Python shell:
   `.\.venv\Scripts\Activate.ps1`

`setup_python_env.ps1` installs `requirements-dev.txt`, CPU-only PyTorch, Ruff, and Pyright. It routes temp and cache files into repo-local `.tmp/`, `.pip-cache/`, `.npm-cache/`, and `.yarn-cache/` folders.

The Windows training wrappers validate the active Node runtime and can resolve
the `.nvmrc` pin through `fnm` when launched from a `-NoProfile` shell. They do
not depend on legacy version-manager installation paths.

## Laptop Training Wrappers

Use the PowerShell wrappers so laptop-safe worker and thread settings stay separate from Linux and RunPod flows.

- Bootstrap or recalibration loop: `.\scripts\run_td_loop_bootstrap_laptop.ps1`
- Self-play loop: `.\scripts\run_td_loop_selfplay_laptop.ps1`
- Self-play resume: `.\scripts\resume_td_loop_selfplay_laptop.ps1 -RunId <interrupted-selfplay-run-id>`

Wrapper behavior:

- requires Node `22.23.1+` within the Node 22 line, `yarn install`, and a populated `.venv`
- sets repo-local temp and cache dirs plus BLAS and OpenMP thread caps
- auto-sizes CPU budget from logical core count
- defaults to `-CpuTargetPercent 60 -ReserveLogicalCores 2`
- maps CPU budget into collect, eval, incumbent eval, and train worker/thread settings
- keeps search-cost tuning explicit through loop args such as `--collect-search-worlds` and `--collect-search-depth`
- self-play laptop runs default to `--train-replay-window-source recent` and `--generator-update-chunks 3`
- streams child collect, train, and eval output into parent logs under `artifacts/logs/`
- resolves warm-start checkpoints from `models/td_checkpoints/manifest.json` before local artifact fallbacks

Useful invocations:

- Inspect the resolved self-play command without running it:
  `.\scripts\run_td_loop_selfplay_laptop.ps1 -DryRun`
- Increase CPU budget while keeping headroom:
  `.\scripts\run_td_loop_selfplay_laptop.ps1 -CpuTargetPercent 70 -DryRun`
- Override loop args:
  `.\scripts\run_td_loop_selfplay_laptop.ps1 -LoopArgs @('--run-label', 'td-loop-selfplay-laptop-test', '--collect-games', '300')`
- Run or resume the frozen extra-data step-9,000 checkpoint's 120-game
  heuristic-v2-medium comparison:
  `.\scripts\run_td_hard_extra_data_heuristic_benchmark.ps1`
