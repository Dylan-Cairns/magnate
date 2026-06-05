# Windows Local Runbook

## Local Setup

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

`setup_python_env.ps1` installs `requirements-dev.txt`, CPU-only PyTorch, Ruff, and Pyright. It routes temp and cache files into repo-local `.tmp/`, `.pip-cache/`, `.npm-cache/`, and `.yarn-cache/` folders.

## Laptop Training Wrappers

Use the PowerShell wrappers so laptop-safe worker and thread settings stay separate from Linux and RunPod flows.

- Bootstrap or recalibration loop: `.\scripts\run_td_loop_bootstrap_laptop.ps1`
- Self-play loop: `.\scripts\run_td_loop_selfplay_laptop.ps1`
- Self-play resume: `.\scripts\resume_td_loop_selfplay_laptop.ps1 -RunId <interrupted-selfplay-run-id>`

Wrapper behavior:

- requires Node `22.12.0+`, `yarn install`, and a populated `.venv`
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
