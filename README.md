# Magnate

Single-player Magnate with a deterministic TypeScript engine, browser UI, and Python training stack.

## At A Glance

- Browser game is playable; default bot is `TD Search Fast`.
- Browser includes selectable `TD Search Fast`, `TD Search (Browser)`, `Rollout Eval Search`, and `Random legal` profiles.
- TypeScript engine is the canonical rules implementation.
- Python training/eval calls the engine through the Node bridge.
- Loop progression: bootstrap/recalibrate with `scripts.run_td_loop`, then continue with self-play-focused `scripts.run_td_loop_selfplay`.

## Local Commands

- Install JS deps: `yarn install`
- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`

## GitHub Pages Deploy

- CI deploy workflow: `.github/workflows/deploy_pages.yml`
- Trigger: push to `main` (or manual `workflow_dispatch`)
- Deploy gates: `yarn test`, `yarn lint`, then `yarn build`
- Artifact: `dist/` uploaded to GitHub Pages
- One-time repo setting: in GitHub, set Pages source to `GitHub Actions`

## Windows Laptop Setup

From repo root:

1. Install/use Node `22.12.0` with `nvm`:
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

`setup_python_env.ps1` is the recommended Windows path for local training. It installs CPU-only PyTorch from the official CPU wheel index and routes temp/cache files into repo-local `.tmp/`, `.pip-cache/`, `.npm-cache/`, and `.yarn-cache/` folders so Windows installs do not explode the default temp directory.

## macOS/Linux Python Setup

Manual equivalent:

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt`

## Windows Laptop Training

Use the dedicated PowerShell wrappers so the laptop-safe worker/thread settings stay separate from the RunPod/Linux scripts.

- Bootstrap/recalibration loop: `.\scripts\run_td_loop_bootstrap_laptop.ps1`
- Self-play loop: `.\scripts\run_td_loop_selfplay_laptop.ps1`

Both wrappers:

- require Node `22.12.0+`, `yarn install`, and a populated `.venv`
- set repo-local temp/cache dirs plus BLAS/OpenMP thread caps
- log full output under `artifacts/logs/`
- pin the Dell laptop profile:
  - `collect-workers=2`
  - `eval-workers=2`
  - `incumbent-eval-workers=2` for self-play
  - `train-num-threads=4`
  - `train-num-interop-threads=1`

To inspect the resolved command without running it:

```powershell
.\scripts\run_td_loop_selfplay_laptop.ps1 -DryRun
```

To override or append loop arguments, pass them via `-LoopArgs`. Later args win, so you can intentionally override the wrapper defaults:

```powershell
.\scripts\run_td_loop_selfplay_laptop.ps1 -LoopArgs @('--run-label', 'td-loop-selfplay-laptop-test', '--collect-games', '300')
```

## RunPod Install (CPU)

Use this for Linux CPU pods with a persistent `/workspace` volume.

One-time setup on the persistent volume:

```bash
cd /workspace
git clone <your-repo-url> magnate
cd /workspace/magnate

apt-get update
apt-get install -y curl ca-certificates gnupg
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs python3.11 python3.11-venv
npm install -g yarn
npm install -g npm@11.11.0

yarn install

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Keep pip temp files on /workspace (many pods have only ~5 GB on /tmp).
export TMPDIR=/workspace/tmp
mkdir -p /workspace/tmp

# CPU-only torch install (avoids downloading NVIDIA/CUDA wheels).
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements.txt
```

Each new pod session (same persistent volume):

```bash
cd /workspace/magnate
git pull --ff-only
source .venv/bin/activate
export TMPDIR=/workspace/tmp
npm install -g yarn
yarn install
```

Run smoke first, then bootstrap, then self-play loop:

```bash
python -m scripts.run_td_loop --run-label td-loop-smoke --chunks-per-loop 1 --collect-games 12 --collect-search-worlds 2 --collect-search-depth 8 --collect-search-max-root-actions 4 --train-steps 30 --train-save-every-steps 15 --train-hidden-dim 64 --train-value-batch-size 32 --train-opponent-batch-size 16 --eval-games-per-side 10 --eval-opponent-policy search --eval-workers 1 --eval-search-worlds 2 --eval-search-depth 8 --eval-search-max-root-actions 4 --promotion-min-ci-low 0.5
python -m scripts.run_td_loop --cloud --cloud-vcpus 16 --run-label td-loop-r2-overnight --chunks-per-loop 3 --collect-games 1200 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search --promotion-min-ci-low 0.5 --progress-heartbeat-minutes 30 --eval-progress-log-minutes 30
# New self-play loop (requires promoted warm start or explicit warm-start checkpoints):
python -m scripts.run_td_loop_selfplay --cloud --cloud-vcpus 16 --run-label td-loop-selfplay-r1 --chunks-per-loop 18 --collect-games 600 --train-steps 10000 --eval-games-per-side 200 --incumbent-eval-games-per-side 200 --progress-heartbeat-minutes 30 --eval-progress-log-minutes 30
```

For long runs, use `tmux`:

```bash
tmux new -s train
# run training command
# detach: Ctrl+b then d
# reattach: tmux attach -t train
```

## Python Commands (Current)

With `.venv` active:

- Smoke: `python -m scripts.smoke_trainer`
- Canonical side-swapped eval: `python -m scripts.eval_suite --mode certify --games-per-side 200 --workers 2 --candidate-policy search --opponent-policy heuristic`
- Manual promotion-style eval: `python -m scripts.eval_suite --mode certify --games-per-side 200 --workers 2 --candidate-policy td-search --opponent-policy search --td-search-value-checkpoint artifacts/td_checkpoints/<run>/value-step-0002000.pt --td-search-opponent-checkpoint artifacts/td_checkpoints/<run>/opponent-step-0002000.pt`
- Search sweep: `python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse`
- Teacher data generation (warm-start labels; teacher policy must emit root action probabilities): `python -m scripts.generate_teacher_data --games 200 --teacher-policy search --teacher-players both --out artifacts/teacher_data/teacher_search.jsonl`
- TD self-play replay generation: `python -m scripts.collect_td_self_play --games 200 --player-a-policy search --player-b-policy search --out-dir artifacts/td_replay --run-label td-replay-search`
- TD training run: `python -m scripts.train_td --value-replay artifacts/td_replay/<run>.value.jsonl --opponent-replay artifacts/td_replay/<run>.opponent.jsonl --steps 2000 --run-label td-v1`
- TD checkpoint eval: `python -m scripts.eval_suite --mode certify --games-per-side 200 --candidate-policy td-value --opponent-policy heuristic --td-value-checkpoint artifacts/td_checkpoints/<run>/value-step-0002000.pt --td-worlds 8`
- TD-search checkpoint eval: `python -m scripts.eval_suite --mode certify --games-per-side 200 --candidate-policy td-search --opponent-policy heuristic --td-search-value-checkpoint artifacts/td_checkpoints/<run>/value-step-0002000.pt --td-search-opponent-checkpoint artifacts/td_checkpoints/<run>/opponent-step-0002000.pt`
- Bootstrap/recalibration loop (local defaults): `python -m scripts.run_td_loop --run-label td-loop-r1 --chunks-per-loop 3 --collect-games 1200 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search --promotion-min-ci-low 0.5`
- Bootstrap/recalibration loop (cloud profile): `python -m scripts.run_td_loop --cloud --cloud-vcpus 16 --run-label td-loop-r2-overnight --chunks-per-loop 3 --collect-games 1200 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search --promotion-min-ci-low 0.5 --progress-heartbeat-minutes 30 --eval-progress-log-minutes 30`
- Bootstrap/recalibration loop (Windows laptop wrapper): `.\scripts\run_td_loop_bootstrap_laptop.ps1`
- Self-play loop automation (cloud profile): `python -m scripts.run_td_loop_selfplay --cloud --cloud-vcpus 16 --run-label td-loop-selfplay-r1 --chunks-per-loop 18 --collect-games 600 --train-steps 10000 --eval-games-per-side 200 --incumbent-eval-games-per-side 200 --progress-heartbeat-minutes 30 --eval-progress-log-minutes 30`
- Self-play loop automation (Windows laptop wrapper): `.\scripts\run_td_loop_selfplay_laptop.ps1`

Use `--help` on each script for full options.

## Source-of-Truth Docs

- Memory Bank: `memoryBank/`
- Rules reference: `memoryBank/magnateRules.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
