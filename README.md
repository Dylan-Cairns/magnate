# Magnate (Web) + TD/Keldon Training Pivot

Single-player Magnate with a deterministic TypeScript engine, browser UI, and Python training stack.

## Project Direction

- TypeScript engine is the canonical rules implementation.
- Python training/eval calls TS through a Node bridge.
- Current mission is singular:
  - build a TD-Gammon / Keldon-like training pipeline,
  - keep rollout search as warm-start signal only,
  - deploy a stronger learned bot in this web app.

## Training Ethos

- Fail fast over silent fallback in Python training/eval code.
- Invalid bridge payloads, missing checkpoints, or malformed policy probabilities are treated as hard errors.
- Training scripts require explicit policy selection and an active `.venv`.

## Current Status

- Browser game is playable; default web bot is rollout-eval search.
- Bridge runtime is implemented (`metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`).
- Python policy surface is `random`, `heuristic`, `search`, `td-value`, and `td-search`.
- TD Phase 1 foundation is implemented in `trainer/td` (models, replay, targets, checkpointing, self-play utilities, value trainer).
- TD Phase 2 orchestration is implemented:
  - `scripts.collect_td_self_play` to generate replay artifacts,
  - `scripts.train_td` to train value/opponent models from replay,
  - `td-value` policy support in `scripts.eval` and `scripts.eval_suite` for checkpoint benchmarking.
- TD Phase 3 initial integration is implemented:
  - `td-search` policy combines determinized search with TD value leaf evaluation and required opponent-model rollout guidance.
- PPO, MCTS, and guidance/distillation codepaths were intentionally removed.

## Local Commands

- Install JS deps: `yarn`
- Dev server: `yarn dev`
- Bridge runtime: `yarn bridge`
- Test: `yarn test`
- Lint + typecheck: `yarn lint`
- Format: `yarn format`

## Python Setup

From repo root:

1. `./scripts/setup_python_env.ps1`
2. `./.venv/Scripts/Activate.ps1`

macOS/Linux equivalent:

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`

## RunPod Install (CPU)

Use this for Linux CPU pods with a persistent `/workspace` volume.

One-time setup on the persistent volume:

```bash
cd /workspace
git clone <your-repo-url> magnate
cd /workspace/magnate

apt-get update
apt-get install -y curl ca-certificates gnupg
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs python3.11 python3.11-venv
npm install -g yarn

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
yarn install
```

Run smoke first, then full loop:

```bash
python -m scripts.run_td_loop --run-label td-loop-smoke --collect-games 12 --collect-search-worlds 2 --collect-search-depth 8 --collect-search-max-root-actions 4 --train-steps 30 --train-save-every-steps 15 --train-hidden-dim 64 --train-value-batch-size 32 --train-opponent-batch-size 16 --eval-games-per-side 20 --eval-opponent-policy search --eval-search-worlds 2 --eval-search-depth 8 --eval-search-max-root-actions 4
python -m scripts.run_td_loop --cloud --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search
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
- Canonical side-swapped eval: `python -m scripts.eval_suite --games-per-side 200 --workers 2 --candidate-policy search --opponent-policy heuristic`
- Search sweep: `python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse`
- Teacher data generation (warm-start labels): `python -m scripts.generate_teacher_data --games 200 --teacher-policy search --teacher-players both --out artifacts/teacher_data/teacher_search.jsonl`
- TD self-play replay generation: `python -m scripts.collect_td_self_play --games 200 --player-a-policy search --player-b-policy search --out-dir artifacts/td_replay --run-label td-replay-search`
- TD training run: `python -m scripts.train_td --value-replay artifacts/td_replay/<run>.value.jsonl --opponent-replay artifacts/td_replay/<run>.opponent.jsonl --steps 2000 --run-label td-v1`
- TD checkpoint eval: `python -m scripts.eval_suite --games-per-side 200 --candidate-policy td-value --opponent-policy heuristic --td-value-checkpoint artifacts/td_checkpoints/<run>/value-step-0002000.pt --td-worlds 8`
- TD-search checkpoint eval: `python -m scripts.eval_suite --games-per-side 200 --candidate-policy td-search --opponent-policy heuristic --td-search-value-checkpoint artifacts/td_checkpoints/<run>/value-step-0002000.pt --td-search-opponent-checkpoint artifacts/td_checkpoints/<run>/opponent-step-0002000.pt`
- Full loop automation (local defaults): `python -m scripts.run_td_loop --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search`
- Full loop automation (cloud 8 vCPU profile): `python -m scripts.run_td_loop --cloud --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search`
- Full loop with built-in begin/end improvement readout: `python -m scripts.run_td_loop --cloud --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --train-save-every-steps 200 --eval-games-per-side 200 --eval-opponent-policy search --eval-first-last-checkpoints`

Use `--help` on each script for full options.

## Source-of-Truth Docs

- Agent manifest: `AGENTS.md`
- Memory workflow: `docs/AGENT_GUIDE.md`
- Training handoff/restart: `docs/TRAINING_HANDOFF.md`
- Training plan: `docs/TRAINING_PLAN_SEARCH_FIRST.md`
- Command cookbook: `docs/TRAINING_COMMANDS.md`
- Encoding contract: `docs/TRAINING_ENCODING.md`
- Memory Bank: `memoryBank/`
- Rules reference: `memoryBank/magnateRules.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
