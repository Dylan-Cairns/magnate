# Magnate (Web) + RL Bot

Single-player Magnate with a trained bot opponent.

## Project Direction

- TypeScript engine is the canonical rules implementation.
- Python training calls the TS engine through a Node bridge.
- Shared contract scope stays small: bridge payloads, action IDs, observation layout, model I/O names.
- Do not duplicate full Magnate rules in Python unless bridge throughput is a proven bottleneck.

## Current Status

- Engine core exists in `src/engine/` and is deterministic:
  - setup/deck lifecycle
  - legality + reducer transitions
  - phase resolution (`advanceToDecision`)
  - scoring + terminal resolution
- Canonical initialization is available via `newGame(seed, { firstPlayer })`.
- Player-view projection exists (`toPlayerView` / `toActivePlayerView`) with hidden opponent hand contents and hidden draw order.
- React gameplay shell exists (`index.html`, `src/main.tsx`, `src/App.tsx`) for human vs bot play using engine APIs.
- UI exposes a bot profile selector:
  - champion PPO profile is available in browser and set as default
  - random legal profile remains available for baseline play
  - champion browser weights are served from `public/models/ppo_champion_2026-02-23_seed7.browser.json`
- Controller boundaries are extracted for bot swapping:
  - `createSession` / `stepToDecision` (`src/engine/session.ts`)
  - async-capable `ActionPolicy` + profile catalog (`src/policies/`)
  - action grouping/picker helpers (`src/ui/actionPresentation.ts`)
- Bridge runtime is implemented (`src/bridge/runtime.ts`, `src/bridge/cli.ts`) with NDJSON commands:
  - `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`
- Versioned bridge contract artifact exists at `contracts/magnate_bridge.v1.json`.
- Python trainer scaffold is now available (`trainer/`, `scripts/train.py`, `scripts/eval.py`):
  - bridge client + env wrapper
  - fixed-size observation/action encoders
  - baseline random/heuristic policies and matchup evaluation harness
  - behavior-cloning warm-start optimizer + checkpoint save/load
  - behavior-cloned policy support in eval (`--player-*-policy bc --player-*-checkpoint <path>`)
  - stabilized RL fine-tuning from BC checkpoints (`scripts/finetune.py`):
    - mixed opponents (self/heuristic/random)
    - BC-anchor regularization
    - fixed-holdout eval-based best-checkpoint selection
  - PyTorch PPO scaffold (`scripts/train_ppo.py`) with candidate-action actor-critic model
  - canonical fixed-holdout benchmark CLI (`scripts/benchmark.py`) for consistent checkpoint comparison
  - browser PPO export utility (`scripts/export_ppo_browser_checkpoint.py`)
- Competitive RL tuning vs human-level play is still in progress.

## Local Commands

- Install: `yarn`
- Dev: `yarn dev`
- Bridge: `yarn bridge`
- Test: `yarn test`
- Lint: `yarn lint`
- Format: `yarn format`

## Python Setup (Required)

From repo root (`c:\Users\dcairns\Documents\src\magnate`):

1. Create/update venv (Windows PowerShell): `.\scripts\setup_python_env.ps1`
2. Activate venv (Windows PowerShell): `.\.venv\Scripts\Activate.ps1`

macOS/Linux equivalent:

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`

With `.venv` active:

- Python smoke: `python -m scripts.smoke_trainer`
- Python eval: `python -m scripts.eval --games 20`
- Canonical benchmark (BC/PPO/random/heuristic candidate as PlayerA):
  - `python -m scripts.benchmark --candidate-policy bc --candidate-checkpoint artifacts/bc_checkpoint.json`
- Python sample collection + BC warm-start: `python -m scripts.train --games 20`
- Python BC from existing samples: `python -m scripts.train --samples-in artifacts/training_samples.jsonl`
- Python RL fine-tune from BC checkpoint:
  - `python -m scripts.finetune --checkpoint-in artifacts/bc_checkpoint.json --checkpoint-out artifacts/rl_checkpoint.json --episodes 300 --eval-games 100 --eval-every 50 --eval-mode fixed-holdout`
- Python PPO scaffold training:
  - `python -m scripts.train_ppo --checkpoint-out artifacts/ppo_checkpoint.pt --episodes 1024 --episodes-per-update 32 --eval-games 100 --eval-every-updates 5 --eval-mode fixed-holdout`
  - progress logging is enabled by default every 5 updates (`--progress-every-updates 0` disables it)
- Queue PPO training runs by seed (sequential):
  - `python -m scripts.train_ppo_queue --seeds 2 3 4 --episodes 1024 --episodes-per-update 32 --eval-games 100 --eval-every-updates 5 --eval-mode fixed-holdout --progress-every-updates 5`
- Queue canonical benchmarks by seed (sequential, with ranked summary):
  - `python -m scripts.benchmark_queue --seeds 1 2 3 4 --candidate-policy ppo`
- Export a trained PPO `.pt` checkpoint for browser inference:
  - `python -m scripts.export_ppo_browser_checkpoint --checkpoint-in models/ppo_champion_2026-02-23_seed7.pt --out public/models/ppo_champion_2026-02-23_seed7.browser.json`

## Source-of-Truth Docs

- Agent manifest: `AGENTS.md`
- Memory Bank: `memoryBank/`
- Memory workflow: `docs/AGENT_GUIDE.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
- Training encoding spec: `docs/TRAINING_ENCODING.md`
- Rules reference: `memoryBank/magnateRules.md`
