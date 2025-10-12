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
  - random legal policy is available now
  - trained profile entries are scaffolded with explicit random fallback until model/runtime wiring lands
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
  - REINFORCE self-play fine-tuning from BC checkpoints (`scripts/finetune.py`)
- Broader experiment scheduling/metrics automation is not implemented yet.

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

- Python smoke: `python scripts/smoke_trainer.py`
- Python eval: `python scripts/eval.py --games 20`
- Python sample collection + BC warm-start: `python scripts/train.py --games 20`
- Python BC from existing samples: `python scripts/train.py --samples-in artifacts/training_samples.jsonl`
- Python RL fine-tune from BC checkpoint:
  - `python scripts/finetune.py --checkpoint-in artifacts/bc_checkpoint.json --checkpoint-out artifacts/rl_checkpoint.json --episodes 200 --eval-games 100`

## Source-of-Truth Docs

- Agent manifest: `AGENTS.md`
- Memory Bank: `memoryBank/`
- Memory workflow: `docs/AGENT_GUIDE.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
- Training encoding spec: `docs/TRAINING_ENCODING.md`
- Rules reference: `memoryBank/magnateRules.md`
