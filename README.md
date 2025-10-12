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
- Full learning loop/model training is still a scaffold stage.

## Local Commands

- Install: `yarn`
- Dev: `yarn dev`
- Bridge: `yarn bridge`
- Test: `yarn test`
- Lint: `yarn lint`
- Format: `yarn format`
- Python smoke: `py -3.12 scripts/smoke_trainer.py`
- Python eval: `py -3.12 scripts/eval.py --games 20`
- Python sample collection: `py -3.12 scripts/train.py --games 20`

## Source-of-Truth Docs

- Agent manifest: `AGENTS.md`
- Memory Bank: `memoryBank/`
- Memory workflow: `docs/AGENT_GUIDE.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
- Training encoding spec: `docs/TRAINING_ENCODING.md`
- Rules reference: `memoryBank/magnateRules.md`
