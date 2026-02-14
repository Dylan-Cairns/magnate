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
- React gameplay shell exists (`index.html`, `src/main.tsx`, `src/App.tsx`) for human vs random bot using engine APIs.
- Controller boundaries are extracted for bot swapping:
  - `createSession` / `stepToDecision` (`src/engine/session.ts`)
  - `ActionPolicy` + `randomPolicy` (`src/policies/`)
  - action grouping/picker helpers (`src/ui/actionPresentation.ts`)
- Bridge runtime and training pipeline are not implemented yet.

## Local Commands

- Install: `yarn`
- Dev: `yarn dev`
- Test: `yarn test`
- Lint: `yarn lint`
- Format: `yarn format`

## Source-of-Truth Docs

- Agent manifest: `AGENTS.md`
- Memory Bank: `memoryBank/`
- Memory workflow: `docs/AGENT_GUIDE.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
- Rules reference: `memoryBank/magnateRules.md`
