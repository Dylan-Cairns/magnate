# Magnate (Web) + RL Bot

Single-player Magnate with a trained bot opponent.

## Project Direction

- TypeScript engine is the canonical rules implementation.
- Python training calls the TS engine through a Node bridge.
- Shared contract scope stays small: bridge payloads, action IDs, observation layout, model I/O names.
- Do not duplicate full Magnate rules in Python unless bridge throughput is a proven bottleneck.

## Current Status

- Engine core exists (`src/engine/`) with deterministic setup, deck flow, low-level actions, and early turn-flow resolution.
- Tax/income baseline is implemented in the turn-flow resolver.
- UI, bridge runtime, training pipeline, and scoring/endgame completion are still in progress.

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
