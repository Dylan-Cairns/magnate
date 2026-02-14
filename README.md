# Magnate (Web) + RL Bot

Single-player Magnate with a trained bot opponent.

## Project Direction

- TypeScript engine is the canonical rules implementation.
- Python training calls the TS engine through a Node bridge.
- Shared contract scope stays small: bridge payloads, action IDs, observation layout, model I/O names.
- Do not duplicate full Magnate rules in Python unless bridge throughput is a proven bottleneck.

## Current Status

- Engine core exists (`src/engine/`) with deterministic setup, deck flow, action legality/reducer logic, and turn-flow resolution.
- Tax/income, scoring, and terminal resolution are implemented.
- Canonical game initialization now exists via `newGame(seed, { firstPlayer })`.
- Player-scoped observation views are implemented (`toPlayerView` / `toActivePlayerView`) so opponent hand cards and draw order stay hidden.
- React gameplay shell is now wired (`index.html`, `src/main.tsx`, `src/App.tsx`) with:
  - board rendering for districts/stacks/deeds/resources/crowns/hand
  - legal-action button panel for the human player
  - random-bot opponent using engine `legalActions` + `applyAction`
  - turn/status/dice/log HUD
- Bridge runtime and training pipeline are still in progress.

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
