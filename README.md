# Magnate (Web) + RL Bot

Single-player Magnate with a competent computer opponent.

Core rule engine is TypeScript and runs in both browser and Node. Python training consumes the TypeScript engine through a small stable bridge interface.

Quick links:
- Agent manifest: `AGENTS.md`
- Memory Bank index: `memoryBank/`
- Memory workflows: `docs/AGENT_GUIDE.md`
- Bridge interface contract: `memoryBank/bridgeInterfaceContract.md`
- Official rules: http://wiki.decktet.com/game:magnate

Architecture decisions (current)
- TypeScript engine is the canonical rules implementation.
- Python trainer does not re-implement rules in v1; it calls the TS engine through a bridge.
- Shared contract scope is intentionally small: bridge payloads, action IDs, observation layout, model I/O names.
- Full cross-language rules schema is out of scope for now.

Current status (2026-02-19)
- Engine foundations exist under `src/engine/` (types, cards, deterministic deck, early action/reducer scaffolding).
- UI, bridge runtime, trainer runtime, and end-to-end tests are not complete yet.
- Documentation and Memory Bank are now aligned to the TS-canonical + small-contract direction.

Getting started (current scripts)
- Node: v20+
- Package manager: Yarn
- Install: `yarn`
- Dev server: `yarn dev`
- Tests: `yarn test`
- Lint/format: `yarn lint` / `yarn format`

Repo conventions
- Engine is deterministic and pure (seeded RNG, no side effects).
- UI is a thin React layer over the engine.
- Bridge exposes stable commands (`reset`, `step`, `legalActions`, `observation`, `serialize`, metadata handshake) for training.
- Keep the Memory Bank current after meaningful changes.
- Keep rule semantics centralized in TypeScript; do not fork game rules into Python by default.
- No code comments: express intent via clear names, types, and tests; capture rationale in the Memory Bank.

Memory Bank
- Location: `memoryBank/`
  - `projectBrief.md`: scope, goals, architecture
  - `activeContext.md`: current focus, decisions, next steps
  - `systemPatterns.md`: architecture and patterns
  - `techContext.md`: tools, versions, scripts
  - `progress.md`: status, what works, what’s next
  - `magnateRules.md`: official rules reference for the engine
  - `bridgeInterfaceContract.md`: stable TS<->Python boundary contract

Using Codex CLI
- Start each session in `AGENTS.md`, then review the Memory Bank before planning.
- Ask to “update memory bank” to refresh context; see `docs/AGENT_GUIDE.md`.
- Prefer making small, focused changes and keeping docs aligned with code.

Next major milestone
- Finish "reset truth sources" follow-through by implementing bridge-contract scaffolding, completing TS engine rules flow, and adding targeted tests before RL training work begins.
