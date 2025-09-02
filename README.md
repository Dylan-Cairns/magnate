# Magnate (Web) + RL Bot

Single-player Magnate with a competent computer opponent. Core game engine in TypeScript; browser UI for play; simple Node bridge for Python training.

Quick links:
- Agent manifest: `AGENTS.md`
- Memory Bank index: `memoryBank/`
- Memory workflows: `docs/AGENT_GUIDE.md`
- Official rules: http://wiki.decktet.com/game:magnate

Getting started
- Node: v20+
- Package manager: Yarn
- Install: `yarn`
- Dev server: `yarn dev`
- Tests: `yarn test`
- Lint/format: `yarn lint` / `yarn format`

Repo conventions
- Engine is deterministic and pure (seeded RNG, no side effects).
- UI is a thin React layer over the engine.
- Bridge exposes `reset`, `step`, `legalActions`, `observation`, `serialize` for training.
- Keep the Memory Bank current after meaningful changes.
- No code comments: express intent via clear names, types, and tests; capture rationale in the Memory Bank.

Memory Bank
- Location: `memoryBank/`
  - `projectBrief.md`: scope, goals, architecture
  - `activeContext.md`: current focus, decisions, next steps
  - `systemPatterns.md`: architecture and patterns
  - `techContext.md`: tools, versions, scripts
  - `progress.md`: status, what works, what’s next
  - `magnateRules.md`: official rules reference for the engine

Using Codex CLI
- Start each session in `AGENTS.md`, then review the Memory Bank before planning.
- Ask to “update memory bank” to refresh context; see `docs/AGENT_GUIDE.md`.
- Prefer making small, focused changes and keeping docs aligned with code.

Status
- Memory Bank initialized; engine/UI/bridge/trainer stubs to follow.
