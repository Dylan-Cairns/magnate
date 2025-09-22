# System Patterns

## Core Principles

- TS engine is the only gameplay truth source.
- Determinism is required (seeded RNG, pure state transitions).
- UI and trainer consume legality/observations from TS, not re-derived rules.

## Engine Pattern

Primary APIs:

- `legalActions(state)`
- `applyAction(state, action)`
- `advanceToDecision(state)`
- `toPlayerView(state, viewerId)` / `toActivePlayerView(state)`

Design expectations:

- No side effects in rules logic.
- Immutable state updates.
- Phase-driven turn flow.

## Turn-Flow Pattern

- Non-decision phases auto-resolve via `advanceToDecision`.
- Decision phases are where external actors choose actions (`CollectIncome` with pending choices, `OptionalTrade`, `OptionalDevelop`, `PlayCard`).
- Draw/exhaustion handling and final-turn countdown are part of phase resolution.
- Card-play gating is explicit (`cardPlayedThisTurn`):
  - exactly one card-play action per turn
  - post-play optional loop uses `end-optional-trade`, `end-optional-develop`, and `end-turn`
- Income suit-choice actor ownership is explicit:
  - active actor switches to pending choice owner during `CollectIncome`
  - turn owner is restored before normal turn decisions resume

## Bridge Pattern

- Bridge is the runtime boundary between TS engine and Python trainer.
- Contract is versioned and intentionally small.
- Stable items: request/response envelope, commands, action IDs, observation layout metadata, model I/O names.

## Testing Pattern

- Unit tests for helpers, legality generation, reducer behavior, and turn flow.
- Unit tests for visibility boundaries (hidden opponent hand, hidden draw order).
- Deterministic fixtures and seed-based replay paths.
- Contract tests to protect TS/Python integration behavior.

## Versioning Pattern

- Include `schemaVersion` in serialized engine state.
- Include `contractVersion` in bridge metadata/responses where applicable.
- Breaking bridge changes require a major contract bump.
