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

## Client Controller Pattern

- Client loop should stay thin and deterministic:
  - `state = advanceToDecision(newGame(...))`
  - render from `toPlayerView(state, viewerId)`
  - apply only `legalActions(state)` through `applyAction`
- Bot/human action selection should sit behind a shared policy contract so swapping random -> trained does not change controller flow.
- UI score presentation should be derived, not stateful:
  - compute live score from canonical engine state (`scoreGame(state)`) on render
  - reuse same score component for terminal and non-terminal states

## Turn-Flow Pattern

- Non-decision phases auto-resolve via `advanceToDecision`.
- Decision phases are where external actors choose actions (`CollectIncome` with pending choices, `OptionalTrade`, `OptionalDevelop`, `PlayCard`).
- Draw/exhaustion handling and final-turn countdown are part of phase resolution.
- Card-play gating is explicit (`cardPlayedThisTurn`):
  - exactly one card-play action per turn
  - optional phases share one action surface:
    - pre-card: `trade`, `develop-deed`, and card-play actions
    - post-card: `trade`, `develop-deed`, and `end-turn`
  - no explicit optional-phase navigation actions are exposed
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
