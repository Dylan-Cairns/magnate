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
  - `state = createSession(seed, firstPlayer)`
  - render from `toPlayerView(state, viewerId)`
  - apply only `legalActions(state)` through `stepToDecision(state, action)`
- UI-only convenience controls may exist outside `legalActions` if they do not alter rules semantics or policy/bridge contracts:
  - example: human-only turn reset that restores a previously captured engine snapshot.
- Bot/human action selection should sit behind a shared policy contract so swapping random -> trained does not change controller flow.
- Policy contract should allow async selection so browser model inference can plug in without changing controller flow.
- Bot profile selection should resolve through a small catalog:
  - available profiles use their own policy implementation
  - unavailable profiles are disabled in UI; profile resolution throws if selected programmatically (no silent fallback)
  - current browser profiles prioritize deterministic search-based play over legacy model artifact loading
- Policy randomness should be injected by the controller (seed-derived where determinism matters), not hardcoded to `Math.random`.
- Additive policy implementations should not replace existing training/eval paths:
  - new policy kinds (for example `search`) are wired through the same `policy_from_name(...)` factory and existing eval harnesses.
  - policies that spawn external resources (for example bridge subprocesses for simulation) must expose `close()` so scripts can shut them down cleanly.
  - search-class policies may optionally load PPO-format guidance checkpoints for:
    - action priors
    - leaf-value evaluation
    - rollout opponent action modeling
  - guidance integration must remain additive (heuristic priors/value remain fallback when no checkpoint is provided).
- UI score presentation should be derived, not stateful:
  - compute live score from canonical engine state (`scoreGame(state)`) on render
  - reuse same score component for terminal and non-terminal states

## Turn-Flow Pattern

- Non-decision phases auto-resolve via `advanceToDecision`.
- Decision phases are where external actors choose actions (`CollectIncome` with pending choices and `ActionWindow`).
- Draw/exhaustion handling and final-turn countdown are part of phase resolution.
- Draw exhaustion source is canonical in `deck.reshuffles` (no duplicate exhaustion field).
- Income-choice return owner is stored as `PlayerId`.
- Card-play gating is explicit (`cardPlayedThisTurn`):
  - exactly one card-play action per turn
  - `ActionWindow` uses a unified action surface:
    - pre-card: `trade`, `develop-deed`, and card-play actions
    - post-card: `trade`, `develop-deed`, and `end-turn`
- Income suit-choice actor ownership is explicit:
  - active actor switches to pending choice owner during `CollectIncome`
  - turn owner is restored before normal turn decisions resume

## Bridge Pattern

- Bridge is the runtime boundary between TS engine and Python trainer.
- Contract is versioned and intentionally small.
- Stable items: request/response envelope, commands, action IDs, observation layout metadata, model I/O names.
- Runtime transport is NDJSON over stdin/stdout via `src/bridge/cli.ts`.
- Command handling lives in `src/bridge/runtime.ts` and returns strict success/error envelopes.
- Canonical bridge action surface comes from `src/engine/actionSurface.ts`:
  - stable action keys
  - canonical legal-action ordering by lexicographic action key

## Testing Pattern

- Unit tests for helpers, legality generation, reducer behavior, and turn flow.
- Unit tests for visibility boundaries (hidden opponent hand, hidden draw order).
- Deterministic fixtures and seed-based replay paths.
- Contract tests to protect TS/Python integration behavior.
- Python bridge-client/encoding/eval scaffolding should have its own tests (`trainer_tests/`) that run against the live bridge process.
- Search policies should have focused tests for:
  - deterministic action choice under fixed state + RNG
  - legal-action guarantees under live bridge legality responses
  - guidance-checkpoint integration paths for search and MCTS
- Side-swapped promotion evals should run through a single canonical pipeline:
  - paired seeds with swapped policy seats
  - Wilson confidence interval reporting
  - explicit side-gap reporting

## Versioning Pattern

- Include `schemaVersion` in serialized engine state.
- Include `contractVersion` in bridge metadata/responses where applicable.
- Breaking bridge changes require a major contract bump.
