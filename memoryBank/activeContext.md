# Active Context

## Current Focus

- Close remaining rules-parity gaps with scenario-driven integration tests.
- Keep the UI/controller thin and policy-agnostic so random and trained bots share one action-selection boundary.
- Preserve a stable TS-Python contract while bridge runtime and trainer work begin.

## Locked Decisions

- TypeScript engine is the canonical rules implementation.
- Python training calls TS through a Node bridge.
- Shared boundary is a small, versioned interface contract.
- Native Python rules are out of scope unless throughput becomes a proven bottleneck.

## Current State

### Engine

- Deterministic setup/deck lifecycle, legality generation, and reducer transitions are implemented.
- Turn resolution is phase-driven via `advanceToDecision`, with a unified decision phase (`ActionWindow`).
- Income suit choice is explicit (`choose-income-suit`) and actor ownership is handled correctly.
- Scoring and terminal resolution are implemented (`scoreGame`, terminal finalization, `isTerminal`).
- Canonical game init exists via `newGame(seed, { firstPlayer })`.
- Turn-state bookkeeping was simplified:
  - deck reshuffle state is canonical for exhaustion/final-turn flow.
  - income-choice return owner is stored as `PlayerId` in state.
  - transitional `IncomeRoll` phase was removed.

### Observability

- `toPlayerView` and `toActivePlayerView` are implemented and test-covered.
- Opponent hand contents and draw order are hidden; public board/deed/resource state remains visible.

### UI

- Playable React shell exists (`src/App.tsx`) with human vs random bot.
- UI dispatches only engine-legal actions and uses grouped follow-up pickers where options are noisy.
- Controller/policy boundaries are extracted:
  - `createSession` / `stepToDecision` in `src/engine/session.ts`
  - `ActionPolicy` + `randomPolicy` in `src/policies/`
  - action grouping/picker presentation helpers in `src/ui/actionPresentation.ts`
- Right info column includes controls, score, deck state, roll result, and full scrollable log.
- Final-turn warning is surfaced in the actions header during the final-turn window.
- District lanes render centered overlapping card stacks; bot-visible cards (district + crowns) use bot perspective (rank/suits at bottom, progress at top).

## Immediate Next Steps

1. Expand full-turn/full-game scenario coverage for rules parity.
2. Implement bridge runtime against `memoryBank/bridgeInterfaceContract.md`.
3. Add trainer scaffold and baseline loop.
4. Wire trained-policy selection through existing `ActionPolicy` boundary.

_Updated: 2026-02-20._
