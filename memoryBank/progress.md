# Progress

## Implemented

- Architecture is locked:
  - TS engine is canonical.
  - Python integration goes through a small versioned bridge contract.
- Core engine flow is implemented and deterministic:
  - setup/deck lifecycle
  - legality + reducer action handling
  - phase resolver (`advanceToDecision`)
  - scoring and terminal finalization
- Engine state model was simplified:
  - canonical exhaustion source is `deck.reshuffles` (removed duplicated `exhaustionStage`)
  - income-choice return owner is ID-based (`incomeChoiceReturnPlayerId`)
  - removed transitional `IncomeRoll` phase
- Canonical game creation exists via `newGame(seed, { firstPlayer })`.
- Player-scoped visibility projection exists and is test-covered (`toPlayerView` / `toActivePlayerView`).
- Playable browser client exists (human vs random bot) using engine truth APIs.
- Board UI now uses centered overlapping district stacks with player-specific visual perspective for readability.
- Human-only `reset-turn` UX is implemented via a UI snapshot anchor at human turn start (`ActionWindow` pre-card), restoring state prior to `end-turn` without changing engine action contracts.
- Policy/controller boundaries now exist:
  - `src/engine/session.ts` (`createSession`, `stepToDecision`)
  - async-capable `src/policies/types.ts` + `src/policies/randomPolicy.ts`
  - `src/policies/catalog.ts` profile registry with trained-profile placeholders and explicit random fallback
  - `src/ui/actionPresentation.ts` (+ tests)
- UI now exposes bot-profile selection and status text while keeping random legal fallback available.
- Runtime hardening and audit fixes landed:
  - `newGame` now validates `firstPlayer` at runtime
  - income-choice log attribution now reflects the chooser even when active player is restored
- Tooling gates include lint + typecheck + tests.

## In Progress

- Expanding rules-parity scenario coverage, especially full-turn/full-game edges.
- Preparing bridge runtime/trainer integration against the extracted policy and session boundaries.

## Remaining

- Bridge runtime implementation and contract tests.
- Trainer scaffold and baseline training loop.
- Model inference wiring in browser client.
- Deployment polish for static hosting path.

## Risks / Watch Items

- Remaining parity gaps are likely in edge-case sequencing rather than base mechanics.
- Bridge/API stability needs explicit guardrails once runtime integration starts.
- Trained profiles are still placeholders until bridge/runtime inference wiring is complete.

_Updated: 2026-02-21._
