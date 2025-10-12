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
- Canonical game creation exists via `newGame(seed, { firstPlayer })`.
- Player-scoped visibility projection exists and is test-covered (`toPlayerView` / `toActivePlayerView`).
- Playable browser client exists (human vs random bot) using engine truth APIs.
- Board UI now uses centered overlapping district stacks with player-specific visual perspective for readability.
- Tooling gates include lint + typecheck + tests.

## In Progress

- Expanding rules-parity scenario coverage, especially full-turn/full-game edges.
- Cleaning policy boundaries so random/human/trained bots share one selector interface.

## Remaining

- Bridge runtime implementation and contract tests.
- Trainer scaffold and baseline training loop.
- Model inference wiring in browser client.
- Deployment polish for static hosting path.

## Risks / Watch Items

- Remaining parity gaps are likely in edge-case sequencing rather than base mechanics.
- Bridge/API stability needs explicit guardrails once runtime integration starts.

_Updated: 2026-02-20._
