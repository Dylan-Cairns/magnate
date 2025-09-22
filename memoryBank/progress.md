# Progress

## Implemented

- TS-canonical architecture and small bridge-contract strategy are documented and locked.
- Engine foundations are in place (`types`, `cards`, setup/deck/draw modules, legality/reducer paths).
- Deterministic phase resolver scaffold exists (`advanceToDecision`) with tax/income and draw/end-turn baseline behavior.
- Unit tests cover key low-level rules and early turn-flow behavior.

## In Progress

- Completing full turn-loop wiring across optional phases.
- Tightening endgame and scoring behavior to full rules parity.

## Remaining

- Full scoring and terminal resolution.
- Bridge runtime implementation and contract tests.
- Trainer scaffolding and baseline bot loop.
- Browser gameplay shell and model inference integration.
- Deploy-ready web app entry/deployment workflow.

## Risks / Watch Items

- Rule-edge handling in turn flow and scoring still needs broader scenario coverage.
- Bridge contract stability should be guarded as soon as runtime work starts.

_Updated: 2026-02-19._
