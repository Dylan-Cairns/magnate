# Progress

## Implemented

- TS-canonical architecture and small bridge-contract strategy are documented and locked.
- Engine foundations are in place (`types`, `cards`, setup/deck/draw modules, legality/reducer paths).
- Tooling checks now gate TypeScript compilation (`lint` runs `eslint` + `tsc --noEmit`).
- Deterministic phase resolver exists (`advanceToDecision`) with:
  - tax/income baseline behavior
  - explicit deed income suit-choice flow (`choose-income-suit`) during `CollectIncome`
  - draw/end-turn baseline behavior
  - explicit optional-phase progression actions (`OptionalTrade -> OptionalDevelop -> PlayCard`)
- Scoring and terminal resolution are implemented:
  - district scoring with ace district bonus
  - tie-breakers (`districts -> rank total -> resources -> draw`)
  - terminal finalization that discards hands and incomplete deeds
  - final score stored on terminal state
- Unit tests cover key low-level rules and early turn-flow behavior.

## In Progress

- Tightening remaining turn-flow edge handling to full rules parity.

## Remaining

- Remaining turn-loop edge cases and broader full-game scenario coverage.
- Bridge runtime implementation and contract tests.
- Trainer scaffolding and baseline bot loop.
- Browser gameplay shell and model inference integration.
- Deploy-ready web app entry/deployment workflow.

## Risks / Watch Items

- Rule-edge handling in turn flow and scoring still needs broader scenario coverage.
- Bridge contract stability should be guarded as soon as runtime work starts.

_Updated: 2026-02-20._
