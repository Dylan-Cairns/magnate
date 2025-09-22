# Active Context

## Current Focus

- Complete the TS engine turn loop from decision phase through endgame.
- Keep legality and turn progression deterministic.
- Preserve a stable TS-Python bridge boundary while engine work continues.

## Locked Decisions

- TS is canonical for game rules.
- Python uses the TS engine through a Node bridge.
- Shared contract remains narrow and versioned.
- No full Python rules duplication by default.

## Recent Direction

- Low-level action/reducer validation is in place.
- Deck/setup/draw logic has been modularized.
- `advanceToDecision` exists and currently resolves start-turn chain, tax/income baseline, draw, and end-turn handoff.

## Immediate Next Steps

1. Finish optional-phase progression wiring in the turn loop.
2. Implement scoring and terminal resolution details.
3. Add/expand tests for full-game turn flow and scoring outcomes.
4. Start bridge runtime scaffolding against `bridgeInterfaceContract.md`.

_Updated: 2026-02-19._
