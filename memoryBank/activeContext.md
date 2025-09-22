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
- Deed income suit choice is now explicit input:
  - `CollectIncome` becomes a decision phase when multi-suit deed income choices are pending.
  - `choose-income-suit` actions resolve those choices one-by-one, then flow resumes to `OptionalTrade`.
- Optional-phase progression is explicit via actions:
  - `end-optional-trade` moves `OptionalTrade -> OptionalDevelop`
  - `end-optional-develop` moves `OptionalDevelop -> PlayCard`
- Scoring and terminal resolution are now implemented:
  - `scoreGame(state)` computes district points and tie-breakers.
  - Game over finalization discards hands + incomplete deeds, then stores `finalScore`.
  - `isTerminal(state)` helper is available.

## Immediate Next Steps

1. Expand full-game turn-flow scenario coverage for remaining edge cases.
2. Start bridge runtime scaffolding against `bridgeInterfaceContract.md`.
3. Define state initialization entrypoint for complete playable game lifecycle.

_Updated: 2026-02-20._
