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
  - `choose-income-suit` actions resolve those choices one-by-one, with active actor switching to the choice owner and restoration to the turn owner before `OptionalTrade`.
- Optional-phase progression is explicit via actions:
  - `end-optional-trade` moves `OptionalTrade -> OptionalDevelop`
  - `end-optional-develop` moves `OptionalDevelop -> PlayCard` before a card play, and `OptionalDevelop -> OptionalTrade` after a card play.
  - `end-turn` is available only after card play and advances to `DrawCard`.
  - One-card-per-turn lock is enforced via `cardPlayedThisTurn`.
- Scoring and terminal resolution are now implemented:
  - `scoreGame(state)` computes district points and tie-breakers.
  - Game over finalization discards hands + incomplete deeds, then stores `finalScore`.
  - `isTerminal(state)` helper is available.
- Canonical game initialization now exists:
  - `newGame(seed, { firstPlayer })` builds full `GameState` from deterministic setup output.
  - District marker suits now come directly from dealt Pawn/Excuse marker cards instead of fixture assumptions.
- Player-scoped visibility projection now exists:
  - `toPlayerView(state, viewerId)` and `toActivePlayerView(state)` provide observation-safe views.
  - Opponent hand cards are hidden (count only), and draw pile order is hidden (`drawCount` only).
  - Public info remains visible (districts, deeds, resources, crowns, discard pile, phase/turn flags).
- Ace completion interpretation is explicitly locked in tests:
  - Ace deed completion target is 3 total progress tokens.

## Immediate Next Steps

1. Build CLI loop on top of `toActivePlayerView` + policy interface (human/random).
2. Start bridge runtime scaffolding against `bridgeInterfaceContract.md`.
3. Add API-level integration tests that drive full turns from `newGame`.

_Updated: 2026-02-20._
