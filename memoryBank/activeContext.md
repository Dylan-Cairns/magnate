# Active Context

## Current Focus

- Keep legality and turn progression deterministic while wiring playable clients.
- Consolidate UI/controller patterns so random bot and future trained bot share one policy boundary.
- Preserve a stable TS-Python bridge boundary while client work continues.

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
  - District marker order now enforces rules intent:
    - `Excuse` is always the center district marker.
    - The four Pawn markers are shuffled per seed around the center.
- Player-scoped visibility projection now exists:
  - `toPlayerView(state, viewerId)` and `toActivePlayerView(state)` provide observation-safe views.
  - Opponent hand cards are hidden (count only), and draw pile order is hidden (`drawCount` only).
  - Public info remains visible (districts, deeds, resources, crowns, discard pile, phase/turn flags).
- Ace completion interpretation is explicitly locked in tests:
  - Ace deed completion target is 3 total progress tokens.
- React gameplay shell is now in place:
  - Vite/React entry files were added (`index.html`, `src/main.tsx`, `src/App.tsx`, `src/styles.css`).
  - Board UI shows districts, developed cards, deeds + deed tokens, crowns, resources, hands, dice roll, and action log.
  - Human acts via `legalActions` panel; bot currently chooses random legal actions with a short delay.
  - UI consumes engine truth (`newGame`, `advanceToDecision`, `legalActions`, `applyAction`, `toPlayerView`).

## Immediate Next Steps

1. Extract a formal policy interface (`selectAction(view, legalActions)`) and move random bot behind it.
2. Improve action UX from raw action buttons to grouped/intention-level controls where useful.
3. Start bridge runtime scaffolding against `bridgeInterfaceContract.md`.

_Updated: 2026-02-20._
