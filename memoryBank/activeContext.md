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
  - `choose-income-suit` actions resolve those choices one-by-one, with active actor switching to the choice owner and restoration to the turn owner before `ActionWindow`.
- Decision-phase model now uses a single `ActionWindow` phase:
  - legacy `OptionalTrade` / `OptionalDevelop` / `PlayCard` phase literals were removed.
  - Before card play: legal actions are `trade`, `develop-deed`, and all card-play actions.
  - After card play: legal actions are `trade`, `develop-deed`, and `end-turn`.
  - One-card-per-turn lock is enforced via `cardPlayedThisTurn`; card-play actions are unavailable once a card is played.
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
- First-placement district marker rule is now explicitly regression-tested at action/reducer level:
  - `buy-deed` and `develop-outright` are blocked for first placements without Pawn suit overlap (Excuse remains the only first-placement exception).
- React gameplay shell is now in place:
  - Vite/React entry files were added (`index.html`, `src/main.tsx`, `src/App.tsx`, `src/styles.css`).
  - Board UI shows districts, developed cards, deeds + deed tokens, crowns, resources, hands, dice roll, and action log.
  - Human acts via `legalActions` panel; bot currently chooses random legal actions with a short delay.
  - UI consumes engine truth (`newGame`, `advanceToDecision`, `legalActions`, `applyAction`, `toPlayerView`).
  - HUD now shows a derived `Status` label instead of raw engine phase names.
  - Incremental score is now displayed throughout play:
    - UI derives score each state update via `scoreGame(state)`.
    - Single score panel is used; on terminal states it shows the final score (no duplicate live/final panels).
  - Layout is now reorganized for gameplay ergonomics:
    - dedicated left actions column with scrollable action list
    - dedicated right info column ordered as title, seed/new game controls, score, roll result, and log
    - player summary rows now place resources, crowns, and hand horizontally in a single line on desktop
  - Player-row stability polish is now in place:
    - each player section now uses left-aligned title-above-content grouping
    - resources render fixed 2x3 suit slots (with zero-count placeholders) to avoid width/height jitter
    - crowns/hand render fixed slot footprints so card-count changes do not shift layout
    - gameplay shell is viewport-anchored with internal panel scrolling so action-count changes do not resize page height
  - Trade action UX is now collapsed:
    - action list shows one `trade` action per give-suit (`give x3`) instead of one per give/receive pair
    - selecting a trade action opens an anchored popover to the right of the clicked action to choose receive suit
    - this reduces action-list length while preserving engine action semantics
  - District headers now show the marker card name centered in the header row:
    - Pawn/Excuse markers use a unified header style
    - suit icons remain right-aligned for Pawn markers only
  - Player-panel visual consistency pass:
    - crowns/hand now use full-size card tiles matching board card dimensions
    - player sections are horizontally centered in each player panel with slightly increased spacing
    - resource chips are enlarged so the 2x3 resource block height aligns with adjacent card height

## Immediate Next Steps

1. Extract a formal policy interface (`selectAction(view, legalActions)`) and move random bot behind it.
2. Improve action UX from raw action buttons to grouped/intention-level controls where useful.
3. Start bridge runtime scaffolding against `bridgeInterfaceContract.md`.

_Updated: 2026-02-20._
