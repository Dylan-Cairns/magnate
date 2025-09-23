# Progress

## Implemented

- TS-canonical architecture and small bridge-contract strategy are documented and locked.
- Engine foundations are in place (`types`, `cards`, setup/deck/draw modules, legality/reducer paths).
- Tooling checks now gate TypeScript compilation (`lint` runs `eslint` + `tsc --noEmit`).
- Deterministic phase resolver exists (`advanceToDecision`) with:
  - tax/income baseline behavior
  - explicit deed income suit-choice flow (`choose-income-suit`) during `CollectIncome`
  - actor ownership on income choices (active actor switches to choice owner and restores to turn owner)
  - draw/end-turn baseline behavior
  - unified `ActionWindow` decision phase:
    - legacy `OptionalTrade` / `OptionalDevelop` / `PlayCard` literals removed
    - pre-card exposes `trade` + `develop-deed` + card-play actions
    - post-card exposes `trade` + `develop-deed` + `end-turn`
  - one-card lock keeps card-play actions unavailable after the first card play each turn
- Scoring and terminal resolution are implemented:
  - district scoring with ace district bonus
  - tie-breakers (`districts -> rank total -> resources -> draw`)
  - terminal finalization that discards hands and incomplete deeds
  - final score stored on terminal state
- Canonical state initialization entrypoint is implemented:
  - `newGame(seed, { firstPlayer })` builds playable state from setup output and real district marker suits.
  - Setup marker ordering now uses `Excuse` fixed at center with seeded Pawn shuffle around it.
- Ace deed completion interpretation is now explicit and test-covered:
  - ace deed development target is 3 total tokens.
- Player visibility boundary is implemented and test-covered:
  - `toPlayerView`/`toActivePlayerView` hide opponent hand cards and draw order.
  - Public info remains visible for both players (board, deeds, resources, crowns, discard, phase/turn state).
- React playable shell is implemented:
  - entrypoint + app scaffold exists (`index.html`, `src/main.tsx`, `src/App.tsx`, `src/styles.css`)
  - human player can play via engine-generated legal actions
  - random bot opponent executes legal actions through same engine loop
  - UI renders core board/resource/deed/dice/log state
  - HUD status now uses derived user-facing state instead of raw phase labels
  - UI now renders incremental score snapshot every update and reuses the same panel as final score at game end
  - UI layout now uses dedicated columns for actions and info:
    - actions list occupies a full vertical left column with internal scrolling
    - right column is ordered as title, seed/new game controls, score, roll result, and log
    - player resource/crown/hand sections are compacted onto one horizontal row on desktop
  - UI stability pass completed:
    - player section headers now sit above content and are left-aligned
    - resources reserve a fixed 2x3 suit footprint, including zero-count slots
    - crowns and hand reserve fixed card slots so counts changing does not shift panel geometry
    - shell uses viewport-height anchoring with internal scroll regions to prevent global page-height jumps from action-list size changes
  - Trade action list is now de-duplicated in UI:
    - one trade action per give suit in the actions column
    - anchored suit popover (positioned beside clicked trade action) selects receive suit before dispatching canonical `trade` action
  - District header display now uses marker card names centered in each district title row, with right-aligned suit icons for non-Excuse markers only
  - Player-area cards now match board card size/shape, with centered player-section layout and slightly increased inter-section spacing
  - Player resource chips are now scaled up so the resources block visually matches neighboring card height
- Unit tests cover key low-level rules and early turn-flow behavior.
- Additional regressions now lock first-placement district legality:
  - first placement must overlap district Pawn suits for both `buy-deed` and `develop-outright` legality/execution paths.

## In Progress

- Tightening remaining turn-flow edge handling to full rules parity and integration coverage.
- Hardening UI/controller boundaries for bot policy swapping.

## Remaining

- Remaining full-game scenario coverage and replay-focused integration tests.
- Policy interface extraction for random/human/trained bot implementations.
- UI action ergonomics and interaction polish.
- Bridge runtime implementation and contract tests.
- Trainer scaffolding and baseline bot loop.
- Browser model inference integration.
- Deploy-ready web app entry/deployment workflow.

## Risks / Watch Items

- Rule-edge handling in turn flow and scoring still needs broader scenario coverage.
- Bridge contract stability should be guarded as soon as runtime work starts.

_Updated: 2026-02-20._
