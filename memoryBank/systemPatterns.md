# System Patterns

## Core Principles

- The TypeScript engine is the only source of gameplay truth.
- Determinism is required: seeded RNG, pure state transitions, no hidden global state.
- UI and Python consume engine legality and observations; they never re-derive rules.

## Shared Finite Values

- Shared string vocabularies use exported `as const` objects and union types
  derived from their values. Consumers use named members in comparisons,
  constructors, discriminant types, and record keys; display labels remain
  independently authored text.
- Engine definitions live in dependency-free `src/engine/values.ts`, with
  existing type import paths re-exported through `src/engine/types.ts`.
  `SUITS`, `PLAYER_IDS`, `GAME_PHASES`, and `ACTION_IDS` preserve their explicit
  compatibility orders. Dice mappings and the ASCII-sorted card catalog keep
  their purpose-specific order.
- Policy values and worker protocols live in `src/policies/values.ts` and
  `workerValues.ts`; evaluation worker tags live in `src/botEval/workerValues.ts`.
  Bridge commands/errors live in `src/bridge/values.ts`. UI presentation tags,
  picker kinds, celebration outcomes, and stored winner outcomes belong to
  their UI/database value modules. Equal spellings in different domains do not
  imply a shared vocabulary.
- Value modules must not initialize controllers, workers, components, or the
  database. Literal expectations in compatibility tests and external JSON/Python
  contracts stay independent of these definitions. Renaming a serialized value
  still requires the applicable save/bridge/model compatibility decision.

## Engine Pattern

Primary APIs:

- `legalActions(state)` / `applyAction(state, action)` / `advanceToDecision(state)`
- `toPlayerView(state, viewerId)` / `toActivePlayerView(state)`
- `decisionPlayerIdForState` / `legalActionsForDecisionPlayer` / `toDecisionPlayerView`
  for the single policy- and bridge-facing decision actor.

Design expectations:

- No side effects in rules logic; immutable state updates; phase-driven turn flow.
- Card metadata lives in `src/engine/cards.ts`, generated from the local Jacynth
  Decktet spec. Card IDs and `ALL_CARDS` order are compatibility surfaces locked
  by tests; extended Courts are IDs `"41"`-`"44"` appended after the standard
  `"0"`-`"40"` catalog.
- Ruleset selection is explicit on state: `GameState.ruleset` is
  `'standard' | 'extended'`, and deck composition comes from
  `propertyDeckForRuleset(ruleset)`. Rollout clones, saved games, and the bridge
  carry the ruleset on state rather than re-deriving it.
- Courts are developable rank-10 property cards (`kind: 'Court'`) handled
  through the shared `DevelopableCard` helpers. Rank 10 keeps them out of rank
  income and out of the TD encoding, which stays standard-only.
- UI card-art filenames derive from card names in `src/ui/cardImages.ts` rather
  than a duplicate table.

## Client Controller Pattern

- The client loop stays thin and deterministic: create a session, render from
  the player view, and apply only legal actions through canonical dispatch.
- UI-only conveniences (for example a human-only turn reset that restores a
  captured snapshot) may exist outside `legalActions` if they do not alter rules
  semantics or policy/bridge contracts.
- Browser-only dev fixtures may be URL-gated under Vite dev mode; they still
  derive legality and decisions through canonical engine APIs.
- Bot and human action selection sit behind a shared async `ActionPolicy`
  contract so policy swaps do not change controller flow.
- Bot profiles resolve through `src/policies/catalog.ts`. Easy/Medium/Hard are
  deterministic `rollout-search-v2-*` profiles with heuristic v2; Experimental
  is `td-root-search-v2-medium` and is standard-ruleset only. Unknown or
  unavailable profiles throw; there is no silent fallback.
- Easy is the fresh-browser default; existing saves and opponent preferences
  retain their chosen profile. New Game keeps ruleset, opponent, and animation
  choices in a UI draft. Start Game validates the ruleset/profile pair, then
  replaces the session and autosave together; dismissing setup discards the draft.
- Policy randomness is injected by the controller (seed-derived where
  determinism matters), not hard-coded to `Math.random`.
- Browser model-backed policies load static model packs:
  `public/model-packs/index.json` selects the default pack, each pack provides
  `manifest.json` + `weights.json`, and the loader validates
  schema/checkpoint/encoding/dimension compatibility before use. URL resolution
  must work from both the main window and Web Workers under `base: './'`. The
  `weights.json` fetch goes through an origin-scoped Cache Storage entry keyed by
  pack id and creation time, so the bot worker and its search workers download a
  multi-megabyte pack once per browser rather than once per worker.
- Search algorithms reuse one deterministic root-search core: stable action
  keys, seeded world sampling, no-log simulation stepping, diagnostics, and
  optional worker-backed execution. Rollout-search and TD-root search share the
  core; TD-root search uses the TD model for root priors, rollout playout
  action choice, and non-terminal leaf values.
- Parallel TD-root search defaults to the paired lockstep worker executor;
  `?tdSearchExecutor=legacy` is the session-scoped rollback, and invalid values
  are hard errors. Executor selection must preserve rollout waves, UCB
  scheduling, visit budgets, RNG streams, ordered result merging, and
  selected-action semantics.
- Additive policy implementations wire through one factory
  (`createPolicyFromBotSpec` in TS, `policy_from_name` in Python) and must not
  replace existing paths. Policies that spawn external resources expose
  `close()`.
- Browser search resources follow one policy: workers are created lazily,
  pooled per bot worker, sized to `hardwareConcurrency - 2` with an 8-worker cap
  and budget clamps, and never more numerous than the scheduled work. Page
  visibility must never reduce bot strength: hiding does not defer, pause,
  cancel, or weaken a decision, and it does not tear down warm workers, so a
  bot turn keeps playing out while the user is on another tab (browser timer
  throttling may only slow it in wall-clock terms). The policy is closed when
  the game is terminal, and a warm worker is torn down after ten idle minutes.
  Teardown sends a `shutdown` request so the owning worker closes its nested
  search pool itself, with a `terminate()` fallback after a short grace period;
  this must not rely on the browser cascading termination to nested workers.
  Fixed per-profile visit budgets remain the anti-overheat lever, so lifecycle
  changes must not alter search semantics or determinism.

## Heuristic Scoring Pattern

- Browser heuristic v1 is one shared TypeScript scorer for direct heuristic play,
  rollout-search root ranking, and TD-search heuristic priors. It stays
  action-level and engine-state-derived: no duplicate legality checks, no
  speculative placement-chain or Ace-bonus preferences, and trades are penalties
  unless the post-trade resources immediately unlock a high-value development or
  deed move.
- Heuristic v2 stays additive and broad-delta based: district-local
  scoring-margin deltas, future suit-access earning deltas, and contextual
  token-bank deltas. Do not reward generic resource hoarding or add one-off
  tactical constants.
- Incomplete Courts are the one exception to the generic scoring path: because
  they have no income channel, `src/policies/courtPotentialV2.ts` values
  `buy-deed` and `develop-deed` on a Court as a feasibility-discounted district
  swing (`swing × available / (available + remainingCost)`), added inside the
  scoring weight and scaled by `SearchPolicyConfig.courtValueScale` (default 1,
  0 disables). Completed Courts and `develop-outright` stay generic, and Court
  deeds contribute zero to `potentialStackScore`, so nothing is double-counted.
  Standard play cannot reach the term because the standard deck has no Courts.
  TD-root search is unaffected because its guidance comes from the TD model.
- Court-valuation experiments predeclare gates, paired seeds, and the extension
  rule in `docs/design/court-valuation.md`; benchmark configs live in
  `configs/bot-eval/court-valuation/` and are analyzed with
  `yarn bot:eval court-value-report`.
- District-potential scoring includes newly bought deeds; opponent deed defense
  pressure scales with completion progress. Non-completing deed progress should
  not receive full new-control-path credit.
- Contextual token value lives in `src/policies/tokenValueV2.ts`: suit value
  tracks remaining earning/scoring demand adjusted by access and
  replaceability, with a concave per-suit marginal curve so surplus tokens are
  discounted rather than hoarded.

## Turn-Flow Pattern

- Non-decision phases auto-resolve via `advanceToDecision`.
- Decision phases are where external actors choose actions: `CollectIncome`
  with unsubmitted income choices and `ActionWindow`.
- Draw/exhaustion handling and the final-turn countdown are part of phase
  resolution; exhaustion is canonical in `deck.reshuffles`.
- Card-play gating is explicit (`cardPlayedThisTurn`): exactly one card-play
  action per turn. The `ActionWindow` surface is `trade` / `develop-deed` /
  card plays pre-card and `trade` / `develop-deed` / `end-turn` post-card.
- Partial deed income is submitted simultaneously: `pendingIncomeChoices` keeps
  the full obligation list, `submittedIncomeChoices` records suit choices
  without applying resources, and selected resources resolve in deterministic
  pending-choice order once every choice is submitted. The original turn owner
  remains the action-window owner afterward.

## Bridge Pattern

- NDJSON over stdin/stdout via `src/bridge/cli.ts`; command handling lives in
  `src/bridge/runtime.ts` and returns strict success/error envelopes.
- The contract is versioned and intentionally small: envelope, commands, action
  IDs, observation layout, and model I/O metadata.
- One policy actor per request: normal phases use the turn owner; simultaneous
  `CollectIncome` uses the first unsubmitted choice owner. `legalActions`,
  legal masks, `step` validation, and returned views align to that decision
  actor.
- The canonical action surface lives in `src/engine/actionSurface.ts`: stable
  action keys and lexicographic canonical ordering.
- Python bridge clients must continuously drain bridge stderr to avoid
  long-run pipe stalls on Windows.

## Training Pattern

- Python trains through the bridge. The policy surface is `random`,
  `heuristic`, `search`, `td-value`, and `td-search`.
- Loop orchestration: `scripts.run_td_loop` for bootstrap or recalibration and
  `scripts.run_td_loop_selfplay` for forward self-play. Both use chunked replay
  collection, checkpointed training, generator/incumbent gates, and
  promotion-gated evaluation through `scripts.eval_suite`
  (`--mode gate|certify`).
- Value training defaults to sequence-aware `td-lambda` targets with
  `lambda=0.7`; `td-lambda` requires complete contiguous per-player
  trajectories. Replay-window manifests reference ordered replay files instead
  of duplicating chunk data on disk.
- The checkpoint registry is `models/td_checkpoints/manifest.json` (schema v2:
  `defaultWarmStart`, `opponentPool`, `checkpoints.<key>.value/.opponent`).
  Successful promotions copy accepted pairs under `models/td_checkpoints/<key>/`
  and update the manifest unless `--disable-manifest-promotion` is set.
- `scripts.train_td` supports opt-in `--district-augmentation none|s4|s4-orbit`
  with an explicit experiment seed; S4 moves only D1/D2/D4/D5 and keeps D3
  fixed. Augmentation experiments bind replay, warm-start, manifest, and
  implementation fingerprints and require matched control runs.
- Training code is fail-fast: invalid bridge payloads, missing checkpoints,
  malformed distributions, missing TD signals, and missing explicit policy args
  are hard errors, never silent fallbacks.
- Platform-specific tuning (CPU/thread caps, temp and cache dirs, worker counts)
  lives in thin wrapper scripts, not canonical loop defaults. Long-running
  orchestration uses the shared `scripts.td_loop_common.run_step` runner for
  merged output, heartbeats, and fail-fast return codes.

## Testing Pattern

- Unit tests cover helpers, legality generation, reducer behavior, turn flow,
  visibility boundaries, and deterministic seeded replay.
- Contract tests protect the TS/Python boundary; `trainer_tests/` covers the
  bridge client, encoding, eval scaffolding, and search policies.
- TypeScript bot evaluation has focused tests for serializable specs, full-game
  deterministic transcripts, paired seat-swapped scheduling, artifacts, exact
  replay divergence reporting, and checkpointed head-to-head resume.
- Promotion evals use paired seeds with swapped seats, Wilson confidence
  intervals, and explicit side-gap reporting.

## Persistence And Versioning

- Browser autosave is one versioned `magnate:savedGame` localStorage entry
  (canonical state, session ID, bot profile, timeline, action history, deferred
  income context). Saves happen when a new human decision window opens plus
  initial and terminal states. Restore validates the snapshot by replaying its
  history through the canonical engine; incompatible saves stay untouched with
  autosave paused until New Game.
- UI preferences (opponent, deck-map visibility, log visibility, animations)
  persist separately. Game history uses IndexedDB with session-ID deduplication.
- Serialized engine state carries `schemaVersion`; bridge metadata and responses
  carry `contractVersion`. Additive fields do not require a version bump;
  breaking bridge changes require a major contract version bump.

## UI Presentation Pattern

- Canonical `state` drives legality, bot scheduling, bug reports, and
  persistence. React renders from a controller-provided `viewState`/player view
  so already-committed results cannot leak into visible UI ahead of animation.
- One `AnimationSequence` per canonical `GameTransaction`, built by
  `buildAnimationSequence`; sequence steps are the single timing source for
  render snapshots, overlays, durations, input unlock, and visual commands.
  Component-local timers must not own sequencing decisions.
- Accepted transitions enter one FIFO presentation backlog keyed by the
  canonical transaction ordinal. Only the head sequence schedules visuals, and
  rendering retains the last presented state between sequences.
- Animations must drive composited properties (`transform`/`opacity`), never
  layout properties. A layout property in `@keyframes` re-styles and re-lays-out
  every frame; the per-turn dice roll bounce animating `top` alone accounted for
  roughly 80% of the app's per-turn style recalculation. The bounce uses
  `translateY` for that reason, and `will-change` must name the animated
  property.
- Human input is gated by a transaction-specific decision-window barrier, not by
  pending presentation. Later actions in the same human window use canonical
  legality immediately and may run ahead of visuals.
- Human-input activity styling (actions-panel and human player-panel glow)
  derives from rendered human action items (canonical legality, readiness-gated),
  not from the presented phase: a bot-only income choice on the human's turn
  must not light the human input area while its action list is empty.
- Browser DOM lookup and flight construction stay outside the engine (for
  example `domTargets.ts`); sequence-derived visual commands carry the semantics
  needed to launch command-specific flights.
- Deed-token rail sides and order are shared presentation memory read by both
  component render and animation flight planning, so render must not mutate it.
  `planDeedTokenLayout` computes from a copy, `useDeedTokenLayout` records new
  suits in a post-render effect, and non-render callers use the mutating
  `commitDeedTokenLayout`. This keeps StrictMode double-renders and discarded
  renders from corrupting the layout.
- Flights that land inside a differently-sized card scope (hand to district lane)
  adopt the destination's resolved card metrics: `laneCardMetrics` measures the
  lane animation anchor's card size plus image-area width/height, the flight
  carries them as `endImageArea*`, and `CardFlightLayer` applies both as inline
  custom properties. Propagate both dimensions because custom properties inherit
  as computed values; the final flight frame must match the real card exactly so
  the landing swap is seamless. The `launch-card-to-district-flight` step carries
  the same `commitBufferMs` settle buffer as `draw-card-flight`, because the
  placement commit starts at the step's end. District flights capture
  `performance.now()` before construction and set the CSS animation's start time
  when mounted. It shares the document timeline's time origin but stays current
  when its last rendered frame is stale after idle time or bot search. This keeps
  React/layout work from delaying the flight past the landing swap;
  the settle buffer still leaves room for the final frame to paint. Destination
  flights keep only the card's inset rim on the tile; their outer shadow lives on
  the non-scaled `.card-flight` container as `filter` drop shadows, and the card
  itself runs the source→destination size fit on an inner `.card-flight-scale`
  wrapper. Separating them keeps the shadow's blur and offset at the destination's
  metric for the whole flight, so it never grows or shrinks with the scale (which
  would read as a darker/tighter in-flight shadow that snapped at the landing
  swap). The human player's own placement lands on a slot its ghost already
  fills for the whole flight, and the ghost renders through the lane-card path,
  so the lane's stack filter paints the outer shadow and the ghost paints the
  inter-card shadow. That flight therefore carries *no* outer shadow
  (`.is-destination:not(.is-bot)` in `flights.css`): any copy on the lone flight
  card is a second single-card shadow whose blur spills past the card edge onto
  the neighbouring existing card for the last frames of the flight and snaps
  back at the swap. Bot flights have no ghost, so they paint the landed card's
  own shadow instead — the lane stack shadow on an empty lane or the hand fan,
  the inter-card shadow (`.lane-stack-card:not(:first-child)`, downward in bot
  lanes) when marked `is-stacked`, or a non-stacked deed's depth shadow plus the
  stack shadow. `CardFlightLayer` syncs the animation start time across the
  subtree (container translate and inner scale).
- The placement ghost and the invisible `.lane-card-animation-target` both render
  through the lane-card path inside the lane stack (`.lane-stack-card.placement-ghost`
  and `.lane-card-animation-target`), sharing the real card's container, centering,
  stack shadow and paint layer; only the ghost's desaturation/opacity and action
  glow are layered on. Rendering the preview as a separately-positioned element
  let its suit tokens land a sub-pixel off the card it becomes. `laneCardCount`
  and the stack-step/target fallbacks ignore `.placement-ghost` so a preview never
  counts as a stacked card.
- Sold-card flights land on the discard pile, whose cards render with the
  deck-pile card scope rather than the board card scope. The flight renders at
  the discard box (`renderAtDestination`) and `CardFlightLayer` marks it
  `is-discard-destination`; `flights.css` remaps `--card-padding`,
  `--card-meta-height`, `--card-meta-gap` and `--card-image-area-*` to the
  `--deck-pile-card-*` values and swaps the lane stack shadow for the discard
  card's own depth shadow (as container filters), so the final frame matches the
  landed card instead of overflowing it with board-card metrics.
- A source card that is transformed (the fanned human hand) reports a larger
  axis-aligned rect; flight construction uses the untransformed layout box
  (`offsetWidth`/`offsetHeight`) for the flight's start size so it matches the
  card it departs. The hand fans with per-slot CSS transforms (`.hand-fan-slot`)
  about each card's center, and an invisible `data-hand-slot-kind="empty"` anchor
  marks the next draw's landing slot. A hand card targeted by the action being
  previewed or presented (`ActionHighlights` committed action) is raised above
  its neighbours until its animation finishes.
- The actions menu is intentionally tooltip-free: `ActionsPanel` and every
  `ActionPicker` variant render no `Tooltip` markup. Picker options are
  self-describing through suit tokens and labels, and the popover's stacking
  context sits above the tooltip layer, so reintroduced tooltips there would
  render behind the popover.
- Play-area lane stacks expose tooltips only on the topmost (front) card:
  `DistrictLane` passes `showTooltip={index === laneCards.length - 1}` to
  `CardTile`, and `CardTile`/`ProgressTracker` gate both the tooltip bubble and
  the `tooltip-trigger` class on it. Covered cards stay tooltip-free so hovering
  a stack always names the visible front card.
- Structure is ownership-based: stateless components under
  `src/ui/components/`, controller logic under `src/ui/hooks/` (notably
  `useGameController`, `useBotTurn`, and `useGameAnimations`), and split style
  files under `src/styles/`. Selector-bearing classes/IDs and `data-*` animation
  anchors are compatibility surfaces.
