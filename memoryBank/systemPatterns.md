# System Patterns

## Core Principles

- The TypeScript engine is the only source of gameplay truth.
- Determinism is required: seeded RNG, pure state transitions, no hidden global state.
- UI and Python consume engine legality and observations; they never re-derive rules.

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
- Policy randomness is injected by the controller (seed-derived where
  determinism matters), not hard-coded to `Math.random`.
- Browser model-backed policies load static model packs:
  `public/model-packs/index.json` selects the default pack, each pack provides
  `manifest.json` + `weights.json`, and the loader validates
  schema/checkpoint/encoding/dimension compatibility before use. URL resolution
  must work from both the main window and Web Workers under `base: './'`.
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
  deterministic transcripts, paired seat-swapped scheduling, artifacts, and
  exact replay divergence reporting.
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
- Human input is gated by a transaction-specific decision-window barrier, not by
  pending presentation. Later actions in the same human window use canonical
  legality immediately and may run ahead of visuals.
- Browser DOM lookup and flight construction stay outside the engine (for
  example `domTargets.ts`); sequence-derived visual commands carry the semantics
  needed to launch command-specific flights.
- Structure is ownership-based: stateless components under
  `src/ui/components/`, controller logic under `src/ui/hooks/` (notably
  `useGameController` and `useGameAnimations`), and split style files under
  `src/styles/`. Selector-bearing classes/IDs and `data-*` animation anchors are
  compatibility surfaces.
