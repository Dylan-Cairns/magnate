# System Patterns

## Core Principles

- TS engine is the only gameplay truth source.
- Determinism is required (seeded RNG, pure state transitions).
- UI and trainer consume legality/observations from TS, not re-derived rules.

## Engine Pattern

Primary APIs:

- `legalActions(state)`
- `applyAction(state, action)`
- `advanceToDecision(state)`
- `toPlayerView(state, viewerId)` / `toActivePlayerView(state)`
- `decisionPlayerIdForState(state)` / `legalActionsForDecisionPlayer(state)` /
  `toDecisionPlayerView(state)` for policy/bridge-facing single-actor decisions

Design expectations:

- No side effects in rules logic.
- Immutable state updates.
- Phase-driven turn flow.

## Client Controller Pattern

- Client loop should stay thin and deterministic:
  - `state = createSession(seed, firstPlayer)`
  - render from `toPlayerView(state, viewerId)`
  - apply only `legalActions(state)` through `stepToDecision(state, action)`
- UI-only convenience controls may exist outside `legalActions` if they do not alter rules semantics or policy/bridge contracts:
  - example: human-only turn reset that restores a previously captured engine snapshot.
- Browser-only dev fixtures may be URL-gated under Vite dev mode for UI testing
  hard-to-reach states; fixtures should still derive legality and decisions
  through canonical engine APIs after constructing the starting state or by
  rolling forward with a deterministic policy.
- Bot/human action selection should sit behind a shared policy contract so swapping random -> trained does not change controller flow.
- Policy contract should allow async selection so browser model inference can plug in without changing controller flow.
- Bot profile selection should resolve through a small catalog:
  - available profiles use their own policy implementation
  - unavailable profiles are disabled in UI; profile resolution throws if selected programmatically (no silent fallback)
  - current browser profiles prioritize deterministic search-based play
- Browser heuristic v1 play should use one shared TypeScript scorer for direct heuristic play, rollout-search root ranking, and TD-search heuristic root priors.
- Rollout-search specs may opt into additive heuristic variants for both root ranking/priors and rollout playout action selection; omitted heuristic config preserves v1 behavior.
- Browser rollout-search leaf evaluation should stay separate from action scoring and estimate unfinished rollout states from canonical district scoring, deed potential/threat pressure, late-game urgency, and contextual resource-bank quality.
- Rollout-search simulations should use deterministic common-random scenarios by action-local visit index: hidden-world samples are reused by scenario index and each rollout replaces the real game seed/rng cursor with a simulated engine seed, so root actions compare matched hidden-card and future dice/tax/shuffle samples without leaking the actual game RNG stream.
- Rollout-search simulations should use simulation-only stepping that suppresses
  game-log growth while preserving normal logged stepping for real games and
  exported transcripts.
- Browser rollout-search parallelism should live below the `ActionPolicy`
  boundary:
  - the UI/controller still awaits one async policy decision;
  - the worker-backed bot may coordinate nested browser search workers for
    deterministic root-visit batches;
  - seeded per-visit RNG and ordered result merging must make worker response
    timing irrelevant;
  - worker-count resolution may choose an explicit serial path, but unexpected
    parallel pool, worker, protocol, or clone failures are hard errors.
- New search algorithms should reuse the deterministic search-session pattern
  where practical: root state/stat ownership in the coordinator, cloneable
  worker tasks/results, stable action keys for merging, and algorithm-specific
  rollout/leaf/opponent kernels behind a small task-result union.
- TD-root rollout-search variants keep the current rollout-search kernel,
  sampled hidden worlds, root visit scheduling/UCB, no-log simulation stepping,
  diagnostics, and worker-backed execution while replacing heuristic guidance
  with TD model-pack signals: root action ordering and priors from
  opponent/action logits, non-terminal leaf values from the value model, and
  rollout playout action choice from opponent/action logits.
- Heuristic v1 scorer rules should stay action-level and engine-state-derived: avoid duplicating rule legality, avoid speculative placement-chain/Ace-bonus preferences, and treat trades as penalties unless the simulated post-trade resources immediately unlock a high-value development or deed move.
- Heuristic v2 should remain additive and broad-delta based: score district-local saturated scoring-margin deltas, future suit-access earning deltas, and contextual token-bank deltas without rewarding generic resource hoarding or v1 tactical patchwork constants.
- Heuristic district-potential scoring must include newly bought deeds, while opponent deed defense pressure should scale with completion progress rather than full card rank from zero progress.
- Non-completing deed progress should not receive full new-control-path credit; spending a last suit token on partial deed progress should usually lose to ending the turn, while completion and surplus-token progress can still be rewarded.
- Browser model-backed policies should load from static model-pack artifacts:
  - `public/model-packs/index.json` selects default pack
  - each pack provides `manifest.json` + `weights.json`
  - loader validates schema/checkpoint/encoding/dim compatibility before policy use
  - URL resolution must work from both the main window and Web Workers under
    `base: './'`, including GitHub Pages subpath hosting; worker URLs under
    built `/assets/` should resolve public model-pack paths from the app root,
    not beside the worker bundle.
- Policy randomness should be injected by the controller (seed-derived where determinism matters), not hardcoded to `Math.random`.
- Browser bot evaluation should run the actual TypeScript `ActionPolicy`
  implementations directly through `src/botEval/`, using serializable `BotSpec`
  definitions and the same versioned state-derived policy RNG helper as browser
  play.
- TypeScript head-to-head bot evaluation should use paired seeds with swapped
  policy seats, record stable action-key transcripts, write JSON plus Markdown
  artifacts, and support exact-game replay checks.
- TypeScript bot evaluation and replay collection should use the policy-facing
  decision actor, not raw turn ownership, so simultaneous `CollectIncome`
  submissions are attributed to the player whose income choice is being
  selected.
- TypeScript TD replay exports should include action-policy targets for every
  opponent/action row:
  - search policies emit visit-derived `actionProbs` aligned to canonical legal
    action order;
  - non-search policies emit a one-hot selected-action target;
  - Python opponent/action training consumes these soft targets rather than
    hard-label cross entropy only.
- Large TypeScript TD replay exports may use process-level contiguous sharding:
  each child owns a disjoint global game-index range, preserves the same
  seed/game-id sequence as serial collection, writes independent shard JSONL
  files, may queue more fixed-size shard jobs than worker processes for
  runtime balancing, and leaves Python training to consume multiple replay files
  directly.
- TypeScript rollout-search sweeps should run candidate configs sequentially
  against one fixed opponent with one shared paired-seed prefix. A configurable
  persistent child-process pool may distribute whole paired seeds for
  throughput while preserving deterministic artifact ordering. Preserve
  replayable child matchups and write aggregate JSON, CSV, and Markdown
  summaries. Record whether latency is isolated (`workers=1`) or loaded
  (`workers>1`) so the two measurements are not conflated.
- Long bot-eval runs should stream human-readable progress to stderr while
  preserving final machine-readable JSON on stdout. Sweep aggregates should be
  written before compute starts and atomically refreshed after each durable
  completed candidate; active-candidate resume is not required.
- Rollout-search evaluation diagnostics should remain optional at the policy
  boundary so browser behavior is unchanged while direct evaluation can record
  legal-root-action counts, actual simulated engine steps, terminal rollout
  rates, per-root-action visits/mean values/terminal rates, and selected-action
  value details.
- Additive policy implementations should not replace existing training/eval paths:
  - policy kinds are wired through one factory (`policy_from_name(...)`)
  - policies that spawn external resources (for example bridge subprocesses for simulation) expose `close()`
- UI score presentation should be derived, not stateful:
  - compute live score from canonical engine state (`scoreGame(state)`) on render
  - reuse same score component for terminal and non-terminal states
- UI animation sequencing should keep pure timing, turn-cycle visual-plan
  derivation, injectable browser-only DOM target resolution, and flight
  planning in `src/ui/animations/`; browser DOM access remains outside the
  engine and isolated in `domTargets.ts`.
- UI animation lifecycle state, timer cleanup, presentation snapshot scheduling,
  and delayed commit coordination should live in `useGameAnimations`; canonical
  state transitions and timeline logging remain injected controller callbacks.
- The next UI animation architecture is being staged under `src/ui/runtime/`:
  action dispatch should produce semantic `GameTransaction` events and a
  presentation timeline, then derive a coherent render-only `viewState` plus
  overlays from that runtime instead of scattering per-component previews and
  holdbacks.
- Browser rendering should prefer the controller-provided `viewState`/player
  view for visual game data while keeping canonical `state` internal for
  legality, bot scheduling, bug reports, and persistence.
- App-level visual helpers should accept `viewState` rather than canonical
  state, so animated transitions cannot leak already-committed engine results
  into visible UI before the presentation timeline reaches them.
- The target animation architecture is one sequence per `GameTransaction`:
  `buildAnimationSequence(transaction)` produces ordered semantic steps with
  durations as the single timing source. React components should eventually
  render snapshots/overlays derived from that sequence rather than owning
  sequence timers locally.
- Presentation `viewState` snapshots should be derived from
  `AnimationSequence` step boundaries. Launch steps may start visual flights,
  but canonical-looking resource/card/count mutations belong to explicit
  sequence apply/commit steps so React cannot leak `nextState` before the
  sequence reaches it.
- Browser token visual effects should be scheduled from sequence-derived visual
  commands. DOM target lookup and flight object construction stay browser-only,
  but the command `atMs`/`startMs` values must come from `AnimationSequence`
  steps rather than a parallel turn-cycle visual timing plan.
- Animation overlays that describe semantic presentation state, including
  income source highlights and active-player visual holdbacks, should be
  emitted by the animation sequence reducer rather than maintained by
  independent hook-local timers.
- Human and bot browser actions should share `prepareActionDispatch` for
  engine validation and animation-plan assembly. Validate with
  `stepToDecision` before DOM-dependent planning so invalid actions cannot
  mutate UI-only deed-layout memory.
- Browser `CollectIncome` controls follow income-choice ownership rather than
  normal turn ownership:
  - human input is available when the human has an unsubmitted
    `choose-income-suit` action, even if the main turn owner is the bot
  - bot scheduling can select bot-owned income choices from the same pending
    income state, including during the turn-cycle input window
  - normal `ActionWindow` controls remain blocked until all simultaneous income
    choices resolve
  - partial-income resource flights are planned only on the final reveal that
    actually applies submitted choices
- Browser session state, timeline logging, reset snapshots, bot scheduling,
  shared action dispatch, and animation-hook composition should live in
  `useGameController`; `App.tsx` should retain UI-local menu, picker, preload,
  and rendering concerns.
- Characterization tests should protect UI commit timing, tax-resource previews,
  composite picker resolution/invalidation, and log presentation while
  `App.tsx` is decomposed.
- Stateless browser rendering blocks should live under `src/ui/components/`
  and receive derived data plus callbacks from `App.tsx`; selector-bearing
  classes, IDs, and `data-*` animation anchors are compatibility surfaces.
- Browser styles should stay split by ownership under `src/styles/`: global
  primitives, layout, board, card primitives, flights, rendered side-panel
  components, and a final responsive override layer. Preserve import order
  when moving rules so selector-bearing compatibility surfaces do not change.
- Action-panel and picker components stay controlled: `App.tsx` owns picker
  state, positioning callbacks, refs, dismiss hooks, and action execution,
  while pure category and picker-conversion helpers live under `src/ui/`.

## Turn-Flow Pattern

- Non-decision phases auto-resolve via `advanceToDecision`.
- Decision phases are where external actors choose actions (`CollectIncome` with unsubmitted income choices and `ActionWindow`).
- Draw/exhaustion handling and final-turn countdown are part of phase resolution.
- Draw exhaustion source is canonical in `deck.reshuffles` (no duplicate exhaustion field).
- Income-choice return owner is stored as `PlayerId`.
- Card-play gating is explicit (`cardPlayedThisTurn`):
  - exactly one card-play action per turn
  - `ActionWindow` uses a unified action surface:
    - pre-card: `trade`, `develop-deed`, and card-play actions
    - post-card: `trade`, `develop-deed`, and `end-turn`
- Partial deed income choices are submitted before resolution:
  - `pendingIncomeChoices` keeps the full obligation list for the roll
  - `submittedIncomeChoices` records selected suits without applying resources immediately
  - once every pending choice has a submission, selected resources are applied in deterministic pending-choice order
  - the original turn owner remains the normal action-window owner after simultaneous income resolves

## Bridge Pattern

- Bridge is the runtime boundary between TS engine and Python trainer.
- Contract is versioned and intentionally small.
- Stable items: request/response envelope, commands, action IDs, observation layout metadata.
- Runtime transport is NDJSON over stdin/stdout via `src/bridge/cli.ts`.
- Command handling lives in `src/bridge/runtime.ts` and returns strict success/error envelopes.
- Python bridge clients must continuously drain bridge stderr to avoid long-run pipe stalls on Windows.
- The bridge preserves a single policy actor per request:
  - normal phases use the engine turn owner
  - simultaneous `CollectIncome` uses the first unsubmitted
    `pendingIncomeChoices` owner
  - bridge `legalActions`, legal masks, raw `step` action validation, and
    returned actor views are filtered/aligned to that decision actor
- Canonical bridge action surface comes from `src/engine/actionSurface.ts`:
  - stable action keys
  - canonical legal-action ordering by lexicographic action key

## Training Pattern

- Current baseline policy is determinized rollout search.
- Search is treated as warm-start infrastructure, not final architecture.
- TD training architecture is staged around shared primitives in `trainer/td`, replay collection, checkpointed training, and promotion-gated evaluation.
- Bootstrap/recalibration loop (`scripts.run_td_loop`) uses chunked offline replay generation, checkpointed training, and pooled side-swapped certify windows before promotion.
- Forward self-play loop (`scripts.run_td_loop_selfplay`) keeps strict gating while shifting collection toward mixed td-search-heavy opponents and incumbent head-to-head checks.
- Self-play generation is accepted-checkpoint gated:
  - after each chunk trains, saved checkpoints are first compared against the current accepted generator with a cheap td-search eval;
  - the selected chunk checkpoint, not automatically the final training step, advances the learner checkpoint for subsequent training warm starts;
  - `--generator-update-chunks` controls generator cadence; non-boundary chunks defer the generator gate while learner training continues cumulatively;
  - at each full block boundary, and at the final partial block, the loop selects the best chunk candidate from the block with a cheap td-search eval using the `--block-selection-*` settings;
  - the selected block candidate runs a resumable sequential td-search vs current accepted-generator gate;
  - only passing block candidates become the generator for subsequent collection and promotion eligibility;
  - rejected candidates stay in artifacts as trained candidates but are not used for future self-play generation;
  - final promotion eval still runs separately against the fixed search baseline and manifest incumbent.
- Self-play training uses a configurable replay-window artifact:
  - each chunk records an ordered replay-window manifest under `train/replay_window/window.summary.json` that references the current collect replay plus up to `N-1` prior chunks from the selected source;
  - `--train-replay-window-source accepted` uses prior gate-passing chunks for compatibility;
  - `--train-replay-window-source recent` uses prior trained chunks even when generator gates failed, allowing cumulative learner updates while keeping generator promotion conservative;
  - default `N=3` enables a small replay window for normal self-play runs;
  - `scripts.train_td` now accepts ordered replay-file lists directly, so replay windows do not duplicate chunk replay data on disk;
  - value training defaults to sequence-aware `td-lambda` targets with `lambda=0.7`;
  - replay-window value line caps are fail-fast under `td-lambda` because raw line caps can split complete trajectories;
  - rejected chunks are not eligible for accepted-source replay windows, but can be included by the recent-source learner window.
- Per-chunk durability lives in `chunks/chunk-XXX/chunk.summary.json`; block generator decisions live in `blocks/block-XXX/block.summary.json`. Resume requires the current chunk-summary schema, including checkpoint-selection metadata, reconstructs learner/generator checkpoints separately, restores accepted and recent replay histories, and carries any trailing non-boundary chunk candidates into the next block gate.
- Replay regime in loop orchestration is explicit `chunk-local` for bootstrap and `chunk-local-selfplay-mixed` for the self-play loop.
- TD checkpoint registry is source-controlled:
  - `models/td_checkpoints/manifest.json` schema v2 is the canonical source for `defaultWarmStart` and `opponentPool`.
  - referenced promoted checkpoint files live under `models/td_checkpoints/<key>/` so they move with the repo; ignored `artifacts/td_loops/*/loop.summary.json` files are fallback/history only.
  - successful loop promotions copy the latest accepted value/opponent pair into the checkpoint registry and update the manifest unless `--disable-manifest-promotion` is set.
  - self-play opponent-pool loading reads manifest pool entries first, then promoted artifact summaries, and de-duplicates by checkpoint pair.
- Platform-specific runtime tuning lives in thin wrapper scripts, not the canonical Python loop defaults:
  - RunPod/Linux uses bash launchers with cloud presets.
  - Windows laptop runs use PowerShell launchers that set temp/cache dirs, CPU thread caps, and explicit worker counts.
- Long-running TD orchestration should use the shared Python step runner in
  `scripts.td_loop_common.run_step` for merged stdout/stderr, live output,
  heartbeat/progress writes, and fail-fast return-code handling; PowerShell
  wrappers should not duplicate per-stage train execution.
- Canonical evaluation is `scripts.eval_suite` with explicit modes:
  - loop default: `--mode certify` for fixed-size side-swapped promotion evals
  - self-play chunk gating now uses `--mode gate` for resumable sequential incumbent tests
- Python policy surface is intentionally narrow during TD pivot:
  - `random`
  - `heuristic`
  - `search`
  - `td-value` (checkpoint-backed value policy for benchmarking)
  - `td-search` (checkpoint-backed TD-guided search policy)
- Training code uses fail-fast semantics:
  - no silent fallback to heuristic labels/actions when required TD/search signals are missing
  - `td-lambda` value training requires sequence-aware replay rows with full contiguous player trajectories
  - teacher label generation requires policies that emit root action probabilities
  - malformed bridge payloads and invalid distributions raise immediately with context
  - script entrypoints require explicit policy args and active virtualenv runtime

## Testing Pattern

- Unit tests for helpers, legality generation, reducer behavior, and turn flow.
- Unit tests for visibility boundaries (hidden opponent hand, hidden draw order).
- Deterministic fixtures and seed-based replay paths.
- Contract tests to protect TS/Python integration behavior.
- Python bridge-client/encoding/eval scaffolding has focused tests in `trainer_tests/`.
- Search policies have focused tests for deterministic action choice and legal-action guarantees.
- TypeScript browser-bot evaluation has focused tests for serializable specs,
  full-game deterministic transcripts, legal-action enforcement, paired
  seat-swapped scheduling, artifacts, and exact replay divergence reporting.
- Side-swapped promotion evals run through a canonical pooled-window pipeline:
  - paired seeds with swapped policy seats
  - Wilson confidence interval reporting
  - explicit side-gap reporting

## Versioning Pattern

- Include `schemaVersion` in serialized engine state.
- Include `contractVersion` in bridge metadata/responses where applicable.
- Breaking bridge changes require a major contract bump.
