# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Improve TD policy quality through the staged training loop: collect, train, gate, promote.
- Keep optionality repetitions 24-47 reserved after both controlled
  district-symmetry training interventions failed the required action-symmetry
  gates. The motivation remains the replay-wide direct audit, not the invalid
  catalog-v1 coordinates.
- Use `scripts.run_td_loop` for bootstrap or recalibration and `scripts.run_td_loop_selfplay` for forward self-play.
- Keep promoted checkpoint registration portable through `models/td_checkpoints/manifest.json`.
- Keep the Python training/eval runtime fail-fast and bridge-backed.

## Locked Decisions

- TS engine is canonical gameplay truth.
- Python training/eval calls TS through the bridge contract (`contracts/magnate_bridge.v1.json`).
- Shared TS/Python boundary remains a small interface contract, not a duplicated rules schema.
- Native Python rules are out of scope unless throughput becomes a proven bottleneck.
- Training/eval scripts should fail fast on invalid bridge payloads, missing checkpoints, malformed policy probabilities, or unsupported policy signals.

## Current State

- Browser play is functional with selectable bot profiles behind a shared async policy contract.
- TypeScript engine partial deed income is submitted simultaneously: pending obligations remain intact, submitted suit choices do not pay resources until all required choices are submitted, then selected resources resolve in deterministic pending-choice order.
- Browser controller/UI partial deed income follows the simultaneous engine phase: human and bot can submit owned income choices from the shared pending state, and resource flights occur on final reveal.
- Vite dev builds support `?fixture=multi-income` for multiple partial-income choices and `?fixture=late-game` for a heuristic-rollout late-game board.
- Bridge, direct TypeScript evaluation, replay collection, and TD policy lookahead use a single policy-facing decision actor during simultaneous income: one unsubmitted income-choice owner is exposed at a time in pending-choice order, with observations and masks aligned to that actor.
- Browser rollout search and TD-root search share deterministic policy plumbing where appropriate; the old standalone browser `td-search` policy path has been retired.
- Paired TD rollout inference passed three increasingly realistic browser
  evaluations on the deployed July checkpoint: exact scalar/paired logits and
  argmax with a 1.313x kernel speedup; exact selected actions, root diagnostics,
  and a complete-game transcript with a 1.252x direct-search speedup; and the
  full outer-worker production stack with exact parity over 128 corpus
  decisions plus 599 searched decisions in four complete shadow games. The
  outer-worker run observed eight workers, batch size 16, root budget 160, no
  p95 regression, and a 1.289x total speedup. The complete evidence and artifact
  location are documented in
  `docs/runbooks/td-outer-worker-shadow-benchmark.md`.
- Eligible parallel browser TD-root searches with TD rollout guidance now
  resolve omitted execution mode to the paired lockstep executor. Synchronous,
  ordinary search, and heuristic-rollout paths remain unchanged. The browser
  query `tdSearchExecutor=legacy` is the session rollback;
  `tdSearchExecutor=paired` is an explicit diagnostic selection, and invalid
  values fail rather than falling back. The paired worker path verifies that
  its loaded action scorer exposes the paired kernel before executing.
  Effective executor mode is returned as outer-worker response metadata,
  separate from search diagnostics/teacher targets. The legacy-authoritative
  outer shadow harness now omits the candidate override and verifies that the
  production default reports `resumable-paired-td`. Its 2026-07-24 browser
  activation smoke passed: the omitted candidate mode resolved to paired, the
  rollback lane resolved to legacy, action and full diagnostics matched exactly,
  and the expected eight workers, batch size 16, root budget 160, and deployed
  checkpoint SHA were observed.
- Rollout-search includes an additive v2 heuristic profile/config path with contextual token-bank valuation; omitted heuristic config preserves v1 root and playout behavior.
- Experimental `StrategicStateSummaryV0` exposes player-view-safe score, clock,
  resource, income-source, deed-feasibility, placement-support, and card-support
  facts without strategic values or probabilities. Exact ActionWindow deltas
  remain factual and do not alter heuristic v2, rollout search, TD encoding, or
  the bridge contract.
- Strategic-position catalog v2 covers district portfolio, tiebreak,
  known-hand and unknown-pool optionality mirrors, deed feasibility, clock, and
  Ace-aware cases. Bot eval supports filtered variants/positions and
  non-overlapping repetition ranges. JSON/Markdown diagnostics report
  information-safe summaries, full case-payload fingerprints, selection
  stability, preferred/alternative/unassessed counts, pairwise focus gaps and
  visits, and counterfactual transitions. Catalog preferences remain reviewed
  hypotheses, not current-bot golden assertions. Catalog validation now
  requires real-game coordinates `D1`-`D5` and the sole Excuse at `D3`; the
  optionality mirrors swap only complete Pawn lanes.
- The original 2026-07-13/14 catalog-v1 optionality work found useful
  continuation and resource-preservation mechanisms, but its TD
  physical-location attribution was invalid: v1 used `D0`-`D4`, shifting the
  observation/action association from normal training, and some mirrors moved
  the Excuse away from its normal fixed `D3` slot. Those artifacts remain
  historical diagnostics, not evidence for permutation training.
- The 2026-07-14 catalog-v2 audit reran all eight optionality positions at 160
  and 800 visits over exposed repetitions 0-23. Destructive overwrites dropped
  from the v1 totals of 44/192 and 6/192 to 8/192 and 0/192. The eight 160-visit
  failures split exactly 4-4 across physical orientations, while every assessed
  800-visit focus choice preserved the continuation. Corrected final choices
  therefore reject catalog v1's orientation-dependent failure, but do not prove
  model symmetry: corrected root priors still showed local D5 preference, and
  deeper search compensated. Detailed rows are in ignored
  `strategic-position-canonical-v2-exposed` artifacts.
- A replay-wide direct symmetry audit now provides valid broad evidence. It
  scanned all 163,194 decisions from the complete 900-game V2 Hard replay and
  evaluated a deterministic 10,000-row sample under all 24 exact permutations
  of D1, D2, D4, and D5 with D3 fixed. For the deployed July pack, pairwise
  agreement was 80.44% (a 19.56% flip rate per non-identity relabeling), while
  4,763 sampled decisions changed under at least one of 23 relabelings. Mean
  maximum probability change was 0.0618 and mean absolute value change was
  0.1036; high-margin choices also changed. Balanced
  slot means favored D4 most broadly (uniquely highest in 462/900 shards), then
  D5 (272), rather than showing one universal D5 rule. This is a meaningful
  violation of an exact game symmetry and justifies a controlled permutation
  training ablation; it does not itself prove a strength gain. Results are in
  ignored `td-symmetry-v2-hard-900-primary` artifacts.
- Symmetry-ablation preflight is complete without starting training. A
  raw-state metamorphic test proves all 24 fixed-D3 transformations match fresh
  encoding of genuinely relabeled legal states/actions and invert exactly. The
  complete 900-game value replay is now local (900 shards, 163,194 rows) beside
  the opponent replay. The missing deployed-July `.pt` pair was reconstructed
  from the checked-in browser pack with strict schema/tensor/provenance checks,
  canonical checkpoint reload, zero internal parity difference, and real-row
  Python/TypeScript differences below `1e-6`. The optimizer-free warm starts are
  in ignored
  `artifacts/td_checkpoints/reconstructed/td-two-stage-imported-20260706-hard-step-30000/`.
- Matched S4 augmentation plumbing and training are complete.
  The Python transformer covers all 24 D1/D2/D4/D5 permutations with fixed D3,
  remaps observations and action candidates, and transforms complete
  TD-lambda player trajectories coherently. Replay sampling and augmentation
  have independent RNG streams; run summaries hash raw sampled indices so
  control/candidate batch equality is directly checkable. Control mode is an
  exact no-op. The frozen checked-in pilot manifest resolves to an 800/100
  shard split and eight commands, each capped at four PyTorch threads and
  intended for sequential execution. The PowerShell pilot launcher validates
  that profile, resumes from completed summaries, prevents duplicate trainers,
  and logs each command. All eight sequential four-thread commands completed
  5,000 updates on 2026-07-16 and produced their final checkpoints; the source
  manifest remains `review-required` with `launchAuthorized=false` as required
  by its non-launching preparation contract.
- The prelaunch guardrail pass is complete. Full, training, and validation
  replay contents now have byte-level fingerprints in addition to inventory
  and membership hashes; each prepared training command verifies replay,
  warm-start, source-manifest, and implementation fingerprints, then embeds
  them in summaries and checkpoints. Four one-update smokes each loaded the
  complete 800-shard/145,014-row training side with four threads. Control and
  S4 used identical raw samples: value trace
  `412d7f3de1755d52bc49410281a9a89e20f3da028e03c419052f6d633f628214`
  and opponent trace
  `8c9ab989480de4306a5cad78ac952861a419905bd03f699fcf4916c254a3fc34`.
  The deployed model-pack index remained unchanged.
- Post-training selection is frozen and stage-one evaluation is complete. The
  preparer passed all trace, provenance, and final-step checks and froze only
  the `pilot-a` S4 step-5,000 pair as the primary candidate before reserved
  repetitions 24-47. `pilot-b` remains replication and cannot replace it;
  contradictory replication blocks promotion. The plan uses all 100
  validation shards for direct value/opponent metrics and prepares
  deterministic heldout symmetry, strategic, and paired full-game commands.
  Candidate/control/cross-component packs remain export commands targeting an
  ignored experimental index; no deployed default or checkpoint registry was
  changed.
- A dedicated stage-one PowerShell launcher completed the seven isolated
  pack exports, four complete heldout checks, and four direct symmetry audits.
  It runs sequentially, caps heldout inference at four threads, validates
  resumable outputs, and does not include strategic or full-game evaluation.
- Completed pilot summaries passed the hard consistency audit: all used the
  frozen replay, warm starts, source manifest, and implementation fingerprint;
  all four control/S4 pairs had identical raw sampling traces. Training alone
  did not by itself establish symmetry improvement or playing strength.
- Stage-one results show a reproducible partial intervention, not a
  promotion-eligible candidate. Both seeds passed heldout noninferiority and
  modestly improved value MSE and opponent cross-entropy. Mean value drift
  under district relabeling fell 69.5% and 64.6%, passing its 50% gate, but
  preferred-action agreement reached only 82.88% and 83.38% versus the 95%
  requirement, while mean probability drift fell only 23.1% and 22.7% versus
  the 50% requirement. The S4 candidate therefore fails the predeclared
  required symmetry diagnostic despite clear improvement; reserved strategic
  repetitions and multi-day promotion games remain unspent.
- The opponent-only complete-orbit follow-up completed both 5,000-update seeds
  with exact matched raw sampling traces. Its control sampled one random
  fixed-D3 S4 permutation per raw row; its treatment averaged ordinary
  soft-target cross-entropy over all 24 permutations and kept the successful
  first-pilot augmented value checkpoint fixed per seed. Both treatments
  passed heldout noninferiority and slightly improved opponent cross-entropy.
  Direct all-24-permutation audits nevertheless missed both required gates:
  primary top-action agreement rose only 82.85% -> 86.36% and probability
  drift fell 25.0%; replication rose 83.03% -> 86.09% and drift fell 25.2%,
  versus requirements of 95% agreement and 50% reduction. This reproducible
  partial improvement is not promotion-eligible. Reserved strategic and
  full-game tests remain unspent. The result favors an opponent model whose
  district symmetry is enforced by architecture rather than more augmentation
  weighting or subjective action boosts.
- Matched forced traces still show a separate heuristic-v2 blind spot:
  heuristic rollout can trade away resources needed for a valuable uncertain
  draw, while TD preserved and realized those continuations. Replacing TD
  rollout with heuristic-v2 rollout increased harmful choices in the v1
  holdout, so that hybrid is not a general remedy.
- Rollout-search and TD-root search use a deterministic root-search core with stable action keys, seeded world sampling, no-log simulation stepping, diagnostics, and optional worker-backed execution.
- TD-root search is the canonical TD-guided browser rollout path and now supports per-hook guidance selection for experiments: root ranking/priors, rollout playout actions, and non-terminal leaf evaluation can each use either TD model guidance or the existing heuristic fallback. Omitted guidance config preserves the original all-TD behavior. It loads `td-root-search-v1` static model packs and fails fast when a requested TD-guided hook has no valid pack available.
- Rollout-search simulations use no-log engine stepping so simulated playouts do
  not grow/copy human-readable game logs; real games and exported transcripts
  still use normal logged stepping.
- Direct TypeScript bot evaluation lives under `src/botEval/` and can run
  head-to-head evals, rollout-search sweeps, strategic comparisons, matched
  forced-rollout traces, direct TD district-symmetry audits, replay checks, and
  serial or sharded rollout-search TD replay exports. Node bot-eval installs a
  local `public/` fetch shim so
  serialized TD-root model-pack specs can run in the parent process and
  child-process matchup workers.
- `scripts/calibrate_bot_latency.ts` samples representative decision states and times bot specs on identical states, producing JSON/CSV/Markdown latency calibration artifacts for matching TD and heuristic configs before expensive full-game evals.
- TypeScript TD replay action rows include required `actionProbs` targets:
  search policies derive them from root visit counts, and Python opponent/action
  training uses them as soft policy targets.
- Browser UI code is split across controller hooks, animation helpers, stateless components, and ownership-based style files; `App.tsx` remains the composition layer.
- Browser presentation runtime now uses one `AnimationSequence` per animated
  `GameTransaction`. `useGameAnimations` builds that sequence and schedules
  render-only `viewState` snapshots plus sequence-derived visual commands for
  tax pulses, tax-token flights, and income-token flights from sequence step
  boundaries, so visible resource counts, visual launches, and presentation
  commits follow the
  central sequence timing. The old presentation timeline, turn-cycle visual
  timing plan, eager income-choice flight planner, and hook-level turn-cycle
  plan contract have been removed. Actor-neutral canonical action dispatch now
  rejects stale source states and actor mismatches, validates/applies one engine
  transition synchronously, advances timeline state immediately, and assigns a
  unique per-session transaction ordinal before presentation begins. Ordinary
  visual flights and visible mutation timing remain derived from the animation
  sequence. Canonical state drives legality, bot scheduling, bug reports, and
  persistence; `useGameAnimations` no longer owns canonical commits. Every
  accepted canonical transition is enqueued for presentation in transaction
  ordinal order. The hook runs one sequence at a time and retains the last
  presented engine state between sequences, so a canonical state that has run
  ahead cannot leak through a temporary null presentation snapshot.
- Presentation event derivation now emits semantic coverage for all canonical
  action families: card placement, action resource payments, deed token/progress
  and completion, sell-card resource gains, and trades. These new events are
  now produce ordered `AnimationSequence` steps for action payments, card
  placement, deed token flights/progress/completion, sell gains, and trades.
  Sequence-derived visual commands now launch draw/sell/card-placement flights,
  action payment token flights, deed token flights, tax flights, and income
  flights from step boundaries; action dispatch no longer queues ordinary
  action card/deed flights. The presentation reducer now applies action payment,
  card placement, deed token landing, deed progress/completion, sell gain, and
  trade resource mutations at their sequence steps, so those visible state
  changes no longer depend on generic commit timing. Deed token development now
  removes player resources at flight start, applies in-card token badges after
  token flight landing, advances the progress tracker afterward, and reveals
  completion last. Sequence-launched resource flight visual commands carry
  their duration from the owning `AnimationSequence` step. Staggered income
  flights now have individual sequence landing steps; each step increments the
  corresponding visible resource count and removes that flight overlay in the
  same scheduled React update, so completed overlays no longer wait for a
  batch income apply. The deed progress tracker uses its original explicit SVG
  arc geometry with local requestAnimationFrame interpolation, but its duration
  is sourced from the shared animation timing constant. Its interpolated ratio
  is scoped to the mounted card, initialized and bounded by the current
  presentation snapshot, and does not persist across deed placements or game
  sessions. Presentation finalization and input unlock are derived from
  `AnimationSequence`
  `commitMs`/`inputUnlockMs` rather than action-type timing rules. Human input is
  blocked only when a transition opens a new human decision window; the barrier
  clears when that specific queued transaction reaches its ordered input-unlock
  boundary. Further actions within the same human action or income-choice
  window use canonical legality immediately and may enqueue while presentation
  is still active. Bot policy selection also follows canonical state without
  waiting for presentation; generation and state-identity checks discard stale
  async policy results before dispatch. Reset cancels the active sequence and
  backlog, while disabling animations finishes queued unlock/settle callbacks
  in order and snaps rendering to canonical state. Turn-cycle tax animation is
  driven by an explicit
  transaction `tax-resolved` semantic event; sticky canonical `lastTaxSuit`
  history must not create tax animation steps for later non-tax actions.
  Dice visual phases are also derived from the sequence presentation overlay;
  `RollResult` is a render-only component and no longer owns d10/d6 sequencing
  timers. Pre-commit presentation snapshots now avoid broad `nextState` copies
  for sell staging, card placement, deed completion, income roll reveal, and
  income-choice request reveal; those visible mutations are driven by the
  current sequence step payload. Sequence-derived draw, sell, and
  card-to-district visual commands now call command-specific DOM flight builders
  instead of re-interpreting the whole action/state diff at launch time. Deed
  token flight commands now also launch from semantic `deed-token-paid` event
  payloads that carry the before/after deed token pools needed for target rail
  layout, so the hook no longer passes action/state diffs into deed flight
  planning. The old queued flight settle path and terminal cleanup flight stub
  have been removed; `useGameAnimations` settles transitions from
  `AnimationSequence.durationMs`.
  Income-choice bot thinking visibility now keys off bot thinking state during
  visible income choice resolution rather than normal turn ownership.
  Buy-deed and develop-outright sequences move/place the card into the district
  before launching and applying payment token removal.
- Bridge runtime command surface is stable: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Python policy surface is intentionally narrow: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- Self-play training uses checkpoint selection, accepted-generator gating, replay windows, and `td-lambda` value targets.
- Imported teacher-replay training uses the shared Python loop process runner for
  stage execution; PowerShell launchers should remain thin environment handoffs,
  not duplicate long-running train orchestration.
- Promoted TD checkpoint warm starts and opponent-pool entries are registered in `models/td_checkpoints/manifest.json`.
- Typed bridge payloads cover the main `trainer/` package; some script orchestration remains dynamic.

## Remaining Work

- Design a district-equivariant opponent/action model intervention. Both
  random-S4 and complete-orbit continuation training improved but failed the
  required action-symmetry gates; do not spend reserved/full-game promotion
  tests on either candidate.
- Keep repetitions 24-47 reserved for evaluating a future candidate selected
  on independent evidence, followed by full-game promotion tests.
- Keep uncertain-draw resource preservation as a separate diagnostic; the
  holdout rejects heuristic-v2 rollout substitution but does not yet identify
  a safe general training feature or target for that mechanism.
- Calibrate self-play loop cadence, replay-window settings, and promotion thresholds from repeated runs.
- Continue improving throughput for direct TypeScript TD-root matchups; child-process paired-seed parallelism is available, while individual search decisions remain synchronous in Node.
- Continue shrinking untyped or dynamic payload handling in Python scripts as those surfaces are touched.
- Keep setup, wrapper, training, and bot-eval procedures in `docs/runbooks/` rather than expanding Memory Bank files.

## Immediate Next Steps

1. Write a short design and guardrail plan for enforcing fixed-D3 S4 symmetry
   in the opponent/action architecture, while preserving the existing replay,
   checkpoint, and browser-export contracts where practical.
2. Keep the uncertain-resource diagnostic separate from symmetry augmentation.
3. Continue self-play iterations with promoted manifest warm starts,
   `td-lambda` value targets, checkpoint selection, replay windows, and
   generator gating.
4. Track checkpoint-selection winners, block-selection winners, generator-gate
   outcomes, final promotion outcomes, and side-gap stability in artifacts, not
   Memory Bank prose.
5. Use `yarn bot:eval collect-td-replay-sharded --config configs/bot-eval/collect-td-replay.v2-hard.json --workers <count> --shard-games <games-per-shard>`
   for large TypeScript teacher replay exports; use `collect-td-replay` for
   serial debugging.
6. Keep docs aligned by replacing stale Memory Bank bullets rather than
   appending task history.

_Updated: 2026-07-24._
