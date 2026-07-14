# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Improve TD policy quality through the staged training loop: collect, train, gate, promote.
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
- Rollout-search includes an additive v2 heuristic profile/config path with contextual token-bank valuation; omitted heuristic config preserves v1 root and playout behavior.
- Rollout-search and TD-root search use a deterministic root-search core with stable action keys, seeded world sampling, no-log simulation stepping, diagnostics, and optional worker-backed execution.
- TD-root search is the canonical TD-guided browser rollout path and now supports per-hook guidance selection for experiments: root ranking/priors, rollout playout actions, and non-terminal leaf evaluation can each use either TD model guidance or the existing heuristic fallback. Omitted guidance config preserves the original all-TD behavior. It loads `td-root-search-v1` static model packs and fails fast when a requested TD-guided hook has no valid pack available.
- Rollout-search simulations use no-log engine stepping so simulated playouts do
  not grow/copy human-readable game logs; real games and exported transcripts
  still use normal logged stepping.
- Direct TypeScript bot evaluation lives under `src/botEval/` and can run head-to-head evals, rollout-search sweeps, replay checks, and serial or sharded rollout-search TD replay exports. Node bot-eval installs a local `public/` fetch shim so serialized TD-root model-pack specs can run in the parent process and child-process matchup workers.
- `scripts/calibrate_bot_latency.ts` samples representative decision states and times bot specs on identical states, producing JSON/CSV/Markdown latency calibration artifacts for matching TD and heuristic configs before expensive full-game evals.
- TypeScript TD replay action rows include required `actionProbs` targets:
  search policies derive them from root visit counts, and Python opponent/action
  training uses them as soft policy targets.
- Browser UI code is split across controller hooks, animation helpers, stateless components, and ownership-based style files; `App.tsx` remains the composition layer.
- Browser presentation runtime now uses one `AnimationSequence` per animated
  `GameTransaction`. `useGameAnimations` builds that sequence and schedules
  render-only `viewState` snapshots plus sequence-derived visual commands for
  tax pulses, tax-token flights, and income-token flights from sequence step
  boundaries, so resource counts, visual launches, and commits follow the
  central sequence timing. The old presentation timeline, turn-cycle visual
  timing plan, eager income-choice flight planner, and hook-level turn-cycle
  plan contract have been removed. Action dispatch now only validates/applies
  the engine transition and reports terminal entry; ordinary visual flights and
  commit timing are derived from the animation sequence. Canonical state still
  drives legality, bot scheduling, bug reports, and persistence.
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
  their duration from the owning `AnimationSequence` step, and the overlay copy
  self-removes from that command duration before later sequence steps reveal
  landed state. The deed progress tracker uses its original explicit SVG arc
  geometry with local requestAnimationFrame interpolation, but its duration is
  sourced from the shared animation timing constant. Action commit/input unlock
  are derived from `AnimationSequence` `commitMs`/`inputUnlockMs` rather than
  action-type timing rules. Turn-cycle tax animation is driven by an explicit
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

- Calibrate self-play loop cadence, replay-window settings, and promotion thresholds from repeated runs.
- Continue improving throughput for direct TypeScript TD-root matchups; child-process paired-seed parallelism is available, while individual search decisions remain synchronous in Node.
- Continue shrinking untyped or dynamic payload handling in Python scripts as those surfaces are touched.
- Keep setup, wrapper, training, and bot-eval procedures in `docs/runbooks/` rather than expanding Memory Bank files.

## Immediate Next Steps

1. Before long training runs, confirm `models/td_checkpoints/manifest.json` and referenced checkpoint files are present and committed.
2. Continue self-play iterations with promoted manifest warm starts, `td-lambda` value targets, checkpoint selection, replay windows, and generator gating.
3. Track checkpoint-selection winners, block-selection winners, generator-gate outcomes, final promotion outcomes, and side-gap stability in artifacts, not Memory Bank prose.
4. Use `yarn bot:eval collect-td-replay-sharded --config configs/bot-eval/collect-td-replay.v2-hard.json --workers <count> --shard-games <games-per-shard>` for large TypeScript teacher replay exports; use `collect-td-replay` for serial debugging.
5. Keep docs aligned by replacing stale Memory Bank bullets rather than appending task history.

_Updated: 2026-07-09._
