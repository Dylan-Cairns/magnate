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
- TD-root search is the canonical TD-guided browser rollout path: root ranking/priors use opponent/action logits, rollout playouts use opponent/action logits, and non-terminal leaves use TD value predictions. It loads `td-root-search-v1` static model packs and fails fast when no valid pack is available.
- Rollout-search simulations use no-log engine stepping so simulated playouts do
  not grow/copy human-readable game logs; real games and exported transcripts
  still use normal logged stepping.
- Direct TypeScript bot evaluation lives under `src/botEval/` and can run head-to-head evals, rollout-search sweeps, replay checks, and serial or sharded rollout-search TD replay exports.
- TypeScript TD replay action rows include required `actionProbs` targets:
  search policies derive them from root visit counts, and Python opponent/action
  training uses them as soft policy targets.
- Browser UI code is split across controller hooks, animation helpers, stateless components, and ownership-based style files; `App.tsx` remains the composition layer.
- Bridge runtime command surface is stable: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Python policy surface is intentionally narrow: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- Self-play training uses checkpoint selection, accepted-generator gating, replay windows, and `td-lambda` value targets.
- Promoted TD checkpoint warm starts and opponent-pool entries are registered in `models/td_checkpoints/manifest.json`.
- Typed bridge payloads cover the main `trainer/` package; some script orchestration remains dynamic.

## Remaining Work

- Calibrate self-play loop cadence, replay-window settings, and promotion thresholds from repeated runs.
- Add Node-local model-pack loading for direct TypeScript evaluation of serialized TD-root rollout specs outside browser/worker runtime.
- Continue shrinking untyped or dynamic payload handling in Python scripts as those surfaces are touched.
- Keep setup, wrapper, training, and bot-eval procedures in `docs/runbooks/` rather than expanding Memory Bank files.

## Immediate Next Steps

1. Before long training runs, confirm `models/td_checkpoints/manifest.json` and referenced checkpoint files are present and committed.
2. Continue self-play iterations with promoted manifest warm starts, `td-lambda` value targets, checkpoint selection, replay windows, and generator gating.
3. Track checkpoint-selection winners, block-selection winners, generator-gate outcomes, final promotion outcomes, and side-gap stability in artifacts, not Memory Bank prose.
4. Use `yarn bot:eval collect-td-replay-sharded --config configs/bot-eval/collect-td-replay.v2-hard.json --workers <count> --shard-games <games-per-shard>` for large TypeScript teacher replay exports; use `collect-td-replay` for serial debugging.
5. Keep docs aligned by replacing stale Memory Bank bullets rather than appending task history.

_Updated: 2026-06-18._
