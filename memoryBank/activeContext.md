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
- Vite dev builds support `?fixture=multi-income` to open a browser state with multiple rank-2 partial-income deed choices for UI testing.
- Bridge, direct TypeScript evaluation, replay collection, and TD policy lookahead use a single policy-facing decision actor during simultaneous income: one unsubmitted income-choice owner is exposed at a time in pending-choice order, with observations and masks aligned to that actor.
- Browser heuristic, rollout search, TD search, and TD-root search share deterministic policy plumbing where appropriate.
- Rollout-search includes an additive v2 heuristic profile/config path with contextual token-bank valuation; omitted heuristic config preserves v1 root and playout behavior.
- Rollout-search and TD-root search use a deterministic root-search core with stable action keys, seeded world sampling, and optional worker-backed execution.
- Direct TypeScript bot evaluation lives under `src/botEval/` and can run head-to-head evals, rollout-search sweeps, replay checks, and rollout-search TD replay exports.
- Browser UI code is split across controller hooks, animation helpers, stateless components, and ownership-based style files; `App.tsx` remains the composition layer.
- Bridge runtime command surface is stable: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Python policy surface is intentionally narrow: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- Self-play training uses checkpoint selection, accepted-generator gating, replay windows, and `td-lambda` value targets.
- Promoted TD checkpoint warm starts and opponent-pool entries are registered in `models/td_checkpoints/manifest.json`.
- Typed bridge payloads cover the main `trainer/` package; some script orchestration remains dynamic.

## Remaining Work

- Improve `td-search` strength and throughput.
- Calibrate self-play loop cadence, replay-window settings, and promotion thresholds from repeated runs.
- Add Node-local model-pack loading for direct TypeScript evaluation of serialized `td-search` specs.
- Continue shrinking untyped or dynamic payload handling in Python scripts as those surfaces are touched.
- Keep setup, wrapper, training, and bot-eval procedures in `docs/runbooks/` rather than expanding Memory Bank files.

## Immediate Next Steps

1. Before long training runs, confirm `models/td_checkpoints/manifest.json` and referenced checkpoint files are present and committed.
2. Continue self-play iterations with promoted manifest warm starts, `td-lambda` value targets, checkpoint selection, replay windows, and generator gating.
3. Track checkpoint-selection winners, block-selection winners, generator-gate outcomes, final promotion outcomes, and side-gap stability in artifacts, not Memory Bank prose.
4. Use `yarn bot:eval collect-td-replay --config configs/bot-eval/collect-td-replay.rollout-search.example.json` when direct TypeScript rollout-search replay exports are needed for TD training.
5. Keep docs aligned by replacing stale Memory Bank bullets rather than appending task history.

_Updated: 2026-06-09._
