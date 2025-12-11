# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Iterate TD loops using chunked `collect -> train` with single fixed-size promotion eval.
- Improve model quality against search baseline while keeping loop runtime practical.

## Locked Decisions

- TS engine is canonical gameplay truth.
- Python training/eval calls TS through the bridge contract (`contracts/magnate_bridge.v1.json`).
- Training/eval scripts are fail-fast (no silent fallback labels/actions/probabilities).
- Default promotion decision uses one side-swapped certify eval (`200` games/side).

## Implemented Snapshot

- Browser app is playable; default bot is rollout-search.
- Bridge runtime commands are stable: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Trainer supports policies: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- Canonical eval pipeline is `scripts.eval_suite` with explicit `--mode gate|certify`.
- TD pipeline is operational:
  - replay collection: `scripts.collect_td_self_play`
  - training: `scripts.train_td`
  - loop orchestration: `scripts.run_td_loop`
- `scripts.run_td_loop` supports cloud profile scaling (`--cloud --cloud-vcpus 8|16|32`), collect sharding (`--collect-workers`), and explicit promotion thresholds.
- Overnight runner auto-resolves warm start from latest promoted loop summary (`scripts/run_overnight_td_loop_r2.sh`).

## Remaining

- Keep calibrating loop cadence/thresholds from repeated overnight results.
- Improve `td-search` strength and throughput.
- Wire an online replay refresh loop (beyond chunk-local offline replay files).
- Define browser deployment path for learned TD checkpoints.

## Immediate Next Steps

1. Continue overnight loop iterations with promoted warm starts.
2. Track promotion outcomes and side-gap stability across runs.
3. Rank promoted checkpoints against search baseline with certify evals.

_Updated: 2026-03-05._
