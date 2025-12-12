# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Continue the self-play-focused TD loop after recent cadence/threshold recalibration.
- Use `scripts.run_td_loop` for bootstrap/recalibration passes, then advance primarily with `scripts.run_td_loop_selfplay`.
- Improve model quality against search baseline and incumbent td-search while keeping runtime practical.

## Locked Decisions

- TS engine is canonical gameplay truth.
- Python training/eval calls TS through the bridge contract (`contracts/magnate_bridge.v1.json`).
- Training/eval scripts are fail-fast (no silent fallback labels/actions/probabilities).
- Default promotion decision uses pooled side-swapped certify eval windows (`200` games/side per window).

## Implemented Snapshot

- Browser app is playable; default bot is now browser `td-search fast`.
- Browser app loads exported model packs from static `public/model-packs/` artifacts; UI currently exposes `td-search fast`, `td-search (browser)`, rollout-search, and random profiles.
- Browser startup now preloads card images and TD model packs behind a centered modal with blurred backdrop; bot turns are blocked until preload completes.
- Browser bootstrap now renders an immediate loading shell from `index.html` and lazy-loads the React `App` module, so first paint is no longer gated on eager engine/policy module evaluation.
- Bridge runtime commands are stable: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Trainer supports policies: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- Canonical eval pipeline is `scripts.eval_suite` with explicit `--mode gate|certify`.
- TD pipeline is operational:
  - replay collection: `scripts.collect_td_self_play`
  - training: `scripts.train_td`
  - bootstrap/recalibration orchestration: `scripts.run_td_loop`
  - self-play-forward orchestration: `scripts.run_td_loop_selfplay`
- `scripts.run_td_loop` supports cloud profile scaling (`--cloud --cloud-vcpus 8|16|32`), collect sharding (`--collect-workers`), explicit promotion thresholds, and pooled multi-window promotion evals (`--eval-seed-start-indices`).
- Added `scripts.run_td_loop_selfplay` as a separate post-bootstrap loop: shorter collect/train cadence, td-search-heavy mixed collection, promoted opponent-pool sampling, and dual promotion gates (baseline vs `search` plus candidate vs incumbent `td-search`).
- Overnight runner auto-resolves warm start from latest promoted loop summary (`scripts/run_overnight_td_loop_r2.sh`).
- Overnight runner now persists full console logs and exit status under `artifacts/logs/` before pod teardown.
- Added one-off interrupted-run recovery helper: `scripts/resume_td_loop_run.py` (resume from chunk-003 train, then promotion eval + loop summary); now supports cloud/thread scaling overrides (`--cloud --cloud-vcpus 8|16|32`, `--train-num-threads`, `--train-num-interop-threads`).

## Remaining

- Keep calibrating loop cadence/thresholds from repeated overnight results.
- Improve `td-search` strength and throughput.
- Wire an online replay refresh loop (beyond chunk-local offline replay files).
- Tune browser `td-search` latency/throughput now that `td-search fast` is the default profile.

## Immediate Next Steps

1. Continue overnight self-play loop iterations with promoted warm starts.
2. Track dual-gate outcomes (baseline vs search and candidate vs incumbent td-search) plus side-gap stability.
3. Re-run bootstrap/recalibration loop only when balance or cadence retuning is needed.

_Updated: 2026-03-07._
