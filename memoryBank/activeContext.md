# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Continue the self-play-focused TD loop after recent cadence/threshold recalibration, using a `12`-chunk promotion cadence.
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
- GitHub Pages deploy automation is now in place via `.github/workflows/deploy_pages.yml` (push to `main`, gated by `yarn test` + `yarn lint`).
- Bridge runtime commands are stable: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Python bridge client now drains bridge stderr in a background thread to avoid long-run Windows pipe stalls during self-play collection.
- Trainer supports policies: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- Canonical eval pipeline is `scripts.eval_suite` with explicit `--mode gate|certify`.
- TD pipeline is operational:
  - replay collection: `scripts.collect_td_self_play`
  - training: `scripts.train_td`
  - bootstrap/recalibration orchestration: `scripts.run_td_loop`
  - self-play-forward orchestration: `scripts.run_td_loop_selfplay`
- `scripts.run_td_loop` supports cloud profile scaling (`--cloud --cloud-vcpus 8|16|32`), collect sharding (`--collect-workers`), explicit promotion thresholds, and pooled multi-window promotion evals (`--eval-seed-start-indices`).
- Added `scripts.run_td_loop_selfplay` as a separate post-bootstrap loop: chunk-local td-search-heavy mixed collection, promoted opponent-pool sampling, and dual promotion gates (baseline vs `search` plus candidate vs incumbent `td-search`).
- `scripts.run_td_loop_selfplay` collection now supports shard parallelism via `--collect-workers` (cloud profile sets this automatically) while preserving profile-level mixed-opponent collection semantics.
- `scripts.run_td_loop_selfplay` now defaults to `12` chunks before promotion eval so each candidate accumulates materially more collect/train work before certify gating.
- `scripts.run_td_loop_selfplay` orchestration is now split into explicit run-setup, per-chunk execution, promotion-stage, summary, and terminal-report helpers; shared eval/promotion helpers live in `scripts/td_loop_selfplay_common.py`, and targeted orchestration tests plus a self-play smoke test pin the refactor.
- Added `scripts.benchmark_selfplay_collect_setup` to benchmark single-vs-sharded self-play collect throughput on the current machine and recommend a safe `--collect-workers` setting.
- Added `scripts.benchmark_collect_search_profiles` to benchmark laptop-friendly td-search collect throughput across a small `search-worlds` / `search-depth` profile matrix.
- Added Windows laptop entrypoints (`scripts/run_td_loop_bootstrap_laptop.ps1`, `scripts/run_td_loop_selfplay_laptop.ps1`) so local Dell runs can use repo-local temp/cache dirs, manifest-backed warm-start fallback, and an auto-sized CPU budget (`-CpuTargetPercent`, `-ReserveLogicalCores`) without changing the RunPod bash launchers.
- TD loop stage runner now captures child stdout/stderr directly, so Windows wrapper logs include shard-level collect/train/eval progress instead of only parent heartbeats.
- Overnight runner auto-resolves warm start from latest promoted loop summary (`scripts/run_overnight_td_loop_r2.sh`).
- Overnight runner now persists full console logs and exit status under `artifacts/logs/` before pod teardown.
- Added interrupted-run recovery helpers:
  - `scripts/resume_td_loop_run.py` for the older bootstrap chunk-003 recovery path
  - `scripts/resume_td_loop_selfplay.py` for interrupted self-play loops; it resumes from the latest fully completed chunk, reruns the next partial chunk from scratch, preserves the original incumbent checkpoint, and then completes the remaining chunks plus dual promotion evals
  - `scripts/resume_td_loop_selfplay_laptop.ps1` so the same recovery path is available through the Windows laptop runtime wrapper

## Remaining

- Keep calibrating loop cadence/thresholds from repeated overnight results.
- Improve `td-search` strength and throughput.
- Wire an online replay refresh loop (beyond chunk-local offline replay files).
- Tune browser `td-search` latency/throughput now that `td-search fast` is the default profile.

## Immediate Next Steps

1. Use the Windows laptop wrappers for local runs while preserving the existing RunPod bash launchers for cloud re-entry.
2. Continue overnight self-play loop iterations with promoted warm starts and the `12`-chunk cadence.
3. Track dual-gate outcomes (baseline vs search and candidate vs incumbent td-search) plus side-gap stability.

_Updated: 2026-04-19._
