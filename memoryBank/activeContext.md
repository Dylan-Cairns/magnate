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
- Python trainer/eval runtime now standardizes on Python `3.12+`; Windows local setup already provisions `.venv` with `3.12`, and runtime guards/docs now match that floor.
- Local Python setup now installs `requirements-dev.txt`, and the repo includes a Ruff baseline (`ruff.toml`) targeting Python `3.12` with high-signal lint checks plus import sorting.
- Repo-local Python typing now standardizes on the project `.venv`: VS Code workspace settings pin `.venv\Scripts\python.exe`, `requirements-dev.txt` includes `pyright`, and `pyrightconfig.json` checks the `trainer/` package against that environment.
- Python bridge client now drains bridge stderr in a background thread to avoid long-run Windows pipe stalls during self-play collection.
- Python bridge ingress is now typed and validated once at the boundary: `trainer/bridge_payloads.py` defines the consumed TS bridge payload subset, `trainer/bridge_parsing.py` parses raw bridge JSON into that subset, and `trainer/bridge_client.py`/`trainer/env.py` expose typed state, view, action, and metadata payloads to the trainer stack.
- Trainer supports policies: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- Trainer policy internals are now split across focused modules (`trainer/basic_policies.py`, `trainer/value_policy.py`, `trainer/search_policy.py`, `trainer/policy_factory.py`); `trainer/policies.py` remains the stable public facade for scripts/tests.
- Core trainer policy/search flow now consumes the typed bridge boundary directly: the policy interface, search determinization helpers, forward model, teacher-data collection, self-play collection, and evaluation/training sampling paths no longer rely on raw `Dict[str, Any]` bridge payloads at their main entry points.
- Core model-facing internals now use the typed bridge payloads directly: `trainer/encoding.py` and `trainer/search/leaf_evaluator.py` no longer treat state/view payloads as generic nested mappings.
- Trainer-owned JSON/checkpoint edges are now typed too: `DecisionSample` JSONL, TD replay JSONL (`trainer/td/io.py`), and TD checkpoint payloads (`trainer/td/checkpoint.py`) now parse into explicit payload models instead of leaking `Any` through the trainer package.
- Checked-in static analysis now covers both `trainer/` and the corresponding trainer-side tests under `trainer_tests/`; eval-suite script tests remain outside that checked-in pyright scope because the `scripts/` tree is still intentionally excluded.
- Canonical eval pipeline is `scripts.eval_suite` with explicit `--mode gate|certify`.
- TD pipeline is operational:
  - replay collection: `scripts.collect_td_self_play`
  - training: `scripts.train_td`
  - bootstrap/recalibration orchestration: `scripts.run_td_loop`
  - self-play-forward orchestration: `scripts.run_td_loop_selfplay`
- `scripts.run_td_loop` supports cloud profile scaling (`--cloud --cloud-vcpus 8|16|32`), collect sharding (`--collect-workers`), explicit promotion thresholds, and pooled multi-window promotion evals (`--eval-seed-start-indices`).
- `scripts.run_td_loop` orchestration is now split into explicit run-setup, per-chunk execution, promotion-stage, summary, and terminal-report helpers, and targeted bootstrap-loop orchestration tests plus a smoke test pin that structure.
- Added `scripts.run_td_loop_selfplay` as a separate post-bootstrap loop: chunk-local td-search-heavy mixed collection, promoted opponent-pool sampling, and dual promotion gates (baseline vs `search` plus candidate vs incumbent `td-search`).
- `scripts.run_td_loop_selfplay` collection now supports shard parallelism via `--collect-workers` (cloud profile sets this automatically) while preserving profile-level mixed-opponent collection semantics.
- `scripts.run_td_loop_selfplay` now defaults to `12` chunks before promotion eval so each candidate accumulates materially more collect/train work before certify gating.
- Bootstrap, bootstrap-resume, and self-play TD paths now share generic eval row parsing, pooled-window payload building, and promotion-gate evaluation through `scripts/td_loop_eval_common.py`; self-play-specific dual-gate eval command wiring now lives in `scripts/td_loop_selfplay_eval.py`.
- `scripts.run_td_loop_selfplay` orchestration is now split into explicit run-setup, per-chunk execution, promotion-stage, summary, and terminal-report helpers, and targeted orchestration tests plus a self-play smoke test pin the refactor.
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
- Continue shrinking untyped/dynamic payload handling in the remaining Python script/orchestration layer outside the typed `trainer/` package.

## Immediate Next Steps

1. Use the Windows laptop wrappers for local runs while preserving the existing RunPod bash launchers for cloud re-entry.
2. Continue overnight self-play loop iterations with promoted warm starts and the `12`-chunk cadence.
3. Track dual-gate outcomes (baseline vs search and candidate vs incumbent td-search) plus side-gap stability.
4. Extend the typed rollout from `trainer/` into the remaining `scripts/` orchestration and export helpers as those surfaces are touched.

_Updated: 2026-04-20._
