# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Improve teacher strength through search-first tuning (determinized rollout search + optional guidance).
- Use `scripts.eval_suite` as the canonical promotion protocol (paired seeds, side-swapped, CI + side-gap).
- Run staged sweep gates (`120 -> 400 -> 2000` total games per preset).
- Train guidance from teacher data only after strong teacher configs are confirmed.

## Locked Decisions

- TS engine is canonical gameplay truth.
- Python training/eval uses the TS bridge contract, not duplicated rules.
- Bridge contract remains small and versioned.
- Legacy BC/REINFORCE benchmark pipeline is out of scope.

## Current State

### UI

- Browser bot catalog now includes rollout search (default) and random.
- Legacy browser PPO profile/model path was removed.
- Bot policy failures are explicit (no silent fallback).

### Bridge

- NDJSON runtime is stable (`metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`).
- Contract artifact: `contracts/magnate_bridge.v1.json`.

### Trainer

- Canonical side-swapped eval suite is implemented:
  - `scripts/eval_suite.py`
  - `trainer/eval_suite.py`
  - supports deterministic game sharding via `--workers` (seed-offset shards merged into one artifact)
- Search/MCTS internals are modularized:
  - `trainer/search/belief_sampler.py`
  - `trainer/search/forward_model.py`
  - `trainer/search/leaf_evaluator.py`
  - `trainer/search/root_selector.py`
- Search and MCTS support optional PPO-format guidance checkpoints for priors/value/opponent modeling.
- Teacher samples support soft policy targets (`actionProbs`).
- Encoding is v2 (`OBSERVATION_DIM=206`) with TS parity encoder.
- Search sweep runner is now eval-suite based with modern pack definitions (`scripts/search_teacher_sweep.py`).
  - supports preset-level parallelism via `--jobs` and forwards eval sharding via `--workers`
- Guidance A/B pipeline uses paired eval seeds for baseline vs guided comparison (`scripts/run_guidance_ab_pipeline.py`).

## Immediate Next Steps

1. Run coarse search sweep (`coarse-v1`, 60 games/side) and keep top 2-3 presets.
2. Confirm top presets at 200 games/side.
3. Run final promotion gate at 1000 games/side for best preset.
4. Generate teacher data from promoted teacher and train guidance checkpoint.
5. Re-run matched-seed eval_suite A/B for guided vs unguided teacher.

_Updated: 2026-03-01._
