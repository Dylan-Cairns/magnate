# Progress

## Implemented

- Deterministic TS engine and bridge runtime are in place.
- Browser app is playable with policy-agnostic bot selection.
- Default web bot is rollout-eval search; legacy browser PPO path removed.
- Canonical side-swapped eval suite is implemented (`scripts/eval_suite.py`).
- Search/MCTS modular internals and guidance integration are implemented.
- Teacher sample schema includes optional soft policy targets (`actionProbs`).
- Encoding upgraded to v2 (`OBSERVATION_DIM=206`, `encodingVersion` enforced for PPO-format checkpoints).
- Search sweep runner was modernized:
  - now eval-suite based (one artifact per preset with win rate/CI/side-gap)
  - legacy `t1..t8` presets removed in favor of modern preset packs
- Eval throughput controls added:
  - `scripts.eval_suite` now supports `--workers` deterministic shard parallelism
  - `scripts.search_teacher_sweep` now supports `--jobs` and forwards `--workers`
- Guidance A/B pipeline now uses paired eval seeds for cleaner baseline vs guided comparison.

## Removed (Intentional Cleanup)

- `scripts/export_ppo_browser_checkpoint.py`
- `scripts/benchmark.py`
- `scripts/benchmark_queue.py`
- `scripts/train.py`
- `scripts/finetune.py`
- `trainer/benchmarking.py`
- `trainer/behavior_cloning.py`
- `trainer/reinforcement.py`
- Legacy BC/reinforcement/benchmark tests.

## In Progress

- Search preset tuning to reach promotion-grade dominance vs heuristic.
- Guidance checkpoint quality tuning on top of promoted teacher configs.

## Remaining

- Lock promotion thresholds and enforce them in repeated sweeps.
- Complete search->student distillation workflow once teacher dominance is stable.
- Standardize lightweight experiment reporting across multi-run sweeps.

## Risks / Watch Items

- Determinized search strength can improve slower than latency cost.
- Side-gap instability can hide seat bias; treat as a hard promotion risk.
- Guidance may increase compute if injected too broadly at inference.

_Updated: 2026-03-01._
