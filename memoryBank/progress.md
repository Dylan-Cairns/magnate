# Progress

## Implemented

- Deterministic TS engine and bridge runtime are in place.
- Browser app is playable with policy-agnostic bot selection.
- Default web bot is rollout-eval search.
- Canonical side-swapped eval suite is implemented (`scripts/eval_suite.py`).
- Search root logic parity was implemented between Python and browser TS policy.
- Search sweep runner is eval-suite based (`scripts/search_teacher_sweep.py`).
- Eval throughput controls are in place:
  - `scripts.eval_suite --workers`
  - `scripts.search_teacher_sweep --jobs` (preset-level parallelism)

## Removed (Intentional Cleanup)

- PPO training stack:
  - `scripts/train_ppo.py`
  - `scripts/train_ppo_queue.py`
  - `trainer/ppo_model.py`
  - `trainer/ppo_training.py`
- MCTS policy stack (policy + CLI surface + tests).
- Guidance/distillation stack:
  - `scripts/train_search_guidance.py`
  - `scripts/run_guidance_ab_pipeline.py`
  - `trainer/guidance_training.py`
- Related PPO/MCTS/guidance tests and documentation references.

## In Progress

- Search preset tuning to lock a stable warm-start baseline vs heuristic.
- Preparation for TD/Keldon training module implementation.

## Remaining

- Add TD value training loop (bootstrapped from self-play trajectories).
- Add opponent action model.
- Integrate TD value + opponent model into search (`td-search` path).
- Define and document TD checkpoint contract.

## Risks / Watch Items

- Search-only strength can plateau below strong-human target.
- Side-gap instability can hide seat bias; treat as a hard promotion risk.
- Warm-start data can encode heuristic biases; TD training must move beyond it.

_Updated: 2026-03-01._
