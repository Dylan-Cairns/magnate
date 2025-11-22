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
- TD Phase 1 foundation is implemented:
  - `trainer/td/models.py` (ValueNet, OpponentModel)
  - `trainer/td/replay.py` (value/opponent replay buffers)
  - `trainer/td/targets.py` (`n-step`, `TD(lambda)` target helpers)
  - `trainer/td/checkpoint.py` (TD checkpoint contracts and load/save)
  - `trainer/td/self_play.py` (self-play trajectory collection + flatten helpers)
  - `trainer/td/train.py` (value batch training + target network sync trainer)
  - TD unit coverage in `trainer_tests/test_td_*.py`
- TD Phase 2 orchestration is implemented:
  - `scripts/collect_td_self_play.py` (bridge-driven replay generation)
  - `scripts/train_td.py` (value/opponent training from replay + checkpoint cadence)
  - `scripts/run_td_loop.py` (single-command collect -> train -> eval orchestration)
    - includes collect-stage sharding via `--collect-workers` and merged replay summaries
    - includes `--cloud` fixed 8 vCPU profile (`collect-workers=6`, `eval-workers=6`)
    - sharded replay merge now appends + deletes shard JSONL files to avoid large temporary disk spikes
    - includes `--eval-first-last-checkpoints` and an `improvement` summary block (`fromStep`, `toStep`, win-rate delta, side-gap delta)
  - `td-value` policy path in eval scripts (`scripts/eval.py`, `scripts/eval_suite.py`)
  - TD replay JSONL I/O helpers in `trainer/td/io.py`
- TD Phase 3 initial integration is implemented:
  - `td-search` policy path (`trainer/policies.py`) with TD leaf evaluation and required opponent-model rollout guidance
  - script support for `td-search` args in:
    - `scripts/eval.py`
    - `scripts/eval_suite.py`
    - `scripts/generate_teacher_data.py`
    - `scripts/collect_td_self_play.py`
- TD leaf perspective correctness fix landed:
  - `td-value` / `td-search` now evaluate non-terminal leaves from active-player observations and convert to root-player perspective with explicit sign handling
  - focused unit tests added for root/active perspective conversion behavior
- Fail-fast cleanup pass implemented across Python training/eval:
  - no silent fallback action when determinization sampling fails
  - no heuristic fallback inside `td-search` opponent rollout
  - no silent winner/probability fallback in training label pipelines
  - stricter payload parsing in encoding/search helpers
  - scripts require active `.venv`, explicit policy args, and required TD checkpoints
- End-to-end loop automation validated with a small run:
  - run id `20260301-164449Z-td-loop-smoke`
  - artifacts under `artifacts/td_loops/20260301-164449Z-td-loop-smoke/`
  - produced replay, checkpoints, eval artifact, and loop summary successfully

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
- TD checkpoint iteration/evaluation cycle quality (which `td-value` / `td-search` checkpoints materially beat heuristic/search baselines).

## Remaining

- Add online/self-play replay refresh loop (not only offline replay files).
- Define checkpoint promotion/reporting conventions across TD runs.
- Improve `td-search` quality (deeper integration than current rollout-guided form, plus caching/throughput improvements).
- Define browser deployment path for learned TD checkpoint inference.

## Risks / Watch Items

- Search-only strength can plateau below strong-human target.
- Side-gap instability can hide seat bias; treat as a hard promotion risk.
- Warm-start data can encode heuristic biases; TD training must move beyond it.

_Updated: 2026-03-02._
