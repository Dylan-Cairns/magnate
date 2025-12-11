# Progress

## Implemented

- Deterministic TS engine and bridge runtime are in place.
- Browser app is playable with policy-agnostic bot selection.
- Default web bot is rollout-eval search.
- Canonical side-swapped eval suite is implemented (`scripts/eval_suite.py`).
- Eval suite now supports fixed-size promotion eval as canonical loop usage:
  - `scripts.eval_suite --mode certify --games-per-side 200` (400 total games)
  - explicit per-worker thread caps (`--worker-torch-threads`, `--worker-torch-interop-threads`, `--worker-blas-threads`)
  - minute-based progress cadence (`--progress-log-minutes`)
  - in-run progress artifact output (`eval.progress.json` default sibling to `--out`)
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
  - `scripts/run_td_loop.py` (single-command chunked collect/train -> single promotion eval orchestration)
    - includes collect-stage sharding via `--collect-workers` and merged replay summaries
    - includes `--cloud` preset profile via `--cloud-vcpus` (8/16/32) for worker scaling
    - train stage now supports explicit torch CPU threading controls (`--train-num-threads`, `--train-num-interop-threads`)
    - sharded replay merge appends + deletes shard JSONL files to avoid large temporary disk spikes
    - promotion decision is explicit in `loop.summary.json` (`promoted`, `reason`)
    - parent stage heartbeat and progress artifact use minutes (`--progress-heartbeat-minutes`, default `30`)
  - overnight defaults rebalanced toward training:
    - collect per chunk reduced (`1500 -> 800`)
    - train steps per chunk increased (`15000 -> 30000`)
    - promotion eval fixed to `200` games/side (`400` total)
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
- TD replay integrity hardening landed:
  - replay JSONL read/write now fail fast on invalid `done`/`nextObservation` combinations
  - value training batch ingestion now rejects malformed terminal/non-terminal transition shape
  - targeted unit tests added for strict transition invariants
- Eval/reporting seat-collision fix landed:
  - matchup/leg summaries now consistently use seat-keyed fields (`winsBySeat`, `policyBySeat`)
  - shard-merge logic validates `policyBySeat` consistency across shards
  - removed residual name-keyed `winsByPolicy` references from scripts/tests/smoke path
- TD target-mode decision is now explicit:
  - one-step TD(0) remains default in `trainer.td.train`
  - `scripts.train_td` summary records `valueTargetMode`
- Sequence-aware replay + TD(lambda) landed:
  - value replay rows now carry `episodeId` + `timestep`
  - `scripts.train_td` supports `--value-target-mode td-lambda --td-lambda <0..1>`
  - TD(lambda) training uses contiguous per-episode/per-player sequence indexing with fail-fast invariants
- Single-eval promotion protocol landed:
  - latest checkpoint only in `scripts.run_td_loop`
  - one fixed-size side-swapped eval for promotion decision
  - threshold checks (`minWinRate`, `maxSideGap`, `minCiLow`) are applied directly on the single eval artifact
- Promotion/eval reliability fixes landed:
  - `scripts.eval_suite --mode gate` no longer crashes due to pre-init `history` usage
  - gate smoke coverage added (`trainer_tests/test_eval_suite_gate.py`) to catch regressions
- CLI/docs parity hardening landed:
  - `README.md` and `docs/TRAINING_COMMANDS.md` examples aligned to current `scripts.run_td_loop` flags (`--chunks-per-loop`, `--eval-*`)
  - arg-surface smoke coverage added for canonical loop commands (`trainer_tests/test_run_td_loop_args.py`)
- Teacher label-generation guardrail landed:
  - `scripts.generate_teacher_data` rejects teacher policies that do not emit root action probabilities at validation time
  - validation coverage added (`trainer_tests/test_generate_teacher_data_script.py`)
- Runtime/platform guardrails improved:
  - `scripts.search_teacher_sweep` now resolves default `--python-bin` cross-platform (`sys.executable`, then Unix/Windows `.venv` paths)
  - `scripts.train_td` now fail-fast enforces Python 3.11+ and active `.venv`
- Promotion evidence defaults tightened:
  - `scripts.run_td_loop` default `--promotion-min-ci-low` raised from `0.0` to `0.5`
  - loop summary now records explicit replay regime metadata (`chunk-local`)
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
- Continue tuning fixed-eval promotion thresholds and chunk cadence based on observed variance/runtime.
- Improve `td-search` quality (deeper integration than current rollout-guided form, plus caching/throughput improvements).
- Define browser deployment path for learned TD checkpoint inference.

## Risks / Watch Items

- Search-only strength can plateau below strong-human target.
- Side-gap instability can hide seat bias; treat as a hard promotion risk.
- Warm-start data can encode heuristic biases; TD training must move beyond it.

_Updated: 2026-03-05._
