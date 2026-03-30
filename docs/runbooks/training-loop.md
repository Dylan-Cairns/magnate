# Training Loop Runbook

Use the project `.venv` for all Python commands in this repo.

## Smoke And Primitive Commands

- `python -m scripts.smoke_trainer`
  Quick trainer smoke test.
- `python -m scripts.eval`
  Simple evaluation entrypoint.
- `python -m scripts.generate_teacher_data --games 200 --teacher-policy search --teacher-players both --out artifacts/teacher_data/teacher_search.jsonl`
  Teacher-label generation; teacher policy must emit root action probabilities.
- `python -m scripts.collect_td_self_play --games 200 --player-a-policy search --player-b-policy search --out-dir artifacts/td_replay --run-label td-replay-search`
  Replay generation for TD training primitives.
- `python -m scripts.train_td --value-replay artifacts/td_replay/<run>.value.jsonl --opponent-replay artifacts/td_replay/<run>.opponent.jsonl --steps 2000 --run-label td-v1`
  TD training primitive over replay files.

## Evaluation

- `python -m scripts.eval_suite --mode certify --games-per-side 200 --workers 2 --candidate-policy search --opponent-policy heuristic`
  Canonical side-swapped evaluation. Supports `--mode gate|certify`, deterministic worker sharding, worker thread caps, and separate td-search checkpoints per side.
- `python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse`
  Search profile sweep for teacher-data and search tuning work.

## Bootstrap Or Recalibration Loop

```powershell
python -m scripts.run_td_loop --run-label td-loop-r1 --chunks-per-loop 3 --collect-games 1200 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search --promotion-min-ci-low 0.5
```

This path supports `--collect-workers`, `--eval-workers`, `--eval-seed-start-indices`, explicit `--train-value-target-mode td0|td-lambda`, and cloud presets via `--cloud --cloud-vcpus 8|16|32`.

## Self-Play Loop

```powershell
python -m scripts.run_td_loop_selfplay --cloud --cloud-vcpus 16 --run-label td-loop-selfplay-r1 --chunks-per-loop 12 --collect-games 600 --train-steps 10000 --eval-games-per-side 200 --incumbent-eval-games-per-side 200 --progress-heartbeat-minutes 30 --eval-progress-log-minutes 30
```

The self-play loop uses mixed td-search-heavy collection, promoted opponent-pool sampling, accepted-generator gates, and final dual promotion gates versus fixed `search` plus incumbent `td-search`.

Important controls:

- `--checkpoint-selection-games-per-side` cheaply compares saved checkpoints before selecting a chunk candidate.
- `--generator-update-chunks` controls generator gate cadence.
- `--block-selection-*` settings select the best candidate in a generator block.
- `--chunk-gate-*` settings configure resumable sequential generator gates through `scripts.eval_suite --mode gate`.
- `--train-value-target-mode td-lambda --train-td-lambda 0.7` is the normal value-target mode.
- `--train-replay-window-chunks` controls replay-window width.
- `--train-replay-window-source accepted|recent` selects gate-passing chunks or recent trained chunks for replay windows.

## Resume And Promotion

- `python -m scripts.resume_td_loop_selfplay --run-id <interrupted-selfplay-run-id>`
  Resume an interrupted self-play loop from the latest fully completed chunk while preserving separate learner/generator checkpoints and replay histories.
- `python -m scripts.resume_td_loop_run`
  Bootstrap recovery helper.
- `python -m scripts.promote_td_checkpoint --key <key> --value-checkpoint <value.pt> --opponent-checkpoint <opponent.pt> --source-run-id <run-id> --set-default --add-to-opponent-pool`
  Copy a promoted value/opponent checkpoint pair into `models/td_checkpoints/<key>/` and register it in the checkpoint manifest.

## Benchmarks

- `python -m scripts.benchmark_collect_search_profiles --workers 4 --games 8`
  Benchmark td-search collect throughput across a small `search-worlds` and `search-depth` matrix.
- `python -m scripts.benchmark_selfplay_collect_setup`
  Compare single-process versus sharded self-play collection and recommend a `--collect-workers` setting for the current machine.

Use `--help` on each script for the full option surface.
