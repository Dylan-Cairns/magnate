# Training Loop Runbook

Use the project `.venv` for all Python commands in this repo.

## Loop Commands

Bootstrap or recalibration with `scripts.run_td_loop`:

```powershell
python -m scripts.run_td_loop --run-label td-loop-r1 --chunks-per-loop 3 --collect-games 1200 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search --promotion-min-ci-low 0.5
```

Supports `--collect-workers`, `--eval-workers`, `--eval-seed-start-indices`,
explicit `--train-value-target-mode td0|td-lambda`, and cloud presets via
`--cloud --cloud-vcpus 8|16|32`.

Forward self-play with `scripts.run_td_loop_selfplay`:

```powershell
python -m scripts.run_td_loop_selfplay --cloud --cloud-vcpus 16 --run-label td-loop-selfplay-r1 --chunks-per-loop 12 --collect-games 600 --train-steps 10000 --eval-games-per-side 200 --incumbent-eval-games-per-side 200 --progress-heartbeat-minutes 30 --eval-progress-log-minutes 30
```

The self-play loop uses mixed td-search-heavy collection, promoted
opponent-pool sampling, accepted-generator gates, and final dual promotion
gates versus fixed `search` plus incumbent `td-search`.

Important controls:

- `--checkpoint-selection-games-per-side` compares saved checkpoints before
  selecting a chunk candidate.
- `--generator-update-chunks` controls generator gate cadence; non-boundary
  chunks defer the generator gate while learner training continues.
- `--block-selection-*` selects the best candidate in a generator block.
- `--chunk-gate-*` configures resumable sequential generator gates through
  `scripts.eval_suite --mode gate`.
- `--train-value-target-mode td-lambda --train-td-lambda 0.7` is the normal
  value-target mode.
- `--train-replay-window-chunks` controls replay-window width.
- `--train-replay-window-source accepted|recent` selects gate-passing chunks or
  recent trained chunks for replay windows.

Windows laptop wrappers (`run_td_loop_bootstrap_laptop.ps1`,
`run_td_loop_selfplay_laptop.ps1`, `resume_td_loop_selfplay_laptop.ps1`) set
laptop-safe worker/thread settings; see `windows-local.md`.

## Resume And Promotion

- `python -m scripts.resume_td_loop_selfplay --run-id <interrupted-run-id>`
  resumes a self-play loop from the latest fully completed chunk while
  preserving learner/generator checkpoints and replay histories.
- `python -m scripts.resume_td_loop_run` is the bootstrap recovery helper.
- `python -m scripts.promote_td_checkpoint --key <key> --value-checkpoint <value.pt> --opponent-checkpoint <opponent.pt> --source-run-id <run-id> --set-default --add-to-opponent-pool`
  copies a promoted pair into `models/td_checkpoints/<key>/` and registers it in
  the checkpoint manifest.

Resume state comes from `chunks/chunk-XXX/chunk.summary.json` (per-chunk
durability) and `blocks/block-XXX/block.summary.json` (generator decisions).
Resume requires the current chunk-summary schema and restores accepted and
recent replay histories separately.

## Smoke And Primitive Commands

- `python -m scripts.smoke_trainer`: quick trainer smoke test.
- `python -m scripts.eval`: simple evaluation entrypoint.
- `python -m scripts.generate_teacher_data --games 200 --teacher-policy search --teacher-players both --out artifacts/teacher_data/teacher_search.jsonl`:
  teacher-label generation; the teacher policy must emit root action
  probabilities.
- `python -m scripts.collect_td_self_play --games 200 --player-a-policy search --player-b-policy search --out-dir artifacts/td_replay --run-label td-replay-search`:
  replay generation for TD training primitives.
- `python -m scripts.train_td --value-replay <value.jsonl> --opponent-replay <opponent.jsonl> --steps 2000 --run-label td-v1`:
  TD training over replay files.

`scripts.train_td` also accepts `--value-replay-list` and
`--opponent-replay-list`, each pointing to a UTF-8 file with one replay path per
line, avoiding platform command-length limits for large shard sets. Replay lists
that span collection runs must pass `--replay-key-mode run-qualified-canonical-v1`.

## Evaluation

- `python -m scripts.eval_suite --mode certify --games-per-side 200 --workers 2 --candidate-policy search --opponent-policy heuristic`:
  canonical side-swapped evaluation. Supports `--mode gate|certify`,
  deterministic worker sharding, worker thread caps, and separate td-search
  checkpoints per side.
- `python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse`:
  search profile sweep for teacher-data and search tuning work.
- `python -m scripts.evaluate_td_replay_holdout --help`: evaluate a final
  value/opponent pair on the complete replay holdout.

## Benchmarks

- `python -m scripts.benchmark_collect_search_profiles --workers 4 --games 8`:
  td-search collect throughput across a `search-worlds`/`search-depth` matrix.
- `python -m scripts.benchmark_selfplay_collect_setup`: compares
  single-process and sharded self-play collection and recommends a
  `--collect-workers` setting for the current machine.

## Frozen Experiment Launchers

The extra-data continuation, district-symmetry pilot, and opponent-orbit pilot
launchers remain under `scripts/` with frozen manifests under
`configs/td-training/`. They are completed one-off experiments; outcomes and
promotion blocks are summarized in `memoryBank/activeContext.md` (extra-data
continuation) and `docs/design/district-symmetry.md` (district symmetry), with
detailed artifacts ignored under `artifacts/`. Do not re-run them as part of
normal loop work.

Use `--help` on each script for the full option surface.
