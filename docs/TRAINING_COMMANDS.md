# Training Command Cookbook

This file contains copy/paste commands for the current cleanup state:

- search remains as the warm-start baseline,
- PPO/MCTS/guidance paths are removed,
- TD/Keldon Phase 2 orchestration is implemented.

## Fail-Fast Rules

- Run commands from the project `.venv`; scripts exit if virtualenv is not active.
- Policy roles must be explicit (`--candidate-policy`, `--opponent-policy`, `--player-a-policy`, `--player-b-policy`, `--teacher-policy`).
- `td-search` now requires both `--td-search-value-checkpoint` and `--td-search-opponent-checkpoint`.
- Search/TD training paths no longer silently fall back to heuristic action selection.

## 1) Coarse Search Sweep (Warm-Start Teacher Tuning)

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse
```

List available packs/presets:

```powershell
python -m scripts.search_teacher_sweep --list-packs
```

## 2) Confirm Top Presets

Replace preset ids with top rows from the coarse summary.

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --presets s03 s04 s06 --games-per-side 200 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-confirm
```

## 3) Final Promotion Gate for Search Baseline

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --presets s04 --games-per-side 1000 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-final
```

## 4) One-Off Canonical Eval

```powershell
python -m scripts.eval_suite --games-per-side 200 --workers 1 --seed-prefix eval-suite-search --candidate-policy search --opponent-policy heuristic --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.04 --out artifacts/evals/search_v_heur_eval_suite_400.json
```

## 5) Warm-Start Teacher Data Generation

```powershell
python -m scripts.generate_teacher_data --games 200 --seed-prefix teacher-search --teacher-policy search --teacher-players both --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.04 --out artifacts/teacher_data/teacher_search.jsonl --summary-out artifacts/teacher_data/teacher_search.summary.json
```

## 6) Smoke Check

```powershell
python -m scripts.smoke_trainer
```

## 7) TD Replay Collection (Phase 2)

```powershell
python -m scripts.collect_td_self_play --games 200 --seed-prefix td-replay --player-a-policy search --player-b-policy search --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.04 --out-dir artifacts/td_replay --run-label td-replay-search
```

## 8) TD Training (Phase 2)

Replace `<stamp-run-label>` with the replay artifact stem from step 7.

```powershell
python -m scripts.train_td --value-replay artifacts/td_replay/<stamp-run-label>.value.jsonl --opponent-replay artifacts/td_replay/<stamp-run-label>.opponent.jsonl --steps 2000 --value-batch-size 128 --opponent-batch-size 64 --target-sync-interval 200 --save-every-steps 200 --progress-every-steps 20 --out-dir artifacts/td_checkpoints --run-label td-v1
```

## 9) TD Checkpoint Eval (Phase 2)

Replace checkpoint path with a saved value checkpoint from step 8.

```powershell
python -m scripts.eval_suite --games-per-side 200 --workers 1 --seed-prefix td-eval-v1 --candidate-policy td-value --opponent-policy heuristic --td-value-checkpoint artifacts/td_checkpoints/<run-dir>/value-step-0002000.pt --td-worlds 8 --out artifacts/evals/td_value_v_heur_400.json
```

## 10) TD-Search Eval (Phase 3)

Use both value and opponent checkpoints from a TD run.

```powershell
python -m scripts.eval_suite --games-per-side 200 --workers 1 --seed-prefix td-search-eval-v1 --candidate-policy td-search --opponent-policy heuristic --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.04 --td-search-value-checkpoint artifacts/td_checkpoints/<run-dir>/value-step-0002000.pt --td-search-opponent-checkpoint artifacts/td_checkpoints/<run-dir>/opponent-step-0002000.pt --td-search-opponent-temperature 1.0 --out artifacts/evals/td_search_v_heur_400.json
```

## 11) One-Command TD Loop (Collect -> Train -> Eval)

Runs all three stages end-to-end and writes one loop summary artifact.

```powershell
python -m scripts.run_td_loop --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search
```

Cloud 8 vCPU profile (single flag):

```powershell
python -m scripts.run_td_loop --cloud --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search --collect-search-worlds 6 --collect-search-depth 14 --collect-search-max-root-actions 6 --eval-search-worlds 6 --eval-search-depth 14 --eval-search-max-root-actions 6
```

Cloud run with built-in begin/end improvement report (no separate checkpoint sweep):

```powershell
python -m scripts.run_td_loop --cloud --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --train-save-every-steps 200 --eval-games-per-side 200 --eval-opponent-policy search --eval-first-last-checkpoints --collect-search-worlds 6 --collect-search-depth 14 --collect-search-max-root-actions 6 --eval-search-worlds 6 --eval-search-depth 14 --eval-search-max-root-actions 6
```

Quick validation-sized run:

```powershell
python -m scripts.run_td_loop --run-label td-loop-smoke --collect-games 12 --collect-search-worlds 2 --collect-search-depth 8 --collect-search-max-root-actions 4 --train-steps 30 --train-save-every-steps 15 --train-hidden-dim 64 --train-value-batch-size 32 --train-opponent-batch-size 16 --eval-games-per-side 20 --eval-opponent-policy search --eval-search-worlds 2 --eval-search-depth 8 --eval-search-max-root-actions 4
```
