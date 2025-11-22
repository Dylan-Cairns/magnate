# Training Handoff (2026-03-01)

Use this file as restart context for Python training work.

## Goal

Keep the repo focused on one direction:

1. retain rollout search as a warm-start signal generator,
2. remove all legacy PPO/MCTS/guidance complexity,
3. iterate on the TD-Gammon / Keldon-like training loop with `td-search` integration now available.

## Canonical Approach (Current)

1. Tune and evaluate rollout search with side-swapped paired seeds.
2. Use search teacher to generate warm-start decision data.
3. Run TD replay collection + TD training + TD checkpoint eval cycles.
4. Evaluate `td-search` checkpoints (TD-guided search) against heuristic/search baselines.

## Fail-Fast Policy (Locked)

- No silent fallback action selection in training/eval pipelines.
- No silent winner-label fallbacks (`Draw`) when terminal metadata is malformed.
- No silent probability fallbacks (for example one-hot substitution) when policy distributions are invalid/missing.
- Scripts require active `.venv` and explicit policy flags.
- `td-search` requires both value and opponent checkpoints.

## Removed in Cleanup

- PPO training stack and queue runner.
- MCTS policy stack and all MCTS CLI options.
- Guidance training/checkpoint pipeline.
- Related tests and documentation references.

## Canonical Scripts (Now)

- Matchup eval (single-seat): `scripts.eval`
- Side-swapped paired-seed eval: `scripts.eval_suite`
- Search sweep runner (eval_suite-based): `scripts.search_teacher_sweep`
- Teacher data collection (warm-start labels): `scripts.generate_teacher_data`
- TD replay collection: `scripts.collect_td_self_play`
- TD training loop: `scripts.train_td`
- Full TD loop orchestration: `scripts.run_td_loop`
- TD-guided search policy path: `td-search` through `scripts.eval` / `scripts.eval_suite`
- Smoke check: `scripts.smoke_trainer`

## Default Search Baseline

- `worlds=6`
- `rollouts=1`
- `depth=14`
- `max_root_actions=6`
- `rollout_epsilon=0.04`

## Promotion Protocol for Search Baseline

Stage A (coarse):

- Sweep coarse pack with `games_per_side=60` (120 total per preset).
- Rank by win rate, then side gap, then CI width.
- Keep top 2-3 presets.

Stage B (confirm):

- Re-run top presets with `games_per_side=200` (400 total).
- Require low side gap before promoting.

Stage C (final):

- Final gate at `games_per_side=1000` (2000 total).
- Use this as search baseline evidence for TD warm-start.

## Command Templates

Coarse sweep:

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse
```

Confirm top presets:

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --presets s03 s04 s06 --games-per-side 200 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-confirm
```

Final gate:

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --presets s04 --games-per-side 1000 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-final
```

Teacher-data generation:

```powershell
python -m scripts.generate_teacher_data --games 200 --seed-prefix teacher-search --teacher-policy search --teacher-players both --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.04 --out artifacts/teacher_data/teacher_search.jsonl --summary-out artifacts/teacher_data/teacher_search.summary.json
```

TD replay generation:

```powershell
python -m scripts.collect_td_self_play --games 200 --seed-prefix td-replay --player-a-policy search --player-b-policy search --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.04 --out-dir artifacts/td_replay --run-label td-replay-search
```

Full TD loop orchestration (8 vCPU cloud profile):

```powershell
python -m scripts.run_td_loop --cloud --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search
```

Built-in begin/end improvement comparison:

```powershell
python -m scripts.run_td_loop --cloud --run-label td-loop-r1 --collect-games 2000 --train-steps 20000 --train-save-every-steps 200 --eval-games-per-side 200 --eval-opponent-policy search --eval-first-last-checkpoints
```

TD training:

```powershell
python -m scripts.train_td --value-replay artifacts/td_replay/<stamp-run>.value.jsonl --opponent-replay artifacts/td_replay/<stamp-run>.opponent.jsonl --steps 2000 --save-every-steps 200 --target-sync-interval 200 --out-dir artifacts/td_checkpoints --run-label td-v1
```

TD checkpoint eval:

```powershell
python -m scripts.eval_suite --games-per-side 200 --workers 1 --seed-prefix td-eval-v1 --candidate-policy td-value --opponent-policy heuristic --td-value-checkpoint artifacts/td_checkpoints/<run-dir>/value-step-0002000.pt --td-worlds 8 --out artifacts/evals/td_value_v_heur_400.json
```

TD-search checkpoint eval:

```powershell
python -m scripts.eval_suite --games-per-side 200 --workers 1 --seed-prefix td-search-eval-v1 --candidate-policy td-search --opponent-policy heuristic --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.04 --td-search-value-checkpoint artifacts/td_checkpoints/<run-dir>/value-step-0002000.pt --td-search-opponent-checkpoint artifacts/td_checkpoints/<run-dir>/opponent-step-0002000.pt --out artifacts/evals/td_search_v_heur_400.json
```

## Artifacts

- Evals: `artifacts/evals/*.json`
- Sweeps:
  - manifest: `artifacts/sweeps/<run-id>-manifest.json`
  - summary: `artifacts/sweeps/<run-id>-summary.md`
- Teacher data: `artifacts/teacher_data/*.jsonl` + `*.summary.json`

## Risks / Watch Items

- Search quality and latency trade off sharply with `worlds * rollouts * depth`.
- Side-gap instability means determinization bias or seat dependence; treat as a promotion blocker.
- Search warm-start data can encode heuristic bias; TD training must eventually self-correct beyond it.

## Next Session Checklist

1. Read this file and `docs/TRAINING_PLAN_SEARCH_FIRST.md`.
2. Confirm a promoted search baseline via `scripts.search_teacher_sweep`.
3. Generate a clean warm-start dataset with `scripts.generate_teacher_data`.
4. Run Phase 2/3 TD loops (`collect_td_self_play` -> `train_td` -> `eval_suite --candidate-policy td-value|td-search`) and rank checkpoints.
