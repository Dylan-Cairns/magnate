# Training Handoff (2026-03-01)

Use this file as restart context for Python training work.

## Goal

Keep the repo focused on one direction:

1. retain rollout search as a warm-start signal generator,
2. remove all legacy PPO/MCTS/guidance complexity,
3. implement a TD-Gammon / Keldon-like training loop next.

## Canonical Approach (Current)

1. Tune and evaluate rollout search with side-swapped paired seeds.
2. Use search teacher to generate warm-start decision data.
3. Start TD value + opponent-model implementation on the cleaned stack.

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
4. Begin TD value/opponent model implementation on the cleaned trainer stack.
