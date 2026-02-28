# Training Handoff (2026-03-01)

Use this file as the restart context for training work in a new session.

## Goal

Ship a search-first bot pipeline that clearly outperforms heuristic, then distill/guidance for practical deployment speed.

## Canonical Approach

1. Tune determinized rollout search first (teacher quality).
2. Evaluate with side-swapped paired-seed protocol (`scripts.eval_suite`).
3. Train guidance from teacher data (`scripts.generate_teacher_data` + `scripts.train_search_guidance`).
4. Re-evaluate search/MCTS with matched paired seeds.
5. Promote only when stability gates pass at larger sample sizes.

## What Was Removed

- Browser PPO export path (`scripts/export_ppo_browser_checkpoint.py`).
- BC/REINFORCE training entrypoints (`scripts/train.py`, `scripts/finetune.py`).
- Legacy benchmark/queue pipeline (`scripts/benchmark.py`, `scripts/benchmark_queue.py`).
- Legacy BC/reinforcement modules and tests.
- Legacy named sweep presets (`t1..t8`) and non-canonical two-leg sweep flow.

## Canonical Scripts

- Matchup eval (single-seat): `scripts.eval`
- Side-swapped paired-seed eval: `scripts.eval_suite`
- Search sweep runner (eval_suite-based): `scripts.search_teacher_sweep`
- Teacher data collection: `scripts.generate_teacher_data`
- Guidance training: `scripts.train_search_guidance`
- End-to-end A/B pipeline: `scripts.run_guidance_ab_pipeline`

## Default Search Baseline (T3-style)

- `worlds=6`
- `rollouts=1`
- `depth=14`
- `max_root_actions=6`
- `rollout_epsilon=0.08`

These are the defaults in current search-oriented scripts unless explicitly overridden.

## Promotion Protocol

Stage A (coarse):

- Sweep coarse pack with `games_per_side=60` (120 total per preset).
- Rank by win rate, then side gap, then CI width.
- Keep top 2-3 presets.

Stage B (confirm):

- Re-run top presets with `games_per_side=200` (400 total).
- Require low side gap before promoting.

Stage C (final):

- Final gate at `games_per_side=1000` (2000 total).
- Use this result for promotion decisions.

## Command Templates

Coarse sweep:

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --opponent-policy heuristic --run-label search-coarse
```

Confirm top presets:

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --presets s03 s04 s06 --games-per-side 200 --opponent-policy heuristic --run-label search-confirm
```

Final gate:

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --presets s04 --games-per-side 1000 --opponent-policy heuristic --run-label search-final
```

Guidance pipeline:

```powershell
python -m scripts.run_guidance_ab_pipeline --run-label guidance-pilot --games 200
```

The A/B pipeline now uses a shared eval seed prefix for baseline and guided eval_suite runs.

## Artifacts

- Evals: `artifacts/evals/*.json`
- Sweeps:
  - manifest: `artifacts/sweeps/<run-id>-manifest.json`
  - summary: `artifacts/sweeps/<run-id>-summary.md`
- Teacher data: `artifacts/teacher_data/*.jsonl` + `*.summary.json`
- Guidance checkpoints: `artifacts/search_guidance_*.pt`
- Pipeline manifests: `artifacts/*-pipeline-manifest.json`

## Risks / Watch Items

- Search quality and latency trade off sharply with `worlds * rollouts * depth`.
- Side-gap instability means determinization bias or seat dependence; treat as a promotion blocker.
- Guidance can help strength while increasing per-decision cost if overused.

## Next Session Checklist

1. Read this file and `docs/TRAINING_PLAN_SEARCH_FIRST.md`.
2. Start with `scripts.search_teacher_sweep --pack coarse-v1`.
3. Promote only through 120 -> 400 -> 2000 total-game stages.
4. Keep documentation updated when promotion criteria or defaults change.
