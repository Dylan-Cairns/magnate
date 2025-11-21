# Training Command Cookbook

This file contains copy/paste commands for the active search-first workflow.

## 1) Coarse Sweep (Stage A)

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --opponent-policy heuristic --run-label search-coarse
```

List available packs/presets:

```powershell
python -m scripts.search_teacher_sweep --list-packs
```

## 2) Confirm Top Presets (Stage B)

Replace preset ids with top rows from the coarse summary.

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --presets s03 s04 s06 --games-per-side 200 --opponent-policy heuristic --run-label search-confirm
```

## 3) Final Promotion Gate (Stage C)

```powershell
python -m scripts.search_teacher_sweep --pack coarse-v1 --presets s04 --games-per-side 1000 --opponent-policy heuristic --run-label search-final
```

## 4) Epsilon Tuning Pass

```powershell
python -m scripts.search_teacher_sweep --pack epsilon-v1 --games-per-side 200 --opponent-policy heuristic --run-label search-epsilon
```

## 5) Rollout Count Pass

```powershell
python -m scripts.search_teacher_sweep --pack rollouts-v1 --games-per-side 200 --opponent-policy heuristic --run-label search-rollouts
```

## 6) One-Off Canonical Eval

```powershell
python -m scripts.eval_suite --games-per-side 200 --seed-prefix eval-suite-search --candidate-policy search --opponent-policy heuristic --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.08 --out artifacts/evals/search_v_heur_eval_suite_400.json
```

## 7) Teacher Data + Guidance Training

Teacher data:

```powershell
python -m scripts.generate_teacher_data --games 200 --seed-prefix teacher-search --teacher-policy search --teacher-players both --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.08 --out artifacts/teacher_data/teacher_search.jsonl --summary-out artifacts/teacher_data/teacher_search.summary.json
```

Guidance train:

```powershell
python -m scripts.train_search_guidance --samples-in artifacts/teacher_data/teacher_search.jsonl --checkpoint-out artifacts/search_guidance_checkpoint.pt --epochs 12 --batch-size 128 --learning-rate 3e-4 --value-loss-coef 0.5 --hidden-dim 256
```

## 8) Baseline vs Guided A/B Pipeline

```powershell
python -m scripts.run_guidance_ab_pipeline --run-label guidance-pilot --games 200
```

The pipeline uses the same eval seed prefix for baseline and guided runs, so comparisons are paired and less noisy.
