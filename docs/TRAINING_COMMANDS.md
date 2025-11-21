# Training Command Cookbook

This file contains copy/paste commands for the current cleanup state:

- search remains as the warm-start baseline,
- PPO/MCTS/guidance paths are removed,
- TD/Keldon training loop is the next implementation step.

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
