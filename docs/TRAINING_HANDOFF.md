# Training Handoff (2026-02-27)

Use this document as the single restart context for training work in a new chat/session.

## 1. Goal

Build a bot that reliably beats the heuristic baseline while remaining practical for browser play.

Current direction:

- Keep existing BC/REINFORCE/PPO pipeline intact.
- Use deterministic search as a stronger teacher signal.
- Move toward search-to-student distillation once teacher config is locked.

## 2. Important Constraints

- TypeScript engine is canonical for rules.
- Python consumes TS via bridge; no Python-side rule duplication.
- Determinism matters for training, eval, and reproducibility.
- Avoid unnecessary defensive branches and optional dependency fallbacks in this project.

## 3. Pipeline Status

### Implemented and stable

- BC warm-start (`scripts.train`)
- REINFORCE fine-tune (`scripts.finetune`)
- PPO training (`scripts.train_ppo`, `scripts.train_ppo_queue`)
- Eval and benchmark harnesses (`scripts.eval`, `scripts.benchmark`, queue helpers)
- Additive search policy in eval/benchmark (`--player-*-policy search`, `--candidate-policy search`)

### Not yet implemented

- Search-to-student distillation training loop
- Formal promotion gate automation (currently manual review of benchmark/eval artifacts)

## 4. Measured Results Snapshot

### PPO / BC benchmark summary (200-game holdout, score = `0.7*heur + 0.3*random`)

Source artifacts: `artifacts/benchmarks/*.json`

- `ppo_tuneC_seed11_preview.json`: score `0.6055`, heuristic `0.505`, random `0.84`
- `ppo_tuneB_seed7.json`: score `0.595`, heuristic `0.49`, random `0.84`
- `ppo_tuneB_seed8.json`: score `0.594`, heuristic `0.495`, random `0.825`
- `20260222-001540Z-ppo.json` (run_001): score `0.5875`, heuristic `0.505`, random `0.78`
- `20260222-001914Z-bc.json`: score `0.485`, heuristic `0.395`, random `0.695`

Interpretation:

- PPO is clearly stronger than BC.
- PPO advantage vs heuristic is modest in 200-game holdout benchmarks.

### PPO vs heuristic (high-sample 2000-game evals from recorded CLI runs)

These are the better indicator of actual head-to-head strength than 200-game benchmarks.

- `ppo_checkpoint_tuneB_seed7.pt`: `1082/2000` wins vs heuristic => `54.1%`
- `ppo_checkpoint_tuneB_seed8.pt`: `1024/2000` wins vs heuristic => `51.2%`
- `ppo_checkpoint_run_001.pt`: `1028/2000` wins vs heuristic => `51.4%`
- Holdout-prefix recheck (`bench-heuristic-holdout`) for `tuneB_seed7`: `1074/2000` => `53.7%`

Interpretation:

- Best PPO checkpoints are above `50%` vs heuristic on large samples.
- Expectation should be "low-to-mid 50s vs heuristic" for current PPO path, not sub-50.

### Search teacher sweep vs heuristic (700 games each)

Source artifacts: `artifacts/evals/search_teacher_t*_v_heur_700.json`

- `t1` (`worlds=2 depth=8 maxRoot=4 eps=0.12`): `479-220-1` => `68.4%` win
- `t2` (`worlds=4 depth=12 maxRoot=6 eps=0.10`): `524-176-0` => `74.9%` win
- `t3` (`worlds=6 depth=14 maxRoot=6 eps=0.08`): `553-146-1` => `79.0%` win

Interpretation:

- Search policy is already substantially stronger than heuristic.
- Cost rises steeply with search depth/world count.

## 5. In-Flight Run

Currently running (user started):

```powershell
python -m scripts.eval --games 700 --seed-prefix teacher-t4-v-heur --player-a-policy search --player-b-policy heuristic --search-worlds 8 --search-rollouts 1 --search-depth 16 --search-max-root-actions 6 --search-rollout-epsilon 0.06 --out artifacts/evals/search_teacher_t4_v_heur_700.json
```

First checkpoint observed:

- `25/700`, win rate `0.64`, elapsed `30.0` min (too early for conclusion)

## 6. Decision Logic After t4 Finishes

Read result artifact:

- `artifacts/evals/search_teacher_t4_v_heur_700.json`

Then apply:

1. If `t4` improves materially over `t3` (>= ~2% absolute win-rate gain), keep deeper teacher config candidate list.
2. If `t4` is flat or worse vs `t3`, lock `t3` as teacher and stop deepening search.
3. In either case, next major step is distillation (teacher -> fast student), not more blind PPO-only sweeps.

Rationale:

- Search strength is already high; biggest remaining gap is inference cost and deployment practicality.
- Distillation targets both strength retention and UI latency.

## 7. Immediate Next Steps (Ordered)

1. Finish/inspect `t4`.
2. Lock teacher config (`t3` or `t4`) based on objective comparison.
3. Implement teacher-data generation script:
   - status: implemented via `python -m scripts.generate_teacher_data`
   - next use: run it with locked teacher config to generate production dataset artifacts
4. Implement student supervised training stage:
   - consume teacher dataset
   - train candidate-action model (same observation/action feature system)
5. Evaluate student against heuristic and PPO baselines using `scripts.eval` + `scripts.benchmark`.
6. Only then decide whether PPO opponent-pool fine-tuning is still needed.

## 8. Useful Commands

### Compare teacher eval artifacts quickly

```powershell
Get-ChildItem artifacts\evals\search_teacher_t*_v_heur_700.json | ForEach-Object { $j = Get-Content $_.FullName -Raw | ConvertFrom-Json; $g = [double]$j.results.games; $w = [double]$j.results.winners.PlayerA; [PSCustomObject]@{ file=$_.Name; winRate=[math]::Round($w/$g,4); winners=($j.results.winners | ConvertTo-Json -Compress) } } | Sort-Object file | Format-Table -AutoSize
```

### Compare benchmark artifacts quickly

```powershell
Get-ChildItem artifacts\benchmarks\*.json | ForEach-Object { $j = Get-Content $_.FullName -Raw | ConvertFrom-Json; if ($j.results.selectionScore -ne $null) { [PSCustomObject]@{ file=$_.Name; score=[double]$j.results.selectionScore; heuristic=[double]$j.results.heuristicWinRate; random=[double]$j.results.randomWinRate } } } | Sort-Object score -Descending | Select-Object -First 15 | Format-Table -AutoSize
```

### Generate teacher dataset (search labels)

```powershell
python -m scripts.generate_teacher_data --games 200 --seed-prefix teacher-distill-v1 --teacher-policy search --teacher-players both --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.08 --out artifacts/teacher_data/teacher_distill_v1.jsonl --summary-out artifacts/teacher_data/teacher_distill_v1.summary.json
```

## 9. Artifact Conventions

- Checkpoints: `artifacts/ppo_checkpoint_<label>_seed<seed>.pt`
- Benchmarks: `artifacts/benchmarks/<label>.json`
- Evals: `artifacts/evals/<label>.json`
- Seed prefixes should include run label + seed for reproducibility.

## 10. Known Risks

- Search policy is expensive at inference time (not suitable as-is for responsive browser play).
- PPO seed variance is real; single short benchmarks can mislead.
- Over-optimizing teacher depth may waste compute if distillation quality becomes the actual bottleneck.

---

If resuming in a fresh chat, provide this file plus the latest `t4` artifact (if finished), then proceed from section 6.
