# Training Handoff (2026-02-28)

Use this document as the single restart context for training work in a new chat/session.

## 1. Goal

Build a bot that is significantly stronger than the current heuristic baseline, with practical browser play options.

Current direction:

- Keep existing BC/REINFORCE/PPO pipeline intact.
- Keep rollout-eval search as a strong baseline teacher.
- Evaluate newly added MCTS for a higher-strength policy tier.
- Use learned guidance (policy prior + value + opponent model) inside search/MCTS to raise teacher strength.
- Distill stronger search/MCTS behavior into fast student policies after teacher strength is locked.

## 2. Important Constraints

- TypeScript engine is canonical for rules.
- Python consumes TS via bridge; no Python-side rule duplication.
- Determinism matters for training, eval, and reproducibility.
- No silent policy fallback; active bot must be explicit.

## 3. Pipeline Status

### Implemented and stable

- BC warm-start (`scripts.train`)
- REINFORCE fine-tune (`scripts.finetune`)
- PPO training (`scripts.train_ppo`, `scripts.train_ppo_queue`)
- Eval and benchmark harnesses (`scripts.eval`, `scripts.benchmark`, queue helpers)
- Additive rollout search policy in eval/benchmark (`--player-*-policy search`, `--candidate-policy search`)
- Additive MCTS policy in eval/benchmark/teacher-data:
  - `--player-*-policy mcts`
  - `--candidate-policy mcts`
  - `--teacher-policy mcts`
  - upgraded implementation details:
    - progressive root widening (instead of permanent hard top-K root filtering)
    - cached state-transition stepping in tree search
    - stronger non-terminal value proxy aligned to district/tiebreak pressure
- Guidance-enabled search/MCTS paths are now implemented:
  - optional PPO-format guidance checkpoints for:
    - search root priors
    - search rollout opponent action modeling
    - MCTS priors
    - MCTS leaf value cutoffs
  - integrated CLI support in:
    - `scripts.eval`
    - `scripts.benchmark`
    - `scripts.generate_teacher_data`
- Guidance checkpoint training from teacher data:
  - `scripts/train_search_guidance`
- Browser bot profiles:
  - `Champion PPO` (default)
  - `Rollout Eval Search` (T3 settings)
  - `Random`

### Not yet implemented

- Browser MCTS profile integration (MCTS is currently Python-tooling only)
- Full search/MCTS-to-student distillation training loop with promotion automation
- Formal promotion gate automation (currently manual artifact review)

## 4. Measured Results Snapshot

### PPO vs heuristic (high-sample)

- `ppo_checkpoint_tuneB_seed7.pt`: `1082/2000` => `54.1%`
- `ppo_checkpoint_tuneB_seed8.pt`: `1024/2000` => `51.2%`
- `ppo_checkpoint_run_001.pt`: `1028/2000` => `51.4%`

Interpretation:

- PPO beats heuristic, but only modestly.

### Rollout search vs heuristic (700-game teacher sweeps)

Source artifacts: `artifacts/evals/search_teacher_t*_v_heur_700.json`

- `t1` (`worlds=2 depth=8 maxRoot=4 eps=0.12`): `68.4%`
- `t2` (`worlds=4 depth=12 maxRoot=6 eps=0.10`): `74.9%`
- `t3` (`worlds=6 depth=14 maxRoot=6 eps=0.08`): `79.0%`

Interpretation:

- Rollout search is much stronger than heuristic and stronger than PPO.

### MCTS current checkpoint

- Smoke eval completed:
  - `artifacts/evals/mcts_v_heur_25.json`
  - `19/25` wins => `76.0%` (`mcts` vs `heuristic`)

Interpretation:

- Promising, but 25 games is too small for strong claims.

## 5. Current Experiment State

Latest request was to run three identical 25-game MCTS evaluations with different seed prefixes to reduce luck effects.

Chained command:

```powershell
python -m scripts.eval --games 25 --seed-prefix eval-mcts-a --player-a-policy mcts --player-b-policy heuristic --mcts-worlds 6 --mcts-simulations 192 --mcts-depth 20 --mcts-max-root-actions 10 --mcts-c-puct 1.15 --out artifacts/evals/mcts_a_25.json && python -m scripts.eval --games 25 --seed-prefix eval-mcts-b --player-a-policy mcts --player-b-policy heuristic --mcts-worlds 6 --mcts-simulations 192 --mcts-depth 20 --mcts-max-root-actions 10 --mcts-c-puct 1.15 --out artifacts/evals/mcts_b_25.json && python -m scripts.eval --games 25 --seed-prefix eval-mcts-c --player-a-policy mcts --player-b-policy heuristic --mcts-worlds 6 --mcts-simulations 192 --mcts-depth 20 --mcts-max-root-actions 10 --mcts-c-puct 1.15 --out artifacts/evals/mcts_c_25.json
```

## 6. Decision Logic For MCTS

Short-screen gate (25-game triad):

1. Check all three artifacts (`mcts_a_25.json`, `mcts_b_25.json`, `mcts_c_25.json`).
2. If each run is at least `23/25` (92%), promote this config to larger-sample evaluation.
3. If not, retune MCTS config before scaling games.

Scale-up gate:

1. Run best config for `200` games.
2. If still strong, run `700` then `2000` games.
3. Use 2000-game results as promotion evidence, not 25-game results.

## 7. Immediate Next Steps (Ordered)

1. Train a first guidance checkpoint from strongest available teacher data.
2. Run side-swapped holdouts for:
   - search (no guidance vs guidance)
   - MCTS (no guidance vs guidance)
3. Promote only after `200+`, then `700+`, then `2000` game stability checks.
4. Lock strongest teacher + guidance config and generate larger teacher datasets.
5. Proceed to student distillation planning once teacher dominance is confirmed.

## 8. Useful Commands

### Summarize recent MCTS eval artifacts

```powershell
Get-ChildItem artifacts\evals\mcts*_25.json | ForEach-Object { $j = Get-Content $_.FullName -Raw | ConvertFrom-Json; $g = [double]$j.games; $w = [double]$j.winners.PlayerA; [PSCustomObject]@{ file=$_.Name; winRate=[math]::Round($w/$g,4); winners=($j.winners | ConvertTo-Json -Compress) } } | Sort-Object file | Format-Table -AutoSize
```

### 200-game MCTS evaluation template

```powershell
python -m scripts.eval --games 200 --seed-prefix eval-mcts-v-heur-200 --player-a-policy mcts --player-b-policy heuristic --mcts-worlds 6 --mcts-simulations 192 --mcts-depth 20 --mcts-max-root-actions 10 --mcts-c-puct 1.15 --out artifacts/evals/mcts_v_heur_200.json
```

### Rollout-search teacher-data generation template

```powershell
python -m scripts.generate_teacher_data --games 200 --seed-prefix teacher-distill-v1 --teacher-policy search --teacher-players both --search-worlds 6 --search-rollouts 1 --search-depth 14 --search-max-root-actions 6 --search-rollout-epsilon 0.08 --out artifacts/teacher_data/teacher_distill_v1.jsonl --summary-out artifacts/teacher_data/teacher_distill_v1.summary.json
```

### Train guidance checkpoint from teacher data

```powershell
python -m scripts.train_search_guidance --samples-in artifacts/teacher_data/teacher_distill_v1.jsonl --checkpoint-out artifacts/search_guidance_checkpoint_v1.pt --epochs 12 --batch-size 128 --learning-rate 3e-4 --value-loss-coef 0.5 --hidden-dim 256
```

### Evaluate MCTS with guidance checkpoint

```powershell
python -m scripts.eval --games 200 --seed-prefix eval-mcts-guidance-v-heur-200 --player-a-policy mcts --player-b-policy heuristic --mcts-worlds 6 --mcts-simulations 192 --mcts-depth 20 --mcts-max-root-actions 10 --mcts-c-puct 1.15 --mcts-guidance-checkpoint artifacts/search_guidance_checkpoint_v1.pt --guidance-temperature 1.0 --out artifacts/evals/mcts_guidance_v_heur_200.json
```

### One-command unattended A/B pipeline

```powershell
python -m scripts.run_guidance_ab_pipeline --run-label guidance-pilot --games 200
```

## 9. Artifact Conventions

- Checkpoints: `artifacts/ppo_checkpoint_<label>_seed<seed>.pt`
- Benchmarks: `artifacts/benchmarks/<label>.json`
- Evals: `artifacts/evals/<label>.json`
- Teacher data: `artifacts/teacher_data/<label>.jsonl` and summary JSON
- Guidance checkpoints: `artifacts/search_guidance_checkpoint_<label>.pt`
- Seed prefixes should include run label + seed for reproducibility.

## 10. Known Risks

- 25-game runs are noisy; they are only screening runs.
- Current MCTS is determinized-tree search and not full chance-node MCTS.
- MCTS runtime can increase sharply with `worlds * simulations * depth`.
- MCTS root widening improves action coverage, but poor prior quality can still slow convergence.
- Browser MCTS is not yet integrated; only rollout-search and PPO are available in UI.

---

If resuming in a fresh chat, provide this file plus the latest MCTS eval artifacts under `artifacts/evals/`, then proceed from section 6.
