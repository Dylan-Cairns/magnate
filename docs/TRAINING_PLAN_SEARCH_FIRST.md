# Search Warm-Start Plan

This file defines the post-cleanup plan before TD implementation lands.

## Objective

Use rollout search only for warm-start leverage:

1. establish the strongest practical search baseline vs heuristic,
2. collect high-quality warm-start decision data,
3. transition to TD value + opponent-model training.

## Why This Plan

- The repo is intentionally stripped of PPO/MCTS/guidance paths.
- Search is useful for immediate signal, but not the long-run ceiling.
- TD/Keldon approach is the target architecture for stronger play.

## Evaluation Standard

Use `scripts.eval_suite` outputs:

- `candidateWinRate`
- `candidateWinRateCi95` (Wilson)
- `sideGap`

Never promote based on one-seat or tiny-sample runs.

## Staged Sweep Strategy

### Stage A: Coarse Ranking

- `games_per_side = 60` (120 total/preset)
- Sweep `coarse-v1` pack.
- Rank by:
  1. win rate (desc)
  2. side gap (asc)
  3. CI width (asc)

### Stage B: Confirmation

- Re-run top 2-3 presets at `games_per_side = 200` (400 total/preset).
- Keep presets with low side dependence and stable ordering.

### Stage C: Promotion Gate

- Run best preset at `games_per_side = 1000` (2000 total).
- Use this as the warm-start teacher baseline.

## Parameter-Tuning Order

To avoid a full 5D sweep:

1. Structure first:
   - tune `worlds`, `depth`, `max_root_actions`
   - keep `rollouts=1`, `rollout_epsilon=0.04`
2. Exploration next:
   - tune `rollout_epsilon` (`epsilon-v1` pack)
3. Extra compute last:
   - tune `rollouts` (`rollouts-v1` pack)

## Promotion Targets for Search Baseline

- Stage B (400 total):
  - strong candidate if win rate >= 0.78
  - side gap <= 0.06
- Stage C (2000 total):
  - baseline target win rate >= 0.85
  - CI low >= 0.82
  - side gap <= 0.04

## Exit Criteria to TD Build Phase

Move to TD implementation when:

- a stable search baseline is confirmed, and
- warm-start teacher data quality is acceptable.

First TD priorities:

1. value network with TD targets,
2. opponent action prediction model,
3. integration into `td-search` rollout/leaf evaluation.
