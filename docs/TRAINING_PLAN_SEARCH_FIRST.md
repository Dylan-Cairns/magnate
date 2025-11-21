# Search-First Training Plan

This is the active plan for reaching a bot that decisively outperforms heuristic.

## Objective

Build the strongest practical teacher quickly:

1. Improve determinized rollout search through evaluation-driven tuning.
2. Stabilize results with side-swapped paired-seed evidence.
3. Add guidance/distillation only after teacher strength is validated.

## Why This Plan

- Current search path already outperforms legacy policy training.
- Determinization + lookahead quality has high immediate leverage.
- Promotion decisions need robust statistics, not short noisy runs.

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
- Use this as promotion evidence.

## Parameter-Tuning Order

To avoid 5D grid explosion:

1. Structure first:
   - tune `worlds`, `depth`, `max_root_actions`
   - keep `rollouts=1`, `rollout_epsilon=0.04`
2. Exploration next:
   - tune `rollout_epsilon` (`epsilon-v1` pack)
3. Extra compute last:
   - tune `rollouts` (`rollouts-v1` pack)

## Promotion Targets

Use these as operational gates (adjust only with explicit team decision):

- Stage B (400 total):
  - strong candidate if win rate >= 0.78
  - side gap <= 0.06
- Stage C (2000 total):
  - promotion target win rate >= 0.85
  - CI low >= 0.82
  - side gap <= 0.04

## When to Stop Tuning and Add Infra

Switch to additional infrastructure work only if:

- best confirmed search config plateaus below target strength, or
- runtime costs are unacceptable at winning configs.

Priority order for next infra:

1. Explicit dice/chance expectation in lookahead.
2. Stronger opponent-response modeling.
3. Further value/policy guidance improvements.
