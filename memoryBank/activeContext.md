# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Keep rollout search as the temporary warm-start baseline.
- Use `scripts.eval_suite` as the canonical promotion protocol (paired seeds, side-swapped, CI + side-gap).
- Keep Python training surface clean while preparing for TD/Keldon implementation.

## Locked Decisions

- TS engine is canonical gameplay truth.
- Python training/eval uses the TS bridge contract, not duplicated rules.
- Bridge contract remains small and versioned.
- PPO, MCTS, and guidance codepaths are removed from active scope.

## Current State

### UI

- Browser bot catalog includes rollout search (default) and random.
- Browser rollout-search policy mirrors Python search root logic:
  - heuristic prior softmax
  - progressive root widening
  - root UCB selection across a fixed visit budget

### Bridge

- NDJSON runtime is stable (`metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`).
- Contract artifact: `contracts/magnate_bridge.v1.json`.

### Trainer

- Canonical side-swapped eval suite is implemented:
  - `scripts/eval_suite.py`
  - `trainer/eval_suite.py`
  - deterministic game sharding via `--workers`
- Search internals remain modularized:
  - `trainer/search/belief_sampler.py`
  - `trainer/search/forward_model.py`
  - `trainer/search/leaf_evaluator.py`
  - `trainer/search/root_selector.py`
- Active Python policy surface is now only:
  - `random`
  - `heuristic`
  - `search`
- Search sweep runner remains eval-suite based with preset packs (`scripts/search_teacher_sweep.py`).

## Immediate Next Steps

1. Confirm promoted search baseline with sweep gates (`120 -> 400 -> 2000` total games per preset).
2. Generate warm-start teacher data from promoted search baseline.
3. Implement TD value + opponent-model training pipeline on this cleaned stack.

_Updated: 2026-03-01._
