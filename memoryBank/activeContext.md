# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Keep rollout search as the temporary warm-start baseline.
- Use `scripts.eval_suite` with explicit modes:
  - `--mode gate` for sequential SPRT gating vs search
  - `--mode certify` for fixed-size side-swapped confidence evals
- Execute and iterate gate-first TD loops (`collect/train` chunks -> gate -> optional certify).

## Locked Decisions

- TS engine is canonical gameplay truth.
- Python training/eval uses the TS bridge contract, not duplicated rules.
- Bridge contract remains small and versioned.
- PPO, MCTS, and guidance codepaths are removed from active scope.
- Python training/eval paths are fail-fast (no silent fallback labels/actions/probabilities).

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
  - shard/summary leg reporting is seat-keyed (`winsBySeat` + `policyBySeat`) to prevent same-policy-name collisions
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
- TD Phase 1 primitives are now implemented under `trainer/td/`:
  - value/opponent model definitions
  - replay buffers
  - TD target helpers (`n-step`, `TD(lambda)`)
  - checkpoint save/load contracts for TD value/opponent models
  - self-play episode collection into value transitions + opponent samples
  - value trainer utilities (target network sync + clipped gradient updates)
- TD-focused unit tests are in place under `trainer_tests/test_td_*.py`.
- TD Phase 2 orchestration is implemented:
  - `scripts.collect_td_self_play` writes replay artifacts:
    - `<stamp>-<label>.value.jsonl`
    - `<stamp>-<label>.opponent.jsonl`
  - `scripts.train_td` trains value/opponent models from replay with checkpoint cadence.
  - `scripts.run_td_loop` orchestrates:
    - multiple collect/train chunks (`--chunks-per-gate`)
    - gate decision on latest checkpoint only (`eval_suite --mode gate`)
    - certify panel only on gate pass (`eval_suite --mode certify`)
    - explicit promotion decision in `loop.summary.json`
    - collect-stage sharding via `--collect-workers` to use multiple CPU cores during replay generation
    - `--cloud` applies fixed 8 vCPU worker profile for hosted runs
    - shard merge deletes shard JSONL files after append to reduce peak disk usage on small-volume hosts
  - `td-value` policy is available through `scripts.eval` and `scripts.eval_suite` via:
    - `--candidate-policy td-value` / `--opponent-policy td-value`
    - `--td-value-checkpoint`
    - `--td-worlds`
- TD Phase 3 initial search integration is implemented:
  - `td-search` policy in `trainer.policies`:
    - determinized search root logic retained
    - leaf evaluation replaced with TD value checkpoint
    - non-terminal TD leaf values are interpreted in active-player perspective and converted to root perspective via sign flip when needed
    - opponent rollout guidance via required opponent checkpoint
  - surfaced via `scripts.eval`, `scripts.eval_suite`, `scripts.generate_teacher_data`, and `scripts.collect_td_self_play`.
- Runtime guardrails:
  - training/eval scripts require active `.venv` and explicit policy flags
  - `scripts.eval_suite` requires explicit `--mode gate|certify`
  - `td-search` configuration requires both value and opponent checkpoints
  - TD value transitions are strict: `done=False` requires `nextObservation`; `done=True` requires `nextObservation=null`

### TD Value Semantics

- `V(obs)` is locked to active-player perspective:
  - model output is expected outcome for `obs.activePlayerId`.
- TD replay collection is aligned with that definition (active-player observations).
- Any root-player search/eval leaf score must convert with:
  - `v_root = v_active` if `activePlayerId == rootPlayer`
  - `v_root = -v_active` otherwise
- TD training target modes:
  - default TD(0): `target = reward + gamma * (1 - done) * V_target(next_obs)`
  - optional TD(lambda): enabled with sequence-aware replay rows (`episodeId`, `timestep`) and `scripts.train_td --value-target-mode td-lambda --td-lambda <0..1>`

## Immediate Next Steps

1. Confirm promoted search baseline with sweep gates (`120 -> 400 -> 2000` total games per preset).
2. Generate warm-start teacher data from promoted search baseline.
3. Run repeated TD cycles and track promotion decisions:
   - run `scripts.run_td_loop` for standardized chunked collect/train + gate/certify cycles,
   - rank promoted checkpoints against search baselines.

_Updated: 2026-03-03._
