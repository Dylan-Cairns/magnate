# Active Context

## Current Focus

- Keep rules behavior deterministic and validated in the TypeScript engine.
- Keep the existing BC/REINFORCE/PPO training path stable.
- Evaluate and tune the additive `search` policy as a stronger teacher candidate.
- Decide whether to proceed with teacher-data distillation into a fast browser policy.

## Locked Decisions

- TypeScript engine is the canonical rules implementation.
- Python training calls TS through a Node bridge.
- Shared boundary is a small, versioned interface contract.
- Native Python rules are out of scope unless throughput becomes a proven bottleneck.

## Current State

### Engine

- Core setup/turn-flow/reducer/scoring loops are implemented and deterministic.
- Canonical initialization is `newGame(seed, { firstPlayer })` with runtime validation.
- Canonical action surface exists (`actionId`, stable `actionKey`, deterministic legal-action order).
- Recent edge-case tests now cover:
  - tax-before-income sequencing with deed-token immunity
  - mixed d10 income branches (`1/x`, higher-die handling, no-match zero-income)
  - generated multi-choice income queue and player handoff/restore
  - Excuse follow-on placement overlap
  - ace buy/develop cost legality paths
  - full second-exhaustion final-turn countdown

### UI

- Browser game is playable (human vs bot) with engine-legal action dispatch.
- Policy boundary is unified (`ActionPolicy`), and bot profiles are selector-driven.
- Champion browser PPO profile is enabled and set as default bot.

### Bridge

- NDJSON runtime is implemented and stable (`metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`).
- Contract artifact is versioned at `contracts/magnate_bridge.v1.json`.

### Trainer

- Bridge environment, encoders, eval/benchmark harnesses, and queue scripts are implemented.
- Supported policy families in tooling:
  - `random`, `heuristic`
  - checkpoint-backed `bc`, `ppo`
  - additive deterministic `search` baseline with configurable search knobs

## Immediate Next Steps

1. Finalize a search-teacher configuration based on larger holdout evals.
2. Document promotion criteria for teacher/champion checkpoints.
3. Build teacher-data generation + distillation path (search -> fast policy).
4. Reuse opponent-pool PPO only if distillation still needs RL fine-tuning.

_Updated: 2026-02-27._
