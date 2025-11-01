# Active Context

## Current Focus

- Keep rules behavior deterministic and validated in the TypeScript engine.
- Keep the existing BC/REINFORCE/PPO training path stable.
- Raise search/MCTS strength using learned guidance (policy prior + value + opponent modeling), not heuristic-only search tuning.
- Train and tune guidance checkpoints from teacher data, then validate lift on side-swapped holdouts.
- Decide promotion order: stronger teacher first, then distillation for speed/browser deployment.
- Keep `docs/TRAINING_HANDOFF.md` as the restart document for training continuity across fresh sessions.

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
- Browser rollout-eval search profile (T3 config) is wired for direct play/testing.
- Bot policy failures are surfaced explicitly in the UI (no silent policy fallback).
- Development-card progress rings now animate only on fill increases using UI-local interpolation; canonical `progress/target` math remains unchanged and deterministic.
- Deed development now triggers a UI-only linear resource-flight animation from the acting player's resource rail to the deed token-side target on the in-development card (human and bot actions), and deed-token UI state is deferred until flight completion for visual consistency.

### Bridge

- NDJSON runtime is implemented and stable (`metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`).
- Contract artifact is versioned at `contracts/magnate_bridge.v1.json`.

### Trainer

- Bridge environment, encoders, eval/benchmark harnesses, and queue scripts are implemented.
- Teacher-data collection for distillation is now implemented (`scripts/generate_teacher_data.py`).
- MCTS policy was upgraded with:
  - progressive root widening (no permanent hard top-K root lockout)
  - LRU-style state-transition cache for repeated bridge step transitions
  - stronger non-terminal value proxy emphasizing district control/tiebreak structure
- Supported policy families in tooling:
  - `random`, `heuristic`
  - checkpoint-backed `bc`, `ppo`
  - additive deterministic `search` and `mcts` baselines with configurable knobs
- Search/MCTS now support optional PPO-format guidance checkpoints:
  - search root ranking can use learned policy priors
  - search rollouts can model opponent actions from learned policy distributions
  - MCTS priors can come from learned policy distributions
  - MCTS leaf evaluation can come from learned value head
- Guidance checkpoint training from teacher datasets is now implemented (`scripts/train_search_guidance.py`).
- Rollout-search sweep automation is now implemented (`scripts/search_teacher_sweep.py`) with side-swapped eval per preset and ranked summary outputs.
- Guidance A/B pipeline now defaults eval policy A to `search` (instead of `mcts`) in `scripts/run_guidance_ab_pipeline.py`.

## Immediate Next Steps

1. Run side-swapped rollout-search sweep (`scripts/search_teacher_sweep.py`) to select best teacher config beyond T1/T2/T3.
2. Confirm top sweep config on larger holdouts (`200+` then `700+`) against heuristic.
3. Train guidance checkpoint from the strongest confirmed search teacher data.
4. Run side-swapped search A/B (no guidance vs guidance) before additional MCTS work.
5. Proceed to student distillation for fast browser inference once teacher dominance is validated.

_Updated: 2026-02-28._
