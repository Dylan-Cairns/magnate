# Active Context

## Current Focus

- Keep rules behavior deterministic and validated in the TypeScript engine.
- Keep the existing BC/REINFORCE/PPO training path stable.
- Raise search/MCTS strength using learned guidance (policy prior + value + opponent modeling), not heuristic-only search tuning.
- Use canonical side-swapped paired-seed evaluation (`scripts/eval_suite.py`) for promotion decisions.
- Train and tune guidance checkpoints from teacher data with soft policy targets, then validate lift on side-swapped holdouts.
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
- Browser rollout-eval search profile (T3 config) is wired for direct play/testing.
- Browser rollout-eval search profile is enabled and set as default bot.
- Legacy browser PPO profile/model path was removed from the web app catalog.
- Bot policy failures are surfaced explicitly in the UI (no silent policy fallback).
- Development-card progress rings now animate only on fill increases using UI-local interpolation; canonical `progress/target` math remains unchanged and deterministic.
- Deed development now triggers a UI-only linear resource-flight animation from the acting player's resource rail to the deed token-side target on the in-development card (human and bot actions), and deed-token UI state is deferred until flight completion for visual consistency.
- Buying/playing a property card (`buy-deed` or `develop-outright`) now triggers a UI-only linear card-flight animation from the acting player's hand area to the destination district lane, with board state commit deferred until flight completion; `buy-deed` flights render with in-development (deed) styling.
- Card flights now also cover `sell-card` (acting hand -> discard pile) and `end-turn` draw resolution (deck -> acting hand); `sell-card` commits state at flight start so the sold card leaves hand immediately, and discard visibility is UI-held until settle so destination appearance aligns with flight completion.
- Card flights now interpolate width/height across source and destination anchors so deck/discard size differences animate smoothly instead of snapping.
- In-development deed token side assignment is now stable per card/perspective in UI layout memory, preventing existing suit tokens from switching columns when new suits are added.
- Options menu now includes an Animations toggle persisted in browser `localStorage` (`magnate:animationsEnabled`); when disabled, flight animations are skipped and deed-progress rings render without tweening.
- `develop-outright` action presentation now groups by card (not payment permutation); when a card has only one payment pattern it uses a district-only picker and district selection commits immediately, and multi-payment cases use a combined district+payment picker that auto-commits once a valid pair is selected.
- Multi-source trade selection now also uses a single anchored popover with source selector + receive selector shown immediately and auto-commits once a valid pair is selected.

### Bridge

- NDJSON runtime is implemented and stable (`metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`).
- Contract artifact is versioned at `contracts/magnate_bridge.v1.json`.

### Trainer

- Bridge environment, encoders, eval/benchmark harnesses, and queue scripts are implemented.
- Canonical side-swapped evaluation suite is now implemented (`scripts/eval_suite.py`, `trainer/eval_suite.py`) with Wilson CI + side-gap reporting.
- Teacher-data collection for distillation is now implemented (`scripts/generate_teacher_data.py`).
- Teacher decision samples now carry optional soft policy targets (`actionProbs`) from search/MCTS root visit distributions.
- MCTS policy was upgraded with:
  - progressive root widening (no permanent hard top-K root lockout)
  - LRU-style state-transition cache for repeated bridge step transitions
  - stronger non-terminal value proxy emphasizing district control/tiebreak structure
- Search/MCTS shared internals are now split into reusable modules:
  - belief sampler (`trainer/search/belief_sampler.py`)
  - bridge forward model + transition cache (`trainer/search/forward_model.py`)
  - leaf evaluator + value cache (`trainer/search/leaf_evaluator.py`)
  - root ranking/priors/progressive widening helpers (`trainer/search/root_selector.py`)
- Determinized search root action selection now uses progressive widening and no permanent hard top-K lockout.
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
- Guidance training now supports soft policy targets from teacher samples (not only hard action indices).
- Rollout-search sweep automation is now implemented (`scripts/search_teacher_sweep.py`) with side-swapped eval per preset and ranked summary outputs.
- Guidance A/B pipeline now defaults eval policy A to `search` (instead of `mcts`) in `scripts/run_guidance_ab_pipeline.py`.
- Training encoding was bumped to v2 (`trainer.encoding.ENCODING_VERSION = 2`) with added hand-composition and endgame/tiebreak features; TS browser parity encoder was updated.

## Immediate Next Steps

1. Standardize all promotion checks on `scripts/eval_suite.py` (paired seeds, side-swapped, CI + side-gap).
2. Re-run rollout-search preset sweeps with eval-suite-style promotion gates (`200+`, `700+`, `2000`).
3. Retrain guidance on fresh teacher data using soft policy targets (`actionProbs`) and encoding v2 checkpoints only.
4. Compare guided vs unguided search and MCTS using matched eval-suite runs before additional algorithmic changes.
5. Proceed to student distillation for fast browser inference once teacher dominance is validated.

_Updated: 2026-02-28._
