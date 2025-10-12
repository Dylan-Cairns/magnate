# Progress

## Implemented

- Architecture is locked:
  - TS engine is canonical.
  - Python integration goes through a small versioned bridge contract.
- Core engine flow is implemented and deterministic:
  - setup/deck lifecycle
  - legality + reducer action handling
  - phase resolver (`advanceToDecision`)
  - scoring and terminal finalization
- Engine state model was simplified:
  - canonical exhaustion source is `deck.reshuffles` (removed duplicated `exhaustionStage`)
  - income-choice return owner is ID-based (`incomeChoiceReturnPlayerId`)
  - removed transitional `IncomeRoll` phase
- Canonical game creation exists via `newGame(seed, { firstPlayer })`.
- Player-scoped visibility projection exists and is test-covered (`toPlayerView` / `toActivePlayerView`).
- Playable browser client exists (human vs random bot) using engine truth APIs.
- Board UI now uses centered overlapping district stacks with player-specific visual perspective for readability.
- Human-only `reset-turn` UX is implemented via a UI snapshot anchor at human turn start (`ActionWindow` pre-card), restoring state prior to `end-turn` without changing engine action contracts.
- Policy/controller boundaries now exist:
  - `src/engine/session.ts` (`createSession`, `stepToDecision`)
  - async-capable `src/policies/types.ts` + `src/policies/randomPolicy.ts`
  - `src/policies/catalog.ts` profile registry with strict fail-fast resolution for unknown/unavailable profiles
  - `src/ui/actionPresentation.ts` (+ tests)
- UI now exposes bot-profile selection and status text, with unavailable trained placeholders shown as disabled options.
- Runtime hardening and audit fixes landed:
  - `newGame` now validates `firstPlayer` at runtime
  - income-choice log attribution now reflects the chooser even when active player is restored
- Bridge runtime and deterministic action surface are implemented:
  - canonical action IDs/keys/order in `src/engine/actionSurface.ts`
  - NDJSON runtime in `src/bridge/runtime.ts` + CLI entrypoint `src/bridge/cli.ts`
  - command coverage: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`
  - bridge tests in `src/bridge/runtime.test.ts`
  - versioned contract artifact in `contracts/magnate_bridge.v1.json`
- Python trainer scaffold is implemented:
  - bridge client + env wrapper in `trainer/bridge_client.py` and `trainer/env.py`
  - training encoders in `trainer/encoding.py` with spec in `docs/TRAINING_ENCODING.md`
  - baseline policies/eval in `trainer/policies.py` and `trainer/evaluate.py`
  - sample collection + sample IO in `trainer/training.py`
  - behavior-cloning optimizer/checkpointing in `trainer/behavior_cloning.py`
  - stabilized RL fine-tuning in `trainer/reinforcement.py`:
    - mixed opponents (self/heuristic/random)
    - BC-anchor regularization
    - fixed-holdout eval-based best-checkpoint selection (default)
  - PyTorch PPO scaffold in `trainer/ppo_model.py` + `trainer/ppo_training.py` + `scripts/train_ppo.py`
  - scripts: `scripts/smoke_trainer.py`, `scripts/eval.py`, `scripts/train.py` (collect + BC), `scripts/finetune.py` (BC -> RL), `scripts/train_ppo.py` (PPO scaffold)
  - project Python venv bootstrap is standardized (`requirements.txt`, `scripts/setup_python_env.ps1`)
  - unittest coverage in `trainer_tests/`
- Tooling gates include lint + typecheck + tests.

## In Progress

- Expanding rules-parity scenario coverage, especially full-turn/full-game edges.
- Tuning RL schedules/hyperparameters and benchmark tracking across BC, stabilized REINFORCE, and PPO scaffold loops.

## Remaining

- Experiment tracking/metrics for BC baseline, RL fine-tuning, and self-play runs.
- Surpass heuristic baseline consistently with current stabilized REINFORCE controls and/or PPO path.
- Model inference wiring in browser client.
- Deployment polish for static hosting path.

## Risks / Watch Items

- Remaining parity gaps are likely in edge-case sequencing rather than base mechanics.
- Bridge/API stability needs explicit guardrails as trainer integration starts.
- Trained profiles are still placeholders until bridge/runtime inference wiring is complete.

_Updated: 2026-02-21._
