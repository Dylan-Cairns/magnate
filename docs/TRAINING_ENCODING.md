# Training Encoding (v1)

This file defines the first-pass Python-side training encoding used by the bridge trainer scaffold.

## Context

- Canonical game rules/state transitions remain in TypeScript.
- Python reads state through `yarn bridge` commands (`observation`, `legalActions`, `step`).
- Training uses action scoring over the legal action list instead of a single global fixed action index.

## Observation Vector

- Function: `trainer.encoding.encode_observation(view)`
- Input: bridge `observation.view` payload (active-player view by default).
- Output dimension: `OBSERVATION_DIM = 186`.

Feature groups:

1. Phase one-hot (`StartTurn`, `TaxCheck`, `CollectIncome`, `ActionWindow`, `DrawCard`, `GameOver`).
2. Turn/deck/timing scalars:
   - turn
   - `cardPlayedThisTurn`
   - `finalTurnsRemaining`
   - draw count, discard count, reshuffles
   - last income roll (`die1`, `die2`)
   - last tax suit one-hot.
3. Active + opponent economy:
   - resources by suit
   - hand counts
   - crown-suit counts.
4. District board state (5 districts, active then opponent stack features):
   - marker suit counts
   - developed count
   - developed rank sum
   - deed presence/progress/target
   - deed token suit counts.

All numeric features are normalized to bounded ranges for stable learning input.

## Action Candidate Features

- Function: `trainer.encoding.encode_action_candidates(actions)`
- Input: bridge `legalActions.actions` (canonical key-sorted list).
- Output per action dimension: `ACTION_FEATURE_DIM = 40`.

Per-candidate feature groups:

1. Action-ID one-hot (`buy-deed`, `choose-income-suit`, `develop-deed`, `develop-outright`, `end-turn`, `sell-card`, `trade`).
2. Card/district metadata:
   - normalized card id
   - normalized card rank
   - normalized district index.
3. Actor/suit metadata:
   - `playerId` one-hot (for `choose-income-suit`)
   - selected suit one-hot
   - trade give/receive suit one-hots.
4. Token payload:
   - token vector by suit from `payment`/`tokens`
   - normalized token total.
5. Presence flags:
   - has card id
   - has district id
   - card is property.

## Action Selection Model Shape

Training-time policy input is `(observation, [action_feature_0 ... action_feature_n])`.

- The policy scores each legal action candidate.
- The chosen action is mapped back via canonical `actionKey`.
- This avoids a brittle giant global action index and supports dynamic legality naturally.

## Baseline Policies and Eval

Baseline policies currently implemented in `trainer.policies`:

- `RandomLegalPolicy`
- `HeuristicPolicy`
- `BehaviorCloningPolicy` (checkpoint-backed)
- `TorchPpoPolicy` (checkpoint-backed, requires PyTorch)

Evaluation harness:

- `trainer.evaluate.evaluate_matchup`
- Script (with project `.venv` active): `python -m scripts.eval --games 50 --player-a-policy heuristic --player-b-policy random`
- BC eval example:
  - `python -m scripts.eval --games 50 --player-a-policy bc --player-a-checkpoint artifacts/bc_checkpoint.json --player-b-policy random`
- PPO eval example:
  - `python -m scripts.eval --games 50 --player-a-policy ppo --player-a-checkpoint artifacts/ppo_checkpoint.pt --player-b-policy heuristic`

Canonical benchmark protocol:

- Script: `python -m scripts.benchmark --candidate-policy bc --candidate-checkpoint artifacts/bc_checkpoint.json`
- Fixed holdout seed prefixes:
  - `bench-random-holdout`
  - `bench-heuristic-holdout`
- Default games per matchup: `200`
- Selection score:
  - `0.7 * win_rate_vs_heuristic + 0.3 * win_rate_vs_random`
- Artifact output:
  - one JSON file per run under `artifacts/benchmarks/` by default

## Behavior-Cloning Warm Start

Training script:

- `python -m scripts.train --games 20` (with project `.venv` active)
  - collects decision samples (`artifacts/training_samples.jsonl` by default)
  - optimizes a supervised action-ranking model from chosen legal actions
  - writes checkpoint (`artifacts/bc_checkpoint.json` by default)

Model scoring form (`trainer.behavior_cloning`):

- Input: observation vector + legal action feature vectors
- Score for candidate `a`: `score(a) = obs^T W a + w_action^T a`
- Optimization: SGD on per-decision softmax cross-entropy over the legal action list
- Checkpoint payload includes:
  - `checkpointType = magnate_behavior_cloning_v1`
  - observation/action dimensions
  - `obsActionWeights`, `actionWeights`
  - metadata (sample source + optimizer params)

## RL Fine-Tuning (Stabilized REINFORCE)

Fine-tuning script:

- `python -m scripts.finetune --checkpoint-in artifacts/bc_checkpoint.json --checkpoint-out artifacts/rl_checkpoint.json --episodes 300 --eval-games 100 --eval-every 50`
  - default checkpoint-selection mode now uses fixed holdout seeds (`--eval-mode fixed-holdout`)
  - loads a BC checkpoint
  - runs seeded mixed-opponent training with stochastic learner sampling
    - self-play snapshot opponent
    - heuristic opponent
    - random opponent
  - applies REINFORCE policy-gradient updates to the same model weights
  - anchors weights toward source BC checkpoint (`--bc-anchor-coeff`)
  - evaluates every `N` episodes and keeps the best-scoring checkpoint

Update shape:

- Policy logits are produced from the same candidate-scoring form:
  - `score(a) = obs^T W a + w_action^T a`
- REINFORCE gradient update uses:
  - `advantage = winner_reward / decisions_by_player_in_episode`
  - `winner_reward in {-1, 0, +1}`
  - L2 weight decay from config
  - BC-anchor penalty toward source checkpoint weights

## PPO Scaffold (PyTorch)

Dependency:

- PyTorch is required in the trainer environment (`requirements.txt`)

Training script:

- `python -m scripts.train_ppo --checkpoint-out artifacts/ppo_checkpoint.pt --episodes 1024 --episodes-per-update 32 --eval-games 100 --eval-every-updates 5`
  - uses the same bridge observations and legal action candidate features
  - actor-critic policy/value network scores legal action candidates directly
  - PPO clipped objective over collected bridge trajectories
  - fixed-holdout checkpoint selection (`--eval-mode fixed-holdout` by default)
  - default progress heartbeat prints to stderr every 5 updates (`--progress-every-updates 0` disables)

Checkpoint type:

- `magnate_ppo_policy_v1` (`trainer/ppo_model.py`)
