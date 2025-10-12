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

Evaluation harness:

- `trainer.evaluate.evaluate_matchup`
- Script: `py -3.12 scripts/eval.py --games 50 --player-a-policy heuristic --player-b-policy random`
