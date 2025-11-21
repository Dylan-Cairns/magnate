# Training Encoding (v2)

This document defines the canonical training encoding shared by Python tooling and TS browser parity code.

## Scope

- Canonical rules/state transitions remain in TypeScript.
- Python receives state through bridge commands (`observation`, `legalActions`, `step`).
- Models score the current legal-action list directly (no giant global action index).

## Observation Vector

- Python encoder: `trainer.encoding.encode_observation(view)`
- TS parity encoder: `src/policies/trainingEncoding.ts`
- `OBSERVATION_DIM = 206`
- `ENCODING_VERSION = 2`

Feature groups:

1. Phase one-hot (`StartTurn`, `TaxCheck`, `CollectIncome`, `ActionWindow`, `DrawCard`, `GameOver`).
2. Turn/deck/timing scalars:
   - turn
   - `cardPlayedThisTurn`
   - `finalTurnsRemaining`
   - draw/discard counts, reshuffles
   - last income roll (`die1`, `die2`)
   - last tax suit one-hot
3. Active + opponent economy:
   - resources by suit
   - hand counts
   - crown-suit counts
4. Active-player hand composition:
   - suit histogram
   - rank histogram (`1..10`)
5. Endgame/tiebreak context:
   - endgame flag
   - district-control proxy differential
   - developed-rank differential proxy
   - resource differential proxy
6. District board state (5 districts, active then opponent):
   - marker suit counts
   - developed count
   - developed rank sum
   - deed presence/progress/target
   - deed token suit counts

All numeric features are normalized to bounded ranges.

## Action Candidate Features

- Encoder: `trainer.encoding.encode_action_candidates(actions)`
- Per-candidate dimension: `ACTION_FEATURE_DIM = 40`

Feature groups:

1. Action-ID one-hot:
   - `buy-deed`
   - `choose-income-suit`
   - `develop-deed`
   - `develop-outright`
   - `end-turn`
   - `sell-card`
   - `trade`
2. Card/district metadata:
   - normalized card id
   - normalized card rank
   - normalized district index
3. Actor/suit metadata:
   - `playerId` one-hot (`choose-income-suit`)
   - selected suit one-hot
   - trade give/receive suit one-hots
4. Token payload:
   - token vector by suit from `payment` / `tokens`
   - normalized token total
5. Presence flags:
   - has card id
   - has district id
   - card is property

## Checkpoint Contract

- PPO/guidance checkpoint type: `magnate_ppo_policy_v1` (`trainer/ppo_model.py`)
- Checkpoints must include `encodingVersion` and it must match `ENCODING_VERSION`.
- Legacy checkpoints without `encodingVersion` are rejected.

## Compatibility Rule

- Any change to observation or action candidate layout requires:
  - encoder updates in both Python and TS parity code
  - checkpoint compatibility/version update
  - documentation update in this file
