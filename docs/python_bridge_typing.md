# Python Bridge Typing

This note records the bridge typing sources used by the Python trainer code.

## Canonical TS Sources

- Engine state and player view shapes:
  - `src/engine/types.ts`
  - `src/engine/view.ts`
- Bridge result and payload envelopes:
  - `src/bridge/protocol.ts`
  - `src/bridge/runtime.ts`

The TypeScript producer side remains the source of truth. Python only models the
subset of fields it currently consumes.

## Live Payload Validation

Bridge typing changes should be checked against live bridge output from the repo
virtualenv:

```powershell
.\.venv\Scripts\python - <<'PY'
from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv

with BridgeClient() as client:
    env = MagnateBridgeEnv(client=client)
    step = env.reset(seed="bridge-typing-check", first_player="PlayerA")
    print(sorted(step.state.keys()))
    print(sorted(step.view.keys()))
PY
```

## Current Python Consumers

The current boundary subset is derived from these Python readers:

- `trainer/encoding.py`
- `trainer/search/leaf_evaluator.py`
- `trainer/search/belief_sampler.py`
- `trainer/value_policy.py`
- `trainer/search_policy.py`
- `trainer/training.py`
- `trainer/teacher_data.py`
- `trainer/evaluate.py`
- `trainer/td/self_play.py`

## Boundary Model

- `trainer/bridge_payloads.py` contains the typed bridge payload subset used by
  Python.
- `trainer/bridge_parsing.py` parses raw bridge JSON results once at ingress.
- `trainer/bridge_client.py`, `trainer/types.py`, and `trainer/env.py` consume
  those parsed payloads.

The boundary model is intentionally smaller than the full TS engine schema. New
fields should be added only when Python starts consuming them or when bridge
result parsing needs to validate them directly.
