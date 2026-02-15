# Bridge Interface Contract (TS <-> Python)

## Purpose

Define the minimal stable boundary for Python training to call the canonical TS Magnate engine.

- Contract name: `magnate_bridge`
- Contract version: `v1`
- Versioned contract artifact: `contracts/magnate_bridge.v1.json`

## Scope

In scope:

- Bridge command names
- Request/response envelope
- Action ID stability
- Observation layout metadata
- Model I/O metadata
- Error envelope and codes

Out of scope:

- Full rules schema duplication in Python
- Python-native legality/scoring logic

## Transport

- Newline-delimited JSON over stdin/stdout (`yarn bridge`)
- One request -> one response
- `requestId` must be echoed

## Envelope

Request:

```json
{
  "requestId": "string",
  "command": "metadata | reset | step | legalActions | observation | serialize",
  "payload": {}
}
```

Success response:

```json
{
  "requestId": "string",
  "ok": true,
  "result": {}
}
```

Error response:

```json
{
  "requestId": "string",
  "ok": false,
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

## Commands (v1)

- `metadata`
  - payload: optional object
  - returns `contractName`, `contractVersion`, `schemaVersion`, command list, action IDs, action-surface metadata, observation spec, model I/O names
- `reset`
  - payload:
    - `seed?`: string
    - `firstPlayer?`: `PlayerA | PlayerB`
    - `serializedState?`: canonical state snapshot to load
    - `skipAdvanceToDecision?`: boolean (only used with `serializedState`)
  - returns state snapshot + active actor view + terminal flag
- `legalActions`
  - payload: optional object
  - returns canonical legal actions for current state, each with:
    - `actionId`
    - `actionKey`
    - `action` payload
  - canonical order is lexicographic by `actionKey`
- `observation`
  - payload:
    - `viewerId?`: `PlayerA | PlayerB` (default active player)
    - `includeLegalActionMask?`: boolean
  - returns `toPlayerView` payload (+ optional legal-action-key mask)
- `step`
  - payload:
    - `action?`: full action payload
    - `actionKey?`: stable action key from `legalActions`
  - requires either `action` or `actionKey`
  - if both are provided, they must refer to the same action
  - applies action, advances to decision, returns state snapshot + active actor view + terminal flag
- `serialize`
  - payload: optional object
  - returns canonical state snapshot with `schemaVersion`

## Deterministic Action Surface

- Stable action IDs:
  - `buy-deed`
  - `choose-income-suit`
  - `develop-deed`
  - `develop-outright`
  - `end-turn`
  - `sell-card`
  - `trade`
- Stable action keys are canonicalized by `src/engine/actionSurface.ts`.
- `legalActions` canonical order is deterministic: ascending lexicographic `actionKey`.

## Error Codes

- `INVALID_COMMAND`
- `INVALID_PAYLOAD`
- `ILLEGAL_ACTION`
- `STATE_DESERIALIZATION_FAILED`
- `INTERNAL_ENGINE_ERROR`

## Stability Rules

- Action IDs are stable once used by training/checkpoints.
- Action key generation and canonical legal-action ordering are stable within a contract version.
- Observation layout is stable within a contract version.
- Model I/O names are stable within a contract version.
- Additive fields are allowed without a major version bump.
- Breaking changes require a major contract version bump.
