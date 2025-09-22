# Bridge Interface Contract (TS <-> Python)

## Purpose

Define the minimal stable boundary for Python training to call the canonical TS Magnate engine.

- Contract name: `magnate_bridge`
- Contract version: `v1`

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

- Newline-delimited JSON over stdin/stdout (initial implementation)
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
  - returns `contractName`, `contractVersion`, action ID maps, observation spec, model I/O names
- `reset`
  - optional seed; returns fresh state + active actor view
- `legalActions`
  - returns legal action payloads for current state
- `observation`
  - returns actor observation (+ optional mask)
- `step`
  - applies action; returns next state view and terminal info if game ended
- `serialize`
  - returns canonical state snapshot with `schemaVersion`

## Error Codes

- `INVALID_COMMAND`
- `INVALID_PAYLOAD`
- `ILLEGAL_ACTION`
- `STATE_DESERIALIZATION_FAILED`
- `INTERNAL_ENGINE_ERROR`

## Stability Rules

- Action IDs are stable once used by training/checkpoints.
- Observation layout is stable within a contract version.
- Model I/O names are stable within a contract version.
- Additive fields are allowed without a major version bump.
- Breaking changes require a major contract version bump.
