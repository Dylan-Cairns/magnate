# Bridge Interface Contract (TS <-> Python)

## Purpose

Define the smallest stable cross-runtime contract required for Python training to call the canonical TypeScript Magnate engine.

This contract does not duplicate game rules in Python. Rules remain in TypeScript.

## Scope

In scope:
- Bridge command names and request/response shapes
- Stable action IDs
- Observation vector layout metadata
- Model I/O naming metadata used by export/inference
- Error envelope shape

Out of scope:
- Full game-rules schema duplication in Python
- Python-native legality or scoring logic

## Versioning

- Contract name: `magnate_bridge`
- Initial version: `v1`
- Backward-compatible changes can add fields.
- Breaking changes require a new major contract version.

## Transport

- JSON messages over newline-delimited stdin/stdout for the first implementation.
- One request in, one response out.
- Each request carries a `requestId` echoed by the response.

## Request Envelope

```json
{
  "requestId": "string",
  "command": "reset | step | legalActions | observation | serialize | metadata",
  "payload": {}
}
```

## Response Envelope

```json
{
  "requestId": "string",
  "ok": true,
  "result": {}
}
```

or

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

## Commands

### `metadata`

Returns static contract metadata for Python bootstrapping and validation.

Result must include:
- `contractName`
- `contractVersion`
- `actionIdByName`
- `actionNameById`
- `observationSpec`:
  - `size`
  - `segments` (name, offset, size)
- `modelIo`:
  - `inputNames`
  - `outputNames`

### `reset`

Payload:
- optional `seed` (string)

Returns:
- fresh serialized state
- active player
- legal actions
- observation for active player

### `legalActions`

Payload:
- optional state token/reference if stateless bridge mode is later added

Returns:
- array of legal action objects with stable action IDs

### `observation`

Returns:
- observation vector for current actor
- optional action mask aligned with action IDs

### `step`

Payload:
- chosen action ID plus any action parameters needed by the action type

Returns:
- next serialized state
- terminal flag
- legal actions for next actor (if non-terminal)
- observation for next actor (if non-terminal)
- final score/winner fields (if terminal)

### `serialize`

Returns:
- canonical JSON state snapshot
- `schemaVersion` for saved-state compatibility checks

## Error Codes (initial set)

- `INVALID_COMMAND`
- `INVALID_PAYLOAD`
- `ILLEGAL_ACTION`
- `STATE_DESERIALIZATION_FAILED`
- `INTERNAL_ENGINE_ERROR`

## Stability Rules

- Action IDs are stable once training data/checkpoints depend on them.
- Observation segment offsets/sizes are stable within a contract version.
- Model I/O names are stable within a contract version.
- Additive metadata fields are allowed without a version bump.

## Adoption Notes (2026-02-19)

- This contract is the replacement for a larger shared rules schema approach.
- It is intentionally narrow to preserve momentum while preventing TS/Python drift.
