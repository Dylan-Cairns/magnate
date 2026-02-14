# System Patterns

## Purpose

- Capture architecture and patterns used across Magnate.
- Keep decisions explicit, testable, and synchronized with implementation.
- Scope: official Magnate rules, single-player human vs computer.

## High-level architecture

- **TypeScript Core Engine (canonical rules)**: deterministic/pure game logic.
- **React UI**: thin state presenter/controller over engine APIs.
- **Node Bridge**: stable JSON protocol exposing engine operations to Python.
- **Python Trainer**: RL loop that treats bridge as the environment runtime.
- **Browser Inference**: static model artifact loaded client-side.

## Core design principles

- **Canonical rules runtime**: TS is the only source of gameplay truth.
- **Determinism**: fixed seed + action sequence -> identical outcomes.
- **Purity**: reducers/helpers are side-effect free.
- **Immutability**: return new state objects; do not mutate inputs.
- **Type safety**: strict TS, discriminated action unions, no `any` in core paths.

## Game loop and phases

- Explicit turn finite state machine.
- Exactly one card-play choice each turn (develop outright, buy deed, or sell).
- Trading and deed advancement occur in legal windows defined by rules.
- No pass action.

Primary engine API targets:
- `setupGame(seed)`
- `legalActions(state)`
- `applyAction(state, action)`
- `isTerminal(state)`
- `scoreGame(state)`

## Domain model

- **Card**: immutable identity, suits, rank, kind.
- **District**: marker + per-player developed stack + optional deed state.
- **Player**: hand, crowns, resource pool.
- **GameState**: players, districts, deck/discard, phase, turn, rng cursor, history/log.
- **GameAction**: buy deed, develop deed, develop outright, trade, sell.

## Randomness pattern

- Use seeded PRNG helpers only.
- Avoid `Math.random` in engine.
- Keep shuffle/reshuffle deterministic under seed and state.

## Legality and action space

- `legalActions(state)` emits only actions legal in current phase/state.
- Stable action IDs are required once training/checkpoints depend on them.
- Trainer and UI both consume the same legality output.

## Bridge boundary pattern

Use a **small interface contract** for TS/Python coordination (not a full rules schema).

Contract document: `memoryBank/bridgeInterfaceContract.md`

Bridge command set (v1 target):
- `metadata`
- `reset`
- `step`
- `legalActions`
- `observation`
- `serialize`

Metadata handshake must provide:
- action ID maps
- observation spec (size/segments)
- model I/O names
- contract version

## Observation pattern (bot-facing)

- Observation vector emitted from TS state only.
- Python does not infer hidden rules fields independently.
- Layout is contract-versioned and exposed through `metadata`.
- Legal action mask is derived from `legalActions(state)`.

## Training pattern

- Python RL loop interacts only through bridge contract.
- Start with random and scripted baselines before self-play complexity.
- Reward starts simple; shaping is optional and justified by measured need.
- Single-process execution is acceptable for v1.

## Browser inference pattern

- Load static model artifact from deployed assets.
- Build observation using the same contract layout used in training.
- Apply legal mask before action selection.
- Deterministic policy mode is default for predictable UX.

## Testing strategy

- Unit tests for engine rules helpers/reducers.
- Snapshot/fixture tests for hard rule sequences.
- Bridge contract tests for command payload stability.
- Parity checks: UI and Python consume matching action IDs/observation layout.

## Serialization and versioning

- Canonical JSON state includes `schemaVersion`.
- Contract payloads include `contractVersion`.
- Breaking bridge changes require a major version bump.

## Security and deployment

- Static hosting on GitHub Pages.
- No runtime backend required.
- No external API dependency for gameplay or inference.

## Open questions

- Final observation feature set and dimensionality.
- Final model artifact format (ONNX vs equivalent static format).
- Exact scripted baseline heuristic policy and tie-breakers.
- Whether bridge transport should remain stdin/stdout or adopt local HTTP later.