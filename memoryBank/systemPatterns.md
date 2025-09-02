# System Patterns

## Purpose

- Capture the architecture and patterns used across the project.
- Keep decisions short, explicit, and testable.
- Scope: official Magnate rules only; single-player vs computer.

## High-level architecture

- **TypeScript Core Engine**: pure game logic, no I/O.
- **React UI**: thin view over the engine.
- **Node Bridge**: JSON over stdin/stdout for training.
- **Python Trainer**: reinforcement learning loop calling the bridge.
- **Browser Inference**: loads a static model and picks actions.

## Core design principles

- **Determinism**: seeded randomness; same inputs → same outputs.
- **Purity**: reducers are pure functions; no hidden state, no I/O, no clocks.
- **Immutability**: return new state; never mutate inputs.
- **Type safety**: strict TypeScript, discriminated unions for actions, no `any`.

## Game loop & phases

- Finite state machine for turn flow.
- Exactly one “play” per turn (per official rules).
- Central reducer: `next = reduce(prev, action)`.
- Helper functions: `setup`, `legalActions`, `score`, `isTerminal`.

## Domain model (plain language)

- **Card**: identity, suits, rank.
- **District**: ordered stacks along the district line.
- **Player**: hand, resources by suit, deeds/developments.
- **Game state**: players, board, turn/phase, RNG seed, history of actions for UI.
- **Actions**: buy deed, buy outright, develop, trade (3:1), sell (per rules), pass (only when legal).

## Randomness pattern

- Single master seed feeds all randomness (e.g., shuffles).
- Use a pure PRNG; avoid `Math.random`.
- Deterministic results when seed and action sequence are the same.

## Legality & action space

- `legalActions(state)` returns only the actions allowed right now.
- Stable action IDs for UI and trainer.
- The bot is never offered illegal actions.

## Observation format (for the bot)

- **Hand**: fixed “multi-hot” vector over the full card list (1 for each card you hold, 0 otherwise).
- **Board**: compact features per district (stack depth, top card’s suits/rank).
- **Counts & flags**: deck/discard remaining as needed, current player flag, resource counts.
- **Scaling**: numeric features normalized to a common range (e.g., 0..1) for steady learning.

## Bridge protocol (Node ⇄ Python)

- **Commands**: `reset(seed?)`, `step(actionId)`, `observation()`, `legalActions()`, `serialize()`.
- **Message shape**: simple JSON with a type and payload, one message per line.
- **Errors**: typed codes with enough context to reproduce locally.

## Training patterns

- Trial-and-error learning through many simulated games.
- Reward: win = 1, loss = 0 (optional small progress points only if needed).
- Baselines: random and a simple scripted heuristic.
- **Execution model**: single thread, single process.

## Inference in the browser (static model)

- **Approach**: export a small neural network as a `model.json` file (weights and layer descriptions).
- **Flow**:
  1. UI builds the observation vector from the current game state (same layout used during training).
  2. UI obtains the legal-action mask from `legalActions(state)`.
  3. The model produces a score for each action.
  4. Mask out illegal actions and choose the highest-scoring legal action (deterministic).
- All assets (app and `model.json`) are static files served from GitHub Pages.

## Testing strategy

- **Unit tests** for reducers and helpers.
- **Snapshot fixtures**: serialize states after known sequences; fail on unexpected changes.
- **Property tests** for invariants (e.g., resources never negative; legal actions are truly legal).

## Serialization & versioning

- Canonical JSON for game state with a `schemaVersion`.
- Stable ordering for arrays where needed.
- Migration handlers if the schema changes.

## Performance & UX guidelines

- Correctness first; keep the engine allocation-light where practical.
- Never block the main thread during bot turns; show clear “thinking” feedback.
- Simple animations and clear affordances over raw speed.

## Error handling & logging

- Reducers validate inputs; boundary layers surface errors with context.
- **Logging is always on** to aid user feedback and debugging.
- **Turn History**: visible list of all human and AI actions with key deltas (resources, scores) for user review.

## Security & licensing

- Static hosting on GitHub Pages; no external servers or APIs.
- Use permissive/CC-compatible assets for suits/icons; attribute appropriately.

## Coding conventions

- Strict TypeScript, discriminated unions for actions, functional helpers.
- File layout by domain: `engine/`, `ui/`, `bridge/`, `trainer/` (stubs).
- Naming: `VerbNoun` for actions; `getX`, `isX`, `toX` helpers.

## Open questions

- Exact observation vector layout and size.
- Scripted baseline heuristics and tie-breakers.
- PRNG choice and seeding details.
- Minimal district features that are sufficient for learning without leaking hidden info.
