# Magnate (Web) + RL Bot — Project Brief

## One-liner

Build a web-playable, single-player **Magnate** with a competent computer opponent.

The **TypeScript engine is canonical for rules** and runs in browser + Node. Python training calls that engine through a narrow, stable bridge contract. **Official rules only.**

## Goals

- **Accurate rules**: Implement Magnate exactly as written on the Decktet wiki.
- **Competent bot**: Beat random and simple heuristic baselines; feel fair and capable to humans.
- **Single rules implementation**: Keep game legality/scoring/setup logic in TS only for v1.
- **Reproducible & testable**: Seeded randomness, pure state updates, deterministic fixtures.
- **Static deploy**: Ship browser app to GitHub Pages with model artifact as static asset.
- **Execution order**: Build a complete rules engine and scripted opponent before RL training.

## Non-Goals (v1)

- Multiplayer, accounts, persistence, mobile polish, localization.
- Full cross-language rules schema/codegen.
- Native Python rules implementation unless a real bridge-throughput bottleneck appears.

## Game scope (rules parity)

Two-player Magnate per official write-up: setup, turn order, taxation/income, developing, deeds, trade, sell, draw exhaustion, endgame, and scoring.

Rule source of truth: `memoryBank/magnateRules.md`.

## Architecture

- **TypeScript Engine (canonical rules)**
  - Pure functions over immutable state.
  - Deterministic under seed + action sequence.
  - No I/O in reducer logic.

- **Browser UI (TypeScript/React)**
  - Thin rendering/controller layer over engine APIs.
  - Human vs bot only for v1.

- **Node Bridge (TS runtime boundary)**
  - Hosts the TS engine.
  - Speaks newline-delimited JSON over stdin/stdout.
  - Provides stable command surface to Python.

- **Python Trainer (bridge client)**
  - Treats engine as external environment via bridge.
  - Consumes legal actions and observations from TS runtime.
  - Avoids duplicated rules logic.

## Shared Contract Strategy

Use a **small interface contract** (not full rules schema) to lock:

- Bridge request/response envelope
- Command names and payload requirements
- Stable action IDs
- Observation layout metadata
- Model I/O names for export/inference
- Error code taxonomy

Contract reference: `memoryBank/bridgeInterfaceContract.md`.

## Data model (engine)

- **State**
  - Deck/discard, player hands, district line markers, player resources, deeds/developments, turn/phase, seed/rng cursor, minimal action log.

- **Actions (discrete families)**
  - Buy deed, develop deed, develop outright, trade (3:1 per action), sell.
  - No pass action.

- **Legality**
  - `legalActions(state)` returns exactly legal actions for current state.

## Reproducibility and tests

- Seeded PRNG only; no `Math.random` in rules.
- Reducer purity: `next = reduce(prev, action)`.
- Snapshot fixtures for tricky rule paths.
- Contract tests for bridge payload stability.

## Training plan

- Start with random and heuristic baselines.
- Train via PPO-style loop in Python against bridge environment.
- Illegal actions are masked by legal action set from TS engine.
- Reward starts simple (win/loss), then only add shaping if required.

## Packaging and deploy

- Build static web app for GitHub Pages.
- Model artifact loaded client-side (no backend).
- Inference format can be ONNX or equivalent static artifact, but contract names/dims stay stable.

## Quality bar

- Rules parity confirmed against `magnateRules.md`.
- Deterministic replays from seed + action log.
- Unit/snapshot coverage for setup, legality, taxation, income, selling, and scoring.
- Bridge contract tests guard TS/Python integration stability.

## Milestones

1. **Truth-source reset**: Align docs and architecture decisions.
2. **Interface contract**: Finalize bridge contract v1 and metadata handshake.
3. **TS engine completion**: Setup, turn FSM, legality, scoring, terminal flow.
4. **Engine tests**: Unit + snapshot fixtures for representative rule paths.
5. **Scripted bot + minimal UI**: Human vs heuristic bot in browser.
6. **Bridge runtime**: TS engine accessible from Python with contract tests.
7. **Python trainer**: PPO baseline loop through bridge.
8. **Model export + browser inference**: Static artifact and release.

## References

- Magnate rules: http://wiki.decktet.com/game:magnate
- Memory workflows: `docs/AGENT_GUIDE.md`
- Agent manifest: `AGENTS.md`