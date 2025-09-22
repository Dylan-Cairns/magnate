# Project Brief

## Goal

Build a web-playable, single-player Magnate with a competent bot.

## Scope (v1)

- Official two-player Magnate rules parity.
- Human vs bot in browser.
- TS engine used by both browser and training through a bridge.
- Static deploy target (GitHub Pages style hosting).

## Non-Goals (v1)

- Multiplayer/accounts/backend services.
- Full cross-language rules schema/codegen.
- Native Python rules engine.

## Architecture

- **TS Engine (canonical)**: deterministic rules/state transitions.
- **UI (React/TS)**: thin layer over engine APIs.
- **Node Bridge**: stable JSON protocol to expose engine to Python.
- **Python Trainer**: RL loop as a bridge client.

## Success Criteria

- Rules behavior matches `memoryBank/magnateRules.md`.
- Deterministic replay from seed + actions.
- Stable bridge contract for TS/Python integration.
- Bot can play full games in browser without server runtime.

## Execution Order

1. Complete TS rules loop (setup, turn flow, legality, scoring, terminal).
2. Expand tests for full-turn and scoring behavior.
3. Implement bridge runtime using contract v1.
4. Add trainer scaffold and baseline policy.
5. Integrate model inference in browser.

## References

- Rules: `memoryBank/magnateRules.md`
- Contract: `memoryBank/bridgeInterfaceContract.md`
- Active work: `memoryBank/activeContext.md`
