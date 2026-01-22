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

## References

- Rules: `memoryBank/magnateRules.md`
- Contract: `memoryBank/bridgeInterfaceContract.md`
- Active work: `memoryBank/activeContext.md`
