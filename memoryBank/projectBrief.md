# Project Brief

## Goal

Build a web-playable, single-player Magnate with a competent bot.

## Scope (v1)

- Official two-player Magnate rules parity.
- Human vs bot in browser.
- TS engine used by both browser and training through a bridge.
- Static deploy target (GitHub Pages style hosting).

## Non-Goals (v1)

- Multiplayer, accounts, or backend services.
- Full cross-language rules schema or codegen.
- Native Python rules engine.

## Architecture

- **TS Engine (canonical)**: deterministic rules and state transitions.
- **UI (React/TS)**: thin layer over engine APIs.
- **Node Bridge**: stable JSON protocol exposing the engine to Python.
- **Python Trainer**: RL loop as a bridge client.

## Success Criteria

- Rules behavior matches `memoryBank/magnateRules.md`.
- Deterministic replay from seed plus actions.
- Stable bridge contract for TS/Python integration.
- Bot can play full games in browser without a server runtime.

## Licensing

Project license is CC BY-NC-SA 4.0 (`LICENSE`; matching SPDX identifier in
`package.json`). Game credits appear in the info modal and `README.md`; bundled
software notices live in `public/third-party-notices.txt`.

## References

- Rules: `memoryBank/magnateRules.md`
- Contract: `memoryBank/bridgeInterfaceContract.md`
- Active work: `memoryBank/activeContext.md`
