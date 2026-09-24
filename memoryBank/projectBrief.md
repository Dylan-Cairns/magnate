# Project Brief

## Goal

Build a web-playable, single-player Magnate with strong bots: a deterministic
TypeScript rules engine, a browser UI, and a Python training stack that improves
the trained opponent through gated self-play.

## Scope

- Official two-player Magnate rules parity, plus the optional extended ruleset
  that shuffles the four Courts into the deck.
- Human vs bot in the browser with selectable profiles: Easy, Medium, and Hard
  rollout-search profiles for both rulesets, and an Experimental TD-root search
  profile for the standard ruleset only.
- One deterministic TypeScript engine used by browser play, bot evaluation, and
  Python training through the Node bridge.
- Python training and evaluation: collect, train, gate, and promote, with the
  checkpoint registry in `models/td_checkpoints/manifest.json` and deployed
  browser model packs under `public/model-packs/`.
- Local-only browser persistence (autosave, game history); no backend runtime.
- Static deploy target (GitHub Pages style hosting).

## Non-Goals

- Multiplayer, accounts, or backend services.
- Full cross-language rules schema or codegen; the TS/Python boundary stays a
  small interface contract.
- Native Python rules engine.
- Court/extended-rules support in TD training, encoding, or the Experimental
  browser profile (standard-only unless explicitly re-approved).

## Architecture

- **TS Engine (canonical)**: deterministic rules and state transitions, shared
  by browser and training.
- **UI (React/TS)**: thin layer over engine APIs with controller-owned
  presentation.
- **Node Bridge**: versioned NDJSON protocol exposing the engine to Python.
- **Python Trainer**: RL loop as a bridge client.
- **TS Bot Evaluation** (`src/botEval/`): browser-policy head-to-head,
  calibration, and replay-export tooling.
- **Browser Model Packs**: committed static packs load the deployed TD model
  with no server runtime.

## Success Criteria

- Rules behavior matches `memoryBank/magnateRules.md` for both rulesets.
- Deterministic replay from seed plus actions across engine, bridge, and saves.
- Stable bridge contract for TS/Python integration.
- Bot strength improves only through predeclared paired gates; promotions are
  registered in the checkpoint manifest and the deployed browser pack matches
  the registered model.
- Bot plays full games in the browser without a server runtime.

## Licensing

Project license is CC BY-NC-SA 4.0 (`LICENSE`; matching SPDX identifier in
`package.json`). Game credits appear in the info modal and `README.md`; bundled
software notices live in `public/third-party-notices.txt`.

## References

- Rules: `memoryBank/magnateRules.md`
- Contract: `memoryBank/bridgeInterfaceContract.md`
- Architecture: `memoryBank/systemPatterns.md`
- Tooling: `memoryBank/techContext.md`
- Active work: `memoryBank/activeContext.md`
