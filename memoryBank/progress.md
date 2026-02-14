# Progress
> Reminder: Start at `AGENTS.md` and review every core Memory Bank file before planning or acting.

## What Works

- Memory Bank and top-level docs are aligned to current architecture decisions.
- TypeScript engine foundations exist (`types`, `cards`, deterministic `deck`, early legality/reducer scaffolding).
- Rules references remain centralized in `memoryBank/magnateRules.md`.
- TS-canonical + bridge-client Python direction is now explicit and documented.
- Small interface contract strategy is documented in `memoryBank/bridgeInterfaceContract.md`.
- Tooling baseline has been modernized and aligned closer to Kuhn:
  - React/Vite/Vitest versions adjusted
  - ESLint flat config in place
  - root TS/Vite config files added
  - lint/test commands now execute successfully
- Low-level rule enforcement and setup/draw helpers are substantially improved:
  - reducer now rejects illegal actions against `legalActions(state)`
  - deed/develop/outright/sell handlers validate ownership, placement, suits, and spend
  - deed purchase now allows same-turn partial development path
  - Excuse district first-card placement rule is respected
  - setup deals crowns/hands and computes starting resources
  - reshuffle RNG now uses seed + cursor and draw helper marks second-exhaustion final turns

## What's Left to Build

- Complete TS engine flow:
  - setup/deal
  - full phase machine (taxation/income/play/draw/end)
  - legality completeness
  - scoring and terminal resolution
- Add TS unit and snapshot tests.
- Implement bridge runtime with v1 contract.
- Build scripted baseline bot.
- Build minimal browser UI (human vs bot).
- Add Python trainer scaffolding that uses bridge contract.
- Add model export + browser inference integration.
- Add GitHub Pages deployment workflow.
- Add minimal web app entry files required for `vite build`.

## Current Status

- Up to date as of 2026-02-19.
- Project is in architecture-reset plus tooling-baseline phase.
- Package/config changes have been made; engine gameplay code was not changed in this pass.

## Known Issues

- No TS tests currently present.
- Vite build fails due missing `index.html` entrypoint.
- Python interpreter in this shell is 3.7.9, below planned training baseline.

## Evolution of Project Decisions

- Kept no-pass interpretation for play action flow.
- Kept 3:1 trade as one exchange per action, chainable.
- Locked TS engine as canonical rules source.
- Dropped plan for full shared cross-language rules schema.
- Adopted narrow TS<->Python bridge contract as the shared boundary.
- Deferred native Python rules implementation unless bridge throughput proves insufficient.

_Progress updated on 2026-02-19._
