# Progress
> Reminder: Start at `AGENTS.md` and review every core Memory Bank file before planning or acting.

## What Works

- Memory Bank and top-level docs are aligned to current architecture decisions.
- TypeScript engine foundations exist (`types`, `cards`, deterministic deck flow, legality/reducer scaffolding).
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
- Deck module refactor completed with compatibility preserved:
  - `src/engine/rng.ts`
  - `src/engine/deckCore.ts`
  - `src/engine/setup.ts`
  - `src/engine/drawPolicy.ts`
  - `src/engine/deck.ts` now re-exports and adapts legacy imports
- Turn-flow scaffolding now exists:
  - `src/engine/turnFlow.ts` with `advanceToDecision(state)`
  - auto-advances non-decision phases through the start-turn chain
  - resolves `DrawCard` into draw + end-turn handoff + final-turn countdown
  - resolves deterministic taxation/income baseline:
    - TaxCheck rolls d10/d10 and applies one taxation event on 1-trigger with d6 suit mapping
    - CollectIncome pays crowns on 10, rank matches on 2-9, and ace income on double ones
    - deed income currently uses deterministic first-suit selection until explicit choice flow is added
- New engine unit test suite is in place:
  - `src/engine/deck.test.ts`
  - `src/engine/stateHelpers.test.ts`
  - `src/engine/actionBuilders.test.ts`
  - `src/engine/reducer.test.ts`
  - `src/engine/turnFlow.test.ts`
  - Current result: 61 tests passing.

## What's Left to Build

- Complete TS engine flow:
  - setup/deal
  - remaining turn-loop wiring around optional-phase exits and action progression
  - legality completeness
  - scoring and terminal resolution
- Expand test coverage into higher-level turn-flow, taxation/income, and scoring fixtures.
- Implement bridge runtime with v1 contract.
- Build scripted baseline bot.
- Build minimal browser UI (human vs bot).
- Add Python trainer scaffolding that uses bridge contract.
- Add model export + browser inference integration.
- Add GitHub Pages deployment workflow.
- Add minimal web app entry files required for `vite build`.

## Current Status

- Up to date as of 2026-02-19.
- Project is in engine-hardening and rules-loop implementation phase.
- Tooling baseline and low-level engine correctness work are complete enough to proceed with higher-level rule flow implementation.

## Known Issues

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
