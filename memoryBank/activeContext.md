# Active Context
> Reminder: Start at `AGENTS.md` and review every core Memory Bank file before planning or acting.

## Current Focus

- Finish TS-canonical engine completion (turn FSM, taxation/income, scoring, terminal flow).
- Expand engine tests from low-level action coverage into turn-flow and scoring scenarios.
- Use the bridge interface contract as the integration boundary for upcoming bridge/trainer work.

## Recent Changes

- Confirmed architecture direction:
  - TypeScript engine is canonical for Magnate rules.
  - Python trainer will call TS engine through a stable bridge.
  - Full shared rules schema was dropped in favor of a small interface contract.
- Added `memoryBank/bridgeInterfaceContract.md`.
- Updated `README.md`, `AGENTS.md`, and Memory Bank docs to align with these decisions.
- Removed stale docs language implying pass actions or outdated architecture assumptions.
- Aligned frontend/tooling versions closer to Kuhn baseline in `package.json`.
- Added missing root config files:
  - `eslint.config.js` (flat config)
  - `tsconfig.json`
  - `vite.config.ts`
- Removed legacy `.eslintrc.json`.
- Refactored deck implementation into focused modules:
  - `src/engine/rng.ts` (seeded PRNG + shuffle)
  - `src/engine/deckCore.ts` (deck primitives)
  - `src/engine/setup.ts` (setup/deal/resources)
  - `src/engine/drawPolicy.ts` (draw/exhaustion policy)
  - `src/engine/deck.ts` kept as a compatibility facade
- Hardened current low-level engine behavior:
  - `applyAction` now enforces phase/action legality via `legalActions(state)`.
  - Action handlers now validate card ownership, district existence, placement legality, and resource affordability.
  - Buying a deed now transitions to `OptionalDevelop` instead of immediately forcing `DrawCard`.
  - Excuse district first-placement rule is now handled in `placementAllowed`.
  - Deed development now prevents overspend and requires exact completion to finish a deed.
  - Setup now shuffles/deals crowns, deals opening hands, and computes starting resources by crown suit.
  - Draw helper now uses seed + RNG cursor for reshuffles and tracks second-exhaustion final-turn state.
- Added phase resolver scaffolding via `src/engine/turnFlow.ts`:
  - `advanceToDecision(state)` auto-advances non-decision phases.
  - Start-turn chain now resolves deterministic taxation/income and advances `StartTurn -> TaxCheck -> IncomeRoll -> CollectIncome -> OptionalTrade`.
  - `DrawCard` now resolves draw/exhaustion, hands off active player, increments turn, and applies final-turn countdown to `GameOver`.
- Added comprehensive TS engine unit tests (61 passing) covering:
  - setup/deck draw behavior and exhaustion markers
  - placement rules (including Excuse behavior)
  - legal action generation across trade/develop/play phases
  - reducer legality gate and low-level action semantics (buy/develop/sell)
  - turn-flow auto-advance, draw/end-turn handling, deterministic taxation/income
  - direct regressions for issues 2, 3, 4, 5, 6, 7, and 9.

## Next Steps

- Finalize bridge command payloads and metadata fields for v1.
- Implement missing TS engine rule flow:
  - setup game
  - remaining turn FSM details beyond current tax/income baseline (optional-phase exits, full action loop wiring)
  - full legality coverage
  - scoring and terminal logic
- Add targeted tests for taxation/income behavior and scoring.
- Expand higher-level turn-flow fixtures beyond phase scaffolding.
- Scaffold bridge runtime and validate contract with a Python client smoke test.
- Add minimal web app entry scaffolding (`index.html` + app entrypoint) so Vite build succeeds.

## Active Decisions and Considerations

- Keep all game-rule semantics in TS for v1.
- Python remains a consumer of engine outputs, not a second rules engine.
- Preserve deterministic behavior end to end (seed + action log replay).
- No pass action; if no legal develop/deed is available, selling is mandatory.
- Trade remains 3:1 one exchange per action and can be chained.
- Keep tooling versions close to Kuhn baseline unless a clear Magnate-specific reason exists.

## Insights and Learnings

- Magnate complexity increases drift risk; a single canonical rules runtime is the safest path.
- A narrow bridge contract yields most of the cross-language stability benefit with much less maintenance cost than full rules schema duplication.
- Keeping docs synchronized early prevents architecture confusion during implementation.

_Active context updated on 2026-02-19._
