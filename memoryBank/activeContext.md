# Active Context
> Reminder: Start at `AGENTS.md` and review every core Memory Bank file before planning or acting.

## Current Focus

- Core TypeScript engine scaffolding: types, cards, deterministic deck.
- Finalize action spec and legality (no pass; single-trade actions).
- Implement setupGame and legalActions skeleton.

## Recent Changes

- Added engine types, suits/ranks, and strict card kinds.
- Adapted provided card list; split into Property/Crown/Pawn/Excuse.
- Implemented deterministic deck with seeded shuffle and reshuffle.
- Removed Pass action; Trade is one exchange per action (repeat to chain).
- Adopted “no code comments” policy; removed comments from engine files.
- Updated docs (projectBrief, systemPatterns, README) to reflect the above.

## Next Steps

- Define Action union and `legalActions(state)` skeleton with stable IDs.
- Implement placement validators (Pawn tri-suits, Excuse rules) and turn FSM.
- Implement `setupGame(seed)` to deal Crowns, starting resources, and hands.
- Add unit/snapshot tests for deck, setup, and legality basics.

## Active Decisions & Considerations

- No code comments; express intent with names/types/tests; rationale in Memory Bank.
- No Pass; if development/deed is impossible, Sell is mandatory.
- Trade is 3:1, one per action (repeatable).
- Crowns persist with players; properties deck excludes Crowns/Pawns/Excuse.
- District markers are Pawn tri-suits plus the Excuse.
- Deterministic PRNG (seeded sfc32) for shuffle/reshuffle.

## Insights & Learnings

- Keep rule interpretations explicit in Memory Bank to guide engine purity.
- Deterministic helpers greatly simplify snapshot tests and RL reproducibility.

_Active context updated on 2025-09-03._
