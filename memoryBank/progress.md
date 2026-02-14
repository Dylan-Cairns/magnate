# Progress
> Reminder: Start at `AGENTS.md` and review every core Memory Bank file before planning or acting.

## What Works

- Memory Bank is current and reflects design decisions.
- Engine scaffolding in place: strict types, normalized cards, deterministic deck.
- Docs aligned to rules: no Pass; Trade is one exchange per action.
- No code comments policy enforced; engine files cleaned.

## What's Left to Build

- TypeScript engine reducers, legality, scoring, turn FSM.
- `setupGame(seed)` to deal Crowns, starting resources, and hands.
- `legalActions(state)` skeleton with stable IDs and action union.
- React UI (human vs rules-based bot).
- Node bridge (stdin/stdout JSON API).
- Rules-based opponent (before any RL tools).
- Python trainer (RL loop, PPO, baselines).
- Static model export and browser inference.
- Testing: unit, snapshot, property tests.
- Deployment: static build for GitHub Pages.

## Current Status

- Up to date as of 2025-09-03.
- Types/cards/deck done; moving into legality and setup implementation.

## Known Issues

- None currently tracked.

## Evolution of Project Decisions

- Removed Pass action; Sell is mandatory when no develop/deed is legal.
- Trade is 3:1, one per action; repeat to chain.
- District markers are Pawn tri-suits plus the Excuse.
- Crowns persist with players and power 10-income; not in property deck.
- No code comments; record rationale in Memory Bank.

_Progress updated on 2025-09-03._
