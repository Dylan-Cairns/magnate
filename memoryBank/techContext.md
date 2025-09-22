# Tech Context

## Stack

- Node.js 20+
- TypeScript (strict)
- React + Vite
- Vitest
- ESLint + Prettier
- Python 3.11+ target for training

## Project Layout Direction

- TS engine and browser app in this repo.
- Python trainer as a bridge client.
- Shared cross-runtime contract is small and versioned.

## Tooling Notes

- Package manager: Yarn
- Primary scripts: `dev`, `build`, `test`, `lint`, `typecheck`, `format`
- `lint` now includes `tsc --noEmit` via `yarn typecheck`.
- Vite base uses relative paths for static hosting compatibility.

## Constraints

- Static deploy target (no gameplay backend).
- Deterministic gameplay required for tests, replay, and training.
- Rules logic remains in TS unless explicitly re-approved otherwise.

## Known Gaps

- Full rules-parity scenario coverage is not complete yet.
- Bridge runtime and trainer scaffolding are not complete yet.
- Web app entry/deploy flow is not complete yet.
- Browser/bridge wiring to drive games from `newGame(seed, { firstPlayer })` is not complete yet.
