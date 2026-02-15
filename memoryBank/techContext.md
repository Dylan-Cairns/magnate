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
- Python trainer scaffold now exists in-repo as a bridge client (`trainer/` + `scripts/*.py`).
- Shared cross-runtime contract is small and versioned.
- Browser app entry now exists at `index.html` + `src/main.tsx` with gameplay shell in `src/App.tsx`.

## Tooling Notes

- Package manager: Yarn
- Primary scripts: `dev`, `build`, `bridge`, `test`, `lint`, `typecheck`, `format`
- `lint` now includes `tsc --noEmit` via `yarn typecheck`.
- Vite base uses relative paths for static hosting compatibility.
- Python scripts are run with `py -3.12` in this workspace:
  - `scripts/smoke_trainer.py`
  - `scripts/eval.py`
  - `scripts/train.py`

## Constraints

- Static deploy target (no gameplay backend).
- Deterministic gameplay required for tests, replay, and training.
- Rules logic remains in TS unless explicitly re-approved otherwise.

## Known Gaps

- Full rules-parity scenario coverage is not complete yet.
- Trainer scaffold exists but does not yet implement a model optimization loop/checkpointing.
- Model-backed trained-policy implementations are not complete yet (UI trained profiles currently use explicit random fallback).
- Browser/bridge wiring to drive games from `newGame(seed, { firstPlayer })` is not complete yet.
