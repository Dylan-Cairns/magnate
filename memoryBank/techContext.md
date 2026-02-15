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
- Python training commands run from a project-local virtualenv (`.venv`):
  - bootstrap script: `scripts/setup_python_env.ps1`
  - activate in PowerShell: `.\.venv\Scripts\Activate.ps1`
  - run script entrypoints as modules once activated (`python -m scripts.<name>`)
  - `python -m scripts.smoke_trainer`
  - `python -m scripts.eval`
  - `python -m scripts.benchmark`
  - `python -m scripts.benchmark_queue`
  - `python -m scripts.train`
  - `python -m scripts.finetune`
  - `python -m scripts.train_ppo_queue`
  - PPO uses PyTorch from `requirements.txt`
  - PPO scaffold entrypoint: `python -m scripts.train_ppo`

## Constraints

- Static deploy target (no gameplay backend).
- Deterministic gameplay required for tests, replay, and training.
- Rules logic remains in TS unless explicitly re-approved otherwise.

## Known Gaps

- Full rules-parity scenario coverage is not complete yet.
- Trainer supports BC warm-start, stabilized seeded RL fine-tuning, and a PPO scaffold path, but experiment automation/tuning depth is still limited.
- Model-backed trained-policy implementations are not complete yet (UI trained profiles are currently disabled placeholders).
- Browser/bridge wiring to drive games from `newGame(seed, { firstPlayer })` is not complete yet.
