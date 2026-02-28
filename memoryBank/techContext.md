# Tech Context

## Stack

- Node.js 20+
- TypeScript (strict)
- React + Vite
- Vitest
- ESLint + Prettier
- Python 3.11+ (local `.venv`)
- PyTorch + NumPy for training

## Layout

- Engine + browser app in this repo (`src/`).
- Bridge runtime in `src/bridge/`.
- Python trainer/tooling in `trainer/` and `scripts/`.
- Versioned bridge contract in `contracts/`.

## Tooling Notes

- Package manager: Yarn
- Main JS scripts: `dev`, `build`, `bridge`, `test`, `lint`, `typecheck`, `format`
- Python environment bootstrap:
  - `scripts/setup_python_env.ps1`
  - activate: `.\.venv\Scripts\Activate.ps1`
- Python entrypoints run as modules:
  - `python -m scripts.eval`
  - `python -m scripts.benchmark`
  - `python -m scripts.train`
  - `python -m scripts.finetune`
  - `python -m scripts.train_ppo`
  - `python -m scripts.train_search_guidance`
  - queue helpers for PPO and benchmark sweeps

## Constraints

- Static deployment target (no gameplay backend).
- Deterministic gameplay required for replay/test/training.
- Rules semantics remain in TS unless explicitly re-approved.

## Known Gaps

- Guidance training path now exists, but checkpoint quality and tuning are not yet standardized.
- Search teacher is stronger than PPO, but still below target dominance; learned priors/value/opponent-model tuning remains in progress.
- Experiment tracking and promotion criteria are still lightweight/manual (despite canonical side-swapped eval-suite artifacts).
- Long-run training automation exists, but tuning strategy and reporting are not yet standardized.
