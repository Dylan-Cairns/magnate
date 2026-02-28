# Tech Context

## Stack

- Node.js 20+
- TypeScript (strict)
- React + Vite
- Vitest
- ESLint + Prettier
- Python 3.11+ (`.venv`)
- PyTorch + NumPy

## Layout

- Engine + browser app: `src/`
- Bridge runtime: `src/bridge/`
- Python trainer/tooling: `trainer/`, `scripts/`
- Bridge contract: `contracts/`

## Tooling Notes

- Package manager: Yarn
- JS scripts: `dev`, `build`, `bridge`, `test`, `lint`, `typecheck`, `format`
- Python bootstrap:
  - `scripts/setup_python_env.ps1`
  - `.\.venv\Scripts\Activate.ps1`
- Active Python entrypoints:
  - `python -m scripts.eval`
  - `python -m scripts.eval_suite`
  - `python -m scripts.search_teacher_sweep`
  - `python -m scripts.generate_teacher_data`
  - `python -m scripts.train_search_guidance`
  - `python -m scripts.run_guidance_ab_pipeline`
  - `python -m scripts.train_ppo` / `python -m scripts.train_ppo_queue` (kept for PPO-format model workflow)

## Constraints

- Static deployment target (no gameplay backend).
- Deterministic gameplay required for replay/eval/training.
- Rule semantics stay in TS unless explicitly re-approved.

## Known Gaps

- Promotion thresholds still need repeated high-sample confirmation.
- Search strength vs latency tuning remains active.
- Student distillation path is not yet productionized.

_Updated: 2026-03-01._
