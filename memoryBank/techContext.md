# Tech Context

## Technologies and Frameworks

- **Rules runtime**: TypeScript (strict), Node.js 20+
- **Frontend**: React 19 + Vite
- **Testing (TS)**: Vitest
- **Lint/format**: ESLint + Prettier
- **Training runtime**: Python 3.11+ (3.12 preferred)
- **RL stack target**: PyTorch, Stable-Baselines3, Gymnasium, NumPy
- **Bridge transport**: Node stdin/stdout JSON protocol

## Baseline policy

Use `kuhn-poker` as the workflow/template baseline for:

- repo shape and separation between TS runtime and Python runtime
- deterministic testing discipline
- static deployment pattern
- model export + browser inference integration approach

Magnate diverges where game complexity requires it, but should not diverge without explicit justification.

## Development setup

- **Package manager (current)**: Yarn
- **Node target**: v20+
- **Python target for training tooling**: 3.11+
- **Core scripts (current package.json)**: `dev`, `build`, `preview`, `test`, `lint`, `format`

## Technical constraints

- Static hosting on GitHub Pages (no gameplay backend)
- Deterministic engine and replayability
- TS engine remains canonical rules source
- Python trainer consumes bridge outputs; no duplicated rules engine by default
- Shared contract scope stays narrow (bridge + observation/action/model metadata)

## Current known gaps (2026-02-19)

- ESLint is on v9 while config is still legacy `.eslintrc.json` format.
- No `tsconfig.json` yet in repo root.
- No `vite.config.*` yet in repo root.
- No TS test files currently present.
- Python environment in this shell is currently 3.7.9, below the target for RL stack work.

## Planned config direction

- Add strict `tsconfig.json` and explicit Vite config (Kuhn-style baseline).
- Migrate ESLint config to flat config format or pin compatible ESLint strategy.
- Add bridge contract tests and engine unit/snapshot tests before trainer work.
- Add Python project config (`pyproject.toml`) when bridge client/trainer scaffolding starts.

## Contract reference

- `memoryBank/bridgeInterfaceContract.md` is the TS<->Python boundary source of truth.