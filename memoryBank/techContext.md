# Tech Context

## Technologies and Frameworks

- **Rules runtime**: TypeScript (strict), Node.js 20+
- **Frontend**: React 19.2.x + Vite 5.4.x
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
- **Core scripts (current package.json)**: `dev`, `build`, `preview`, `test`, `test:watch`, `lint`, `format`
- **TypeScript config**: root `tsconfig.json` (strict, bundler module resolution, React JSX)
- **Vite config**: root `vite.config.ts` with `base: "./"` for static-host compatibility
- **ESLint config**: root `eslint.config.js` (flat config)

## Technical constraints

- Static hosting on GitHub Pages (no gameplay backend)
- Deterministic engine and replayability
- TS engine remains canonical rules source
- Python trainer consumes bridge outputs; no duplicated rules engine by default
- Shared contract scope stays narrow (bridge + observation/action/model metadata)

## Current known gaps (2026-02-19)

- No TS test files currently present.
- Vite build currently fails because no `index.html`/app entrypoint exists yet.
- Python environment in this shell is currently 3.7.9, below the target for RL stack work.

## Planned config direction

- Keep frontend/runtime versions close to Kuhn's proven baseline unless a justified divergence is needed.
- Add bridge contract tests and engine unit/snapshot tests before trainer work.
- Add Python project config (`pyproject.toml`) when bridge client/trainer scaffolding starts.

## Contract reference

- `memoryBank/bridgeInterfaceContract.md` is the TS<->Python boundary source of truth.
