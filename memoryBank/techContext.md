# Tech Context

## Technologies & Frameworks

- **Languages/Runtimes**: TypeScript (strict), Node.js (LTS), Python 3.11+
- **Frontend**: React 18, Vite
- **Styling**: Tailwind CSS, PostCSS, Autoprefixer
- **State & Engine**: Pure TS reducers (no side effects), seeded PRNG
- **Testing**: Vitest (unit + snapshots)
- **RL Training (Python)**: PyTorch, Stable-Baselines3 (PPO), Gymnasium, NumPy
- **Bridge**: Node child process (stdin/stdout JSON) — built-in to Node
- **Tooling**: Yarn, ESLint, Prettier

## Development Setup

- **Package manager**: Yarn (node-linker: `node-modules`)
- **Versions**: Node LTS; Python 3.11+
- **Scripts**: `dev`, `build` (static), `test`, `lint`, `format`
- **Vite**: configure `base` for GitHub Pages subpath

## Technical Constraints

- **Static hosting on GitHub Pages** (no backend services)
- **Single-thread execution** for training and inference
- **Strict TypeScript** (no `any`, discriminated unions for actions)
- **Deterministic RNG** (seeded; reproducible runs)
- **Model format**: static `assets/model.json` (weights + layer spec); TypeScript forward pass
- **No external APIs** (all assets loaded from the site)

## Dependencies & Tool Configurations

- **TypeScript**: strict mode; path aliases via `tsconfig.json`
- **ESLint**: `@typescript-eslint/parser`, `@typescript-eslint/eslint-plugin`, `eslint-config-prettier`, `eslint-plugin-import` (TS resolver)
- **Prettier**: project-wide formatting
- **Vite**: React plugin; `base` set for GH Pages
- **Tailwind**: `tailwind.config.js`
- **PostCSS**: `postcss.config.js` with **Autoprefixer**
- **Browserslist**: targets for Autoprefixer in `package.json` (optional)
- **Vitest**: jsdom for UI tests if needed; snapshot testing enabled
- **Python env**: `torch`, `stable-baselines3`, `gymnasium`, `numpy` (pin versions)
