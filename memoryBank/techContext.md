# Tech Context

## Stack

- Node.js `22.23.1` (`.nvmrc`; `engines.node` allows `>=22.23.1 <23`)
- TypeScript strict, React + Vite, Vitest
- Python 3.12+ through the project `.venv`: Pytest, Ruff, Pyright, PyTorch + NumPy
- ESLint + Prettier

## Layout

- Engine + browser app: `src/`
- Browser policies: `src/policies/`
- TypeScript bot evaluation: `src/botEval/`
- Bridge runtime: `src/bridge/`
- Python trainer/tooling: `trainer/`, `scripts/`
- Trainer tests: `trainer_tests/`
- Bridge contract: `contracts/`
- Operational runbooks: `docs/runbooks/`

## Tooling Notes

- License is CC BY-NC-SA 4.0 (`LICENSE`). Card/glyph credits and third-party
  software notices live in `README.md`, the in-app info modal, and
  `public/third-party-notices.txt`.
- Playable card artwork is lossless WebP at 242 x 376 under
  `src/assets/decktet-card-art/`, named `decktet-card-<normalized-name>.webp`.
  The mapping and eager URL glob live in `src/ui/cardImages.ts`; startup
  preloads all playable cards. Court WebPs map to the extended IDs `"41"`-`"44"`.
- The Court rank symbol is `src/assets/icons/court.svg`, rendered by
  `src/ui/courtIcon.tsx` and used by `CardRank`; the Excuse keeps the `X`
  placeholder.
- Card facts are authored from the local Jacynth Decktet extraction rather than
  any third-party card catalog. Card IDs and `ALL_CARDS` ordering remain
  compatibility surfaces.
- Node version manager: fnm via the checked-in `.nvmrc`, including
  `-NoProfile` Windows wrappers. Package manager: Yarn 4.15.0 through Corepack;
  CI runs `yarn install --immutable`.
- JS scripts: `dev`, `build`, `bridge`, `bot:eval`, `test`, `lint`,
  `typecheck`, `format`.
- Vite dev watching excludes the local Python environment, generated artifacts,
  and local test/tool caches.
- GitHub Pages deploy (`.github/workflows/deploy_pages.yml`) reads the `.nvmrc`
  pin, activates the Yarn `packageManager` pin, and gates on `yarn test`,
  `yarn lint`, and `yarn build`.
- Checked-in pyright scope covers `trainer/` plus `trainer_tests/`; some
  `scripts/` orchestration remains outside it.
- TypeScript bridge output is canonical. Python models the consumed subset in
  `trainer/bridge_payloads.py`.
- Strategic-position diagnostics support `--positions`, `--variants`,
  `--repetitions`, and `--start-repetition`. The command does not resume or
  merge prior output, so targeted extensions need a separate output directory.

## Core Commands

- Install/select Node: `fnm install`, `fnm use`; activate Yarn:
  `corepack enable`, `corepack install`
- Install JS deps: `yarn install`
- Dev server: `yarn dev`; bridge runtime: `yarn bridge`
- JS test / lint+typecheck / format: `yarn test`, `yarn lint`, `yarn format`
- Python test: `.\.venv\Scripts\python -m pytest`
- Python lint: `python -m ruff check scripts trainer trainer_tests`
- Python typecheck: `.\.venv\Scripts\python -m pyright -p .`
- Register a checkpoint pair:
  `.\.venv\Scripts\python -m scripts.promote_td_checkpoint --help`
- Export a browser TD-root model pack:
  `.\.venv\Scripts\python -m scripts.export_browser_td_root_pack --value-checkpoint <value.pt> --opponent-checkpoint <opponent.pt> --set-default`
- Reconstruct trainer checkpoints from a browser pack:
  `.\.venv\Scripts\python -m scripts.reconstruct_browser_td_root_checkpoints --manifest <pack-manifest.json> --output-dir <dir>`

## Python Workflow

- Use the project `.venv` for any Python command in this repo.
- When changing Python code, run targeted pytest tests for touched behavior plus
  Ruff and Pyright before handoff.
- Note explicitly when a change touches Python outside checked-in pyright scope.

## Checkpoint Manifest

- `models/td_checkpoints/manifest.json` (schema v2) is the canonical
  source-controlled registry: `defaultWarmStart`, `opponentPool`, and
  `checkpoints.<key>.value` / `.opponent`.
- Referenced checkpoint files live under `models/td_checkpoints/<key>/` so they
  move with the repo; commit them when the manifest changes.
- Successful promotions in TD loop scripts copy accepted pairs into the
  registry and update the manifest unless `--disable-manifest-promotion` is set.

## Runbooks

- Windows local setup and laptop wrappers: `docs/runbooks/windows-local.md`
- RunPod/Linux CPU setup: `docs/runbooks/runpod-linux.md`
- Python training and evaluation loops: `docs/runbooks/training-loop.md`
- TypeScript browser-bot evaluation: `docs/runbooks/bot-eval.md`
- TD browser benchmarks and executor rollback:
  `docs/runbooks/td-browser-benchmarks.md`

## Constraints

- Static deployment target; no gameplay backend.
- Deterministic gameplay is required for replay, evaluation, and training.
- Rule semantics stay in TypeScript unless explicitly re-approved.
- Python training scripts are fail-fast and require the active project
  virtualenv; `scripts.train_td` enforces Python 3.12+ and active `.venv`.
- Replay path lists spanning multiple run directories must use
  `--replay-key-mode run-qualified-canonical-v1`; basename mode remains the
  default for single-run fingerprints.
- Augmentation modes (`--district-augmentation s4|s4-orbit`) require an explicit
  experiment seed and a matched control. Experimental browser packs use the
  ignored `public/model-packs-experiments/` index and never change the deployed
  default.

## Known Gaps

- Search baseline promotion thresholds still need repeated confirmation.
- Browser TD deployment uses the committed step-9,000 pack
  (`public/model-packs/td-hard-extra-data-primary-treatment-step-09000/`), the
  `defaultPackId` in `public/model-packs/index.json`.
- Direct TypeScript TD-root matchup throughput can still improve; each
  individual Node search decision remains synchronous.
