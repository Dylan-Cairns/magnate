# Progress
> Reminder: Start at `AGENTS.md` and review every core Memory Bank file before planning or acting.

## What Works

- Memory Bank and top-level docs are aligned to current architecture decisions.
- TypeScript engine foundations exist (`types`, `cards`, deterministic `deck`, early legality/reducer scaffolding).
- Rules references remain centralized in `memoryBank/magnateRules.md`.
- TS-canonical + bridge-client Python direction is now explicit and documented.
- Small interface contract strategy is documented in `memoryBank/bridgeInterfaceContract.md`.

## What's Left to Build

- Complete TS engine flow:
  - setup/deal
  - full phase machine (taxation/income/play/draw/end)
  - legality completeness
  - scoring and terminal resolution
- Add TS unit and snapshot tests.
- Implement bridge runtime with v1 contract.
- Build scripted baseline bot.
- Build minimal browser UI (human vs bot).
- Add Python trainer scaffolding that uses bridge contract.
- Add model export + browser inference integration.
- Add GitHub Pages deployment workflow.

## Current Status

- Up to date as of 2026-02-19.
- Project is in architecture-reset/implementation-readiness phase.
- No code/package changes were made in this pass; docs only.

## Known Issues

- Lint tooling mismatch: ESLint v9 with legacy `.eslintrc.json`.
- No TS tests currently present.
- Missing root TS/Vite config files.
- Python interpreter in this shell is 3.7.9, below planned training baseline.

## Evolution of Project Decisions

- Kept no-pass interpretation for play action flow.
- Kept 3:1 trade as one exchange per action, chainable.
- Locked TS engine as canonical rules source.
- Dropped plan for full shared cross-language rules schema.
- Adopted narrow TS<->Python bridge contract as the shared boundary.
- Deferred native Python rules implementation unless bridge throughput proves insufficient.

_Progress updated on 2026-02-19._