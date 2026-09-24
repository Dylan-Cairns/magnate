# Agent Workflow Guide

Use this guide with `AGENTS.md`. It covers how to read the project, how to
verify changes, and how to keep project docs current.

## Reading Map

Read the files for the surfaces you will touch; do not load everything up
front.

| Working on                                 | Read first                                                                                          |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| Engine rules, legality, scoring, turn flow | `memoryBank/magnateRules.md`, `memoryBank/systemPatterns.md` (Engine, Turn-Flow), `src/engine/`     |
| UI, controller, animations, persistence    | `memoryBank/systemPatterns.md` (Client Controller, UI Presentation, Persistence), `src/ui/`         |
| Bridge protocol or trainer client          | `memoryBank/bridgeInterfaceContract.md`, `contracts/`, `src/bridge/`, `trainer/bridge_client.py`    |
| Training and evaluation loops              | `docs/runbooks/training-loop.md`, `memoryBank/systemPatterns.md` (Training), `trainer/`, `scripts/` |
| Browser bot evaluation                     | `docs/runbooks/bot-eval.md`, `src/botEval/`                                                         |
| Browser TD inference and workers           | `docs/runbooks/td-browser-benchmarks.md`                                                            |
| Heuristics and policies                    | `memoryBank/systemPatterns.md` (Heuristic Scoring), `src/policies/`                                 |
| District symmetry and TD architecture      | `docs/design/district-symmetry.md`                                                                  |
| Environment and setup                      | `docs/runbooks/windows-local.md` or `docs/runbooks/runpod-linux.md`, `memoryBank/techContext.md`    |
| Current focus and next steps               | `memoryBank/activeContext.md`                                                                       |
| Experiment designs and gates               | `docs/design/`                                                                                      |

`memoryBank/activeContext.md` is the only always-read status file.

## Verification Contract

Run the narrowest relevant check before handoff, then report what you ran.

TypeScript (canonical engine, UI, bot evaluation):

- Focused test first: `yarn vitest run <test-file-or-pattern>`
- Full gate: `yarn test`
- Lint and typecheck: `yarn lint`
- Formatting: `yarn format` (Prettier), or `npx prettier --check <files>` to
  check only what you touched.

Python (training stack):

- Use the project `.venv` for every command (`.\.venv\Scripts\python` on
  Windows, `.venv/bin/python` on Linux/macOS).
- Targeted tests for touched behavior:
  `.\.venv\Scripts\python -m pytest trainer_tests/<test-file>.py`
- Ruff: `.\.venv\Scripts\python -m ruff check scripts trainer trainer_tests`
- Pyright: `.\.venv\Scripts\python -m pyright -p .` (scope is `trainer/` plus
  `trainer_tests/`, excluding `trainer_tests/test_eval_suite*.py`; `scripts/`
  orchestration is outside it).

Bridge changes: run the bridge contract tests, keep action IDs and action keys
stable, and follow the versioning rules in
`memoryBank/bridgeInterfaceContract.md`. Python clients must keep draining
bridge stderr.

Fail-fast expectations: invalid payloads, missing checkpoints, malformed policy
probabilities, and missing explicit policy arguments are hard errors. Do not add
silent fallbacks.

CI runs `yarn test`, `yarn lint`, and `yarn build` on pushes to `main`
(`.github/workflows/deploy_pages.yml`). Python checks are local-only; run them
yourself when Python changes.

## Planning And Scope

- Non-trivial work starts with a short plan: goal, files to touch, verification,
  risks. Plans are session artifacts; durable decisions belong in
  `docs/design/` or `memoryBank/`.
- Keep diffs minimal and scoped to the request. Avoid unrelated refactors and
  broad reformatting.
- Ask before choosing between interpretations that would produce materially
  different changes.
- Determinism is a hard requirement: seeded RNG only, no hidden global state,
  and no changes to canonical action ordering without an explicit decision.

## Doc Maintenance

When to update project docs:

1. Architecture or bridge decisions change.
2. Significant implementation milestones land.
3. The user asks to update the docs (for example "update memory bank").
4. Current focus or next steps are no longer accurate.

Update standard:

- Keep docs concise and decision-focused; prefer durable facts over changelog
  noise.
- Avoid volatile details (for example exact test counts).
- Promote detail to its home: experiment blow-by-blow to `docs/design/`,
  operational procedures to `docs/runbooks/`, bridge specifics to
  `memoryBank/bridgeInterfaceContract.md`.
- Keep `README.md`, `AGENTS.md`, and affected Memory Bank files aligned in the
  same pass.
- Replace stale bullets instead of appending near-duplicates; delete obsolete
  docs rather than leaving stubs (git history preserves prior detail).
- Do not record task-completion logs or agent handoff chatter.

End-of-task check:

- Current focus and status in `activeContext.md` are accurate and link out for
  detail.
- `AGENTS.md` and `README.md` still match behavior you changed.
- Contract or architecture changes are reflected in their source file.
- You can state what verification you ran, or why none applied.

## Document Ownership

- `README.md`: human overview, quickstart, common commands, links.
- `AGENTS.md`: agent entry point: startup checklist, locked decisions,
  verification, and task routing.
- `docs/AGENT_GUIDE.md`: this guide.
- `memoryBank/projectBrief.md`: goal, scope, non-goals, success criteria.
- `memoryBank/systemPatterns.md`: stable architecture and implementation
  patterns.
- `memoryBank/techContext.md`: stack, tooling, command index, constraints,
  runbook links.
- `memoryBank/magnateRules.md`: rules reference only.
- `memoryBank/bridgeInterfaceContract.md`: TS/Python bridge contract only.
- `memoryBank/activeContext.md`: current focus, state, remaining work, and
  immediate next steps.
- `docs/design/`: experiment designs, predeclared gates, and design notes.
- `docs/runbooks/`: operational procedures.
