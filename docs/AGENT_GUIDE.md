# Memory Bank Guide

Use this guide with `AGENTS.md`.

## Core Files (always review)

- `memoryBank/projectBrief.md`: scope and goals.
- `memoryBank/systemPatterns.md`: architecture and key design constraints.
- `memoryBank/techContext.md`: stack, tooling, and environment constraints.
- `memoryBank/magnateRules.md`: rules source of truth.
- `memoryBank/bridgeInterfaceContract.md`: TS-Python contract.
- `memoryBank/activeContext.md`: current focus and immediate next steps.

## When To Update

Update Memory Bank files when:

1. Architecture decisions change.
2. Significant implementation milestones land.
3. The user asks to "update memory bank".
4. Current focus/next steps are no longer accurate.

## Update Standard

- Keep docs concise and decision-focused.
- Prefer durable facts over changelog noise.
- Avoid tracking volatile details (for example exact test counts).
- Keep operational runbook material in `docs/runbooks/`; keep `memoryBank/techContext.md` as tooling context and runbook index.
- Keep `memoryBank/activeContext.md` focused on current work, not release-note history.
- Keep `README.md`, `AGENTS.md`, and relevant Memory Bank files aligned in the same pass.
- Replace stale bullets instead of appending near-duplicates.
- Do not record task-completion logs, experiment blow-by-blow, or agent handoff chatter in Memory Bank files.
- Do not document volatile UI selections unless the user explicitly asks for that to be part of project direction.

## Document Ownership

- `README.md`: short project overview, quickstart, common commands, and links.
- `AGENTS.md`: agent startup checklist, durable decisions, and working rules.
- `memoryBank/projectBrief.md`: project goal, scope, non-goals, and success criteria.
- `memoryBank/systemPatterns.md`: stable architecture and implementation patterns.
- `memoryBank/techContext.md`: stack, tooling, command index, constraints, and runbook links.
- `memoryBank/magnateRules.md`: rules reference only.
- `memoryBank/bridgeInterfaceContract.md`: TS/Python bridge contract only.
- `memoryBank/activeContext.md`: current focus, current state, remaining work, and immediate next steps.
- `docs/runbooks/`: operational setup, training, evaluation, wrapper, and recovery procedures.

## Minimum End-of-Task Check

Before handoff, confirm:

- Current focus is accurate in `activeContext.md`.
- Implemented/remaining status is accurate in `activeContext.md`.
- Any changed contract/architecture is reflected in its source file.
