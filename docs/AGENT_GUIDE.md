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
- Keep `README.md`, `AGENTS.md`, and relevant Memory Bank files aligned in the same pass.

## Minimum End-of-Task Check

Before handoff, confirm:

- Current focus is accurate in `activeContext.md`.
- Implemented/remaining status is accurate in `activeContext.md`.
- Any changed contract/architecture is reflected in its source file.
