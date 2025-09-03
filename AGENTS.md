# Agent Manifest

This is the landing pad for GPT-5 Codex agents working in this repo. Start every session here, then dive into the Memory Bank for full context.

## Core Ritual

1. Read this manifest.
2. Review all core Memory Bank files (`memoryBank/`) before planning or executing.
3. Use the plan tool for any task beyond a trivial edit; keep the plan current.
4. Confirm whether the Memory Bank needs an update when work is done.
5. Hand off with a concise summary and next-step suggestions.

See `docs/AGENT_GUIDE.md` for the detailed Memory Bank procedures.

## Locked Architecture Decisions (2026-02-19)

- TypeScript engine is the canonical rules implementation.
- Python training calls the TS engine through a stable Node bridge.
- Use a small shared interface contract for bridge I/O, action IDs, observation layout, and model I/O names.
- Do not duplicate full Magnate rules in Python unless bridge throughput becomes a real bottleneck.

## Agent Roster

- **Builder**
  - Implements code, tests, and docs.
  - Maintains determinism and type safety.
  - Runs available checks (`yarn test`, `yarn lint`) when changes warrant.
- **Reviewer**
  - Performs code reviews and risk assessments.
  - Highlights bugs, regressions, and missing coverage first.
  - Suggests follow-up actions when approval is blocked.
- **Memory Steward**
  - Keeps the Memory Bank synchronized with reality.
  - Triggers "update memory bank" sweeps when significant context shifts.
  - Ensures insights live in docs, not inline code comments.

Any agent can wear multiple hats in a session; call out the role you are assuming when it matters.

## Working Agreements

- **Planning**: Declare a plan before touching files (unless the task is truly trivial). Update plans after each major sub-task.
- **Context Refresh**: When the user says "update memory bank", review every file under `memoryBank/` and revise as needed.
- **Determinism**: Never introduce nondeterministic behavior into the engine or training loop.
- **Canonical Rules**: Keep rules logic in TypeScript. Python is a bridge client unless the team explicitly approves a native Python rules engine.
- **Bridge Contract**: Keep the bridge interface stable and documented in `memoryBank/bridgeInterfaceContract.md`.
- **Testing**: Prefer running targeted tests that cover touched areas; note any gaps in the final message if tests were skipped.
- **Documentation**: Keep README, this manifest, and the Memory Bank aligned after changes.

## Reference Map

- Quick project overview: `README.md`
- Memory Bank index: `memoryBank/`
- Detailed Memory workflows: `docs/AGENT_GUIDE.md`
- Official Magnate rules: `memoryBank/magnateRules.md`
- Bridge interface contract: `memoryBank/bridgeInterfaceContract.md`

Stay disciplined, keep context fresh, and document decisions.
