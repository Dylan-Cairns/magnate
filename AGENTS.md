# Agent Manifest

Start every session here.

## Session Checklist

1. Read this file.
2. Read all core Memory Bank files in `memoryBank/`.
3. Create and maintain a plan for any non-trivial task.
4. Update Memory Bank files when project context or decisions change.
5. End with a concise summary and concrete next steps.

Detailed workflow: `docs/AGENT_GUIDE.md`

## Locked Decisions (2026-02-19)

- TypeScript engine is the canonical rules implementation.
- Python training calls TS through a stable Node bridge.
- Shared boundary is a small interface contract, not a full cross-language rules schema.
- Native Python rules are out of scope unless throughput becomes a real bottleneck.

## Working Rules

- Keep engine behavior deterministic (seeded RNG only).
- Keep rule semantics in TypeScript.
- Keep bridge contract stable (`memoryBank/bridgeInterfaceContract.md`).
- Prefer targeted tests for touched behavior.
- Keep docs aligned with code changes.

## Reference Map

- Overview: `README.md`
- Memory Bank: `memoryBank/`
- Rules reference: `memoryBank/magnateRules.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
