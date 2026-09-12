# Agent Manifest

Start every session here.

## Session Checklist

1. Read this file and `memoryBank/activeContext.md`.
2. Read the docs for the surfaces you will touch; routing lives in
   `docs/AGENT_GUIDE.md`.
3. Create a plan for any non-trivial task and keep changes scoped.
4. Update project docs when context or decisions change.
5. End with a concise summary: what changed, what verification ran, remaining
   risk.

Detailed workflow: `docs/AGENT_GUIDE.md`

## Locked Decisions

- (2026-02-19) TypeScript engine is the canonical rules implementation.
- (2026-02-19) Python training calls TS through a stable Node bridge; the
  shared boundary is a small interface contract, not a full cross-language
  rules schema.
- (2026-02-19) Native Python rules are out of scope unless throughput becomes a
  real bottleneck.
- (2026-09-15) The extended ruleset (four Courts) is supported in browser play
  and the heuristic/search profiles; TD training, encoding, and the
  Experimental profile stay standard-only unless explicitly re-approved.
- (2026-07-24) Parallel browser TD search uses the paired lockstep executor by
  default; `?tdSearchExecutor=legacy` is the session rollback, and invalid
  values are hard errors.

## Working Rules

- Keep engine behavior deterministic (seeded RNG only).
- Keep rule semantics in TypeScript; UI and Python consume engine legality and
  observations, never re-derive rules.
- Keep the bridge contract stable (`memoryBank/bridgeInterfaceContract.md`);
  breaking changes require a contract version bump.
- Use the project `.venv` for any Python command in this repo. Training and
  evaluation are fail-fast: invalid payloads, missing checkpoints, or malformed
  policy probabilities are hard errors, never silent fallbacks.
- Verify before handoff:
  - TypeScript changes: focused `yarn vitest run <pattern>`, then `yarn test`
    and `yarn lint`.
  - Python changes: targeted pytest for touched behavior, then Ruff and
    Pyright.
  - Bridge changes: contract tests plus stable action IDs and keys.
- Default to **user verification** for visual/UI changes: make the change, run
  the non-visual checks (lint, typecheck, tests, build), then hand off for the
  user to review in the running app. Do not self-verify UI with the
  Playwright/CDP harness unless the user explicitly asks you to iterate without
  their feedback on that task.
- Scope to what was asked. Do not convert your own coverage gaps, hypotheses, or
  "worth confirming" items into tasks. If something seems worth investigating
  beyond the request, ask first and let the user decide. If you notice a
  suspected defect incidentally, report it — investigating it is the user's call,
  not yours. If an automated run does not reach its goal on the first attempt,
  stop and report rather than iterating unprompted.
- Do not build a bespoke harness for a one-off task. Write one short throwaway
  script, run it, delete it; only generalize after a third use.
- Run focused tests while iterating; run the full suite and build once per
  change-set (per "Verify before handoff" above). Skip checks the change cannot
  affect — CSS-only changes do not need vitest.
- Set up browser tooling at most once per session, at a sensible window size,
  and tear it down at the end. Prefer handing off to the user for visual review.
- Batch visual changes and review them in one pass rather than re-verifying
  after each edit.
- Prefer promoted checkpoints as warm start; register promotions through
  `models/td_checkpoints/manifest.json`.
- Keep docs aligned with code changes; replace stale docs instead of appending
  history; delete obsolete documentation rather than leaving stubs.

## Project Context

- Browser play supports the standard and extended rulesets with four profiles:
  Easy/Medium/Hard (`rollout-search-v2-*`, both rulesets) and Experimental
  (`td-root-search-v2-medium`, standard only). Autosave and local game history
  are browser-local; there is no gameplay backend.
- Training is TD-focused and runs `collect -> train -> promotion eval`;
  bootstrap/recalibration uses `python -m scripts.run_td_loop`, ongoing
  self-play uses `python -m scripts.run_td_loop_selfplay`.
- The Python policy surface is `random`, `heuristic`, `search`, `td-value`,
  `td-search`.
- The deployed browser model pack is selected by
  `public/model-packs/index.json`.

## Reference Map

- Goal and scope: `memoryBank/projectBrief.md`
- Architecture and patterns: `memoryBank/systemPatterns.md`
- Tooling, commands, and runbook links: `memoryBank/techContext.md`
- Rules reference: `memoryBank/magnateRules.md`
- Bridge contract: `memoryBank/bridgeInterfaceContract.md`
- Current focus and next steps: `memoryBank/activeContext.md`
- Agent workflow, verification, and doc ownership: `docs/AGENT_GUIDE.md`
- Runbooks: `docs/runbooks/`
- Design notes and experiment gates: `docs/design/`
