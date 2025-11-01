# Progress

## Implemented

- Architecture is locked:
  - TS engine is canonical.
  - Python integrates through a small versioned bridge contract.
- Engine core is implemented and deterministic:
  - setup/deck lifecycle
  - legality + reducer transitions
  - phase resolver (`advanceToDecision`)
  - scoring + terminal resolution
- Rules edge-case coverage was expanded with targeted regression tests (tax/income sequencing, mixed-roll handling, income-choice queue order, ace cost paths, Excuse follow-on placement, final-turn countdown).
- Browser UI is playable with policy-agnostic controller boundaries and a bot profile catalog.
- Browser PPO inference path is wired, and a champion profile is available as default bot.
- Browser determinized search policy path is wired into the UI bot catalog (T2/T3 configs) for direct in-browser play/testing.
- Search policy failure behavior in UI remains explicit (no silent fallback to another bot).
- Bridge runtime and contract tests are in place.
- Trainer scaffold is in place for:
  - sample collection + BC warm-start
  - stabilized REINFORCE fine-tuning
  - PPO training
  - search-policy evaluation/benchmarking
  - queued multi-seed train/benchmark workflows
  - teacher-labeled data generation (`scripts/generate_teacher_data.py`) for distillation
- Training handoff documentation now exists at `docs/TRAINING_HANDOFF.md` with:
  - measured benchmark/eval snapshot
  - in-flight run context
  - objective next-step decision logic
  - restart-ready command playbook

## In Progress

- Search-teacher performance tuning and holdout validation.
- Determining next training stage: direct PPO tuning vs search-to-policy distillation.

## Remaining

- Add distillation training loop and evaluation path that consume teacher datasets.
- Define and enforce model promotion criteria (teacher/champion gate).
- Improve experiment tracking/reporting for long-running sweeps.

## Risks / Watch Items

- Search policy strength is improving, but inference latency is high for UI-time play.
- PPO seed variance remains significant; single-run conclusions are unreliable.
- Rules parity is strong but still depends on scenario tests staying current as features evolve.

_Updated: 2026-02-27._
