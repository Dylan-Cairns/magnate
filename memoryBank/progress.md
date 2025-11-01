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
- Browser rollout-eval search policy path is wired into the UI bot catalog (T3 config) for direct in-browser play/testing.
- Search policy failure behavior in UI remains explicit (no silent fallback to another bot).
- Development-card progress ring fill now animates on upward progress only (including across remounts), with helper-level tests to protect canonical ratio/arc math.
- Bridge runtime and contract tests are in place.
- Trainer scaffold is in place for:
  - sample collection + BC warm-start
  - stabilized REINFORCE fine-tuning
  - PPO training
  - search and MCTS policy evaluation/benchmarking
  - queued multi-seed train/benchmark workflows
  - teacher-labeled data generation (`scripts/generate_teacher_data.py`) for distillation
- Guidance training path from teacher data is implemented:
  - `scripts/train_search_guidance.py`
  - `trainer/guidance_training.py`
  - guidance checkpoints reuse PPO checkpoint format (`magnate_ppo_policy_v1`)
- MCTS implementation quality pass landed:
  - root search now uses progressive widening instead of permanent hard top-K pruning
  - repeated serialized-state + action transitions are cached to reduce duplicate bridge steps
  - non-terminal MCTS leaf value is more score-aligned (district pressure and tiebreak structure)
- Search/MCTS guidance integration landed:
  - optional learned policy priors for search and MCTS
  - optional learned leaf value for MCTS depth cutoffs
  - optional learned opponent action modeling in search rollouts
  - new eval/benchmark/teacher-data CLI flags for guidance checkpoint injection
- Unattended A/B pipeline script landed:
  - `scripts/run_guidance_ab_pipeline.py` runs teacher-data -> guidance-train -> baseline eval -> guided eval sequentially with fail-fast behavior and manifest output
  - default eval policy A is now `search` (was `mcts`)
- Unattended rollout-search sweep script landed:
  - `scripts/search_teacher_sweep.py` runs side-swapped search-vs-opponent eval for named presets, then writes ranked JSON/Markdown summaries
- Training handoff documentation now exists at `docs/TRAINING_HANDOFF.md` with:
  - measured benchmark/eval snapshot
  - in-flight run context
  - objective next-step decision logic
  - restart-ready command playbook

## In Progress

- Guidance checkpoint quality tuning and holdout validation.
- Determining best teacher configuration: heuristic-guided search vs guidance-assisted search/MCTS.

## Remaining

- Add full student distillation training/evaluation path with promotion gates for browser deployment.
- Define and enforce model promotion criteria (teacher/champion gate).
- Improve experiment tracking/reporting for long-running sweeps.

## Risks / Watch Items

- Guidance can improve search quality but may increase simulation-time compute if overused in deep trees.
- Search policy strength is improving, but inference latency is high for UI-time play.
- PPO seed variance remains significant; single-run conclusions are unreliable.
- Rules parity is strong but still depends on scenario tests staying current as features evolve.

_Updated: 2026-02-28._
