# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Continue the self-play-focused TD loop using the current `12`-chunk promotion cadence.
- Use `scripts.run_td_loop` for bootstrap/recalibration passes, then advance primarily with `scripts.run_td_loop_selfplay`.
- Improve model quality against search baseline and incumbent td-search while keeping runtime practical.

## Locked Decisions

- TS engine is canonical gameplay truth.
- Python training/eval calls TS through the bridge contract (`contracts/magnate_bridge.v1.json`).
- Training/eval scripts are fail-fast (no silent fallback labels/actions/probabilities).
- Default promotion decision uses pooled side-swapped certify eval windows (`200` games/side per window).

## Current State

- Browser app is playable; default bot is `TD Search Fast`.
- Bridge runtime command surface is stable: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Trainer policy surface is intentionally narrow: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- Search-path forward-model cache infrastructure now exists in `trainer/search/forward_model.py`: exact-state keyed transition, legal-actions, and observation caches with stats and dedicated tests.
- Plain `search` rollout now uses the cached state-query APIs; cache limits still default to `0`, and `td-search` / `td-value` have not been switched over yet.
- `scripts.run_td_loop` is the bootstrap or recalibration path; `scripts.run_td_loop_selfplay` is the primary forward loop.
- Typed bridge payloads cover the main `trainer/` package, while `scripts/` still contains some dynamic orchestration surfaces outside checked-in pyright scope.

## Remaining

- Keep calibrating loop cadence/thresholds from repeated overnight results.
- Improve `td-search` strength and throughput.
- Wire an online replay refresh loop (beyond chunk-local offline replay files).
- Tune browser `td-search` latency/throughput now that `TD Search Fast` is the default profile.
- Wire the new exact-state cache APIs into the remaining live `td-search` and `td-value` rollout paths, then benchmark hit rates and wall-clock impact before enabling nonzero defaults.
- Continue shrinking untyped/dynamic payload handling in the remaining Python script/orchestration layer outside the typed `trainer/` package.

## Immediate Next Steps

1. Continue overnight self-play loop iterations with promoted warm starts and the `12`-chunk cadence.
2. Track dual-gate outcomes (baseline vs search and candidate vs incumbent td-search) plus side-gap stability.
3. Extend the typed rollout from `trainer/` into the remaining `scripts/` orchestration and export helpers as those surfaces are touched.
4. Keep the Windows laptop wrappers and Linux cloud flows aligned with the runbook in `memoryBank/techContext.md`.

_Updated: 2026-04-20._
