# Active Context

## Current Focus

- Keep the TypeScript engine deterministic and canonical.
- Court valuation is settled: keep `courtValueScale = 1` everywhere. The term
  and the recalibrated leaf are validated as no-harm at medium/easy and as a
  screen-v1 no-harm call at hard; see `docs/design/court-valuation.md` for the
  protocol, gates, and full results.
- Improve TD policy quality through the staged loop: collect, train, gate,
  promote.
- Move district symmetry from training augmentation to an architecture
  intervention. Both controlled symmetry-training pilots improved heldout
  metrics but failed the predeclared action-symmetry gates and are not
  promotion-eligible; see `docs/design/district-symmetry.md`.
- Keep the Python training/eval runtime fail-fast and bridge-backed.
- Keep promoted checkpoint registration portable through
  `models/td_checkpoints/manifest.json`.

## Current State

- Browser play is functional and deterministic with four profiles:
  Easy/Medium/Hard (`rollout-search-v2-*` with heuristic v2, both rulesets) and
  Experimental (`td-root-search-v2-medium`, standard-ruleset only). Games
  support standard and extended rulesets, autosave/restore, and local game
  history. Paired TD rollout inference is the browser default, with
  `?tdSearchExecutor=legacy` as the session rollback.
- Court valuation attempt two is implemented and settled at
  `courtValueScale = 1`: the failed flat deed floor was removed, standard
  scoring is restored exactly, and incomplete Courts are priced by
  `src/policies/courtPotentialV2.ts` as a feasibility-discounted district swing
  (`swing × feasibility × courtValueScale`). The next lever, if Courts ever
  need to add strength, is the action term's recommended-buy pricing.
- The TD hard extra-data step-9,000 checkpoint is promoted as the
  `experimental` manifest entry, is the default training warm start and
  opponent-pool entry, and is deployed as the default browser pack
  (`td-hard-extra-data-primary-treatment-step-09000`). Its sealed 100-game
  final test is unspent. The pair earned the `experimental` slot through a
  separate 120-game browser benchmark (99-21, 82.5%, versus heuristic-v2
  medium) after mixed replication on the frozen development metrics.
- District symmetry is an open defect: the 10,000-decision audit confirmed an
  exact-symmetry violation in the deployed model, and both augmentation
  interventions were closed without promotion. The audit, predeclared gates,
  outcomes, constraints, and open architecture design live in
  `docs/design/district-symmetry.md`; reserved repetitions 24-47 and full-game
  promotion tests remain unspent.
- Heuristic rollout's uncertain-draw resource-preservation blind spot remains a
  separate diagnostic; substituting heuristic rollout is not a general fix.

## Remaining Work

- Complete and implement the district-equivariant architecture described in
  `docs/design/district-symmetry.md` (fixed-D3 S4 symmetry by construction, not
  more augmentation weight or subjective action boosts).
- Keep repetitions 24-47 reserved for a future candidate selected on
  independent evidence, followed by full-game promotion tests.
- Calibrate self-play loop cadence, replay-window settings, and promotion
  thresholds from repeated runs.
- Continue improving throughput for direct TypeScript TD-root matchups;
  individual Node search decisions remain synchronous.
- Continue shrinking untyped or dynamic payload handling in Python scripts as
  those surfaces are touched.

## Immediate Next Steps

1. Decide whether to spend the sealed 100-game final test on the promoted
   step-9,000 candidate.
2. Complete the open architecture design in `docs/design/district-symmetry.md`
   and freeze a new experiment manifest with predeclared gates and a matched
   control before implementing.
3. Continue self-play iterations with promoted manifest warm starts,
   `td-lambda` value targets, checkpoint selection, replay windows, and
   generator gating.
4. Keep docs aligned by replacing stale content rather than appending task
   history.

_Updated: 2026-09-24._
