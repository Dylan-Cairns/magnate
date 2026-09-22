# Active Context

## Current Focus

- Keep the TypeScript engine deterministic and canonical.
- Court valuation attempt two is implemented and unbenchmarked
  (`docs/design/court-valuation.md`): the failed flat deed floor was removed,
  standard scoring is restored exactly, and a dedicated term now prices
  incomplete Courts as a feasibility-discounted district swing
  (`courtValueScale`, default 1, control 0). Next: run the predeclared
  extended-hard A/B series with `yarn bot:eval court-value-report`.
- Improve TD policy quality through the staged loop: collect, train, gate, promote.
- Move district symmetry from training augmentation to an architecture
  intervention. Both controlled symmetry-training pilots improved heldout
  metrics but failed the predeclared action-symmetry gates and are not
  promotion-eligible.
- Keep the Python training/eval runtime fail-fast and bridge-backed.
- Keep promoted checkpoint registration portable through
  `models/td_checkpoints/manifest.json`.

## Current State

- Court valuation attempt two is implemented and unbenchmarked. The flat
  `deedPotentialBase` floor and its knob were deleted (restoring standard
  scoring bit-for-bit), and incomplete Courts are now valued by
  `src/policies/courtPotentialV2.ts` as `swing × feasibility × courtValueScale`
  for buy-deed and develop-deed only; develop-outright and completed Courts stay
  on the generic path. Deployed profiles default to `courtValueScale = 1`; the
  benchmark control is 0. Configs live in `configs/bot-eval/court-valuation/`.
- Browser play is functional and deterministic with four profiles:
  Easy/Medium/Hard (`rollout-search-v2-*` with heuristic v2, both rulesets) and
  Experimental (`td-root-search-v2-medium`, standard-ruleset only). Games
  support standard and extended rulesets, autosave/restore, and local game
  history.
- The TD hard extra-data step-9,000 checkpoint is promoted as the `experimental`
  manifest entry, is the default training warm start and opponent-pool entry,
  and is deployed as the default browser pack
  (`td-hard-extra-data-primary-treatment-step-09000`). Its sealed 100-game
  final test is unspent.
- The extra-data candidate scored 99-21 (82.5%) against heuristic-v2 medium
  over 120 paired games, decisively outperforming the July incumbent's 73.3%
  (88-32). The development split was mixed replication on the frozen primary
  metrics; the pair is nevertheless the checked-in `experimental` entry on the
  strength of the separate 120-game browser benchmark.
- Both symmetry interventions (random-S4 augmentation, opponent-only
  complete-orbit augmentation) passed heldout noninferiority and improved value
  or opponent metrics, but failed the required top-action agreement (>= 95%)
  and probability-drift reduction (>= 50%) gates. Reserved optionality
  repetitions 24-47 and full-game promotion tests remain unspent.
- A replay-wide direct audit over 10,000 ordinary game decisions under all 24
  fixed-D3 district permutations confirmed a meaningful exact-symmetry
  violation in the deployed model. This result, not the invalid catalog-v1
  fixtures, is the broad evidence behind the symmetry work.
- Heuristic rollout's uncertain-draw resource-preservation blind spot remains a
  separate diagnostic; substituting heuristic rollout is not a general fix.
- Browser presentation runs one `AnimationSequence` per canonical transaction
  with FIFO input barriers and `viewState` rendering; canonical state stays
  controller-owned. Paired TD rollout inference is the browser default, with
  `?tdSearchExecutor=legacy` as the session rollback.

## Remaining Work

- Design a district-equivariant opponent/action model that enforces fixed-D3 S4
  symmetry in the architecture rather than through more augmentation weight or
  subjective action boosts.
- Keep repetitions 24-47 reserved for a future candidate selected on
  independent evidence, followed by full-game promotion tests.
- Calibrate self-play loop cadence, replay-window settings, and promotion
  thresholds from repeated runs.
- Continue improving throughput for direct TypeScript TD-root matchups;
  individual Node search decisions remain synchronous.
- Continue shrinking untyped or dynamic payload handling in Python scripts as
  those surfaces are touched.

## Immediate Next Steps

1. Run the predeclared court-valuation benchmark: extended-hard A/B screen at
   30 pairs (`configs/bot-eval/court-valuation/extended-hard-ab.json`), extend
   once to 60 pairs if the point estimate is positive and the interval includes
   0, then call gates with `yarn bot:eval court-value-report`. Do not spend the
   standard smoke on gate calls; standard parity is structural.
2. Decide whether to spend the sealed 100-game final test on the promoted
   step-9,000 candidate.
3. Write a short design and guardrail plan for architectural fixed-D3 S4
   symmetry while preserving existing replay, checkpoint, and browser-export
   contracts.
4. Continue self-play iterations with promoted manifest warm starts,
   `td-lambda` value targets, checkpoint selection, replay windows, and
   generator gating.
5. Keep docs aligned by replacing stale content rather than appending task
   history.

_Updated: 2026-09-15._
