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
- Head-to-head evals checkpoint every completed pair to
  `<out-dir>/checkpoint.json` and resume with `--resume <checkpoint-path>`;
  writes are synchronous atomic replaces because run loops do not yield to the
  event loop, and the checkpoint is removed once the final artifact is written.
- Court decision calibration (`yarn bot:eval court-decision-eval`) compares
  term-owned Court actions against the best non-Court action under matched
  worlds and terminal rollouts, from a completed artifact or an interrupted
  run's checkpoint. On the round-1 floor-era artifacts (150 hard / 300 medium
  positions) term recommendations were break-even (hard -0.011 ci95
  [-0.192, +0.170]; medium -0.002 ci95 [-0.093, +0.090]) and rejections mildly
  correct; the repeated weakness was low-feasibility and early-phase Court
  actions. It is the pre-screen for behavioral runs, not a gate.
- Court valuation screen v1 (`court-valuation-extended-hard-screen-v1`, 10 pairs
  at 25 worlds) finished 12W-8L (60%), paired margin +0.20 ci95
  [-0.192, +0.592]; utilization and follow-through passed, and the coarse dump
  gate failed at 6 while same-card dumps were 1 per arm (all events were
  late-game resource sells of unowned hand cards that the search valued above
  the alternatives). Calibration on the screen's own positions (300 sampled):
  recommendations +0.021 [-0.052, +0.094], rejections -0.030; develop
  recommendations +0.059 [-0.008, +0.125], buy recommendations -0.146
  [-0.405, +0.112], rejected buys -0.556 [-1.104, -0.007]. Verdict: keep
  `courtValueScale = 1`; buy discrimination is directionally right and
  recommended-buy softness is not significant.
- The search leaf evaluator prices incomplete Courts through
  `courtPotentialValueForPlayerV2` at `courtValueScale > 0`, while scale 0 keeps
  the legacy generic curve as the control. Leaf v1 (the raw action value) failed
  its medium gate (`court-valuation-extended-medium-ab-v1`: 25W-34L-1D, 41.7%,
  paired margin -0.15 [-0.376, +0.076]) because a fresh deed was priced at
  almost a completed Court's swing, switching medium play from the control's
  outright path to 75 deed buys per 60 games. The leaf now applies a convex
  progress discount (`progressRatio^2`): fresh deeds carry no option value and
  value accrues as the deed is developed. The leaf v2 screen
  (`court-valuation-extended-medium-screen-v1`, 10 pairs) came back 10W-10L
  (paired margin 0.00, 2-2-6) with all behavioral gates passing, and the 30-pair
  v2 validation (`court-valuation-extended-medium-ab-v2`: 29W-31L, 48.3%, paired
  margin -0.033 [-0.253, +0.187]) failed the paired-improvement gate by rule
  while removing the v1 regression (v1 margin -0.15); utilization,
  follow-through (0.72), and dump gates pass. Term + calibrated leaf is
  approximately neutral versus scale 0 at medium. The v2 CDC keeps the
  decision-level verdict neutral (recommendations +0.019, recommended develops
  +0.068, recommended buys -0.047, all n.s.; rejections -0.116 with CI
  excluding 0; the negative buy bucket is carried by rejected buys the term
  avoids). Pooled across both medium runs the scale-1 arm is 9-15 on discordant
  pairs (p ≈ 0.31, pooled margin ≈ -0.05); at 30 pairs the CI half-width is
  ≈0.22, so the gate certifies large effects only. Verdict (2026-09-24): keep
  `courtValueScale = 1` everywhere; medium/easy are validated as no-harm, not
  as strength gains. The next lever, if Courts ever need to add strength, is
  the action term's recommended-buy pricing (hard -0.146, medium -0.047, both
  n.s.), which requires a fresh hard screen.
  The dump gate uses the same-card metric with the coarse count as a diagnostic.
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

1. Decide whether to spend the sealed 100-game final test on the promoted
   step-9,000 candidate.
2. Write a short design and guardrail plan for architectural fixed-D3 S4
   symmetry while preserving existing replay, checkpoint, and browser-export
   contracts.
3. Continue self-play iterations with promoted manifest warm starts,
   `td-lambda` value targets, checkpoint selection, replay windows, and
   generator gating.
4. Keep docs aligned by replacing stale content rather than appending task
   history.

_Updated: 2026-09-24._
