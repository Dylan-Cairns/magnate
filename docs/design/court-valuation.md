# Court Valuation (Heuristic V2)

Status: implemented 2026-09-21. Benchmark series are run manually by the
project owner and analyzed by an agent.

## Background

Heuristic v2 scored a fresh Court deed at roughly zero: the generic scoring
term weights a deed's district potential by `(progress/target)²`, which is 0 at
purchase, and Courts have zero income potential so the earning term is 0 too.
The first attempt (a flat `deedPotentialBase` floor on every fresh deed) fixed
Court engagement behavior but lost its A/B: it double-counted income cards whose
future value already flows through the earning-access term, and the candidate
started 1.1-1.4 extra deeds per game with about twice as many unfinished deeds.
That floor has been removed; standard-ruleset scoring is restored exactly.

## Design

Courts are the only developable cards with no income channel, so they get a
dedicated valuation owned by `src/policies/courtPotentialV2.ts`.

**Ownership.** The court term covers incomplete court states only: `buy-deed`
and `develop-deed` on a Court. Inside `potentialStackScore`, a Court deed
contributes zero to the generic potential. `develop-outright` on a Court stays
on the generic scoring path (it was always priced correctly, because the
projection puts the card straight into `developed`). No effect is counted
twice, and standard-ruleset play cannot reach the term because the standard
deck has no Courts.

The search leaf evaluator prices incomplete Courts through
`courtPotentialValueForPlayerV2`. The leaf keeps the action term's swing and
feasibility but applies a convex progress discount
(`progressRatio^COURT_LEAF_PROGRESS_EXPONENT`, exponent 2): a leaf state is a
position, not an action, so a fresh deed carries no option value and value
accrues as the deed is developed. Leaf v1 shared the action term's value
exactly and failed its medium validation (see Decision-Level Calibration). At
`courtValueScale = 0` the leaf keeps the legacy generic curve, which keeps the
control arm comparable to pre-change behavior.

**Value.** For a Court in a district:

```
swing         D = tanh((own + increment − opponent)/5) − tanh((own − opponent)/5)
turnsLeft     = finalTurnsRemaining ?? max(0, 42 − turn), capped at 24
available     = stock + flow × turnsLeft × 0.7 − tokensSpentByThisAction
feasibility   F = available / (available + remainingCost)
value         P = D × F × courtValueScale
court delta   = P(after) − P(before)
leaf value    L = P × (progress/target)²
```

The action term uses `P` because it prices a real option bought this turn; the
leaf uses `L` because it prices a standing position. `stock` and `flow` are the
Court's three suits only: `flow` is the existing expected income access per suit
(`suitAccessBySuitForPlayerV2`), and `stock` is the current resource count. The
swing reuses the generic tanh margin machinery, so a Court in an already-decided
district is worth little and one in a contested district is worth a lot.

The action score adds the court delta inside the existing scoring weight:
`scoringWeight × (genericScoringDelta + courtDelta)`. Phase behavior therefore
comes for free: Courts lose to engine-building early and compete late.

**Coherence.** `F` is strictly increasing as `remainingCost` shrinks and
`available` grows, so whenever the buy clears the bar, progress on that deed is
also positive, and completing the deed always nets the full generic swing minus
the already-credited discounted value. The tail completion value shrinks as
feasibility approaches certainty, which is expected-value accounting; at the
default scale of 1 the completion total stays positive. Scales above 1 shift
value earlier and can make the final completing token unattractive, so treat
values above 1 as an experiment, not a default.

## Knobs

- `courtValueScale` (`SearchPolicyConfig`, default `1`): the only A/B and
  tuning multiplier; `0` disables the term and is the control arm.
- `COURT_FEASIBILITY_HAIRCUT = 0.7`: share of projected income assumed
  available to the Court.
- `COURT_HORIZON_CAP = 24`: maximum projected turns, preventing tiny early
  flows from implying false feasibility.
- `COURT_LEAF_PROGRESS_EXPONENT = 2`: convexity of the leaf's progress
  discount; 0 would restore leaf v1's shared action value.

## Invariants

- Standard-ruleset scores are unchanged; the term returns undefined for
  non-extended states and non-Court cards.
- `develop-outright` on a Court scores identically at any `courtValueScale`.
- `P` is monotone in progress, stock, flow, and turns left.
- The leaf value `L` is 0 for a fresh deed and grows convexly with progress.
- Buy coherence: buy, progress, and completion totals are positive at scale 1.
- Zero stock, zero flow, and no turns give feasibility 0.

## Benchmark Protocol (predeclared)

The 30/60-pair win-rate series is retired: at hard settings it costs 35-80
hours and cannot resolve a term that changes a subset of decisions. Validation
is one decision-level calibration pre-screen plus one behavioral screen. Runs
are extended-ruleset hard-profile with paired seeds: candidate
`courtValueScale = 1` versus control `courtValueScale = 0`. Hard is the
deployment default and its rollouts reach terminal states (depth 270 exceeds the
measured maximum game length), so behavior is priced by real game results
rather than the leaf evaluator.

- Calibration pre-screen: `yarn bot:eval court-decision-eval` on existing
  extended artifacts or an interrupted run's checkpoint; see Decision-Level
  Calibration above.
- Screen: `configs/bot-eval/court-valuation/extended-hard-screen.json`, 10
  pairs, 25 worlds, depth 270. Width is halved to cut cost; depth keeps
  terminal pricing. The win-rate diagnostic is reported but not gated at this
  N.
- No extension and no standard smoke: standard-ruleset parity is structural
  (invariants plus unit tests), and the screen exists to catch behavior
  problems, not to estimate win rate.
- Medium screen: `configs/bot-eval/court-valuation/extended-medium-screen.json`,
  10 pairs at medium (10 worlds, depth 40). Medium is leaf-priced, so this is
  where leaf calibration shows up; the screen reads Court path usage (deed buys
  versus outrights) and reports the win-rate diagnostic.
- Medium validation: `configs/bot-eval/court-valuation/extended-medium-ab.json`,
  30 pairs; at 30 pairs `extended-paired-improvement` is a real gate call.

Gates:

- `extended-court-utilization`: candidate must acquire at least one Court and
  complete at least one.
- `extended-court-follow-through`: completions / acquisitions at least 0.5
  passes, 0.25-0.5 observes, below 0.25 fails.
- `extended-court-dump`: zero same-card dumps (a sold Court whose own buy was
  legal) passes; one or two observe; more fail. The coarse count (any legal
  Court build) stays a diagnostic in the detail text. Screen v1 was scored
  before this refinement and recorded a coarse fail (6) that decomposed to one
  same-card dump per arm; every event was a late-game resource sell of an
  unowned hand card that the search valued above the alternatives.
- `extended-paired-improvement` is reported as a diagnostic only; the tool
  marks it observe below 30 pairs and no extension follows.

Report with
`yarn bot:eval court-value-report --artifact <matchup.json> [--out-dir <path>]`.
The report includes court-decision diagnostics: how often a Court action ranked
top-1/top-4/top-16, and the mean swing, feasibility, and delta of the best
Court option.

## Decision-Level Calibration

`yarn bot:eval court-decision-eval` is the pre-screen. It replays an extended
artifact, finds decisions with a term-owned Court option, and compares the
term's top Court action against the best non-Court action under matched hidden
worlds and terminal rollouts, split into recommendations and rejections with
swing, feasibility, and phase buckets.

On the round-1 floor-era artifacts (150 hard positions, 300 medium positions)
term recommendations were break-even overall (hard -0.011, ci95
[-0.192, +0.170]; medium -0.002, ci95 [-0.093, +0.090]), and rejections were
mildly correct. The repeated weakness was low-feasibility recommendations
(clearly negative on the medium set) and early-phase actions in both sets;
high-feasibility recommendations were neutral rather than clearly positive.
These are calibration observations from an aggressive-play position set, not a
gate: behavioral gates still come from the head-to-head screen.

Screen v1 (`court-valuation-extended-hard-screen-v1`, 10 pairs at 25 worlds,
2026-09-22): candidate 12W-8L (60%), paired margin +0.20 ci95
[-0.192, +0.592]; utilization and follow-through passed (14 acquired, 12
completed, 0.857); the coarse dump gate recorded 6 while same-card dumps were 1
per arm (the refined gate observes). Calibration on the screen's own positions
(300 sampled): recommendations +0.021 [-0.052, +0.094], rejections -0.030
[-0.088, +0.028]; develop recommendations +0.059 [-0.008, +0.125]; buy
recommendations -0.146 [-0.405, +0.112] while rejected buys were -0.556
[-1.104, -0.007], so the term's buy discrimination is directionally right and
recommended-buy softness is not significant.

Leaf v1 (`court-valuation-extended-medium-ab-v1`, 30 pairs, 2026-09-23) failed
its medium gate: candidate 25W-34L-1D (41.7%), paired margin -0.15
[-0.376, +0.076], `extended-paired-improvement` fail. The usage table showed
the mechanism: the candidate switched from the control's outright path (10
develop-outrights, 3 deed buys, 11 completions) to the deed path (75 buys, 512
develops, 51 completions), because a fresh deed was priced at almost a
completed Court's swing. Calibration on the same artifact (300 positions) kept
the term verdict neutral (recommendations +0.014 [-0.062, +0.090], rejections
-0.069 [-0.139, +0.001]; develop recommendations +0.065 [-0.018, +0.148]; buy
recommendations -0.107 [-0.267, +0.052] while rejected buys were -0.727
[-1.276, -0.178]), so the failure was the leaf's option pricing, not the action
term's discrimination. The leaf now applies the convex progress discount.

Leaf v2 screen (`court-valuation-extended-medium-screen-v1`, 10 pairs,
2026-09-23): candidate 10W-10L, paired margin 0.00 (2-2-6), all behavioral
gates pass (17 acquired, 10 completed, follow-through 0.588, zero same-card
dumps). The progress discount cut deed activity from 1.25 to 0.85 buys per game
(75 buys over 60 games to 17 over 20) and removed the negative point estimate;
the 30-pair v2 validation is the gate call.

Leaf v2 validation (`court-valuation-extended-medium-ab-v2`, 30 pairs,
2026-09-23): candidate 29W-31L (48.3%), paired margin -0.033
[-0.253, +0.187], `extended-paired-improvement` fail by rule (margin ≤ 0), but
the v1 regression is gone (v1 was -0.15). Utilization and follow-through (0.72)
pass; the dump gate observes at 1 same-card. The term + calibrated leaf is
approximately neutral versus scale 0 at medium, not harmful.

The v2 CDC (300 positions) keeps the decision-level verdict neutral:
recommendations +0.019 [-0.061, +0.100], recommended develops +0.068
[-0.012, +0.147], recommended buys -0.047 [-0.205, +0.110], rejections -0.116
[-0.231, -0.001]. The buy bucket is negative overall (-0.181) but that is
carried by rejected buys (-0.713 [-1.186, -0.240]), which the term already
avoids. Pooled across both medium runs the scale-1 arm is 9-15 on discordant
pairs (p ≈ 0.31, pooled margin ≈ -0.05). At 30 pairs the paired CI half-width
is ≈0.22, so this gate certifies large effects only; a small coherence effect
can neither pass nor be excluded at this N.

Verdict (2026-09-24): keep `courtValueScale = 1` for every profile. Hard keeps
the screen-v1 no-harm call; medium/easy are validated as no-harm after the leaf
recalibration (pooled margin ≈ -0.05, n.s.), not as strength gains. The
remaining soft spot is the action term's recommended buys (hard -0.146, medium
-0.047, both n.s.); tightening the term's buy pricing is the next lever if
Courts ever need to add strength rather than coherence, and it requires a fresh
hard screen.

## Ownership

- Agent: implementation, tests, configs, report tooling, artifact analysis,
  gate calls, docs.
- Owner: run the calibration pre-screen and the behavioral screen on the
  project machine, then hand artifact paths back.

## Rollback

`courtValueScale = 0` disables the term. Deleting the field restores the
deployed default of 1.
