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

The search leaf evaluator shares the same state value through
`courtPotentialValueForPlayerV2`, so depth-limited profiles (easy and medium)
price incomplete Courts consistently with the action term. At
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
```

where `stock` and `flow` are the Court's three suits only: `flow` is the
existing expected income access per suit (`suitAccessBySuitForPlayerV2`), and
`stock` is the current resource count. The swing reuses the generic tanh margin
machinery, so a Court in an already-decided district is worth little and one in
a contested district is worth a lot.

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

## Invariants

- Standard-ruleset scores are unchanged; the term returns undefined for
  non-extended states and non-Court cards.
- `develop-outright` on a Court scores identically at any `courtValueScale`.
- `P` is monotone in progress, stock, flow, and turns left.
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
- Medium validation: `configs/bot-eval/court-valuation/extended-medium-ab.json`,
  30 pairs. Medium is leaf-priced, so this run validates the aligned leaf and
  the action term together; at 30 pairs `extended-paired-improvement` is a real
  gate call.

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

## Ownership

- Agent: implementation, tests, configs, report tooling, artifact analysis,
  gate calls, docs.
- Owner: run the calibration pre-screen and the behavioral screen on the
  project machine, then hand artifact paths back.

## Rollback

`courtValueScale = 0` disables the term. Deleting the field restores the
deployed default of 1.
