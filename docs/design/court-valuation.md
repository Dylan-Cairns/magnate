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

Runs are extended-ruleset hard-profile A/B with paired seeds: candidate
`courtValueScale = 1` versus control `courtValueScale = 0`. Hard is the
deployment default and its rollouts reach terminal states, so outcomes are
priced by real game results rather than the leaf evaluator.

- Screen: `configs/bot-eval/court-valuation/extended-hard-ab.json`, 30 pairs.
- Extension: `extended-hard-extension.json`, 60 pairs, same seed prefix.
  Extend once when the screen's paired margin point estimate is positive and
  its ci95 includes 0. Never extend a clearly negative or clearly positive
  result.
- Standard smoke: `standard-hard-smoke.json`, 10 pairs, sanity only; standard
  parity is structural and proven by unit tests.

Gates:

- `extended-paired-improvement`: pass when the paired margin ci95 low is above
  0; observe when positive but the interval includes 0; fail otherwise.
- `extended-court-utilization`: candidate must acquire at least one Court and
  complete at least one.
- `extended-court-follow-through`: completions / acquisitions at least 0.5
  passes, 0.25-0.5 observes, below 0.25 fails.
- `extended-court-dump`: zero Court sells while a Court build was legal passes;
  one or two observe; more fail.
- `standard-paired-noninferiority` and `standard-deed-buy-behavior` remain the
  standard-ruleset gates for the smoke run.

Report with
`yarn bot:eval court-value-report --artifact <matchup.json> [--out-dir <path>]`.
The report includes court-decision diagnostics: how often a Court action ranked
top-1/top-4/top-16, and the mean swing, feasibility, and delta of the best
Court option.

## Ownership

- Agent: implementation, tests, configs, report tooling, artifact analysis,
  gate calls, docs.
- Owner: run the benchmark series (and the single predeclared extension) on
  the project machine, then hand artifact paths back.

## Rollback

`courtValueScale = 0` disables the term. Deleting the field restores the
deployed default of 1.
