# Strategic State Summary and Position Catalog v0

Status: implemented experimental diagnostic; no policy behavior changes.

## Purpose

Step 1 creates a shared factual vocabulary for strategic work before assigning
values to those facts. It has three parts:

1. `StrategicStateSummaryV0`: a deterministic, player-relative description of
   the visible position.
2. A typed catalog of strategic positions with reviewed qualitative
   expectations.
3. A seeded one-decision comparator for characterizing heuristic v2,
   V2 Hard, and the current TD profile on those positions.

The intended flow is:

```text
canonical GameState
  -> player-view-safe StrategicStateSummaryV0
  -> future projection or learned value model
  -> action comparison
```

The summary is useful to a future heuristic v3, TD observation revision, search
leaf evaluator, and diagnostics. It is not itself heuristic v3.

## Design Boundary

The v0 summary contains only:

- integers and booleans;
- rule enums;
- card, player, district, and suit identities;
- exact consequences of current rules.

It deliberately excludes:

- match equity or win probability;
- expected remaining turns or likely last mover;
- district security, criticality, or contestability labels;
- deed completion probability;
- expected income, engine payback, or token shadow value;
- option value or unknown-card probabilities;
- normalized game phase;
- recommended actions, weights, bonuses, or scores.

For example, v0 reports that a deed has one work remaining and that its owner
has enough matching loose resources to complete it. It does not claim that the
completion is currently legal, report that the deed is “valuable,” or assign a
completion probability.

Likewise, `currentLexicographicOutcome` means “who would win if this exact
position were scored now.” It is not a forecast of the actual game result.

## Contract

The implementation is
`src/policies/strategicStateSummary.ts`. The contract identifier is
`magnate.strategic-state-summary`, version `0`.

The main sections are:

| section           | facts                                                                                                                                    |
| :---------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| identity          | source state schema, perspective player, opponent player, visibility boundary                                                            |
| turn              | turn number, phase, raw turn owner, card-play status, terminal status                                                                    |
| clock             | draw count, discard count, reshuffle count, exact final-turn counter                                                                     |
| live score        | Ace-aware district points, raw developed-rank totals, resources, margins, provisional lexicographic outcome                              |
| players           | Crowns and suit counts, loose resources, exact tax loss by suit, developed/deed counts, exact income sources for results 1–10            |
| districts         | board order, marker suits, score/control, developed order, raw rank, Ace bonus, deed work and resource feasibility, placement constraint |
| placement support | own known compatible hand cards and compatible cards in the combined unknown support for each player’s stack                             |
| card knowledge    | own hand, opponent hand count, public discard, combined draw/opponent-hand support set                                                   |

Set-like card arrays use catalog order. Developed cards retain placement order
because the top card controls subsequent placement. Suit records always include
all six suits.

### Information safety

The builder applies the same visibility boundary as `toPlayerView`. It never
emits opponent hand identities or draw identities/order. The unknown-card field
is the support set remaining after subtracting the perspective hand, public
discard, and public board; it does not assign those cards to the draw pile or
opponent hand.

For a fixed perspective, the summary is invariant to:

- permutations of the hidden draw pile;
- redistributing the same unknown support between opponent hand and draw while
  preserving their public counts;
- opponent hand order;
- seed, RNG cursor, and game log changes.

The implementation validates that the visible zones and unknown support form a
complete, disjoint partition of all 30 property cards.

### Canonical rule reuse

The summary does not reproduce policy-local versions of scoring, placement, or
income rules:

- district and match score use `districtScore` and `scoreGame`;
- placement support uses `placementAllowed`;
- deed targets use `developmentCost`;
- exact income sources use the shared engine helper `incomeForResult`.

`incomeForResult` was extracted from turn flow so normal turn resolution and
read-only summaries share the same canonical logic. No income behavior changed.

## Objective Action Deltas

`strategicActionDeltasV0` applies each canonical legal `ActionWindow` action
without advancing hidden randomness and compares the resulting factual
summaries. It reports:

- district-point-margin change;
- developed-rank-margin change;
- resource-margin change;
- acted-district score-margin change;
- provisional outcome before and after;
- whether the normal card play remains available;
- where a played card goes: developed, deed, first-reshuffle discard, or dead
  discard.

These are exact state deltas, not Q-values. In particular,
`first-reshuffle-discard` records the rule fact that the card will enter the
first reshuffle; it does not estimate how beneficial the longer horizon is.

## Strategic Position Catalog

The executable catalog is
`src/botEval/strategicPositionCatalog.ts`. Every position:

- has a stable ID and catalog version;
- contains a complete 30-card partition accepted by rollout determinization;
- has a fixed Player A perspective and canonical legal focus actions;
- states a strategic thesis and expected factual relationships;
- may name a qualitative preferred focus action.

The initial catalog contains:

| ID                            | concept                                                  |
| :---------------------------- | :------------------------------------------------------- |
| `minimum-winning-coalition`   | pivotal fifth district versus fortress reinforcement     |
| `tie-denial-restores-match`   | loss-to-tie denial and the global district count         |
| `rank-tiebreak-conversion`    | conditional value of developed rank at 2–2               |
| `endpoint-optionality`        | equal immediate result with unequal continuation support |
| `deed-fork-affordable`        | immediate completion plus a remaining card play          |
| `deed-fork-inaccessible`      | identical progress with different current feasibility    |
| `sale-before-first-reshuffle` | sale remains in future draw circulation                  |
| `sale-after-first-reshuffle`  | the same sale goes to a dead discard                     |
| `ace-aware-control`           | Ace bonuses reverse the raw-rank district comparison     |

The catalog’s preferred action is a reviewed strategic hypothesis. Tests assert
that the setup and stated factual relationships are correct; they do not force
existing bots to select the preferred action. That distinction allows the suite
to expose blind spots rather than baking current behavior in as truth.

Future positions should prefer relational expectations—A changes the district
margin while B does not, or A preserves more compatible support than B—over
fragile exact floating-point scores.

## Baseline Comparator

Run:

```powershell
yarn bot:eval strategic-positions --repetitions 1
```

The default variants are:

- direct heuristic v2;
- the catalog’s V2 Hard rollout profile;
- the catalog’s current TD V2 Medium profile and default checked-in model pack.

Each `(position, repetition)` supplies the same explicit random seed to every
variant, independent of bot ID. Positions in a declared counterfactual pair,
such as the two reshuffle-boundary cases, also share that seed. Stochastic
variants therefore consume matched scenario-stream prefixes, although their
different search budgets need not consume the same number of scenarios.

Results include selected stable action keys, pairwise preference assessments,
direct-v2 focus scores/ranks, search visits/mean values, full search diagnostics,
the information-safe state summary, and a canonical SHA-256 fingerprint of the
exact executable case. Selecting an action outside a preference's declared
comparison set is recorded as unassessed, not as a mismatch. Latency is measured
for in-process Node execution and is not representative of browser/worker
latency or part of deterministic behavior expectations.

Generated `positions.json` and `summary.md` files live under ignored
`artifacts/ts-bot-evals/` by default; `--out-dir` overrides that location. They
are observations, not source-controlled golden answers.

## Invariants Protected by Tests

- complete/disjoint card partition and hidden-world determinization;
- hidden-assignment invariance;
- canonical score, Ace bonus, deed, placement, tax, and income consistency;
- stable ordering and JSON-safe plain data;
- focus actions remain canonical and legal;
- exact global-district, tiebreak, endpoint, deed, and reshuffle relationships;
- common random seeds across comparator variants;
- deterministic comparison output apart from measured latency and artifact
  metadata.

## Non-Goals and Next Step

V0 does not change heuristic v2, rollout backup, TD encoding, model dimensions,
or the bridge contract. It also does not attempt the horizon distribution,
district outcome kernels, shared future-action allocation, or calibrated match
equity model discussed for later work.

A candidate next experiment is a deliberately narrow heuristic v3:
consume these facts in one whole-state potential, replace rather than stack on
top of v2’s local district term, and use this catalog as a diagnostic gate before
paired full-game evaluation. The same catalog should first identify which blind
spots survive V2 Hard and which are already repaired by rollout search.
