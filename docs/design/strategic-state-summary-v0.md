# Strategic State Summary v0 and Position Catalog v1

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

Catalog v1 contains 12 positions:

| ID                                  | concept                                                   |
| :---------------------------------- | :-------------------------------------------------------- |
| `minimum-winning-coalition`         | pivotal fifth district versus fortress reinforcement      |
| `tie-denial-restores-match`         | loss-to-tie denial and the global district count          |
| `rank-tiebreak-conversion`          | conditional value of developed rank at 2–2                |
| `known-hand-optionality-original`   | preserving a guaranteed hand continuation                 |
| `known-hand-optionality-mirror`     | the same known-hand option with district roles reversed   |
| `unknown-pool-optionality-original` | preserving placement support for possible future draws    |
| `unknown-pool-optionality-mirror`   | the same unknown-pool option with district roles reversed |
| `deed-fork-affordable`              | immediate completion plus a remaining card play           |
| `deed-fork-inaccessible`            | identical progress with different current feasibility     |
| `sale-before-first-reshuffle`       | sale remains in future draw circulation                   |
| `sale-after-first-reshuffle`        | the same sale goes to a dead discard                      |
| `ace-aware-control`                 | Ace bonuses reverse the raw-rank district comparison      |

The four optionality cases replace the confounded catalog-v0 endpoint case.
Both focus actions now play The Sailor with the same payment and identical
immediate score, rank, resource, hand, card-knowledge, and income consequences.
In the known-hand family, preserving The Forest keeps a tax-safe Author play
reachable on Player A's final turn; it changes a 5-7 district to 7-7 and a
rank-tiebreak loss at 2-2 into a 2-1 match win. In the unknown-pool family,
preserving The Discovery retains placement support for The Penitent, the only
unknown card whose support differs between the two actions. The canonical
hidden assignment draws it before the final-turn window to verify the
continuation is executable; playing it on Player A's final turn changes the
corresponding district to 11-7 and the match to 3-1. Each mirror swaps both
complete target lanes, including their marker masks and both players' stacks,
so a policy that values the preserved option must reverse its physical district
choice rather than follow stable district or action ordering.

The catalog’s preferred action is a reviewed strategic hypothesis. Tests assert
that the setup and stated factual relationships are correct; they do not force
existing bots to select the preferred action. That distinction allows the suite
to expose blind spots rather than baking current behavior in as truth.

Future positions should prefer relational expectations—A changes the district
margin while B does not, or A preserves more compatible support than B—over
fragile exact floating-point scores.

## Baseline Comparator

Run a one-seed smoke characterization with:

```powershell
yarn bot:eval strategic-positions --repetitions 1
```

Use eight repetitions as the initial fixed-position stability screen:

```powershell
yarn bot:eval strategic-positions --repetitions 8
```

The default variants are:

- direct heuristic v2;
- the catalog’s V2 Hard rollout profile;
- the catalog’s current TD V2 Medium profile and default checked-in model pack.

Their selectable IDs are `heuristic-v2-direct`, `rollout-search-v2-hard`, and
`td-root-search-v2-medium`. `--positions` and `--variants` accept unique,
comma-separated IDs. `--start-repetition` offsets the deterministic repetition
and seed indices for a targeted extension; it does not append to or merge an
earlier artifact.

The opt-in `td-root-search-v2-800-visits` diagnostic is excluded from the
defaults. It clones TD V2 Medium and changes only sampled worlds from 10 to 50,
giving it 800 root visits while retaining depth 40, the same default model-pack
selection, and TD guidance at root, rollout, and leaf. This matches V2 Hard's
root-visit count, not its search configuration or total computation.

Each `(position, repetition)` supplies the same explicit random seed to every
variant, independent of bot ID. Positions in a declared counterfactual group,
including the optionality mirrors, deed-affordability pair, and reshuffle
boundary, also share that seed. Stochastic variants therefore consume matched
scenario-stream prefixes, although their different search budgets need not
consume the same number of scenarios.

Results include selected stable action keys, pairwise preference assessments,
direct-v2 focus scores/ranks, search visits/mean values, full search diagnostics,
the information-safe state summary, and a canonical SHA-256 fingerprint of the
full catalog-case payload, including its descriptive metadata. Markdown reports
per-position/variant selection stability, within-position pairwise focus gaps
and expansion coverage,
counterfactual-group transitions, raw decisions, and focus-action signals.
Selecting an action outside a preference's declared comparison set is recorded
as unassessed, not as a mismatch. Search values come from adaptive, potentially
unequal visits, so their gaps are diagnostic within one position and variant,
not controlled paired estimates. Latency is measured for in-process Node
execution and is not representative of browser/worker latency or part of
deterministic behavior expectations.

Repeated seeds characterize policy stability in the same fixed cases. They are
not independent games and do not prove that the reviewed preference improves
match equity. Direct v2 is deterministic here, so repeating it establishes
repeatability rather than adding strategic evidence.

Generated `positions.json` and `summary.md` files live under ignored
`artifacts/ts-bot-evals/` by default; `--out-dir` overrides that location. They
are observations, not source-controlled golden answers.

## Step 2 Characterization Outcome

The 2026-07-13 catalog-v1 screen produced these diagnostic results:

- Direct heuristic v2 selected the reviewed action in all eight assessed
  comparisons. Its optionality gaps were positive but small: about `+0.008` for
  each known-hand mirror and `+0.004` for each unknown-pool mirror.
- V2 Hard selected the reviewed action in every assessed row of the initial
  eight-seed screen. Across 24 optionality seeds, the known-hand pair never
  selected the destructive overwrite: it preserved 43 times and sold the
  option-preserving Ace five times. The unknown-pool pair preserved 47 of 48
  times; the single overwrite was a 299-299 visit tie whose fixed physical-lane
  tie-break reversed its strategic label in only one mirror. This is stable
  selection behavior, but the near-zero unknown-pool adaptive value gaps do not
  prove a continuation-value advantage.
- Current TD V2 Medium retained a persistent mirror asymmetry across the same
  24 optionality seeds. Known-hand original selected preserve/overwrite 14/10,
  while its mirror selected 24/0. Unknown-pool original selected preserve 21
  times with three outside-pair sales; its mirror selected preserve/overwrite
  7/7 with ten outside-pair sales. The default TD profile has 160 root visits,
  versus 800 for V2 Hard, and often gives a focus branch only one to three
  visits, so this is not a compute-matched attribution.

These fixed cases did not justify a heuristic-v3 potential: direct v2 and V2
Hard already captured every reviewed relationship, at least as root selection
behavior. They instead made an 800-root-visit TD comparison the next diagnostic.

## Step 3 800-Visit TD Outcome

The 2026-07-13 follow-up compared TD V2 Medium's original 160 root visits with
the opt-in 800-visit variant over the four optionality positions and seeds
0–23, for 96 matched decisions. The random seed, information-safe state, legal
actions, focus-action consequences, and all TD root priors matched between the
two budgets. The known-hand case fingerprints also matched. The two
unknown-pool fingerprints changed because one expected-fact sentence was
reworded between artifact runs; their executable inputs and priors still
matched exactly.

- Destructive option overwrites fell from 17 of 96 decisions at 160 visits to
  5 of 96 at 800 visits.
- The unknown-pool mirror improved from seven overwrites to zero. The
  unknown-pool original already had zero and remained at zero.
- The known-hand original improved from ten overwrites to five, while its
  mirror remained at zero.
- Seed by seed, 13 harmful choices became safe, one safe choice became harmful,
  four remained harmful, and 78 remained safe.
- The new sell and trade selections in the unknown-pool cases are safe deferrals:
  they do not play The Sailor or destroy either valuable endpoint, so they are
  not counted as strategic failures.

Every 800-visit search used the full budget, expanded both focus actions, and
reached a terminal game state in every simulation. The 160-visit baseline had
also reached terminal states throughout these fixtures, so extra depth was not
the explanation. More simulations substantially overcame the TD model's
unchanged physical-lane prior, but did not eliminate the bias: all five
remaining harmful choices occur in the known-hand original, versus none in its
mirror.

The next useful diagnostic is therefore narrow: on the known-hand pair and the
remaining failure seeds, switch TD root, rollout, and leaf guidance separately
to identify which hook carries the asymmetry. An encoding-collision holdout is
useful only after that localization. No heuristic, observation, or training
change is yet justified by this experiment alone.

## Invariants Protected by Tests

- complete/disjoint card partition and hidden-world determinization;
- hidden-assignment invariance;
- canonical score, Ace bonus, deed, placement, tax, and income consistency;
- stable ordering and JSON-safe plain data;
- focus actions remain canonical and legal;
- exact global-district, tiebreak, mirrored known-hand/unknown-pool optionality,
  deed, and reshuffle relationships;
- common random seeds across comparator variants and counterfactual groups;
- deterministic repetition offsets and explicit comparison subsets;
- per-position/variant stability, focus-gap, and counterfactual reporting;
- deterministic comparison output apart from measured latency and artifact
  metadata.

## Non-Goals and Next Step

V0 does not change heuristic v2, rollout backup, TD encoding, model dimensions,
or the bridge contract. It also does not attempt the horizon distribution,
district outcome kernels, shared future-action allocation, or calibrated match
equity model discussed for later work.

The immediate next step is the narrow 800-visit TD guidance ablation described
above, not heuristic v3. Only a concept that survives repeated characterization
should become a bot change. The catalog is characterization, not proof of
strategic correctness; a surviving candidate still needs controlled
continuation outcomes, holdout position families, and paired full-game
evaluation.
