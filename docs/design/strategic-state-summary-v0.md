# Strategic State Summary v0 and Position Catalog

Status: implemented experimental diagnostic; no policy behavior changes.

## Purpose

Strategic work starts from a shared factual vocabulary before assigning values.
It has three parts:

1. `StrategicStateSummaryV0`: a deterministic, player-relative description of
   the visible position.
2. A typed catalog of strategic positions with reviewed qualitative
   expectations.
3. A seeded one-decision comparator for characterizing heuristic v2, V2 Hard,
   and the current TD profile.

Intended flow:

```text
canonical GameState
  -> player-view-safe StrategicStateSummaryV0
  -> future projection or learned value model
  -> action comparison
```

The summary is a reusable input for a future heuristic v3, TD observation
revision, search leaf evaluator, and diagnostics. It is not itself heuristic v3.

## Design Boundary

The v0 summary contains only integers and booleans, rule enums,
card/player/district/suit identities, and exact consequences of current rules.

It deliberately excludes match equity, win probability, expected remaining
turns, district security or criticality labels, deed completion probability,
expected income, option value, unknown-card probabilities, normalized game
phase, and any recommended actions, weights, or scores.

For example, v0 reports that a deed has one work remaining and whether its owner
has matching loose resources. It does not claim the completion is legal, call
the deed valuable, or assign a probability. `currentLexicographicOutcome` means
"who would win if this exact position were scored now"; it is not a forecast.

## Contract

Implementation: `src/policies/strategicStateSummary.ts`. Contract identifier:
`magnate.strategic-state-summary`, version `0`.

| section           | facts                                                                                                                                    |
| :---------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| identity          | source state schema, perspective player, opponent player, visibility boundary                                                            |
| turn              | turn number, phase, raw turn owner, card-play status, terminal status                                                                    |
| clock             | draw count, discard count, reshuffle count, exact final-turn counter                                                                     |
| live score        | Ace-aware district points, raw developed-rank totals, resources, margins, provisional lexicographic outcome                              |
| players           | Crowns and suit counts, loose resources, exact tax loss by suit, developed/deed counts, exact income sources for results 1-10            |
| districts         | board order, marker suits, score/control, developed order, raw rank, Ace bonus, deed work and resource feasibility, placement constraint |
| placement support | own known compatible hand cards and compatible cards in the combined unknown support for each player's stack                             |
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

For a fixed perspective, the summary is invariant to hidden-draw permutations,
redistribution of unknown support between opponent hand and draw at fixed public
counts, opponent hand order, and seed/RNG/log changes. The implementation
validates that visible zones and unknown support form a complete, disjoint
partition of all 30 property cards.

### Canonical rule reuse

The summary does not reproduce policy-local rules:

- district and match score use `districtScore` and `scoreGame`;
- placement support uses `placementAllowed`;
- deed targets use `developmentCost`;
- exact income sources use the shared engine helper `incomeForResult`.

## Objective Action Deltas

`strategicActionDeltasV0` applies each canonical legal `ActionWindow` action
without advancing hidden randomness and compares the resulting factual
summaries. It reports district-point-margin change, developed-rank-margin
change, resource-margin change, acted-district score-margin change, provisional
outcome before and after, whether the card play remains available, and where a
played card goes (developed, deed, first-reshuffle discard, or dead discard).
These are exact state deltas, not Q-values.

## Strategic Position Catalog

Implementation: `src/botEval/strategicPositionCatalog.ts`, current catalog
version 2. Every position has a stable ID and catalog version, a complete
30-card partition accepted by rollout determinization, a fixed Player A
perspective with canonical legal focus actions, a strategic thesis with
expected factual relationships, and may name a qualitative preferred action.

Catalog v2 uses canonical `D1`-`D5` coordinates with the sole Excuse fixed at
`D3`; validation rejects noncanonical layouts; optionality mirrors swap two
complete Pawn lanes only.

| ID                                          | concept                                                   |
| :------------------------------------------ | :-------------------------------------------------------- |
| `minimum-winning-coalition`                 | pivotal fifth district versus fortress reinforcement      |
| `tie-denial-restores-match`                 | loss-to-tie denial and the global district count          |
| `rank-tiebreak-conversion`                  | conditional value of developed rank at 2-2                |
| `known-hand-optionality-original`           | preserving a guaranteed hand continuation                 |
| `known-hand-optionality-mirror`             | the same known-hand option with district roles reversed   |
| `unknown-pool-optionality-original`         | preserving placement support for possible future draws    |
| `unknown-pool-optionality-mirror`           | the same unknown-pool option with district roles reversed |
| `known-hand-optionality-holdout-original`   | independent guaranteed-continuation holdout               |
| `known-hand-optionality-holdout-mirror`     | the Cave/Castle holdout with district roles reversed      |
| `unknown-pool-optionality-holdout-original` | independent hidden-draw resource holdout                  |
| `unknown-pool-optionality-holdout-mirror`   | the Painter/Desert holdout with district roles reversed   |
| `deed-fork-affordable`                      | immediate completion plus a remaining card play           |
| `deed-fork-inaccessible`                    | identical progress with different current feasibility     |
| `sale-before-first-reshuffle`               | sale remains in future draw circulation                   |
| `sale-after-first-reshuffle`                | the same sale goes to a dead discard                      |
| `ace-aware-control`                         | Ace bonuses reverse the raw-rank district comparison      |

The optionality families test continuation preservation: a focus action that
keeps a guaranteed or possible future continuation versus one that forfeits it.
Each mirror swaps complete target lanes, including marker masks and both
players' stacks, so a policy that values the preserved option must reverse its
physical district choice rather than follow stable ordering. Holdout cases
repeat the relationships with different cards, payment suits, and lane pairs.

Catalog preferences are reviewed hypotheses. Tests assert that setups and
stated factual relationships are correct; they do not require existing bots to
select the preferred action. Future positions should prefer relational
expectations over fragile exact floating-point scores.

## Comparator

```powershell
yarn bot:eval strategic-positions --repetitions 1   # smoke
yarn bot:eval strategic-positions --repetitions 8   # initial stability screen
```

Default variants are direct heuristic v2 (`heuristic-v2-direct`), V2 Hard
(`rollout-search-v2-hard`), and TD V2 Medium (`td-root-search-v2-medium`). The
opt-in `td-root-search-v2-800-visits` diagnostic clones TD V2 Medium and changes
only sampled worlds from 10 to 50. `--positions` and `--variants` accept unique
comma-separated IDs; unknown or duplicate IDs fail fast.

Each `(position, repetition)` supplies the same explicit random seed to every
variant, independent of bot ID. Positions in a declared counterfactual group,
including the optionality mirrors, share that seed as well. Results include
selected stable action keys, pairwise preference assessments, focus
scores/ranks, search visits/values, full diagnostics, the information-safe
summary, and a canonical payload fingerprint. Actions outside a preference's
declared comparison set are recorded as unassessed, not as mismatches. Search
means come from adaptive, potentially unequal visits, so gaps are diagnostics
within one position and variant, not fixed-budget paired estimates.

Repeated seeds characterize stability in the same fixed cases. They are not
independent games and do not prove match-equity improvement. Generated
`positions.json` and `summary.md` files live under ignored
`artifacts/ts-bot-evals/` by default and are observations, not golden answers.

## Experiment Outcome Summary

Durable conclusions from the characterization work; detailed blow-by-blow
results live in ignored artifacts and git history.

- Catalog-v1 optionality fixtures used `D0`-`D4` and sometimes moved the Excuse
  lane. Model-attribution claims from those results are invalid; the artifacts
  remain historical only.
- Corrected catalog-v2 reruns showed that deeper search (800 visits) can
  compensate for physical-lane priors in the fixtures, so the fixtures do not
  prove model symmetry either way.
- The replay-wide direct audit over 10,000 ordinary decisions under all 24
  fixed-D3 permutations did establish meaningful symmetry violation in the
  deployed model. The audit, augmentation outcomes, and the architecture plan
  live in [the district-symmetry design note](district-symmetry.md).
- Heuristic rollout's uncertain-draw resource-preservation blind spot is a
  separate diagnostic; heuristic-rollout substitution is not a general fix.

## Invariants Protected by Tests

- complete/disjoint card partition and hidden-world determinization;
- canonical `D1`-`D5` coordinates with the sole Excuse fixed at `D3`;
- hidden-assignment invariance;
- canonical score, Ace bonus, deed, placement, tax, and income consistency;
- stable ordering and JSON-safe plain data; focus actions remain legal;
- exact global-district, tiebreak, optionality, deed, and reshuffle
  relationships; common seeds across variants and counterfactual groups;
- exact D1/D2/D4/D5 observation-block and action-feature permutation with D3
  fixed, including inverse restoration;
- S4 training augmentation preserves targets and candidate order, transforms
  TD-lambda trajectories coherently, and leaves control mode untouched;
- forced-root traces reuse one hidden-world sample and one engine/rollout seed
  pair across both roots and guides, remain terminal, and cannot mutate normal
  search behavior.

## Non-Goals

V0 does not change heuristic v2, rollout backup, TD encoding, model dimensions,
or the bridge contract. It does not attempt horizon distributions, district
outcome kernels, shared future-action allocation, or a calibrated match-equity
model. The next district-symmetry step is an architecture change, not another
augmentation weight or heuristic patch; see
[the district-symmetry design note](district-symmetry.md).
