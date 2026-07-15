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

Catalog v1 contains 16 positions. The catalog version remains `1` for this
additive holdout extension so the legacy comparison seed namespace stays
stable; fresh pair IDs give the holdouts independent random groups.
Adding machine-readable trace metadata intentionally changes the full-payload
fingerprints of the four pre-existing optionality positions once. Unrelated
position fingerprints stay stable because absent metadata is omitted from the
fingerprint payload.

| ID                                          | concept                                                   |
| :------------------------------------------ | :-------------------------------------------------------- |
| `minimum-winning-coalition`                 | pivotal fifth district versus fortress reinforcement      |
| `tie-denial-restores-match`                 | loss-to-tie denial and the global district count          |
| `rank-tiebreak-conversion`                  | conditional value of developed rank at 2–2                |
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

The first four optionality cases replace the confounded catalog-v0 endpoint case.
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

The four holdout cases repeat those relationships without reusing the original
mechanism. The known-hand holdout plays The Desert with a Suns-Wyrms payment
across complete Cave/Castle lanes in D0/D3. Preserving The Cave keeps a
tax-safe Origin play reachable after selling the Ace of Leaves; it flips the
remaining 7-8 district to 9-8 and the match to 3-1. The unknown-pool holdout
plays The Mountain with a Moons-Suns payment across complete Painter/Desert
lanes in D2/D3. Only a preserved Painter can receive a possible Market draw;
the configured hidden distractors have equal support, and the controlled
continuation changes the target from 3-5 to 9-5 and the match to 3-1. These
positions carry explicit optionality-trace metadata naming their source type,
target card, semantic lanes, and focus actions. The tracer consumes that
metadata rather than inferring cards or lanes from position IDs.

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

Three additional opt-in variants form a per-hook guidance matrix. Each retains
the 800-visit configuration and TD leaf setting, uses heuristic v2 for any
heuristic hook, and remains excluded from defaults:

| ID                                                    | root         | rollout      | leaf |
| :---------------------------------------------------- | :----------- | :----------- | :--- |
| `td-root-search-v2-800-visits-heuristic-root`         | heuristic v2 | TD           | TD   |
| `td-root-search-v2-800-visits-heuristic-rollout`      | TD           | heuristic v2 | TD   |
| `td-root-search-v2-800-visits-heuristic-root-rollout` | heuristic v2 | heuristic v2 | TD   |

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

This motivated a narrow root-versus-rollout guidance ablation on the known-hand
pair. Leaf guidance was held fixed because terminal simulations never invoke
the non-terminal leaf estimator.

## Step 4 Guidance-Ablation Outcome

The 2026-07-14 diagnostic used the default model pack
`td-two-stage-imported-20260706-hard-step-30000`, 800 root visits, and the same
positions, seeds, legal actions, and action consequences. A fresh all-TD rerun
on the five failure seeds reproduced every prior choice, focus signal, and
search diagnostic exactly.

On known-hand seeds 7, 10, 14, 17, and 18 and their mirrors, destructive
overwrites were:

| root         | rollout      | harmful decisions (of 10) |
| :----------- | :----------- | ------------------------: |
| TD           | TD           |                         5 |
| heuristic v2 | TD           |                         5 |
| TD           | heuristic v2 |                         0 |
| heuristic v2 | heuristic v2 |                         0 |

Changing only root priorities removed none of the five failures. Changing only
rollout play removed all five harmful overwrites, despite retaining the same TD
root priors. The rollout-only variant then produced no harmful overwrite across
all known-hand seeds 0–47: 96 mirrored decisions including the fresh 24–47
seed extension.

The broader optionality check ruled out treating that result as a ready-made
hybrid bot:

- Across all four optionality cases and seeds 0–23, all-TD at 800 visits made 5
  harmful choices in 96 decisions.
- TD root plus heuristic-v2 rollout made 24 harmful choices, all in the
  unknown-pool mirror. Its TD root prior strongly favored the same physical D4
  lane, and heuristic rollout did not overcome that allocation.
- Heuristic-v2 root plus heuristic-v2 rollout made 1 harmful choice in 96. It
  occurred in unknown-pool original seed 10, where preserve and overwrite each
  received 299 visits and their terminal means differed by about `0.000025`.
- Heuristic-v2 root plus TD rollout made no harmful unknown-pool choice, but it
  fixed none of the five selected known-hand failures.

Every new decision used exactly 800 visits, expanded every legal root action,
and had terminal rate 1. Leaf guidance is therefore ruled out for these
fixtures. The known-hand residual is localized to TD rollout play, while the
unknown-pool result exposes a separate root/rollout interaction. No single-hook
substitution is a general remedy, and the nearly all-heuristic combined variant
is effectively another heuristic-search diagnostic rather than evidence for a
TD improvement.

## Step 5 Matched Forced-Rollout Outcome

The 2026-07-14 follow-up removed root selection and adaptive visit allocation
from the comparison. For each action-local scenario index, it sampled the
hidden world once, then forced preserve and overwrite through the same hidden
assignment, engine seed, and rollout seed under both TD and heuristic-v2
rollout play. Every trace reached terminal scoring before depth 40, so the TD
leaf evaluator was again inactive.

The known-hand artifact contains 2,000 terminal traces: both mirrors, failure
seeds 7, 10, 14, 17, and 18, 50 matched scenarios per seed, two forced roots,
and two rollout guides. Its aggregate results were:

| position | rollout      | preserve mean | overwrite mean | preserve minus overwrite |
| :------- | :----------- | ------------: | -------------: | -----------------------: |
| original | TD           |        -0.214 |         -0.109 |                   -0.105 |
| mirror   | TD           |         0.461 |         -0.670 |                    1.132 |
| original | heuristic v2 |         0.007 |         -0.330 |                    0.337 |
| mirror   | heuristic v2 |         0.013 |         -0.330 |                    0.343 |

Heuristic-v2 rollout play valued preserve consistently in both lane
orientations. TD rollout play instead changed by more than a full score unit
when the two complete lanes were swapped. Across the five individual seeds,
the TD preserve-minus-overwrite gap was negative in four original cases and
strongly positive in all five mirrors. After normalizing D1/D4 to valuable and
alternative lane roles, the Player A preserve trajectory matched across the
mirror in only 109 of 250 TD scenarios, versus 229 of 250 heuristic scenarios.
The most common first TD mismatch, in 53 scenarios, played The Journey into
the same physical D4: that was the alternative lane in the original and the
valuable lane in the mirror.

The Author is not simply invisible to TD at the final decision. On TD-driven
preserve trajectories it was developed in the valuable lane in 128 of 250
original scenarios and 140 of 250 mirrors. TD also declined it after it became
legal in 50 original and 40 mirror scenarios; in another 72 and 70 scenarios,
respectively, its valuable-lane play never became legal. Recurring earlier
losses included selling The Diplomat or The Mill instead of the setup Ace of
Knots, developing another card or deed into a pivotal lane, and using the last
card play on another development. On states reached by heuristic rollout play,
TD would often choose The Author too. The central TD blind spot is therefore
the multi-turn preparation and lane-role consistency needed to preserve the
continuation, not merely the final Author action score.

The unknown-pool artifact contains 800 terminal traces over repetitions 1 and
6, their mirrors, and the same 50-scenario cycle. The Penitent was the next
Player A draw in 20 of the 100 sampled scenarios and was assigned to the
opponent in the other 80. The following continuation and win counts held in
each mirror orientation when restricted to those 20 reachable draws:

| forced root and rollout  | Penitent realized | Player A wins |
| :----------------------- | ----------------: | ------------: |
| preserve + TD            |             19/20 |         20/20 |
| overwrite + TD           |              0/20 |          1/20 |
| preserve + heuristic v2  |              0/20 |          0/20 |
| overwrite + heuristic v2 |              0/20 |          0/20 |

In every reachable-draw scenario, the first Player A disagreement was the
same. TD ended the Sailor turn and retained Suns/Wyrms; heuristic v2 traded a
Wyrms token for Moons and then traded Suns for Moons before drawing. The
Penitent consequently reached the heuristic trajectory's hand with its
payment gone and was never legal in the valuable lane. Heuristic-v2 rollout
gave preserve and overwrite essentially identical terminal means, because it
discarded the option's token support before learning whether the card would be
drawn. TD preserve exceeded overwrite in both mirrors and missed the Penitent
only once after it was legal.

Together with Step 4, this explains the unknown-pool interaction. TD root
guidance has a physical D4 prior. Heuristic rollout removes the future
Penitent value that could counter that prior, so the physical-location bias is
left to decide the root. This is a fixture-level causal diagnosis of the
recorded search behavior, not evidence that either rollout guide is generally
stronger in full games.

## Step 6 Independent Holdout Outcome

The 2026-07-14 holdout repeated both optionality mechanisms with different
cards, payment suits, lane pairs, and district stacks. Repetitions 0-23 covered
four positions and seven variants, for 672 root decisions. A destructive
overwrite means playing the root card into the only lane that can receive the
valuable continuation. Sales and trades outside the two focus actions remain
unassessed rather than being treated automatically as either correct or
harmful.

The characterization and unknown-pool trace artifacts were generated before
one Market expected-fact sentence was corrected to remove an inaccurate
reference to taxation. Expected facts participate in the case fingerprint, so
those saved descriptive fingerprints differ from a fresh run. The executable
state, legal actions, random seeds, and measured results are unchanged; the
continuation test now explicitly confirms that Leaves and Knots both remain at
five through the intervening opponent turn.

| variant                               | preserve | destructive overwrite | unassessed other |
| :------------------------------------ | -------: | --------------------: | ---------------: |
| direct heuristic v2                   |       96 |                     0 |                0 |
| V2 Hard                               |       72 |                     0 |               24 |
| TD V2 Medium, 160 visits              |       69 |                    14 |               13 |
| all-TD, 800 visits                    |       73 |                     1 |               22 |
| heuristic root + TD rollout           |       82 |                     0 |               14 |
| TD root + heuristic-v2 rollout        |       44 |                    18 |               34 |
| heuristic root + heuristic-v2 rollout |       72 |                     0 |               24 |

The extra TD search reduced harmful choices from 14 to one. The sole all-TD
failure was known-hand mirror repetition 13. In that matched pair the model's
root prior heavily favored physical D3 in both orientations: D3 was the
preserve action in the original and the overwrite action in the mirror. The
search consequently gave the mirror's overwrite 789 visits and its preserve
only six.

A 400-trace forced-root check on that repetition showed that TD rollout did
not actually prefer destroying the Origin option. Preserve beat overwrite by
`+0.294` in the original and `+0.318` in the mirror. Heuristic-v2 rollout's
corresponding gaps were `+0.382` and `+0.379`. TD realized Origin in 37 of 50
original preserve traces and 40 of 50 mirror preserve traces; every overwrite
made Origin permanently illegal. The old Author fixture's severe
orientation-dependent rollout valuation therefore did not reproduce. Recurring physical
D4 placements and longer setup choices remain visible inside the traces, but
this holdout's one 800-visit failure is primarily a root-prior and
under-exploration failure.

The hidden-draw mechanism did reproduce. Across 100 matched scenarios per
orientation, Market was assigned to Player A 29 times. TD preserve retained
its payment and realized Market in all 29 in both orientations. Under
heuristic-v2 rollout, preserve realized it only twice and was never able to
make it legal in the other 27. The first disagreement was consistently that
heuristic v2 traded Leaves for Wyrms while TD ended the turn; a later trade
spent more of the Knots reserve. As a result, heuristic-v2 rollout gave
preserve and overwrite effectively identical outcomes even though its overall
play won more of these particular traces. This isolates an optionality blind
spot rather than claiming that the guide is generally weak.

The holdout therefore supports three conclusions:

- TD has a learned physical-district bias that more visits reduce but do not
  remove.
- Heuristic v2 does not consistently protect resources for an uncertain but
  valuable hidden draw.
- Replacing TD rollout play with heuristic-v2 rollout is not a general fix: it
  increased harmful holdout choices from one to 18 at the 800-visit budget.

The narrowest justified TD experiment is district-permutation augmentation (or
an equivalent permutation-aware encoding), because it teaches an exact game
symmetry without assigning subjective strategic bonuses. A trained candidate
should first be checked on untouched repetitions 24-47, then in full games.
The resource-option mechanism should remain a separate diagnostic; these
traces do not justify hand-authoring a heuristic-v3 bonus or adopting the
failed rollout hybrid.

## Invariants Protected by Tests

- complete/disjoint card partition and hidden-world determinization;
- hidden-assignment invariance;
- canonical score, Ace bonus, deed, placement, tax, and income consistency;
- stable ordering and JSON-safe plain data;
- focus actions remain canonical and legal;
- exact global-district, tiebreak, both mirrored known-hand/unknown-pool
  optionality families, deed, and reshuffle relationships;
- common random seeds across comparator variants and counterfactual groups;
- deterministic repetition offsets and explicit comparison subsets;
- per-position/variant stability, focus-gap, and counterfactual reporting;
- deterministic comparison output apart from measured latency and artifact
  metadata;
- forced-root traces reuse one hidden-world sample and one engine/rollout seed
  pair across both roots and both guides, remain terminal, and cannot mutate
  normal search behavior or shared sampled worlds.

## Non-Goals and Reserved Follow-Up

V0 does not change heuristic v2, rollout backup, TD encoding, model dimensions,
or the bridge contract. It also does not attempt the horizon distribution,
district outcome kernels, shared future-action allocation, or calibrated match
equity model discussed for later work.

The distinct mirrored holdout is now part of the catalog. Repetitions 0-23 were
consumed by the characterization screen; 24-47 remain untouched for evaluating
the next bot change. No training-data change, feature change, search hybrid, or
narrow heuristic potential is a promotion candidate without that reserved
screen and full-game evaluation.
