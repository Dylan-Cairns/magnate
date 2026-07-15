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

These catalog-v1 results initially suggested three conclusions: a learned
physical-district bias, an uncertain-draw resource blind spot in heuristic v2,
and failure of heuristic-v2 rollout substitution as a general fix. Catalog v1
could not support the first conclusion because its coordinates were invalid.
The resource trace and the failed rollout hybrid remain informative because
they come from matched continuations rather than the model's district encoding.

## Step 7 Canonical-Coordinate Audit

The catalog-v1 states used district IDs `D0`-`D4`, while normal games and TD
training use `D1`-`D5`. Encoding v2 sorts five 28-feature district blocks by ID
and also encodes the numeric district suffix on each action. Catalog v1
therefore introduced an unseen `D0` action and shifted every action-to-board
block association. Some mirrors also moved the empty Excuse lane even though
normal setup always fixes the Excuse at `D3`. The apparent physical-location
bias was consequently confounded with out-of-distribution input.

Catalog v2 corrects that mismatch without changing bot behavior:

- every exposed state has exactly `D1`-`D5`;
- `D3` is always the only Excuse district;
- optionality mirrors swap two complete Pawn lanes, not the Excuse;
- catalog validation rejects noncanonical layouts;
- state and comparison seeds use the fresh catalog-v2 namespace.

The corrected 2026-07-14 rerun used all eight optionality positions,
repetitions 0-23, and current all-TD search at 160 and 800 visits. It produced
384 decisions:

| budget     | preserve | destructive overwrite | unassessed other |
| :--------- | -------: | --------------------: | ---------------: |
| 160 visits |      168 |                     8 |               16 |
| 800 visits |      167 |                     0 |               25 |

For comparison, catalog v1 recorded 44 destructive overwrites at 160 visits
and six at 800 across the same number of optionality decisions. More
importantly, the eight remaining catalog-v2 overwrites at 160 visits split
exactly 4-4 across the two physical orientations. At 800 visits every assessed
focus choice preferred preserving the continuation in every original and
mirror position. Non-focus sales and trades remain unassessed, so this does not
claim perfect strategic play.

The corrected final choices therefore do not reproduce catalog v1's apparent
orientation-dependent failure. Normal 160-visit search still made eight harmful
focus choices, but they split evenly across orientations, and at 800 visits
every assessed focus choice preserved the continuation. This shows that deeper
search can compensate in these fixtures; it does not prove that the underlying
TD model is district-symmetric. Corrected root priors still favored physical D5
in both orientations of several optionality pairs, so the stronger question
required ordinary full-game states rather than another hand-built fixture.

The heuristic-v2 uncertain-resource blind spot and rejection of heuristic
rollout substitution remain separate findings. Repetitions 24-47 remain
untouched for evaluating a future candidate.

## Step 8 Replay-Wide Direct Symmetry Audit

The 2026-07-14 direct audit used the deployed July model pack
`td-two-stage-imported-20260706-hard-step-30000` and the complete 900-game V2
Hard teacher replay. That replay contains 163,194 decisions produced with 50
worlds and depth 270. A seeded reservoir selected 10,000 decisions across all
900 game shards. Each selected row was evaluated under all 24 permutations of
the four Pawn districts D1, D2, D4, and D5 while D3 remained fixed. The audit
moved each complete 28-feature district block and every corresponding action
district feature together. This is an exact game symmetry, not a subjective
strategic preference.

The reproducible command surface is `yarn bot:eval td-symmetry --replay-dir
<path> --sample-size <n> --sampling-seed <seed>`; it writes bounded JSON and
Markdown diagnostics under the bot-evaluation artifact directory.

Across 230,000 non-identity comparisons:

- 4,763 of 10,000 decisions changed preferred action under at least one
  equivalent relabeling;
- pairwise preferred-action agreement was 80.44%;
- the mean maximum action-probability change was 0.0618 and the largest was
  0.8262;
- mean absolute value-estimate change was 0.1036 and the largest was 1.7853;
- even when the baseline top action led the runner-up by at least 0.10,
  7,874 of 137,195 relabelings changed the preferred action (94.26% agreement);
- the largest recorded reversal began with a baseline probability margin of
  0.8944, so the effect is not confined to near-ties.

The balanced slot comparison found a replay-wide physical preference, but not
a universal D5 preference. Mean centered action logits were D1 `0.4984`, D2
`0.5318`, D4 `0.6451`, and D5 `0.5971`. D4 had the uniquely highest per-game
mean in 462 of 900 shards, followed by D5 in 272, D1 in 111, and D2 in 55.
Preferred-action agreement was 42% for buy-deed roots, 59% for
develop-outright, 84% for develop-deed, and 71% for choose-income-suit. The D5
prior seen in the optionality fixtures was therefore a real local asymmetry,
but it was not a complete description of the broader learned slot effect.

This audit does not show that every changed choice is strategically worse, and
the replay is not a held-out playing-strength evaluation. It does establish
that the fixed value and action models violate an exact rules symmetry by
meaningful amounts across ordinary game states. A controlled
district-permutation training ablation is therefore justified by direct broad
evidence, independently of the invalid catalog-v1 diagnosis. The ablation must
retain an unaugmented control and unchanged training data, seeds, steps, model
shape, and promotion tests. No model, training path, browser default, or bot
behavior changed during this audit. Detailed results are in the ignored
`td-symmetry-v2-hard-900-primary` artifact.

## Invariants Protected by Tests

- complete/disjoint card partition and hidden-world determinization;
- canonical `D1`-`D5` coordinates with the sole Excuse fixed at `D3`;
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
- exact D1/D2/D4/D5 observation-block and action-feature permutation with D3
  fixed, deterministic replay sampling, symmetric/biased mock-model detection,
  and bounded worst-case retention;
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
