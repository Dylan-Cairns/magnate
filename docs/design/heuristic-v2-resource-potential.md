# Heuristic V2 Resource Potential

Status: implemented 2026-10-07, not yet gated. Bulk benchmark series are run
manually by the project owner and analyzed by an agent.

## Background

Log `magnate-log-2026-10-04T16-10-35.json` (Easy profile, standard ruleset, seed
`seed-1791129873281`) shows PlayerB at turn 22 making three trades in a row:
`Knots -> Suns`, then `Suns -> Moons`, then `Moons -> Suns`. Trades are 3:1
(`src/engine/reducer.ts`), so the sequence destroyed six tokens and the last two
trades cancelled each other. Only the first trade had any effect, and that effect
was `Knots -1, Suns +1`; the trades that would have enabled a district action
(into Waves or Wyrms, the suits of the incomplete rank-7 deed `22`) were never
taken.

Root causes found by deterministic replay (`policyRandomSeedForState` reproduces
the browser decision through `selectRolloutSearchActionSync`):

- Heuristic v2 priced a trade by a smooth, recomputed demand proxy
  (`src/policies/tokenValueV2.ts`). A 3:1 loss could score positive once the
  received suit looked "demanded": at decision 2, `Knots -> Waves` scored
  `+0.47` bank value.
- `SMALL_ACTION_BASELINE = 0.05` was added to every non-`end-turn` action, so a
  value-destroying trade (`Knots -> Suns`, `-0.034` before baseline) still
  outranked ending the turn.
- The search's rollout playout uses the same heuristic, so rollouts traded too;
  because trades leave the districts unchanged, the root decision reduced to a
  weak resource term whose 20-world estimate was noise-dominated (in world 0
  `Knots -> Suns` was the worst option, on the 20-world mean the best). Sync and
  browser runs diverged on the near-tie at decision 4.

Heuristic v1 had an explicit "a trade is only worth what it unlocks" rule
(`heuristicScorer.ts` `tradeContextScore`); v2 replaced it with the demand proxy
and dropped the requirement.

## Principle

Resources are means, not score. A token is worth only what it lets a player
_do_: progress a concrete unfinished deed, or pay for the next placement. That
value is step-shaped at completion thresholds, and it is stable within a turn —
it is anchored to the board, not recomputed from the current portfolio. Under
this rule a 3:1 trade is a strict loss unless the received token is worth more
than the three surplus tokens it costs, which is exactly the "unlock" idea
expressed continuously rather than as a guard. The same potential prices both
actions and search leaves, so the two valuations cannot disagree.

The same principle extends globally: heuristic v2 is the single heuristic for
all profiles and the training/reference surfaces. This change is applied
everywhere rather than forking a browser-only variant, and past
heuristic-dependent gates are expected to be re-baselined.

## Design

**Target-anchored potential** (`src/policies/resourcePotentialV2.ts`).

For each unfinished, non-Court deed on the board, the per-token value is the
deed's own scoring-plus-earning demand divided by its remaining cost, and the
suit is capped at that remaining count. A suit's need value is the best such
deed wanting it. The potential is

```
sum over suits of  covered(count, need.remaining) * max(need.perTokenValue, SURPLUS)
                 + surplus(count) * SURPLUS
```

- `SURPLUS_TOKEN_VALUE = 0.05` is one surplus token.
- Monotone non-decreasing in every suit count, so holding is never penalized.
- `3 -> 1` into surplus loses two surplus tokens; the same conversion into a
  needed suit wins only if `need.perTokenValue > 3 * SURPLUS`.
- Courts are excluded: they stay owned by `courtPotentialV2.ts`.

**Action term** (`heuristicScorerV2.ts`). A trade has no scoring or earning
delta, so its only value signal is the resource term. Trades now add
`TRADE_RESOURCE_POTENTIAL_WEIGHT * resourcePotentialDeltaForActionV2(...)`
(`0.1`) on top of the existing demand-bank term, and the `SMALL_ACTION_BASELINE`
is withheld from a trade whose own value is negative. District actions keep the
existing terms and baseline unchanged, so their exact scores do not move.

**Leaf** (`searchStateEvaluator.ts`). `resourceQualityTerm` gains a
`potentialDiffTerm` over the same `resourcePotentialV2` (weight `0.2`, tanh
scale `1.5`); the other sub-weights are rebalanced to keep the term in `[-1, 1]`.

**Search hard dominance** (`rolloutSearchCore.ts`). After the turn's card has
been committed, a trade that unlocks no new district action and strictly reduces
both total tokens and potential is removed from the root candidate set. It never
prunes before the card is played, where a trade can still set up the turn's own
placement. If pruning leaves only `end-turn`, that is selected.

## Knobs

- `SURPLUS_TOKEN_VALUE` (`resourcePotentialV2.ts`, default `0.05`): surplus
  token value; sets the break-even for a 3:1 conversion.
- `TRADE_RESOURCE_POTENTIAL_WEIGHT` (`heuristicScorerV2.ts`, default `0.1`): the
  trade potential scale relative to the demand-bank term.
- Leaf `potentialDiffTerm` weight/scale (`searchStateEvaluator.ts`).
- `pruneDominatedRootActions` is a hard rule, not a tunable weight.

## Invariants

- Potential is monotone non-decreasing in every held suit count.
- A surplus-only trade strictly lowers potential; a trade into a suit an
  unfinished deed still needs raises it.
- A no-unlock trade scores below `end-turn`; an unlock trade scores above it.
- Non-trade action scores are unchanged.
- After the card is played, no-unlock reducing trades are pruned; before it,
  nothing is pruned.

## Predeclared gates

Tier 0 definitions: an action _unlocks a district action_ if, after applying it,
some `develop-deed` / `develop-outright` / `buy-deed` becomes legal that was not
legal before. A _no-unlock trade_ is a trade that does not.

- **T1.1 Dominance:** every no-unlock trade scores below `end-turn` by at least
  `0.05` heuristic points.
- **T1.2 Unlock:** a trade that crosses a completion/affordability threshold
  scores above `end-turn` by at least `0.05`, and the policy selects it.
- **T1.3 Monotonicity:** potential never decreases as a needed suit grows.
- **T1.4 Consistency:** the sign of a conversion's action-term resource delta
  matches its leaf resource-term delta on a fixture set.
- **T1.5 Cycle-freedom:** greedy same-turn trading from a fixed state takes zero
  non-threshold trades.
- **T1.6 Non-trade invariance:** develop / buy / sell exact scores unchanged.
- **T2.7 Incident replay:** `seed-1791129873281`, turn 22 PlayerB chooses
  `end-turn` at the first no-unlock decision.
- **T2.8 Seed stability:** on a dominated fixture, 32/32 seeds choose `end-turn`.
- **T2.9 Search dominance:** a strictly dominated root action is never selected,
  for all seeds.
- **T3.10 No-unlock-trade rate**: over _post-card_ (`cardPlayedThisTurn`)
  decisions in a fixed, policy-independent corpus of `ActionWindow` decisions
  with a legal trade, drops to `<= 1%`. Pre-card conversions are out of scope
  for this change (see Gate Results).
- **T3.11 Guardrail:** on 32 fixed seeds, district win rate versus the reference
  is within 3pp of the pre-change profile and the resource diagnostic improves.

Thresholds are fixed before measuring. `T1.*`, `T2.7`, and `T2.9` are covered
by tests today; `T2.8` and Tier 3 require the benchmark series below.

## Gate Results

Measured 2026-10-07 against the pre-change commit (stash) and the change, on a
120-decision neutral corpus (games generated by the v1 heuristic, every
`ActionWindow` decision with a legal trade) plus a 16-game guardrail
(`rollout-search-v2-medium` vs the v1 heuristic).

- **T1.\***: pass (`yarn vitest run src/policies/heuristicScorerV2.test.ts`,
  `rolloutSearchCore.test.ts`, `searchStateEvaluator.test.ts`).
- **T2.7**: pass. Turn 22, PlayerB now selects `end-turn` where the log recorded
  `trade:Knots:Suns`; the `Suns->Moons->Suns` spiral is gone.
- **T2.9**: pass (pruned root actions cannot be selected).
- **T3.10 no-unlock-trade rate (post-card scope)**: pass. Post-card no-unlock
  selections go from `2/45 (4.44%)` before to `0/45` after. Over all decisions
  the rate moves `4/120 (3.33%)` to `3/120 (2.5%)`; the three "after" residuals
  are `cardPlayedThisTurn: false` with no unlock trade available, where the
  dominance rule deliberately does not apply. The bar was written over all
  decisions and narrowed to the post-card class with owner sign-off, because the
  change's stated scope is the post-card spiral; the pre-card class is a
  separate, unclaimed question rather than a passed gate.
- **T3.11 guardrail**: pass at reduced size. Candidate win rate rises `10/16
(62.5%)` to `11/16 (68.75%)` on an 8-seed x 2-seat smoke (16 games); the
  predeclared 32-seed guardrail is unspent.

Corpus counts are small (four to six selected trades), so the overall rate is a
weak estimate; a larger neutral corpus is the first thing to widen if the
pre-card class becomes in scope.

## Baseline protocol

The corpus is neutral: games are generated by the v1 heuristic for both seats
and every `ActionWindow` decision with a legal trade is sampled, up to a fixed
cap. It is never filtered by the policy under test, so the before and after runs
see identical states. Record, at the pre-change commit (via `git stash`) and
again after, the selected action key per decision, the post-card
no-unlock-trade rate, and the guardrail win rate. Outputs live under ignored
`artifacts/heuristic-v2-resource-potential/`. Determinism plus git means a missed
baseline remains reconstructable from the pre-change commit.

## Non-goals

This does not retune district scoring, income, or the Court term, and does not
attempt hand-card affordability targets (roughing-out trades are outside
`resourcePotentialV2`). Hard dominance is scoped to post-card decisions;
no-unlock trades before the card is played are unclaimed. It does not change the
bridge contract or TD encodings.
