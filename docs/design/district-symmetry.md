# District Symmetry (TD Models)

Status: defect confirmed; augmentation interventions closed; architectural fix
not started. The "Open Design Space" section is the open work.

## Background

Magnate's board has five districts: four Pawn districts (canonical `D1`, `D2`,
`D4`, `D5`) and the Excuse lane (`D3`). No rule references a Pawn district's
slot identity: placement, income, and scoring depend only on suits and the
local property chains. Permuting the four Pawn districts (24 ways, the
symmetric group S4) while moving their contents with them should therefore
leave the game, the optimal policy, and the position value unchanged. `D3` is
excluded because the Excuse lane behaves differently, so the symmetry group is
"fixed-D3 S4".

The TD observation and action encoding carries absolute district slots, so
invariance is not automatic. It has to be learned from data or built into the
model.

## Defect

The deployed TD model violates the symmetry: relabeling the four Pawn districts
changes action probabilities and value predictions. The model is encoding
location information that the rules guarantee is irrelevant, which is a
correctness-and-generalization defect rather than a tuning question, and it
drives the district-symmetry research line.

## Evidence

- Catalog-v1 optionality fixtures used `D0`-`D4` numbering and sometimes moved
  the Excuse lane. Model-attribution claims from those artifacts are invalid;
  the artifacts remain historical only.
- Corrected catalog-v2 reruns showed deeper search (800 visits) can compensate
  for physical-lane priors in the fixtures, so the fixtures prove neither
  symmetry nor violation.
- The decisive evidence is the replay-wide direct audit: 10,000 ordinary
  decision rows sampled from all validation opponent shards, every one of the
  24 fixed-D3 permutations applied, compared against the identity
  transformation. It established a meaningful exact-symmetry violation in the
  deployed model and justified the controlled augmentation experiments.
- The audit reports aggregate top-action match rate, per-sample top-action
  flips, Jensen-Shannon divergence, max probability delta, max centered-logit
  delta, and value absolute delta, with strata by action type, probability
  margin, district availability, permutation, and Pawn-slot effects. It
  measures symmetry, not playing strength.

## Predeclared Gates

The frozen pilot manifests under `configs/td-training/` are the protocol
source of truth. Both pilots predeclared:

- Heldout noninferiority vs a matched control: maximum value MSE ratio `1.05`
  and maximum opponent cross-entropy increase `0.01`.
- Direct symmetry on 10,000 samples from all 100 validation opponent shards,
  all 24 permutations, with a frozen sampling seed:
  - pairwise top-action agreement `>= 0.95`,
  - probability-drift reduction vs control `>= 0.5`,
  - value-drift reduction vs control `>= 0.5` (in the ablation manifest only).
- Role: `diagnostic-required-but-not-sufficient`. Symmetry improvement alone is
  never sufficient for promotion.
- Strategic repetitions `0-23` are development and may inform candidate
  selection; repetitions `24-47` are reserved and may not. Full-game series
  (100 games per side, paired seeds, seat-swapped) stay blocked until heldout
  noninferiority and the direct symmetry gates pass, and the candidate must be
  frozen before reserved evaluation begins.

Augmentation runs also require an explicit experiment seed and a matched
control, and write experimental packs only under the ignored
`public/model-packs-experiments/` index; the deployed
`public/model-packs/index.json` and bot defaults must not change.

## Interventions And Outcomes

Both interventions are closed and neither is promotion-eligible.

- **Random S4 augmentation** (`district-s4-ablation-pilot-v1`): continued
  training from the reconstructed hard step-30,000 pair, arms
  `continued-control` vs `s4-augmented`, two seeds, 5,000 updates, replay split
  800 training / 100 validation shards, diagnostic cross-component packs to
  separate value and opponent effects. Recorded outcome: improved heldout
  metrics, but failed the direct symmetry gates.
- **Opponent-only complete orbit** (`district-s4-opponent-orbit-pilot-v1`):
  value checkpoints fixed from the pilot-1 augmented runs; opponent training
  averaged over the complete 24-permutation orbit (24 copies per raw sample)
  versus the random single-permutation control, ordinary soft-target
  cross-entropy, no new loss coefficient, value not retrained. Recorded
  outcome: improved heldout metrics, but failed the direct symmetry gates.

Interpretation: augmentation reduces but does not eliminate location bias; the
architecture remains free to encode absolute slot identity. The frozen guardrail
recorded this in advance: if complete-orbit training still fails the symmetry
gate, investigate an equivariant architecture rather than adding subjective
action boosts.

## Constraints For The Architecture Fix

- Preserve the replay row format, the checkpoint registry/manifest contract
  (`models/td_checkpoints/manifest.json`), and the browser model-pack export
  contract.
- Keep determinism: seeded RNG only, stable candidate ordering, no hidden
  global state.
- TD training and encoding stay standard-ruleset only; do not combine the
  architecture change with encoding, reward, replay-mixture, or heuristic
  changes in the same intervention.
- Do not change deployed defaults (`public/model-packs/index.json`, bot
  profiles) from an experiment.
- Follow the promotion protocol: freeze a new experiment manifest with
  predeclared gates and matched controls, freeze the candidate, then run
  reserved strategic evaluation and full games.
- Any model input/output or pack-schema change needs explicit versioned
  handling so existing checkpoints and browser exports stay readable.

## Open Design Space

The goal is a model that is equivariant by construction: a fixed-D3 S4
relabeling should be a no-op (or an exactly transformed output), not something
training has to learn approximately.

Candidate directions, not decisions:

1. **Weight sharing / parameter tying** across the four Pawn district slots with
   relative inputs, so the same parameters apply wherever a district sits.
2. **Relational features**: express district features relative to the
   perspective and to other districts instead of by absolute slot identity.
3. **Group averaging at inference** over the 24 permutations (approximate;
   costs 24x inference and complicates determinism and visit budgets).
4. **Symmetry-aware loss or regularizer** (augmentation-style approaches are
   already shown insufficient).

Open questions to settle in the design:

- Target scope: value model, opponent model, or both.
- Effect on TD-root search, which uses the model for root priors, rollout
  action choice, and leaf values.
- How to keep checkpoint and browser-export compatibility through a model
  change.
- Which acceptance tests become architectural invariants (for example
  permutation-equivariant outputs) versus benchmark evidence.

## Tooling

- Audit:
  `yarn bot:eval td-symmetry (--replay-dir <path> | --replay-list <path>) [--sample-size <count>] [--sampling-seed <text>] [--pack-id <id>] [--model-index-path <path>] [--out-dir <path>]`
- Augmentation training flags: `scripts.train_td --district-augmentation none|s4|s4-orbit`
  with an explicit experiment seed.
- Implementation: `trainer/td/symmetry_augmentation.py`,
  `src/botEval/tdSymmetry.ts`, tests in `src/botEval/tdSymmetry.test.ts`.
- Frozen pilot manifests: `configs/td-training/district-s4-ablation-pilot-v1.json`,
  `configs/td-training/district-s4-opponent-orbit-pilot-v1.json`. These are
  completed one-off experiments; do not re-run them as normal loop work.

## Ownership

- Agent: audit runs, implementation, tests, manifest and report tooling,
  artifact analysis, gate calls, docs.
- Owner: run training and evaluation series on the project machine, then hand
  artifact paths back.

## Next Step

Complete the open design above, freeze a new experiment manifest with
predeclared gates and a matched control, and only then implement. The
browser default stays the promoted step-9,000 pack until a candidate passes the
full promotion protocol.
