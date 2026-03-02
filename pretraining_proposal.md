# Proposal: Fundamentals Pretraining for Magnate TD Search

## Goal

Improve the bot's starting competence before long self-play runs by training an action model on high-signal, human-informed fundamentals, then using that model directly in `td-search` root action priors and rollout opponent play.

The intent is not to hard-code the final bot. Fundamentals are used as supervised training data and search guidance so self-play starts from more sensible decisions and produces less garbage data.

## Motivation

The current self-play loop is struggling to produce clear improvement. One likely issue is that weak checkpoints mostly generate weak training data, so the learner can imitate or stabilize around poor play instead of discovering better play.

The existing `td-search` stack does not currently use the trained opponent/action model as the root action prior. Root priors and root candidate ranking are still heuristic-based, while the opponent model is used mainly for opponent decisions inside rollouts. Because of that, pretraining only `opponent.pt` would help search indirectly, but it would not reliably improve the bot's own first-move preferences.

This proposal therefore has two linked parts:

1. Pretrain the existing action/opponent model from fundamentals soft labels.
2. Use that pretrained model as a blended root prior in `td-search`, first in Python training/eval and then in the browser policy.

## General Design

Add a pretraining pipeline that generates reachable game states, identifies states matching known fundamentals, labels legal actions with domain-informed preferences, and trains the action/opponent model on those labels.

In the first implementation, reuse the existing `OpponentModel` architecture as a legal-action scorer. The name is imperfect, but the model already consumes an observation plus legal action features and emits logits over those actions. Longer term, it may be worth renaming this concept to an action model.

The deployed bot should still use the learned model and search stack. Fundamentals should not become permanent scripted gameplay rules.

Rule semantics remain TypeScript-owned. Python may orchestrate state generation and labeling, but rule-sensitive legality, transitions, and scoring helpers should come from the TypeScript engine or bridge-driven state transitions rather than duplicated Python rules.

## Pipeline

1. Generate reachable states through the existing TypeScript bridge.
2. Bias generation toward useful scenario coverage instead of relying only on random self-play filtering.
3. Detect useful scenario types from active-player observations, legal actions, and bridge-produced state payloads.
4. Score all legal actions in each matched state using small scenario-specific evaluators.
5. Convert action scores into a soft probability distribution over the current legal action list.
6. Store TD opponent/action replay rows with `actionProbs` plus audit metadata for the matched fundamental and score breakdown.
7. Train the action/opponent model with soft-label cross entropy when `actionProbs` is present, preserving hard-label training for existing replay.
8. Use the pretrained checkpoint as:
   - the rollout opponent model, as today;
   - a model-backed root prior/ranker for `td-search`, blended with the existing heuristic prior at first.
9. Use the pretrained checkpoint as the warm start for self-play.
10. Evaluate after meaningful training runs and held-out scenario tests, not with noisy midrun gates alone.

## Proposed Fundamentals

The first milestone should focus on the five strongest, clearest fundamentals:

- Prioritize a path to winning three districts. Prefer actions that flip, contest, or protect districts that matter to a 3-district win path; penalize over-investing in already secure districts while other districts remain unresolved.
- Complete or progress live deeds that improve district control. Strongly prefer deed completion when it wins, contests, or protects a district; give weaker credit for progress on deeds that are not currently strategically live.
- Prefer immediate endgame district swings. Near deck exhaustion or final turns, prioritize actions that improve district points or decisive tiebreakers now.
- Use in-progress deeds to shelter surplus taxable resources. Prefer placing vulnerable surplus tokens onto live deeds instead of leaving them exposed in the bank, while avoiding low-value token dumping.
- Preserve tactical flexibility for future card plays. Prefer maintaining one token in each suit when practical, avoid selling cards central to an active 3-district plan, and prefer trades or sells that directly enable immediate deed completion or strong development.

## Implementation Notes

Reuse existing replay and training formats where possible, but the current TD path needs a targeted extension.

- Add `scripts/generate_fundamentals_data.py` to create labeled TD opponent/action samples.
- Generate states via the bridge instead of constructing raw states manually.
- Use scenario-seeking policies to create enough deed, contest, resource, and endgame examples.
- Add optional `actionProbs` to TD `OpponentSample` JSONL, not only to the older `DecisionSample` format.
- Keep `actionIndex` for compatibility, metrics, and hard-label fallback.
- Validate soft labels fail-fast: finite, non-negative, length matches legal action features, positive total mass, normalized before training.
- Train with soft-label cross entropy when `actionProbs` is present:
  `loss = -sum(actionProbs * log_softmax(logits))`.
- Preserve current hard-label cross entropy for existing replay rows without `actionProbs`.
- Store sample metadata for auditability, either as optional ignored JSONL fields or a sidecar audit file keyed by sample id.
- Blend model-backed root priors with heuristic priors in `td-search` using explicit temperature and blend-weight parameters.
- Keep value-model pretraining conservative; start with action-head pretraining and root-prior integration first.

## Required Code Changes

- Add a fundamentals scenario generator.
- Add scenario matchers and action scorers for the first small set of fundamentals.
- Extend TD opponent/action replay with optional `actionProbs`.
- Add soft-label support to opponent/action training.
- Add Python `td-search` support for model-backed root priors/ranking.
- Add browser `td-search` parity for model-backed root priors before deploying a pretrained pack as default.
- Add tests for:
  - label normalization and malformed-label rejection;
  - legal-action alignment;
  - soft-label loss behavior;
  - backward compatibility with hard-label replay;
  - root-prior blending and deterministic tie-breaking;
  - scenario-specific scoring.
- Add a wrapper command for producing a pretrained `opponent.pt` checkpoint.

## First Milestone

Implement a narrow version focused on the five fundamentals above.

The milestone should produce:

- a fundamentals replay JSONL artifact;
- an audit summary by fundamental type;
- a pretrained `opponent.pt` action model;
- Python `td-search` root-prior integration using that model;
- held-out scenario evaluation results.

Success for this milestone is not measured by final bot strength. It is measured by whether:

- the pretrained action model assigns high probability mass to acceptable moves on held-out fundamental scenarios;
- top-1 model choices are usually in the acceptable action set;
- model-backed root priors improve over the current heuristic prior on those scenarios;
- a small side-swapped `td-search` eval does not show a major regression;
- subsequent self-play starts from visibly less naive behavior.
