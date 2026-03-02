Phase 1: Canonical TS Heuristic

Add src/policies/heuristicScorer.ts.

Inputs: GameState, PlayerView, legal GameAction[].
Outputs:
scored actions
ranked actions
priors over action keys
selected action
Tie-break by actionStableKey.
Keep the actual heuristic rules modular, so future rule tuning does not touch policy plumbing.
Add src/policies/heuristicPolicy.ts.

Implements ActionPolicy.
Calls the canonical scorer.
Returns the selected legal action.
Add app bot profile in src/policies/catalog.ts.

Add BotProfileId = 'heuristic'.
Add kind: 'heuristic'.
Add label Heuristic.
Keep TD Search as default initially.
Remove duplicated browser action scoring.

Update src/policies/searchPolicy.ts.
Update src/policies/tdSearchPolicy.ts.
Both should import root ranking/priors from heuristicScorer.ts.
Phase 2: Bridge Support For Python

Add a bridge command, probably heuristicScores.

Update src/bridge/protocol.ts.
Update src/bridge/runtime.ts.
Update contracts/magnate_bridge.v1.json.
Update memoryBank/bridgeInterfaceContract.md.
Shape:

{
  "activePlayerId": "PlayerA",
  "phase": "ActionWindow",
  "selectedActionKey": "...",
  "actions": [
    {
      "actionKey": "...",
      "actionId": "develop-outright",
      "score": 1.23,
      "prior": 0.42,
      "rank": 0
    }
  ]
}
Add TS bridge tests.

Contract metadata includes the new command.
Command returns only current legal actions.
Priors are finite, non-negative, sum to 1.
Selected key is legal and deterministic.
Phase 3: Python Uses TS Heuristic

Extend Python bridge client.

Add BridgeClient.heuristic_scores().
Add MagnateBridgeEnv.heuristic_scores().
Add BridgeForwardModel.heuristic_scores() for search rollouts.
Replace Python HeuristicPolicy internals.

Keep public name heuristic.
It should delegate to bridge scoring.
root_action_probs() returns the last bridge priors.
For normal eval/collection, use the current env bridge, not a second subprocess.
Add a small dispatch helper.

Example: trainer/policy_action.py.
choose_policy_action(policy, env, view, legal_actions, rng, state).
If policy supports bridge-aware selection, call that.
Otherwise call the existing choose_action_key(...).
Update call sites to use the helper.

trainer/evaluate.py
trainer/td/self_play.py
trainer/teacher_data.py
trainer/training.py
Phase 4: Python Search Uses TS Heuristic Too

Update trainer/search_policy.py.
Remove dependency on Python HeuristicPolicy.score_action.
Root ranking and priors come from BridgeForwardModel.heuristic_scores().
Rollout root-player heuristic choices also come from bridge scorer.
Update trainer/search/root_selector.py.
Either remove heuristic policy dependency or adapt it to consume explicit score rows.
Keep UCB/progressive widening logic unchanged.
Phase 5: Replay And Pretraining

Extend TD opponent replay with optional soft labels.
trainer/td/types.py: add action_probs.
trainer/td/io.py: read/write actionProbs.
Validate length, finiteness, non-negative values, positive total, normalized distribution.
Update trainer/td/self_play.py.
After policy selection, if root_action_probs() exists, write actionProbs.
Heuristic replay then carries soft action targets instead of only one hard selected move.
Update opponent/action training.
trainer/td/train.py
If action_probs exists, use soft-label cross entropy.
Otherwise keep current hard-label cross entropy.
Allow heuristic as a teacher policy where useful.
Update scripts/generate_teacher_data.py.
Add heuristic to root-prob-capable policy names after root_action_probs() works.
Phase 6: TD Search Root Prior Blending

Add optional model-prior blending in Python TD search.
TDSearchPolicyConfig fields:
root_prior_source
root_prior_model_weight
root_prior_temperature
Default stays current heuristic behavior.
Add matching browser TD-search support.
src/policies/tdSearchPolicy.ts
Blend canonical heuristic priors with opponent-model priors from opponentScorer.
This is the point where heuristic pretraining can influence the bot’s own root choices, not only opponent rollout choices.

Phase 7: Self-Play Loop Integration

Add heuristic anchor collection to scripts/run_td_loop_selfplay.py.
Add --collect-heuristic-anchor-share.
Add profile: candidate td-search vs heuristic.
Keep existing search and incumbent promotion gates unchanged.
Use heuristic replay in bootstrap experiments.
heuristic vs heuristic
heuristic vs search
search vs heuristic
current td-search vs heuristic
Phase 8: Verification And Docs

Tests to add/update:
TS heuristic scorer tests.
Browser catalog tests.
Bridge heuristicScores tests.
Python bridge client/env tests.
Python heuristic policy tests.
Soft-label replay IO tests.
Soft-label training tests.
TD-search root-prior blend tests.
Commands before handoff:
yarn test
yarn lint
.\.venv\Scripts\python -m pytest trainer_tests/...
python -m ruff check scripts trainer trainer_tests
.\.venv\Scripts\python -m pyright -p .
Docs:
Update memoryBank/systemPatterns.md.
Update memoryBank/techContext.md.
Update memoryBank/activeContext.md.
Update README.md only if user-facing bot/profile or command docs change.