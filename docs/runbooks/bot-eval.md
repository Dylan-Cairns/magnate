# TypeScript Browser-Bot Evaluation Runbook

`src/botEval/` runs browser `ActionPolicy` implementations directly against the canonical TypeScript engine.

## Commands

- Head-to-head eval:
  `yarn bot:eval head-to-head --config configs/bot-eval/head-to-head.example.json`
- Rollout-search sweep:
  `yarn bot:eval rollout-search-sweep --config configs/bot-eval/rollout-search-width-sweep.example.json`
- Rollout-search TD replay export:
  `yarn bot:eval collect-td-replay --config configs/bot-eval/collect-td-replay.rollout-search.example.json`
- Sharded TD replay export:
  `yarn bot:eval collect-td-replay-sharded --config configs/bot-eval/collect-td-replay.v2-hard.json --workers 8 --shard-games 1`
- Strategic-position characterization:
  `yarn bot:eval strategic-positions --repetitions 8`
- Replay one recorded game:
  `yarn bot:eval replay --artifact artifacts/ts-bot-evals/<run>/matchup.json --game-id pair-0001-candidate-as-a`
- Override heartbeat cadence for head-to-head, sweep, replay, and replay-export
  commands: append `--progress-interval-seconds 10` (`0` disables timed
  heartbeats). Strategic-position characterization reports after each decision
  and does not use this flag.

## Head-To-Head Artifacts

Head-to-head evals use paired seeds, swapped policy seats, alternating first-player seats, Wilson confidence intervals, latency summaries, JSON artifacts, Markdown summaries, and exact stable-action-key replay checks.

Typical outputs:

- `artifacts/ts-bot-evals/<run>/matchup.json`
- `artifacts/ts-bot-evals/<run>/summary.md`

Config inputs can reference catalog presets with `{ "profileId": "rollout-search-v2-medium" }` or define serializable bot specs directly.

## Strategic Position Characterization

`yarn bot:eval strategic-positions [--repetitions <count>] [--start-repetition <nonnegative-integer>] [--positions <comma-separated-ids>] [--variants <comma-separated-ids>] [--out-dir <path>]`
by default runs the typed strategic position catalog against direct heuristic
v2, V2 Hard, and the current TD V2 Medium profile. All variants receive the same
explicit random seed for each position/repetition. The command records choices
and root diagnostics; it does not treat current-bot agreement with the catalog's
reviewed pairwise preference as a test pass condition. A choice outside the
declared pair is reported as unassessed rather than as a mismatch. The JSON also
records the information-safe state summary and a canonical fingerprint of each
full catalog-case payload, including descriptive metadata. Compare the recorded
state, legal actions, and action consequences when checking executable equality
across artifacts; metadata-only wording changes also change the fingerprint.

One repetition is useful as a smoke check; eight is the initial fixed-position
stability screen. Repetitions reveal seed sensitivity in stochastic policies,
not independent strategic evidence. Direct heuristic v2 is deterministic in
these cases, so repeating it only confirms repeatability.

The default selection includes all three variants. `--variants` can select any
unique comma-separated subset of the three defaults and these opt-in
diagnostics:

- `td-root-search-v2-800-visits`;
- `td-root-search-v2-800-visits-heuristic-root`;
- `td-root-search-v2-800-visits-heuristic-rollout`;
- `td-root-search-v2-800-visits-heuristic-root-rollout`.

A valid default model pack under `public/model-packs/` is required when any
selected hook remains TD-guided. The base 800-visit variant clones current TD
V2 Medium and changes only sampled worlds from 10 to 50 while retaining depth
40 and the same default model-pack selection. The three ablations keep that
budget, use heuristic v2 for each named heuristic hook, and retain TD leaf
guidance. They match V2 Hard's root-visit count, not its deeper search or total
computation, and none joins the default variant set.
`--positions` likewise accepts unique catalog position IDs, and unknown or
duplicate IDs fail fast. `--start-repetition` changes the
deterministic repetition/seed index for a targeted extension. It does not resume
or merge an earlier run, so use a separate output directory and combine results
externally if needed. For example:

```powershell
yarn bot:eval strategic-positions `
  --positions known-hand-optionality-original,known-hand-optionality-mirror `
  --variants td-root-search-v2-800-visits,td-root-search-v2-800-visits-heuristic-root,td-root-search-v2-800-visits-heuristic-rollout,td-root-search-v2-800-visits-heuristic-root-rollout `
  --start-repetition 7 `
  --repetitions 1 `
  --out-dir artifacts/ts-bot-evals/optionality-guidance-seed-7
```

The Markdown summary begins with per-position/variant selection histograms,
modal counts, and preferred/alternative/unassessed results. It then reports
within-position focus-score/value gaps with visits and expansion coverage,
matched counterfactual selection transitions, raw decisions, and focus-action
signals. Search means come from adaptive, potentially unequal visits; treat
their gaps as diagnostics within one position and variant, not fixed-budget
paired estimates. Policies run in-process in Node without browser or Web Worker
wrappers, so reported latency is diagnostic for that execution mode.

Typical outputs:

- `artifacts/ts-bot-evals/<run>/positions.json`
- `artifacts/ts-bot-evals/<run>/summary.md`

See [the summary-v0/catalog-v1 design](../design/strategic-state-summary-v0.md)
for the factual summary contract, catalog scope, and interpretation rules.

## TD Replay Export

`yarn bot:eval collect-td-replay --config <path> [--out-dir <path>]` runs full self-play games through Node-compatible policies and writes value, opponent, and summary artifacts under `artifacts/td_replay` by default.

`yarn bot:eval collect-td-replay-sharded --config <path> --workers <count> [--shard-games <count>] [--out-dir <path>]` splits the same config into contiguous game-index ranges and runs those ranges in child Node processes. By default it creates one shard per worker, capped by game count. `--shard-games` instead creates queued shard jobs of at most that many games, which is useful when individual games have uneven runtime and you want a fixed worker pool to keep picking up small jobs. Each shard writes independent `shard-NNN.value.jsonl`, `shard-NNN.opponent.jsonl`, and `shard-NNN.summary.json` files under `artifacts/td_replay/<run>/shards/`; the parent writes `artifacts/td_replay/<run>/summary.json`.

Replay rows use TypeScript `trainingEncoding`, include `episodeId` plus contiguous per-player `timestep` for `td-lambda`, order opponent action candidates by canonical stable action key, and are readable by `scripts.train_td`.

When passing multiple exported value replay files into one td-lambda train run, keep each export's `seedPrefix` globally unique because the sequence key is `(episodeId, playerId, timestep)`. Shards from one sharded export already use one global seed sequence and can be passed together directly.

Example sharded training input:

```powershell
.\.venv\Scripts\python -m scripts.train_td `
  --value-replay artifacts/td_replay/<run>/shards/*.value.jsonl `
  --opponent-replay artifacts/td_replay/<run>/shards/*.opponent.jsonl `
  --steps 2000 `
  --run-label td-v2-teacher
```

## Sweeps And Workers

`yarn bot:eval rollout-search-sweep --config <path> [--workers <count>]` runs explicit rollout `search` candidates sequentially against one fixed opponent with one shared paired-seed prefix.

Worker counts above `1` record latency as loaded latency, so those timings are throughput diagnostics rather than isolated browser latency measurements. Use `--workers 1` for browser-relevant latency.

Sweep aggregate artifacts are written with `status=running` before compute starts and atomically refreshed after each completed candidate. Automatic sweep resume is not implemented; follow-up runs must be constructed explicitly.

## Model-Pack Runtime

Browser and Web Worker TD-root search load `td-root-search-v1` packs from
`public/model-packs`. Node bot evaluation installs a local `public/` fetch shim,
so serialized TD-root specs use those same static model-pack assets.
