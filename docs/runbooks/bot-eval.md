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
  `yarn bot:eval strategic-positions --repetitions 1`
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

`yarn bot:eval strategic-positions [--repetitions <count>] [--out-dir <path>]`
runs the typed strategic position catalog against direct heuristic v2, V2 Hard,
and the current TD V2 Medium profile. All variants receive the same explicit
random seed for each position/repetition. The command records choices and root
diagnostics; it does not treat current-bot agreement with the catalog's reviewed
pairwise preference as a test pass condition. A choice outside the declared
pair is reported as unassessed rather than as a mismatch. The JSON also records
the information-safe state summary and a canonical fingerprint for each exact
catalog case.

This command always includes TD V2 Medium. It requires a valid default model
pack under `public/model-packs/`, and its TD results depend on the pack currently
designated as default. Policies run in-process in Node without browser or Web
Worker wrappers; reported latency is diagnostic for that execution mode.

Typical outputs:

- `artifacts/ts-bot-evals/<run>/positions.json`
- `artifacts/ts-bot-evals/<run>/summary.md`

See [the v0 design](../design/strategic-state-summary-v0.md) for the factual
summary contract, catalog scope, and interpretation rules.

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
