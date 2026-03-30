# TypeScript Browser-Bot Evaluation Runbook

`src/botEval/` runs browser `ActionPolicy` implementations directly against the canonical TypeScript engine.

## Commands

- Head-to-head eval:
  `yarn bot:eval head-to-head --config configs/bot-eval/head-to-head.example.json`
- Rollout-search sweep:
  `yarn bot:eval rollout-search-sweep --config configs/bot-eval/rollout-search-width-sweep.example.json`
- Rollout-search TD replay export:
  `yarn bot:eval collect-td-replay --config configs/bot-eval/collect-td-replay.rollout-search.example.json`
- Replay one recorded game:
  `yarn bot:eval replay --artifact artifacts/ts-bot-evals/<run>/matchup.json --game-id pair-0001-candidate-as-a`
- Override heartbeat cadence:
  append `--progress-interval-seconds 10` (`0` disables timed heartbeats).

## Head-To-Head Artifacts

Head-to-head evals use paired seeds, swapped policy seats, alternating first-player seats, Wilson confidence intervals, latency summaries, JSON artifacts, Markdown summaries, and exact stable-action-key replay checks.

Typical outputs:

- `artifacts/ts-bot-evals/<run>/matchup.json`
- `artifacts/ts-bot-evals/<run>/summary.md`

Config inputs can reference catalog presets with `{ "profileId": "heuristic" }` or define serializable bot specs directly.

## TD Replay Export

`yarn bot:eval collect-td-replay --config <path> [--out-dir <path>]` runs full self-play games through Node-compatible policies and writes value, opponent, and summary artifacts under `artifacts/td_replay` by default.

Replay rows use TypeScript `trainingEncoding`, include `episodeId` plus contiguous per-player `timestep` for `td-lambda`, order opponent action candidates by canonical stable action key, and are readable by `scripts.train_td`.

When passing multiple exported value replay files into one td-lambda train run, keep each export's `seedPrefix` globally unique because the sequence key is `(episodeId, playerId, timestep)`.

## Sweeps And Workers

`yarn bot:eval rollout-search-sweep --config <path> [--workers <count>]` runs explicit rollout `search` candidates sequentially against one fixed opponent with one shared paired-seed prefix.

Worker counts above `1` record latency as loaded latency, so those timings are throughput diagnostics rather than isolated browser latency measurements. Use `--workers 1` for browser-relevant latency.

Sweep aggregate artifacts are written with `status=running` before compute starts and atomically refreshed after each completed candidate. Automatic sweep resume is not implemented; follow-up runs must be constructed explicitly.

## Current Limitation

Browser `td-search` specs are serializable through the shared policy factory, but direct Node eval of those specs still needs a local model-pack loader.
