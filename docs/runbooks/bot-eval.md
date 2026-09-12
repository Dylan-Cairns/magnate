# TypeScript Browser-Bot Evaluation Runbook

`src/botEval/` runs browser `ActionPolicy` implementations directly against the
canonical TypeScript engine.

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
- Matched forced-root rollout tracing:
  `yarn bot:eval strategic-forced-rollouts --repetitions 0 --positions known-hand-optionality-holdout-original,known-hand-optionality-holdout-mirror`
- Replay one recorded game:
  `yarn bot:eval replay --artifact artifacts/ts-bot-evals/<run>/matchup.json --game-id pair-0001-candidate-as-a`
- Heartbeat override for head-to-head, sweep, replay, and replay-export
  commands: append `--progress-interval-seconds 10` (`0` disables timed
  heartbeats). Strategic-position characterization reports per decision and does
  not use this flag.

## Head-To-Head Artifacts

Head-to-head evals use paired seeds, swapped policy seats, alternating
first-player seats, Wilson confidence intervals, latency summaries, JSON
artifacts, Markdown summaries, and exact stable-action-key replay checks.

Typical outputs:

- `artifacts/ts-bot-evals/<run>/matchup.json`
- `artifacts/ts-bot-evals/<run>/summary.md`

Config inputs can reference catalog presets with
`{ "profileId": "rollout-search-v2-medium" }` or define serializable bot specs
directly.

## Strategic Position Characterization

`yarn bot:eval strategic-positions [--repetitions <count>] [--start-repetition <nonnegative-integer>] [--positions <ids>] [--variants <ids>] [--model-index-path <path> --pack-id <id>] [--out-dir <path>]`
runs the typed catalog against direct heuristic v2, V2 Hard, and the current TD
V2 Medium profile by default. It records choices, root diagnostics, the
information-safe state summary, and a canonical fingerprint of each full
catalog-case payload. Actions outside a preference's declared pair are reported
as unassessed, not as mismatches; the reviewed preferences are not test
assertions.

One repetition is a smoke check, eight is the initial stability screen.
Repetitions reveal seed sensitivity, not independent strategic evidence. The
default selection includes all three variants; `--variants` also accepts the
opt-in `td-root-search-v2-800-visits` diagnostic, which clones current TD V2
Medium and changes only sampled worlds from 10 to 50. Any TD-guided variant
requires a valid model pack under `public/model-packs/`. For a frozen
experimental pack, pass both `--model-index-path` and `--pack-id`; those apply
only to TD variants and are also accepted by `strategic-forced-rollouts`.
`--start-repetition` changes the deterministic seed index for targeted
extensions; it does not resume or merge a prior run, so use a separate output
directory.

Typical outputs: `positions.json` and `summary.md` under
`artifacts/ts-bot-evals/<run>/`. See
[the summary/catalog design](../design/strategic-state-summary-v0.md) for the
factual contract and interpretation rules.

## Strategic Forced-Rollout Tracing

`yarn bot:eval strategic-forced-rollouts [--positions <ids>] [--repetitions <id-list>] [--scenarios <id-list>] [--out-dir <path>]`
is the continuation-level diagnostic for the mirrored optionality cases. For
every requested position, repetition, and action-local scenario index, it
samples one hidden world and forces both `preserve-option` and
`overwrite-option` through that same world, engine seed, and rollout seed, then
plays each forced root to terminal once under TD rollout guidance and once under
heuristic v2. It bypasses root search and UCB allocation entirely and records
both guides' proposals at every encountered state without consuming the live
RNG. It fails if a trace reaches the depth limit.

`--repetitions` and `--scenarios` are explicit ID lists. `--scenarios` defaults
to `0`-`49`, one complete cycle of the 50 sampled hidden worlds. Omitting
`--positions` selects all positions carrying optionality-trace metadata.

Typical outputs: `traces.json` and `summary.md` under
`artifacts/ts-bot-evals/<run>/`. These are controlled fixture diagnostics, not
full-game strength estimates.

## TD District-Symmetry Audit

```powershell
yarn bot:eval td-symmetry `
  --replay-list artifacts/training_inputs/<experiment>/validation.opponent.paths.txt `
  --sample-size 10000 `
  --sampling-seed <frozen-seed> `
  --model-index-path model-packs-experiments/<experiment>/index.json `
  --pack-id <frozen-pack-id> `
  --out-dir artifacts/ts-bot-evals/<run>
```

Provide exactly one of `--replay-dir` or `--replay-list`. A path list is
preferred for a frozen holdout because it prevents training shards in the same
source directory from entering the audit. Sampling is deterministic over the
sorted explicit file set. The audit applies all 24 D1/D2/D4/D5 permutations with
D3 fixed and reports action-probability and value drift; it does not measure
playing strength.

## TD Replay Export

`yarn bot:eval collect-td-replay --config <path> [--out-dir <path>]` runs full
self-play games through Node-compatible policies and writes value, opponent, and
summary artifacts under `artifacts/td_replay` by default.

`yarn bot:eval collect-td-replay-sharded --config <path> --workers <count> [--shard-games <count>] [--out-dir <path>]`
splits the same config into contiguous game-index ranges run in child Node
processes. By default it creates one shard per worker, capped by game count.
`--shard-games` instead creates queued shard jobs of at most that many games for
more balanced runtime. Each shard writes `shard-NNN.value.jsonl`,
`shard-NNN.opponent.jsonl`, and `shard-NNN.summary.json` under
`artifacts/td_replay/<run>/shards/`; the parent writes `summary.json`.

Replay rows use TypeScript `trainingEncoding`, include `episodeId` plus
contiguous per-player `timestep` for `td-lambda`, order opponent action
candidates by canonical stable action key, and are readable by
`scripts.train_td`. When passing multiple exported value replay files into one
td-lambda run, keep each export's `seedPrefix` globally unique because the
sequence key is `(episodeId, playerId, timestep)`. Shards from one sharded
export share one global seed sequence and can be passed together directly.

## Sweeps And Workers

`yarn bot:eval rollout-search-sweep --config <path> [--workers <count>]` runs
explicit rollout `search` candidates sequentially against one fixed opponent
with one shared paired-seed prefix.

Worker counts above `1` record latency as loaded latency, so those timings are
throughput diagnostics rather than isolated browser latency measurements. Use
`--workers 1` for browser-relevant latency. Sweep aggregates are written with
`status=running` before compute starts and refreshed after each completed
candidate; automatic sweep resume is not implemented.

## Model-Pack Runtime

Browser and Web Worker TD-root search load `td-root-search-v1` packs from
`public/model-packs`. Node bot evaluation installs a local `public/` fetch shim,
so serialized TD-root specs use those same static model-pack assets.
