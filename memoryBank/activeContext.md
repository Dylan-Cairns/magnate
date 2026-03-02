# Active Context

## Current Focus

- Keep TypeScript rules deterministic and canonical.
- Continue the self-play-focused TD loop using the current `12`-chunk promotion cadence.
- Use `scripts.run_td_loop` for bootstrap/recalibration passes, then advance primarily with `scripts.run_td_loop_selfplay`.
- Improve model quality against search baseline and incumbent td-search while keeping runtime practical.
- Keep `models/td_checkpoints/manifest.json` as the canonical checked-in registry for promoted warm starts and opponent-pool entries.
- Gate self-play generator updates at configurable chunk-block boundaries.
- Train self-play chunks from a small accepted replay window instead of requiring strictly chunk-local replay.
- Select the best cheap-eval saved checkpoint from each training chunk, then select among block candidates before running the generator gate.
- Use sequence-aware `td-lambda` as the default value target for TD training.

## Locked Decisions

- TS engine is canonical gameplay truth.
- Python training/eval calls TS through the bridge contract (`contracts/magnate_bridge.v1.json`).
- Training/eval scripts are fail-fast (no silent fallback labels/actions/probabilities).
- Default promotion decision uses pooled side-swapped certify eval windows (`200` games/side per window).

## Current State

- Browser app is playable; default bot is `Rollout Search`; `TD Search`, `TD Search Fast`, and `Heuristic` remain available as alternate profiles.
- Browser `Heuristic`, rollout-search priors, and TD-search heuristic root priors share the TS heuristic scorer; current scorer intent is resource sheltering without losing suit coverage, three-district control, district flips/defense, close deed completion, avoiding non-completing deed progress that spends a last suit token, projected control from newly bought deeds, progress-aware opponent deed threats, rank-2 deed rejection, high-rank deed caution late or without suit access, and trade penalties unless a trade immediately unlocks a high-value move.
- Browser rollout-search leaf evaluation now uses a dedicated TS state evaluator that weights canonical district control with Ace bonuses, final-score tiebreak pressure, progress-aware deed potential/threats, late-game urgency, and resource quality without using hand-size as a value signal.
- Default browser `Rollout Search` profile now uses a slightly larger search budget (`worlds=10`, `rollouts=4`, `depth=16`, `maxRootActions=8`, `rolloutEpsilon=0`) to make the web opponent stronger while preserving deterministic rollout play.
- Bridge runtime command surface is stable: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Trainer policy surface is intentionally narrow: `random`, `heuristic`, `search`, `td-value`, `td-search`.
- The failed 2026-04-21 cache rollout was reverted with git revert commits; td-search rollouts are back on the stateful bridge simulation path.
- Python unit tests are standardized on pytest via `.\.venv\Scripts\python -m pytest`.
- `scripts.run_td_loop` is the bootstrap or recalibration path; `scripts.run_td_loop_selfplay` is the primary forward loop.
- Checkpoint promotion source of truth is now `models/td_checkpoints/manifest.json` schema v2:
  - `defaultWarmStart` selects the warm-start pair.
  - `opponentPool` selects promoted checkpoints for pool sampling.
  - checkpoint payloads under `models/td_checkpoints/` are intended to travel with source control, while ignored `artifacts/` summaries are fallback/history.
- Loop promotion flows copy passed checkpoints into `models/td_checkpoints/<manifest-key>/` and update the manifest unless `--disable-manifest-promotion` is set.
- `scripts.run_td_loop_selfplay` now distinguishes trained candidate checkpoints from accepted generator checkpoints:
  - every chunk writes `chunk.summary.json`;
  - `trainedLatestCheckpoint` is the final checkpoint emitted by training;
  - `checkpointSelection` records the cheap eval used to choose among saved training checkpoints;
  - `candidateCheckpoint` is the selected chunk learner checkpoint;
  - `learnerCheckpointBefore` / `learnerCheckpointAfter` records the checkpoint used to warm-start subsequent training;
  - `generatorCheckpointBefore` / `generatorCheckpointAfter` records the checkpoint allowed to generate collection data and remain promotion-eligible;
  - `latestCheckpoint` remains a compatibility alias for the post-gate generator checkpoint in loop artifacts;
  - `--generator-update-chunks` controls the generator cadence; non-boundary chunks defer the generator gate, while each full block and final partial block selects the best block candidate and runs the resumable td-search vs td-search sequential gate against the current accepted generator;
  - block summaries live under `blocks/block-XXX/block.summary.json`, and final promotion always evaluates the accepted generator checkpoint, not an ungated learner checkpoint.
- Self-play resume is strict for the current artifact schema; completed chunks missing `chunk.summary.json`, `checkpointSelection`, `replayWindow`, or `replayForTraining` fail instead of being inferred from legacy artifacts. Resume now preserves separate learner/generator checkpoints, recent replay history, and open block candidates for block-gated runs.
- Self-play training writes `train/replay_window/window.summary.json` per chunk:
  - default window size is `3`, enabling a small replay window without wrapper overrides;
  - `--train-replay-window-source accepted` trains on the current chunk plus the last `N-1` accepted chunks;
  - `--train-replay-window-source recent` trains on the current chunk plus the last `N-1` trained chunks, even if generator gates failed;
  - replay-window summaries now reference the ordered chunk replay files directly instead of materializing duplicated `window.value.jsonl` / `window.opponent.jsonl` copies;
  - value target mode defaults to `td-lambda` with `--train-td-lambda 0.7`;
  - value replay line caps are rejected under `td-lambda` because raw line caps can split full episode trajectories;
  - rejected chunks are not added to future accepted replay history, but can participate in the recent learner replay history.
- Typed bridge payloads cover the main `trainer/` package, while `scripts/` still contains some dynamic orchestration surfaces outside checked-in pyright scope.

## Remaining

- Keep calibrating loop cadence/thresholds from repeated overnight results.
- Improve `td-search` strength and throughput.
- Tune replay-window size/caps from repeated runs; a broader online reservoir remains a future option.
- Tune browser rollout-search strength/latency now that `Rollout Search` is the default, while keeping TD-search profiles available for comparison.
- Continue shrinking untyped/dynamic payload handling in the remaining Python script/orchestration layer outside the typed `trainer/` package.

## Immediate Next Steps

1. Before the next long self-play run, confirm `models/td_checkpoints/manifest.json` and the referenced checkpoint files are present and committed on the machine that will run training.
2. Continue self-play loop iterations with promoted manifest warm starts, td-lambda value targets, checkpoint selection, explicit learner/generator checkpoint tracking, `recent` replay windows, `3`-chunk generator blocks on the laptop wrapper, and the current final promotion cadence.
3. Track checkpoint-selection winners, block-selection winners, sequential generator-gate accept/reject or inconclusive outcomes, final dual-gate outcomes, and side-gap stability.
4. Extend the typed rollout from `trainer/` into the remaining `scripts/` orchestration and export helpers as those surfaces are touched.
5. Keep the Windows laptop wrappers and Linux cloud flows aligned with the runbook in `memoryBank/techContext.md`.

_Updated: 2026-05-29._
