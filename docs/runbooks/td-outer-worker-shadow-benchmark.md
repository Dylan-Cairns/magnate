# TD outer-worker shadow benchmark

This is the production-stack validation for the paired TD rollout executor. It
runs the normal worker-backed TD Medium policy through the outer bot worker and
its nested search-worker pool.

The legacy executor remains authoritative inside this benchmark. The candidate
policy omits an executor override, just like normal browser play, and therefore
exercises the production default. Each result records the effective modes
reported by the outer workers, so the harness fails its activation gate unless
the authoritative lane reports `legacy` and the default lane reports
`resumable-paired-td`.

## Browser default and rollback

Parallel `td-root-search` policies with TD rollout guidance now use paired
execution by default. Synchronous search, ordinary rollout search, and
TD-root policies with heuristic rollout guidance retain their prior execution
paths.

To roll an affected browser session back to the legacy executor, add:

```text
?tdSearchExecutor=legacy
```

Use `&tdSearchExecutor=legacy` if the URL already has query parameters. Remove
the parameter to restore the paired default. `tdSearchExecutor=paired` is also
accepted as an explicit diagnostic selection. Any other value is a hard error
rather than a silent fallback.

The executor selection does not change the checkpoint, root-search budget,
rollout waves, UCB allocation, seeds, ordered result merge, diagnostics, or
selected-action policy. Effective executor mode is returned as worker response
metadata and is not inserted into search diagnostics or teacher targets.

## Full quiet-machine workload

The default full run uses:

- the catalog's `td-root-search-v2-medium` specification;
- the production worker-count and batch-size calculations;
- 64 deterministic positions sampled across eight complete heuristic source
  games, with two alternating timing repetitions;
- four complete shadow games using separate seeds;
- exact selected-action and full root-diagnostics comparisons;
- the active checkpoint SHA-256, corpus fingerprint, seeds, root budget,
  observed worker counts, and batch sizes.

Every comparison runs even after a mismatch or a failed performance threshold.
There is no full-mode timeout.

## Quiet-machine command

Close CPU-heavy applications, pause sync/download activity, and keep the laptop
on AC power. From the repository root run:

```powershell
npm run benchmark:td-outer-shadow
```

The run is expected to take roughly one to two hours, but it is safe to leave
overnight. Stop it manually with `Ctrl+C` if necessary.

Results are saved automatically under:

```text
artifacts/benchmarks/td-outer-worker-shadow-browser-<timestamp>/
```

Console output does not need to be preserved. Keep the complete artifact
directory, especially `result.json`.

## Required result

Competitiveness parity requires all four counters to be zero:

- `correctness.selectedActionMismatchCount`
- `correctness.diagnosticsMismatchCount`
- `correctness.shadow.actionMismatchCount`
- `correctness.shadow.diagnosticsMismatchCount`

Also verify:

- `policy.authority` is `legacy`;
- `policy.candidateExecutionModeRequest` is
  `omitted-production-default`;
- `execution.modeRouting.observedLegacy` contains only `legacy`;
- `execution.modeRouting.observedCandidate` contains only
  `resumable-paired-td`;
- `gate.activationPassed` is true;
- `execution.parallelWorkers`, `parallelBatchSizes`, and `rootVisitBudgets`
  contain the expected production values;
- `model.weightsSha256` matches the deployed checkpoint;
- `gate.exactParity` is true.

The performance gate requires at least 1.2x total speedup and no p95 latency
regression. These gates only determine the final recommendation; they never
terminate the run early.

The completed quiet-machine run from 2026-07-23 is stored at
`artifacts/benchmarks/td-outer-worker-shadow-browser-20260723T225606Z/`.
It compared 128 corpus decisions and 599 searched decisions across four
complete shadow games with zero selected-action, diagnostics/teacher-target, or
transcript mismatches. It observed eight workers, batch size 16, root budget
160, no p95 regression, and a 1.289x total speedup on the deployed checkpoint.
That is the performance and competitiveness evidence for enabling the paired
default; a new overnight run is not required for the activation change.

## Daytime smoke

The short wiring check is:

```powershell
npm run benchmark:td-outer-shadow:smoke
```

Smoke mode uses one position, one repetition, one warmup decision, and no
shadow game. Its timing is not performance evidence. It is the activation
check: verify the console reports `Legacy executor observed: legacy` and
`Default executor observed: resumable-paired-td`, and that the saved result has
`gate.activationPassed: true`, exact action/diagnostics parity, eight workers,
batch size 16, and root budget 160.

The 2026-07-24 activation smoke is stored at
`artifacts/benchmarks/td-outer-worker-shadow-browser-20260724T084704Z/`.
It passed exact action and diagnostics parity, reported only `legacy` for the
override lane and only `resumable-paired-td` for the omitted/default lane, and
observed eight workers, batch size 16, root budget 160, and checkpoint SHA-256
`7c6605c6b4f366729082f6e57878bfe3349b6e57010a1e39ec730bb3ff76d819`.
Its one-position 1.340x timing is a smoke observation, not new performance
evidence.
