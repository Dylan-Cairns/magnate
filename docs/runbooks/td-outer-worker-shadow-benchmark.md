# TD outer-worker shadow benchmark

This is the production-stack validation for the two-lane TD rollout executor.
It runs the normal worker-backed TD Medium policy through the outer bot worker
and its nested search-worker pool.

The legacy executor is authoritative. The paired executor receives the same
state, legal actions, checkpoint, search configuration, and seed, but its
selected action is discarded. Catalog-created browser bots omit the
experimental executor field and continue to use legacy execution.

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
- `execution.parallelWorkers`, `parallelBatchSizes`, and `rootVisitBudgets`
  contain the expected production values;
- `model.weightsSha256` matches the deployed checkpoint;
- `gate.exactParity` is true.

The performance gate requires at least 1.2x total speedup and no p95 latency
regression. These gates only determine the final recommendation; they never
terminate the run early. A fully passing run recommends preparing a separate
default-enable change with a legacy rollback switch.

## Daytime smoke

The short wiring check is:

```powershell
npm run benchmark:td-outer-shadow:smoke
```

Smoke mode uses one position, one repetition, one warmup decision, and no
shadow game. Its timing is not performance evidence. The validated 2026-07-23
smoke passed exact action and diagnostics parity, observed eight workers, batch
size 16 and root budget 160, and used checkpoint SHA-256
`7c6605c6b4f366729082f6e57878bfe3349b6e57010a1e39ec730bb3ff76d819`.
