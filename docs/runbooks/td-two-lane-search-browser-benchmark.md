# TD two-lane browser search benchmark

This was the end-to-end follow-up to the isolated two-lane opponent-network
kernel benchmark. It compares explicit legacy execution with the lockstep
executor that pairs the two rollout visits already assigned to each worker.

The paired executor subsequently passed outer-worker shadow validation and is
now the eligible browser default; see
`docs/runbooks/td-outer-worker-shadow-benchmark.md` for the current behavior and
legacy rollback. This benchmark still makes both lane selections explicitly
and does not change the checkpoint, TD Medium search configuration, visit
scheduling, UCB allocation, seeds, merge order, root policy, rollout policy,
leaf policy, or selected-action rule.

## What the full run does

The default full run uses:

- the deployed `td-root-search-v2-medium` configuration (`worlds=10`,
  `depth=40`, `maxRootActions=16`, `rolloutEpsilon=0`);
- eight browser search workers and the production-sized two-task chunk per
  worker;
- 24 deterministic multi-action states, three repetitions per state, with
  legacy and paired order alternated;
- an untimed resumable-scalar lane to audit the state-machine extraction;
- one complete deterministic game per executor to compare action transcripts,
  root diagnostics, and final state;
- the active model-pack ID, weights URL, and SHA-256 in the result.

Every scheduled comparison runs even if an earlier comparison finds a mismatch
or the observed speedup is below the recommendation threshold. Full mode has no
timeout. Stop it manually with `Ctrl+C` if necessary.

## Quiet-machine command

Close CPU-heavy applications, pause downloads/sync, and keep the laptop on AC
power. Then, from the repository root, run:

```powershell
npm run benchmark:td-two-lane-search
```

The browser is headless. The command prints the result path and summary when it
finishes. Artifacts are written under:

```text
artifacts/benchmarks/td-two-lane-search-browser-<timestamp>/
```

The important file is `result.json`. Preserve the whole directory so the Vite
and browser logs remain available if the run fails.

The smoke took about 90 seconds on a 12-logical-core laptop, including model
loading and three diagnostic lanes. The full run includes many more decisions
and a full-game transcript audit; budget at least an hour and let it continue
overnight if needed. Laptop use during the run will not affect parity, but it
can materially distort latency and speedup.

## Reading the result

The competitiveness requirement passes only when all of these are zero:

- `correctness.selectedActionMismatchCount`
- `correctness.diagnosticsMismatchCount`
- `correctness.scalarMachineActionMismatchCount`
- `correctness.scalarMachineDiagnosticsMismatchCount`
- `correctness.transcripts.mismatchGames`

`correctness.mismatchSamples` contains both root visit/value tables when a
decision differs. `gate.exactParity` summarizes the required parity checks.

`timing.speedup` is total legacy decision time divided by total paired decision
time. The recommendation threshold is 1.2x, but it is applied only after the
complete run and never stops work early. The completed passing result was the
evidence used to proceed to production-stack shadow validation.

## Smoke command

The short wiring check is safe to run during ordinary laptop use:

```powershell
npm run benchmark:td-two-lane-search:smoke
```

Smoke mode uses one state, one repetition, two workers, and no full transcript
game. Its timing is not performance evidence. The validated 2026-07-22 smoke
used checkpoint SHA-256
`7c6605c6b4f366729082f6e57878bfe3349b6e57010a1e39ec730bb3ff76d819`,
found zero action/diagnostic/scalar-machine mismatches, and reported a
single-sample 1.321x speedup.
