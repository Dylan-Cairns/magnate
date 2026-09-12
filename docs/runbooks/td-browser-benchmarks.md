# TD Browser Benchmarks

These browser benchmarks validate TD inference and search-execution plumbing.
None of them change the checkpoint, search configuration, visit scheduling, UCB
allocation, seeds, merge order, root policy, or selected-action rule.

Parity checks are valid during ordinary use. Timing evidence requires a quiet
machine: AC power, no builds, training, calls, or other sustained CPU work.
Artifacts land under `artifacts/benchmarks/` with a timestamped directory.

## Two-Lane Kernel Benchmark

Compares the scalar opponent scorer with the fused two-lane JavaScript candidate
inside a dedicated browser worker, on a deterministic corpus built from legal
game decisions.

```powershell
npm run benchmark:td-two-lane:smoke
npm run benchmark:td-two-lane
```

Continue to lockstep integration only when `correctness.exactMismatchCount` and
`correctness.argmaxMismatchCount` are `0`, checksums match, and
`timing.speedup` is at least `1.3`. Smoke timing is not performance evidence.
The benchmark always completes every configured round before computing the
recommendation.

## Search-Lane Benchmark

Compares explicit legacy execution with the lockstep executor that pairs the two
rollout visits already assigned to each worker.

```powershell
npm run benchmark:td-two-lane-search:smoke
npm run benchmark:td-two-lane-search
```

Pass requires zero mismatches in
`correctness.selectedActionMismatchCount`,
`correctness.diagnosticsMismatchCount`,
`correctness.scalarMachineActionMismatchCount`,
`correctness.scalarMachineDiagnosticsMismatchCount`, and
`correctness.transcripts.mismatchGames`. The recommendation threshold is
`timing.speedup >= 1.2`. The completed full run passed and is the evidence for
the paired default.

## Outer-Worker Shadow Benchmark

Production-stack activation check: runs the normal worker-backed TD Medium
policy through the outer bot worker and its nested search-worker pool, with the
legacy executor authoritative and the paired production default as candidate.

```powershell
npm run benchmark:td-outer-shadow:smoke
npm run benchmark:td-outer-shadow
```

Required result:

- zero mismatches for `correctness.selectedActionMismatchCount`,
  `correctness.diagnosticsMismatchCount`, `correctness.shadow.actionMismatchCount`,
  and `correctness.shadow.diagnosticsMismatchCount`;
- `policy.authority` is `legacy`, candidate request is
  `omitted-production-default`;
- observed modes are `legacy` for the authority lane and `resumable-paired-td`
  for the default lane only;
- `gate.activationPassed` and `gate.exactParity` are true;
- production workers, batch sizes, root budget, and deployed checkpoint SHA-256
  match;
- performance gate: at least 1.2x total speedup and no p95 regression.

Full mode never stops early. The completed 2026-07-23 quiet run passed all
parity checks (128 corpus decisions, 599 searched decisions, four shadow games,
eight workers, batch size 16, root budget 160, no p95 regression, 1.289x), and
the 2026-07-24 smoke re-confirmed activation. Re-run the smoke only when
executor or worker plumbing changes.

## Browser Executor Rollback

Parallel TD-root search uses the paired lockstep executor by default. To roll an
affected browser session back:

```text
?tdSearchExecutor=legacy
```

Use `&tdSearchExecutor=legacy` when the URL already has query parameters; remove
the parameter to restore the paired default. `tdSearchExecutor=paired` is an
explicit diagnostic selection. Any other value is a hard error rather than a
silent fallback.

Executor selection does not change the checkpoint, root-search budget, rollout
waves, UCB allocation, seeds, ordered result merge, diagnostics, or
selected-action policy. Effective executor mode is returned as outer-worker
response metadata, separate from search diagnostics and teacher targets.
