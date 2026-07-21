# TD Two-Lane Browser Benchmark

This benchmark is the first stop/go gate for browser TD inference batching. It
compares the current scalar opponent scorer with a fused two-lane JavaScript
candidate inside a dedicated browser worker. It does not change production
search scheduling, rollout waves, search budgets, seeds, checkpoints, or bot
selection.

The deterministic corpus is built from legal game decisions. Every run records
the selected model pack and SHA-256 of its weights, exact-logit and argmax
parity, scalar and paired timing, and the resulting stop/go recommendation.

## Daytime Smoke

The smoke checks browser launch, model loading, provenance, exact parity, and
artifact output. Its timing sample is intentionally too small for a performance
decision.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_td_two_lane_browser_benchmark.ps1 -Mode smoke
```

## Quiet-Machine Run

Run this while the laptop is plugged in and otherwise idle. Close builds,
training jobs, video calls, games, and other sustained CPU workloads. The
benchmark alternates scalar-first and paired-first rounds to reduce order and
thermal bias, but competing work can still distort the result.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_td_two_lane_browser_benchmark.ps1 -Mode full -States 128 -Rounds 2000
```

The command starts a temporary localhost Vite server and an isolated headless
Edge or Chrome profile, writes a timestamped result under
`artifacts/benchmarks/td-two-lane-browser-*`, and shuts both down. It may finish
well before morning. Full mode has no automatic timeout and completes every
configured round unless it is manually interrupted. Pass `-BrowserPath` if
neither browser is in a standard Windows location.

## Decision Gate

Continue to the lockstep two-rollout search spike only when all of these hold:

- `correctness.exactMismatchCount` is `0`;
- `correctness.argmaxMismatchCount` is `0`;
- checksums match; and
- `timing.speedup` is at least `1.3`.

These are post-run decision gates, not early-exit conditions. The benchmark
always completes every configured warmup and measurement round before it
calculates speedup and writes the recommendation.

The kernel gate is necessary but not sufficient. A passing result authorizes a
separate benchmark-only lockstep integration, followed by exact task, root
diagnostic, selected-action, and transcript parity plus a 20% end-to-end
browser decision-latency gate. A failing quiet-machine result means browser
batch-two work should stop; it does not justify increasing rollout-wave size.
