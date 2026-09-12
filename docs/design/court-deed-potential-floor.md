# Court Deed Potential Floor

Status: predeclared experiment. Implementation and benchmark kit landed; the
benchmark series are run manually by the project owner and analyzed by an agent.

## Problem

Heuristic v2 credited a newly bought deed with zero district-score potential:
`scoringDeltaForAction` weights a deed's added score by
`(progress / target)^2`, which is 0 at progress 0, and the earning-access term
is 0 for the four extended-deck Courts because they never provide income. A
Court deed therefore scored ~0.01 at every game phase while a rank-9 deed
scored ~2.5-4.0 from future income access. Diagnostic self-play produced 0
court deeds bought and 0 courts developed across 81 decisions holding a Court.
See `.tmp` probes (removed) and `memoryBank/systemPatterns.md`, which already
states that district-potential scoring should include newly bought deeds.

## Change

`SearchPolicyConfig.deedPotentialBase` (optional, [0, 1]) is threaded through
rollout search into the v2 scorer. `potentialStackScore` becomes:

```text
completionWeight = base + (1 - base) * (progress / target)^2
```

- `base = 0` reproduces the pre-floor scoring bit-for-bit and is the control
  arm. The scorer's own default remains 0 so direct diagnostic callers are
  unchanged; search policies resolve the deployed default
  `DEFAULT_DEED_POTENTIAL_BASE = 0.2`.
- TD-root search is unaffected: its root guide, rollout action choice, and leaf
  value all use the TD model, not heuristic v2.
- Calibration candidates from the proposal probe: 0.15 / 0.2 / 0.25 / 0.3.
  0.2 is the predeclared default.

## Benchmark Protocol

Configs: `configs/bot-eval/court-fix/`. Artifacts:
`artifacts/ts-bot-evals/<run>/`. Both arms are explicit; the control arm is the
current behavior, so no separate pre-change baseline series is needed.

| Series | Config | Games | Purpose |
| :--- | :--- | ---: | :--- |
| Standard A/B | `standard-medium-ab.json` | 60/side | Non-inferiority |
| Extended A/B | `extended-medium-ab.json` | 60/side | Improvement + court usage |
| Hard smoke | `standard-hard-smoke.json` | 10/side | Deployed-profile sanity |
| Hard smoke | `extended-hard-smoke.json` | 10/side | Deployed-profile sanity |
| Contingency sweep | `deed-base-sweep-extended.json` | 40/side/candidate | Only if gates fail or land ambiguous |

Commands (wire artifact paths back for analysis):

```powershell
yarn bot:eval head-to-head --config configs/bot-eval/court-fix/standard-medium-ab.json --workers <n> --progress-interval-seconds 30
yarn bot:eval head-to-head --config configs/bot-eval/court-fix/extended-medium-ab.json --workers <n> --progress-interval-seconds 30
yarn bot:eval head-to-head --config configs/bot-eval/court-fix/standard-hard-smoke.json --workers <n>
yarn bot:eval head-to-head --config configs/bot-eval/court-fix/extended-hard-smoke.json --workers <n>
yarn bot:eval deed-potential-report --artifact <matchup.json> [--out-dir <dir>]
```

### Predeclared Gates

Frozen before any series runs. The report command emits these statuses.

| Gate | Applies | Pass |
| :--- | :--- | :--- |
| `standard-paired-noninferiority` | standard | paired mean win-margin CI95 lower bound > -0.07 (minimum 30 pairs) |
| `standard-deed-buy-behavior` | standard | candidate/control deed buy-rate ratio in [0.8, 1.2]; special-cased when the control rate is ~0 |
| `extended-paired-improvement` | extended | paired mean win-margin CI95 lower bound > 0 |
| `extended-court-utilization` | extended | candidate acquires >= 1 court and completes >= 1 court |
| `extended-court-dump` | extended | candidate never sells a court while a court build is legal |
| `replay-integrity` | both | every recorded game replays through canonical legal actions |

### Extension Rule

- Standard: if the 60/side result lands with a point estimate below 50% and a CI
  that still includes the -0.07 margin, extend the same config to
  `gamesPerSide: 120` (pairs 1-60 are deterministic and replay identically).
  Never extend a good result.
- Extended: `observe` on `extended-paired-improvement` may be extended once to
  120/side; `fail` on court utilization means revisit the design, not the N.
- Ambiguous sweep results select a lower base only through the predeclared
  candidate list; if `standard-deed-buy-behavior` fails, prefer 0.15 before
  abandoning the floor.

### Ownership

- Agent: implementation, tests, configs, report tooling, artifact analysis,
  gate calls, docs.
- Owner: run the four series (and any single predeclared extension) on the
  project machine, then hand artifact paths back.

## Rollback

`deedPotentialBase = 0` restores the pre-change behavior. Removing the field
from search configs returns the deployed default to 0.2.
