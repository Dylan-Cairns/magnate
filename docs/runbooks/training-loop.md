# Training Loop Runbook

Use the project `.venv` for all Python commands in this repo.

## Smoke And Primitive Commands

- `python -m scripts.smoke_trainer`
  Quick trainer smoke test.
- `python -m scripts.eval`
  Simple evaluation entrypoint.
- `python -m scripts.generate_teacher_data --games 200 --teacher-policy search --teacher-players both --out artifacts/teacher_data/teacher_search.jsonl`
  Teacher-label generation; teacher policy must emit root action probabilities.
- `python -m scripts.collect_td_self_play --games 200 --player-a-policy search --player-b-policy search --out-dir artifacts/td_replay --run-label td-replay-search`
  Replay generation for TD training primitives.
- `python -m scripts.train_td --value-replay artifacts/td_replay/<run>.value.jsonl --opponent-replay artifacts/td_replay/<run>.opponent.jsonl --steps 2000 --run-label td-v1`
  TD training primitive over replay files.

`scripts.train_td` also accepts `--value-replay-list` and
`--opponent-replay-list`, each pointing to a UTF-8 file with one replay path per
line. This avoids platform command-length limits for large frozen shard sets.

## Hard-Teacher Extra-Data Continuation Preparation

The imported 890-game V2 Hard replay is frozen as 690 training, 100
development, and 100 final-test games. Prepare the matched continuation
experiment with:

```powershell
python -m scripts.prepare_td_hard_extra_data_continuation
```

This validates
`configs/td-training/td-hard-extra-data-continuation-v1.json`, the imported
split manifest, the original 900-game replay, the reconstructed deployed-July
warm starts, and the frozen implementation fingerprint. It writes ignored
path lists plus eight exact commands for two seeds, two arms, and separate
value/opponent jobs. It also prepares four one-update guardrail smoke commands.
The command never launches training and does not read raw final-test replay.

The matched control continues the July bot on the original 900 games. The
treatment uses the same warm starts, seed, optimizer initialization,
hyperparameters, and 10,000-update budget, but trains on the original 900 plus
the 690 imported training games. Only the primary treatment pair may select a
checkpoint step. Selection ranks the ten 1,000-step checkpoints on all 100
development games by value Monte Carlo MSE and opponent soft-target
cross-entropy, then applies the frozen tie breakers. The control and replication
use that same selected step. The final 100 games remain sealed until the
candidate is frozen and replication direction is reviewed.

Replay lists that span collection runs must pass
`--replay-key-mode run-qualified-canonical-v1`. This fingerprints each shard as
`<run-directory>/<shard-basename>`, so identically named shards from different
runs cannot collide. The default basename scheme remains unchanged for
single-run path lists.

The resolved plan remains `review-required` with `launchAuthorized=false`.
Review it before executing even the one-update smoke commands. No launcher,
promotion, or deployed model-index change is performed by preparation.

Dry-run the first-class Windows launcher with:

```powershell
.\scripts\run_td_hard_extra_data_continuation.ps1 -DryRun
```

Start or resume the eight full jobs with:

```powershell
.\scripts\run_td_hard_extra_data_continuation.ps1
```

The PowerShell wrapper uses `windows_training_common.ps1` for project-local
caches, the project virtualenv, fixed native thread caps, array-safe invocation,
and durable outer log/status files under `artifacts/logs/`. It refuses a
duplicate experiment runner or trainer. The Python orchestrator reruns frozen
preflight, validates all eight command profiles, executes sequentially through
`td_loop_common.run_step`, streams merged output through the background output
pump, writes per-job heartbeat/progress files, and fails on the first nonzero
exit.

Rerunning the launcher is resumable at job boundaries. It skips only summaries
that validate the full 10,000-update configuration, seed, replay and warm-start
fingerprints, manifest and implementation fingerprints, row/file counts,
sampling trace, all ten expected checkpoint steps, and checkpoint paths.
Missing summaries rerun; malformed or mismatched summaries are hard errors. A
completion marker is written only after all eight summaries and checkpoint
sets validate. `-DryRun` performs preflight and reports the exact resume plan
without starting training.

The four one-update guardrail smokes completed on 2026-07-28. Both control
jobs loaded 900 files and 163,194 rows; both treatment jobs loaded 1,590 files
and 288,395 rows. Value training built 1,800 control and 3,180 treatment
TD-lambda sequences. Every summary and step-1 checkpoint embedded the expected
run-qualified replay hash, July warm-start hash, source-manifest hash,
implementation hash, primary seed `2026072801`, and one-update step. Training
provenance contained no development or final-test files.

The smoke sampling traces were:

- control value:
  `375fc286cf9f609bfa4cd42b135809d78500bd58533b1f2a81b5c2a60097c357`;
- control opponent:
  `3749492b93ba12991371cad33f3dc8f10e02047121e2417e27a042d283b96b0c`;
- treatment value:
  `aaa42decbfa42842ec34a7b6456df07511b6f5d8a38f1b4e9dc89ef90685b748`;
  and
- treatment opponent:
  `4de6c785ad5f1fd995393035cbd911ec91fee58ce0ee49ae8f7fe582dcabfb9b`.

Unlike augmentation ablations over one shared buffer, this data-volume
experiment must not require equal control/treatment sampling-index hashes:
the buffer sizes differ by design. The matched controls are the RNG seed,
warm starts, optimizer initialization, hyperparameters, batches, and update
counts. One-update losses are plumbing diagnostics, not selection evidence.

All eight 10,000-update jobs completed and passed the launcher's summary and
checkpoint validation on 2026-07-29.

Dry-run the frozen development-only evaluation with:

```powershell
.\scripts\run_td_hard_extra_data_development.ps1 -DryRun
```

Start or resume it with:

```powershell
.\scripts\run_td_hard_extra_data_development.ps1
```

This launcher uses the same shared Windows cache, array-safe invocation,
logging, status, thread-cap, and duplicate-process infrastructure as training.
The Python orchestrator validates the completed training summaries and all 80
checkpoint hashes, evaluates the primary treatment at steps 1,000 through
10,000 on the frozen development replay, and applies the manifest's rank-sum
and ordered tie-break rules. It then evaluates primary control, both
replication arms, and the unchanged July incumbent at the selected step.
Existing results are reused only after their checkpoint, step, replay-content,
key-mode, row-count, and metric contracts validate. The final-test replay is
not an input to this stage and remains sealed.

The development stage completed on 2026-07-29 and froze step 9,000. Step 9,000
and step 10,000 tied at rank sum 4; step 9,000 won the first tie breaker with
lower value Monte Carlo MAE. Against the matched control, the primary treatment
improved value MSE by 1.74% and opponent cross-entropy by `0.007629`. The
replication improved opponent cross-entropy by `0.011045` and value MAE by
`0.005334`, but its value MSE was 0.21% worse than replication control. Both
treatment seeds beat the unchanged incumbent on value MSE, value MAE, opponent
cross-entropy, and teacher-top-action agreement.

This is mixed replication on the frozen primary metrics: the opponent result
agrees, while value MSE reverses slightly. The manifest says contradictory
replication blocks promotion. No final-test replay or full-game evaluation has
been run. The aggregate is
`artifacts/td_extra_data_evals/td-hard-extra-data-continuation-v1/development/development.result.json`.

### Overnight Heuristic-V2 Comparison

The historical July checkpoint scored 88-32 in a 120-game comparison against
heuristic-v2 medium. Run the same 60 paired seeds and search settings against
the frozen primary-treatment step-9,000 checkpoint with:

```powershell
.\scripts\run_td_hard_extra_data_heuristic_benchmark.ps1
```

Use `-DryRun` to execute the full checkpoint, model-pack, Node, argument, and
resume preflight without starting a game. The launcher verifies the frozen
development result and checkpoint hashes, uses an isolated experimental model
index, invokes TSX through the fnm-resolved `node.exe`, refuses duplicate
matchup parents, and logs through `Invoke-MagnateLoggedCommand`.

The matchup uses five existing paired-seed workers and writes each completed
two-game seat swap atomically. Rerunning the same launcher resumes from those
pair checkpoints and rebuilds the final `matchup.json` and `summary.md` under:

`artifacts/ts-bot-evals/td-hard-extra-data-continuation-v1-primary-treatment-step-09000-vs-heuristic-v2-medium-120/`

This is a useful like-for-like diagnostic against the historical 73.3% result.
It does not spend the sealed replay final test or override the experiment's
mixed-replication promotion block.

## District-Symmetry Pilot Preparation

```powershell
python -m scripts.prepare_td_district_symmetry_ablation
```

This validates the checked-in
`configs/td-training/district-s4-ablation-pilot-v1.json`, verifies replay and
warm-start hashes, computes byte-level content fingerprints for the full
replay and both sides of the deterministic 800/100 shard split, verifies the
frozen training implementation, and writes hashed path lists plus eight exact
commands to the ignored resolved manifest. It also prepares four one-update,
full-800-shard guardrail smoke commands. The preparer never launches training.
The source and resolved manifests must both remain
`review-required` with `launchAuthorized=false` until explicit review.

The training primitive's intervention flags are
`--district-augmentation none|s4|s4-orbit` and
`--district-augmentation-seed <integer>`. Both S4 modes require the explicit
experiment seed; `s4` samples one permutation per row, while the opponent-only
`s4-orbit` mode expands each raw row to all 24 fixed-D3 permutations. Control
mode is an exact no-op. Training summaries include raw sampling-trace
hashes and checkpoints embed the replay, warm-start, source-manifest, and
implementation fingerprints. A run fails before loading/training if a frozen
fingerprint differs. Raw sampling hashes must match between each
control/candidate pair before results are interpreted.

The guardrail pass completed all four prepared one-update smokes against the
full 800-shard lists with four threads. Each path loaded 145,014 rows. The
control and S4 value traces both hashed to
`412d7f3de1755d52bc49410281a9a89e20f3da028e03c419052f6d633f628214`;
the opponent traces both hashed to
`8c9ab989480de4306a5cad78ac952861a419905bd03f699fcf4916c254a3fc34`.
These checks do not authorize or substitute for the 5,000-update runs.

Run the eight prepared training commands sequentially: keep at most one
trainer active at a time. Each command caps PyTorch at four intra-op threads
and one interop thread; launching multiple commands concurrently would exceed
the laptop-oriented CPU budget.

Launch or resume the complete training phase from PowerShell with one command:

```powershell
.\scripts\run_td_district_symmetry_pilot.ps1
```

The launcher validates the frozen command profile, refuses a duplicate pilot
trainer, skips only commands with a readable 5,000-update summary, writes one
log per command, stops on the first failure, and writes a completion marker
only after all eight summaries exist. Use `-DryRun` to validate and list the
pending commands without starting training.

After all eight training summaries exist, prepare—but do not execute—the exact
post-training evaluation plan:

```powershell
python -m scripts.prepare_td_district_symmetry_evaluation
```

Launch or resume only the first evaluation stage—experimental pack exports,
complete heldout replay metrics, and direct symmetry audits—from PowerShell:

```powershell
.\scripts\run_td_district_symmetry_evaluation_stage1.ps1
```

The stage-one launcher validates the prepared plan, executes sequentially,
caps heldout PyTorch work at four threads, regenerates isolated experimental
packs from their frozen checkpoints, skips only validated heldout/symmetry
results, logs every command, and refuses duplicate evaluation processes. It
also verifies that the deployed model-pack index and checkpoint registry did
not change before writing its completion marker. Use `-DryRun` to list pending
work without exporting packs or starting evaluation.

This command rejects intermediate checkpoints, provenance mismatches, and
unequal paired sampling traces. It freezes step 5,000 from `pilot-a` as the
only promotion-eligible candidate before reserved repetitions are used. It
then prepares:

- all-100-shard heldout value and opponent metrics through
  `scripts.evaluate_td_replay_holdout`;
- deterministic 10,000-row, all-24-permutation symmetry audits from the
  validation opponent path list;
- exposed strategic runs for both seeds and the reserved 24-47 run for the
  already-frozen primary candidate;
- paired, seat-swapped full-game configs at the frozen TD V2 Medium search
  budget; and
- candidate/control/cross-component browser packs under ignored
  `public/model-packs-experiments/`, with their own index and explicit
  `--no-set-default` exports.

The deployed `public/model-packs/index.json`, checkpoint registry, and bot
defaults are outside this experiment. Symmetry improvement is a required
diagnostic, not a sufficient promotion condition; heldout noninferiority,
replication direction, strategic behavior, and full-game strength remain
separate gates.

## Opponent-Only Complete-Orbit Follow-up

The first pilot improved opponent symmetry but missed its action-symmetry
gates. The controlled follow-up changes only opponent/action training. Its
matched control continues to sample one random S4 permutation per raw row; the
treatment averages the ordinary soft-target cross-entropy over all 24
permutations of every sampled row. It introduces no additional loss weight.
The successful first-pilot augmented value checkpoint remains fixed per seed.

Validate and resolve its frozen inputs without launching training:

```powershell
python -m scripts.prepare_td_opponent_orbit_ablation
```

The checked-in manifest is
`configs/td-training/district-s4-opponent-orbit-pilot-v1.json`. It reuses the
exact prior 800/100 split, July opponent warm start, two training seeds, raw
batch size 64, 5,000 updates, and four-thread laptop profile. The preparer
verifies full replay content, warm-start, fixed value, source-manifest, and
training-code fingerprints, then writes four opponent-only training commands
and two one-update guardrail commands.

The guardrail smokes completed against all 800 training shards. Both arms
loaded 145,014 opponent rows and produced the same raw sampling trace:
`8c9ab989480de4306a5cad78ac952861a419905bd03f699fcf4916c254a3fc34`.
The random-S4 control used one transformed copy per raw row; the complete-orbit
treatment used 24 copies, for effective batch sizes 64 and 1,536 respectively.

Launch or resume the four long jobs sequentially with:

```powershell
.\scripts\run_td_opponent_orbit_pilot.ps1
```

The launcher reruns the non-launching preflight, caps native and PyTorch work
at four threads, refuses duplicate trainers, validates any existing summary
before skipping it, logs each command, and verifies equal control/treatment
raw-sampling traces per seed before writing its completion marker. Use
`-DryRun` to revalidate everything and list the four jobs without starting
training. Do not run reserved strategic repetitions or full-game promotion
tests until heldout opponent noninferiority and the predeclared direct action-
symmetry gates both pass.

The four training runs and stage-one diagnostics completed on 2026-07-17.
Both complete-orbit checkpoints slightly improved heldout opponent
cross-entropy and passed the `+0.01` noninferiority gate. The direct symmetry
gates failed reproducibly:

| Seed      | Top-action agreement, control -> orbit | Mean probability drift reduction |
| --------- | -------------------------------------: | -------------------------------: |
| `pilot-a` |                       82.85% -> 86.36% |                            25.0% |
| `pilot-b` |                       83.03% -> 86.09% |                            25.2% |

The required thresholds were at least 95% orbit agreement and at least 50%
drift reduction versus the matched control. Complete-orbit augmentation is
therefore a real but insufficient correction; the candidate is not eligible
for reserved strategic or full-game promotion tests. The next intervention
should enforce fixed-D3 S4 symmetry in the opponent/action architecture rather
than add augmentation weights or action-specific heuristic boosts. Stage-one
artifacts are under ignored
`artifacts/td_ablation_evals/district-s4-opponent-orbit-pilot-v1/`; the
deployed model index and checkpoint registry remained unchanged.

## Evaluation

- `python -m scripts.eval_suite --mode certify --games-per-side 200 --workers 2 --candidate-policy search --opponent-policy heuristic`
  Canonical side-swapped evaluation. Supports `--mode gate|certify`, deterministic worker sharding, worker thread caps, and separate td-search checkpoints per side.
- `python -m scripts.search_teacher_sweep --pack coarse-v1 --games-per-side 60 --jobs 1 --workers 1 --opponent-policy heuristic --run-label search-coarse`
  Search profile sweep for teacher-data and search tuning work.

## Bootstrap Or Recalibration Loop

```powershell
python -m scripts.run_td_loop --run-label td-loop-r1 --chunks-per-loop 3 --collect-games 1200 --train-steps 20000 --eval-games-per-side 200 --eval-opponent-policy search --promotion-min-ci-low 0.5
```

This path supports `--collect-workers`, `--eval-workers`, `--eval-seed-start-indices`, explicit `--train-value-target-mode td0|td-lambda`, and cloud presets via `--cloud --cloud-vcpus 8|16|32`.

## Self-Play Loop

```powershell
python -m scripts.run_td_loop_selfplay --cloud --cloud-vcpus 16 --run-label td-loop-selfplay-r1 --chunks-per-loop 12 --collect-games 600 --train-steps 10000 --eval-games-per-side 200 --incumbent-eval-games-per-side 200 --progress-heartbeat-minutes 30 --eval-progress-log-minutes 30
```

The self-play loop uses mixed td-search-heavy collection, promoted opponent-pool sampling, accepted-generator gates, and final dual promotion gates versus fixed `search` plus incumbent `td-search`.

Important controls:

- `--checkpoint-selection-games-per-side` cheaply compares saved checkpoints before selecting a chunk candidate.
- `--generator-update-chunks` controls generator gate cadence.
- `--block-selection-*` settings select the best candidate in a generator block.
- `--chunk-gate-*` settings configure resumable sequential generator gates through `scripts.eval_suite --mode gate`.
- `--train-value-target-mode td-lambda --train-td-lambda 0.7` is the normal value-target mode.
- `--train-replay-window-chunks` controls replay-window width.
- `--train-replay-window-source accepted|recent` selects gate-passing chunks or recent trained chunks for replay windows.

## Resume And Promotion

- `python -m scripts.resume_td_loop_selfplay --run-id <interrupted-selfplay-run-id>`
  Resume an interrupted self-play loop from the latest fully completed chunk while preserving separate learner/generator checkpoints and replay histories.
- `python -m scripts.resume_td_loop_run`
  Bootstrap recovery helper.
- `python -m scripts.promote_td_checkpoint --key <key> --value-checkpoint <value.pt> --opponent-checkpoint <opponent.pt> --source-run-id <run-id> --set-default --add-to-opponent-pool`
  Copy a promoted value/opponent checkpoint pair into `models/td_checkpoints/<key>/` and register it in the checkpoint manifest.

## Benchmarks

- `python -m scripts.benchmark_collect_search_profiles --workers 4 --games 8`
  Benchmark td-search collect throughput across a small `search-worlds` and `search-depth` matrix.
- `python -m scripts.benchmark_selfplay_collect_setup`
  Compare single-process versus sharded self-play collection and recommend a `--collect-workers` setting for the current machine.

Use `--help` on each script for the full option surface.
