# Active Context

## Current Focus

- Close remaining rules-parity gaps with scenario-driven integration tests.
- Keep the UI/controller thin and policy-agnostic so random and trained bots share one action-selection boundary.
- Preserve a stable TS-Python contract while warm-start training/evaluation loops are exercised through the bridge runtime.
- Add a phase-1 determinized search baseline as an additive policy path for evaluation/benchmarking without disrupting PPO/BC workflows.

## Locked Decisions

- TypeScript engine is the canonical rules implementation.
- Python training calls TS through a Node bridge.
- Shared boundary is a small, versioned interface contract.
- Native Python rules are out of scope unless throughput becomes a proven bottleneck.

## Current State

### Engine

- Deterministic setup/deck lifecycle, legality generation, and reducer transitions are implemented.
- Turn resolution is phase-driven via `advanceToDecision`, with a unified decision phase (`ActionWindow`).
- Income suit choice is explicit (`choose-income-suit`) and actor ownership is handled correctly.
- Income-choice logging now attributes entries to the chooser even when turn ownership is restored in the same transition.
- Scoring and terminal resolution are implemented (`scoreGame`, terminal finalization, `isTerminal`).
- Canonical game init exists via `newGame(seed, { firstPlayer })`.
- `newGame` now validates `firstPlayer` at runtime and throws on unknown values.
- Turn-state bookkeeping was simplified:
  - deck reshuffle state is canonical for exhaustion/final-turn flow.
  - income-choice return owner is stored as `PlayerId` in state.
  - transitional `IncomeRoll` phase was removed.
- Canonical bridge action surface exists in `src/engine/actionSurface.ts`:
  - stable `actionStableKey` generation
  - canonical lexicographic `legalActions` ordering by key

### Observability

- `toPlayerView` and `toActivePlayerView` are implemented and test-covered.
- Opponent hand contents and draw order are hidden; public board/deed/resource state remains visible.

### UI

- Playable React shell exists (`src/App.tsx`) with human vs bot and a profile selector.
- UI dispatches only engine-legal actions and uses grouped follow-up pickers where options are noisy.
- Canonical Decktet card-image mapping for UI assets now exists in `src/ui/cardImages.ts`:
  - maps every engine `CardId` to Adaman-derived filenames in `src/assets/CardImages`
  - validates asset resolution at module load and via dedicated regression tests (`src/ui/cardImages.test.ts`)
- Card tiles now render mapped art in `src/App.tsx` with:
  - a dedicated framed image panel (1px gray border)
  - rank/suits and deed progress share a single metadata row
  - image panel sizing follows Decktet source aspect ratio, stretches art with `object-fit: fill`, and draws the 1px frame border via an overlay pseudo-element to avoid edge clipping artifacts
  - deed-token overlays are retained without reserving a separate top/bottom progress lane, using play-area chip sizing in a centered vertical token column with a semi-opaque connected gray enclosure for readability
- Official Decktet suit icons are now canonical in `src/ui/suitIcons.tsx`:
  - shared `{Suit}` text tokens drive action-label formatting and inline suit-icon replacement
  - cards, action text, resource/crown rows, district markers, and deed-token chips all render the same SVG set from `src/assets/icons/*.svg`
  - action labels now reuse the same token-chip component styling as board/resource chips for consistent opaque presentation
  - chip icon sizing is tuned to fill nearly the full circular chip area
- Player/board layout is now split for district-space priority:
  - side columns are sized to three-card hand width; each player panel shows title/score + hand only
  - non-hand side-column panels (actions/info cards) render at 80% width, aligned to outer edges, with ~50% opacity fill so the background texture shows through
  - bot hand panel is at the top of the right column; human hand panel is at the bottom of the left column
  - crowns/resources render inside the middle board column as mirrored edge rows:
    - top row: bot resources-left/crowns-right, right-aligned in center lane
    - bottom row: human crowns-left/resources-right, left-aligned in center lane
    - crown/resource chips are upscaled for readability (3rem token size)
  - full score breakdown is available via hover/focus popovers on each player score badge
- Brand/info controls are now menu-driven:
  - title row (top-left, above Actions) includes a right-aligned hamburger button
  - seed/new-game controls and bot-profile selector moved into a toggleable options dropdown
- Human-only turn reset is available in the actions list:
  - snapshot is captured at the start of the human `ActionWindow` (post roll/income resolution).
  - reset restores that snapshot before `end-turn`; it is not part of engine `legalActions`.
- Controller/policy boundaries are extracted:
  - `createSession` / `stepToDecision` in `src/engine/session.ts`
  - async-capable `ActionPolicy` + bot profile catalog in `src/policies/`
  - browser-trained PPO profile is now enabled (`ppo_champion_2026-02-23_seed7`)
  - champion PPO profile is the default bot selection in UI
  - action grouping/picker presentation helpers in `src/ui/actionPresentation.ts`
- Right info column includes bot hand, roll result, deck state, and full scrollable log.
  - roll panel is above deck state, uses larger dice icons, and renders taxed suit using the shared token-chip component
  - deck state now renders visual deck/discard piles with count badges:
    - deck uses striped card-back styling with "Cards remaining" hover text, showing up to three stacked backs (or a dashed empty placeholder when draw count is zero); deck mini-stack uses a uniform diagonal offset
    - discard shows up to three top discarded cards in a stacked mini-pile, with a slight fan effect when multiple cards are visible ("Discard pile" hover text)
    - reshuffles remaining is shown above the piles
- Hidden opponent hand cards use striped card-back styling.
- Final-turn warning is surfaced in the actions header during the final-turn window.
- District lanes render centered overlapping card stacks; bot-visible cards (district + crowns) use bot perspective (rank/suits at bottom, progress at top).
  - stack shadow now uses a two-layer model:
    - strong outer shadow is applied at the stack container level (`.lane-stack`)
    - each overlapping card gets a subtle directional seam shadow to separate overlaps (human casts upward, bot casts downward)
  - district stack overlap is tighter (`card-stack-step` reduced) so less of underlying cards remains exposed

### Bridge

- Node NDJSON bridge runtime is implemented at `src/bridge/cli.ts` + `src/bridge/runtime.ts` (`yarn bridge`).
- Implemented commands: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`.
- Error-code envelope handling is implemented per contract (`INVALID_COMMAND`, `INVALID_PAYLOAD`, `ILLEGAL_ACTION`, `STATE_DESERIALIZATION_FAILED`, `INTERNAL_ENGINE_ERROR`).
- Bridge contract artifact is versioned at `contracts/magnate_bridge.v1.json`.
- Bridge behavior has dedicated tests in `src/bridge/runtime.test.ts`.

### Trainer Scaffold

- Python bridge client and environment wrapper are implemented:
  - `trainer/bridge_client.py`
  - `trainer/env.py`
- Training encoders are implemented and documented:
  - `trainer/encoding.py` (`OBSERVATION_DIM=186`, `ACTION_FEATURE_DIM=40`)
  - `docs/TRAINING_ENCODING.md`
- Baseline policy/eval/training scripts are implemented:
  - policies: `trainer/policies.py` (random + heuristic + checkpoint-backed BC + checkpoint-backed PPO)
    - additive determinized search policy (`search`) with sampled hidden-state worlds and bounded rollout lookahead
  - eval: `trainer/evaluate.py` + `scripts/eval.py`
    - eval CLI now supports `search` and search knobs (`--search-worlds`, `--search-rollouts`, `--search-depth`, `--search-max-root-actions`, `--search-rollout-epsilon`)
    - eval now emits default progress updates every 25 games (plus final) and persists result artifacts under `artifacts/evals/`
  - canonical holdout benchmark: `trainer/benchmarking.py` + `scripts/benchmark.py`
    - benchmark CLI now supports `search` and the same search knobs
    - fixed seed prefixes: `bench-random-holdout`, `bench-heuristic-holdout`
    - default games per matchup: 200
    - selection score: `0.7 * heuristic + 0.3 * random`
  - benchmark seed queue runner: `scripts/benchmark_queue.py` (sequential multi-seed benchmark automation + ranked summary)
  - sample collection + JSONL IO: `trainer/training.py`
  - behavior cloning warm-start: `trainer/behavior_cloning.py`
  - RL fine-tuning (stabilized REINFORCE): `trainer/reinforcement.py` + `scripts/finetune.py`
    - mixed-opponent training (self/heuristic/random)
    - BC-anchor regularization
    - fixed-holdout eval-based best-checkpoint selection (default)
  - PyTorch PPO scaffold: `trainer/ppo_model.py` + `trainer/ppo_training.py` + `scripts/train_ppo.py`
  - browser checkpoint export utility: `scripts/export_ppo_browser_checkpoint.py`
  - browser checkpoint artifact: `public/models/ppo_champion_2026-02-23_seed7.browser.json`
  - browser PPO runtime policy: `src/policies/ppoBrowserPolicy.ts`
  - TS feature-encoder parity: `src/policies/trainingEncoding.ts`
  - PPO seed queue runner: `scripts/train_ppo_queue.py` (sequential multi-seed automation)
  - training CLI: `scripts/train.py` (collect + optimize + checkpoint)
  - checkpoint-backed policy loading: `policy_from_name(..., checkpoint_path=...)`
  - project Python environment bootstrap: `requirements.txt` + `scripts/setup_python_env.ps1`
- Python scaffold has dedicated unittest coverage in `trainer_tests/`.

## Immediate Next Steps

1. Expand full-turn/full-game scenario coverage for rules parity.
2. Benchmark the new search baseline against heuristic/PPO over larger game counts to estimate real uplift and latency tradeoffs.
3. Validate PPO scaffold against BC/heuristic baselines and tune PPO objective/rollout settings.
4. Improve experiment logging/tracking around checkpoint-selection decisions over long runs.
5. Add model lifecycle docs for promoting trained checkpoints into browser-ready champion profiles.

_Updated: 2026-02-27._
