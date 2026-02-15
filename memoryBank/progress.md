# Progress

## Implemented

- Architecture is locked:
  - TS engine is canonical.
  - Python integration goes through a small versioned bridge contract.
- Core engine flow is implemented and deterministic:
  - setup/deck lifecycle
  - legality + reducer action handling
  - phase resolver (`advanceToDecision`)
  - scoring and terminal finalization
- Engine state model was simplified:
  - canonical exhaustion source is `deck.reshuffles` (removed duplicated `exhaustionStage`)
  - income-choice return owner is ID-based (`incomeChoiceReturnPlayerId`)
  - removed transitional `IncomeRoll` phase
- Canonical game creation exists via `newGame(seed, { firstPlayer })`.
- Player-scoped visibility projection exists and is test-covered (`toPlayerView` / `toActivePlayerView`).
- Playable browser client exists (human vs random bot) using engine truth APIs.
- Board UI now uses centered overlapping district stacks with player-specific visual perspective for readability.
- Human-only `reset-turn` UX is implemented via a UI snapshot anchor at human turn start (`ActionWindow` pre-card), restoring state prior to `end-turn` without changing engine action contracts.
- Policy/controller boundaries now exist:
  - `src/engine/session.ts` (`createSession`, `stepToDecision`)
  - async-capable `src/policies/types.ts` + `src/policies/randomPolicy.ts`
  - `src/policies/catalog.ts` profile registry with strict fail-fast resolution for unknown/unavailable profiles
  - `src/ui/actionPresentation.ts` (+ tests)
- UI card-image mapping is now integrated from Adaman assets:
  - canonical mapping module in `src/ui/cardImages.ts` (`CardId` -> image filename/url)
  - regression tests in `src/ui/cardImages.test.ts` enforce full-card coverage and expected relationships
- UI card rendering now consumes the mapping in `src/App.tsx`:
  - cards display art in a framed image panel with a 1px gray border
  - rank/suit and in-progress deed value render on one metadata line
  - image panel uses Decktet aspect-ratio sizing with `object-fit: fill`; border is rendered via an overlay frame layer to avoid clipped edges
  - in-progress deed suit chips now use play-area chip size, render as a centered vertical stack on the card image, and sit in a connected semi-opaque gray enclosure
- Official Decktet suit SVG icons are now wired across the UI:
  - shared suit-token mapping in `src/ui/suitIcons.tsx` replaces suit emojis in action text, card metadata, and token chips
  - player resources/crowns, district marker tokens, and card/deed chips all render from `src/assets/icons/*.svg`
  - action text now renders suits with the shared token-chip component (not standalone transparent icons)
  - chip icon CSS scales SVGs to fill nearly the full chip area
- Player/board column layout was rebalanced for district space:
  - side columns now anchor player panels by hand only (title/score + 3-card hand footprint)
  - non-hand panels in the side columns are now 80% width, outer-edge aligned, and rendered with ~50% opacity fill
  - bot hand sits at top-right column and human hand at bottom-left column
  - crowns/resources now render in mirrored top/bottom rows inside the center board column
  - center-lane token rows are edge-aligned: bot row packed right, human row packed left
  - crown/resource token chips in the center rows are enlarged (3rem)
  - full score breakdown remains available via hover/focus popovers on score badges
- Deck state panel now uses a visual pile layout:
  - deck pile shows striped card-back styling with hover text ("Cards remaining"), rendering up to three stacked backs with a uniform diagonal offset or a dashed empty placeholder when empty
  - discard pile shows up to three top discarded cards with hover text ("Discard pile"), including a slight fan effect when multiple cards are visible
  - reshuffles remaining is shown as a text line above the piles
- Hidden bot-hand cards use striped card-back styling.
- Roll result panel now appears above deck state, doubles die icon size, and shows taxed suit via the shared token-chip component.
- Card drop shadows now match district-divider strength; stacked district lanes only render outer shadow on the top card.
- District stack shadow behavior now uses a two-layer model (container outer shadow + subtle directional per-card seam shadow), and stack overlap was tightened by reducing `card-stack-step`.
- Brand panel controls were reorganized:
  - title now sits above Actions in the left column with a right-side hamburger trigger
  - seed/new-game and bot-profile controls now live in a dropdown options menu
- UI now exposes bot-profile selection and status text while keeping random legal fallback available.
- Runtime hardening and audit fixes landed:
  - `newGame` now validates `firstPlayer` at runtime
  - income-choice log attribution now reflects the chooser even when active player is restored
- Bridge runtime and deterministic action surface are implemented:
  - canonical action IDs/keys/order in `src/engine/actionSurface.ts`
  - NDJSON runtime in `src/bridge/runtime.ts` + CLI entrypoint `src/bridge/cli.ts`
  - command coverage: `metadata`, `reset`, `legalActions`, `observation`, `step`, `serialize`
  - bridge tests in `src/bridge/runtime.test.ts`
  - versioned contract artifact in `contracts/magnate_bridge.v1.json`
- Python trainer scaffold is implemented:
  - bridge client + env wrapper in `trainer/bridge_client.py` and `trainer/env.py`
  - training encoders in `trainer/encoding.py` with spec in `docs/TRAINING_ENCODING.md`
  - baseline policies/eval in `trainer/policies.py` and `trainer/evaluate.py`
  - canonical benchmark protocol in `trainer/benchmarking.py` + `scripts/benchmark.py`
    - fixed holdout seeds (`bench-random-holdout`, `bench-heuristic-holdout`)
    - default 200 games per matchup
    - locked selection score (`0.7 * heuristic + 0.3 * random`)
    - sequential multi-seed benchmark queue helper (`scripts/benchmark_queue.py`)
  - sample collection + sample IO in `trainer/training.py`
  - behavior-cloning optimizer/checkpointing in `trainer/behavior_cloning.py`
  - stabilized RL fine-tuning in `trainer/reinforcement.py`:
    - mixed opponents (self/heuristic/random)
    - BC-anchor regularization
    - fixed-holdout eval-based best-checkpoint selection (default)
  - PyTorch PPO scaffold in `trainer/ppo_model.py` + `trainer/ppo_training.py` + `scripts/train_ppo.py`
    - `scripts/train_ppo.py` now emits infrequent update-heartbeat progress logs to stderr
    - `scripts/train_ppo_queue.py` adds sequential multi-seed PPO run automation
  - browser PPO checkpoint export path is implemented:
    - `scripts/export_ppo_browser_checkpoint.py` exports `.pt` checkpoints to browser JSON
    - `src/policies/trainingEncoding.ts` provides TS parity for trainer feature encoding
    - `src/policies/ppoBrowserPolicy.ts` runs candidate-action PPO inference in browser
    - champion browser artifact tracked at `public/models/ppo_champion_2026-02-23_seed7.browser.json`
  - scripts: `scripts/smoke_trainer.py`, `scripts/eval.py`, `scripts/train.py` (collect + BC), `scripts/finetune.py` (BC -> RL), `scripts/train_ppo.py` (PPO scaffold)
  - project Python venv bootstrap is standardized (`requirements.txt`, `scripts/setup_python_env.ps1`)
  - unittest coverage in `trainer_tests/`
- Tooling gates include lint + typecheck + tests.

## In Progress

- Expanding rules-parity scenario coverage, especially full-turn/full-game edges.
- Tuning RL schedules/hyperparameters and benchmark tracking across BC, stabilized REINFORCE, and PPO scaffold loops.

## Remaining

- Experiment tracking/metrics for BC baseline, RL fine-tuning, and self-play runs.
- Surpass heuristic baseline consistently with current stabilized REINFORCE controls and/or PPO path.
- Deployment polish for static hosting path.

## Risks / Watch Items

- Remaining parity gaps are likely in edge-case sequencing rather than base mechanics.
- Bridge/API stability needs explicit guardrails as trainer integration starts.
- Trained profiles are still placeholders until bridge/runtime inference wiring is complete.

_Updated: 2026-02-24._
