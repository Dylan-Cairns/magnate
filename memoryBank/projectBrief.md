# Magnate (Web) + RL Bot — Project Brief

## One-liner

Build a web-playable, single-player **Magnate** with a competent computer opponent. Core game engine in **TypeScript** (runs in browser and Node). Train the bot in **Python** by calling the TS engine through a tiny Node bridge. **Official rules only.**

## Goals

- **Accurate rules**: Implement Magnate exactly as written on the Decktet wiki.
- **Competent bot**: Beats a random baseline and a simple scripted baseline; feels fair and capable to humans.
- **Single codebase**: Same TS engine works in browser (UI) and in Node (training).
- **Reproducible & testable**: Seeded randomness, pure state updates, and snapshot tests.
- **Static deploy**: Ship to GitHub Pages; model loaded as a static asset (no server required).

## Non-Goals (v1)

- Multiplayer, accounts, persistence, mobile polish, localization.
- Fancy training tricks (tree search, model-based planning, etc.).

## Game scope (rules parity)

Two-player Magnate per the official write-up: setup, turn order, resource collection/taxation, buying, developing, trading, selling, end, and scoring. We will cross-check every rule against the wiki text. **Only official rules.**

## Architecture

- **TS Engine (core)**
  - Pure functions over immutable state. No I/O or hidden globals inside reducers.
  - Deterministic when given a seed.
- **Browser UI**
  - Thin React view over the engine. Human vs bot only.
- **Node Bridge (for training)**
  - Tiny Node process that hosts the TS engine and speaks JSON over stdin/stdout (or local HTTP).
  - Exposes: `reset`, `step`, `legalActions`, `observation`, `seed`, `serialize`.
- **Python Trainer**
  - Simple reinforcement learning loop (e.g., PPO from Stable-Baselines).
  - Presents only legal moves to the agent (illegal moves are hidden).

## Data model (engine)

- **State**
  - Deck & discard; hands; district line; player resources by suit; deeds/developments; turn/phase; RNG seed; minimal history for debugging.
- **Actions (discrete)**
  - Buy deed, buy outright, develop, trade (3:1), sell (per rules), pass (only when legal).
- **Legality**
  - `legalActions(state)` returns the exact set allowed now—used by UI and training.

## Observations (for the bot)

- **Cards**
  - Fixed-length “one-hot” vectors for possible cards in hand (1 at a card’s position if held, else 0).
- **Board**
  - Compact features for each district (stack depth, top card’s suits/rank), remaining deck counts, whose turn, resource counts.
- **Scaling**
  - Numeric values normalized to a common range (e.g., 0..1) so no single feature dominates learning.

## Reproducibility & tests

- **Seeded RNG**
  - Start randomness from a known seed so the same game can be replayed exactly.
- **Pure reducers**
  - `nextState = reduce(prevState, action)` with no side effects.
- **Snapshot fixtures**
  - Save serialized states after known move sequences; fail fast if a refactor changes outcomes.

## Training plan

- **Method**
  - The bot improves by playing many simulated games and favoring choices that lead to wins (we can use PPO under the hood).
- **Illegal-move mask**
  - Only present legal moves to the agent to speed learning and avoid nonsense.
- **Reward**
  - 1 for a win, 0 for a loss. Optional tiny “progress” points for clearly beneficial milestones if needed (can start without).
- **Opponents**
  - Random baseline and a simple scripted baseline (e.g., greedy buy/develop). Later, occasional self-play snapshots are optional.
- **Throughput**
  - Start single process; add parallel workers later only if training feels too slow (not required).

## Packaging & deploy (GitHub Pages)

- **Model as static asset** (no server): export trained weights to a small JSON file; implement a tiny TS forward pass to choose actions.
- The UI loads `assets/model.json` at start; all inference runs in the browser.

## Quality bar

- Rule parity confirmed against the official wiki.
- Deterministic seeds and solid unit coverage for setup, turn flow, resource changes, selling, and scoring.
- Snapshot tests for tricky sequences.

## Milestones

1. **Spec + deck data** (card list, suits/ranks) and rule checklist from the wiki.
2. **TS engine**: reducers, legality, scoring, seeded RNG + unit/snapshot tests.
3. **Minimal UI**: human vs simple scripted bot.
4. **Node bridge**: stdin/stdout JSON API.
5. **Python trainer**: basic PPO training with illegal-move mask; beat random + scripted baselines.
6. **Static model export** + browser inference; release on GitHub Pages.
7. **Polish**: usability, small hints, and test hardening.

## References

- Magnate rules: http://wiki.decktet.com/game:magnate
- SIMPLE project (self-play RL pattern): https://github.com/davidADSP/SIMPLE
- Walkthrough article: https://medium.com/applied-data-science/how-to-train-ai-agents-to-play-multiplayer-games-using-self-play-deep-reinforcement-learning-247d0b440717
- Cline Memory Bank (context for this file): https://docs.cline.bot/prompting/cline-memory-bank
