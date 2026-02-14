# Active Context
> Reminder: Start at `AGENTS.md` and review every core Memory Bank file before planning or acting.

## Current Focus

- Complete the project truth-source reset so all docs match current decisions.
- Prepare implementation sequencing for TS-canonical engine completion.
- Define and lock bridge interface contract v1 before bridge/trainer coding.

## Recent Changes

- Confirmed architecture direction:
  - TypeScript engine is canonical for Magnate rules.
  - Python trainer will call TS engine through a stable bridge.
  - Full shared rules schema was dropped in favor of a small interface contract.
- Added `memoryBank/bridgeInterfaceContract.md`.
- Updated `README.md`, `AGENTS.md`, and Memory Bank docs to align with these decisions.
- Removed stale docs language implying pass actions or outdated architecture assumptions.

## Next Steps

- Finalize bridge command payloads and metadata fields for v1.
- Implement missing TS engine rule flow:
  - setup game
  - turn FSM (taxation/income/play/draw)
  - full legality coverage
  - scoring and terminal logic
- Add targeted tests for setup, legality, taxation/income, and scoring.
- Scaffold bridge runtime and validate contract with a Python client smoke test.

## Active Decisions and Considerations

- Keep all game-rule semantics in TS for v1.
- Python remains a consumer of engine outputs, not a second rules engine.
- Preserve deterministic behavior end to end (seed + action log replay).
- No pass action; if no legal develop/deed is available, selling is mandatory.
- Trade remains 3:1 one exchange per action and can be chained.

## Insights and Learnings

- Magnate complexity increases drift risk; a single canonical rules runtime is the safest path.
- A narrow bridge contract yields most of the cross-language stability benefit with much less maintenance cost than full rules schema duplication.
- Keeping docs synchronized early prevents architecture confusion during implementation.

_Active context updated on 2026-02-19._