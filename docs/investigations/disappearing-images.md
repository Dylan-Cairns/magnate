# Disappearing images: code investigation

Reviewed 2026-09-07 at `caedcf2`, using
`magnate-log-2026-07-22T01-33-43.json`. Scope: code inspection, direct component
rendering, mocked failure checks, existing targeted tests, and production build.
No prolonged browser session or Firefox memory reproduction was attempted.
The log does not identify its source commit; these conclusions concern the
current checkout, not necessarily every detail of the July executable.

## Conclusion

No confirmed cause of the disappearing images, and no demonstrated unbounded
leak across ordinary games with the same bot profile. There are actionable
worker-lifecycle and image-error-handling gaps. Neither is proof that memory
pressure caused this report.

## Logged payment window

- Turn 11, PlayerA, ActionWindow, V2 Hard, animations enabled, error null.
- The Diplomat (card 25, rank 8, Moons/Suns) is a deed in D2 with two Moons
  already invested. PlayerA has one Moon and one Sun available.
- Running canonical legal-action generation and rendering the actual
  `ActionPicker` produces `{Moons}x1` and `{Suns}x1`. The title and payment
  buttons contain the correct Moon/Sun `<img>` elements.
- `SuitText -> TokenChip -> SuitIcon` uses the same imported suit URLs used
  elsewhere. There is no separate payment-window fetch, lazy-loading condition,
  or per-game asset URL. No Sun-specific hiding rule was found in the reviewed CSS.
- All six suit SVGs parse as XML and contain no external href dependencies.
  The production build includes and references the Sun SVG.

This excludes missing action data or missing image markup for this state in the
current code. Server rendering does not verify browser layout, decoding, or paint.

## Findings

### Bot resources outlive the selected profile

`src/policies/catalog.ts` constructs one persistent policy per profile.
`src/policies/workerPolicy.ts` lazily creates an outer worker and exposes `close()`.
`src/ui/hooks/useGameController.ts` never calls that method on profile switch,
session reset, or unmount. Effect cleanup only invalidates results and clears
the scheduling timeout; a search already running can continue.

Each used profile can retain an outer worker plus up to eight search workers
(`src/policies/botWorker.ts`). Search workers retain the latest sampled world
states until the next initialization; TD profiles also cache their models.
Switching profiles can therefore leave multiple pools resident. The same profile
reuses its pool across games, so this is bounded retention in the normal path,
not evidence of a new pool leaking on every New Game click. Worker errors and
superseding selections do have termination paths.

### Image preload ignores decode failures; displayed images lack diagnostics

`src/ui/cardImages.ts:139` catches a rejected `image.decode()` promise and then
marks the URL successfully preloaded. A controlled mock confirmed that a failed
decode resolves the preload call and a second call creates no new image.
Actual load errors through `onerror` do reject and remove the pending cache entry.

The visible images in `SuitIcon` and `CardTile` have no application `onError`
handler, retry, or failure telemetry. The game log therefore cannot distinguish
failed image loading from successful loading followed by a rendering failure.
Closing and reopening the picker creates ordinary images at the same URL; it
does not invoke a deliberate recovery path.

The preload mock demonstrates an error-handling weakness, not the reported
Firefox failure. Preloading runs at startup, so it is not a sufficient explanation
for assets that render normally and disappear much later. An SVG already visible
elsewhere does not prove that a particular new image element painted correctly.
For decode promise semantics, see
[MDN](https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement/decode).

### Detailed bot diagnostics also run in production

`src/ui/hooks/useGameController.ts:57` emits `console.info` and `console.table`
for each searched decision, including root-action arrays. There is no development
or opt-in guard and session reset does not clear these messages. This creates
avoidable allocation/logging work and is a possible console-retention contributor;
its Firefox memory impact has not been measured. Do not interpret it as an
unbounded log leak or an explanation of 1 GB by itself.

## Suspects reduced by inspection

- The startup image cache is keyed by 41 fixed card URLs plus six suit URLs.
  Repeated games do not create additional cache keys in normal application use.
- All 41 used card PNGs are 242 by 376 pixels: about 14.23 MiB for one RGBA
  surface per card, excluding browser overhead, scaled surfaces, SVG rendering,
  and graphics caches. The source art is not unexpectedly enormous.
- Session reset clears action history, timeline, turn snapshots, animation
  queues, flights, timers, and deed-token layout memory.
- Reviewed DOM listeners, bot-thinking intervals, and card animation frames have
  cleanup. Animation timer IDs are removed when fired, and flights are cleared
  on sequence settlement/reset.
- Simulated rollout steps suppress game-log growth; worker world-state contexts
  replace the prior context rather than accumulate every decision.
- Bug-report download blob URLs are revoked. Game history stores compact score
  records rather than full board/image snapshots; the history view loads all
  records, which is a scaling concern but not a plausible multi-game image leak
  established here.

## Validation and next work

- Directly rendered the reported payment state and inspected its image markup.
- Injected a rejected decode promise to verify preload success caching.
- A mocked outer-worker probe selected Hard, Medium, then Hard again: two workers
  remained active, and explicit `close()` calls reduced that count to zero. This
  checks policy ownership/reuse, not real browser worker memory.
- Parsed suit SVG metadata and card PNG dimensions.
- 48 existing targeted tests passed across 11 files: payment/suit rendering,
  card assets/startup preload, animation timers/frames, deed layout, turn reset,
  and worker policy/pool behavior. These tests do not exercise Firefox painting.
- `yarn build` passed; the emitted JavaScript still contains the detailed bot
  console logging and references the emitted Sun asset.

## Implemented follow-up

- Switching bot profiles and unmounting the controller now closes the prior
  worker-backed policy. Starting a new game or resetting the current turn also
  closes the selected policy, which cancels its active search and tears down its
  nested search-worker pool.
- Full search diagnostics are off by default. Add `?botDiagnostics=1` to a
  browser URL when root-action diagnostics are needed; only then does the app
  build, clone, send, and log those detailed payloads.
- A rejected image `decode()` now fails startup preloading and exposes the
  existing retry UI instead of marking the URL as ready. Visible suit, card,
  discard, and die images now emit one bounded console error per failed asset
  context, making a late rendering failure identifiable.

Validation after the changes: focused UI/worker tests (34 assertions),
TypeScript typecheck, lint (one pre-existing HistoryModal warning only), and a
production build passed. A prolonged browser memory reproduction remains
unnecessary unless the symptom continues after these changes.
