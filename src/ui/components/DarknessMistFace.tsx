import { useEffect, useRef, useState, type CSSProperties } from 'react';

import { DARKNESS_STIR_DURATION_MS } from '../animations/timing';
import type { Suit } from '../../engine/types';

function deedTokenSignature(
  tokens: Partial<Record<Suit, number>> | undefined
): string {
  if (!tokens) {
    return '';
  }
  return Object.entries(tokens)
    .filter(([, count]) => (count ?? 0) > 0)
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([suit, count]) => `${suit}:${count}`)
    .join(',');
}

/**
 * The Darkness face: grey mist that drifts and settles for a few seconds when
 * the card is played, developed, or sold, then freezes.
 *
 * The look is a continuous procedural-noise field rather than particles: an
 * inline `feTurbulence` SVG (the same technique the app already uses for its
 * grain texture) is tiled into two oversized layers that drift and rotate at
 * different rates. Their interference makes the mist shape-shift without ever
 * popping in or out, and because the layers are full-bleed there are no
 * hotspots or bald patches. A settled base supplies even density, so the card
 * reads as mist at rest and never as blank.
 *
 * `stirSignal` is an app-supplied trigger token (for example the presenting
 * action that touched the card). Changing it restarts the stir. A card that
 * mounts inside a stir-enabled surface stirs once on arrival, so playing or
 * buying the deed animates without a separate signal.
 *
 * `stirResetSignal` is an explicit forced restart (a turn reset). It bypasses
 * the mid-run cooldown, so the mist stirs afresh even right after the action it
 * undoes.
 *
 * A single action fires several trigger changes (its presentation goes
 * pending -> settled, then deeds apply tokens and progress). Those all land
 * inside the stir window, so a `DARKNESS_STIR_DURATION_MS` cooldown swallows
 * them: the run is never restarted partway through. A genuinely new action later
 * still restarts it.
 */
export function DarknessMistFace({
  deedTokens,
  deedProgress,
  inDevelopment,
  animationsEnabled,
  stirEnabled,
  stirSignal,
  stirResetSignal,
}: {
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  inDevelopment?: boolean;
  animationsEnabled: boolean;
  stirEnabled: boolean;
  stirSignal?: string;
  stirResetSignal?: string;
}) {
  const stateSignature = `${inDevelopment ? 'deed' : 'placed'}:${
    deedProgress ?? ''
  }:${deedTokenSignature(deedTokens)}`;
  const [stirGeneration, setStirGeneration] = useState(0);
  const stirRef = useRef<{
    trigger: string | null;
    startedAt: number;
    resetSignal: string | undefined;
  }>({
    trigger: null,
    startedAt: Number.NEGATIVE_INFINITY,
    resetSignal: undefined,
  });

  useEffect(() => {
    if (!stirEnabled) {
      return;
    }
    // Fold the surface state and the app trigger into one token. Arriving in a
    // stir-enabled surface (mount) also counts as a stir.
    const trigger = `${stirSignal ?? 'mount'}|${stirResetSignal ?? ''}|${stateSignature}`;
    const state = stirRef.current;
    if (state.trigger === trigger) {
      return;
    }
    // A turn reset is an explicit, discrete user action, so it restarts the stir
    // even if the previous run has not finished. Ordinary mid-run trigger churn
    // (a presentation settling, deed tokens applying) is swallowed by the
    // cooldown so the animation is never snapped back to its first frame.
    const forced =
      stirResetSignal !== undefined && stirResetSignal !== state.resetSignal;
    state.trigger = trigger;
    state.resetSignal = stirResetSignal;
    const now =
      typeof performance === 'undefined' ? Date.now() : performance.now();
    if (!forced && now - state.startedAt < DARKNESS_STIR_DURATION_MS) {
      return;
    }
    state.startedAt = now;
    setStirGeneration((generation) => generation + 1);
  }, [stirEnabled, stirSignal, stirResetSignal, stateSignature]);

  const animated = stirEnabled && animationsEnabled;
  const fogStyle = {
    '--darkness-stir-duration': `${DARKNESS_STIR_DURATION_MS}ms`,
  } as CSSProperties;

  return (
    <div
      className="darkness-fog"
      data-face-effect="darkness"
      data-stir-generation={stirGeneration}
      style={fogStyle}
    >
      {/* The settled face: even grey density, always present, and the animation's
          end state once the drift comes to rest. */}
      <div className="darkness-fog__base" />
      <div
        key={stirGeneration}
        className={`darkness-fog__drift${animated ? ' is-animated' : ''}`}
      >
        <div className="darkness-fog__cloud darkness-fog__cloud-a" />
        <div className="darkness-fog__cloud darkness-fog__cloud-b" />
      </div>
    </div>
  );
}
