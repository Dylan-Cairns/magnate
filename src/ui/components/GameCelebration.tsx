import type { CSSProperties } from 'react';

import { CelebrationOutcome } from '../outcomes';
export { CelebrationOutcome } from '../outcomes';

type ChipParticle = {
  /** Horizontal start, in percent of the viewport width. */
  left: number;
  delayMs: number;
  durationMs: number;
  /** Horizontal drift across the fall, in rem. */
  driftRem: number;
  spinDeg: number;
  sizeRem: number;
  tone: number;
};

/*
  The layout is generated from a fixed xorshift seed rather than Math.random so
  the shower is identical on every render and test: it is pure decoration, and
  a stable table keeps snapshots and re-renders from reshuffling mid-animation.
*/
function createChips(count: number, seed: number): readonly ChipParticle[] {
  let state = seed >>> 0;
  const next = () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0xffffffff;
  };

  const chips: ChipParticle[] = [];
  for (let index = 0; index < count; index += 1) {
    chips.push({
      left: round(next() * 96 + 2, 2),
      delayMs: Math.round(next() * 900),
      durationMs: Math.round(2200 + next() * 1500),
      driftRem: round((next() * 2 - 1) * 6, 1),
      spinDeg: Math.round((next() * 2 - 1) * 540),
      sizeRem: round(0.5 + next() * 0.5, 2),
      tone: Math.floor(next() * 3),
    });
  }
  return chips;
}

const WIN_CHIPS = createChips(30, 0x9e3779b9);

export function GameCelebration({
  outcome,
  animationsEnabled,
}: {
  outcome: CelebrationOutcome | null;
  animationsEnabled: boolean;
}) {
  if (!animationsEnabled || outcome === null) {
    return null;
  }

  const chips = outcome === CelebrationOutcome.Win ? WIN_CHIPS : [];

  return (
    <div className={`game-celebration is-${outcome}`} aria-hidden="true">
      {outcome === CelebrationOutcome.Win ? (
        <span className="celebration-glow" />
      ) : null}

      {outcome === CelebrationOutcome.Draw ? (
        <>
          <span className="celebration-draw-ring" />
          <span className="celebration-draw-ring is-delayed" />
        </>
      ) : null}

      {chips.map((chip, index) => (
        <span
          key={index}
          className="celebration-chip"
          data-tone={chip.tone}
          style={chipStyle(chip)}
        />
      ))}
    </div>
  );
}

function chipStyle(chip: ChipParticle): CSSProperties {
  return {
    '--p-left': `${chip.left}%`,
    '--p-delay': `${chip.delayMs}ms`,
    '--p-duration': `${chip.durationMs}ms`,
    '--p-drift': `${chip.driftRem}rem`,
    '--p-spin': `${chip.spinDeg}deg`,
    '--p-size': `${chip.sizeRem}rem`,
  } as CSSProperties;
}

function round(value: number, places: number): number {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}
