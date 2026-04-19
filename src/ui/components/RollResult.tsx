import { useEffect, useState } from 'react';
import type { IncomeRollResult, Suit } from '../../engine/types';
import { D6Die } from './D6Die';
import { D10Die } from './D10Die';

// Match the d10 CSS transition duration so the d6 animates after the d10s settle
const D10_TRANSITION_MS = 1000;

export function RollResult({
  roll,
  taxSuit,
  gameKey,
}: {
  roll: IncomeRollResult | undefined;
  taxSuit: Suit | undefined;
  gameKey?: string;
}) {
  const [displayedSuit, setDisplayedSuit] = useState<Suit | undefined>(
    undefined
  );
  const [isPulsing, setIsPulsing] = useState(false);
  const [d6Dimmed, setD6Dimmed] = useState(false);

  useEffect(() => {
    if (taxSuit === undefined) {
      setDisplayedSuit(undefined);
      return;
    }
    // Chain: wait for d10 animations to finish before spinning the tax die
    const timer = setTimeout(
      () => setDisplayedSuit(taxSuit),
      D10_TRANSITION_MS
    );
    return () => clearTimeout(timer);
    // gameKey+rollId ensures this re-arms on every new roll and every new game
  }, [gameKey, roll?.rollId, taxSuit]);

  useEffect(() => {
    setIsPulsing(false);
    if (!roll) return;
    // When tax die is shown, wait for it to finish animating before pulsing anything
    const delay = taxSuit !== undefined ? D10_TRANSITION_MS * 2 : D10_TRANSITION_MS;
    const timer = setTimeout(() => setIsPulsing(true), delay);
    return () => clearTimeout(timer);
  }, [gameKey, roll?.rollId, taxSuit]);

  useEffect(() => {
    if (!roll) return;
    if (taxSuit !== undefined) {
      // Tax roll: un-dim d6 immediately so it brightens before rolling
      setD6Dimmed(false);
    } else {
      // Non-tax roll: dim d6 when the d10 animations settle
      const timer = setTimeout(() => setD6Dimmed(true), D10_TRANSITION_MS);
      return () => clearTimeout(timer);
    }
  }, [gameKey, roll?.rollId, taxSuit]);

  if (!roll) {
    return <p className="roll-value">-</p>;
  }

  const pulseDie1 = isPulsing && roll.die1 >= roll.die2;
  const pulseDie2 = isPulsing && roll.die2 >= roll.die1;

  return (
    <div className="roll-value" aria-label="Roll result">
      <D10Die result={roll.die1} rollKey={gameKey !== undefined ? `${gameKey}:${roll.rollId}` : roll.rollId} pulsing={pulseDie1} dimmed={isPulsing && !pulseDie1} />
      <D10Die result={roll.die2} rollKey={gameKey !== undefined ? `${gameKey}:${roll.rollId}` : roll.rollId} pulsing={pulseDie2} dimmed={isPulsing && !pulseDie2} />
      <D6Die suit={displayedSuit} pulsing={isPulsing && displayedSuit !== undefined} dimmed={d6Dimmed} />
    </div>
  );
}
