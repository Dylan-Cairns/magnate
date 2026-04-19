import type { Suit } from '../../engine/types';
import { D6Die } from './D6Die';
import { D10Die } from './D10Die';

export function RollResult({
  roll,
  taxSuit,
}: {
  roll: { die1: number; die2: number } | undefined;
  taxSuit: Suit | undefined;
}) {
  if (!roll) {
    return <p className="roll-value">-</p>;
  }

  return (
    <div className="roll-value" aria-label="Roll result">
      <D10Die result={roll.die1} />
      <D10Die result={roll.die2} />
      <D6Die suit={taxSuit} />
    </div>
  );
}
