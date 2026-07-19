import type { DiceVisualState } from '../runtime/types';
import { D6Die } from './D6Die';
import { D10Die } from './D10Die';

export function RollResult({
  dice,
  gameKey,
  animationsEnabled = true,
}: {
  dice: DiceVisualState | null;
  gameKey?: string;
  animationsEnabled?: boolean;
}) {
  if (!dice) {
    return <p className="roll-value">-</p>;
  }

  const rollIdentity =
    dice.incomeRoll.rollId ?? `${dice.incomeRoll.die1}-${dice.incomeRoll.die2}`;
  const visibleRollKey =
    gameKey !== undefined ? `${gameKey}:${rollIdentity}` : rollIdentity;
  const die1Wins = dice.incomeRoll.die1 >= dice.incomeRoll.die2;
  const die2Wins = dice.incomeRoll.die2 > dice.incomeRoll.die1;
  const incomeSettled = dice.incomePhase === 'settled';
  const taxSettled = dice.taxPhase === 'settled';
  const taxDimmed = dice.taxPhase === 'hidden' || dice.taxPhase === 'dimmed';
  const taxSuit =
    taxDimmed ? undefined : dice.taxSuit;

  return (
    <div className="roll-value" aria-label="Roll result">
      <D10Die
        result={dice.incomeRoll.die1}
        rollKey={visibleRollKey}
        glowing={incomeSettled && die1Wins}
        dimmed={incomeSettled && !die1Wins}
        animationsEnabled={animationsEnabled}
      />
      <D10Die
        result={dice.incomeRoll.die2}
        rollKey={visibleRollKey}
        glowing={incomeSettled && die2Wins}
        dimmed={incomeSettled && !die2Wins}
        animationsEnabled={animationsEnabled}
      />
      <D6Die
        suit={taxSuit}
        rollKey={visibleRollKey}
        glowing={taxSettled && taxSuit !== undefined}
        dimmed={taxDimmed}
        animationsEnabled={animationsEnabled}
      />
    </div>
  );
}
