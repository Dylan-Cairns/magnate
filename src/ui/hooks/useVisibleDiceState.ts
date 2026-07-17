import { useState } from 'react';

import type { IncomeRollResult, Suit } from '../../engine/types';
import type { DiceVisualState } from '../runtime/types';

type RetainedDiceState = {
  gameKey: string;
  dice: DiceVisualState;
};

export function useVisibleDiceState({
  animationDice,
  gameKey,
  incomeRoll,
  taxSuit,
  terminal,
}: {
  animationDice: DiceVisualState | null;
  gameKey: string;
  incomeRoll: IncomeRollResult | undefined;
  taxSuit: Suit | undefined;
  terminal: boolean;
}): DiceVisualState | null {
  const [retainedDiceState, setRetainedDiceState] =
    useState<RetainedDiceState | null>(null);
  const settledDice = settledDiceVisualState(incomeRoll, taxSuit);
  const retainedDice =
    retainedDiceState?.gameKey === gameKey ? retainedDiceState.dice : null;

  if (
    settledDice &&
    !sameSettledDice(retainedDiceState, gameKey, settledDice)
  ) {
    setRetainedDiceState({ gameKey, dice: settledDice });
  } else if (
    !settledDice &&
    retainedDiceState !== null &&
    retainedDiceState.gameKey !== gameKey
  ) {
    setRetainedDiceState(null);
  }

  return animationDice ?? settledDice ?? (terminal ? retainedDice : null);
}

function sameSettledDice(
  retained: RetainedDiceState | null,
  gameKey: string,
  dice: DiceVisualState
): boolean {
  return (
    retained?.gameKey === gameKey &&
    retained.dice.incomeRoll.die1 === dice.incomeRoll.die1 &&
    retained.dice.incomeRoll.die2 === dice.incomeRoll.die2 &&
    retained.dice.incomeRoll.rollId === dice.incomeRoll.rollId &&
    retained.dice.taxSuit === dice.taxSuit
  );
}

function settledDiceVisualState(
  incomeRoll: IncomeRollResult | undefined,
  taxSuit: Suit | undefined
): DiceVisualState | null {
  if (!incomeRoll) {
    return null;
  }

  return {
    incomeRoll,
    taxSuit,
    incomePhase: 'settled',
    taxPhase: taxSuit ? 'settled' : 'dimmed',
  };
}
