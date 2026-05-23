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
  holdPrevious = false,
  animationsEnabled = true,
}: {
  roll: IncomeRollResult | undefined;
  taxSuit: Suit | undefined;
  gameKey?: string;
  holdPrevious?: boolean;
  animationsEnabled?: boolean;
}) {
  const [heldRoll, setHeldRoll] = useState<IncomeRollResult | undefined>(roll);
  const [heldTaxSuit, setHeldTaxSuit] = useState<Suit | undefined>(taxSuit);
  const [displayedSuit, setDisplayedSuit] = useState<Suit | undefined>(
    undefined
  );
  const [displayedSuitRollKey, setDisplayedSuitRollKey] = useState<
    number | string | undefined
  >(undefined);
  const [isPulsing, setIsPulsing] = useState(false);
  const [d6Dimmed, setD6Dimmed] = useState(false);

  const [prevGameKey, setPrevGameKey] = useState(gameKey);
  const [prevRollId, setPrevRollId] = useState(roll?.rollId);
  const [prevTaxSuit, setPrevTaxSuit] = useState(taxSuit);
  const visibleRoll = roll ?? (holdPrevious ? heldRoll : undefined);
  const visibleTaxSuit = roll
    ? taxSuit
    : holdPrevious
      ? heldTaxSuit
      : undefined;
  const rollIdentity =
    visibleRoll === undefined
      ? undefined
      : (visibleRoll.rollId ?? `${visibleRoll.die1}-${visibleRoll.die2}`);
  const visibleRollKey =
    gameKey !== undefined ? `${gameKey}:${rollIdentity}` : rollIdentity;

  if (roll) {
    if (heldRoll !== roll) {
      setHeldRoll(roll);
    }
    if (heldTaxSuit !== taxSuit) {
      setHeldTaxSuit(taxSuit);
    }
  }

  if (
    gameKey !== prevGameKey ||
    roll?.rollId !== prevRollId ||
    taxSuit !== prevTaxSuit
  ) {
    setPrevGameKey(gameKey);
    setPrevRollId(roll?.rollId);
    setPrevTaxSuit(taxSuit);
    setIsPulsing(false);
    if (gameKey !== prevGameKey && !roll) {
      setHeldRoll(undefined);
      setHeldTaxSuit(undefined);
    }
    if (taxSuit === undefined) {
      setDisplayedSuit(undefined);
      setDisplayedSuitRollKey(undefined);
    }
    if (roll && taxSuit !== undefined) {
      setD6Dimmed(false);
    }
  }

  useEffect(() => {
    if (taxSuit === undefined) return;
    // Chain: wait for d10 animations to finish before spinning the tax die
    const delay = animationsEnabled ? D10_TRANSITION_MS : 0;
    const timer = setTimeout(() => {
      setDisplayedSuit(taxSuit);
      setDisplayedSuitRollKey(visibleRollKey);
    }, delay);
    return () => clearTimeout(timer);
  }, [animationsEnabled, gameKey, roll?.rollId, taxSuit, visibleRollKey]);

  useEffect(() => {
    if (roll?.rollId === undefined) return;
    // When tax die is shown, wait for it to finish animating before pulsing anything
    const delay = animationsEnabled
      ? taxSuit !== undefined ? D10_TRANSITION_MS * 2 : D10_TRANSITION_MS
      : 0;
    const timer = setTimeout(() => setIsPulsing(true), delay);
    return () => clearTimeout(timer);
  }, [animationsEnabled, gameKey, roll?.rollId, taxSuit]);

  useEffect(() => {
    if (roll?.rollId === undefined || taxSuit !== undefined) return;
    // Non-tax roll: dim d6 when the d10 animations settle
    const delay = animationsEnabled ? D10_TRANSITION_MS : 0;
    const timer = setTimeout(() => setD6Dimmed(true), delay);
    return () => clearTimeout(timer);
  }, [animationsEnabled, gameKey, roll?.rollId, taxSuit]);

  if (!visibleRoll) {
    return <p className="roll-value">-</p>;
  }

  const die1Wins = visibleRoll.die1 >= visibleRoll.die2;
  const die2Wins = visibleRoll.die2 > visibleRoll.die1;
  const isSettled = animationsEnabled ? isPulsing : roll !== undefined;
  const pulseDie1 = animationsEnabled && roll !== undefined && isPulsing && die1Wins;
  const pulseDie2 = animationsEnabled && roll !== undefined && isPulsing && die2Wins;
  const d6ShouldBeDimmed = (roll !== undefined || holdPrevious) && d6Dimmed;

  return (
    <div className="roll-value" aria-label="Roll result">
      <D10Die
        result={visibleRoll.die1}
        rollKey={visibleRollKey}
        pulsing={pulseDie1}
        dimmed={roll !== undefined && isSettled && !die1Wins}
        animationsEnabled={animationsEnabled}
      />
      <D10Die
        result={visibleRoll.die2}
        rollKey={visibleRollKey}
        pulsing={pulseDie2}
        dimmed={roll !== undefined && isSettled && !die2Wins}
        animationsEnabled={animationsEnabled}
      />
      <D6Die
        suit={roll === undefined ? visibleTaxSuit : displayedSuit}
        rollKey={displayedSuitRollKey}
        pulsing={animationsEnabled && roll !== undefined && isPulsing && displayedSuit !== undefined}
        dimmed={d6ShouldBeDimmed}
        animationsEnabled={animationsEnabled}
      />
    </div>
  );
}
