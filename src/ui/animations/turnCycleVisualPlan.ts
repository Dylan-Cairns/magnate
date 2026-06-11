import type { CardId } from '../../engine/cards';
import type { GameAction, GameState, PlayerId, Suit } from '../../engine/types';
import type { TurnCycleEvents, TurnCycleIncomeToken } from '../turnCycleEvents';
import { deriveTurnCycleEvents } from '../turnCycleEvents';
import {
  ACTION_FLIGHT_COMMIT_BUFFER_MS,
  DICE_D10_SETTLE_MS,
  DICE_TAX_SETTLE_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_INCOME_PRE_FLIGHT_MS,
  TURN_CYCLE_POST_INCOME_HOLD_MS,
  TURN_CYCLE_STAGE_GAP_MS,
  TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TAX_PRE_FLIGHT_MS,
} from './timing';

export type TurnCycleVisualPlan = {
  taxPulseStartAtMs: number | null;
  taxPulseEndAtMs: number | null;
  taxFlightLaunchAtMs: number | null;
  taxResourcesApplyAtMs: number | null;
  taxPulseTargets: ReadonlyArray<{
    playerId: PlayerId;
    suit: Suit;
  }>;
  taxFlightTokens: ReadonlyArray<{
    playerId: PlayerId;
    suit: Suit;
  }>;
  taxLossesByPlayer: ReadonlyArray<{
    playerId: PlayerId;
    count: number;
  }>;
  incomeFlightLaunchAtMs: number;
  incomeFlightTokens: ReadonlyArray<TurnCycleIncomeToken>;
  incomeHighlightStartAtMs: number;
  incomeHighlightEndAtMs: number;
  hideAllAtMs: number;
  taxSuit: Suit | null;
  highlightCardIds: ReadonlyArray<CardId>;
  highlightCrowns: ReadonlyArray<{
    playerId: PlayerId;
    suit: Suit;
  }>;
};

export type TurnCycleAnimationPlan = {
  visualPlan: TurnCycleVisualPlan;
  totalDurationMs: number;
  incomeAnimationEndMs: number;
};

export function buildTurnCycleVisualPlan(
  cycle: TurnCycleEvents,
): TurnCycleAnimationPlan {
  const diceSettleMs = cycle.tax ? DICE_TAX_SETTLE_MS : DICE_D10_SETTLE_MS;
  let cursorMs = diceSettleMs;

  let taxPulseStartAtMs: number | null = null;
  let taxPulseEndAtMs: number | null = null;
  let taxFlightLaunchAtMs: number | null = null;
  let taxResourcesApplyAtMs: number | null = null;
  const taxPulseTargets: Array<{ playerId: PlayerId; suit: Suit }> = [];
  const taxFlightTokens: Array<{ playerId: PlayerId; suit: Suit }> = [];
  const taxLossesByPlayer: Array<{ playerId: PlayerId; count: number }> = [];
  let taxSuit: Suit | null = null;
  if (cycle.tax) {
    taxSuit = cycle.tax.suit;
    const taxedPlayers = cycle.tax.lossesByPlayer.filter(
      (entry) => entry.count > 0
    );
    taxLossesByPlayer.push(...taxedPlayers);
    if (taxedPlayers.length > 0) {
      taxPulseStartAtMs = cursorMs;
      for (const taxedPlayer of taxedPlayers) {
        taxPulseTargets.push({
          playerId: taxedPlayer.playerId,
          suit: cycle.tax.suit,
        });
        for (let count = 0; count < taxedPlayer.count; count += 1) {
          taxFlightTokens.push({
            playerId: taxedPlayer.playerId,
            suit: cycle.tax.suit,
          });
        }
      }
    }
    const taxFlightStartMs = cursorMs + TURN_CYCLE_TAX_PRE_FLIGHT_MS;
    taxFlightLaunchAtMs =
      taxFlightTokens.length > 0 ? taxFlightStartMs : null;
    const taxAnimationEndMs =
      taxFlightTokens.length > 0
        ? taxFlightStartMs +
          (taxFlightTokens.length - 1) * TURN_CYCLE_TAX_FLIGHT_STAGGER_MS +
          TURN_CYCLE_TAX_FLIGHT_DURATION_MS
        : taxFlightStartMs;
    taxResourcesApplyAtMs =
      taxFlightTokens.length > 0 ? taxAnimationEndMs : null;
    if (taxedPlayers.length > 0) {
      taxPulseEndAtMs = taxAnimationEndMs;
    }
    cursorMs = taxAnimationEndMs + TURN_CYCLE_STAGE_GAP_MS;
  }

  const incomeAnimationStartMs = cursorMs + TURN_CYCLE_INCOME_PRE_FLIGHT_MS;
  const incomeAnimationEndMs =
    cycle.incomeTokens.length > 0
      ? incomeAnimationStartMs +
        (cycle.incomeTokens.length - 1) * TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS +
        TURN_CYCLE_INCOME_FLIGHT_DURATION_MS
      : incomeAnimationStartMs;

  const highlightCardIds = [
    ...new Set(cycle.incomeHighlights.map((entry) => entry.cardId)),
  ];
  const highlightCrowns: Array<{ playerId: PlayerId; suit: Suit }> = [];
  const highlightCrownKeys = new Set<string>();
  for (const token of cycle.incomeTokens) {
    if (token.source.kind !== 'crown') {
      continue;
    }
    const key = `${token.playerId}:${token.suit}`;
    if (highlightCrownKeys.has(key)) {
      continue;
    }
    highlightCrownKeys.add(key);
    highlightCrowns.push({
      playerId: token.playerId,
      suit: token.suit,
    });
  }
  const incomeHighlightStartAtMs = cursorMs;
  const hasIncomeHighlightTargets =
    highlightCardIds.length > 0 || highlightCrowns.length > 0;
  const minimumHighlightDurationMs = hasIncomeHighlightTargets
    ? TURN_CYCLE_INCOME_PRE_FLIGHT_MS + TURN_CYCLE_INCOME_FLIGHT_DURATION_MS
    : 0;
  const incomeHighlightEndAtMs = Math.max(
    incomeHighlightStartAtMs + minimumHighlightDurationMs,
    incomeAnimationEndMs
  );

  const hideAllAtMs =
    Math.max(incomeHighlightEndAtMs, incomeAnimationEndMs) +
    TURN_CYCLE_POST_INCOME_HOLD_MS;

  return {
    visualPlan: {
      taxPulseStartAtMs,
      taxPulseEndAtMs,
      taxFlightLaunchAtMs,
      taxResourcesApplyAtMs,
      taxPulseTargets,
      taxFlightTokens,
      taxLossesByPlayer,
      incomeFlightLaunchAtMs: incomeAnimationStartMs,
      incomeFlightTokens: cycle.incomeTokens,
      incomeHighlightStartAtMs,
      incomeHighlightEndAtMs,
      hideAllAtMs,
      taxSuit,
      highlightCardIds,
      highlightCrowns,
    },
    totalDurationMs: hideAllAtMs + ACTION_FLIGHT_COMMIT_BUFFER_MS,
    incomeAnimationEndMs,
  };
}

export function collectTurnCycleAnimationPlan(
  previousState: GameState,
  nextState: GameState,
  action: GameAction,
): TurnCycleAnimationPlan | null {
  const cycle = deriveTurnCycleEvents(previousState, nextState, action);
  return cycle ? buildTurnCycleVisualPlan(cycle) : null;
}
