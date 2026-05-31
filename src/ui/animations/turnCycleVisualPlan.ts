import type { CardId } from '../../engine/cards';
import type { GameAction, GameState, PlayerId, Suit } from '../../engine/types';
import type { TurnCycleEvents, TurnCycleIncomeToken } from '../turnCycleEvents';
import { deriveTurnCycleEvents } from '../turnCycleEvents';
import {
  ACTION_FLIGHT_COMMIT_BUFFER_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_INCOME_PRE_FLIGHT_MS,
  TURN_CYCLE_POST_INCOME_HOLD_MS,
  TURN_CYCLE_STAGE_GAP_MS,
  TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TEXT_ONLY_MS,
} from './timing';

export type TurnCycleVisualPlan = {
  taxLabelAtMs: number | null;
  taxLabelHideAtMs: number | null;
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
  incomeLabelAtMs: number;
  incomeLabelHideAtMs: number;
  incomeFlightLaunchAtMs: number;
  incomeFlightTokens: ReadonlyArray<TurnCycleIncomeToken>;
  incomeHighlightStartAtMs: number;
  incomeHighlightEndAtMs: number;
  hideAllAtMs: number;
  taxSuit: Suit | null;
  incomeRank: number;
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
  baseDelayMs: number
): TurnCycleAnimationPlan {
  let cursorMs = baseDelayMs;

  let taxLabelAtMs: number | null = null;
  let taxLabelHideAtMs: number | null = null;
  let taxPulseStartAtMs: number | null = null;
  let taxPulseEndAtMs: number | null = null;
  let taxFlightLaunchAtMs: number | null = null;
  let taxResourcesApplyAtMs: number | null = null;
  const taxPulseTargets: Array<{ playerId: PlayerId; suit: Suit }> = [];
  const taxFlightTokens: Array<{ playerId: PlayerId; suit: Suit }> = [];
  const taxLossesByPlayer: Array<{ playerId: PlayerId; count: number }> = [];
  let taxSuit: Suit | null = null;
  if (cycle.tax) {
    const taxLabelAtTime = cursorMs;
    const taxLabelHideAtTime = taxLabelAtTime + TURN_CYCLE_TEXT_ONLY_MS;
    taxLabelAtMs = taxLabelAtTime;
    taxLabelHideAtMs = taxLabelHideAtTime;
    taxSuit = cycle.tax.suit;
    const taxedPlayers = cycle.tax.lossesByPlayer.filter(
      (entry) => entry.count > 0
    );
    taxLossesByPlayer.push(...taxedPlayers);
    if (taxedPlayers.length > 0) {
      taxPulseStartAtMs = taxLabelAtTime;
      taxPulseEndAtMs = taxLabelHideAtTime;
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
    const taxAnimationStartMs = taxLabelHideAtTime + 120;
    taxFlightLaunchAtMs =
      taxFlightTokens.length > 0 ? taxAnimationStartMs : null;
    const taxAnimationEndMs =
      taxFlightTokens.length > 0
        ? taxAnimationStartMs +
          (taxFlightTokens.length - 1) * TURN_CYCLE_TAX_FLIGHT_STAGGER_MS +
          TURN_CYCLE_TAX_FLIGHT_DURATION_MS
        : taxAnimationStartMs;
    taxResourcesApplyAtMs =
      taxFlightTokens.length > 0 ? taxAnimationEndMs : null;
    cursorMs = taxAnimationEndMs + TURN_CYCLE_STAGE_GAP_MS;
  }

  const incomeLabelAtMs = cursorMs;
  const incomeLabelHideAtMs = incomeLabelAtMs + TURN_CYCLE_INCOME_PRE_FLIGHT_MS;
  const incomeAnimationStartMs = incomeLabelHideAtMs;
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
  const incomeHighlightStartAtMs = incomeLabelAtMs;
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
      taxLabelAtMs,
      taxLabelHideAtMs,
      taxPulseStartAtMs,
      taxPulseEndAtMs,
      taxFlightLaunchAtMs,
      taxResourcesApplyAtMs,
      taxPulseTargets,
      taxFlightTokens,
      taxLossesByPlayer,
      incomeLabelAtMs,
      incomeLabelHideAtMs,
      incomeFlightLaunchAtMs: incomeAnimationStartMs,
      incomeFlightTokens: cycle.incomeTokens,
      incomeHighlightStartAtMs,
      incomeHighlightEndAtMs,
      hideAllAtMs,
      taxSuit,
      incomeRank: cycle.incomeRank,
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
  baseDelayMs: number
): TurnCycleAnimationPlan | null {
  const cycle = deriveTurnCycleEvents(previousState, nextState, action);
  return cycle ? buildTurnCycleVisualPlan(cycle, baseDelayMs) : null;
}
