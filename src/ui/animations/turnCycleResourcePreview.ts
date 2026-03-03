import type {
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../../engine/types';
import {
  applySingleTaxLossToPreview,
  buildResourcePreviewByPlayer,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
} from './timing';
import type { TurnCycleAnimationPlan } from './turnCycleVisualPlan';

export type ResourcePreviewByPlayer = Partial<
  Record<PlayerId, ResourcePool>
> | null;

export type TurnCycleResourcePreviewEvent =
  | {
      kind: 'tax-loss';
      atMs: number;
      token: {
        playerId: PlayerId;
        suit: Suit;
      };
    }
  | {
      kind: 'tax-loss-summary';
      atMs: number;
      suit: Suit;
      lossesByPlayer: ReadonlyArray<{
        playerId: PlayerId;
        count: number;
      }>;
    }
  | {
      kind: 'replace';
      atMs: number;
      preview: Partial<Record<PlayerId, ResourcePool>>;
    };

export type TurnCycleResourcePreviewPlan = {
  initialPreview: Partial<Record<PlayerId, ResourcePool>>;
  events: ReadonlyArray<TurnCycleResourcePreviewEvent>;
};

export function buildTurnCycleResourcePreviewPlan(
  previousState: GameState,
  nextState: GameState,
  turnCyclePlan: TurnCycleAnimationPlan
): TurnCycleResourcePreviewPlan {
  const visualPlan = turnCyclePlan.visualPlan;
  const events: TurnCycleResourcePreviewEvent[] = [];

  if (
    visualPlan.taxFlightLaunchAtMs !== null &&
    visualPlan.taxFlightTokens.length > 0
  ) {
    for (const [index, token] of visualPlan.taxFlightTokens.entries()) {
      events.push({
        kind: 'tax-loss',
        atMs:
          visualPlan.taxFlightLaunchAtMs +
          index * TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
        token,
      });
    }
  } else if (
    visualPlan.taxResourcesApplyAtMs !== null &&
    visualPlan.taxSuit &&
    visualPlan.taxLossesByPlayer.length > 0
  ) {
    events.push({
      kind: 'tax-loss-summary',
      atMs: visualPlan.taxResourcesApplyAtMs,
      suit: visualPlan.taxSuit,
      lossesByPlayer: visualPlan.taxLossesByPlayer,
    });
  }

  events.push({
    kind: 'replace',
    atMs: turnCyclePlan.incomeAnimationEndMs,
    preview: buildResourcePreviewByPlayer(nextState),
  });

  return {
    initialPreview: buildResourcePreviewByPlayer(previousState),
    events,
  };
}

export function applyTurnCycleResourcePreviewEvent(
  preview: ResourcePreviewByPlayer,
  event: TurnCycleResourcePreviewEvent
): ResourcePreviewByPlayer {
  if (event.kind === 'replace') {
    return event.preview;
  }
  if (event.kind === 'tax-loss') {
    return applySingleTaxLossToPreview(preview, event.token);
  }

  let nextPreview = preview;
  for (const loss of event.lossesByPlayer) {
    for (let count = 0; count < loss.count; count += 1) {
      nextPreview = applySingleTaxLossToPreview(nextPreview, {
        playerId: loss.playerId,
        suit: event.suit,
      });
    }
  }
  return nextPreview;
}
