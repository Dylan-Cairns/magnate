import type {
  GameAction,
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../../engine/types';
import type { CardFlight, ResourceFlight } from './types';

export const RESOURCE_FLIGHT_DURATION_MS = 280;
export const RESOURCE_FLIGHT_STAGGER_MS = 75;
export const CARD_FLIGHT_DURATION_MS = 280;
export const CARD_DRAW_FLIGHT_DELAY_MS = CARD_FLIGHT_DURATION_MS;
export const ACTION_FLIGHT_COMMIT_BUFFER_MS = 20;

// Dice animation settle durations — must match the CSS transition durations.
export const DICE_D10_SETTLE_MS = 1000;
export const DICE_D6_SETTLE_MS = 1000;
export const DICE_TAX_SETTLE_MS = DICE_D10_SETTLE_MS + DICE_D6_SETTLE_MS;

// Turn-cycle pre-flight timing knobs.
export const TURN_CYCLE_TAX_PRE_FLIGHT_MS = 550;
export const TURN_CYCLE_INCOME_PRE_FLIGHT_MS = 400;
export const TURN_CYCLE_STAGE_GAP_MS = 220;
export const TURN_CYCLE_TAX_FLIGHT_DURATION_MS = 900;
export const TURN_CYCLE_TAX_FLIGHT_STAGGER_MS = 500;
export const TURN_CYCLE_INCOME_FLIGHT_DURATION_MS = 560;
export const TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS = 95;
export const TURN_CYCLE_POST_INCOME_HOLD_MS = 220;
export const TERMINAL_CLEANUP_FLIGHT_DURATION_MS = 900;
export const TERMINAL_CLEANUP_VERTICAL_TRAVEL_PX = 220;

export function resourceFlightSettleMs(
  flights: readonly ResourceFlight[]
): number {
  if (flights.length === 0) {
    return 0;
  }
  const latestEndMs = Math.max(
    ...flights.map(
      (flight) =>
        flight.delayMs + (flight.durationMs ?? RESOURCE_FLIGHT_DURATION_MS)
    )
  );
  return latestEndMs + ACTION_FLIGHT_COMMIT_BUFFER_MS;
}

export function cardFlightSettleMs(flights: readonly CardFlight[]): number {
  if (flights.length === 0) {
    return 0;
  }
  const latestEndMs = Math.max(
    ...flights.map(
      (flight) =>
        flight.delayMs + (flight.durationMs ?? CARD_FLIGHT_DURATION_MS)
    )
  );
  return latestEndMs + ACTION_FLIGHT_COMMIT_BUFFER_MS;
}

export function shouldCommitBeforeAnimationSettle(action: GameAction): boolean {
  return action.type === 'sell-card' || action.type === 'end-turn';
}

export function shouldAllowHumanActionsDuringAnimationSettle(
  action: GameAction
): boolean {
  return action.type === 'end-turn' || action.type === 'choose-income-suit';
}

export function buildResourcePreviewByPlayer(
  state: GameState
): Partial<Record<PlayerId, ResourcePool>> {
  const preview: Partial<Record<PlayerId, ResourcePool>> = {};
  for (const player of state.players) {
    preview[player.id] = { ...player.resources };
  }
  return preview;
}

export function applySingleTaxLossToPreview(
  preview: Partial<Record<PlayerId, ResourcePool>> | null,
  token: { playerId: PlayerId; suit: Suit }
): Partial<Record<PlayerId, ResourcePool>> | null {
  if (!preview) {
    return preview;
  }

  const resources = preview[token.playerId];
  if (!resources) {
    return preview;
  }

  const count = resources[token.suit];
  if (count <= 0) {
    return preview;
  }

  return {
    ...preview,
    [token.playerId]: {
      ...resources,
      [token.suit]: Math.max(0, count - 1),
    },
  };
}
