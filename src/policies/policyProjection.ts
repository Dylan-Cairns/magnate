import {
  developmentCost,
  findProperty,
  mergeTokens,
  sumTokens,
} from '../engine/stateHelpers';
import type {
  DistrictStack,
  DistrictState,
  GameAction,
  GameState,
  PlayerId,
} from '../engine/types';

export type DistrictAction = Extract<
  GameAction,
  { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
>;

export function projectStateDistrictAction(
  action: GameAction,
  state: GameState,
  playerId: PlayerId
): GameState {
  if (!isDistrictAction(action)) {
    return state;
  }
  return {
    ...state,
    districts: state.districts.map((district) =>
      district.id === action.districtId
        ? projectDistrictAction(district, action, playerId)
        : district
    ),
  };
}

export function projectDistrictAction(
  district: DistrictState,
  action: DistrictAction,
  playerId: PlayerId
): DistrictState {
  const currentStack = district.stacks[playerId];
  const projectedStack = projectStackAction(currentStack, action);
  return {
    ...district,
    stacks: {
      ...district.stacks,
      [playerId]: projectedStack,
    },
  };
}

export function projectStackAction(
  stack: DistrictStack,
  action: DistrictAction
): DistrictStack {
  if (action.type === 'develop-outright') {
    return {
      ...stack,
      developed: [...stack.developed, action.cardId],
    };
  }
  if (action.type === 'buy-deed') {
    return {
      ...stack,
      deed: {
        cardId: action.cardId,
        progress: 0,
        tokens: {},
      },
    };
  }

  const deed = stack.deed;
  if (!deed) {
    return stack;
  }
  const card = findProperty(deed.cardId);
  const progress = deed.progress + sumTokens(action.tokens);
  const target = card ? developmentCost(card) : 0;
  if (target > 0 && progress >= target) {
    return {
      ...stack,
      developed: [...stack.developed, deed.cardId],
      deed: undefined,
    };
  }
  return {
    ...stack,
    deed: {
      ...deed,
      progress,
      tokens: mergeTokens(deed.tokens, action.tokens),
    },
  };
}

export function isDistrictAction(action: GameAction): action is DistrictAction {
  return (
    action.type === 'buy-deed' ||
    action.type === 'develop-deed' ||
    action.type === 'develop-outright'
  );
}

export function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

export function smoothstep(value: number): number {
  const x = clamp(value, 0, 1);
  return x * x * (3 - 2 * x);
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
