import type { CardId } from '../cards';
import type {
  DeckState,
  DeedState,
  DistrictState,
  DistrictStack,
  GameAction,
  GamePhase,
  GameState,
  IncomeChoice,
  IncomeRollResult,
  PlayerId,
  PlayerState,
  ResourcePool,
  Suit,
} from '../types';
import { legalActions } from '../actionBuilders';

export const PLAYER_A = 'PlayerA';
export const PLAYER_B = 'PlayerB';

export function makeResources(
  overrides: Partial<Record<Suit, number>> = {}
): ResourcePool {
  return {
    Moons: overrides.Moons ?? 0,
    Suns: overrides.Suns ?? 0,
    Waves: overrides.Waves ?? 0,
    Leaves: overrides.Leaves ?? 0,
    Wyrms: overrides.Wyrms ?? 0,
    Knots: overrides.Knots ?? 0,
  };
}

export function makePlayer(
  id: PlayerId,
  overrides: Partial<Omit<PlayerState, 'id'>> = {}
): PlayerState {
  return {
    id,
    hand: overrides.hand ? [...overrides.hand] : ['6'],
    crowns: overrides.crowns ? [...overrides.crowns] : ['30', '31', '32'],
    resources: overrides.resources
      ? makeResources(overrides.resources)
      : makeResources(),
  };
}

export function makeDistrict(
  id: string,
  markerSuitMask: readonly Suit[],
  stacks?: Partial<Record<PlayerId, DistrictStack>>
): DistrictState {
  return {
    id,
    markerSuitMask,
    stacks: {
      [PLAYER_A]: stacks?.[PLAYER_A] ?? { developed: [] },
      [PLAYER_B]: stacks?.[PLAYER_B] ?? { developed: [] },
    },
  };
}

export function makeDefaultDistricts(): DistrictState[] {
  return [
    makeDistrict('D1', ['Moons']),
    makeDistrict('D2', ['Suns']),
    makeDistrict('D3', ['Waves']),
    makeDistrict('D4', ['Leaves']),
    makeDistrict('D5', []),
  ];
}

export interface GameStateOverrides {
  phase?: GamePhase;
  players?: readonly [PlayerState, PlayerState];
  districts?: DistrictState[];
  deck?: DeckState;
  activePlayerIndex?: number;
  seed?: string;
  rngCursor?: number;
  turn?: number;
  cardPlayedThisTurn?: boolean;
  exhaustionStage?: 0 | 1 | 2;
  finalTurnsRemaining?: number;
  lastIncomeRoll?: IncomeRollResult;
  pendingIncomeChoices?: readonly IncomeChoice[];
  incomeChoiceReturnPlayerIndex?: number;
}

export function makeGameState(overrides: GameStateOverrides = {}): GameState {
  const players =
    overrides.players ??
    ([
      makePlayer(PLAYER_A),
      makePlayer(PLAYER_B),
    ] as const satisfies readonly [PlayerState, PlayerState]);

  return {
    schemaVersion: 1,
    seed: overrides.seed ?? 'test-seed',
    rngCursor: overrides.rngCursor ?? 0,
    deck: overrides.deck ?? { draw: ['6', '7', '8'], discard: [], reshuffles: 0 },
    players,
    activePlayerIndex: overrides.activePlayerIndex ?? 0,
    turn: overrides.turn ?? 1,
    phase: overrides.phase ?? 'ActionWindow',
    districts: overrides.districts ?? makeDefaultDistricts(),
    cardPlayedThisTurn: overrides.cardPlayedThisTurn ?? false,
    exhaustionStage: overrides.exhaustionStage ?? 0,
    finalTurnsRemaining: overrides.finalTurnsRemaining,
    lastIncomeRoll: overrides.lastIncomeRoll,
    pendingIncomeChoices: overrides.pendingIncomeChoices,
    incomeChoiceReturnPlayerIndex: overrides.incomeChoiceReturnPlayerIndex,
    log: [],
  };
}

export function withDeed(
  state: GameState,
  districtId: string,
  playerId: PlayerId,
  deed: DeedState
): GameState {
  return {
    ...state,
    districts: state.districts.map((district) => {
      if (district.id !== districtId) {
        return district;
      }
      return {
        ...district,
        stacks: {
          ...district.stacks,
          [playerId]: {
            ...district.stacks[playerId],
            deed,
          },
        },
      };
    }),
  };
}

export function findLegalActionByType<TType extends GameAction['type']>(
  state: GameState,
  type: TType
): Extract<GameAction, { type: TType }> {
  const action = legalActions(state).find(
    (item): item is Extract<GameAction, { type: TType }> => item.type === type
  );
  if (!action) {
    throw new Error(`No legal action found for type "${type}".`);
  }
  return action;
}

export function asCardId(value: string): CardId {
  return value as CardId;
}
