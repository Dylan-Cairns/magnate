import type { CardId } from '../cards';
import {
  type DeckState,
  type DeedState,
  type DistrictState,
  type DistrictStack,
  type GameAction,
  GamePhase,
  type GameState,
  type IncomeChoice,
  type IncomeRollResult,
  PlayerId,
  type PlayerState,
  type ResourcePool,
  Ruleset,
  type SubmittedIncomeChoice,
  Suit,
} from '../types';
import { legalActions } from '../actionBuilders';

export const PLAYER_A = PlayerId.PlayerA;
export const PLAYER_B = PlayerId.PlayerB;

export function makeResources(
  overrides: Partial<Record<Suit, number>> = {}
): ResourcePool {
  return {
    [Suit.Moons]: overrides[Suit.Moons] ?? 0,
    [Suit.Suns]: overrides[Suit.Suns] ?? 0,
    [Suit.Waves]: overrides[Suit.Waves] ?? 0,
    [Suit.Leaves]: overrides[Suit.Leaves] ?? 0,
    [Suit.Wyrms]: overrides[Suit.Wyrms] ?? 0,
    [Suit.Knots]: overrides[Suit.Knots] ?? 0,
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
    makeDistrict('D1', [Suit.Moons]),
    makeDistrict('D2', [Suit.Suns]),
    makeDistrict('D3', [Suit.Waves]),
    makeDistrict('D4', [Suit.Leaves]),
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
  ruleset?: Ruleset;
  finalTurnsRemaining?: number;
  lastIncomeRoll?: IncomeRollResult;
  pendingIncomeChoices?: readonly IncomeChoice[];
  submittedIncomeChoices?: readonly SubmittedIncomeChoice[];
  incomeChoiceReturnPlayerId?: PlayerId;
}

export function makeGameState(overrides: GameStateOverrides = {}): GameState {
  const players =
    overrides.players ??
    ([makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const satisfies readonly [
      PlayerState,
      PlayerState,
    ]);

  return {
    schemaVersion: 1,
    seed: overrides.seed ?? 'test-seed',
    rngCursor: overrides.rngCursor ?? 0,
    ruleset: overrides.ruleset ?? Ruleset.Standard,
    deck: overrides.deck ?? {
      draw: ['6', '7', '8'],
      discard: [],
      reshuffles: 0,
    },
    players,
    activePlayerIndex: overrides.activePlayerIndex ?? 0,
    turn: overrides.turn ?? 1,
    phase: overrides.phase ?? GamePhase.ActionWindow,
    districts: overrides.districts ?? makeDefaultDistricts(),
    cardPlayedThisTurn: overrides.cardPlayedThisTurn ?? false,
    finalTurnsRemaining: overrides.finalTurnsRemaining,
    lastIncomeRoll: overrides.lastIncomeRoll,
    pendingIncomeChoices: overrides.pendingIncomeChoices,
    submittedIncomeChoices: overrides.submittedIncomeChoices,
    incomeChoiceReturnPlayerId: overrides.incomeChoiceReturnPlayerId,
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
