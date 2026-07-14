import { actionStableKey } from '../engine/actionSurface';
import { PROPERTY_CARDS, type CardId } from '../engine/cards';
import { legalActionsForDecisionPlayer } from '../engine/decisionActor';
import { SUITS } from '../engine/stateHelpers';
import type {
  DeedState,
  DistrictState,
  GameAction,
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../engine/types';

export const STRATEGIC_POSITION_CATALOG_VERSION = 0 as const;

export type StrategicPositionThemeV0 =
  | 'global-district-portfolio'
  | 'conditional-tiebreak'
  | 'placement-optionality'
  | 'deed-feasibility'
  | 'game-clock'
  | 'ace-scoring';

export interface StrategicFocusActionV0 {
  readonly id: string;
  readonly label: string;
  readonly actionKey: string;
}

export interface StrategicPreferenceV0 {
  readonly preferredFocusActionId: string;
  readonly overFocusActionIds: readonly string[];
  readonly rationale: string;
}

export interface StrategicPositionV0 {
  readonly catalogVersion: typeof STRATEGIC_POSITION_CATALOG_VERSION;
  readonly id: string;
  readonly title: string;
  readonly theme: StrategicPositionThemeV0;
  readonly perspectivePlayerId: PlayerId;
  readonly thesis: string;
  readonly expectedFacts: readonly string[];
  readonly pairId: string | null;
  readonly state: GameState;
  readonly focusActions: readonly StrategicFocusActionV0[];
  readonly expectedPreference: StrategicPreferenceV0 | null;
}

interface FocusActionRecipe {
  readonly id: string;
  readonly label: string;
  readonly selector: ActionSelector;
}

interface StrategicPositionRecipe {
  readonly id: string;
  readonly title: string;
  readonly theme: StrategicPositionThemeV0;
  readonly thesis: string;
  readonly expectedFacts: readonly string[];
  readonly pairId?: string;
  readonly state: GameState;
  readonly focusActions: readonly FocusActionRecipe[];
  readonly expectedPreference?: StrategicPreferenceV0;
}

interface ActionSelector {
  readonly type: GameAction['type'];
  readonly cardId?: CardId;
  readonly districtId?: string;
}

interface PositionStateRecipe {
  readonly id: string;
  readonly turn: number;
  readonly districts: readonly DistrictState[];
  readonly ownHand: readonly CardId[];
  readonly selfResources?: Partial<Record<Suit, number>>;
  readonly opponentResources?: Partial<Record<Suit, number>>;
  readonly reshuffles?: 0 | 1 | 2;
  readonly finalTurnsRemaining?: number;
  readonly drawCount?: number;
  readonly unknownCardIds?: readonly CardId[];
  readonly opponentHandCount?: number;
}

export function createStrategicPositionCatalogV0(): StrategicPositionV0[] {
  return strategicPositionRecipes().map(resolvePositionRecipe);
}

function strategicPositionRecipes(): StrategicPositionRecipe[] {
  const deedForkDistricts = [
    district('D0', ['Waves', 'Leaves', 'Wyrms'], ['14']),
    district('D1', ['Moons', 'Suns', 'Leaves'], ['15']),
    district('D2', ['Suns', 'Waves', 'Knots'], [], ['17']),
    district('D3', ['Moons', 'Wyrms', 'Knots'], [], ['19']),
    district('D4', [], [], ['25'], {
      cardId: '29',
      progress: 8,
      tokens: { Moons: 8 },
    }),
  ];

  return [
    {
      id: 'minimum-winning-coalition',
      title: 'Pivotal fifth district over fortress reinforcement',
      theme: 'global-district-portfolio',
      thesis:
        'When the board is 2-2, winning the tied fifth district is worth more than adding the same card to an already secure district.',
      expectedFacts: [
        'The pivotal placement changes the live district margin from 0 to +1.',
        "The fortress placement does not change either player's district count.",
      ],
      state: positionState({
        id: 'minimum-winning-coalition',
        turn: 34,
        ownHand: ['6', '7', '8'],
        selfResources: { Moons: 1, Knots: 1 },
        districts: [
          district('D0', ['Waves', 'Leaves', 'Wyrms'], ['26', '16'], ['1']),
          district('D1', ['Moons', 'Suns', 'Leaves'], ['25']),
          district('D2', ['Suns', 'Waves', 'Knots'], [], ['14']),
          district('D3', ['Moons', 'Wyrms', 'Knots'], [], ['17']),
          district('D4', []),
        ],
      }),
      focusActions: [
        focus('pivotal', 'Develop The Author in D4', {
          type: 'develop-outright',
          cardId: '6',
          districtId: 'D4',
        }),
        focus('fortress', 'Develop The Author in D1', {
          type: 'develop-outright',
          cardId: '6',
          districtId: 'D1',
        }),
      ],
      expectedPreference: {
        preferredFocusActionId: 'pivotal',
        overFocusActionIds: ['fortress'],
        rationale:
          'The pivotal action changes the provisional match result; the fortress action only widens an existing win.',
      },
    },
    {
      id: 'tie-denial-restores-match',
      title: 'Denying an opponent district restores a tied match',
      theme: 'global-district-portfolio',
      thesis:
        'Turning an opponent-controlled district into a tie can be match-decisive even though it does not create a district point for the mover.',
      expectedFacts: [
        'The denial action removes one opponent district point.',
        'The resulting 2-2 district score is provisionally won by developed-rank total.',
      ],
      state: positionState({
        id: 'tie-denial-restores-match',
        turn: 36,
        ownHand: ['13', '7', '8'],
        selfResources: { Moons: 2, Suns: 2 },
        districts: [
          district('D0', ['Waves', 'Leaves', 'Wyrms'], ['26']),
          district('D1', ['Moons', 'Suns', 'Leaves'], ['25']),
          district('D2', ['Suns', 'Waves', 'Knots'], ['4'], ['14']),
          district('D3', ['Moons', 'Wyrms', 'Knots'], ['5'], ['17']),
          district('D4', [], [], ['12']),
        ],
      }),
      focusActions: [
        focus('deny', 'Develop The Mountain in D4', {
          type: 'develop-outright',
          cardId: '13',
          districtId: 'D4',
        }),
        focus('fortress', 'Develop The Mountain in D1', {
          type: 'develop-outright',
          cardId: '13',
          districtId: 'D1',
        }),
      ],
      expectedPreference: {
        preferredFocusActionId: 'deny',
        overFocusActionIds: ['fortress'],
        rationale:
          'Loss-to-tie denies the opponent one district and activates a favorable rank tiebreak.',
      },
    },
    {
      id: 'rank-tiebreak-conversion',
      title: 'Rank conversion in a tied district match',
      theme: 'conditional-tiebreak',
      thesis:
        'With the district score locked at 2-2, developing a rank-2 card in an already won district can convert the rank tiebreak while selling cannot.',
      expectedFacts: [
        'Neither focus action changes the 2-2 district score.',
        'Development changes the provisional outcome from behind to ahead at the rank-total level.',
      ],
      state: positionState({
        id: 'rank-tiebreak-conversion',
        turn: 43,
        ownHand: ['6', '7', '8'],
        selfResources: { Moons: 1, Knots: 1 },
        reshuffles: 2,
        finalTurnsRemaining: 1,
        drawCount: 0,
        districts: [
          district('D0', ['Waves', 'Leaves', 'Wyrms'], ['26']),
          district('D1', ['Moons', 'Suns', 'Leaves'], ['25']),
          district('D2', ['Suns', 'Waves', 'Knots'], ['4'], ['24', '12']),
          district('D3', ['Moons', 'Wyrms', 'Knots'], ['11'], ['27']),
          district(
            'D4',
            [],
            [],
            [],
            { cardId: '14', progress: 0, tokens: {} },
            { cardId: '17', progress: 0, tokens: {} }
          ),
        ],
      }),
      focusActions: [
        focus('convert-rank', 'Develop The Author in D1', {
          type: 'develop-outright',
          cardId: '6',
          districtId: 'D1',
        }),
        focus('sell', 'Sell The Author', {
          type: 'sell-card',
          cardId: '6',
        }),
      ],
      expectedPreference: {
        preferredFocusActionId: 'convert-rank',
        overFocusActionIds: ['sell'],
        rationale:
          'District count remains tied, so the otherwise secondary developed-rank total is the live winning condition.',
      },
    },
    {
      id: 'endpoint-optionality',
      title: 'Equal immediate score, unequal continuation support',
      theme: 'placement-optionality',
      thesis:
        'Two rank-2 cards both win the fifth district now, but The Author preserves a compatible card in hand and matches much more of the public unknown-card support.',
      expectedFacts: [
        'Both focus actions have identical immediate district, rank, and resource deltas.',
        'After The Author, D4 is compatible with The Journey and every configured unknown card; after The Desert it is not.',
      ],
      state: positionState({
        id: 'endpoint-optionality',
        turn: 31,
        ownHand: ['6', '7', '9'],
        selfResources: { Moons: 1, Knots: 1, Suns: 1, Wyrms: 1 },
        unknownCardIds: ['0', '2', '16', '18', '19', '23', '28', '29'],
        drawCount: 5,
        districts: [
          district('D0', ['Waves', 'Leaves', 'Wyrms'], ['14']),
          district('D1', ['Moons', 'Suns', 'Leaves'], ['26']),
          district('D2', ['Suns', 'Waves', 'Knots'], ['8'], ['13']),
          district('D3', ['Moons', 'Wyrms', 'Knots'], ['11', '1'], ['24']),
          district('D4', []),
        ],
      }),
      focusActions: [
        focus('flexible-endpoint', 'Develop The Author in D4', {
          type: 'develop-outright',
          cardId: '6',
          districtId: 'D4',
        }),
        focus('dead-endpoint', 'Develop The Desert in D4', {
          type: 'develop-outright',
          cardId: '7',
          districtId: 'D4',
        }),
      ],
      expectedPreference: {
        preferredFocusActionId: 'flexible-endpoint',
        overFocusActionIds: ['dead-endpoint'],
        rationale:
          'Immediate match facts are equal, so preserving responses to future draws is the distinguishing strategic consideration.',
      },
    },
    {
      id: 'deed-fork-affordable',
      title: 'Affordable deed completion creates a two-front turn',
      theme: 'deed-feasibility',
      thesis:
        'Completing the rank-9 deed flips the fifth district without consuming the mandatory card play, leaving a second action front in the same turn.',
      expectedFacts: [
        'The deed has one remaining work and one matching loose resource.',
        'Completion changes the district score from 2-3 to 3-2 and leaves card play available.',
      ],
      pairId: 'deed-affordability',
      state: positionState({
        id: 'deed-fork-affordable',
        turn: 38,
        ownHand: ['7', '8', '9'],
        selfResources: { Suns: 1 },
        districts: deedForkDistricts,
      }),
      focusActions: [
        focus('complete-deed', 'Complete The Pact deed', {
          type: 'develop-deed',
          cardId: '29',
          districtId: 'D4',
        }),
        focus('sell', 'Sell The Desert', {
          type: 'sell-card',
          cardId: '7',
        }),
      ],
      expectedPreference: {
        preferredFocusActionId: 'complete-deed',
        overFocusActionIds: ['sell'],
        rationale:
          "The deed completion is an immediate match swing and does not spend the turn's card play.",
      },
    },
    {
      id: 'deed-fork-inaccessible',
      title: 'Identical deed progress without matching liquidity',
      theme: 'deed-feasibility',
      thesis:
        'Nominal deed progress is unchanged, but no matching loose resource makes immediate completion impossible.',
      expectedFacts: [
        'The deed still has one remaining work.',
        'Its completion shortfall is one and no develop-deed action is legal.',
      ],
      pairId: 'deed-affordability',
      state: positionState({
        id: 'deed-fork-inaccessible',
        turn: 38,
        ownHand: ['7', '8', '9'],
        selfResources: { Knots: 1 },
        districts: deedForkDistricts,
      }),
      focusActions: [
        focus('sell', 'Sell The Desert', {
          type: 'sell-card',
          cardId: '7',
        }),
      ],
    },
    ...clockBoundaryRecipes(),
    {
      id: 'ace-aware-control',
      title: 'Ace bonus reverses the raw-rank comparison',
      theme: 'ace-scoring',
      thesis:
        "The Moons Ace gives two bonus points, so a raw rank total of 8 controls D4 over the opponent's raw rank 9.",
      expectedFacts: [
        'Player A has developed rank 8, Ace bonus 2, and district score 10 in D4.',
        'Player B has developed rank and district score 9 in D4.',
      ],
      state: positionState({
        id: 'ace-aware-control',
        turn: 24,
        ownHand: ['6', '7', '8'],
        districts: [
          district('D0', ['Waves', 'Leaves', 'Wyrms']),
          district('D1', ['Moons', 'Suns', 'Leaves']),
          district('D2', ['Suns', 'Waves', 'Knots']),
          district('D3', ['Moons', 'Wyrms', 'Knots']),
          district('D4', [], ['2', '13', '9'], ['29']),
        ],
      }),
      focusActions: [
        focus('sell', 'Sell The Author', {
          type: 'sell-card',
          cardId: '6',
        }),
      ],
    },
  ];
}

function clockBoundaryRecipes(): StrategicPositionRecipe[] {
  const base = {
    turn: 28,
    ownHand: ['6', '7', '8'] as const,
    selfResources: { Moons: 1, Knots: 1 },
    drawCount: 4,
    districts: [
      district('D0', ['Waves', 'Leaves', 'Wyrms'], ['26', '16']),
      district('D1', ['Moons', 'Suns', 'Leaves'], ['25']),
      district('D2', ['Suns', 'Waves', 'Knots'], [], ['14']),
      district('D3', ['Moons', 'Wyrms', 'Knots'], [], ['17']),
      district('D4', []),
    ],
  } satisfies Omit<PositionStateRecipe, 'id' | 'reshuffles'>;
  const focusActions = [
    focus('sell', 'Sell The Desert', {
      type: 'sell-card',
      cardId: '7',
    }),
    focus('develop', 'Develop The Author in D4', {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D4',
    }),
  ];
  return [
    {
      id: 'sale-before-first-reshuffle',
      title: 'Sale before the first reshuffle',
      theme: 'game-clock',
      thesis:
        'A sold card enters the first reshuffle and therefore remains eligible for a future draw.',
      expectedFacts: [
        'The sale destination is first-reshuffle-discard.',
        'The developed card is removed from future draw circulation.',
      ],
      pairId: 'reshuffle-boundary',
      state: positionState({
        ...base,
        id: 'sale-before-first-reshuffle',
        reshuffles: 0,
      }),
      focusActions,
    },
    {
      id: 'sale-after-first-reshuffle',
      title: 'Same sale after the first reshuffle',
      theme: 'game-clock',
      thesis:
        'After the only reshuffle has occurred, a sold card goes to a discard pile that will not return to the draw pile.',
      expectedFacts: [
        'The sale destination is dead-discard.',
        'All non-clock board, hand, resource, and focus-action facts match the paired pre-reshuffle position.',
      ],
      pairId: 'reshuffle-boundary',
      state: positionState({
        ...base,
        id: 'sale-after-first-reshuffle',
        reshuffles: 1,
      }),
      focusActions,
    },
  ];
}

function resolvePositionRecipe(
  recipe: StrategicPositionRecipe
): StrategicPositionV0 {
  const focusActions = recipe.focusActions.map((focusAction) => ({
    id: focusAction.id,
    label: focusAction.label,
    actionKey: resolveActionKey(recipe.state, focusAction.selector),
  }));
  if (
    new Set(focusActions.map((action) => action.id)).size !==
    focusActions.length
  ) {
    throw new Error(
      `Strategic position ${recipe.id} has duplicate focus action ids.`
    );
  }
  const focusIds = new Set(focusActions.map((action) => action.id));
  const preference = recipe.expectedPreference;
  if (
    preference &&
    (!focusIds.has(preference.preferredFocusActionId) ||
      preference.overFocusActionIds.some((id) => !focusIds.has(id)))
  ) {
    throw new Error(
      `Strategic position ${recipe.id} preference references an unknown focus action.`
    );
  }
  return {
    catalogVersion: STRATEGIC_POSITION_CATALOG_VERSION,
    id: recipe.id,
    title: recipe.title,
    theme: recipe.theme,
    perspectivePlayerId: 'PlayerA',
    thesis: recipe.thesis,
    expectedFacts: [...recipe.expectedFacts],
    pairId: recipe.pairId ?? null,
    state: structuredClone(recipe.state),
    focusActions,
    expectedPreference: preference ? structuredClone(preference) : null,
  };
}

function focus(
  id: string,
  label: string,
  selector: ActionSelector
): FocusActionRecipe {
  return { id, label, selector };
}

function resolveActionKey(state: GameState, selector: ActionSelector): string {
  const actions = legalActionsForDecisionPlayer(state, 'PlayerA').filter(
    (action) => actionMatchesSelector(action, selector)
  );
  if (actions.length !== 1) {
    throw new Error(
      `Strategic action selector ${JSON.stringify(selector)} matched ${String(actions.length)} actions in state ${state.seed}.`
    );
  }
  return actionStableKey(actions[0]);
}

function actionMatchesSelector(
  action: GameAction,
  selector: ActionSelector
): boolean {
  if (action.type !== selector.type) {
    return false;
  }
  if (
    selector.cardId !== undefined &&
    (!('cardId' in action) || action.cardId !== selector.cardId)
  ) {
    return false;
  }
  return !(
    selector.districtId !== undefined &&
    (!('districtId' in action) || action.districtId !== selector.districtId)
  );
}

function positionState(recipe: PositionStateRecipe): GameState {
  const opponentHandCount = recipe.opponentHandCount ?? 3;
  const knownCards = new Set<CardId>();
  for (const cardId of recipe.ownHand) {
    addUniqueCard(knownCards, cardId, `${recipe.id} own hand`);
  }
  for (const districtState of recipe.districts) {
    for (const playerId of ['PlayerA', 'PlayerB'] as const) {
      const stack = districtState.stacks[playerId];
      for (const cardId of stack.developed) {
        addUniqueCard(knownCards, cardId, `${recipe.id} board`);
      }
      if (stack.deed) {
        addUniqueCard(knownCards, stack.deed.cardId, `${recipe.id} deed`);
      }
    }
  }

  const remaining = PROPERTY_CARDS.map((card) => card.id).filter(
    (cardId) => !knownCards.has(cardId)
  );
  const requestedDrawCount =
    recipe.drawCount ?? Math.min(10, remaining.length - opponentHandCount);
  const requestedUnknownCount = opponentHandCount + requestedDrawCount;
  const unknown = recipe.unknownCardIds
    ? [...recipe.unknownCardIds]
    : remaining.slice(0, requestedUnknownCount);
  if (unknown.length !== requestedUnknownCount) {
    throw new Error(
      `Strategic position ${recipe.id} requires ${String(requestedUnknownCount)} unknown cards; received ${String(unknown.length)}.`
    );
  }
  const unknownSet = new Set<CardId>();
  for (const cardId of unknown) {
    if (!remaining.includes(cardId)) {
      throw new Error(
        `Strategic position ${recipe.id} unknown card ${cardId} is already public.`
      );
    }
    addUniqueCard(unknownSet, cardId, `${recipe.id} unknown pool`);
  }
  const discard = remaining.filter((cardId) => !unknownSet.has(cardId));
  const opponentHand = unknown.slice(0, opponentHandCount);
  const draw = unknown.slice(opponentHandCount);

  return {
    schemaVersion: 1,
    seed: `strategic-v0:${recipe.id}`,
    rngCursor: 0,
    deck: {
      draw,
      discard,
      reshuffles: recipe.reshuffles ?? 1,
    },
    players: [
      {
        id: 'PlayerA',
        hand: [...recipe.ownHand],
        crowns: ['30', '32', '34'],
        resources: resources(recipe.selfResources),
      },
      {
        id: 'PlayerB',
        hand: opponentHand,
        crowns: ['31', '33', '35'],
        resources: resources(recipe.opponentResources),
      },
    ],
    activePlayerIndex: 0,
    turn: recipe.turn,
    phase: 'ActionWindow',
    districts: recipe.districts.map((districtState) =>
      structuredClone(districtState)
    ),
    cardPlayedThisTurn: false,
    finalTurnsRemaining: recipe.finalTurnsRemaining,
    log: [],
  };
}

function district(
  id: string,
  markerSuitMask: readonly Suit[],
  playerADeveloped: readonly CardId[] = [],
  playerBDeveloped: readonly CardId[] = [],
  playerADeed?: DeedState,
  playerBDeed?: DeedState
): DistrictState {
  return {
    id,
    markerSuitMask: [...markerSuitMask],
    stacks: {
      PlayerA: {
        developed: [...playerADeveloped],
        ...(playerADeed ? { deed: structuredClone(playerADeed) } : {}),
      },
      PlayerB: {
        developed: [...playerBDeveloped],
        ...(playerBDeed ? { deed: structuredClone(playerBDeed) } : {}),
      },
    },
  };
}

function resources(
  overrides: Partial<Record<Suit, number>> = {}
): ResourcePool {
  return Object.fromEntries(
    SUITS.map((suit) => [suit, overrides[suit] ?? 0])
  ) as ResourcePool;
}

function addUniqueCard(
  cards: Set<CardId>,
  cardId: CardId,
  label: string
): void {
  if (cards.has(cardId)) {
    throw new Error(`${label} duplicates property card ${cardId}.`);
  }
  cards.add(cardId);
}
