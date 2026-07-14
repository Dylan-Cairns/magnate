import { stepToDecision as defaultStepToDecision } from '../../engine/session';
import { developmentCost, findProperty } from '../../engine/stateHelpers';
import type {
  GameAction,
  GameState,
  IncomeChoice,
  PlayerId,
  ResourcePool,
  SubmittedIncomeChoice,
  Suit,
} from '../../engine/types';
import { deriveTurnCycleEvents } from '../turnCycleEvents';
import type {
  GamePresentationEvent,
  GameTransaction,
  IncomeTokenSource,
} from './types';

export type BuildGameTransactionOptions = {
  previousState: GameState;
  action: GameAction;
  actingPlayerId: PlayerId;
  transactionId?: string;
  stepToDecision?: (state: GameState, action: GameAction) => GameState;
};

export function buildGameTransaction({
  previousState,
  action,
  actingPlayerId,
  transactionId,
  stepToDecision = defaultStepToDecision,
}: BuildGameTransactionOptions): GameTransaction {
  const nextState = stepToDecision(previousState, action);
  const id =
    transactionId ??
    `${previousState.seed}:${previousState.turn}:${previousState.phase}:${action.type}`;

  return {
    id,
    previousState,
    nextState,
    action,
    actingPlayerId,
    events: deriveGamePresentationEvents(
      previousState,
      nextState,
      action,
      actingPlayerId
    ),
  };
}

export function deriveGamePresentationEvents(
  previousState: GameState,
  nextState: GameState,
  action: GameAction,
  actingPlayerId: PlayerId
): readonly GamePresentationEvent[] {
  const events: GamePresentationEvent[] = [
    {
      type: 'action-started',
      action,
      actingPlayerId,
    },
  ];

  if (action.type === 'end-turn') {
    events.push(...deriveEndTurnEvents(previousState, nextState, action));
  }
  if (action.type === 'sell-card') {
    events.push({
      type: 'card-sold',
      playerId: actingPlayerId,
      cardId: action.cardId,
    });
    events.push(
      ...deriveSellResourceGainEvents(
        previousState,
        nextState,
        action,
        actingPlayerId
      )
    );
  }
  if (action.type === 'buy-deed') {
    events.push(
      ...deriveCardPlayEvents(
        previousState,
        nextState,
        action,
        actingPlayerId,
        'buy-deed',
        'deed'
      )
    );
  }
  if (action.type === 'develop-outright') {
    events.push(
      ...deriveCardPlayEvents(
        previousState,
        nextState,
        action,
        actingPlayerId,
        'develop-outright',
        'developed'
      )
    );
  }
  if (action.type === 'develop-deed') {
    events.push(
      ...deriveDevelopDeedEvents(
        previousState,
        nextState,
        action,
        actingPlayerId
      )
    );
  }
  if (action.type === 'trade') {
    events.push({
      type: 'trade-resources-applied',
      playerId: actingPlayerId,
      give: action.give,
      receive: action.receive,
      giveCount: 3,
      receiveCount: 1,
    });
  }
  if (action.type === 'choose-income-suit') {
    events.push(...deriveIncomeChoiceEvents(previousState, nextState, action));
  }

  const previousPlayerId = activePlayerId(previousState);
  const nextPlayerId = activePlayerId(nextState);
  if (previousPlayerId !== nextPlayerId) {
    events.push({
      type: 'active-player-changed',
      previousPlayerId,
      nextPlayerId,
    });
  }
  if (previousState.phase !== nextState.phase) {
    events.push({
      type: 'phase-changed',
      previousPhase: previousState.phase,
      nextPhase: nextState.phase,
    });
  }
  events.push({ type: 'transaction-settled' });
  return events;
}

function deriveCardPlayEvents(
  previousState: GameState,
  nextState: GameState,
  action: Extract<GameAction, { type: 'buy-deed' | 'develop-outright' }>,
  actingPlayerId: PlayerId,
  reason: 'buy-deed' | 'develop-outright',
  placement: 'deed' | 'developed'
): GamePresentationEvent[] {
  const payment = resourcePaymentForPlayer(
    previousState,
    nextState,
    actingPlayerId
  );
  return [
    {
      type: 'resource-payment-started',
      playerId: actingPlayerId,
      reason,
      cardId: action.cardId,
      districtId: action.districtId,
      payment,
    },
    {
      type: 'resource-payment-applied',
      playerId: actingPlayerId,
      reason,
      cardId: action.cardId,
      districtId: action.districtId,
      payment,
    },
    {
      type: 'card-played-to-district',
      playerId: actingPlayerId,
      cardId: action.cardId,
      districtId: action.districtId,
      placement,
    },
  ];
}

function deriveDevelopDeedEvents(
  previousState: GameState,
  nextState: GameState,
  action: Extract<GameAction, { type: 'develop-deed' }>,
  actingPlayerId: PlayerId
): GamePresentationEvent[] {
  const payment = resourcePaymentForPlayer(
    previousState,
    nextState,
    actingPlayerId
  );
  const previousDeed = previousState.districts.find(
    (district) => district.id === action.districtId
  )?.stacks[actingPlayerId]?.deed;
  const nextDeed = nextState.districts.find(
    (district) => district.id === action.districtId
  )?.stacks[actingPlayerId]?.deed;
  const card = findProperty(action.cardId);
  const targetProgress = card
    ? developmentCost(card)
    : (nextDeed?.progress ?? previousDeed?.progress ?? 0);
  const previousProgress = previousDeed?.progress ?? 0;
  const nextProgress = nextDeed?.progress ?? targetProgress;
  const completed = !nextDeed && nextProgress >= targetProgress;
  const events: GamePresentationEvent[] = [
    {
      type: 'resource-payment-started',
      playerId: actingPlayerId,
      reason: 'develop-deed',
      cardId: action.cardId,
      districtId: action.districtId,
      payment,
    },
  ];

  for (const entry of tokenEntries(payment)) {
    for (let tokenIndex = 0; tokenIndex < entry.count; tokenIndex += 1) {
      events.push({
        type: 'deed-token-paid',
        playerId: actingPlayerId,
        districtId: action.districtId,
        cardId: action.cardId,
        suit: entry.suit,
        tokenIndex,
      });
    }
  }

  events.push(
    {
      type: 'resource-payment-applied',
      playerId: actingPlayerId,
      reason: 'develop-deed',
      cardId: action.cardId,
      districtId: action.districtId,
      payment,
    },
    {
      type: 'deed-progress-applied',
      playerId: actingPlayerId,
      districtId: action.districtId,
      cardId: action.cardId,
      previousProgress,
      nextProgress,
      targetProgress,
      completed,
    }
  );
  if (completed) {
    events.push({
      type: 'deed-completed',
      playerId: actingPlayerId,
      districtId: action.districtId,
      cardId: action.cardId,
    });
  }
  return events;
}

function deriveSellResourceGainEvents(
  previousState: GameState,
  nextState: GameState,
  action: Extract<GameAction, { type: 'sell-card' }>,
  actingPlayerId: PlayerId
): GamePresentationEvent[] {
  const gains = resourceGainForPlayer(previousState, nextState, actingPlayerId);
  const events: GamePresentationEvent[] = [];
  for (const entry of tokenEntries(gains)) {
    for (let tokenIndex = 0; tokenIndex < entry.count; tokenIndex += 1) {
      events.push({
        type: 'sell-resource-gained',
        playerId: actingPlayerId,
        cardId: action.cardId,
        suit: entry.suit,
        tokenIndex,
      });
    }
  }
  return events;
}

function deriveEndTurnEvents(
  previousState: GameState,
  nextState: GameState,
  action: Extract<GameAction, { type: 'end-turn' }>
): GamePresentationEvent[] {
  const events: GamePresentationEvent[] = [];
  const previousPlayer = previousState.players.find(
    (player) =>
      player.id === previousState.players[previousState.activePlayerIndex]?.id
  );
  const nextPlayer = previousPlayer
    ? nextState.players.find((player) => player.id === previousPlayer.id)
    : undefined;
  if (
    previousPlayer &&
    nextPlayer &&
    nextPlayer.hand.length === previousPlayer.hand.length + 1
  ) {
    events.push({
      type: 'draw-card',
      playerId: previousPlayer.id,
      cardId: nextPlayer.hand[nextPlayer.hand.length - 1],
    });
  }

  const cycle = deriveTurnCycleEvents(previousState, nextState, action);
  if (!cycle) {
    return events;
  }

  events.push({
    type: 'income-roll',
    playerId: cycle.cycleOwner,
    roll: cycle.roll,
    incomeRank: cycle.incomeRank,
  });

  if (cycle.tax) {
    events.push({
      type: 'tax-resolved',
      suit: cycle.tax.suit,
    });
    for (const loss of cycle.tax.lossesByPlayer) {
      for (let tokenIndex = 0; tokenIndex < loss.count; tokenIndex += 1) {
        events.push({
          type: 'tax-token-lost',
          playerId: loss.playerId,
          suit: cycle.tax.suit,
          tokenIndex,
        });
      }
    }
  }

  for (const token of cycle.incomeTokens) {
    events.push({
      type: 'income-token-gained',
      playerId: token.playerId,
      suit: token.suit,
      source: token.source,
    });
  }

  if (cycle.pendingChoices.length > 0) {
    events.push({
      type: 'income-choice-required',
      choices: cloneIncomeChoices(cycle.pendingChoices),
    });
  }

  return events;
}

function deriveIncomeChoiceEvents(
  previousState: GameState,
  nextState: GameState,
  action: Extract<GameAction, { type: 'choose-income-suit' }>
): GamePresentationEvent[] {
  const events: GamePresentationEvent[] = [
    {
      type: 'income-choice-submitted',
      playerId: action.playerId,
      districtId: action.districtId,
      cardId: action.cardId,
      suit: action.suit,
    },
  ];

  if ((nextState.pendingIncomeChoices?.length ?? 0) > 0) {
    return events;
  }

  const submissions: SubmittedIncomeChoice[] = [
    ...(previousState.submittedIncomeChoices ?? []),
    {
      playerId: action.playerId,
      districtId: action.districtId,
      cardId: action.cardId,
      suit: action.suit,
    },
  ];
  for (const choice of previousState.pendingIncomeChoices ?? []) {
    const submission = submissions.find((entry) =>
      incomeChoiceMatches(choice, entry)
    );
    if (!submission) {
      continue;
    }
    events.push({
      type: 'income-token-gained',
      playerId: submission.playerId,
      suit: submission.suit,
      source: {
        kind: 'income-choice',
        cardId: submission.cardId,
        districtId: submission.districtId,
      },
    });
  }
  return events;
}

function activePlayerId(state: GameState): PlayerId | null {
  return state.players[state.activePlayerIndex]?.id ?? null;
}

const SUITS: readonly Suit[] = [
  'Moons',
  'Suns',
  'Waves',
  'Leaves',
  'Wyrms',
  'Knots',
];

function resourcePaymentForPlayer(
  previousState: GameState,
  nextState: GameState,
  playerId: PlayerId
): Partial<Record<Suit, number>> {
  const previous = resourcesForPlayer(previousState, playerId);
  const next = resourcesForPlayer(nextState, playerId);
  if (!previous || !next) {
    return {};
  }
  const payment: Partial<Record<Suit, number>> = {};
  for (const suit of SUITS) {
    const spent = previous[suit] - next[suit];
    if (spent > 0) {
      payment[suit] = spent;
    }
  }
  return payment;
}

function resourceGainForPlayer(
  previousState: GameState,
  nextState: GameState,
  playerId: PlayerId
): Partial<Record<Suit, number>> {
  const previous = resourcesForPlayer(previousState, playerId);
  const next = resourcesForPlayer(nextState, playerId);
  if (!previous || !next) {
    return {};
  }
  const gains: Partial<Record<Suit, number>> = {};
  for (const suit of SUITS) {
    const gained = next[suit] - previous[suit];
    if (gained > 0) {
      gains[suit] = gained;
    }
  }
  return gains;
}

function resourcesForPlayer(
  state: GameState,
  playerId: PlayerId
): ResourcePool | undefined {
  return state.players.find((player) => player.id === playerId)?.resources;
}

function tokenEntries(
  tokens: Partial<Record<Suit, number>>
): Array<{ suit: Suit; count: number }> {
  return SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 })).filter(
    (entry) => entry.count > 0
  );
}

function incomeChoiceMatches(
  choice: Pick<IncomeChoice, 'playerId' | 'districtId' | 'cardId'>,
  submission: Pick<SubmittedIncomeChoice, 'playerId' | 'districtId' | 'cardId'>
): boolean {
  return (
    choice.playerId === submission.playerId &&
    choice.districtId === submission.districtId &&
    choice.cardId === submission.cardId
  );
}

function cloneIncomeChoices(
  choices: readonly IncomeChoice[]
): readonly IncomeChoice[] {
  return choices.map((choice) => ({
    ...choice,
    suits: [...choice.suits],
  }));
}

export function incomeTokenSourceKey(source: IncomeTokenSource): string {
  switch (source.kind) {
    case 'district-card':
      return `district-card:${source.districtId}:${source.cardId}`;
    case 'crown':
      return `crown:${source.cardId}`;
    case 'income-choice':
      return `income-choice:${source.districtId}:${source.cardId}`;
  }
}
