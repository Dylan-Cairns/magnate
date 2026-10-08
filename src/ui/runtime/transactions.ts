import { ActionId, SUITS } from '../../engine/values';
import { GamePresentationEventType, IncomeTokenSourceKind } from './values';
import { stepToDecision as defaultStepToDecision } from '../../engine/session';
import {
  developmentCost,
  findDevelopableCard,
} from '../../engine/stateHelpers';
import {
  type GameAction,
  type GameState,
  type IncomeChoice,
  type PlayerId,
  type ResourcePool,
  type SubmittedIncomeChoice,
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
      type: GamePresentationEventType.ActionStarted,
      action,
      actingPlayerId,
    },
  ];

  if (action.type === ActionId.EndTurn) {
    events.push(...deriveEndTurnEvents(previousState, nextState, action));
  }
  if (action.type === ActionId.SellCard) {
    events.push({
      type: GamePresentationEventType.CardSold,
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
  if (action.type === ActionId.BuyDeed) {
    events.push(
      ...deriveCardPlayEvents(
        previousState,
        nextState,
        action,
        actingPlayerId,
        ActionId.BuyDeed,
        'deed'
      )
    );
  }
  if (action.type === ActionId.DevelopOutright) {
    events.push(
      ...deriveCardPlayEvents(
        previousState,
        nextState,
        action,
        actingPlayerId,
        ActionId.DevelopOutright,
        'developed'
      )
    );
  }
  if (action.type === ActionId.DevelopDeed) {
    events.push(
      ...deriveDevelopDeedEvents(
        previousState,
        nextState,
        action,
        actingPlayerId
      )
    );
  }
  if (action.type === ActionId.Trade) {
    events.push({
      type: GamePresentationEventType.TradeResourcesApplied,
      playerId: actingPlayerId,
      give: action.give,
      receive: action.receive,
      giveCount: 3,
      receiveCount: 1,
    });
  }
  if (action.type === ActionId.ChooseIncomeSuit) {
    events.push(...deriveIncomeChoiceEvents(previousState, nextState, action));
  }

  const previousPlayerId = activePlayerId(previousState);
  const nextPlayerId = activePlayerId(nextState);
  if (previousPlayerId !== nextPlayerId) {
    events.push({
      type: GamePresentationEventType.ActivePlayerChanged,
      previousPlayerId,
      nextPlayerId,
    });
  }
  if (previousState.phase !== nextState.phase) {
    events.push({
      type: GamePresentationEventType.PhaseChanged,
      previousPhase: previousState.phase,
      nextPhase: nextState.phase,
    });
  }
  events.push({ type: GamePresentationEventType.TransactionSettled });
  return events;
}

function deriveCardPlayEvents(
  previousState: GameState,
  nextState: GameState,
  action: Extract<
    GameAction,
    { type: typeof ActionId.BuyDeed | typeof ActionId.DevelopOutright }
  >,
  actingPlayerId: PlayerId,
  reason: typeof ActionId.BuyDeed | typeof ActionId.DevelopOutright,
  placement: 'deed' | 'developed'
): GamePresentationEvent[] {
  const payment = resourcePaymentForPlayer(
    previousState,
    nextState,
    actingPlayerId
  );
  return [
    {
      type: GamePresentationEventType.ResourcePaymentStarted,
      playerId: actingPlayerId,
      reason,
      cardId: action.cardId,
      districtId: action.districtId,
      payment,
    },
    {
      type: GamePresentationEventType.ResourcePaymentApplied,
      playerId: actingPlayerId,
      reason,
      cardId: action.cardId,
      districtId: action.districtId,
      payment,
    },
    {
      type: GamePresentationEventType.CardPlayedToDistrict,
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
  action: Extract<GameAction, { type: typeof ActionId.DevelopDeed }>,
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
  const card = findDevelopableCard(action.cardId);
  const targetProgress = card
    ? developmentCost(card)
    : (nextDeed?.progress ?? previousDeed?.progress ?? 0);
  const previousProgress = previousDeed?.progress ?? 0;
  const nextProgress = nextDeed?.progress ?? targetProgress;
  const completed = !nextDeed && nextProgress >= targetProgress;
  const previousTokens = { ...(previousDeed?.tokens ?? {}) };
  const nextTokens: Partial<Record<Suit, number>> = { ...previousTokens };
  for (const entry of tokenEntries(payment)) {
    nextTokens[entry.suit] = (nextTokens[entry.suit] ?? 0) + entry.count;
  }
  const events: GamePresentationEvent[] = [
    {
      type: GamePresentationEventType.ResourcePaymentStarted,
      playerId: actingPlayerId,
      reason: ActionId.DevelopDeed,
      cardId: action.cardId,
      districtId: action.districtId,
      payment,
    },
  ];

  for (const entry of tokenEntries(payment)) {
    for (let tokenIndex = 0; tokenIndex < entry.count; tokenIndex += 1) {
      events.push({
        type: GamePresentationEventType.DeedTokenPaid,
        playerId: actingPlayerId,
        districtId: action.districtId,
        cardId: action.cardId,
        suit: entry.suit,
        tokenIndex,
        previousTokens,
        nextTokens,
      });
    }
  }

  events.push(
    {
      type: GamePresentationEventType.ResourcePaymentApplied,
      playerId: actingPlayerId,
      reason: ActionId.DevelopDeed,
      cardId: action.cardId,
      districtId: action.districtId,
      payment,
    },
    {
      type: GamePresentationEventType.DeedProgressApplied,
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
      type: GamePresentationEventType.DeedCompleted,
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
  action: Extract<GameAction, { type: typeof ActionId.SellCard }>,
  actingPlayerId: PlayerId
): GamePresentationEvent[] {
  const gains = resourceGainForPlayer(previousState, nextState, actingPlayerId);
  const events: GamePresentationEvent[] = [];
  for (const entry of tokenEntries(gains)) {
    for (let tokenIndex = 0; tokenIndex < entry.count; tokenIndex += 1) {
      events.push({
        type: GamePresentationEventType.SellResourceGained,
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
  action: Extract<GameAction, { type: typeof ActionId.EndTurn }>
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
      type: GamePresentationEventType.DrawCard,
      playerId: previousPlayer.id,
      cardId: nextPlayer.hand[nextPlayer.hand.length - 1],
    });
  }

  const cycle = deriveTurnCycleEvents(previousState, nextState, action);
  if (!cycle) {
    return events;
  }

  events.push({
    type: GamePresentationEventType.IncomeRoll,
    playerId: cycle.cycleOwner,
    turn: nextState.turn,
    roll: cycle.roll,
    incomeRank: cycle.incomeRank,
  });

  if (cycle.tax) {
    events.push({
      type: GamePresentationEventType.TaxResolved,
      suit: cycle.tax.suit,
    });
    for (const loss of cycle.tax.lossesByPlayer) {
      for (let tokenIndex = 0; tokenIndex < loss.count; tokenIndex += 1) {
        events.push({
          type: GamePresentationEventType.TaxTokenLost,
          playerId: loss.playerId,
          suit: cycle.tax.suit,
          tokenIndex,
        });
      }
    }
  }

  for (const token of cycle.incomeTokens) {
    events.push({
      type: GamePresentationEventType.IncomeTokenGained,
      playerId: token.playerId,
      suit: token.suit,
      source: token.source,
    });
  }

  if (cycle.pendingChoices.length > 0) {
    events.push({
      type: GamePresentationEventType.IncomeChoiceRequired,
      choices: cloneIncomeChoices(cycle.pendingChoices),
      returnPlayerId: nextState.incomeChoiceReturnPlayerId,
    });
  }

  return events;
}

function deriveIncomeChoiceEvents(
  previousState: GameState,
  nextState: GameState,
  action: Extract<GameAction, { type: typeof ActionId.ChooseIncomeSuit }>
): GamePresentationEvent[] {
  const events: GamePresentationEvent[] = [
    {
      type: GamePresentationEventType.IncomeChoiceSubmitted,
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
      type: GamePresentationEventType.IncomeTokenGained,
      playerId: submission.playerId,
      suit: submission.suit,
      source: {
        kind: IncomeTokenSourceKind.IncomeChoice,
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
    case IncomeTokenSourceKind.DistrictCard:
      return `district-card:${source.districtId}:${source.cardId}`;
    case IncomeTokenSourceKind.Crown:
      return `crown:${source.cardId}`;
    case IncomeTokenSourceKind.IncomeChoice:
      return `income-choice:${source.districtId}:${source.cardId}`;
  }
}
