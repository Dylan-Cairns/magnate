import { stepToDecision as defaultStepToDecision } from '../../engine/session';
import type {
  GameAction,
  GameState,
  IncomeChoice,
  PlayerId,
  SubmittedIncomeChoice,
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
