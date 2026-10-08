import { Suit } from './values';
import {
  type DistrictStack,
  type FinalScore,
  type DistrictState,
  type GameLogEntry,
  type GameState,
  type IncomeChoice,
  type IncomeRollResult,
  type ObservedPlayerState,
  PlayerId,
  type PlayerState,
  type PlayerView,
  type ResourcePool,
  type SubmittedIncomeChoice,
} from './types';

export function toPlayerView(state: GameState, viewerId: PlayerId): PlayerView {
  assertPlayerExists(state, viewerId);

  const activePlayer = state.players[state.activePlayerIndex];
  if (!activePlayer) {
    throw new Error(
      `Active player index ${state.activePlayerIndex} is out of bounds.`
    );
  }

  // Income choice selection is double-blind: while some players have submitted
  // but others haven't, hide the already-submitted choices and their log entries
  // from players who haven't submitted yet, so no one can react to an opponent's
  // choice before making their own.
  const pendingCount = state.pendingIncomeChoices?.length ?? 0;
  const submittedCount = state.submittedIncomeChoices?.length ?? 0;
  const selectionInProgress = pendingCount > submittedCount;

  const filteredSubmitted = selectionInProgress
    ? state.submittedIncomeChoices?.filter((c) => c.playerId === viewerId)
    : state.submittedIncomeChoices;
  // Normalize an empty filtered array to undefined to match the convention that
  // undefined means "no submissions visible yet" (avoids [] vs undefined ambiguity).
  const visibleSubmitted =
    filteredSubmitted?.length === 0 ? undefined : filteredSubmitted;

  const visibleLog = visibleLogEntriesForPlayer(state.log, state, viewerId);

  return {
    viewerId,
    activePlayerId: activePlayer.id,
    turn: state.turn,
    phase: state.phase,
    districts: state.districts.map(cloneDistrict),
    players: state.players.map((player) => toObservedPlayer(player, viewerId)),
    deck: {
      drawCount: state.deck.draw.length,
      discard: [...state.deck.discard],
      reshuffles: state.deck.reshuffles,
    },
    cardPlayedThisTurn: state.cardPlayedThisTurn,
    finalTurnsRemaining: state.finalTurnsRemaining,
    lastIncomeRoll: cloneIncomeRoll(state.lastIncomeRoll),
    lastTaxSuit: state.lastTaxSuit,
    pendingIncomeChoices: cloneIncomeChoices(state.pendingIncomeChoices),
    submittedIncomeChoices: cloneSubmittedIncomeChoices(visibleSubmitted),
    incomeChoiceReturnPlayerId: state.incomeChoiceReturnPlayerId,
    finalScore: cloneFinalScore(state.finalScore),
    log: visibleLog.map(cloneLogEntry),
  };
}

export function visibleLogEntriesForPlayer(
  entries: ReadonlyArray<GameLogEntry>,
  state: Pick<
    GameState,
    'turn' | 'pendingIncomeChoices' | 'submittedIncomeChoices'
  >,
  viewerId: PlayerId
): ReadonlyArray<GameLogEntry> {
  const selectionInProgress =
    (state.pendingIncomeChoices?.length ?? 0) >
    (state.submittedIncomeChoices?.length ?? 0);
  if (!selectionInProgress) return entries;

  return entries.filter(
    (entry) =>
      entry.turn !== state.turn ||
      !entry.summary.startsWith('income choice ') ||
      entry.player === viewerId
  );
}

export function toActivePlayerView(state: GameState): PlayerView {
  const activePlayer = state.players[state.activePlayerIndex];
  if (!activePlayer) {
    throw new Error(
      `Active player index ${state.activePlayerIndex} is out of bounds.`
    );
  }
  return toPlayerView(state, activePlayer.id);
}

function toObservedPlayer(
  player: PlayerState,
  viewerId: PlayerId
): ObservedPlayerState {
  const ownPerspective = player.id === viewerId;
  return {
    id: player.id,
    crowns: [...player.crowns],
    resources: cloneResources(player.resources),
    hand: ownPerspective ? [...player.hand] : [],
    handCount: player.hand.length,
    handHidden: !ownPerspective,
  };
}

function cloneDistrict(district: DistrictState): DistrictState {
  return {
    id: district.id,
    markerSuitMask: [...district.markerSuitMask],
    stacks: {
      [PlayerId.PlayerA]: cloneStack(district.stacks[PlayerId.PlayerA]),
      [PlayerId.PlayerB]: cloneStack(district.stacks[PlayerId.PlayerB]),
    },
  };
}

function cloneStack(stack: DistrictStack): DistrictStack {
  if (!stack.deed) {
    return { developed: [...stack.developed] };
  }
  return {
    developed: [...stack.developed],
    deed: {
      cardId: stack.deed.cardId,
      progress: stack.deed.progress,
      tokens: { ...stack.deed.tokens },
    },
  };
}

function cloneResources(resources: ResourcePool): ResourcePool {
  return {
    [Suit.Moons]: resources[Suit.Moons],
    [Suit.Suns]: resources[Suit.Suns],
    [Suit.Waves]: resources[Suit.Waves],
    [Suit.Leaves]: resources[Suit.Leaves],
    [Suit.Wyrms]: resources[Suit.Wyrms],
    [Suit.Knots]: resources[Suit.Knots],
  };
}

function cloneIncomeRoll(
  roll: IncomeRollResult | undefined
): IncomeRollResult | undefined {
  if (!roll) {
    return undefined;
  }
  return {
    die1: roll.die1,
    die2: roll.die2,
    rollId: roll.rollId,
  };
}

function cloneIncomeChoices(
  choices: ReadonlyArray<IncomeChoice> | undefined
): ReadonlyArray<IncomeChoice> | undefined {
  if (!choices) {
    return undefined;
  }
  return choices.map((choice) => ({
    playerId: choice.playerId,
    districtId: choice.districtId,
    cardId: choice.cardId,
    suits: [...choice.suits],
  }));
}

function cloneSubmittedIncomeChoices(
  choices: ReadonlyArray<SubmittedIncomeChoice> | undefined
): ReadonlyArray<SubmittedIncomeChoice> | undefined {
  if (!choices) {
    return undefined;
  }
  return choices.map((choice) => ({
    playerId: choice.playerId,
    districtId: choice.districtId,
    cardId: choice.cardId,
    suit: choice.suit,
  }));
}

function cloneLogEntry(entry: GameLogEntry): GameLogEntry {
  return {
    turn: entry.turn,
    player: entry.player,
    phase: entry.phase,
    summary: entry.summary,
    details: entry.details ? { ...entry.details } : undefined,
  };
}

function cloneFinalScore(
  score: FinalScore | undefined
): FinalScore | undefined {
  if (!score) {
    return undefined;
  }
  return {
    districtPoints: {
      [PlayerId.PlayerA]: score.districtPoints[PlayerId.PlayerA],
      [PlayerId.PlayerB]: score.districtPoints[PlayerId.PlayerB],
    },
    rankTotals: {
      [PlayerId.PlayerA]: score.rankTotals[PlayerId.PlayerA],
      [PlayerId.PlayerB]: score.rankTotals[PlayerId.PlayerB],
    },
    resourceTotals: {
      [PlayerId.PlayerA]: score.resourceTotals[PlayerId.PlayerA],
      [PlayerId.PlayerB]: score.resourceTotals[PlayerId.PlayerB],
    },
    winner: score.winner,
    decidedBy: score.decidedBy,
  };
}

function assertPlayerExists(state: GameState, playerId: PlayerId): void {
  const exists = state.players.some((player) => player.id === playerId);
  if (!exists) {
    throw new Error(`Unknown player: ${playerId}`);
  }
}
