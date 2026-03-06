import type {
  GameAction,
  GameLogEntry,
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../engine/types';

const SUIT_ORDER: readonly Suit[] = [
  'Moons',
  'Suns',
  'Waves',
  'Leaves',
  'Wyrms',
  'Knots',
];

export function transitionLogEntries(
  previousState: GameState,
  nextState: GameState,
  action?: GameAction
): GameLogEntry[] {
  const startIndex =
    nextState.log.length >= previousState.log.length ? previousState.log.length : 0;
  const engineEntries = nextState.log.slice(startIndex);

  if (!action || action.type !== 'end-turn') {
    return engineEntries;
  }

  return [
    ...engineEntries,
    ...resolveTurnCycleEntries(previousState, nextState),
  ];
}

function resolveTurnCycleEntries(
  previousState: GameState,
  nextState: GameState
): GameLogEntry[] {
  const roll = nextState.lastIncomeRoll;
  if (!roll) {
    return [];
  }

  const playerId = cycleOwner(previousState, nextState);
  if (!playerId) {
    return [];
  }

  const incomeResult = Math.max(roll.die1, roll.die2);
  const taxSuit = nextState.lastTaxSuit;
  const postTaxResources = new Map<PlayerId, ResourcePool>();
  const taxLossByPlayer = new Map<PlayerId, number>();

  for (const player of previousState.players) {
    const postTax = taxSuit
      ? {
          ...player.resources,
          [taxSuit]: Math.min(player.resources[taxSuit], 1),
        }
      : player.resources;
    postTaxResources.set(player.id, postTax);
    if (taxSuit) {
      taxLossByPlayer.set(player.id, player.resources[taxSuit] - postTax[taxSuit]);
    }
  }

  const taxSummary = summarizeTax(nextState, taxLossByPlayer);
  const incomeSummaries = summarizeIncome(nextState, postTaxResources);

  const taxEntries: GameLogEntry[] = [];
  if (taxSummary) {
    taxEntries.push({
      turn: nextState.turn,
      player: playerId,
      phase: 'TaxCheck',
      summary: taxSummary,
    });
  }

  return [
    {
      turn: nextState.turn,
      player: playerId,
      phase: 'TaxCheck',
      summary: `Roll d10 ${roll.die1}/${roll.die2} (income ${incomeResult})`,
    },
    ...taxEntries,
    ...incomeSummaries.map((summary) => ({
      turn: nextState.turn,
      player: playerId,
      phase: 'CollectIncome' as const,
      summary,
    })),
  ];
}

function cycleOwner(
  previousState: GameState,
  nextState: GameState
): PlayerId | undefined {
  if (nextState.incomeChoiceReturnPlayerId) {
    return nextState.incomeChoiceReturnPlayerId;
  }
  const nextActive = nextState.players[nextState.activePlayerIndex];
  if (nextActive) {
    return nextActive.id;
  }
  return previousState.players[previousState.activePlayerIndex]?.id;
}

function summarizeTax(
  nextState: GameState,
  taxLossByPlayer: Map<PlayerId, number>
): string | null {
  const taxSuit = nextState.lastTaxSuit;
  if (!taxSuit) {
    return null;
  }

  const losses = nextState.players.map((player) => ({
    playerId: player.id,
    loss: taxLossByPlayer.get(player.id) ?? 0,
  }));
  const totalLoss = losses.reduce((sum, entry) => sum + entry.loss, 0);
  if (totalLoss <= 0) {
    return `Tax ${taxSuit} (no losses)`;
  }

  const details = losses
    .filter((entry) => entry.loss > 0)
    .map((entry) => `${entry.playerId} -${entry.loss}`)
    .join(', ');
  return `Tax ${taxSuit} (${details})`;
}

function summarizeIncome(
  nextState: GameState,
  postTaxResources: Map<PlayerId, ResourcePool>
): string[] {
  const byPlayer = nextState.players.map((player) => {
    const baseline = postTaxResources.get(player.id) ?? player.resources;
    const delta = resourceDelta(baseline, player.resources);
    return `Income ${player.id} ${formatDelta(delta)}`;
  });

  const pendingChoiceCount = nextState.pendingIncomeChoices?.length ?? 0;
  if (pendingChoiceCount > 0) {
    byPlayer.push(`Income pending choices ${pendingChoiceCount}`);
  }

  return byPlayer;
}

function resourceDelta(
  baseline: ResourcePool,
  next: ResourcePool
): Partial<Record<Suit, number>> {
  const delta: Partial<Record<Suit, number>> = {};
  for (const suit of SUIT_ORDER) {
    const value = next[suit] - baseline[suit];
    if (value !== 0) {
      delta[suit] = value;
    }
  }
  return delta;
}

function formatDelta(delta: Partial<Record<Suit, number>>): string {
  const parts: string[] = [];
  for (const suit of SUIT_ORDER) {
    const count = delta[suit] ?? 0;
    if (count === 0) {
      continue;
    }
    const sign = count > 0 ? '+' : '';
    parts.push(`${sign}${count} ${suit}`);
  }
  if (parts.length === 0) {
    return 'none';
  }
  return parts.join(', ');
}
