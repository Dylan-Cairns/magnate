import type {
  GameAction,
  GameLogEntry,
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../engine/types';
import {
  deriveTurnCycleEvents,
  type TurnCycleTaxSummary,
} from './turnCycleEvents';

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

  return [...engineEntries, ...resolveTurnCycleEntries(previousState, nextState, action)];
}

function resolveTurnCycleEntries(
  previousState: GameState,
  nextState: GameState,
  action: Extract<GameAction, { type: 'end-turn' }>
): GameLogEntry[] {
  const cycle = deriveTurnCycleEvents(previousState, nextState, action);
  if (!cycle) {
    return [];
  }

  const postTaxResources = postTaxResourcesByPlayer(previousState, cycle.tax);
  const taxSummary = summarizeTax(cycle.tax);
  const incomeSummaries = summarizeIncome(
    nextState,
    postTaxResources,
    cycle.pendingChoices.length
  );

  const taxEntries: GameLogEntry[] = [];
  if (taxSummary) {
    taxEntries.push({
      turn: nextState.turn,
      player: cycle.cycleOwner,
      phase: 'TaxCheck',
      summary: taxSummary,
    });
  }

  return [
    {
      turn: nextState.turn,
      player: cycle.cycleOwner,
      phase: 'TaxCheck',
      summary: `Roll d10 ${cycle.roll.die1}/${cycle.roll.die2} (income ${cycle.incomeRank})`,
    },
    ...taxEntries,
    ...incomeSummaries.map((summary) => ({
      turn: nextState.turn,
      player: cycle.cycleOwner,
      phase: 'CollectIncome' as const,
      summary,
    })),
  ];
}

function postTaxResourcesByPlayer(
  previousState: GameState,
  tax: TurnCycleTaxSummary | null
): Map<PlayerId, ResourcePool> {
  const resources = new Map<PlayerId, ResourcePool>();
  for (const player of previousState.players) {
    if (!tax) {
      resources.set(player.id, player.resources);
      continue;
    }
    resources.set(player.id, {
      ...player.resources,
      [tax.suit]: Math.min(player.resources[tax.suit], 1),
    });
  }
  return resources;
}

function summarizeTax(tax: TurnCycleTaxSummary | null): string | null {
  if (!tax) {
    return null;
  }

  const totalLoss = tax.lossesByPlayer.reduce((sum, entry) => sum + entry.count, 0);
  if (totalLoss <= 0) {
    return `Tax ${tax.suit} (no losses)`;
  }

  const details = tax.lossesByPlayer
    .filter((entry) => entry.count > 0)
    .map((entry) => `${entry.playerId} -${entry.count}`)
    .join(', ');
  return `Tax ${tax.suit} (${details})`;
}

function summarizeIncome(
  nextState: GameState,
  postTaxResources: Map<PlayerId, ResourcePool>,
  pendingChoiceCount: number
): string[] {
  const byPlayer = nextState.players.map((player) => {
    const baseline = postTaxResources.get(player.id) ?? player.resources;
    const delta = resourceDelta(baseline, player.resources);
    return `Income ${player.id} ${formatDelta(delta)}`;
  });

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
