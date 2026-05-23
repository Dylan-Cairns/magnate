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
import { playerDisplayName } from './playerDisplay';

const SUIT_ORDER: readonly Suit[] = [
  'Moons',
  'Suns',
  'Waves',
  'Leaves',
  'Wyrms',
  'Knots',
];

export interface DeferredIncomeLogContext {
  turn: number;
  player: PlayerId;
  postTaxResourcesByPlayer: ReadonlyArray<{
    playerId: PlayerId;
    resources: ResourcePool;
  }>;
}

export interface TimelineLogUpdate {
  entries: GameLogEntry[];
  deferredIncomeLogContext: DeferredIncomeLogContext | null;
}

export function transitionLogEntries(
  previousState: GameState,
  nextState: GameState,
  action: GameAction | undefined,
  humanPlayerId: PlayerId
): GameLogEntry[] {
  return transitionLogUpdate(
    previousState,
    nextState,
    action,
    humanPlayerId,
    null
  ).entries;
}

export function transitionLogUpdate(
  previousState: GameState,
  nextState: GameState,
  action: GameAction | undefined,
  humanPlayerId: PlayerId,
  deferredIncomeLogContext: DeferredIncomeLogContext | null
): TimelineLogUpdate {
  const startIndex =
    nextState.log.length >= previousState.log.length
      ? previousState.log.length
      : 0;
  const engineEntries = nextState.log.slice(startIndex);

  if (!action) {
    return {
      entries: engineEntries,
      deferredIncomeLogContext: null,
    };
  }

  if (action.type === 'end-turn') {
    const resolved = resolveTurnCycleEntries(
      previousState,
      nextState,
      action,
      humanPlayerId
    );
    return {
      entries: [...engineEntries, ...resolved.entries],
      deferredIncomeLogContext: resolved.deferredIncomeLogContext,
    };
  }

  if (action.type === 'choose-income-suit') {
    const resolved = resolveDeferredIncomeEntries(
      nextState,
      deferredIncomeLogContext,
      humanPlayerId
    );
    return {
      entries: [...engineEntries, ...resolved.entries],
      deferredIncomeLogContext: resolved.deferredIncomeLogContext,
    };
  }

  return {
    entries: engineEntries,
    deferredIncomeLogContext:
      nextState.phase === 'CollectIncome' &&
      (nextState.pendingIncomeChoices?.length ?? 0) > 0
        ? deferredIncomeLogContext
        : null,
  };
}

function resolveTurnCycleEntries(
  previousState: GameState,
  nextState: GameState,
  action: Extract<GameAction, { type: 'end-turn' }>,
  humanPlayerId: PlayerId
): TimelineLogUpdate {
  const cycle = deriveTurnCycleEvents(previousState, nextState, action);
  if (!cycle) {
    return {
      entries: [],
      deferredIncomeLogContext: null,
    };
  }

  const postTaxResources = postTaxResourcesByPlayer(previousState, cycle.tax);
  const taxSummary = summarizeTax(cycle.tax, humanPlayerId);

  const taxEntries: GameLogEntry[] = [];
  if (taxSummary) {
    taxEntries.push({
      turn: nextState.turn,
      player: cycle.cycleOwner,
      phase: 'TaxCheck',
      summary: taxSummary,
    });
  }

  const rollAndTaxEntries: GameLogEntry[] = [
    {
      turn: nextState.turn,
      player: cycle.cycleOwner,
      phase: 'TaxCheck',
      summary: `Roll d10 ${cycle.roll.die1}/${cycle.roll.die2} (income ${cycle.incomeRank})`,
    },
    ...taxEntries,
  ];

  if (cycle.pendingChoices.length > 0) {
    return {
      entries: rollAndTaxEntries,
      deferredIncomeLogContext: {
        turn: nextState.turn,
        player: cycle.cycleOwner,
        postTaxResourcesByPlayer: resourceMapToEntries(postTaxResources),
      },
    };
  }

  return {
    entries: [
      ...rollAndTaxEntries,
      ...incomeEntries(
        nextState,
        cycle.cycleOwner,
        postTaxResources,
        humanPlayerId
      ),
    ],
    deferredIncomeLogContext: null,
  };
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

function summarizeTax(
  tax: TurnCycleTaxSummary | null,
  humanPlayerId: PlayerId
): string | null {
  if (!tax) {
    return null;
  }

  const totalLoss = tax.lossesByPlayer.reduce(
    (sum, entry) => sum + entry.count,
    0
  );
  if (totalLoss <= 0) {
    return `Tax ${tax.suit} (no losses)`;
  }

  const details = tax.lossesByPlayer
    .filter((entry) => entry.count > 0)
    .map(
      (entry) =>
        `${playerDisplayName(entry.playerId, humanPlayerId)} -${entry.count}`
    )
    .join(', ');
  return `Tax ${tax.suit} (${details})`;
}

function summarizeIncome(
  nextState: GameState,
  postTaxResources: Map<PlayerId, ResourcePool>,
  humanPlayerId: PlayerId
): string[] {
  return nextState.players.map((player) => {
    const baseline = postTaxResources.get(player.id) ?? player.resources;
    const delta = resourceDelta(baseline, player.resources);
    return `Income ${playerDisplayName(player.id, humanPlayerId)} ${formatDelta(delta)}`;
  });
}

function resolveDeferredIncomeEntries(
  nextState: GameState,
  context: DeferredIncomeLogContext | null,
  humanPlayerId: PlayerId
): TimelineLogUpdate {
  if (!context) {
    return {
      entries: [],
      deferredIncomeLogContext: null,
    };
  }

  if (
    nextState.phase === 'CollectIncome' &&
    (nextState.pendingIncomeChoices?.length ?? 0) > 0
  ) {
    return {
      entries: [],
      deferredIncomeLogContext: context,
    };
  }

  if (nextState.turn !== context.turn) {
    return {
      entries: [],
      deferredIncomeLogContext: null,
    };
  }

  return {
    entries: incomeEntries(
      nextState,
      context.player,
      resourceEntriesToMap(context.postTaxResourcesByPlayer),
      humanPlayerId
    ),
    deferredIncomeLogContext: null,
  };
}

function incomeEntries(
  nextState: GameState,
  player: PlayerId,
  postTaxResources: Map<PlayerId, ResourcePool>,
  humanPlayerId: PlayerId
): GameLogEntry[] {
  return summarizeIncome(nextState, postTaxResources, humanPlayerId).map(
    (summary) => ({
      turn: nextState.turn,
      player,
      phase: 'CollectIncome' as const,
      summary,
    })
  );
}

function resourceMapToEntries(
  resources: Map<PlayerId, ResourcePool>
): DeferredIncomeLogContext['postTaxResourcesByPlayer'] {
  return [...resources.entries()].map(([playerId, pool]) => ({
    playerId,
    resources: { ...pool },
  }));
}

function resourceEntriesToMap(
  entries: DeferredIncomeLogContext['postTaxResourcesByPlayer']
): Map<PlayerId, ResourcePool> {
  return new Map(
    entries.map((entry) => [entry.playerId, { ...entry.resources }])
  );
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
