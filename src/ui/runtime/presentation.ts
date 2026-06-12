import type {
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../../engine/types';
import type {
  AnimationOverlayState,
  GameTransaction,
  PresentationTimeline,
  PresentationTimelineEvent,
} from './types';

export type PresentationSnapshot = {
  viewState: GameState;
  overlays: AnimationOverlayState;
};

export type DerivePresentationSnapshotOptions = {
  transaction: GameTransaction;
  timeline: PresentationTimeline;
  elapsedMs: number;
};

const EMPTY_OVERLAYS: AnimationOverlayState = {
  resourceFlights: [],
  cardFlights: [],
  incomeHighlightCardIds: [],
  incomeHighlightCrowns: [],
  activePlayerHighlightOverride: null,
};

export function derivePresentationSnapshot({
  transaction,
  timeline,
  elapsedMs,
}: DerivePresentationSnapshotOptions): PresentationSnapshot {
  const commitEvent = timeline.events.find(
    (event) => event.type === 'commit-view-to-next-state'
  );
  if (commitEvent && elapsedMs >= commitEvent.atMs) {
    return {
      viewState: transaction.nextState,
      overlays: EMPTY_OVERLAYS,
    };
  }

  let viewState = cloneGameState(transaction.previousState);
  let overlays = initialOverlays(transaction);
  for (const event of timeline.events) {
    if (event.atMs > elapsedMs) {
      break;
    }
    const updated = applyTimelineEvent(viewState, overlays, transaction, event);
    viewState = updated.viewState;
    overlays = updated.overlays;
  }

  return { viewState, overlays };
}

function applyTimelineEvent(
  viewState: GameState,
  overlays: AnimationOverlayState,
  transaction: GameTransaction,
  event: PresentationTimelineEvent
): PresentationSnapshot {
  switch (event.type) {
    case 'hold-previous-state':
      return { viewState, overlays };
    case 'reveal-drawn-card':
      return {
        viewState: revealDrawnCard(viewState, transaction.nextState, event),
        overlays: {
          ...overlays,
          activePlayerHighlightOverride: null,
        },
      };
    case 'show-income-roll':
      return {
        viewState: revealIncomeRoll(viewState, transaction.nextState, event),
        overlays,
      };
    case 'apply-tax-token-loss':
      return {
        viewState: applyResourceDelta(viewState, event.event.playerId, {
          [event.event.suit]: -1,
        }),
        overlays,
      };
    case 'launch-income-token-flight':
      return {
        viewState,
        overlays,
      };
    case 'show-income-highlights':
      return {
        viewState,
        overlays: {
          ...overlays,
          incomeHighlightCardIds: event.cardIds,
          incomeHighlightCrowns: event.crowns,
        },
      };
    case 'clear-income-highlights':
      return {
        viewState,
        overlays: {
          ...overlays,
          incomeHighlightCardIds: [],
          incomeHighlightCrowns: [],
        },
      };
    case 'apply-income-token-gain':
      return {
        viewState: applyResourceDelta(viewState, event.event.playerId, {
          [event.event.suit]: 1,
        }),
        overlays,
      };
    case 'reveal-income-choice-request':
      return {
        viewState: {
          ...viewState,
          phase: 'CollectIncome',
          pendingIncomeChoices: event.event.choices,
          incomeChoiceReturnPlayerId:
            transaction.nextState.incomeChoiceReturnPlayerId,
        },
        overlays,
      };
    case 'reveal-income-choice-submission':
      return {
        viewState: {
          ...viewState,
          submittedIncomeChoices: [
            ...(viewState.submittedIncomeChoices ?? []),
            {
              playerId: event.event.playerId,
              districtId: event.event.districtId,
              cardId: event.event.cardId,
              suit: event.event.suit,
            },
          ],
        },
        overlays,
      };
    case 'commit-view-to-next-state':
      return {
        viewState: transaction.nextState,
        overlays: EMPTY_OVERLAYS,
      };
  }
}

function initialOverlays(transaction: GameTransaction): AnimationOverlayState {
  const hasDraw = transaction.events.some(
    (event) => event.type === 'draw-card'
  );
  return {
    ...EMPTY_OVERLAYS,
    activePlayerHighlightOverride: hasDraw ? transaction.actingPlayerId : null,
  };
}

function revealDrawnCard(
  viewState: GameState,
  nextState: GameState,
  event: Extract<PresentationTimelineEvent, { type: 'reveal-drawn-card' }>
): GameState {
  return {
    ...viewState,
    deck: nextState.deck,
    players: viewState.players.map((player) =>
      player.id === event.event.playerId
        ? {
            ...player,
            hand: player.hand.includes(event.event.cardId)
              ? player.hand
              : [...player.hand, event.event.cardId],
          }
        : player
    ),
  };
}

function revealIncomeRoll(
  viewState: GameState,
  nextState: GameState,
  event: Extract<PresentationTimelineEvent, { type: 'show-income-roll' }>
): GameState {
  return {
    ...viewState,
    activePlayerIndex: nextState.activePlayerIndex,
    turn: nextState.turn,
    lastIncomeRoll: event.event.roll,
    lastTaxSuit: nextState.lastTaxSuit,
  };
}

function applyResourceDelta(
  state: GameState,
  playerId: PlayerId,
  delta: Partial<Record<Suit, number>>
): GameState {
  return {
    ...state,
    players: state.players.map((player) =>
      player.id === playerId
        ? {
            ...player,
            resources: applyDeltaToResources(player.resources, delta),
          }
        : player
    ),
  };
}

function applyDeltaToResources(
  resources: ResourcePool,
  delta: Partial<Record<Suit, number>>
): ResourcePool {
  return {
    ...resources,
    Moons: Math.max(0, resources.Moons + (delta.Moons ?? 0)),
    Suns: Math.max(0, resources.Suns + (delta.Suns ?? 0)),
    Waves: Math.max(0, resources.Waves + (delta.Waves ?? 0)),
    Leaves: Math.max(0, resources.Leaves + (delta.Leaves ?? 0)),
    Wyrms: Math.max(0, resources.Wyrms + (delta.Wyrms ?? 0)),
    Knots: Math.max(0, resources.Knots + (delta.Knots ?? 0)),
  };
}

function cloneGameState(state: GameState): GameState {
  return {
    ...state,
    deck: {
      ...state.deck,
      draw: [...state.deck.draw],
      discard: [...state.deck.discard],
    },
    players: state.players.map((player) => ({
      ...player,
      hand: [...player.hand],
      crowns: [...player.crowns],
      resources: { ...player.resources },
    })),
    districts: state.districts.map((district) => ({
      ...district,
      markerSuitMask: [...district.markerSuitMask],
      stacks: {
        PlayerA: cloneDistrictStack(district.stacks.PlayerA),
        PlayerB: cloneDistrictStack(district.stacks.PlayerB),
      },
    })),
    log: [...state.log],
    pendingIncomeChoices: state.pendingIncomeChoices?.map((choice) => ({
      ...choice,
      suits: [...choice.suits],
    })),
    submittedIncomeChoices: state.submittedIncomeChoices?.map((choice) => ({
      ...choice,
    })),
  };
}

function cloneDistrictStack(
  stack: GameState['districts'][number]['stacks'][PlayerId]
): GameState['districts'][number]['stacks'][PlayerId] {
  return {
    developed: [...stack.developed],
    deed: stack.deed
      ? {
          ...stack.deed,
          tokens: { ...stack.deed.tokens },
        }
      : undefined,
  };
}
