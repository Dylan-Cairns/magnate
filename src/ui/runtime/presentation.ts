import type {
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../../engine/types';
import type { CardId } from '../../engine/cards';
import type { AnimationOverlayState, GameTransaction } from './types';
import type {
  AnimationSequence,
  ScheduledAnimationStep,
} from './animationSequence';

export type PresentationSnapshot = {
  viewState: GameState;
  overlays: AnimationOverlayState;
};

export type DerivePresentationSnapshotFromSequenceOptions = {
  transaction: GameTransaction;
  sequence: AnimationSequence;
  elapsedMs: number;
};

const EMPTY_OVERLAYS: AnimationOverlayState = {
  incomeHighlightCardIds: [],
  incomeHighlightCrowns: [],
  activePlayerHighlightOverride: null,
};

export function derivePresentationSnapshotFromSequence({
  transaction,
  sequence,
  elapsedMs,
}: DerivePresentationSnapshotFromSequenceOptions): PresentationSnapshot {
  const commitStep = sequence.steps.find(
    (step) => step.type === 'commit-view-state'
  );
  if (commitStep && elapsedMs >= commitStep.startMs) {
    return {
      viewState: transaction.nextState,
      overlays: EMPTY_OVERLAYS,
    };
  }

  let viewState = cloneGameState(transaction.previousState);
  let overlays = initialOverlays(transaction);
  for (const step of sequence.steps) {
    if (step.startMs > elapsedMs) {
      break;
    }
    const updated = applySequenceStep(
      viewState,
      overlays,
      transaction,
      step,
      elapsedMs
    );
    viewState = updated.viewState;
    overlays = updated.overlays;
  }

  return { viewState, overlays };
}

function applySequenceStep(
  viewState: GameState,
  overlays: AnimationOverlayState,
  transaction: GameTransaction,
  step: ScheduledAnimationStep,
  elapsedMs: number
): PresentationSnapshot {
  switch (step.type) {
    case 'hold-previous-state':
    case 'pulse-income-die':
    case 'pulse-tax-die':
    case 'hold-before-tax-flights':
    case 'launch-tax-token-flights':
    case 'stage-gap':
    case 'hold-before-income-flights':
    case 'launch-income-token-flights':
    case 'launch-payment-token-flights':
    case 'apply-resource-payment':
    case 'launch-card-to-district-flight':
    case 'place-card-in-district':
    case 'launch-deed-token-flights':
    case 'apply-deed-progress':
    case 'reveal-deed-completion':
    case 'apply-sell-resource-gains':
    case 'apply-trade-resources':
      return { viewState, overlays };
    case 'draw-card-flight':
      if (elapsedMs < step.endMs) {
        return { viewState, overlays };
      }
      return {
        viewState: revealDrawnCard(viewState, transaction.nextState, step),
        overlays: {
          ...overlays,
          activePlayerHighlightOverride: null,
        },
      };
    case 'stage-sold-card':
      return {
        viewState: stageSoldCard(viewState, transaction.nextState),
        overlays,
      };
    case 'roll-income-dice':
      return {
        viewState: revealIncomeRoll(viewState, transaction.nextState, step),
        overlays: {
          ...overlays,
          activePlayerHighlightOverride: null,
        },
      };
    case 'roll-tax-die':
      return {
        viewState: {
          ...viewState,
          lastTaxSuit: step.suit,
        },
        overlays,
      };
    case 'apply-tax-losses':
      return {
        viewState: applyResourceDeltas(
          viewState,
          step.losses.map((event) => ({
            playerId: event.playerId,
            delta: { [event.suit]: -1 },
          }))
        ),
        overlays,
      };
    case 'highlight-income-sources':
      return {
        viewState,
        overlays: {
          ...overlays,
          incomeHighlightCardIds: step.cardIds,
          incomeHighlightCrowns: step.crowns,
        },
      };
    case 'apply-income-gains':
      return {
        viewState: applyResourceDeltas(
          viewState,
          step.gains.map((event) => ({
            playerId: event.playerId,
            delta: { [event.suit]: 1 },
          }))
        ),
        overlays,
      };
    case 'post-income-hold':
      return {
        viewState,
        overlays: {
          ...overlays,
          incomeHighlightCardIds: [],
          incomeHighlightCrowns: [],
        },
      };
    case 'reveal-income-choice-request':
      return {
        viewState: {
          ...viewState,
          phase: 'CollectIncome',
          pendingIncomeChoices: step.choices,
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
              playerId: step.event.playerId,
              districtId: step.event.districtId,
              cardId: step.event.cardId,
              suit: step.event.suit,
            },
          ],
        },
        overlays,
      };
    case 'commit-view-state':
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

function stageSoldCard(viewState: GameState, nextState: GameState): GameState {
  return {
    ...nextState,
    deck: {
      ...nextState.deck,
      discard: viewState.deck.discard,
    },
  };
}

function revealDrawnCard(
  viewState: GameState,
  nextState: GameState,
  event: { playerId: PlayerId; cardId: CardId }
): GameState {
  return {
    ...viewState,
    deck: nextState.deck,
    players: viewState.players.map((player) =>
      player.id === event.playerId
        ? {
            ...player,
            hand: player.hand.includes(event.cardId)
              ? player.hand
              : [...player.hand, event.cardId],
          }
        : player
    ),
  };
}

function revealIncomeRoll(
  viewState: GameState,
  nextState: GameState,
  event: { roll: GameState['lastIncomeRoll'] }
): GameState {
  return {
    ...viewState,
    activePlayerIndex: nextState.activePlayerIndex,
    turn: nextState.turn,
    lastIncomeRoll: event.roll,
    lastTaxSuit: nextState.lastTaxSuit,
  };
}

function applyResourceDeltas(
  state: GameState,
  deltas: readonly {
    playerId: PlayerId;
    delta: Partial<Record<Suit, number>>;
  }[]
): GameState {
  return deltas.reduce(
    (updated, entry) =>
      applyResourceDelta(updated, entry.playerId, entry.delta),
    state
  );
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
