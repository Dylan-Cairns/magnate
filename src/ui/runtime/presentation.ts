import type {
  DistrictStack,
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../../engine/types';
import type { CardId } from '../../engine/cards';
import type {
  AnimationOverlayState,
  DiceVisualState,
  GameTransaction,
} from './types';
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
  dice: null,
};

const SUITS: readonly Suit[] = [
  'Moons',
  'Suns',
  'Waves',
  'Leaves',
  'Wyrms',
  'Knots',
];

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
    case 'launch-tax-token-flights':
    case 'stage-gap':
    case 'launch-income-token-flights':
    case 'launch-payment-token-flights':
    case 'launch-trade-token-flights':
    case 'launch-card-to-district-flight':
    case 'launch-deed-token-flights':
      return { viewState, overlays };
    case 'apply-resource-payment':
      return {
        viewState: applyResourcePayment(viewState, step.event),
        overlays,
      };
    case 'apply-resource-payment-token':
      return {
        viewState: applyResourceDelta(viewState, step.playerId, {
          [step.suit]: -1,
        }),
        overlays,
      };
    case 'apply-deed-tokens':
      return {
        viewState: applyDeedTokens(viewState, step.tokens),
        overlays,
      };
    case 'place-card-in-district':
      return {
        viewState: placeCardInDistrict(viewState, step.event),
        overlays,
      };
    case 'apply-deed-progress':
      return {
        viewState: applyDeedProgress(viewState, step.event),
        overlays,
      };
    case 'reveal-deed-completion':
      return {
        viewState: revealDeedCompletion(viewState, step.event),
        overlays,
      };
    case 'apply-sell-resource-gains':
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
    case 'apply-trade-token-loss':
      return {
        viewState: applyResourceDelta(viewState, step.playerId, {
          [step.suit]: -1,
        }),
        overlays,
      };
    case 'apply-trade-token-gain':
      return {
        viewState: applyResourceDelta(viewState, step.event.playerId, {
          [step.event.receive]: step.event.receiveCount,
        }),
        overlays,
      };
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
        viewState: stageSoldCard(viewState, step),
        overlays,
      };
    case 'roll-income-dice':
      return {
        viewState: revealIncomeRoll(viewState, step),
        overlays: {
          ...overlays,
          activePlayerHighlightOverride: null,
          dice: {
            incomeRoll: step.roll,
            taxSuit: undefined,
            incomePhase: 'rolling',
            taxPhase: 'hidden',
          },
        },
      };
    case 'roll-tax-die':
      return {
        viewState: {
          ...viewState,
          lastTaxSuit: step.suit,
        },
        overlays: {
          ...overlays,
          dice: updateDiceVisualState(overlays.dice, {
            taxSuit: step.suit,
            incomePhase: 'settled',
            taxPhase: 'rolling',
          }),
        },
      };
    case 'hold-before-tax-flights':
      return {
        viewState,
        overlays: {
          ...overlays,
          dice: updateDiceVisualState(overlays.dice, {
            taxSuit: overlays.dice?.taxSuit,
            incomePhase: 'settled',
            taxPhase: 'settled',
          }),
        },
      };
    case 'hold-before-income-flights':
      return {
        viewState,
        overlays: {
          ...overlays,
          dice: updateDiceVisualState(overlays.dice, {
            taxSuit: overlays.dice?.taxSuit,
            incomePhase: 'settled',
            taxPhase: overlays.dice?.taxPhase ?? 'hidden',
          }),
        },
      };
    case 'apply-tax-token-loss':
      return {
        viewState: applyResourceDelta(viewState, step.loss.playerId, {
          [step.loss.suit]: -1,
        }),
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
          incomeChoiceReturnPlayerId: step.returnPlayerId,
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

function updateDiceVisualState(
  current: DiceVisualState | null,
  update: Pick<DiceVisualState, 'taxSuit' | 'incomePhase' | 'taxPhase'>
): DiceVisualState | null {
  if (!current) {
    return null;
  }
  return {
    ...current,
    ...update,
  };
}

function stageSoldCard(
  viewState: GameState,
  event: { playerId: PlayerId; cardId: CardId }
): GameState {
  return {
    ...viewState,
    phase: 'ActionWindow',
    cardPlayedThisTurn: true,
    players: viewState.players.map((player) =>
      player.id === event.playerId
        ? {
            ...player,
            hand: player.hand.filter((cardId) => cardId !== event.cardId),
          }
        : player
    ),
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

function applyResourcePayment(
  state: GameState,
  event: Extract<
    GameTransaction['events'][number],
    { type: 'resource-payment-applied' }
  >
): GameState {
  return applyResourceDelta(
    state,
    event.playerId,
    negateResourceDelta(event.payment)
  );
}

function placeCardInDistrict(
  state: GameState,
  event: Extract<
    GameTransaction['events'][number],
    { type: 'card-played-to-district' }
  >
): GameState {
  const currentStack = districtStackFor(
    state,
    event.districtId,
    event.playerId
  );
  if (!currentStack) {
    return state;
  }
  const nextStack =
    event.placement === 'deed'
      ? {
          developed: [...currentStack.developed],
          deed: {
            cardId: event.cardId,
            progress: 0,
            tokens: {},
          },
        }
      : {
          developed: [...currentStack.developed, event.cardId],
          deed: currentStack.deed
            ? { ...currentStack.deed, tokens: { ...currentStack.deed.tokens } }
            : undefined,
        };

  return {
    ...state,
    phase: 'ActionWindow',
    cardPlayedThisTurn: true,
    players: state.players.map((player) =>
      player.id === event.playerId
        ? {
            ...player,
            hand: player.hand.filter((cardId) => cardId !== event.cardId),
          }
        : player
    ),
    districts: replaceDistrictStack(
      state,
      event.districtId,
      event.playerId,
      cloneDistrictStack(nextStack)
    ),
  };
}

function applyDeedProgress(
  state: GameState,
  event: Extract<
    GameTransaction['events'][number],
    { type: 'deed-progress-applied' }
  >
): GameState {
  const currentStack = districtStackFor(state, event.districtId, event.playerId);
  if (!currentStack?.deed) {
    return state;
  }

  return {
    ...state,
    districts: replaceDistrictStack(state, event.districtId, event.playerId, {
      developed: [...currentStack.developed],
      deed: {
        cardId: event.cardId,
        progress: event.nextProgress,
        tokens: { ...currentStack.deed.tokens },
      },
    }),
  };
}

function applyDeedTokens(
  state: GameState,
  tokens: readonly Extract<
    GameTransaction['events'][number],
    { type: 'deed-token-paid' }
  >[]
): GameState {
  if (tokens.length === 0) {
    return state;
  }
  const first = tokens[0];
  const currentStack = districtStackFor(
    state,
    first.districtId,
    first.playerId
  );
  if (!currentStack?.deed) {
    return state;
  }

  return {
    ...state,
    districts: replaceDistrictStack(state, first.districtId, first.playerId, {
      developed: [...currentStack.developed],
      deed: {
        cardId: first.cardId,
        progress: currentStack.deed.progress,
        tokens: mergeResourceTokens(
          currentStack.deed.tokens,
          deedTokenPayment(tokens)
        ),
      },
    }),
  };
}

function revealDeedCompletion(
  state: GameState,
  event: Extract<GameTransaction['events'][number], { type: 'deed-completed' }>
): GameState {
  const currentStack = districtStackFor(
    state,
    event.districtId,
    event.playerId
  );
  if (!currentStack) {
    return state;
  }

  return {
    ...state,
    districts: replaceDistrictStack(
      state,
      event.districtId,
      event.playerId,
      {
        developed: [...currentStack.developed, event.cardId],
        deed: undefined,
      }
    ),
  };
}

function revealIncomeRoll(
  viewState: GameState,
  event: { playerId: PlayerId; turn: number; roll: GameState['lastIncomeRoll'] }
): GameState {
  return {
    ...viewState,
    activePlayerIndex: playerIndexFor(viewState, event.playerId),
    turn: event.turn,
    lastIncomeRoll: event.roll,
    lastTaxSuit: undefined,
  };
}

function playerIndexFor(state: GameState, playerId: PlayerId): number {
  const index = state.players.findIndex((player) => player.id === playerId);
  return index >= 0 ? index : state.activePlayerIndex;
}

function districtStackFor(
  state: GameState,
  districtId: string,
  playerId: PlayerId
): DistrictStack | undefined {
  return state.districts.find((district) => district.id === districtId)?.stacks[
    playerId
  ];
}

function replaceDistrictStack(
  state: GameState,
  districtId: string,
  playerId: PlayerId,
  stack: DistrictStack
): GameState['districts'] {
  return state.districts.map((district) =>
    district.id === districtId
      ? {
          ...district,
          stacks: {
            ...district.stacks,
            [playerId]: stack,
          },
        }
      : district
  );
}

function deedTokenPayment(
  deedTokens: readonly Extract<
    GameTransaction['events'][number],
    { type: 'deed-token-paid' }
  >[]
): Partial<Record<Suit, number>> {
  const tokens: Partial<Record<Suit, number>> = {};
  for (const token of deedTokens) {
    tokens[token.suit] = (tokens[token.suit] ?? 0) + 1;
  }
  return tokens;
}

function negateResourceDelta(
  resources: Partial<Record<Suit, number>>
): Partial<Record<Suit, number>> {
  const delta: Partial<Record<Suit, number>> = {};
  for (const [suit, count] of Object.entries(resources) as Array<
    [Suit, number]
  >) {
    delta[suit] = -count;
  }
  return delta;
}

function mergeResourceTokens(
  left: Partial<Record<Suit, number>>,
  right: Partial<Record<Suit, number>>
): Partial<Record<Suit, number>> {
  const merged: Partial<Record<Suit, number>> = {};
  for (const suit of SUITS) {
    const count = (left[suit] ?? 0) + (right[suit] ?? 0);
    if (count > 0) {
      merged[suit] = count;
    }
  }
  return merged;
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
