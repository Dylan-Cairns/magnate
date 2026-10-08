import { GamePhase, SUITS } from '../../engine/values';
import {
  AnimationStepType,
  DicePhase,
  GamePresentationEventType,
} from './values';
import {
  type DistrictStack,
  type GameState,
  type PlayerId,
  type ResourcePool,
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

export function derivePresentationSnapshotFromSequence({
  transaction,
  sequence,
  elapsedMs,
}: DerivePresentationSnapshotFromSequenceOptions): PresentationSnapshot {
  const commitStep = sequence.steps.find(
    (step) => step.type === AnimationStepType.CommitViewState
  );
  if (commitStep && elapsedMs >= commitStep.startMs) {
    return {
      viewState: transaction.nextState,
      overlays: EMPTY_OVERLAYS,
    };
  }

  // Immutable structural sharing: every step returns new objects only along the
  // branches it changes and reuses the rest of `previousState`. Cloning here
  // would give every district/player a fresh identity on every snapshot tick,
  // defeating memoization and forcing the whole board to restyle and re-render.
  let viewState = transaction.previousState;
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
    case AnimationStepType.HoldPreviousState:
    case AnimationStepType.LaunchTaxTokenFlights:
    case AnimationStepType.StageGap:
    case AnimationStepType.LaunchIncomeTokenFlights:
    case AnimationStepType.LaunchPaymentTokenFlights:
    case AnimationStepType.LaunchTradeTokenFlights:
    case AnimationStepType.LaunchCardToDistrictFlight:
    case AnimationStepType.LaunchDeedTokenFlights:
    case AnimationStepType.LaunchSellTokenFlights:
      return { viewState, overlays };
    case AnimationStepType.ApplyResourcePayment:
      return {
        viewState: applyResourcePayment(viewState, step.event),
        overlays,
      };
    case AnimationStepType.ApplyResourcePaymentToken:
      return {
        viewState: applyResourceDelta(viewState, step.playerId, {
          [step.suit]: -1,
        }),
        overlays,
      };
    case AnimationStepType.ApplyDeedTokens:
      return {
        viewState: applyDeedTokens(viewState, step.tokens),
        overlays,
      };
    case AnimationStepType.PlaceCardInDistrict:
      return {
        viewState: placeCardInDistrict(viewState, step.event),
        overlays,
      };
    case AnimationStepType.ApplyDeedProgress:
      return {
        viewState: applyDeedProgress(viewState, step.event),
        overlays,
      };
    case AnimationStepType.RevealDeedCompletion:
      return {
        viewState: revealDeedCompletion(viewState, step.event),
        overlays,
      };
    case AnimationStepType.LandSellToken:
      if (elapsedMs < step.endMs) {
        return { viewState, overlays };
      }
      return {
        viewState: applyResourceDelta(viewState, step.gain.playerId, {
          [step.gain.suit]: 1,
        }),
        overlays,
      };
    case AnimationStepType.ApplyTradeTokenLoss:
      return {
        viewState: applyResourceDelta(viewState, step.playerId, {
          [step.suit]: -1,
        }),
        overlays,
      };
    case AnimationStepType.LandTradeToken:
      return {
        viewState,
        overlays: {
          ...overlays,
          tradeProgress: {
            transactionId: transaction.id,
            playerId: step.event.playerId,
            suit: step.event.receive,
            landed: step.landed,
            total: step.event.giveCount,
          },
        },
      };
    case AnimationStepType.ApplyTradeTokenGain:
      return {
        viewState: applyResourceDelta(viewState, step.event.playerId, {
          [step.event.receive]: step.event.receiveCount,
        }),
        overlays: { ...overlays, tradeProgress: undefined },
      };
    case AnimationStepType.DrawCardFlight:
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
    case AnimationStepType.StageSoldCard:
      return {
        viewState: stageSoldCard(viewState, step),
        overlays,
      };
    case AnimationStepType.RollIncomeDice:
      return {
        viewState: revealIncomeRoll(viewState, step),
        overlays: {
          ...overlays,
          activePlayerHighlightOverride: null,
          dice: {
            incomeRoll: step.roll,
            taxSuit: undefined,
            incomePhase: DicePhase.Rolling,
            taxPhase: DicePhase.Hidden,
          },
        },
      };
    case AnimationStepType.RollTaxDie:
      return {
        viewState: {
          ...viewState,
          lastTaxSuit: step.suit,
        },
        overlays: {
          ...overlays,
          dice: updateDiceVisualState(overlays.dice, {
            taxSuit: step.suit,
            incomePhase: DicePhase.Settled,
            taxPhase: DicePhase.Rolling,
          }),
        },
      };
    case AnimationStepType.HoldBeforeTaxFlights:
      return {
        viewState,
        overlays: {
          ...overlays,
          dice: updateDiceVisualState(overlays.dice, {
            taxSuit: overlays.dice?.taxSuit,
            incomePhase: DicePhase.Settled,
            taxPhase: DicePhase.Settled,
          }),
        },
      };
    case AnimationStepType.HoldBeforeIncomeFlights:
      return {
        viewState,
        overlays: {
          ...overlays,
          dice: updateDiceVisualState(overlays.dice, {
            taxSuit: overlays.dice?.taxSuit,
            incomePhase: DicePhase.Settled,
            taxPhase: overlays.dice?.taxPhase ?? DicePhase.Hidden,
          }),
        },
      };
    case AnimationStepType.ApplyTaxTokenLoss:
      return {
        viewState: applyResourceDelta(viewState, step.loss.playerId, {
          [step.loss.suit]: -1,
        }),
        overlays,
      };
    case AnimationStepType.HighlightIncomeSources:
      return {
        viewState,
        overlays: {
          ...overlays,
          incomeHighlightCardIds: step.cardIds,
          incomeHighlightCrowns: step.crowns,
        },
      };
    case AnimationStepType.LandIncomeToken:
      if (elapsedMs < step.endMs) {
        return { viewState, overlays };
      }
      return {
        viewState: applyResourceDelta(viewState, step.gain.playerId, {
          [step.gain.suit]: 1,
        }),
        overlays,
      };
    case AnimationStepType.PostIncomeHold:
      return {
        viewState,
        overlays: {
          ...overlays,
          incomeHighlightCardIds: [],
          incomeHighlightCrowns: [],
        },
      };
    case AnimationStepType.RevealIncomeChoiceRequest:
      return {
        viewState: {
          ...viewState,
          phase: GamePhase.CollectIncome,
          pendingIncomeChoices: step.choices,
          incomeChoiceReturnPlayerId: step.returnPlayerId,
        },
        overlays,
      };
    case AnimationStepType.RevealIncomeChoiceSubmission:
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
    case AnimationStepType.CommitViewState:
      return {
        viewState: transaction.nextState,
        overlays: EMPTY_OVERLAYS,
      };
  }
}

function initialOverlays(transaction: GameTransaction): AnimationOverlayState {
  const hasDraw = transaction.events.some(
    (event) => event.type === GamePresentationEventType.DrawCard
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
    phase: GamePhase.ActionWindow,
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
    { type: typeof GamePresentationEventType.ResourcePaymentApplied }
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
    { type: typeof GamePresentationEventType.CardPlayedToDistrict }
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
    phase: GamePhase.ActionWindow,
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
    { type: typeof GamePresentationEventType.DeedProgressApplied }
  >
): GameState {
  const currentStack = districtStackFor(
    state,
    event.districtId,
    event.playerId
  );
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
    { type: typeof GamePresentationEventType.DeedTokenPaid }
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
  event: Extract<
    GameTransaction['events'][number],
    { type: typeof GamePresentationEventType.DeedCompleted }
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

  return {
    ...state,
    districts: replaceDistrictStack(state, event.districtId, event.playerId, {
      developed: [...currentStack.developed, event.cardId],
      deed: undefined,
    }),
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
    { type: typeof GamePresentationEventType.DeedTokenPaid }
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
    [Suit.Moons]: Math.max(0, resources[Suit.Moons] + (delta[Suit.Moons] ?? 0)),
    [Suit.Suns]: Math.max(0, resources[Suit.Suns] + (delta[Suit.Suns] ?? 0)),
    [Suit.Waves]: Math.max(0, resources[Suit.Waves] + (delta[Suit.Waves] ?? 0)),
    [Suit.Leaves]: Math.max(
      0,
      resources[Suit.Leaves] + (delta[Suit.Leaves] ?? 0)
    ),
    [Suit.Wyrms]: Math.max(0, resources[Suit.Wyrms] + (delta[Suit.Wyrms] ?? 0)),
    [Suit.Knots]: Math.max(0, resources[Suit.Knots] + (delta[Suit.Knots] ?? 0)),
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
