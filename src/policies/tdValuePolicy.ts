import { actionStableKey } from '../engine/actionSurface';
import { PROPERTY_CARDS } from '../engine/cards';
import { shuffleInPlace } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import { stepToDecision } from '../engine/session';
import { toPlayerView } from '../engine/view';
import type { GameAction, GameState, PlayerId, PlayerView } from '../engine/types';
import type { ActionPolicy } from './types';
import { encodeObservation } from './trainingEncoding';
import {
  DEFAULT_TD_VALUE_MODEL_INDEX_PATH,
  preloadTdValueBrowserModel,
} from './modelRuntimeCache';
import { type TdValueScorer } from './tdValueModelPack';

const PROPERTY_CARD_IDS = PROPERTY_CARDS.map((card) => card.id);

const DEFAULT_TD_VALUE_WORLDS = 8;

export interface TdValuePolicyOptions {
  worlds?: number;
  modelIndexPath?: string;
  loadModel?: () => Promise<TdValueScorer>;
}

export function createTdValuePolicy(options: TdValuePolicyOptions = {}): ActionPolicy {
  const worlds = integerWithFloor(options.worlds ?? DEFAULT_TD_VALUE_WORLDS, 1);
  const configuredLoader =
    options.loadModel ??
    (() =>
      preloadTdValueBrowserModel(
        options.modelIndexPath ?? DEFAULT_TD_VALUE_MODEL_INDEX_PATH
      ));

  let modelPromise: Promise<TdValueScorer> | null = null;
  function getModel(): Promise<TdValueScorer> {
    if (modelPromise === null) {
      modelPromise = configuredLoader();
    }
    return modelPromise;
  }

  return {
    async selectAction({ view, state, legalActions: candidateActions, random }) {
      if (candidateActions.length === 0) {
        return undefined;
      }
      if (candidateActions.length === 1) {
        return candidateActions[0];
      }

      const model = await getModel();
      const rootPlayer = view.activePlayerId;
      const worldStates = sampleWorldStates(
        state,
        view,
        rootPlayer,
        worlds,
        random
      );
      const rankedActions = [...candidateActions].sort((left, right) =>
        actionStableKey(left).localeCompare(actionStableKey(right))
      );
      if (worldStates.length === 0) {
        return rankedActions[0];
      }

      let bestAction = rankedActions[0];
      let bestActionKey = actionStableKey(bestAction);
      let bestScore = Number.NEGATIVE_INFINITY;
      for (const action of rankedActions) {
        const actionKey = actionStableKey(action);
        let total = 0;
        for (const world of worldStates) {
          total += scoreActionInWorld({
            world,
            action,
            rootPlayer,
            scorer: model,
          });
        }
        const score = total / worldStates.length;
        if (
          score > bestScore
          || (approximatelyEqual(score, bestScore)
            && actionKey.localeCompare(bestActionKey) < 0)
        ) {
          bestAction = action;
          bestActionKey = actionKey;
          bestScore = score;
        }
      }
      return bestAction;
    },
  };
}

function scoreActionInWorld({
  world,
  action,
  rootPlayer,
  scorer,
}: {
  world: GameState;
  action: GameAction;
  rootPlayer: PlayerId;
  scorer: TdValueScorer;
}): number {
  const next = stepToDecision(world, action);
  if (isTerminal(next)) {
    return terminalValue(next, rootPlayer);
  }

  const activePlayer = next.players[next.activePlayerIndex]?.id;
  if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
    throw new Error('TD value policy could not resolve active player from next state.');
  }
  const nextView = toPlayerView(next, activePlayer);
  const observation = encodeObservation(nextView);
  const activeValue = scorer.predict(observation);
  const rootValue = activePlayer === rootPlayer ? activeValue : -activeValue;
  return clamp(rootValue, -1, 1);
}

function sampleWorldStates(
  state: GameState,
  view: PlayerView,
  rootPlayer: PlayerId,
  worldCount: number,
  random: () => number
): GameState[] {
  const opponentPlayer = otherPlayerId(rootPlayer);
  const rootView = requiredPlayerView(view, rootPlayer);
  const opponentView = requiredPlayerView(view, opponentPlayer);

  const rootHand = [...rootView.hand];
  const opponentHandCount = opponentView.handCount;
  const drawCount = view.deck.drawCount;

  const knownCards = new Set<string>(rootHand);
  for (const cardId of view.deck.discard) {
    knownCards.add(cardId);
  }
  for (const cardId of districtPropertyCards(view)) {
    knownCards.add(cardId);
  }

  const hiddenPool = PROPERTY_CARD_IDS.filter((cardId) => !knownCards.has(cardId));
  const expectedHiddenCount = opponentHandCount + drawCount;
  if (hiddenPool.length !== expectedHiddenCount) {
    throw new Error(
      `TD value determinization mismatch: expected hidden=${String(expectedHiddenCount)} but found ${String(hiddenPool.length)}.`
    );
  }

  const worlds: GameState[] = [];
  for (let index = 0; index < worldCount; index += 1) {
    const sampledHidden = [...hiddenPool];
    shuffleInPlace(sampledHidden, random);
    const opponentHand = sampledHidden.slice(0, opponentHandCount);
    const draw = sampledHidden.slice(
      opponentHandCount,
      opponentHandCount + drawCount
    );

    const world = structuredClone(state) as GameState;
    world.players = world.players.map((player) => {
      if (player.id === rootPlayer) {
        return { ...player, hand: [...rootHand] };
      }
      if (player.id === opponentPlayer) {
        return { ...player, hand: [...opponentHand] };
      }
      return player;
    });
    world.deck = {
      ...world.deck,
      draw: [...draw],
    };
    worlds.push(world);
  }
  return worlds;
}

function terminalValue(state: GameState, rootPlayer: PlayerId): number {
  const winner = state.finalScore?.winner;
  if (!winner || winner === 'Draw') {
    return 0;
  }
  return winner === rootPlayer ? 1 : -1;
}

function districtPropertyCards(view: PlayerView): Set<string> {
  const cards = new Set<string>();
  for (const district of view.districts) {
    for (const playerId of ['PlayerA', 'PlayerB'] as const) {
      const stack = district.stacks[playerId];
      for (const cardId of stack.developed) {
        cards.add(cardId);
      }
      if (stack.deed) {
        cards.add(stack.deed.cardId);
      }
    }
  }
  return cards;
}

function requiredPlayerView(
  view: PlayerView,
  playerId: PlayerId
): PlayerView['players'][number] {
  const player = view.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`TD value policy view is missing player ${playerId}.`);
  }
  return player;
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

function integerWithFloor(value: number, floor: number): number {
  if (!Number.isFinite(value)) {
    throw new Error(
      `TD value policy expected a finite number; received ${String(value)}.`
    );
  }
  const rounded = Math.trunc(value);
  if (rounded < floor) {
    throw new Error(
      `TD value policy value must be >= ${String(floor)}; received ${String(value)}.`
    );
  }
  return rounded;
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
