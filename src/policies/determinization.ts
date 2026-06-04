import { PROPERTY_CARDS } from '../engine/cards';
import { shuffleInPlace, type RandomFn } from '../engine/rng';
import type { GameState, PlayerId, PlayerView } from '../engine/types';

const PROPERTY_CARD_IDS = PROPERTY_CARDS.map((card) => card.id);

export function sampleHiddenWorldStates({
  state,
  view,
  rootPlayer,
  worldCount,
  random,
  errorPrefix,
}: {
  state: GameState;
  view: PlayerView;
  rootPlayer: PlayerId;
  worldCount: number;
  random: RandomFn;
  errorPrefix: string;
}): GameState[] {
  const opponentPlayer = otherPlayerId(rootPlayer);
  const rootView = requiredPlayerView(view, rootPlayer, errorPrefix);
  const opponentView = requiredPlayerView(view, opponentPlayer, errorPrefix);

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

  const hiddenPool = PROPERTY_CARD_IDS.filter(
    (cardId) => !knownCards.has(cardId)
  );
  const expectedHiddenCount = opponentHandCount + drawCount;
  if (hiddenPool.length !== expectedHiddenCount) {
    throw new Error(
      `${errorPrefix} determinization mismatch: expected hidden=${String(expectedHiddenCount)} but found ${String(hiddenPool.length)}.`
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

function requiredPlayerView(
  view: PlayerView,
  playerId: PlayerId,
  errorPrefix: string
): PlayerView['players'][number] {
  const player = view.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`${errorPrefix} view is missing player ${playerId}.`);
  }
  return player;
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

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}
