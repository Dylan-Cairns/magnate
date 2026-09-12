import type { CardId } from '../engine/cards';
import { findDevelopableCard, SUITS } from '../engine/stateHelpers';
import type { DistrictId, GameAction, Suit } from '../engine/types';

export type ResourceEffect = 'gain' | 'spend';

// Player-owned targets are scoped to the human player's rendered components.
export type HighlightTarget =
  | { kind: 'hand-card'; cardId: CardId }
  | { kind: 'played-card'; cardId: CardId }
  | { kind: 'resource'; suit: Suit; effect: ResourceEffect }
  | {
      kind: 'district-lane';
      districtId: DistrictId;
      cardId: CardId;
      placement: 'deed' | 'developed';
    }
  | { kind: 'pile'; pile: 'discard' };

export function highlightTargetKey(target: HighlightTarget): string {
  switch (target.kind) {
    case 'hand-card':
    case 'played-card':
      return `${target.kind}:${target.cardId}`;
    case 'resource':
      return `${target.kind}:${target.effect}:${target.suit}`;
    case 'district-lane':
      return `${target.kind}:${target.districtId}:${target.cardId}:${target.placement}`;
    case 'pile':
      return `${target.kind}:${target.pile}`;
  }
}

export function resourceGainSuits(
  targets: readonly HighlightTarget[]
): Set<Suit> {
  const suits = new Set<Suit>();
  for (const target of targets) {
    if (target.kind === 'resource' && target.effect === 'gain') {
      suits.add(target.suit);
    }
  }
  return suits;
}

function resources(
  suits: readonly Suit[],
  effect: ResourceEffect
): HighlightTarget[] {
  return suits.map((suit) => ({ kind: 'resource', suit, effect }));
}

function cardResources(
  cardId: CardId,
  effect: ResourceEffect
): HighlightTarget[] {
  const card = findDevelopableCard(cardId);
  if (!card) throw new Error(`Expected a property card: ${cardId}`);
  return resources(card.suits, effect);
}

export function actionHighlightTargets(action: GameAction): HighlightTarget[] {
  switch (action.type) {
    case 'buy-deed':
      return [
        { kind: 'hand-card', cardId: action.cardId },
        ...cardResources(action.cardId, 'spend'),
        {
          kind: 'district-lane',
          districtId: action.districtId,
          cardId: action.cardId,
          placement: 'deed',
        },
      ];
    case 'develop-outright':
      return [
        { kind: 'hand-card', cardId: action.cardId },
        ...resources(
          SUITS.filter((suit) => (action.payment[suit] ?? 0) > 0),
          'spend'
        ),
        {
          kind: 'district-lane',
          districtId: action.districtId,
          cardId: action.cardId,
          placement: 'developed',
        },
      ];
    case 'develop-deed':
      return [
        { kind: 'played-card', cardId: action.cardId },
        ...resources(
          SUITS.filter((suit) => (action.tokens[suit] ?? 0) > 0),
          'spend'
        ),
      ];
    case 'sell-card':
      return [
        { kind: 'hand-card', cardId: action.cardId },
        ...cardResources(action.cardId, 'gain'),
        { kind: 'pile', pile: 'discard' },
      ];
    case 'trade':
      return [
        ...resources([action.give], 'spend'),
        ...resources([action.receive], 'gain'),
      ];
    case 'choose-income-suit':
      return [
        { kind: 'played-card', cardId: action.cardId },
        ...resources([action.suit], 'gain'),
      ];
    case 'end-turn':
      return [];
  }
}

// A grouped entry previews only effects shared by every remaining option.
export function sharedActionHighlightTargets(
  actions: readonly GameAction[]
): HighlightTarget[] {
  const [first, ...rest] = actions;
  if (!first) return [];
  const otherKeys = rest.map(
    (action) => new Set(actionHighlightTargets(action).map(highlightTargetKey))
  );
  return actionHighlightTargets(first).filter((target) =>
    otherKeys.every((keys) => keys.has(highlightTargetKey(target)))
  );
}
