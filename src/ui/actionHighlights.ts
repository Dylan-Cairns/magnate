import type { CardId } from '../engine/cards';
import { findProperty, SUITS } from '../engine/stateHelpers';
import type { DistrictId, GameAction, Suit } from '../engine/types';

// Player-owned targets are scoped to the human player's rendered components.
export type HighlightTarget =
  | { kind: 'hand-card'; cardId: CardId }
  | { kind: 'played-card'; cardId: CardId }
  | { kind: 'resource'; suit: Suit }
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
      return `${target.kind}:${target.suit}`;
    case 'district-lane':
      return `${target.kind}:${target.districtId}:${target.cardId}:${target.placement}`;
    case 'pile':
      return `${target.kind}:${target.pile}`;
  }
}

function resources(suits: readonly Suit[]): HighlightTarget[] {
  return suits.map((suit) => ({ kind: 'resource', suit }));
}

function cardResources(cardId: CardId): HighlightTarget[] {
  const card = findProperty(cardId);
  if (!card) throw new Error(`Expected a property card: ${cardId}`);
  return resources(card.suits);
}

export function actionHighlightTargets(action: GameAction): HighlightTarget[] {
  switch (action.type) {
    case 'buy-deed':
      return [
        { kind: 'hand-card', cardId: action.cardId },
        ...cardResources(action.cardId),
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
        ...resources(SUITS.filter((suit) => (action.payment[suit] ?? 0) > 0)),
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
        ...resources(SUITS.filter((suit) => (action.tokens[suit] ?? 0) > 0)),
      ];
    case 'sell-card':
      return [
        { kind: 'hand-card', cardId: action.cardId },
        ...cardResources(action.cardId),
        { kind: 'pile', pile: 'discard' },
      ];
    case 'trade':
      return resources([action.give, action.receive]);
    case 'choose-income-suit':
      return [
        { kind: 'played-card', cardId: action.cardId },
        ...resources([action.suit]),
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
