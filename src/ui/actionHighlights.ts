import { ActionId } from '../engine/values';
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
    case ActionId.BuyDeed:
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
    case ActionId.DevelopOutright:
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
    case ActionId.DevelopDeed:
      return [
        { kind: 'played-card', cardId: action.cardId },
        ...resources(
          SUITS.filter((suit) => (action.tokens[suit] ?? 0) > 0),
          'spend'
        ),
      ];
    case ActionId.SellCard:
      return [
        { kind: 'hand-card', cardId: action.cardId },
        ...cardResources(action.cardId, 'gain'),
        { kind: 'pile', pile: 'discard' },
      ];
    case ActionId.Trade:
      return [
        ...resources([action.give], 'spend'),
        ...resources([action.receive], 'gain'),
      ];
    case ActionId.ChooseIncomeSuit:
      return [
        { kind: 'played-card', cardId: action.cardId },
        ...resources([action.suit], 'gain'),
      ];
    case ActionId.EndTurn:
      return [];
  }
}

// A confirmed human action keeps its hand card lit and its destination ghost
// filled while the animation plays, so the card stays on top from hover through
// the effect instead of snapping away when the picker closes. Placement is
// opt-in and only the human's own action may request it; a bot's action never
// produces a ghost.
export function committedHighlightTargets(
  action: GameAction,
  options: { includePlacement?: boolean } = {}
): HighlightTarget[] {
  const includePlacement = options.includePlacement ?? false;
  return actionHighlightTargets(action).filter(
    (target) =>
      target.kind === 'hand-card' ||
      (includePlacement && target.kind === 'district-lane')
  );
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
