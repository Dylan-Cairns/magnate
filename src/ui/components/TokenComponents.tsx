import { SUITS } from '../../engine/stateHelpers';
import type { ResourcePool, Suit } from '../../engine/types';
import { SuitTokenFace } from './SuitTokenFace';
import { Tooltip } from './Tooltip';

export { SUIT_TOKEN_BG } from './SuitTokenFace';

export function tokenEntries(
  tokens: Partial<Record<Suit, number>> | ResourcePool
): Array<{ suit: Suit; count: number }> {
  return SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 })).filter(
    (entry) => entry.count > 0
  );
}

export function TokenRow({
  tokens,
  compact,
  emptyLabel,
  fixedSuitSlots,
  className,
  highlightedSuits,
}: {
  tokens: Partial<Record<Suit, number>> | ResourcePool;
  compact?: boolean;
  emptyLabel?: string;
  fixedSuitSlots?: boolean;
  className?: string;
  highlightedSuits?: ReadonlySet<Suit>;
}) {
  const entries = fixedSuitSlots
    ? SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 }))
    : tokenEntries(tokens);

  if (!fixedSuitSlots && entries.length === 0) {
    return <span className="empty-note">{emptyLabel ?? 'None'}</span>;
  }

  return (
    <div
      className={`token-row${compact ? ' compact' : ''}${fixedSuitSlots ? ' fixed-suits' : ''}${
        className ? ` ${className}` : ''
      }`}
    >
      {entries.map(({ suit, count }) => (
        <TokenChip
          key={suit}
          suit={suit}
          count={count}
          compact={compact}
          className={
            highlightedSuits?.has(suit) ? 'is-income-highlighted' : undefined
          }
        />
      ))}
    </div>
  );
}

export function TokenChip({
  suit,
  count,
  compact,
  className,
}: {
  suit: Suit;
  count: number;
  compact?: boolean;
  className?: string;
}) {
  const isEmpty = count === 0;
  return (
    <span
      className={`token-chip tooltip-trigger${compact ? ' compact' : ''}${isEmpty ? ' empty' : ''}${
        className ? ` ${className}` : ''
      }`}
      data-token-suit={suit}
    >
      <SuitTokenFace suit={suit} empty={isEmpty} />
      {count > 1 && <span className="token-count">x{count}</span>}
      <Tooltip>{suit}</Tooltip>
    </span>
  );
}
