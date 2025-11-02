import type { CSSProperties } from 'react';

import { SUITS } from '../../engine/stateHelpers';
import type { ResourcePool, Suit } from '../../engine/types';
import { SuitIcon } from '../suitIcons';

const SUIT_TOKEN_BG: Record<Suit, string> = {
  Moons: '#e4e7eb',
  Suns: '#f7cc95',
  Waves: '#cfe3f5',
  Leaves: '#dfc8b2',
  Wyrms: '#bfe3b3',
  Knots: '#f6f4bf',
};

export function tokenEntries(tokens: Partial<Record<Suit, number>> | ResourcePool): Array<{ suit: Suit; count: number }> {
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
}: {
  tokens: Partial<Record<Suit, number>> | ResourcePool;
  compact?: boolean;
  emptyLabel?: string;
  fixedSuitSlots?: boolean;
  className?: string;
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
        <TokenChip key={suit} suit={suit} count={count} compact={compact} />
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
      className={`token-chip${compact ? ' compact' : ''}${isEmpty ? ' empty' : ''}${
        className ? ` ${className}` : ''
      }`}
      data-token-suit={suit}
      title={`${suit} x${count}`}
      style={{ '--token-bg': SUIT_TOKEN_BG[suit] } as CSSProperties}
    >
      <SuitIcon suit={suit} className="chip-suit-icon" />
      {count > 1 && <span className="token-count">x{count}</span>}
    </span>
  );
}
