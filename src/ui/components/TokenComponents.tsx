import { useHighlightClass } from './ActionHighlights';
import type { TradeProgress } from '../runtime/types';
import { ProgressTracker } from './ProgressTracker';
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
  highlightResources = false,
  tradeProgress,
}: {
  tokens: Partial<Record<Suit, number>> | ResourcePool;
  compact?: boolean;
  emptyLabel?: string;
  fixedSuitSlots?: boolean;
  className?: string;
  highlightedSuits?: ReadonlySet<Suit>;
  highlightResources?: boolean;
  tradeProgress?: TradeProgress;
}) {
  const highlightClass = useHighlightClass();
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
          tradeProgress={
            tradeProgress?.suit === suit ? tradeProgress : undefined
          }
          compact={compact}
          className={`${highlightedSuits?.has(suit) ? 'is-income-highlighted' : ''}${highlightClass({ kind: 'resource', suit }, highlightResources)}`}
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
  showTooltip = true,
  tradeProgress,
}: {
  suit: Suit;
  count: number;
  compact?: boolean;
  className?: string;
  showTooltip?: boolean;
  tradeProgress?: TradeProgress;
}) {
  const isEmpty = count === 0 && !tradeProgress;
  return (
    <span
      className={`token-chip${showTooltip ? ' tooltip-trigger' : ''}${compact ? ' compact' : ''}${isEmpty ? ' empty' : ''}${
        className ? ` ${className}` : ''
      }`}
      data-token-suit={suit}
    >
      <SuitTokenFace suit={suit} empty={isEmpty} />
      {count > 1 && <span className="token-count">x{count}</span>}
      {tradeProgress ? (
        <span className="trade-progress">
          <ProgressTracker
            key={tradeProgress.transactionId}
            cardId={tradeProgress.transactionId}
            deedProgress={tradeProgress.landed}
            deedTarget={tradeProgress.total}
            label="Trade progress"
          />
        </span>
      ) : null}
      {showTooltip ? <Tooltip>{suit}</Tooltip> : null}
    </span>
  );
}
