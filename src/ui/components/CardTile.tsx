import type { CSSProperties } from 'react';

import { CARD_BY_ID, type CardId } from '../../engine/cards';
import type { Suit } from '../../engine/types';
import { getCardImage } from '../cardImages';
import { SuitIcon } from '../suitIcons';
import { TokenChip, tokenEntries } from './TokenComponents';

export type CardPerspective = 'human' | 'bot';

function splitDeedTokensBySide(
  entries: Array<{ suit: Suit; count: number }>,
  perspective: CardPerspective
): {
  left: Array<{ suit: Suit; count: number }>;
  right: Array<{ suit: Suit; count: number }>;
} {
  if (entries.length === 1) {
    return perspective === 'bot'
      ? { left: [], right: entries }
      : { left: entries, right: [] };
  }

  const left: Array<{ suit: Suit; count: number }> = [];
  const right: Array<{ suit: Suit; count: number }> = [];
  for (const [index, entry] of entries.entries()) {
    const placeLeft = perspective === 'bot' ? index % 2 === 1 : index % 2 === 0;
    if (placeLeft) {
      left.push(entry);
    } else {
      right.push(entry);
    }
  }
  return { left, right };
}

export function CardTile({
  cardId,
  hidden,
  placeholder,
  deedTokens,
  deedProgress,
  deedTarget,
  inDevelopment,
  perspective = 'human',
}: {
  cardId?: CardId;
  hidden?: boolean;
  placeholder?: boolean;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
  inDevelopment?: boolean;
  perspective?: CardPerspective;
}) {
  if (placeholder) {
    return <div className="card-tile card-placeholder" aria-hidden="true" />;
  }

  if (hidden) {
    return <div className="card-tile card-back" title="Hidden card" />;
  }

  if (!cardId) {
    return null;
  }

  const card = CARD_BY_ID[cardId];
  const suits = card.kind === 'Excuse' ? [] : [...card.suits];
  const rank =
    card.kind === 'Property' || card.kind === 'Crown'
      ? String(card.rank)
      : card.kind === 'Pawn'
        ? 'P'
        : 'X';
  const deedTokenEntries = deedTokens ? tokenEntries(deedTokens) : [];
  const hasDeedTokens = deedTokenEntries.length > 0;
  const deedTokensBySide = splitDeedTokensBySide(deedTokenEntries, perspective);
  const hasDeedProgress = deedProgress !== undefined && deedTarget !== undefined;
  const progressValue = deedProgress ?? 0;
  const progressTarget = deedTarget ?? 0;
  const deedProgressRatio =
    progressTarget > 0
      ? Math.max(0, Math.min(1, progressValue / progressTarget))
      : 0;
  const deedProgressStyle = hasDeedProgress
    ? ({
        '--deed-progress-ratio': deedProgressRatio.toFixed(4),
      } as CSSProperties)
    : undefined;
  const cardImage = getCardImage(cardId);

  const metadataRow = (
    <div className="card-row card-meta">
      <div className="card-meta-leading">
        <span className="card-rank">{rank}</span>
        <div className="card-suits-row">
          {suits.length > 0 ? (
            suits.map((suit) => <SuitIcon key={`${cardId}-${suit}`} suit={suit} className="card-suit-icon" />)
          ) : (
            <span className="card-suit-placeholder" />
          )}
        </div>
      </div>
      {hasDeedProgress ? (
        <div
          className="deed-progress"
          title="development progress"
          aria-label="development progress"
          style={deedProgressStyle}
        >
          {deedProgress}/{deedTarget}
        </div>
      ) : (
        <span className="deed-progress-placeholder" aria-hidden="true" />
      )}
    </div>
  );

  const imageBody = (
    <div className="card-row card-body">
      <div className="card-image-frame" aria-hidden="true">
        <img className="card-image" src={cardImage} alt="" />
      </div>
      {hasDeedTokens ? (
        <>
          <div className="card-side-token-rail card-side-token-rail-left" aria-hidden="true">
            {deedTokensBySide.left.map((entry) => (
              <TokenChip key={`left-${cardId}-${entry.suit}`} suit={entry.suit} count={entry.count} compact />
            ))}
          </div>
          <div className="card-side-token-rail card-side-token-rail-right" aria-hidden="true">
            {deedTokensBySide.right.map((entry) => (
              <TokenChip key={`right-${cardId}-${entry.suit}`} suit={entry.suit} count={entry.count} compact />
            ))}
          </div>
        </>
      ) : null}
    </div>
  );

  return (
    <div
      className={`card-tile${perspective === 'bot' ? ' perspective-bot' : ''}${inDevelopment ? ' is-in-development' : ''}`}
      title={card.name}
    >
      {perspective === 'bot' ? imageBody : metadataRow}
      {perspective === 'bot' ? metadataRow : imageBody}
    </div>
  );
}
