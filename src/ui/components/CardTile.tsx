import { CARD_BY_ID, type CardId } from '../../engine/cards';
import type { Suit } from '../../engine/types';
import { getCardImage } from '../cardImages';
import { SuitIcon } from '../suitIcons';
import { TokenChip, tokenEntries } from './TokenComponents';

export type CardPerspective = 'human' | 'bot';
const DEED_PROGRESS_RING_RADIUS = 16;
const DEED_PROGRESS_RING_CENTER = 18;
const DEED_PROGRESS_START_ANGLE_DEGREES = -90;

function pointOnProgressRing(angleDegrees: number): { x: number; y: number } {
  const angleRadians = (angleDegrees * Math.PI) / 180;
  return {
    x: DEED_PROGRESS_RING_CENTER + DEED_PROGRESS_RING_RADIUS * Math.cos(angleRadians),
    y: DEED_PROGRESS_RING_CENTER + DEED_PROGRESS_RING_RADIUS * Math.sin(angleRadians),
  };
}

function buildDeedProgressArcPath(progressRatio: number): string | null {
  if (progressRatio <= 0 || progressRatio >= 1) {
    return null;
  }
  const start = pointOnProgressRing(DEED_PROGRESS_START_ANGLE_DEGREES);
  const end = pointOnProgressRing(DEED_PROGRESS_START_ANGLE_DEGREES + progressRatio * 360);
  const largeArcFlag = progressRatio > 0.5 ? 1 : 0;
  return [
    `M ${start.x} ${start.y}`,
    `A ${DEED_PROGRESS_RING_RADIUS} ${DEED_PROGRESS_RING_RADIUS} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`,
  ].join(' ');
}

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
  const deedProgressArcPath = buildDeedProgressArcPath(deedProgressRatio);
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
        >
          <svg className="deed-progress-ring" viewBox="0 0 36 36" aria-hidden="true">
            <circle
              className="deed-progress-ring-track"
              cx="18"
              cy="18"
              r={DEED_PROGRESS_RING_RADIUS}
            />
            {deedProgressRatio >= 1 ? (
              <circle className="deed-progress-ring-value" cx="18" cy="18" r={DEED_PROGRESS_RING_RADIUS} />
            ) : deedProgressArcPath ? (
              <path className="deed-progress-ring-value" d={deedProgressArcPath} />
            ) : null}
          </svg>
          <span className="deed-progress-value">
            {deedProgress}/{deedTarget}
          </span>
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
