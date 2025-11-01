import { CARD_BY_ID, type CardId } from '../../engine/cards';
import type { Suit } from '../../engine/types';
import { getCardImage } from '../cardImages';
import { SuitIcon } from '../suitIcons';
import { TokenRow, tokenEntries } from './TokenComponents';

export type CardPerspective = 'human' | 'bot';

export function CardTile({
  cardId,
  hidden,
  placeholder,
  deedTokens,
  deedProgress,
  deedTarget,
  perspective = 'human',
}: {
  cardId?: CardId;
  hidden?: boolean;
  placeholder?: boolean;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
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
  const hasDeedTokens = deedTokens ? tokenEntries(deedTokens).length > 0 : false;
  const hasDeedProgress = deedProgress !== undefined && deedTarget !== undefined;
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
        <div className="deed-progress" title="development progress" aria-label="development progress">
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
      {hasDeedTokens && deedTokens ? <TokenRow tokens={deedTokens} compact className="card-token-row" /> : null}
    </div>
  );

  return (
    <div className={`card-tile${perspective === 'bot' ? ' perspective-bot' : ''}`} title={card.name}>
      {perspective === 'bot' ? imageBody : metadataRow}
      {perspective === 'bot' ? metadataRow : imageBody}
    </div>
  );
}
