import { CARD_BY_ID, type CardId } from '../../engine/cards';
import type { PlayerId, Suit } from '../../engine/types';
import { getCardImage, reportImageRenderFailure } from '../cardImages';
import { SuitIcon } from '../suitIcons';
import { TokenChip, tokenEntries } from './TokenComponents';
import { ProgressTracker } from './ProgressTracker';
import { layoutDeedTokensBySide } from './deedTokenLayout';
import { Tooltip } from './Tooltip';

export type CardPerspective = 'human' | 'bot';

export function CardTile({
  cardId,
  hidden,
  placeholder,
  deedTokens,
  deedProgress,
  deedTarget,
  inDevelopment,
  perspective = 'human',
  handOwnerId,
  handCardId,
  handSlotKind,
  animateDeedProgress = true,
  incomeHighlighted = false,
}: {
  cardId?: CardId;
  hidden?: boolean;
  placeholder?: boolean;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
  inDevelopment?: boolean;
  perspective?: CardPerspective;
  handOwnerId?: PlayerId;
  handCardId?: CardId;
  handSlotKind?: 'occupied' | 'hidden' | 'empty';
  animateDeedProgress?: boolean;
  incomeHighlighted?: boolean;
}) {
  if (placeholder) {
    return (
      <div
        className="card-tile card-placeholder"
        aria-hidden="true"
        data-hand-owner-id={handOwnerId}
        data-hand-card-id={handCardId}
        data-hand-slot-kind={handSlotKind}
      />
    );
  }

  if (hidden) {
    return (
      <div
        className="card-tile card-back"
        data-hand-owner-id={handOwnerId}
        data-hand-card-id={handCardId}
        data-hand-slot-kind={handSlotKind}
      />
    );
  }

  if (!cardId) {
    return null;
  }

  return (
    <CardTileCard
      cardId={cardId}
      deedTokens={deedTokens}
      deedProgress={deedProgress}
      deedTarget={deedTarget}
      inDevelopment={inDevelopment}
      perspective={perspective}
      handOwnerId={handOwnerId}
      handCardId={handCardId}
      handSlotKind={handSlotKind}
      animateDeedProgress={animateDeedProgress}
      incomeHighlighted={incomeHighlighted}
    />
  );
}

function CardTileCard({
  cardId,
  deedTokens,
  deedProgress,
  deedTarget,
  inDevelopment,
  perspective = 'human',
  handOwnerId,
  handCardId,
  handSlotKind,
  animateDeedProgress = true,
  incomeHighlighted = false,
}: {
  cardId: CardId;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
  inDevelopment?: boolean;
  perspective?: CardPerspective;
  handOwnerId?: PlayerId;
  handCardId?: CardId;
  handSlotKind?: 'occupied' | 'hidden' | 'empty';
  animateDeedProgress?: boolean;
  incomeHighlighted?: boolean;
}) {
  const card = CARD_BY_ID[cardId];
  const cardImage = getCardImage(cardId);
  const suits = card.kind === 'Excuse' ? [] : [...card.suits];
  const rank =
    card.kind === 'Property' || card.kind === 'Crown'
      ? String(card.rank)
      : card.kind === 'Pawn'
        ? 'P'
        : 'X';
  const deedTokenEntries = deedTokens ? tokenEntries(deedTokens) : [];
  const hasDeedTokens = deedTokenEntries.length > 0;
  const showDeedTokenRails = Boolean(inDevelopment) || hasDeedTokens;
  const deedTokensBySide = layoutDeedTokensBySide(
    cardId,
    perspective,
    deedTokenEntries,
    {
      resetWhenEmpty: Boolean(inDevelopment) && deedTokenEntries.length === 0,
    }
  );
  const hasDeedProgress =
    deedProgress !== undefined && deedTarget !== undefined;

  const metadataRow = (
    <div className="card-row card-meta">
      <div className="card-meta-leading">
        <span className="card-rank">{rank}</span>
        <div className="card-suits-row">
          {suits.length > 0 ? (
            suits.map((suit) => (
              <SuitIcon
                key={`${cardId}-${suit}`}
                suit={suit}
                className="card-suit-icon"
              />
            ))
          ) : (
            <span className="card-suit-placeholder" />
          )}
        </div>
      </div>
      {hasDeedProgress ? (
        <ProgressTracker
          deedProgress={deedProgress}
          deedTarget={deedTarget}
          animateDeedProgress={animateDeedProgress}
          cardId={cardId}
        />
      ) : (
        <span className="deed-progress-placeholder" aria-hidden="true" />
      )}
    </div>
  );

  const imageBody = (
    <div className="card-row card-body">
      <div className="card-image-frame" aria-hidden="true">
        <img
          className="card-image"
          src={cardImage}
          alt=""
          onError={() => reportImageRenderFailure(cardImage, 'card image')}
        />
      </div>
      {showDeedTokenRails ? (
        <>
          <div
            className="card-side-token-rail card-side-token-rail-left"
            data-deed-token-rail="left"
            aria-hidden="true"
          >
            {deedTokensBySide.left.map((entry) => (
              <TokenChip
                key={`left-${cardId}-${entry.suit}`}
                suit={entry.suit}
                count={entry.count}
                compact
              />
            ))}
          </div>
          <div
            className="card-side-token-rail card-side-token-rail-right"
            data-deed-token-rail="right"
            aria-hidden="true"
          >
            {deedTokensBySide.right.map((entry) => (
              <TokenChip
                key={`right-${cardId}-${entry.suit}`}
                suit={entry.suit}
                count={entry.count}
                compact
              />
            ))}
          </div>
        </>
      ) : null}
    </div>
  );

  return (
    <div
      className={`card-tile${perspective === 'bot' ? ' perspective-bot' : ''}${inDevelopment ? ' is-in-development' : ''}${incomeHighlighted ? ' is-income-highlighted' : ''} tooltip-trigger`}
      data-card-id={cardId}
      data-in-development={inDevelopment ? 'true' : undefined}
      data-hand-owner-id={handOwnerId}
      data-hand-card-id={handCardId}
      data-hand-slot-kind={handSlotKind}
    >
      {perspective === 'bot' ? imageBody : metadataRow}
      {perspective === 'bot' ? metadataRow : imageBody}
      <Tooltip
        placement={
          perspective === 'human' && handOwnerId === undefined
            ? 'below'
            : 'above'
        }
      >
        {card.name}
      </Tooltip>
    </div>
  );
}
