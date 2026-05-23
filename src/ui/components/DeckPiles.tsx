import { CARD_BY_ID, type CardId } from '../../engine/cards';
import type { Suit } from '../../engine/types';
import { getCardImage } from '../cardImages';
import { SuitIcon, SUIT_TEXT_TOKEN } from '../suitIcons';
import { SuitText } from './SuitText';

export function DeckPiles({
  drawCount,
  reshuffles,
  discard,
  pendingDiscardHoldback,
  terminal,
}: {
  drawCount: number;
  reshuffles: number;
  discard: readonly CardId[];
  pendingDiscardHoldback: number;
  terminal: boolean;
}) {
  const deckStackCount = Math.min(3, drawCount);
  const deckOverlayShiftClass =
    deckStackCount >= 3
      ? 'overlay-shift-2'
      : deckStackCount === 2
        ? 'overlay-shift-1'
        : 'overlay-shift-0';
  const showSecondShuffleLabel =
    reshuffles > 0 && !(terminal && drawCount === 0);
  const visibleDiscardCards =
    pendingDiscardHoldback > 0
      ? discard.slice(pendingDiscardHoldback)
      : discard;
  const discardStackCardIds = visibleDiscardCards.slice(0, 3).reverse();
  const discardCardDetails = visibleDiscardCards.map((cardId) => {
    const card = CARD_BY_ID[cardId];
    const rank =
      card.kind === 'Property' || card.kind === 'Crown'
        ? String(card.rank)
        : card.kind;
    const suitTokenText =
      card.kind === 'Excuse'
        ? ''
        : card.suits.map((suit) => SUIT_TEXT_TOKEN[suit]).join(' ');
    return {
      id: card.id,
      name: card.name,
      rank,
      suitTokenText,
    };
  });

  return (
    <section className="panel">
      <h2>Deck State</h2>
      <div className="deck-piles" aria-label="Deck and discard piles">
        <div className="deck-pile">
          <div
            className={`deck-pile-stack is-deck ${deckOverlayShiftClass}`}
            title="Cards remaining"
            aria-label="Cards remaining"
          >
            {deckStackCount === 0 ? (
              <div className="deck-pile-card deck-pile-card-empty deck-pile-stack-card" />
            ) : (
              Array.from({ length: deckStackCount }).map((_, index) => (
                <div
                  key={`deck-back-${index}`}
                  className="deck-pile-card deck-pile-card-back deck-pile-stack-card"
                />
              ))
            )}
            {showSecondShuffleLabel ? (
              <span className="deck-pile-overlay-label" aria-hidden="true">
                2nd shuffle
              </span>
            ) : null}
            <div className="deck-pile-animation-anchor" aria-hidden="true" />
          </div>
          <strong className="deck-pile-count">{drawCount}</strong>
        </div>
        <div className="deck-pile">
          <div className="player-score-wrap discard-pile-wrap">
            <div
              className={`deck-pile-stack is-discard${discardStackCardIds.length > 0 ? ' is-fanned' : ''}`}
              title="Discard pile"
              aria-label="Discard pile"
              tabIndex={0}
            >
              {discardStackCardIds.length > 0 ? (
                discardStackCardIds.map((cardId, index) => {
                  const isTopCard = index === discardStackCardIds.length - 1;
                  let topMeta: { rank: string; suits: Suit[] } | null = null;
                  if (isTopCard) {
                    const card = CARD_BY_ID[cardId];
                    topMeta = {
                      rank:
                        card.kind === 'Property' || card.kind === 'Crown'
                          ? String(card.rank)
                          : card.kind === 'Pawn'
                            ? 'P'
                            : 'X',
                      suits: card.kind === 'Excuse' ? [] : [...card.suits],
                    };
                  }
                  return (
                    <div
                      key={`discard-${cardId}-${index}`}
                      className="deck-pile-card deck-pile-card-discard deck-pile-stack-card"
                    >
                      <div className="deck-pile-card-meta">
                        {topMeta !== null && (
                          <>
                            <span className="card-rank">{topMeta.rank}</span>
                            {topMeta.suits.length > 0 && (
                              <div className="deck-pile-card-suits">
                                {topMeta.suits.map((suit) => (
                                  <SuitIcon
                                    key={suit}
                                    suit={suit}
                                    className="card-suit-icon"
                                  />
                                ))}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                      <img
                        className="deck-pile-image"
                        src={getCardImage(cardId)}
                        alt=""
                      />
                    </div>
                  );
                })
              ) : (
                <div className="deck-pile-card deck-pile-card-empty deck-pile-stack-card" />
              )}
              <div className="discard-pile-animation-anchor" aria-hidden="true" />
            </div>
            <section
              className="player-score-popover discard-pile-popover"
              role="tooltip"
              aria-label="Discard pile details"
            >
              <p className="score-result">
                Discarded Cards: <strong>{discardCardDetails.length}</strong>
              </p>
              {discardCardDetails.length === 0 ? (
                <p className="score-line">
                  <span>None yet</span>
                  <strong>-</strong>
                </p>
              ) : (
                <ol className="discard-pile-list">
                  {discardCardDetails.map((card, index) => (
                    <li
                      key={`discard-detail-${card.id}-${index}`}
                      className="discard-pile-item"
                    >
                      <p className="discard-pile-card-row">
                        <strong className="discard-pile-card-rank">
                          {card.rank}
                        </strong>
                        <span className="discard-pile-card-suits">
                          {card.suitTokenText.length > 0 ? (
                            <SuitText text={card.suitTokenText} />
                          ) : (
                            <strong>-</strong>
                          )}
                        </span>
                        <span className="discard-pile-card-name">
                          {card.name}
                        </span>
                      </p>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </div>
          <strong className="deck-pile-count">
            {visibleDiscardCards.length}
          </strong>
        </div>
      </div>
    </section>
  );
}
