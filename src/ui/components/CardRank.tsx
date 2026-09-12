import { CARD_BY_ID, type CardId } from '../../engine/cards';
import { CourtIcon } from '../courtIcon';

export function CardRank({ cardId }: { cardId: CardId }) {
  const card = CARD_BY_ID[cardId];
  if (card.kind === 'Court') {
    return <CourtIcon className="card-rank-court-icon" />;
  }
  const label =
    card.kind === 'Property' || card.kind === 'Crown'
      ? String(card.rank)
      : card.kind === 'Pawn'
        ? 'P'
        : 'X';
  return <>{label}</>;
}
