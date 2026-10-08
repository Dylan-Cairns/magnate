import { CardKind } from '../../engine/values';
import { CARD_BY_ID, type CardId } from '../../engine/cards';
import { CourtIcon } from '../courtIcon';

export function CardRank({
  cardId,
  courtFilled = false,
}: {
  cardId: CardId;
  courtFilled?: boolean;
}) {
  const card = CARD_BY_ID[cardId];
  if (card.kind === CardKind.Court) {
    return <CourtIcon className="card-rank-court-icon" filled={courtFilled} />;
  }
  const label =
    card.kind === CardKind.Property || card.kind === CardKind.Crown
      ? String(card.rank)
      : card.kind === CardKind.Pawn
        ? 'P'
        : 'X';
  return <>{label}</>;
}
