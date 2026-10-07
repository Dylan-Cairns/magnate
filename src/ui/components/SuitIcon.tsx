import type { Suit } from '../../engine/types';
import { SuitTokenFace } from './SuitTokenFace';

/*
  The small suit marks on card faces. They render through the shared token face
  so a card's suits carry the same ring as the board tokens, district markers
  and action entries, rather than a bare image with no edge of its own.
*/
export function SuitIcon({
  suit,
  className,
}: {
  suit: Suit;
  className?: string;
}) {
  return <SuitTokenFace suit={suit} simplified className={className} />;
}
