import knotsIcon from '../assets/icons/knots.svg';
import leavesIcon from '../assets/icons/leaves.svg';
import moonsIcon from '../assets/icons/moons.svg';
import sunsIcon from '../assets/icons/suns.svg';
import wavesIcon from '../assets/icons/waves.svg';
import wyrmsIcon from '../assets/icons/wyrms.svg';
import type { Suit } from '../engine/types';

export const SUIT_ICON_BY_SUIT: Record<Suit, string> = {
  Moons: moonsIcon,
  Suns: sunsIcon,
  Waves: wavesIcon,
  Leaves: leavesIcon,
  Wyrms: wyrmsIcon,
  Knots: knotsIcon,
};

export const SUIT_TEXT_TOKEN: Record<Suit, string> = {
  Moons: '{Moons}',
  Suns: '{Suns}',
  Waves: '{Waves}',
  Leaves: '{Leaves}',
  Wyrms: '{Wyrms}',
  Knots: '{Knots}',
};

export const SUIT_TOKEN_TO_SUIT: Record<string, Suit> = Object.freeze(
  Object.fromEntries(
    (Object.entries(SUIT_TEXT_TOKEN) as Array<[Suit, string]>).map(([suit, token]) => [token, suit])
  ) as Record<string, Suit>
);

export const SUIT_TOKEN_REGEX = new RegExp(
  (Object.values(SUIT_TEXT_TOKEN) as string[])
    .sort((left, right) => right.length - left.length)
    .map((token) => escapeRegex(token))
    .join('|'),
  'g'
);

export function SuitIcon({
  suit,
  className,
}: {
  suit: Suit;
  className?: string;
}) {
  return (
    <img
      src={SUIT_ICON_BY_SUIT[suit]}
      alt={suit}
      title={suit}
      className={`suit-icon${className ? ` ${className}` : ''}`}
    />
  );
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
