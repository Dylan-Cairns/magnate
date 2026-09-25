import type { CSSProperties } from 'react';

import knotsIcon from '../assets/icons/knots.svg';
import leavesIcon from '../assets/icons/leaves.svg';
import moonsIcon from '../assets/icons/moons.svg';
import sunsIcon from '../assets/icons/suns.svg';
import wavesIcon from '../assets/icons/waves.svg';
import wyrmsIcon from '../assets/icons/wyrms.svg';
import type { Suit } from '../engine/types';
import { reportImageRenderFailure } from './cardImages';

// Shared opaque fills for suits. The artwork has transparent gaps between its
// layers, so it only reads correctly sitting on its own fill rather than
// whatever happens to be behind it — a white card face, the in-development
// grey, a dark chip. Tokens, dice, deck-map nodes and the log all back the
// artwork this way; SuitIcon does now too.
export const SUIT_TOKEN_BG: Record<Suit, string> = {
  Moons: '#e4e7eb',
  Suns: '#f7cc95',
  Waves: '#cfe3f5',
  Leaves: '#dfc8b2',
  Wyrms: '#bfe3b3',
  Knots: '#f6f4bf',
};

export const SUIT_ICON_BY_SUIT: Record<Suit, string> = {
  Moons: moonsIcon,
  Suns: sunsIcon,
  Waves: wavesIcon,
  Leaves: leavesIcon,
  Wyrms: wyrmsIcon,
  Knots: knotsIcon,
};

export const ALL_SUIT_ICON_URLS: readonly string[] =
  Object.values(SUIT_ICON_BY_SUIT);

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
    (Object.entries(SUIT_TEXT_TOKEN) as Array<[Suit, string]>).map(
      ([suit, token]) => [token, suit]
    )
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
      className={`suit-icon${className ? ` ${className}` : ''}`}
      style={{ '--suit-bg': SUIT_TOKEN_BG[suit] } as CSSProperties}
      onError={() =>
        reportImageRenderFailure(SUIT_ICON_BY_SUIT[suit], `${suit} token`)
      }
    />
  );
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
