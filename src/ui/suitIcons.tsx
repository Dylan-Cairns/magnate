import type { CSSProperties } from 'react';

import knotsIcon from '../assets/icons/knots.svg';
import leavesIcon from '../assets/icons/leaves.svg';
import moonsIcon from '../assets/icons/moons.svg';
import sunsIcon from '../assets/icons/suns.svg';
import wavesIcon from '../assets/icons/waves.svg';
import wyrmsIcon from '../assets/icons/wyrms.svg';
import knotsSimpleIcon from '../assets/icons/simplified/knots.svg';
import leavesSimpleIcon from '../assets/icons/simplified/leaves.svg';
import moonsSimpleIcon from '../assets/icons/simplified/moons.svg';
import sunsSimpleIcon from '../assets/icons/simplified/suns.svg';
import wavesSimpleIcon from '../assets/icons/simplified/waves.svg';
import wyrmsSimpleIcon from '../assets/icons/simplified/wyrms.svg';
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

// Flat, two-tone suit marks taken from the Decktet's own simplified symbols
// (the glyphs the printed cards use in their corners). They read clearly at
// card/action sizes where the shaded emblem turns to mush. The artwork leaves
// its negative space transparent, so it sits on the same pale SUIT_TOKEN_BG
// field as the shaded emblems — the saturated colour is in the mark itself.
export const SUIT_ICON_SIMPLIFIED_BY_SUIT: Record<Suit, string> = {
  Moons: moonsSimpleIcon,
  Suns: sunsSimpleIcon,
  Waves: wavesSimpleIcon,
  Leaves: leavesSimpleIcon,
  Wyrms: wyrmsSimpleIcon,
  Knots: knotsSimpleIcon,
};

export const ALL_SUIT_ICON_URLS: readonly string[] = [
  ...Object.values(SUIT_ICON_BY_SUIT),
  ...Object.values(SUIT_ICON_SIMPLIFIED_BY_SUIT),
];

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
  const src = SUIT_ICON_SIMPLIFIED_BY_SUIT[suit];
  return (
    <img
      src={src}
      alt={suit}
      className={`suit-icon${className ? ` ${className}` : ''}`}
      style={{ '--suit-bg': SUIT_TOKEN_BG[suit] } as CSSProperties}
      onError={() => reportImageRenderFailure(src, `${suit} token`)}
    />
  );
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
