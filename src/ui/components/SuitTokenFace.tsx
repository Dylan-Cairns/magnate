import { useId } from 'react';

import type { Suit } from '../../engine/types';
import { reportImageRenderFailure } from '../cardImages';
import {
  SUIT_ICON_BY_SUIT,
  SUIT_ICON_SIMPLIFIED_BY_SUIT,
  SUIT_TOKEN_BG,
} from '../suitIcons';

/** One coordinate system for the rim and artwork, shared with the deck map. */
const FACE_VIEWBOX = 54;
const FACE_CENTER = 27;
const FACE_RADIUS = 26.5;
// The shaded emblems carry their own outlines, and several run right up to the
// disc edge. The artwork is clipped just outside the ring and the ring drawn on
// top, so the ring — not the art — is the token's edge for every suit.
const SHADED_ART_SIZE = 50;
const SIMPLIFIED_ART_SIZE = 50;
const RING_RADIUS = 23.5;
const RING_STROKE_WIDTH = 1.5;
const SHADED_CLIP_RADIUS = 22.8;
// Matches the ink baked into the shaded emblems, so the ring reads as the same
// line weight and colour as the artwork it encloses.
const TOKEN_INK = '#221e1f';

export function SuitTokenFace({
  suit,
  x,
  y,
  size = '100%',
  simplified = false,
  className,
}: {
  suit: Suit;
  x?: number;
  y?: number;
  size?: number | string;
  simplified?: boolean;
  className?: string;
}) {
  const clipId = `suit-token-clip-${useId().replace(/:/g, '')}`;
  const artwork = simplified
    ? SUIT_ICON_SIMPLIFIED_BY_SUIT[suit]
    : SUIT_ICON_BY_SUIT[suit];
  const artSize = simplified ? SIMPLIFIED_ART_SIZE : SHADED_ART_SIZE;
  const artOffset = (FACE_VIEWBOX - artSize) / 2;
  return (
    <svg
      className={`suit-token-face${className ? ` ${className}` : ''}`}
      viewBox={`0 0 ${FACE_VIEWBOX} ${FACE_VIEWBOX}`}
      x={x}
      y={y}
      width={size}
      height={size}
      role="img"
      aria-label={suit}
    >
      {!simplified && (
        <defs>
          <clipPath id={clipId}>
            <circle cx={FACE_CENTER} cy={FACE_CENTER} r={SHADED_CLIP_RADIUS} />
          </clipPath>
        </defs>
      )}
      <circle
        cx={FACE_CENTER}
        cy={FACE_CENTER}
        r={FACE_RADIUS}
        fill={SUIT_TOKEN_BG[suit]}
      />
      <image
        href={artwork}
        x={artOffset}
        y={artOffset}
        width={artSize}
        height={artSize}
        preserveAspectRatio="xMidYMid meet"
        clipPath={!simplified ? `url(#${clipId})` : undefined}
        ref={(image) => {
          if (!image) return;
          const onError = () =>
            reportImageRenderFailure(artwork, `${suit} token`);
          image.addEventListener('error', onError);
          return () => image.removeEventListener('error', onError);
        }}
      />
      <circle
        cx={FACE_CENTER}
        cy={FACE_CENTER}
        r={RING_RADIUS}
        fill="none"
        stroke={TOKEN_INK}
        strokeWidth={RING_STROKE_WIDTH}
      />
    </svg>
  );
}
