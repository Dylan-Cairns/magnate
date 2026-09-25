import type { Suit } from '../../engine/types';
import { reportImageRenderFailure } from '../cardImages';
import { SUIT_ICON_BY_SUIT, SUIT_TOKEN_BG } from '../suitIcons';

/** One coordinate system for the rim and artwork, shared with the deck map. */
export function SuitTokenFace({
  suit,
  x,
  y,
  size = '100%',
  empty = false,
}: {
  suit: Suit;
  x?: number;
  y?: number;
  size?: number | string;
  empty?: boolean;
}) {
  return (
    <svg
      className="suit-token-face"
      viewBox="0 0 54 54"
      x={x}
      y={y}
      width={size}
      height={size}
      role="img"
      aria-label={suit}
    >
      <circle
        cx={27}
        cy={27}
        r={26.5}
        fill={SUIT_TOKEN_BG[suit]}
        stroke="var(--chip-border, #4a545f)"
        strokeWidth={1}
        strokeDasharray={empty ? '2 2' : undefined}
      />
      <image
        href={SUIT_ICON_BY_SUIT[suit]}
        x={2}
        y={2}
        width={50}
        height={50}
        preserveAspectRatio="xMidYMid meet"
        ref={(image) => {
          if (!image) return;
          const onError = () =>
            reportImageRenderFailure(SUIT_ICON_BY_SUIT[suit], `${suit} token`);
          image.addEventListener('error', onError);
          return () => image.removeEventListener('error', onError);
        }}
      />
    </svg>
  );
}
