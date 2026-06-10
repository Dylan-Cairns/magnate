import type { CSSProperties } from 'react';
import { useEffect, useRef, useState } from 'react';
import type { Suit } from '../../engine/types';
import '../../styles/d6-die.css';
import { SUIT_TOKEN_BG } from './TokenComponents';
import { SUIT_ICON_BY_SUIT } from '../suitIcons';

const SUITS_BY_FACE: [Suit, Suit, Suit, Suit, Suit, Suit] = [
  'Moons', 'Suns', 'Waves', 'Leaves', 'Wyrms', 'Knots',
];

const SUIT_TO_FACE: Record<Suit, number> = {
  Moons: 1, Suns: 2, Waves: 3, Leaves: 4, Wyrms: 5, Knots: 6,
};

// rotateX/Y in degrees to bring each face to face the camera
const FACE_OFFSET: Record<number, { x: number; y: number }> = {
  1: { x: 0, y: 0 },
  2: { x: 0, y: 180 },
  3: { x: 0, y: -90 },
  4: { x: 0, y: 90 },
  5: { x: -90, y: 0 },
  6: { x: 90, y: 0 },
};

// Show three faces at rest so the die reads as 3D before any roll
const INITIAL_ROT = { x: -15, y: 20 };

export function D6Die({ suit }: { suit: Suit | undefined }) {
  const prevSuitRef = useRef<Suit | undefined>(undefined);
  const [rotX, setRotX] = useState(INITIAL_ROT.x);
  const [rotY, setRotY] = useState(INITIAL_ROT.y);
  const [rotZ, setRotZ] = useState(0);

  useEffect(() => {
    if (suit === prevSuitRef.current) return;
    prevSuitRef.current = suit;

    if (suit === undefined) {
      setRotX(prev => Math.round(prev / 360) * 360 + INITIAL_ROT.x);
      setRotY(prev => Math.round(prev / 360) * 360 + INITIAL_ROT.y);
      setRotZ(prev => Math.round(prev / 360) * 360);
      return;
    }

    const face = SUIT_TO_FACE[suit];
    const { x: faceX, y: faceY } = FACE_OFFSET[face];
    // Round to nearest multiple of 360 then add 720 for two full spins before landing.
    // Z adds a 720° tumble (always a multiple of 360, so it doesn't affect resting face).
    setRotX(prev => Math.round(prev / 360) * 360 + 720 + faceX);
    setRotY(prev => Math.round(prev / 360) * 360 + 720 + faceY);
    setRotZ(prev => prev - 720);
  }, [suit]);

  return (
    <div
      className="die-scene-d6"
      aria-label={suit !== undefined ? `d6: ${suit}` : 'd6'}
    >
      <div
        className="die-d6"
        style={{ transform: `rotateX(${rotX}deg) rotateY(${rotY}deg) rotateZ(${rotZ}deg)` }}
      >
        {SUITS_BY_FACE.map((faceSuit, i) => (
          <div
            key={faceSuit}
            className={`die-face die-face-d6-${i + 1}`}
            style={{ '--suit-bg': SUIT_TOKEN_BG[faceSuit] } as CSSProperties}
          >
            <img src={SUIT_ICON_BY_SUIT[faceSuit]} alt="" className="die-suit-icon" />
          </div>
        ))}
      </div>
    </div>
  );
}
