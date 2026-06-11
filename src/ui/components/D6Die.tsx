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

// rotateX/Y to bring each face toward the camera with a slight Y offset so it never looks flat.
// Side faces (1-4) keep x:0 so the die stays upright. Top/bottom (5-6) need a modest X tilt.
const FACE_OFFSET: Record<number, { x: number; y: number }> = {
  1: { x: 0, y: 15 },
  2: { x: 0, y: 195 },
  3: { x: 0, y: -75 },
  4: { x: 0, y: 105 },
  5: { x: -90, y: 15 },
  6: { x: 90, y: 15 },
};

const INITIAL_ROT = { x: 0, y: 15 };

export function D6Die({ suit, pulsing, dimmed }: { suit: Suit | undefined; pulsing?: boolean; dimmed?: boolean }) {
  const prevSuitRef = useRef<Suit | undefined>(undefined);
  const [rotX, setRotX] = useState(INITIAL_ROT.x);
  const [rotY, setRotY] = useState(INITIAL_ROT.y);
  const [rotZ, setRotZ] = useState(0);

  useEffect(() => {
    if (suit === prevSuitRef.current) return;
    prevSuitRef.current = suit;

    if (suit === undefined) {
      return; // stay at current position; parent handles dimming
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
      className={`die-scene-d6${pulsing ? ' is-pulsing' : ''}${dimmed ? ' is-dimmed' : ''}`}
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
            <div className="die-suit-circle">
              <img src={SUIT_ICON_BY_SUIT[faceSuit]} alt="" className="die-suit-icon" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
