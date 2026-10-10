import { useEffect, useId, useRef, useState } from 'react';
import '../../styles/d10-die.css';

const SIDE_ANGLE = 72; // 360 / 5 faces
const TILT = 45; // degrees the die leans back at rest

// Container rotation (rotX, rotY) to bring face with value V to face the camera.
// Faces 1,3,5,7,9 are upper (even index 0,2,4,6,8).
// Faces 2,4,6,8,10 are lower (odd index 1,3,5,7,9).
function getFaceOffset(result: number): { x: number; y: number } {
  const index = result - 1;
  if (index % 2 === 0) {
    return { x: -TILT, y: SIDE_ANGLE * (index / 2) };
  } else {
    return { x: -(180 + TILT), y: -SIDE_ANGLE * ((index + 1) / 2) };
  }
}

// Initial tilt to show the die as 3D before any roll
const INITIAL_ROT = { x: -TILT, y: 0 };

export function D10Die({
  result,
  rollKey,
  glowing,
  dimmed,
  animationsEnabled = true,
}: {
  result: number | undefined;
  // Changing rollKey triggers animation even when result is the same number.
  // Uses rollId from IncomeRollResult — increments with rngCursor on each real roll.
  rollKey?: number | string;
  glowing?: boolean;
  dimmed?: boolean;
  animationsEnabled?: boolean;
}) {
  const rollTrigger = rollKey ?? result;
  const lastRolledTriggerRef = useRef<number | string | undefined>(undefined);
  const [rotX, setRotX] = useState(INITIAL_ROT.x);
  const [rotY, setRotY] = useState(INITIAL_ROT.y);
  const [rotZ, setRotZ] = useState(0);
  const [bounceNonce, setBounceNonce] = useState(0);

  useEffect(() => {
    if (result === undefined || rollTrigger === undefined) {
      return;
    }
    if (lastRolledTriggerRef.current === rollTrigger) {
      return;
    }

    lastRolledTriggerRef.current = rollTrigger;
    const { x: faceX, y: faceY } = getFaceOffset(result);
    // 360 added to X (one full tilt), 720 to Y (two full spins).
    // Z adds a 720deg tumble (always a multiple of 360, so it doesn't affect resting face).
    setRotX((prev) => Math.round(prev / 360) * 360 + 360 + faceX);
    setRotY((prev) => Math.round(prev / 360) * 360 + 720 + faceY);
    setRotZ((prev) => prev - 720);
    setBounceNonce((prev) => prev + 1);
  }, [result, rollTrigger]);

  const bounceClass =
    !animationsEnabled || result === undefined || bounceNonce === 0
      ? ''
      : bounceNonce % 2 === 0
        ? ' is-rolling-a'
        : ' is-rolling-b';

  return (
    <div
      className={`die-glow die-glow-d10${glowing ? ' is-glowing' : ''}${dimmed ? ' is-dimmed' : ''}`}
    >
      {glowing && <D10Glow />}
      <div
        className="die-scene-d10"
        aria-label={result !== undefined ? `d10: ${result}` : 'd10'}
      >
        <div className={`die-roll-bounce-wrap${bounceClass}`}>
          <div className="die-d10-viewport">
            <div
              className="die-d10"
              style={{
                transform: `rotateX(${rotX}deg) rotateY(${rotY}deg) rotateZ(${rotZ}deg)`,
                transition: animationsEnabled ? undefined : 'none',
              }}
            >
              {Array.from({ length: 10 }, (_, i) => (
                <div key={i} className={`die-face-d10 die-face-d10-${i}`}>
                  <span className="die-face-number">{i + 1}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function D10Glow() {
  const filterId = useId();

  return (
    <svg
      className="die-d10-glow"
      viewBox="0 0 100 100"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <filter
          id={filterId}
          x="-100%"
          y="-100%"
          width="300%"
          height="300%"
          colorInterpolationFilters="sRGB"
        >
          <feDropShadow
            dx="0"
            dy="6"
            stdDeviation="10"
            floodColor="black"
            floodOpacity="0.4"
          />
          <feDropShadow
            dx="0"
            dy="0"
            stdDeviation="3"
            style={{ floodColor: 'var(--active-ring)' }}
          />
          <feDropShadow
            dx="0"
            dy="0"
            stdDeviation="7"
            style={{ floodColor: 'var(--active-glow-outer)' }}
          />
          <feComposite in2="SourceAlpha" operator="out" />
        </filter>
      </defs>
      <polygon
        points="50,17 90,40 91,62 50,80 9,62 10,40"
        fill="white"
        filter={`url(#${filterId})`}
      />
    </svg>
  );
}
