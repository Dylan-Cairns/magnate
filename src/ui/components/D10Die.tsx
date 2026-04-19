import { useEffect, useState } from 'react';
import '../../styles/d10-die.css';

const SIDE_ANGLE = 72; // 360 / 5 faces
const TILT = 45;       // degrees the die leans back at rest

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
const INITIAL_ROT = { x: -TILT, y: 15 };

export function D10Die({
  result,
  rollKey,
}: {
  result: number | undefined;
  // Changing rollKey triggers animation even when result is the same number.
  // Uses rollId from IncomeRollResult — increments with rngCursor on each real roll.
  rollKey?: number;
}) {
  const [rotX, setRotX] = useState(INITIAL_ROT.x);
  const [rotY, setRotY] = useState(INITIAL_ROT.y);
  const [rotZ, setRotZ] = useState(0);

  useEffect(() => {
    if (result === undefined) return;
    const { x: faceX, y: faceY } = getFaceOffset(result);
    // 360 added to X (one full tilt), 720 to Y (two full spins).
    // Z adds a 720° tumble (always a multiple of 360, so it doesn't affect resting face).
    setRotX(prev => Math.round(prev / 360) * 360 + 360 + faceX);
    setRotY(prev => Math.round(prev / 360) * 360 + 720 + faceY);
    setRotZ(prev => prev - 720);
  // rollKey is the primary trigger; result provides the face target.
  // They always update together so including both is safe and lint-clean.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rollKey]);

  return (
    <div
      className="die-scene-d10"
      aria-label={result !== undefined ? `d10: ${result}` : 'd10'}
    >
      <div
        className="die-d10"
        style={{ transform: `rotateX(${rotX}deg) rotateY(${rotY}deg) rotateZ(${rotZ}deg)` }}
      >
        {Array.from({ length: 10 }, (_, i) => (
          <div key={i} className={`die-face-d10 die-face-d10-${i}`}>
            <span className="die-face-number">{i + 1}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
