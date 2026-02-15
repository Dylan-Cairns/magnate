import cubeDieIcon from '../../assets/icons/cube.png';
import dodecahedronDieIcon from '../../assets/icons/dodecahedron.png';
import type { Suit } from '../../engine/types';
import { TokenChip } from './TokenComponents';

export function RollResult({
  roll,
  taxSuit,
}: {
  roll: { die1: number; die2: number } | undefined;
  taxSuit: Suit | undefined;
}) {
  if (!roll) {
    return <p className="roll-value">-</p>;
  }

  return (
    <div className="roll-value" aria-label="Roll result">
      <span className="roll-item">
        <span className="roll-die-shell roll-die-shell-d10" aria-hidden="true">
          <img src={dodecahedronDieIcon} alt="" title="d10" className="roll-die-icon" />
        </span>
        <strong>{roll.die1}</strong>
      </span>
      <span className="roll-item">
        <span className="roll-die-shell roll-die-shell-d10" aria-hidden="true">
          <img src={dodecahedronDieIcon} alt="" title="d10" className="roll-die-icon" />
        </span>
        <strong>{roll.die2}</strong>
      </span>
      <span className="roll-item">
        <span className="roll-die-shell roll-die-shell-d6" aria-hidden="true">
          <img src={cubeDieIcon} alt="" title="d6" className="roll-die-icon" />
        </span>
        {taxSuit ? <TokenChip suit={taxSuit} count={1} compact className="roll-tax-chip" /> : <strong>-</strong>}
      </span>
    </div>
  );
}
