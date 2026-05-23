import type { CSSProperties } from 'react';

import type {
  FinalScore,
  ObservedPlayerState,
  PlayerId,
} from '../../engine/types';
import { playerDisplayName, winnerDisplayName } from '../playerDisplay';
import { CardTile } from './CardTile';

function ScoreLine({ label, a, b }: { label: string; a: number; b: number }) {
  return (
    <p className="score-line">
      <span>{label}</span>
      <strong>
        P {a} - B {b}
      </strong>
    </p>
  );
}

export function BotHandPanel({
  player,
  isActive,
  score,
  terminal,
  humanPlayerId,
  botPlayerId,
}: {
  player: ObservedPlayerState;
  isActive: boolean;
  score: FinalScore;
  terminal: boolean;
  humanPlayerId: PlayerId;
  botPlayerId: PlayerId;
}) {
  const districtScore = score.districtPoints[player.id];
  const scoreHeadline = terminal ? 'Winner' : 'Leader';
  const title = playerDisplayName(player.id, humanPlayerId);
  const winnerLabel = winnerDisplayName(score.winner, humanPlayerId);
  const hiddenCardCount = Math.max(0, player.handCount);
  const fanSpreadDegrees = Math.min(
    18,
    Math.max(0, (hiddenCardCount - 1) * 6)
  );

  return (
    <section
      className={`player-panel bot-hand-panel${isActive ? ' is-active' : ''}`}
      data-player-id={player.id}
    >
      <header className="player-header">
        <h2>{title}</h2>
        <div className="player-score-wrap">
          <span className="player-score-badge" tabIndex={0}>
            {districtScore} VP
          </span>
          <section
            className="player-score-popover"
            role="tooltip"
            aria-label="Score details"
          >
            <p className="score-result">
              {scoreHeadline}: <strong>{winnerLabel}</strong> (
              {score.decidedBy})
            </p>
            <ScoreLine
              label="Districts"
              a={score.districtPoints.PlayerA}
              b={score.districtPoints.PlayerB}
            />
            <ScoreLine
              label="Rank Total"
              a={score.rankTotals.PlayerA}
              b={score.rankTotals.PlayerB}
            />
            <ScoreLine
              label="Resources"
              a={score.resourceTotals.PlayerA}
              b={score.resourceTotals.PlayerB}
            />
          </section>
        </div>
      </header>

      <div className="bot-hand-card-wrap" aria-label="Bot hand">
        <div
          className="bot-hand-fan"
          style={{ '--bot-hand-count': hiddenCardCount } as CSSProperties}
        >
          <div
            className="bot-hand-animation-anchor"
            aria-hidden="true"
            data-hand-owner-id={botPlayerId}
            data-hand-animation-anchor="true"
          />
          {Array.from({ length: hiddenCardCount }).map((_, index) => {
            const angle =
              hiddenCardCount <= 1
                ? 0
                : -fanSpreadDegrees / 2 +
                  (fanSpreadDegrees * index) / (hiddenCardCount - 1);
            return (
              <div
                key={`bot-hidden-card-${index}`}
                className="bot-hand-fan-card"
                style={
                  {
                    '--bot-hand-card-angle': `${angle}deg`,
                    '--bot-hand-card-z': index + 1,
                  } as CSSProperties
                }
              >
                <CardTile
                  hidden
                  handOwnerId={botPlayerId}
                  handSlotKind="hidden"
                />
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
