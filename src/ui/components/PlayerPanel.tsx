import type { FinalScore, ObservedPlayerState, PlayerId } from '../../engine/types';
import { CardTile, type CardPerspective } from './CardTile';

function ScoreLine({ label, a, b }: { label: string; a: number; b: number }) {
  return (
    <p className="score-line">
      <span>{label}</span>
      <strong>
        A {a} - B {b}
      </strong>
    </p>
  );
}

export function PlayerPanel({
  title,
  player,
  isActive,
  score,
  terminal,
  handSlotCount,
  botPlayerId,
}: {
  title: string;
  player: ObservedPlayerState;
  isActive: boolean;
  score: FinalScore;
  terminal: boolean;
  handSlotCount: number;
  botPlayerId: PlayerId;
}) {
  const handCardCount = player.handHidden ? player.handCount : player.hand.length;
  const handSlots = Math.max(handSlotCount, handCardCount);
  const cardPerspective: CardPerspective = player.id === botPlayerId ? 'bot' : 'human';
  const districtScore = score.districtPoints[player.id];
  const scoreHeadline = terminal ? 'Winner' : 'Leader';

  return (
    <section className={`player-panel${isActive ? ' is-active' : ''}`}>
      <header className="player-header">
        <div className="player-title-line">
          <h2>{title}</h2>
          <div className="player-score-wrap">
            <span className="player-score-badge" tabIndex={0}>
              {districtScore} VP
            </span>
            <section className="player-score-popover" role="tooltip" aria-label="Score details">
              <p className="score-result">
                {scoreHeadline}: <strong>{score.winner}</strong> ({score.decidedBy})
              </p>
              <ScoreLine label="Districts" a={score.districtPoints.PlayerA} b={score.districtPoints.PlayerB} />
              <ScoreLine label="Rank Total" a={score.rankTotals.PlayerA} b={score.rankTotals.PlayerB} />
              <ScoreLine label="Resources" a={score.resourceTotals.PlayerA} b={score.resourceTotals.PlayerB} />
            </section>
          </div>
        </div>
        <span className="player-meta">{player.id}</span>
      </header>

      <div className="player-row">
        <div className="player-section hand-section">
          <h3>Hand</h3>
          <div className="card-row-wrap fixed-slots">
            {Array.from({ length: handSlots }).map((_, index) => {
              if (player.handHidden) {
                return index < player.handCount ? (
                  <CardTile key={`hidden-${player.id}-${index}`} hidden />
                ) : (
                  <CardTile key={`hidden-slot-${player.id}-${index}`} placeholder />
                );
              }

              const cardId = player.hand[index];
              if (!cardId) {
                return <CardTile key={`hand-slot-${player.id}-${index}`} placeholder />;
              }
              return (
                <CardTile
                  key={`hand-${player.id}-${cardId}-${index}`}
                  cardId={cardId}
                  perspective={cardPerspective}
                />
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
