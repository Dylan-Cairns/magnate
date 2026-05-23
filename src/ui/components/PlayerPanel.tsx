import type {
  FinalScore,
  ObservedPlayerState,
  PlayerId,
} from '../../engine/types';
import { playerDisplayName, winnerDisplayName } from '../playerDisplay';
import { CardTile, type CardPerspective } from './CardTile';

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

export function PlayerPanel({
  player,
  isActive,
  score,
  terminal,
  handSlotCount,
  humanPlayerId,
  botPlayerId,
  animateDeedProgress = true,
}: {
  player: ObservedPlayerState;
  isActive: boolean;
  score: FinalScore;
  terminal: boolean;
  handSlotCount: number;
  humanPlayerId: PlayerId;
  botPlayerId: PlayerId;
  animateDeedProgress?: boolean;
}) {
  const handCardCount = player.handHidden
    ? player.handCount
    : player.hand.length;
  const handSlots = Math.max(handSlotCount, handCardCount);
  const cardPerspective: CardPerspective =
    player.id === botPlayerId ? 'bot' : 'human';
  const districtScore = score.districtPoints[player.id];
  const scoreHeadline = terminal ? 'Winner' : 'Leader';
  const title = playerDisplayName(player.id, humanPlayerId);
  const winnerLabel = winnerDisplayName(score.winner, humanPlayerId);

  return (
    <section
      className={`player-panel${isActive ? ' is-active' : ''}`}
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

      <div className="player-row">
        <div className="player-section hand-section" aria-label="Hand">
          <div className="card-row-wrap fixed-slots">
            {Array.from({ length: handSlots }).map((_, index) => {
              if (player.handHidden) {
                return index < player.handCount ? (
                  <CardTile
                    key={`hidden-${player.id}-${index}`}
                    hidden
                    handOwnerId={player.id}
                    handSlotKind="hidden"
                  />
                ) : (
                  <CardTile
                    key={`hidden-slot-${player.id}-${index}`}
                    placeholder
                    handOwnerId={player.id}
                    handSlotKind="empty"
                  />
                );
              }

              const cardId = player.hand[index];
              if (!cardId) {
                return (
                  <CardTile
                    key={`hand-slot-${player.id}-${index}`}
                    placeholder
                    handOwnerId={player.id}
                    handSlotKind="empty"
                  />
                );
              }
              return (
                <CardTile
                  key={`hand-${player.id}-${cardId}-${index}`}
                  cardId={cardId}
                  perspective={cardPerspective}
                  handOwnerId={player.id}
                  handCardId={cardId}
                  handSlotKind="occupied"
                  animateDeedProgress={animateDeedProgress}
                />
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
