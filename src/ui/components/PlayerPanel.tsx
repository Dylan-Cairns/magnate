import { memo, type CSSProperties } from 'react';

import type {
  FinalScore,
  ObservedPlayerState,
  PlayerId,
} from '../../engine/types';
import { useHighlightClass } from './ActionHighlights';
import { playerDisplayName, winnerDisplayName } from '../playerDisplay';
import { CardTile, type CardPerspective } from './CardTile';

const HAND_FAN_MAX_ANGLE_STEP_DEG = 4;
const HAND_FAN_TOTAL_SPREAD_DEG = 10;

function handFanAngleStepDeg(slotCount: number): number {
  if (slotCount <= 1) {
    return 0;
  }
  return Math.min(
    HAND_FAN_MAX_ANGLE_STEP_DEG,
    HAND_FAN_TOTAL_SPREAD_DEG / (slotCount - 1)
  );
}

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

export const PlayerPanel = memo(function PlayerPanel({
  player,
  isActive,
  score,
  terminal,
  handSlotCount,
  humanPlayerId,
  botPlayerId,
  animateDeedProgress = true,
  animationsEnabled = true,
}: {
  player: ObservedPlayerState;
  isActive: boolean;
  score: FinalScore;
  terminal: boolean;
  handSlotCount: number;
  humanPlayerId: PlayerId;
  botPlayerId: PlayerId;
  animateDeedProgress?: boolean;
  animationsEnabled?: boolean;
}) {
  const highlightClass = useHighlightClass();
  const handCardCount = player.handHidden
    ? player.handCount
    : player.hand.length;
  // Fan the cards that are actually present and keep the group centered; a
  // single invisible anchor marks where the next drawn card will land.
  const fanAngleStep = handFanAngleStepDeg(handCardCount);
  const fanMidpoint = (handCardCount - 1) / 2;
  const showDrawAnchor = handCardCount < handSlotCount;
  const cardPerspective: CardPerspective =
    player.id === botPlayerId ? 'bot' : 'human';
  const districtScore = score.districtPoints[player.id];
  const scoreHeadline = terminal ? 'Winner' : 'Leader';
  const title = playerDisplayName(player.id, humanPlayerId);
  const winnerLabel = winnerDisplayName(score.winner, humanPlayerId);

  const fanSlotStyle = (offset: number, angleStep: number): CSSProperties => {
    return {
      '--hand-fan-x': `calc(var(--hand-fan-step) * ${offset})`,
      '--hand-fan-angle': `${offset * angleStep}deg`,
    } as CSSProperties;
  };

  return (
    <section
      className={`player-panel${isActive ? ' is-active' : ''}`}
      data-player-id={player.id}
    >
      <header className="player-header">
        <h2>{title}</h2>
        <div className="player-score-wrap">
          <span className="engraving" tabIndex={0}>
            {districtScore} VP
          </span>
          <section
            className="player-score-popover"
            role="tooltip"
            aria-label="Score details"
          >
            <p className="score-result">
              {scoreHeadline}: <strong>{winnerLabel}</strong> ({score.decidedBy}
              )
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
          <div
            className={`card-row-wrap fixed-slots hand-fan${animationsEnabled ? '' : ' is-static'}`}
          >
            {Array.from({ length: handCardCount }).map((_, index) => {
              const slotStyle = fanSlotStyle(index - fanMidpoint, fanAngleStep);
              if (player.handHidden) {
                return (
                  <div
                    key={`hidden-${player.id}-${index}`}
                    className="hand-fan-slot"
                    style={slotStyle}
                  >
                    <CardTile
                      hidden
                      handOwnerId={player.id}
                      handSlotKind="hidden"
                    />
                  </div>
                );
              }

              const cardId = player.hand[index];
              if (!cardId) {
                return null;
              }
              const raised =
                player.id === humanPlayerId &&
                highlightClass({ kind: 'hand-card', cardId }).length > 0;
              return (
                <div
                  key={`hand-${player.id}-${cardId}`}
                  className={`hand-fan-slot${raised ? ' is-raised' : ''}`}
                  style={slotStyle}
                >
                  <CardTile
                    cardId={cardId}
                    perspective={cardPerspective}
                    handOwnerId={player.id}
                    handCardId={cardId}
                    highlightTarget={
                      player.id === humanPlayerId
                        ? { kind: 'hand-card', cardId }
                        : undefined
                    }
                    handSlotKind="occupied"
                    animateDeedProgress={animateDeedProgress}
                    animationsEnabled={animationsEnabled}
                  />
                </div>
              );
            })}
            {showDrawAnchor ? (
              <div
                key={`hand-anchor-${player.id}`}
                className="hand-fan-slot"
                style={fanSlotStyle(handCardCount / 2, 0)}
                data-hand-owner-id={player.id}
                data-hand-slot-kind="empty"
              />
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
});
