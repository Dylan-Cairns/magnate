import type { FinalScore, PlayerId } from '../../engine/types';
import { playerDisplayName, winnerDisplayName } from '../playerDisplay';

export function TerminalScoreSummary({
  score,
  wonDistrictsByPlayer,
  humanPlayerId,
  botPlayerId,
}: {
  score: FinalScore;
  wonDistrictsByPlayer: Record<PlayerId, readonly string[]>;
  humanPlayerId: PlayerId;
  botPlayerId: PlayerId;
}) {
  return (
    <section
      className="terminal-score-summary"
      aria-label="Final score breakdown"
    >
      <p className="score-result terminal-score-winner">
        Winner: <strong>{winnerDisplayName(score.winner, humanPlayerId)}</strong>
      </p>
      <p className="score-line terminal-score-decider">
        <span>Decided By</span>
        <strong>{deciderLabel(score.decidedBy)}</strong>
      </p>

      <div className="terminal-score-players">
        {([humanPlayerId, botPlayerId] as const).map((playerId) => (
          <article
            key={`terminal-score-${playerId}`}
            className="terminal-score-player"
          >
            <h3>{playerDisplayName(playerId, humanPlayerId)}</h3>
            <p className="score-line">
              <span>Districts Won</span>
              <strong>
                {formatDistrictList(wonDistrictsByPlayer[playerId])}
              </strong>
            </p>
            <p className="score-line">
              <span>Total Properties</span>
              <strong>{score.rankTotals[playerId]}</strong>
            </p>
            <p className="score-line">
              <span>Resources</span>
              <strong>{score.resourceTotals[playerId]}</strong>
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}

function deciderLabel(decidedBy: FinalScore['decidedBy']): string {
  switch (decidedBy) {
    case 'districts':
      return 'Districts';
    case 'rank-total':
      return 'Total Properties';
    case 'resources':
      return 'Resources';
    case 'draw':
      return 'Tie';
  }
}

function formatDistrictList(districtIds: readonly string[]): string {
  if (districtIds.length === 0) {
    return 'None';
  }
  return districtIds.join(', ');
}
