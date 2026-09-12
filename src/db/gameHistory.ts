import type { FinalScore, PlayerId, Ruleset } from '../engine/types';
import { db, type GameRecord, type WinnerOutcome } from './db';

export interface RecordGameParams {
  sessionId: string;
  score: FinalScore;
  humanPlayerId: PlayerId;
  botProfileId: string;
  botLabel: string;
  ruleset: Ruleset;
}

export async function recordGame(params: RecordGameParams): Promise<void> {
  const { score, humanPlayerId, botProfileId, botLabel, ruleset } = params;
  const botPlayerId: PlayerId =
    humanPlayerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';

  const winner: WinnerOutcome =
    score.winner === humanPlayerId
      ? 'player'
      : score.winner === 'Draw'
        ? 'draw'
        : 'bot';

  const record: GameRecord = {
    sessionId: params.sessionId,
    timestamp: Date.now(),
    winner,
    decidedBy: score.decidedBy,
    botProfileId,
    botLabel,
    ruleset,
    playerDistricts: score.districtPoints[humanPlayerId],
    botDistricts: score.districtPoints[botPlayerId],
    playerRankTotal: score.rankTotals[humanPlayerId],
    botRankTotal: score.rankTotals[botPlayerId],
    playerResources: score.resourceTotals[humanPlayerId],
    botResources: score.resourceTotals[botPlayerId],
  };

  await db.transaction('rw', db.games, async () => {
    if (await db.games.where('sessionId').equals(params.sessionId).first())
      return;
    await db.games.add(record);
  });
}

export async function getGames(): Promise<GameRecord[]> {
  return db.games.orderBy('timestamp').reverse().toArray();
}
