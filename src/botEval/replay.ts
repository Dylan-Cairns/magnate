import { createPolicyFromBotSpec, type BotSpec } from '../policies/botSpec';
import { POLICY_RANDOM_SCHEME_VERSION } from '../policies/policyRandom';
import type { ActionPolicy } from '../policies/types';
import { playGame, type RuntimeBot } from './playGame';
import type { HeadToHeadArtifact, PlayedGame } from './types';

export interface ReplayDependencies {
  createPolicy?: (spec: BotSpec) => ActionPolicy;
}

export interface ReplayResult {
  gameId: string;
  decisions: number;
  winner: PlayedGame['finalScore']['winner'];
  matched: true;
}

export async function replayArtifactGame(
  artifact: HeadToHeadArtifact,
  gameId: string,
  dependencies: ReplayDependencies = {}
): Promise<ReplayResult> {
  if (artifact.policyRandomSchemeVersion !== POLICY_RANDOM_SCHEME_VERSION) {
    throw new Error(
      `Replay policy RNG scheme mismatch: expected=${POLICY_RANDOM_SCHEME_VERSION} actual=${artifact.policyRandomSchemeVersion}.`
    );
  }

  const recorded = artifact.games.find((game) => game.gameId === gameId);
  if (!recorded) {
    throw new Error(`Artifact does not include game ${gameId}.`);
  }

  const createPolicy = dependencies.createPolicy ?? createPolicyFromBotSpec;
  const specsById = new Map(
    [artifact.config.candidate, artifact.config.opponent].map((spec) => [
      spec.id,
      spec,
    ])
  );
  const replayed = await playGame({
    gameId: recorded.gameId,
    seed: recorded.seed,
    firstPlayer: recorded.firstPlayer,
    botBySeat: {
      PlayerA: runtimeBotFor(
        recorded.botBySeat.PlayerA,
        specsById,
        createPolicy
      ),
      PlayerB: runtimeBotFor(
        recorded.botBySeat.PlayerB,
        specsById,
        createPolicy
      ),
    },
    maxDecisions: artifact.config.maxDecisionsPerGame,
  });

  assertMatchingTranscript(recorded, replayed);
  if (
    JSON.stringify(recorded.finalScore) !== JSON.stringify(replayed.finalScore)
  ) {
    throw new Error(`Replay final score diverged for game ${gameId}.`);
  }

  return {
    gameId,
    decisions: replayed.transcript.length,
    winner: replayed.finalScore.winner,
    matched: true,
  };
}

function runtimeBotFor(
  botId: string,
  specsById: ReadonlyMap<string, BotSpec>,
  createPolicy: (spec: BotSpec) => ActionPolicy
): RuntimeBot {
  const spec = specsById.get(botId);
  if (!spec) {
    throw new Error(`Artifact is missing bot spec ${botId}.`);
  }
  return {
    spec,
    policy: createPolicy(spec),
  };
}

function assertMatchingTranscript(
  recorded: PlayedGame,
  replayed: PlayedGame
): void {
  const count = Math.max(
    recorded.transcript.length,
    replayed.transcript.length
  );
  for (let index = 0; index < count; index += 1) {
    const expected = recorded.transcript[index]?.actionKey;
    const actual = replayed.transcript[index]?.actionKey;
    if (expected !== actual) {
      throw new Error(
        `Replay diverged for game ${recorded.gameId} at decision ${String(index)}: expected=${String(expected)} actual=${String(actual)}.`
      );
    }
  }
}
