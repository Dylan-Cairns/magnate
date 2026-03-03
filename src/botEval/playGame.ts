import { performance } from 'node:perf_hooks';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey } from '../engine/actionSurface';
import { isTerminal } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import type { PlayerId } from '../engine/types';
import { toPlayerView } from '../engine/view';
import type { BotSpec } from '../policies/botSpec';
import { policyRandomForState } from '../policies/policyRandom';
import type { ActionPolicy } from '../policies/types';
import type { PlayedGame } from './types';

const DEFAULT_MAX_DECISIONS_PER_GAME = 10_000;

export interface RuntimeBot {
  spec: BotSpec;
  policy: ActionPolicy;
}

export interface PlayGameOptions {
  gameId: string;
  seed: string;
  firstPlayer: PlayerId;
  botBySeat: Record<PlayerId, RuntimeBot>;
  maxDecisions?: number;
  now?: () => number;
}

export async function playGame({
  gameId,
  seed,
  firstPlayer,
  botBySeat,
  maxDecisions = DEFAULT_MAX_DECISIONS_PER_GAME,
  now = () => performance.now(),
}: PlayGameOptions): Promise<PlayedGame> {
  if (!Number.isInteger(maxDecisions) || maxDecisions <= 0) {
    throw new Error('maxDecisions must be a positive integer.');
  }

  const gameStartedAt = now();
  const transcript: PlayedGame['transcript'] = [];
  let state = createSession(seed, firstPlayer);

  while (!isTerminal(state)) {
    if (transcript.length >= maxDecisions) {
      throw new Error(
        `Game ${gameId} exceeded maxDecisions=${String(maxDecisions)}.`
      );
    }

    const activePlayerId = state.players[state.activePlayerIndex]?.id;
    if (activePlayerId !== 'PlayerA' && activePlayerId !== 'PlayerB') {
      throw new Error(`Game ${gameId} could not resolve its active player.`);
    }

    const bot = botBySeat[activePlayerId];
    const actions = legalActions(state);
    if (actions.length === 0) {
      throw new Error(
        `Game ${gameId} has no legal actions for ${activePlayerId} during ${state.phase}.`
      );
    }

    const selectedAt = now();
    const selected = await bot.policy.selectAction({
      state,
      view: toPlayerView(state, activePlayerId),
      legalActions: actions,
      random: policyRandomForState(state, bot.spec.id),
    });
    const latencyMs = now() - selectedAt;
    if (!selected) {
      throw new Error(
        `Bot ${bot.spec.id} did not select an action in game ${gameId}.`
      );
    }

    const actionKey = actionStableKey(selected);
    const canonicalAction = actions.find(
      (action) => actionStableKey(action) === actionKey
    );
    if (!canonicalAction) {
      throw new Error(
        `Bot ${bot.spec.id} selected illegal action ${actionKey} in game ${gameId}.`
      );
    }

    transcript.push({
      decisionIndex: transcript.length,
      turn: state.turn,
      phase: state.phase,
      activePlayerId,
      botId: bot.spec.id,
      actionKey,
      latencyMs,
    });
    state = stepToDecision(state, canonicalAction);
  }

  if (!state.finalScore) {
    throw new Error(`Terminal game ${gameId} is missing its final score.`);
  }

  return {
    gameId,
    seed,
    firstPlayer,
    botBySeat: {
      PlayerA: botBySeat.PlayerA.spec.id,
      PlayerB: botBySeat.PlayerB.spec.id,
    },
    transcript,
    finalScore: structuredClone(state.finalScore),
    turns: state.turn,
    elapsedMs: now() - gameStartedAt,
  };
}
