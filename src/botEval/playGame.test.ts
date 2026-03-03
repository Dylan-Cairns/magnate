import { describe, expect, it } from 'vitest';

import { heuristicPolicy } from '../policies/heuristicPolicy';
import { createPolicyFromBotSpec, type BotSpec } from '../policies/botSpec';
import type { ActionPolicy } from '../policies/types';
import { playGame, type RuntimeBot } from './playGame';

describe('TypeScript bot game runner', () => {
  it('replays a full game deterministically from seed and bot ids', async () => {
    const first = await runDeterministicGame();
    const second = await runDeterministicGame();

    expect(actionKeys(first)).toEqual(actionKeys(second));
    expect(first.finalScore).toEqual(second.finalScore);
    expect(first.transcript.length).toBeGreaterThan(0);
    expect(
      first.transcript.every((decision) => decision.legalActionCount >= 1)
    ).toBe(true);
  });

  it('rejects an action that is not legal in the current state', async () => {
    const illegalPolicy: ActionPolicy = {
      selectAction() {
        return { type: 'end-turn' };
      },
    };

    await expect(
      playGame({
        gameId: 'illegal-action',
        seed: 'illegal-action',
        firstPlayer: 'PlayerA',
        botBySeat: {
          PlayerA: runtimeBot(
            { id: 'illegal', kind: 'heuristic' },
            illegalPolicy
          ),
          PlayerB: runtimeBotFor({ id: 'random-b', kind: 'random' }),
        },
      })
    ).rejects.toThrow('selected illegal action');
  });

  it('emits timed heartbeats while a policy is selecting actions', async () => {
    let currentTime = 0;
    const heartbeats: string[] = [];
    const heartbeatPolicy: ActionPolicy = {
      selectAction(context) {
        currentTime += 31_000;
        context.onProgress?.();
        return heuristicPolicy.selectAction(context);
      },
    };

    await playGame({
      gameId: 'heartbeat-game',
      seed: 'heartbeat-game',
      firstPlayer: 'PlayerA',
      botBySeat: {
        PlayerA: runtimeBot(
          { id: 'heartbeat-a', kind: 'heuristic' },
          heartbeatPolicy
        ),
        PlayerB: runtimeBot(
          { id: 'heartbeat-b', kind: 'heuristic' },
          heartbeatPolicy
        ),
      },
      now: () => currentTime,
      progressIntervalMs: 30_000,
      onHeartbeat(heartbeat) {
        heartbeats.push(heartbeat.gameId);
      },
    });

    expect(heartbeats.length).toBeGreaterThan(0);
    expect(new Set(heartbeats)).toEqual(new Set(['heartbeat-game']));
  });
});

async function runDeterministicGame() {
  return playGame({
    gameId: 'deterministic-game',
    seed: 'deterministic-game',
    firstPlayer: 'PlayerA',
    botBySeat: {
      PlayerA: runtimeBotFor({ id: 'heuristic-a', kind: 'heuristic' }),
      PlayerB: runtimeBotFor({ id: 'random-b', kind: 'random' }),
    },
  });
}

function runtimeBotFor(spec: BotSpec): RuntimeBot {
  return runtimeBot(spec, createPolicyFromBotSpec(spec));
}

function runtimeBot(spec: BotSpec, policy: ActionPolicy): RuntimeBot {
  return { spec, policy };
}

function actionKeys(game: Awaited<ReturnType<typeof runDeterministicGame>>) {
  return game.transcript.map((decision) => decision.actionKey);
}
