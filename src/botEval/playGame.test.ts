import { describe, expect, it } from 'vitest';

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
