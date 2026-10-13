import { describe, expect, it } from 'vitest';
import { legalActions } from '../engine/actionBuilders';
import { actionStableKey } from '../engine/actionSurface';
import type { GameAction } from '../engine/types';
import {
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../engine/__tests__/fixtures';
import { toPlayerView } from '../engine/view';
import { botCandidateActions } from './policyUtils';
import { randomPolicy } from './randomPolicy';
import { heuristicPolicy } from './heuristicPolicy';
import { createSearchPolicy } from './searchPolicy';
import { createTdValuePolicy } from './tdValuePolicy';
import { createTdRootSearchPolicy } from './tdRootSearchPolicy';
import { createWorkerBackedPolicy } from './workerPolicy';
import {
  runRolloutSearchTask,
  runRolloutSearchTaskResumable,
  type RolloutSearchTraceStep,
  type RolloutSearchWorkerTask,
} from './rolloutSearchCore';

const state = makeGameState({
  phase: 'ActionWindow',
  players: [
    makePlayer(PLAYER_A, {
      hand: ['6', '7', '8', '9', '41'],
      resources: makeResources({
        Moons: 6,
        Suns: 6,
        Waves: 6,
        Leaves: 6,
        Wyrms: 6,
        Knots: 6,
      }),
    }),
    makePlayer(PLAYER_B),
  ],
  ruleset: 'extended',
});

describe('bot candidate pruning', () => {
  it('removes all three rank-2 purchases, retaining other actions in engine order', () => {
    const legal = legalActions(state);
    const candidates = botCandidateActions(legal);
    for (const cardId of ['6', '7', '8']) {
      expect(
        legal.some((a) => a.type === 'buy-deed' && a.cardId === cardId)
      ).toBe(true);
      expect(
        candidates.some((a) => a.type === 'buy-deed' && a.cardId === cardId)
      ).toBe(false);
      expect(
        candidates.some(
          (a) => a.type === 'develop-outright' && a.cardId === cardId
        )
      ).toBe(true);
    }
    for (const cardId of ['9', '41']) {
      expect(
        candidates.some((a) => a.type === 'buy-deed' && a.cardId === cardId)
      ).toBe(true);
    }
    expect(candidates).toEqual(
      legal.filter(
        (a) => !(a.type === 'buy-deed' && ['6', '7', '8'].includes(a.cardId))
      )
    );
    expect(legalActions(state)).toEqual(legal);
  });

  const noModel = async (): Promise<never> => {
    throw new Error('Pruned forced action must not load a model');
  };
  it.each([
    ['random', randomPolicy],
    ['heuristic', heuristicPolicy],
    ['search', createSearchPolicy()],
    ['TD value', createTdValuePolicy({ loadModel: noModel })],
    ['TD search', createTdRootSearchPolicy({ loadModel: noModel })],
    [
      'worker',
      createWorkerBackedPolicy(
        { id: 'pruning-test', kind: 'random' },
        {
          createWorker: () => {
            throw new Error('Forced action must not spawn a worker');
          },
        }
      ),
    ],
  ] as const)(
    '%s chooses the build when a rank-2 deed is also offered',
    async (_name, policy) => {
      const legal = legalActions(state);
      const deed = legal.find(
        (a): a is Extract<GameAction, { type: 'buy-deed' }> =>
          a.type === 'buy-deed' && a.cardId === '6'
      )!;
      const build = legal.find(
        (a) =>
          a.type === 'develop-outright' &&
          a.cardId === '6' &&
          a.districtId === deed.districtId
      )!;
      expect(
        await policy.selectAction({
          state,
          view: toPlayerView(state, PLAYER_A),
          legalActions: [deed, build],
          random: () => 0,
        })
      ).toEqual(build);
    }
  );

  it.each([
    ['serial', runRolloutSearchTask],
    ['resumable', runRolloutSearchTaskResumable],
  ] as const)(
    '%s rollouts exclude rank-2 deeds during random exploration',
    (_name, run) => {
      const rootAction = legalActions(state).find((a) => a.type === 'trade')!;
      const trace: RolloutSearchTraceStep[] = [];
      const task: RolloutSearchWorkerTask = {
        kind: 'rollout-search',
        contextId: 'pruning-test',
        visitIndex: 0,
        actionVisitIndex: 0,
        scenarioIndex: 0,
        worldIndex: 0,
        engineSeed: 'pruning-engine',
        rootPlayer: PLAYER_A,
        rootAction,
        rootActionKey: actionStableKey(rootAction),
        config: {
          worlds: 1,
          rollouts: 1,
          depth: 1,
          maxRootActions: 1,
          rolloutEpsilon: 1,
        },
      };
      run(task, [state], () => 0, undefined, {
        onStep: (step) => trace.push(step),
      });
      expect(trace).toHaveLength(2);
      expect(
        trace[1].legalActionKeys.some((key) =>
          key.startsWith('develop-outright:6:')
        )
      ).toBe(true);
      expect(
        trace[1].legalActionKeys.some((key) => /^buy-deed:[678]:/.test(key))
      ).toBe(false);
    }
  );
});
