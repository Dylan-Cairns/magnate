import { describe, expect, it } from 'vitest';

import {
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../engine/__tests__/fixtures';
import type { GameLogEntry } from '../engine/types';
import {
  botRandomForState,
  createBrowserSession,
  errorMessage,
  humanActionsAcceptingInputForState,
  makeBrowserSessionSeed,
  shouldScheduleBotAction,
  withSeedLogPrefix,
} from './gameControllerModel';

describe('gameControllerModel', () => {
  it('builds browser seeds from the supplied timestamp', () => {
    expect(makeBrowserSessionSeed(1234)).toBe('seed-1234');
  });

  it('creates an ordinary browser session with the requested seed and first player', () => {
    const session = createBrowserSession('session-characterization', PLAYER_B);

    expect(session.seed).toBe('session-characterization');
    expect(session.players[session.activePlayerIndex]?.id).toBe(PLAYER_B);
  });

  it('prefixes timeline logs with the seed exactly once', () => {
    const state = makeGameState({ seed: 'controller-test' });
    const engineEntry: GameLogEntry = {
      turn: 1,
      player: 'PlayerA',
      phase: 'ActionWindow',
      summary: 'engine entry',
    };

    const prefixed = withSeedLogPrefix(state, [engineEntry], PLAYER_A);

    expect(prefixed.map((entry) => entry.summary)).toEqual([
      'Seed controller-test',
      'engine entry',
    ]);
    expect(withSeedLogPrefix(state, prefixed, PLAYER_A)).toEqual(prefixed);
  });

  it('creates deterministic bot randomness from state and profile identity', () => {
    const state = makeGameState({ seed: 'bot-random-test', turn: 4 });
    const sample = (profileId: string) => {
      const random = botRandomForState(state, profileId);
      return [random(), random(), random()];
    };

    expect(sample('heuristic')).toEqual(sample('heuristic'));
    expect(sample('heuristic')).not.toEqual(sample('random-legal'));
  });

  it('exposes human actions only on the human non-terminal turn', () => {
    const normal = actionsFor(makeGameState());
    const botTurn = actionsFor(makeGameState({ activePlayerIndex: 1 }));
    const terminal = actionsFor(makeGameState({ phase: 'GameOver' }));

    expect(normal.length).toBeGreaterThan(0);
    expect(botTurn).toEqual([]);
    expect(terminal).toEqual([]);
  });

  it('derives the action menu from the supplied canonical resource state', () => {
    const beforeTrade = makeGameState({
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B),
      ],
    });
    const afterTrade = makeGameState({
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Suns: 1 }) }),
        makePlayer(PLAYER_B),
      ],
    });

    expect(
      actionsFor(beforeTrade).some(
        (action) => action.type === 'trade' && action.give === 'Moons'
      )
    ).toBe(true);
    expect(
      actionsFor(afterTrade).some(
        (action) => action.type === 'trade' && action.give === 'Moons'
      )
    ).toBe(false);
  });

  it('blocks ordinary actions during commit settle but allows income choices when enabled', () => {
    expect(
      actionsFor(makeGameState(), {
        actionCommitPending: true,
      })
    ).toEqual([]);

    const incomeChoiceState = makeGameState({
      phase: 'CollectIncome',
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suits: ['Moons', 'Suns'],
        },
      ],
    });
    expect(
      actionsFor(incomeChoiceState, {
        actionCommitPending: true,
        allowHumanActionsWhileCommitPending: true,
      }).map((action) => action.type)
    ).toEqual(['choose-income-suit', 'choose-income-suit']);
  });

  it('exposes human income choices even when the bot owns the main turn', () => {
    const state = makeGameState({
      phase: 'CollectIncome',
      activePlayerIndex: 1,
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suits: ['Moons', 'Suns'],
        },
        {
          playerId: PLAYER_B,
          districtId: 'D2',
          cardId: '8',
          suits: ['Waves', 'Leaves'],
        },
      ],
    });

    expect(actionsFor(state)).toEqual([
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Moons',
      },
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Suns',
      },
    ]);
  });

  it('schedules bot work only for a ready, unlocked bot turn', () => {
    const schedulingAllowed = (
      overrides: Partial<Parameters<typeof shouldScheduleBotAction>[0]> = {}
    ) =>
      shouldScheduleBotAction({
        terminal: false,
        activePlayerId: 'PlayerB',
        botPlayerId: 'PlayerB',
        isIncomeChoicePhase: false,
        actionCommitPending: false,
        allowIncomeChoiceWhileCommitPending: false,
        botIncomeActionCount: 0,
        startupPreloadReady: true,
        ...overrides,
      });

    expect(schedulingAllowed()).toBe(true);
    expect(schedulingAllowed({ terminal: true })).toBe(false);
    expect(schedulingAllowed({ activePlayerId: 'PlayerA' })).toBe(false);
    expect(schedulingAllowed({ actionCommitPending: true })).toBe(false);
    expect(schedulingAllowed({ startupPreloadReady: false })).toBe(false);
    expect(
      schedulingAllowed({
        activePlayerId: 'PlayerA',
        botIncomeActionCount: 2,
        isIncomeChoicePhase: true,
      })
    ).toBe(true);
    expect(
      schedulingAllowed({
        actionCommitPending: true,
        allowIncomeChoiceWhileCommitPending: true,
        botIncomeActionCount: 2,
        isIncomeChoicePhase: true,
      })
    ).toBe(true);
    expect(
      schedulingAllowed({
        isIncomeChoicePhase: true,
        botIncomeActionCount: 0,
      })
    ).toBe(false);
  });

  it('formats Error instances and unknown thrown values', () => {
    expect(errorMessage(new Error('failed'))).toBe('failed');
    expect(errorMessage('failed')).toBe('failed');
  });
});

function actionsFor(
  state: ReturnType<typeof makeGameState>,
  overrides: {
    actionCommitPending?: boolean;
    allowHumanActionsWhileCommitPending?: boolean;
  } = {}
) {
  return humanActionsAcceptingInputForState({
    state,
    humanPlayerId: PLAYER_A,
    actionCommitPending: overrides.actionCommitPending ?? false,
    allowHumanActionsWhileCommitPending:
      overrides.allowHumanActionsWhileCommitPending ?? false,
  });
}
