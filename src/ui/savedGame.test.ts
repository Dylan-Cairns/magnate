import { afterEach, describe, expect, it, vi } from 'vitest';
import { isTerminal } from '../engine/scoring';
import {
  advanceSave,
  initialSave,
  nextTestAction,
} from './__tests__/savedGameFixtures';
import { transitionOpensHumanDecisionWindow } from './gameControllerModel';
import {
  loadSavedGame,
  parseSavedGame,
  SAVED_GAME_KEY,
  writeSavedGame,
} from './savedGame';

afterEach(() => vi.unstubAllGlobals());

describe('saved game checkpoints', () => {
  it('round trips every human window and the final board through a full game', () => {
    let save = initialSave();
    expect(parseSavedGame(JSON.stringify(save), 'PlayerA')).toEqual(save);
    let windows = 0;
    let incomeWindows = 0;
    for (let i = 0; i < 1000 && !isTerminal(save.state); i++) {
      const next = advanceSave(save, nextTestAction(save));
      if (
        transitionOpensHumanDecisionWindow(save.state, next.state, 'PlayerA') ||
        isTerminal(next.state)
      ) {
        const restored = parseSavedGame(JSON.stringify(next), 'PlayerA');
        expect(restored).toEqual(next);
        windows++;
        if (next.state.phase === 'CollectIncome') incomeWindows++;
        if (!isTerminal(next.state)) {
          expect(advanceSave(restored, nextTestAction(restored))).toEqual(
            advanceSave(next, nextTestAction(next))
          );
        }
      }
      save = next;
    }
    expect(isTerminal(save.state)).toBe(true);
    expect(windows).toBeGreaterThan(10);
    expect(incomeWindows).toBeGreaterThan(0);
  });

  it('rejects mid-window saves, changed state, incompatible versions and illegal history', () => {
    const save = initialSave();
    const midTurn = advanceSave(save, nextTestAction(save));
    for (const invalid of [
      midTurn,
      { ...save, version: 2 },
      { ...save, state: { ...save.state, rngCursor: -1 } },
      { ...save, botProfileId: 'removed-bot' },
      {
        ...save,
        actionHistory: [
          {
            turn: 1,
            phase: 'ActionWindow',
            actingPlayerId: 'PlayerA',
            action: { type: 'end-turn' },
          },
        ],
      },
    ]) {
      expect(() =>
        parseSavedGame(JSON.stringify(invalid), 'PlayerA')
      ).toThrow();
    }
    expect(() => parseSavedGame(JSON.stringify(save), 'PlayerB')).toThrow();
  });

  it('leaves unreadable data intact and reports storage failures', () => {
    const setItem = vi.fn(() => {
      throw new Error('quota');
    });
    const getItem = vi.fn(() => '{broken');
    vi.stubGlobal('window', { localStorage: { getItem, setItem } });
    expect(loadSavedGame('PlayerA')).toEqual({
      save: null,
      error: expect.stringContaining('Autosave is paused'),
    });
    expect(setItem).not.toHaveBeenCalled();
    expect(writeSavedGame(initialSave())).toContain('could not be saved');
    expect(getItem).toHaveBeenCalledWith(SAVED_GAME_KEY);
  });
});
