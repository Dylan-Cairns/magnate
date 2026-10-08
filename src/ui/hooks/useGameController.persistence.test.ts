import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Minimal hook runner, including dependency-aware effects and unmount cleanup.
const hooks = vi.hoisted(() => {
  let slots: unknown[] = [];
  let cursor = 0;
  let dirty = false;
  const effects = new Map<number, () => void>();
  const cleanups = new Map<number, () => void>();
  function memo<T>(create: () => T, deps: unknown[]) {
    const index = cursor++;
    const old = slots[index] as { deps: unknown[]; value: T } | undefined;
    if (!old || deps.some((dep, i) => !Object.is(dep, old.deps[i]))) {
      slots[index] = { deps, value: create() };
    }
    return (slots[index] as { value: T }).value;
  }
  return {
    memo,
    state<T>(initial: T | (() => T)) {
      const index = cursor++;
      if (!(index in slots))
        slots[index] =
          typeof initial === 'function' ? (initial as () => T)() : initial;
      return [
        slots[index] as T,
        (next: T | ((old: T) => T)) => {
          const value =
            typeof next === 'function'
              ? (next as (old: T) => T)(slots[index] as T)
              : next;
          dirty ||= !Object.is(value, slots[index]);
          slots[index] = value;
        },
      ] as const;
    },
    ref<T>(value: T) {
      return memo(() => ({ current: value }), []);
    },
    effect(run: () => void | (() => void), deps: unknown[]) {
      const index = cursor;
      memo(() => {
        effects.set(index, () => {
          cleanups.get(index)?.();
          const cleanup = run();
          if (cleanup) cleanups.set(index, cleanup);
          else cleanups.delete(index);
        });
      }, deps);
    },
    render<T>(run: () => T): T {
      let result: T;
      let count = 0;
      do {
        if (++count > 30) throw new Error('Hook render loop');
        cursor = 0;
        dirty = false;
        result = run();
        if (!dirty) {
          const pending = [...effects.values()];
          effects.clear();
          pending.forEach((effect) => effect());
        }
      } while (dirty);
      return result;
    },
    unmount() {
      cleanups.forEach((cleanup) => cleanup());
      cleanups.clear();
      effects.clear();
      slots = [];
      cursor = 0;
    },
  };
});

vi.mock('react', () => ({
  useState: hooks.state,
  useRef: hooks.ref,
  useMemo: hooks.memo,
  useCallback: <T>(callback: T, deps: unknown[]) =>
    hooks.memo(() => callback, deps),
  useEffect: hooks.effect,
  useEffectEvent: <T>(callback: T) => callback,
}));

const animation = vi.hoisted(() => ({
  enabled: false,
  clearAllFlights: vi.fn(),
  clearPresentationQueue: vi.fn(),
  enqueueTransition: vi.fn(),
  setEnabled: vi.fn((enabled: boolean) => {
    animation.enabled = enabled;
  }),
}));
vi.mock('./useGameAnimations', () => ({ useGameAnimations: () => animation }));

const bot = vi.hoisted(() => ({ selectAction: vi.fn(), close: vi.fn() }));
vi.mock('../../policies/catalog', async (original) => {
  const catalog = await original<typeof import('../../policies/catalog')>();
  return {
    ...catalog,
    resolveBotProfile: (id: string, ruleset?: Ruleset) => ({
      ...catalog.resolveBotProfile(id, ruleset),
      policy: bot,
    }),
  };
});

import { legalActions } from '../../engine/actionBuilders';
import type { Ruleset } from '../../engine/types';
import { DEFAULT_BOT_PROFILE_ID } from '../../policies/catalog';
import {
  advanceSave,
  initialSave,
  nextTestAction,
} from '../__tests__/savedGameFixtures';
import {
  incomeChoiceActionsForPlayer,
  transitionOpensHumanDecisionWindow,
} from '../gameControllerModel';
import { parseSavedGame, SAVED_GAME_KEY, type SavedGame } from '../savedGame';
import { useGameController } from './useGameController';

let storage: Map<string, string>;
function render() {
  return hooks.render(() =>
    useGameController({
      humanPlayerId: 'PlayerA',
      botPlayerId: 'PlayerB',
      startupPreloadReady: true,
    })
  );
}
function stored() {
  return parseSavedGame(storage.get(SAVED_GAME_KEY)!, 'PlayerA');
}

beforeEach(() => {
  storage = new Map();
  vi.useFakeTimers();
  vi.clearAllMocks();
  animation.enabled = false;
  bot.selectAction.mockImplementation(
    ({ legalActions: actions }: { legalActions: unknown[] }) =>
      Promise.resolve(actions[0])
  );
  vi.stubGlobal('window', {
    location: { search: '' },
    localStorage: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
    },
    setTimeout: globalThis.setTimeout,
    clearTimeout: globalThis.clearTimeout,
  });
});
afterEach(() => {
  hooks.unmount();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('controller persistence', () => {
  it('starts a fresh browser with Easy and a restorable save', () => {
    const controller = render();
    expect(controller.botProfileId).toBe('rollout-search-v2-easy');
    expect(stored().botProfileId).toBe('rollout-search-v2-easy');
    expect(stored().state).toEqual(controller.state);
  });

  it.each([undefined, 'broken'])(
    'restores the chosen opponent without a usable game save (%s)',
    (save) => {
      if (save !== undefined) storage.set(SAVED_GAME_KEY, save);
      const controller = render();
      controller.resetSession('preferred-opponent', {
        ruleset: 'standard',
        botProfileId: 'rollout-search-v2-medium',
        animationsEnabled: false,
      });
      render();
      expect(storage.get('magnate:botProfileId')).toBe(
        'rollout-search-v2-medium'
      );
      hooks.unmount();
      if (save === undefined) storage.delete(SAVED_GAME_KEY);
      expect(render().botProfileId).toBe('rollout-search-v2-medium');
    }
  );

  it('uses the default opponent when the saved preference is obsolete', () => {
    storage.set('magnate:botProfileId', 'removed-profile');
    expect(render().botProfileId).toBe(DEFAULT_BOT_PROFILE_ID);
  });

  it('keeps the saved game opponent when restoring an existing session', () => {
    const save = { ...initialSave(), botProfileId: 'rollout-search-v2-medium' };
    storage.set(SAVED_GAME_KEY, JSON.stringify(save));
    storage.set('magnate:botProfileId', 'rollout-search-v2-hard');
    expect(render().botProfileId).toBe(save.botProfileId);
    expect(storage.get('magnate:botProfileId')).toBe(save.botProfileId);
  });

  it('uses the preferred ruleset when creating a session without a saved game', () => {
    storage.set('magnate:ruleset', 'extended');
    const controller = render();
    expect(controller.ruleset).toBe('extended');
    expect(controller.state.ruleset).toBe('extended');
    expect(stored().state).toEqual(controller.state);
  });

  it('commits a replacement ruleset and opponent together and restores them', () => {
    storage.set(SAVED_GAME_KEY, JSON.stringify(initialSave('old', 'extended')));
    let controller = render();
    const oldId = controller.gameId;
    controller.resetSession('replacement-standard', {
      ruleset: 'standard',
      botProfileId: 'td-root-search-v2-medium',
      animationsEnabled: false,
    });
    controller = render();
    expect(controller.error).toBeNull();
    expect(controller.gameId).not.toBe(oldId);
    expect(stored().state.seed).toBe('replacement-standard');
    expect(stored().state.ruleset).toBe('standard');
    expect(stored().botProfileId).toBe('td-root-search-v2-medium');
    expect(storage.get('magnate:ruleset')).toBe('standard');
    expect(storage.get('magnate:botProfileId')).toBe(
      'td-root-search-v2-medium'
    );
    const replacement = stored();
    hooks.unmount();
    controller = render();
    expect(controller.storageError).toBeNull();
    expect(controller.state).toEqual(replacement.state);
    expect(controller.botProfileId).toBe(replacement.botProfileId);
  });

  it('rejects an incompatible setup without changing the active session or save', () => {
    storage.set(
      SAVED_GAME_KEY,
      JSON.stringify(initialSave('original', 'extended'))
    );
    const controller = render();
    const original = stored();
    vi.clearAllMocks();
    controller.resetSession('invalid', {
      ruleset: 'extended',
      botProfileId: 'td-root-search-v2-medium',
      animationsEnabled: true,
    });
    const after = render();
    expect(after.error).toContain('not available for the extended ruleset');
    expect(after.gameId).toBe(controller.gameId);
    expect(after.state).toEqual(controller.state);
    expect(after.botProfileId).toBe(controller.botProfileId);
    expect(stored()).toEqual(original);
    expect(bot.close).not.toHaveBeenCalled();
    expect(animation.setEnabled).not.toHaveBeenCalled();
  });

  it('preserves the window through actions and reset, then restores without presentation', () => {
    let controller = render();
    const checkpoint = stored();
    const action = legalActions(controller.state).find(
      (a) => a.type === 'sell-card'
    )!;
    controller.performHumanAction(action);
    controller = render();
    expect(controller.state).not.toEqual(checkpoint.state);
    expect(stored()).toEqual(checkpoint);
    expect(controller.canResetTurn).toBe(true);
    controller.resetTurn();
    controller = render();
    expect(controller.state).toEqual(checkpoint.state);
    controller.performHumanAction(action);
    controller = render();
    controller.performHumanAction({ type: 'end-turn' });
    render();
    // Reload before the bot finishes, including while a turn animation would run.
    hooks.unmount();
    animation.enabled = true;
    controller = render();
    expect(controller.state).toEqual(checkpoint.state);
    expect(controller.timelineLog).toEqual(checkpoint.timelineLog);
    expect(controller.actionHistory).toEqual(checkpoint.actionHistory);
    expect(controller.animations.quietIncomeRollId).toBe(
      checkpoint.state.lastIncomeRoll?.rollId
    );
    expect(controller.canResetTurn).toBe(false);
    expect(animation.enqueueTransition).not.toHaveBeenCalled();
    expect(bot.selectAction).not.toHaveBeenCalled();
    controller.performHumanAction(action);
    expect(animation.enqueueTransition).toHaveBeenCalledTimes(1);
  });

  it('keeps restored shared income idle until human input and saves the next main window', async () => {
    let save = initialSave();
    let checkpoint: SavedGame | undefined;
    for (let i = 0; i < 1000 && save.state.phase !== 'GameOver'; i++) {
      const next = advanceSave(save, nextTestAction(save));
      const actions = legalActions(next.state);
      if (
        transitionOpensHumanDecisionWindow(save.state, next.state, 'PlayerA') &&
        actions.some(
          (a) => a.type === 'choose-income-suit' && a.playerId === 'PlayerA'
        ) &&
        actions.some(
          (a) => a.type === 'choose-income-suit' && a.playerId === 'PlayerB'
        )
      ) {
        checkpoint = next;
        break;
      }
      save = next;
    }
    expect(checkpoint).toBeDefined();
    storage.set(SAVED_GAME_KEY, JSON.stringify(checkpoint));
    let controller = render();
    await vi.runAllTimersAsync();
    expect(bot.selectAction).not.toHaveBeenCalled();
    expect(controller.botThinking).toBe(false);
    expect(controller.humanActionsAcceptingInput.length).toBeGreaterThan(0);
    while (controller.state.phase === 'CollectIncome') {
      const action = controller.humanActionsAcceptingInput[0];
      if (action) controller.performHumanAction(action);
      controller = render();
      await vi.runAllTimersAsync();
      controller = render();
    }
    expect(bot.selectAction).toHaveBeenCalled();
    // If this was bot-owned income, allow cheap bot actions to reach the human.
    for (
      let i = 0;
      i < 50 && !controller.humanActionsAcceptingInput.length;
      i++
    ) {
      bot.selectAction.mockImplementation(
        ({ state }: { state: SavedGame['state'] }) =>
          Promise.resolve(nextTestAction({ ...checkpoint!, state }))
      );
      await vi.runAllTimersAsync();
      controller = render();
    }
    expect(stored().state).toEqual(controller.state);
    expect(stored().state.phase).toBe('ActionWindow');
  });

  it('does not overwrite an unreadable save until New Game, and saves bot selection', () => {
    storage.set(SAVED_GAME_KEY, 'broken');
    let controller = render();
    controller.performHumanAction(
      legalActions(controller.state).find((a) => a.type === 'sell-card')!
    );
    controller = render();
    expect(controller.storageError).toContain('Autosave is paused');
    expect(storage.get(SAVED_GAME_KEY)).toBe('broken');
    const oldId = controller.gameId;
    controller.resetSession('replacement', {
      ruleset: 'extended',
      botProfileId: 'rollout-search-v2-medium',
      animationsEnabled: false,
    });
    controller = render();
    expect(controller.storageError).toBeNull();
    expect(stored().state.seed).toBe('replacement');
    expect(controller.gameId).not.toBe(oldId);
    expect(stored().state.ruleset).toBe('extended');
    expect(stored().botProfileId).toBe('rollout-search-v2-medium');
  });

  it('replaces the unfinished checkpoint on completion and restores the final board', async () => {
    bot.selectAction.mockImplementation(
      ({ state }: { state: SavedGame['state'] }) =>
        Promise.resolve(nextTestAction({ ...initialSave(), state }))
    );
    let controller = render();
    for (let i = 0; i < 1000 && controller.state.phase !== 'GameOver'; i++) {
      if (controller.humanActionsAcceptingInput.length) {
        controller.performHumanAction(
          nextTestAction({ ...initialSave(), state: controller.state })
        );
      }
      controller = render();
      await vi.runAllTimersAsync();
      controller = render();
    }
    expect(controller.state.phase).toBe('GameOver');
    expect(stored().state).toEqual(controller.state);
    const finalState = controller.state;
    hooks.unmount();
    bot.selectAction.mockClear();
    controller = render();
    await vi.runAllTimersAsync();
    expect(controller.state).toEqual(finalState);
    expect(controller.terminal).toBe(true);
    expect(bot.selectAction).not.toHaveBeenCalled();
  });

  it('bypasses saves for developer fixtures', () => {
    storage.set(SAVED_GAME_KEY, 'untouched');
    window.location.search = '?fixture=multi-income';
    const controller = render();
    expect(controller.storageError).toBeNull();
    controller.resetSession();
    expect(storage.get(SAVED_GAME_KEY)).toBe('untouched');
  });
});

describe('controller resource policy', () => {
  function advanceToBotDecision() {
    let controller = render();
    const botMustDecideNow = () =>
      (controller.state.phase === 'ActionWindow' &&
        controller.state.players[controller.state.activePlayerIndex]?.id ===
          'PlayerB') ||
      (controller.state.phase === 'CollectIncome' &&
        incomeChoiceActionsForPlayer(legalActions(controller.state), 'PlayerB')
          .length > 0);
    for (let i = 0; i < 200 && !botMustDecideNow(); i += 1) {
      const action = controller.humanActionsAcceptingInput[0];
      if (!action) break;
      controller.performHumanAction(action);
      controller = render();
    }
    expect(botMustDecideNow()).toBe(true);
    return controller;
  }

  function stubHiddenPage() {
    vi.stubGlobal('document', {
      visibilityState: 'hidden',
      addEventListener: () => {},
      removeEventListener: () => {},
    });
  }

  it('keeps making bot decisions while the page is hidden', async () => {
    stubHiddenPage();

    const before = advanceToBotDecision();
    await vi.runAllTimersAsync();
    const after = render();

    expect(bot.selectAction).toHaveBeenCalled();
    expect(after.actionHistory.length).toBeGreaterThan(
      before.actionHistory.length
    );
    expect(after.error).toBeNull();
  });

  it('releases the bot policy when restoring a terminal game', () => {
    let save = initialSave('terminal-release');
    for (let i = 0; i < 1000 && save.state.phase !== 'GameOver'; i += 1) {
      save = advanceSave(save, nextTestAction(save));
    }
    expect(save.state.phase).toBe('GameOver');
    storage.set(SAVED_GAME_KEY, JSON.stringify(save));

    render();

    expect(bot.close).toHaveBeenCalled();
  });
});
