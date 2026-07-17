import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const reactHookHarness = vi.hoisted(() => {
  let cursor = 0;
  let slots: unknown[] = [];

  return {
    beginRender() {
      cursor = 0;
    },
    reset() {
      cursor = 0;
      slots = [];
    },
    useRef<T>(initialValue: T) {
      const index = cursor;
      cursor += 1;
      if (!(index in slots)) {
        slots[index] = { current: initialValue };
      }
      return slots[index] as { current: T };
    },
    useState<T>(initialValue: T | (() => T)) {
      const index = cursor;
      cursor += 1;
      if (!(index in slots)) {
        slots[index] =
          typeof initialValue === 'function'
            ? (initialValue as () => T)()
            : initialValue;
      }
      const setValue = (nextValue: T | ((previous: T) => T)) => {
        const previous = slots[index] as T;
        slots[index] =
          typeof nextValue === 'function'
            ? (nextValue as (current: T) => T)(previous)
            : nextValue;
      };
      return [slots[index] as T, setValue] as const;
    },
  };
});

vi.mock('react', () => ({
  useCallback: <T>(callback: T) => callback,
  useEffect: (run: () => void | (() => void)) => {
    run();
  },
  useRef: reactHookHarness.useRef,
  useState: reactHookHarness.useState,
}));

import {
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../../engine/__tests__/fixtures';
import { buildAnimationSequence } from '../runtime/animationSequence';
import { buildGameTransaction } from '../runtime/transactions';
import {
  sequencePresentationSnapshotUpdateTimes,
  useGameAnimations,
} from './useGameAnimations';

beforeEach(() => {
  reactHookHarness.reset();
  vi.useFakeTimers();
  vi.stubGlobal('window', {
    setTimeout: (run: () => void, delayMs?: number) =>
      globalThis.setTimeout(run, delayMs),
    clearTimeout: (timerId: number) => globalThis.clearTimeout(timerId),
    localStorage: {
      getItem: () => null,
      setItem: () => undefined,
    },
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('useGameAnimations scheduling helpers', () => {
  it('does not schedule a presentation snapshot at the final commit/unlock boundary', () => {
    const transaction = makeBuyDeedTransaction();
    const sequence = buildAnimationSequence(transaction);
    const finalBoundaryMs = sequence.commitMs;

    const updateTimes = sequencePresentationSnapshotUpdateTimes(
      sequence,
      sequence.durationMs,
      finalBoundaryMs
    );

    expect(sequence.commitMs).toBe(sequence.inputUnlockMs);
    expect(updateTimes).not.toContain(finalBoundaryMs);
    expect(updateTimes.every((atMs) => atMs < finalBoundaryMs)).toBe(true);
  });

  it('unlocks input at the sequence boundary while presentation remains active through settle', () => {
    const transaction = makeBuyDeedTransaction();
    const sequence = buildAnimationSequence(transaction);
    const onInputUnlock = vi.fn();
    const onSettle = vi.fn();
    let animations = AnimationHarness();

    animations.enqueueTransition({
      transactionId: transaction.id,
      previousState: transaction.previousState,
      nextState: transaction.nextState,
      action: transaction.action,
      actingPlayerId: transaction.actingPlayerId,
      onInputUnlock,
      onSettle,
    });
    animations = AnimationHarness();

    expect(animations.presentationPending).toBe(true);
    expect(animations.presentationSnapshot?.viewState).not.toBe(
      transaction.nextState
    );

    vi.advanceTimersByTime(sequence.inputUnlockMs - 1);
    animations = AnimationHarness();
    expect(animations.presentationPending).toBe(true);
    expect(animations.presentationSnapshot).not.toBeNull();
    expect(onInputUnlock).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    animations = AnimationHarness();
    expect(onInputUnlock).toHaveBeenCalledOnce();
    expect(animations.presentationPending).toBe(true);
    expect(animations.presentedState).toBe(transaction.nextState);

    vi.advanceTimersByTime(sequence.durationMs - sequence.inputUnlockMs);
    animations = AnimationHarness();
    expect(animations.presentationPending).toBe(false);
    expect(animations.presentationSnapshot).toBeNull();
    expect(animations.presentedState).toBeNull();
    expect(onSettle).toHaveBeenCalledOnce();
  });

  it('presents queued transitions in order without replacing the active sequence', () => {
    const first = makeBuyDeedTransaction();
    const second = makeTradeTransaction(first.nextState);
    const firstSequence = buildAnimationSequence(first);
    const secondSequence = buildAnimationSequence(second);
    const lifecycle: string[] = [];
    let animations = AnimationHarness();

    animations.enqueueTransition({
      transactionId: first.id,
      previousState: first.previousState,
      nextState: first.nextState,
      action: first.action,
      actingPlayerId: first.actingPlayerId,
      onInputUnlock: () => lifecycle.push('first-unlock'),
      onSettle: () => lifecycle.push('first-settle'),
    });
    animations.enqueueTransition({
      transactionId: second.id,
      previousState: second.previousState,
      nextState: second.nextState,
      action: second.action,
      actingPlayerId: second.actingPlayerId,
      onInputUnlock: () => lifecycle.push('second-unlock'),
      onSettle: () => lifecycle.push('second-settle'),
    });

    vi.advanceTimersByTime(firstSequence.inputUnlockMs);
    animations = AnimationHarness();
    expect(lifecycle).toEqual(['first-unlock']);
    expect(animations.presentationPending).toBe(true);
    expect(animations.presentedState).toBe(first.nextState);

    vi.advanceTimersByTime(
      firstSequence.durationMs - firstSequence.inputUnlockMs
    );
    animations = AnimationHarness();
    expect(lifecycle).toEqual(['first-unlock', 'first-settle']);
    expect(animations.presentationPending).toBe(true);
    expect(animations.presentationSnapshot?.viewState).not.toBe(
      second.nextState
    );

    vi.advanceTimersByTime(secondSequence.durationMs);
    vi.runAllTimers();
    animations = AnimationHarness();
    expect(lifecycle).toEqual([
      'first-unlock',
      'first-settle',
      'second-unlock',
      'second-settle',
    ]);
    expect(animations.presentationPending).toBe(false);
  });

  it('cancels the active transition and backlog without firing lifecycle callbacks', () => {
    const transaction = makeBuyDeedTransaction();
    const queuedTransaction = makeTradeTransaction(transaction.nextState);
    const sequence = buildAnimationSequence(transaction);
    const onInputUnlock = vi.fn();
    const onSettle = vi.fn();
    let animations = AnimationHarness();

    animations.enqueueTransition({
      transactionId: transaction.id,
      previousState: transaction.previousState,
      nextState: transaction.nextState,
      action: transaction.action,
      actingPlayerId: transaction.actingPlayerId,
      onInputUnlock,
      onSettle,
    });
    animations.enqueueTransition({
      transactionId: queuedTransaction.id,
      previousState: queuedTransaction.previousState,
      nextState: queuedTransaction.nextState,
      action: queuedTransaction.action,
      actingPlayerId: queuedTransaction.actingPlayerId,
      onInputUnlock,
      onSettle,
    });
    animations = AnimationHarness();
    expect(animations.presentationPending).toBe(true);

    animations.clearPresentationQueue();
    vi.advanceTimersByTime(sequence.durationMs);
    vi.runAllTimers();
    animations = AnimationHarness();

    expect(onInputUnlock).not.toHaveBeenCalled();
    expect(onSettle).not.toHaveBeenCalled();
    expect(animations.presentationPending).toBe(false);
    expect(animations.presentationSnapshot).toBeNull();
    expect(animations.cardFlights).toEqual([]);
    expect(animations.resourceFlights).toEqual([]);
  });

  it('tracks the animation preference consumed by controller dispatch', () => {
    let animations = AnimationHarness();

    expect(animations.enabled).toBe(true);
    animations.setEnabled(false);
    animations = AnimationHarness();

    expect(animations.enabled).toBe(false);
  });

  it('finishes the active transition and backlog when animations are disabled', () => {
    const first = makeBuyDeedTransaction();
    const second = makeTradeTransaction(first.nextState);
    const lifecycle: string[] = [];
    let animations = AnimationHarness();

    for (const [label, transaction] of [
      ['first', first],
      ['second', second],
    ] as const) {
      animations.enqueueTransition({
        transactionId: transaction.id,
        previousState: transaction.previousState,
        nextState: transaction.nextState,
        action: transaction.action,
        actingPlayerId: transaction.actingPlayerId,
        onInputUnlock: () => lifecycle.push(`${label}-unlock`),
        onSettle: () => lifecycle.push(`${label}-settle`),
      });
    }

    vi.advanceTimersByTime(buildAnimationSequence(first).inputUnlockMs);
    animations.setEnabled(false);
    animations = AnimationHarness();

    expect(lifecycle).toEqual([
      'first-unlock',
      'first-settle',
      'second-unlock',
      'second-settle',
    ]);
    expect(animations.enabled).toBe(false);
    expect(animations.presentationPending).toBe(false);
    expect(animations.presentationSnapshot).toBeNull();
  });
});

function AnimationHarness() {
  reactHookHarness.beginRender();
  return useGameAnimations();
}

function makeBuyDeedTransaction() {
  return buildGameTransaction({
    previousState: makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 2, Knots: 2 }),
        }),
        makePlayer(PLAYER_B),
      ],
    }),
    action: { type: 'buy-deed', cardId: '6', districtId: 'D1' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-buy-deed',
    stepToDecision: () =>
      makeGameState({
        players: [
          makePlayer(PLAYER_A, {
            hand: [],
            resources: makeResources({ Moons: 1, Knots: 1 }),
          }),
          makePlayer(PLAYER_B),
        ],
        districts: [
          makeDistrict('D1', ['Moons'], {
            [PLAYER_A]: {
              developed: [],
              deed: { cardId: '6', progress: 0, tokens: {} },
            },
          }),
        ],
      }),
  });
}

function makeTradeTransaction(previousState: ReturnType<typeof makeGameState>) {
  return buildGameTransaction({
    previousState,
    action: { type: 'trade', give: 'Moons', receive: 'Suns' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-trade-after-buy',
    stepToDecision: () =>
      makeGameState({
        players: [
          makePlayer(PLAYER_A, {
            hand: [],
            resources: makeResources({ Knots: 1, Suns: 1 }),
          }),
          makePlayer(PLAYER_B),
        ],
        districts: [...previousState.districts],
      }),
  });
}
