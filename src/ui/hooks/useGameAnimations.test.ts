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
  useEffect: () => undefined,
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
    const finalBoundaryMs = Math.min(sequence.commitMs, sequence.inputUnlockMs);

    const updateTimes = sequencePresentationSnapshotUpdateTimes(
      sequence,
      sequence.durationMs,
      finalBoundaryMs
    );

    expect(sequence.commitMs).toBe(sequence.inputUnlockMs);
    expect(updateTimes).not.toContain(finalBoundaryMs);
    expect(updateTimes.every((atMs) => atMs < finalBoundaryMs)).toBe(true);
  });

  it('holds canonical commit and input lock until the sequence commit boundary', () => {
    const transaction = makeBuyDeedTransaction();
    const sequence = buildAnimationSequence(transaction);
    const onCommitTransition = vi.fn();
    const onSettle = vi.fn();
    let animations = AnimationHarness(onCommitTransition);

    animations.runTransition({
      previousState: transaction.previousState,
      nextState: transaction.nextState,
      action: transaction.action,
      actingPlayerId: transaction.actingPlayerId,
      onSettle,
    });
    animations = AnimationHarness(onCommitTransition);

    expect(animations.actionCommitPending).toBe(true);
    expect(animations.presentationSnapshot?.viewState).not.toBe(
      transaction.nextState
    );
    expect(onCommitTransition).not.toHaveBeenCalled();

    vi.advanceTimersByTime(sequence.commitMs - 1);
    expect(onCommitTransition).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    animations = AnimationHarness(onCommitTransition);
    expect(onCommitTransition).toHaveBeenCalledOnce();
    expect(onCommitTransition).toHaveBeenCalledWith(
      transaction.previousState,
      transaction.nextState,
      transaction.action
    );
    expect(animations.actionCommitPending).toBe(false);
    expect(animations.presentationSnapshot).toBeNull();

    vi.advanceTimersByTime(sequence.durationMs - sequence.commitMs);
    expect(onCommitTransition).toHaveBeenCalledOnce();
    expect(onSettle).toHaveBeenCalledOnce();
  });

  it('cancels a pending transition without committing when reset cleanup runs', () => {
    const transaction = makeBuyDeedTransaction();
    const sequence = buildAnimationSequence(transaction);
    const onCommitTransition = vi.fn();
    let animations = AnimationHarness(onCommitTransition);

    animations.runTransition({
      previousState: transaction.previousState,
      nextState: transaction.nextState,
      action: transaction.action,
      actingPlayerId: transaction.actingPlayerId,
    });
    animations = AnimationHarness(onCommitTransition);
    expect(animations.actionCommitPending).toBe(true);

    animations.clearPendingActionCommit();
    animations.clearAllFlights();
    vi.advanceTimersByTime(sequence.durationMs);
    animations = AnimationHarness(onCommitTransition);

    expect(onCommitTransition).not.toHaveBeenCalled();
    expect(animations.actionCommitPending).toBe(false);
    expect(animations.presentationSnapshot).toBeNull();
    expect(animations.cardFlights).toEqual([]);
    expect(animations.resourceFlights).toEqual([]);
  });

  it('tracks the animation preference consumed by controller dispatch', () => {
    const onCommitTransition = vi.fn();
    let animations = AnimationHarness(onCommitTransition);

    expect(animations.enabled).toBe(true);
    animations.setEnabled(false);
    animations = AnimationHarness(onCommitTransition);

    expect(animations.enabled).toBe(false);
  });
});

function AnimationHarness(onCommitTransition: ReturnType<typeof vi.fn>) {
  reactHookHarness.beginRender();
  return useGameAnimations({ onCommitTransition });
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
