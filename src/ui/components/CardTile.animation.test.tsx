import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const hookHarness = vi.hoisted(() => {
  let cursor = 0;
  let slots: unknown[] = [];

  return {
    beginRender() {
      cursor = 0;
    },
    resetMount() {
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

      const setValue = (next: T | ((previous: T) => T)) => {
        const previous = slots[index] as T;
        slots[index] =
          typeof next === 'function'
            ? (next as (value: T) => T)(previous)
            : next;
      };

      return [slots[index] as T, setValue] as const;
    },
    useEffect(run: () => void | (() => void)) {
      run();
    },
  };
});

vi.mock('react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react')>();
  return {
    ...actual,
    useEffect: hookHarness.useEffect,
    useRef: hookHarness.useRef,
    useState: hookHarness.useState,
  };
});

import { DEED_PROGRESS_REVEAL_MS } from '../animations/timing';
import { CardTile } from './CardTile';

type AnimationFrameCallback = (timestamp: number) => void;

let nextFrameId = 0;
let animationFrames = new Map<number, AnimationFrameCallback>();

beforeEach(() => {
  hookHarness.resetMount();
  nextFrameId = 0;
  animationFrames = new Map();

  vi.stubGlobal('window', {
    requestAnimationFrame(callback: AnimationFrameCallback) {
      const id = ++nextFrameId;
      animationFrames.set(id, callback);
      return id;
    },
    cancelAnimationFrame(id: number) {
      animationFrames.delete(id);
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('CardTile deed progress animation', () => {
  it('does not carry progress into a remounted card with the same id', () => {
    renderCard(0);
    renderCard(2);

    flushAnimationFrames(0);
    flushAnimationFrames(DEED_PROGRESS_REVEAL_MS);

    const progressedHtml = renderCard(2);
    expect(progressedHtml).toContain('class="deed-progress-ring-value"');

    // Simulates unmount/remount without reloading the CardTile module.
    hookHarness.resetMount();

    const remountedHtml = renderCard(0);
    expect(remountedHtml).toContain('>0/9<');
    expect(remountedHtml).not.toContain('class="deed-progress-ring-value"');
  });
});

function renderCard(progress: number): string {
  hookHarness.beginRender();
  return renderToStaticMarkup(
    <CardTile
      cardId="29"
      deedProgress={progress}
      deedTarget={9}
      inDevelopment
      animateDeedProgress
    />
  );
}

function flushAnimationFrames(timestamp: number): void {
  const callbacks = [...animationFrames.values()];
  animationFrames.clear();

  for (const callback of callbacks) {
    callback(timestamp);
  }
}
