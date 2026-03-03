import { describe, expect, it } from 'vitest';

import {
  createAnimationTimerRegistry,
  type AnimationTimerClock,
} from './animationTimerRegistry';

describe('createAnimationTimerRegistry', () => {
  it('clamps negative delays and runs scheduled callbacks', () => {
    const clock = new FakeClock();
    const registry = createAnimationTimerRegistry(clock);
    const calls: string[] = [];

    registry.schedule(-25, () => calls.push('first'));
    registry.schedule(80, () => calls.push('second'));

    expect(clock.delays).toEqual([0, 80]);
    clock.run(1);
    clock.run(2);
    expect(calls).toEqual(['first', 'second']);
  });

  it('cancels outstanding callbacks without clearing completed timers twice', () => {
    const clock = new FakeClock();
    const registry = createAnimationTimerRegistry(clock);

    registry.schedule(10, () => {});
    registry.schedule(20, () => {});
    clock.run(1);
    registry.clearAll();
    registry.clearAll();

    expect(clock.cleared).toEqual([2]);
  });
});

class FakeClock implements AnimationTimerClock {
  readonly delays: number[] = [];
  readonly cleared: number[] = [];
  private readonly callbacks = new Map<number, () => void>();
  private nextId = 0;

  setTimeout(run: () => void, delayMs: number): number {
    this.nextId += 1;
    this.delays.push(delayMs);
    this.callbacks.set(this.nextId, run);
    return this.nextId;
  }

  clearTimeout(timerId: number): void {
    this.cleared.push(timerId);
    this.callbacks.delete(timerId);
  }

  run(timerId: number): void {
    const callback = this.callbacks.get(timerId);
    this.callbacks.delete(timerId);
    callback?.();
  }
}
