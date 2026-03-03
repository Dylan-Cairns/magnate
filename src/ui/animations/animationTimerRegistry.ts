export interface AnimationTimerClock {
  setTimeout(run: () => void, delayMs: number): number;
  clearTimeout(timerId: number): void;
}

export interface AnimationTimerRegistry {
  schedule(delayMs: number, run: () => void): number;
  clearAll(): void;
}

export function createAnimationTimerRegistry(
  clock: AnimationTimerClock = browserAnimationTimerClock
): AnimationTimerRegistry {
  const timerIds = new Set<number>();

  return {
    schedule(delayMs, run) {
      const timerId = clock.setTimeout(
        () => {
          timerIds.delete(timerId);
          run();
        },
        Math.max(0, delayMs)
      );
      timerIds.add(timerId);
      return timerId;
    },
    clearAll() {
      for (const timerId of timerIds) {
        clock.clearTimeout(timerId);
      }
      timerIds.clear();
    },
  };
}

const browserAnimationTimerClock: AnimationTimerClock = {
  setTimeout: (run, delayMs) => window.setTimeout(run, delayMs),
  clearTimeout: (timerId) => window.clearTimeout(timerId),
};
