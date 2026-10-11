import { useLayoutEffect, useRef, useState } from 'react';

// Presentation may already be settled when a new roll reaches the component.
// Keep the halo off until the die's own rotation and bounce have both finished.
export function useDieMotion(
  rollTrigger: number | string | undefined,
  bounceNonce: number,
  animationsEnabled: boolean
) {
  const motionRef = useRef<HTMLDivElement>(null);
  const [completed, setCompleted] = useState<{
    rollTrigger: number | string;
    bounceNonce: number;
  } | null>(null);

  if (!animationsEnabled && completed !== null) {
    setCompleted(null);
  }

  useLayoutEffect(() => {
    if (!animationsEnabled) {
      return;
    }
    if (rollTrigger === undefined || bounceNonce === 0 || !motionRef.current) {
      return;
    }

    let cancelled = false;
    const motions = motionRef.current.getAnimations({ subtree: true });
    void Promise.allSettled(motions.map((motion) => motion.finished)).then(
      () => {
        if (!cancelled) {
          setCompleted({ rollTrigger, bounceNonce });
        }
      }
    );
    return () => {
      cancelled = true;
    };
  }, [animationsEnabled, bounceNonce, rollTrigger]);

  const moving =
    animationsEnabled &&
    rollTrigger !== undefined &&
    (completed?.rollTrigger !== rollTrigger ||
      completed?.bounceNonce !== bounceNonce);

  return { motionRef, moving };
}
