import { useEffect, useRef, useState } from 'react';
import { DEED_PROGRESS_REVEAL_MS } from '../animations/timing';
import {
  buildDeedProgressArcPath,
  canonicalDeedProgressRatio,
  clampAnimatedDeedProgressRatio,
  DEED_PROGRESS_RING_RADIUS,
  shouldAnimateDeedProgress,
  tweenAnimatedDeedProgressRatio,
} from './deedProgress';
import { Tooltip } from './Tooltip';

export function ProgressTracker({
  deedProgress,
  deedTarget,
  animateDeedProgress = true,
  cardId,
  label = 'Development progress',
  showTooltip = true,
}: {
  deedProgress: number;
  deedTarget: number;
  animateDeedProgress?: boolean;
  cardId: string;
  label?: string;
  showTooltip?: boolean;
}) {
  const progressValue = deedProgress ?? 0;
  const progressTarget = deedTarget ?? 0;
  const deedProgressRatio = canonicalDeedProgressRatio(
    progressValue,
    progressTarget
  );
  const [animatedDeedProgressRatio, setAnimatedDeedProgressRatio] =
    useState<number>(deedProgressRatio);
  const animatedRatioRef = useRef(animatedDeedProgressRatio);
  const animationFrameRef = useRef<number | null>(null);

  const [prevDeedProgressRatio, setPrevDeedProgressRatio] =
    useState(deedProgressRatio);
  const [prevAnimateDeedProgress, setPrevAnimateDeedProgress] =
    useState(animateDeedProgress);

  if (
    deedProgressRatio !== prevDeedProgressRatio ||
    animateDeedProgress !== prevAnimateDeedProgress
  ) {
    setPrevDeedProgressRatio(deedProgressRatio);
    setPrevAnimateDeedProgress(animateDeedProgress);
    if (
      !animateDeedProgress ||
      !shouldAnimateDeedProgress(animatedDeedProgressRatio, deedProgressRatio)
    ) {
      setAnimatedDeedProgressRatio(deedProgressRatio);
    }
  }

  useEffect(() => {
    animatedRatioRef.current = animatedDeedProgressRatio;
  }, [animatedDeedProgressRatio]);

  useEffect(() => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (
      !animateDeedProgress ||
      !shouldAnimateDeedProgress(animatedRatioRef.current, deedProgressRatio)
    ) {
      return;
    }

    let startTime: number | null = null;
    const fromRatio = animatedRatioRef.current;

    const tick = (timestamp: number) => {
      if (startTime === null) {
        startTime = timestamp;
      }
      const elapsed = timestamp - startTime;
      const nextRatio = tweenAnimatedDeedProgressRatio(
        fromRatio,
        deedProgressRatio,
        elapsed,
        DEED_PROGRESS_REVEAL_MS
      );
      animatedRatioRef.current = nextRatio;
      setAnimatedDeedProgressRatio(nextRatio);

      if (elapsed < DEED_PROGRESS_REVEAL_MS) {
        animationFrameRef.current = window.requestAnimationFrame(tick);
        return;
      }

      animatedRatioRef.current = deedProgressRatio;
      setAnimatedDeedProgressRatio(deedProgressRatio);
      animationFrameRef.current = null;
    };

    animationFrameRef.current = window.requestAnimationFrame(tick);
    return () => {
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [animateDeedProgress, cardId, deedProgressRatio]);

  const displayedDeedProgressRatio = animateDeedProgress
    ? clampAnimatedDeedProgressRatio(
        animatedDeedProgressRatio,
        deedProgressRatio
      )
    : deedProgressRatio;
  const deedProgressArcPath = buildDeedProgressArcPath(
    displayedDeedProgressRatio
  );

  return (
    <div
      className="deed-progress tooltip-trigger"
      aria-label={label.toLowerCase()}
    >
      <svg
        className="deed-progress-ring"
        viewBox="0 0 36 36"
        aria-hidden="true"
      >
        <circle
          className="deed-progress-ring-track"
          cx="18"
          cy="18"
          r={DEED_PROGRESS_RING_RADIUS}
        />
        {displayedDeedProgressRatio >= 1 ? (
          <circle
            className="deed-progress-ring-value"
            cx="18"
            cy="18"
            r={DEED_PROGRESS_RING_RADIUS}
          />
        ) : deedProgressArcPath ? (
          <path className="deed-progress-ring-value" d={deedProgressArcPath} />
        ) : null}
      </svg>
      <span className="deed-progress-value">
        {deedProgress}/{deedTarget}
      </span>
      {showTooltip && <Tooltip>{label}</Tooltip>}
    </div>
  );
}
