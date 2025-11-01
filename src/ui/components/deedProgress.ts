export const DEED_PROGRESS_RING_RADIUS = 16;
const DEED_PROGRESS_RING_CENTER = 18;
const DEED_PROGRESS_START_ANGLE_DEGREES = -90;
const DEED_PROGRESS_EPSILON = 1e-6;

export const DEED_PROGRESS_ANIMATION_DURATION_MS = 420;

function pointOnProgressRing(angleDegrees: number): { x: number; y: number } {
  const angleRadians = (angleDegrees * Math.PI) / 180;
  return {
    x: DEED_PROGRESS_RING_CENTER + DEED_PROGRESS_RING_RADIUS * Math.cos(angleRadians),
    y: DEED_PROGRESS_RING_CENTER + DEED_PROGRESS_RING_RADIUS * Math.sin(angleRadians),
  };
}

function easeOutCubic(value: number): number {
  const clamped = Math.max(0, Math.min(1, value));
  return 1 - (1 - clamped) ** 3;
}

export function canonicalDeedProgressRatio(progressValue: number, progressTarget: number): number {
  if (progressTarget <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, progressValue / progressTarget));
}

export function buildDeedProgressArcPath(progressRatio: number): string | null {
  if (progressRatio <= 0 || progressRatio >= 1) {
    return null;
  }

  const start = pointOnProgressRing(DEED_PROGRESS_START_ANGLE_DEGREES);
  const end = pointOnProgressRing(DEED_PROGRESS_START_ANGLE_DEGREES + progressRatio * 360);
  const largeArcFlag = progressRatio > 0.5 ? 1 : 0;
  return [
    `M ${start.x} ${start.y}`,
    `A ${DEED_PROGRESS_RING_RADIUS} ${DEED_PROGRESS_RING_RADIUS} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`,
  ].join(' ');
}

export function shouldAnimateDeedProgress(
  currentRatio: number,
  targetRatio: number
): boolean {
  if (Math.abs(targetRatio - currentRatio) <= DEED_PROGRESS_EPSILON) {
    return false;
  }
  return targetRatio > currentRatio + DEED_PROGRESS_EPSILON;
}

export function tweenAnimatedDeedProgressRatio(
  fromRatio: number,
  targetRatio: number,
  elapsedMs: number,
  durationMs: number = DEED_PROGRESS_ANIMATION_DURATION_MS
): number {
  if (durationMs <= 0) {
    return targetRatio;
  }
  const progress = Math.max(0, Math.min(1, elapsedMs / durationMs));
  return fromRatio + (targetRatio - fromRatio) * easeOutCubic(progress);
}
