export const DEED_PROGRESS_RING_RADIUS = 16;

export function canonicalDeedProgressRatio(
  progressValue: number,
  progressTarget: number
): number {
  if (progressTarget <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, progressValue / progressTarget));
}
