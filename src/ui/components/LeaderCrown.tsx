/* Small crown marking the player currently ahead on the full score tiebreak
   chain (districts -> properties -> resources). Hidden on a complete draw. */
export function LeaderCrown({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className={className ? `leader-crown ${className}` : 'leader-crown'}
      fill="currentColor"
    >
      <path d="M2 8l5 3 5-7 5 7 5-3-2 10H4L2 8z" />
    </svg>
  );
}
