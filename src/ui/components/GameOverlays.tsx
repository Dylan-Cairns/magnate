import type { StartupPreloadProgress } from '../startupPreload';
import { Tooltip } from './Tooltip';

export function StartupPreloadOverlay({
  ready,
  error,
  progress,
  onRetry,
}: {
  ready: boolean;
  error: string | null;
  progress: StartupPreloadProgress;
  onRetry: () => void;
}) {
  if (ready) {
    return null;
  }

  const percent = clamp(progress.percent, 0, 100);
  const completed = Math.min(progress.completed, progress.total);

  return (
    <div
      className="app-bootstrap-shell startup-preload-overlay"
      role="presentation"
    >
      <section
        className="app-bootstrap-card startup-preload-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="startup-preload-title"
      >
        <h2 id="startup-preload-title" className="app-bootstrap-title">
          {error ? 'Loading Failed' : 'Preparing Game Assets'}
        </h2>
        <p className="app-bootstrap-copy startup-preload-message">
          {error ? `Could not preload assets: ${error}` : progress.message}
        </p>
        <div
          className="app-bootstrap-bar startup-preload-progress"
          role="progressbar"
          aria-label="Startup asset preload progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
        >
          <span
            className="startup-preload-progress-fill"
            style={{ width: `${percent}%` }}
          />
        </div>
        <p className="startup-preload-progress-text">
          {completed} / {progress.total}
        </p>
        {error ? (
          <button
            type="button"
            className="reset-button startup-preload-retry tooltip-trigger"
            onClick={onRetry}
          >
            Retry
            <Tooltip>Retry loading game assets</Tooltip>
          </button>
        ) : null}
      </section>
    </div>
  );
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.max(min, Math.min(value, max));
}
