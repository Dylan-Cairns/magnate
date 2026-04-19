import type { StartupPreloadProgress } from '../startupPreload';

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
            className="reset-button startup-preload-retry"
            onClick={onRetry}
          >
            Retry
          </button>
        ) : null}
      </section>
    </div>
  );
}

export function ResolutionWarningOverlay({
  open,
  onDismiss,
}: {
  open: boolean;
  onDismiss: () => void;
}) {
  return open ? (
    <div className="resolution-warning-overlay" role="presentation">
      <section
        className="app-bootstrap-card resolution-warning-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="resolution-warning-title"
      >
        <h2 id="resolution-warning-title" className="app-bootstrap-title">
          Display Warning
        </h2>
        <p className="app-bootstrap-copy resolution-warning-message">
          The UI was designed with a minimum size of 1920 x 1080p and will
          likely be unusable at smaller resolutions :(
        </p>
        <div className="resolution-warning-actions">
          <button type="button" className="reset-button" onClick={onDismiss}>
            OK
          </button>
        </div>
      </section>
    </div>
  ) : null;
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.max(min, Math.min(value, max));
}
