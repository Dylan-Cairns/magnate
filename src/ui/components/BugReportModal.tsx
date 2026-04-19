export function BugReportModal({
  open,
  issueUrl,
  onDownload,
  onClose,
}: {
  open: boolean;
  issueUrl: string;
  onDownload: () => void;
  onClose: () => void;
}) {
  if (!open) {
    return null;
  }

  return (
    <div className="bug-report-overlay" role="presentation">
      <section
        className="panel bug-report-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="bug-report-title"
      >
        <header className="bug-report-header">
          <h2 id="bug-report-title">See a bug?</h2>
          <button
            type="button"
            className="bug-report-close-button"
            aria-label="Close bug report dialog"
            onClick={onClose}
          >
            x
          </button>
        </header>
        <div className="bug-report-body">
          <p>To report a bug:</p>
          <ol>
            <li>
              Click this button to download a log file:{' '}
              <button
                type="button"
                className="bug-report-download-button"
                onClick={onDownload}
              >
                download
              </button>
            </li>
            <li>
              Follow this link to{' '}
              <a href={issueUrl} target="_blank" rel="noreferrer">
                file a new bug
              </a>
              .
            </li>
            <li>Attach the log file and include any notes about the bug</li>
          </ol>
        </div>
      </section>
    </div>
  );
}
