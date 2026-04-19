export function BugReportModal({
  open,
  issueUrl,
  onDownload,
}: {
  open: boolean;
  issueUrl: string;
  onDownload: () => void;
}) {
  if (!open) {
    return null;
  }

  return (
    <section
      className="panel bug-report-modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="bug-report-title"
    >
      <h2 id="bug-report-title">See a bug?</h2>
      <div className="bug-report-body">
        <p>To report a bug:</p>
        <ol>
          <li>
            Click to{' '}
            <button
              type="button"
              className="bug-report-download-link"
              onClick={onDownload}
            >
              download a log file
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
  );
}
