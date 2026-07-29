export function BugReportInstructions({
  issueUrl,
  onDownload,
}: {
  issueUrl: string;
  onDownload: () => void;
}) {
  return (
    <section className="bug-report-instructions">
      <div className="bug-report-body">
        <p>See a bug? A report would be appreciated. To file:</p>
        <ol>
          <li>
            Click to{' '}
            <button
              type="button"
              className="bug-report-download-link"
              title="Download a diagnostic log file"
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
