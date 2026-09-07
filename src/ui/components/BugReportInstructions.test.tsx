import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { BugReportInstructions } from './BugReportInstructions';

const noop = () => {};

describe('BugReportInstructions', () => {
  it('renders the requested report instructions', () => {
    const html = renderToStaticMarkup(
      <BugReportInstructions
        issueUrl="https://github.com/Dylan-Cairns/magnate/issues/new"
        onDownload={noop}
      />
    );

    expect(html).toContain('See a bug?');
    expect(html).toContain('A report would be appreciated. To file:');
    expect(html).toContain('download');
    expect(html).toContain('file a new bug');
    expect(html).toContain(
      'Attach the log file and include any notes about the bug'
    );
  });
});
