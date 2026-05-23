import { createRef } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { OptionsBackdrop, OptionsMenu } from './OptionsMenu';

const noop = () => {};

describe('OptionsMenu', () => {
  it('renders stable control ids and current selections while open', () => {
    const html = renderToStaticMarkup(
      <OptionsMenu
        open
        botProfileId="rollout-eval-search"
        botStatusText="Selected bot status"
        animationsEnabled
        menuRef={createRef<HTMLElement>()}
        buttonRef={createRef<HTMLButtonElement>()}
        seedInputRef={createRef<HTMLInputElement>()}
        newGameExpanded
        newGamePanelRef={createRef<HTMLElement>()}
        newGameButtonRef={createRef<HTMLButtonElement>()}
        onBugReport={noop}
        onToggle={noop}
        onNewGameToggle={noop}
        onBotProfileChange={noop}
        onAnimationsEnabledChange={noop}
        bugReportOpen={false}
        bugReportIssueUrl="https://github.com/Dylan-Cairns/magnate/issues/new"
        onBugReportDownload={noop}
        logVisible
        onToggleLog={noop}
        mapVisible
        onToggleMap={noop}
        deckMapInteractive
        onDeckMapInteractiveChange={noop}
        onHistoryOpen={noop}
      />
    );

    expect(html).toContain('id="brand-options-menu"');
    expect(html).toContain('aria-label="Report a bug"');
    expect(html).toContain('id="seed-input"');
    expect(html).toContain('id="bot-profile-select"');
    expect(html).toContain('value="rollout-eval-search" selected=""');
    expect(html).toContain('id="animations-toggle" type="checkbox" checked=""');
    expect(html).toContain('Selected bot status');
  });

  it('renders the backdrop only while open', () => {
    expect(
      renderToStaticMarkup(<OptionsBackdrop open onClose={noop} />)
    ).toContain('options-backdrop');
    expect(
      renderToStaticMarkup(<OptionsBackdrop open={false} onClose={noop} />)
    ).toBe('');
  });
});
