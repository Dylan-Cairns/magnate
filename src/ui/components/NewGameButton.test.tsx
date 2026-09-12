import { createRef } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { NewGameButton } from './NewGameButton';

const noop = () => {};

function renderButton(expanded: boolean): string {
  return renderToStaticMarkup(
    <NewGameButton
      expanded={expanded}
      panelRef={createRef<HTMLElement>()}
      buttonRef={createRef<HTMLButtonElement>()}
      seedInputRef={createRef<HTMLInputElement>()}
      botProfileId="rollout-search-v2-medium"
      botStatusText="Selected bot status"
      ruleset="standard"
      animationsEnabled
      onToggle={noop}
      onStart={noop}
      onBotProfileChange={noop}
      onRulesetChange={noop}
      onAnimationsEnabledChange={noop}
    />
  );
}

describe('NewGameButton', () => {
  it('keeps the trigger labeled New Game and hides setup until expanded', () => {
    const html = renderButton(false);

    expect(html).toContain('New Game');
    expect(html).not.toContain('Start Game');
    expect(html).not.toContain('id="new-game-panel"');
    expect(html).toContain('aria-expanded="false"');
  });

  it('renders a dedicated start control while expanded', () => {
    const html = renderButton(true);

    expect(html).toContain('New Game');
    expect(html).toContain('Start Game');
    expect(html).toContain('id="new-game-panel"');
    expect(html).toContain('aria-expanded="true"');
  });
});
