import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import {
  ResolutionWarningOverlay,
  StartupPreloadOverlay,
  TurnCycleOverlay,
} from './GameOverlays';

const noop = () => {};

describe('StartupPreloadOverlay', () => {
  it('clamps progress display and renders retry state after an error', () => {
    const html = renderToStaticMarkup(
      <StartupPreloadOverlay
        ready={false}
        error="network failed"
        progress={{
          completed: 8,
          total: 5,
          percent: 120,
          message: 'Loading',
        }}
        onRetry={noop}
      />
    );

    expect(html).toContain('Loading Failed');
    expect(html).toContain('Could not preload assets: network failed');
    expect(html).toContain('aria-valuenow="100"');
    expect(html).toContain('width:100%');
    expect(html).toContain('5 / 5');
    expect(html).toContain('Retry');
  });
});

describe('ResolutionWarningOverlay', () => {
  it('renders only while open', () => {
    expect(
      renderToStaticMarkup(<ResolutionWarningOverlay open onDismiss={noop} />)
    ).toContain('Display Warning');
    expect(
      renderToStaticMarkup(
        <ResolutionWarningOverlay open={false} onDismiss={noop} />
      )
    ).toBe('');
  });
});

describe('TurnCycleOverlay', () => {
  it('renders tax, income, and winner variants', () => {
    const tax = renderToStaticMarkup(
      <TurnCycleOverlay
        overlay={{ kind: 'tax', suit: 'Moons' }}
        terminalWinner={null}
        humanPlayerId="PlayerA"
      />
    );
    const income = renderToStaticMarkup(
      <TurnCycleOverlay
        overlay={{ kind: 'income', rank: 9 }}
        terminalWinner={null}
        humanPlayerId="PlayerA"
      />
    );
    const winner = renderToStaticMarkup(
      <TurnCycleOverlay
        overlay={{ kind: 'income', rank: 9 }}
        terminalWinner="PlayerB"
        humanPlayerId="PlayerA"
      />
    );

    expect(tax).toContain('Taxes:');
    expect(tax).toContain('data-token-suit="Moons"');
    expect(income).toContain('Income:');
    expect(income).toContain('>9</strong>');
    expect(winner).toContain('Winner:');
    expect(winner).toContain('>Bot</strong>');
  });
});
