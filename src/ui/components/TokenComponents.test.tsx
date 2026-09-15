import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { GameAction } from '../../engine/types';
import { makeGameState, makeResources } from '../../engine/__tests__/fixtures';
import { ActionHighlights } from './ActionHighlights';
import { TokenRow } from './TokenComponents';

const TRADE_ACTIONS: GameAction[] = [
  { type: 'trade', give: 'Moons', receive: 'Suns' },
];

function renderResourceRow(tokens: ReturnType<typeof makeResources>): string {
  return renderToStaticMarkup(
    <ActionHighlights
      state={makeGameState()}
      picker={{
        kind: 'trade-combined',
        selectedGive: 'Moons',
        selectedReceive: 'Suns',
        top: 0,
        left: 0,
      }}
      legalActions={TRADE_ACTIONS}
    >
      <TokenRow tokens={tokens} fixedSuitSlots compact highlightResources />
    </ActionHighlights>
  );
}

describe('TokenRow', () => {
  it('previews a gained suit with a ghost chip in the empty slot', () => {
    const html = renderResourceRow(makeResources());

    expect((html.match(/is-token-ghost/g) ?? []).length).toBe(1);
    expect(html).not.toContain('data-token-suit="Suns"');
    expect(html).toMatch(
      /class="token-chip[^"]*empty[^"]*is-action-highlighted[^"]*" data-token-suit="Moons"/
    );
  });

  it('keeps an existing chip highlighted without a ghost when its suit will gain', () => {
    const html = renderResourceRow(makeResources({ Suns: 2 }));

    expect(html).not.toContain('is-token-ghost');
    expect(html).toMatch(
      /class="token-chip[^"]*is-action-highlighted[^"]*" data-token-suit="Suns"/
    );
    expect(html).toContain('x2');
  });
});
