import { createRef } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { GameAction } from '../../engine/types';
import { buildTradeSourceGroups } from '../actionPresentation';
import { ActionPicker } from './ActionPicker';

const noop = () => {};
const TRADE_ACTIONS: GameAction[] = [
  { type: 'trade', give: 'Moons', receive: 'Suns' },
  { type: 'trade', give: 'Moons', receive: 'Waves' },
  { type: 'trade', give: 'Knots', receive: 'Suns' },
];
const OUTRIGHT_ACTIONS: GameAction[] = [
  {
    type: 'develop-outright',
    cardId: '6',
    districtId: 'D1',
    payment: { Moons: 2, Knots: 1 },
  },
  {
    type: 'develop-outright',
    cardId: '6',
    districtId: 'D2',
    payment: { Moons: 1, Knots: 2 },
  },
];

describe('ActionPicker', () => {
  it('renders a positioned standard picker and cancel control', () => {
    const html = renderToStaticMarkup(
      <ActionPicker
        picker={{ kind: 'trade', give: 'Moons', top: 12, left: 34 }}
        pickerRef={createRef<HTMLElement>()}
        legalActions={TRADE_ACTIONS}
        tradeSourceGroups={buildTradeSourceGroups(TRADE_ACTIONS)}
        onPickerChange={noop}
        onSelectAction={noop}
        onClose={noop}
      />
    );

    expect(html).toContain('role="dialog"');
    expect(html).toContain('top:12px;left:34px');
    expect(html).toContain('Trade ');
    expect(html).toContain('x3 for');
    expect(html).toContain('data-token-suit="Suns"');
    expect(html).toContain('Cancel');
  });

  it('renders selected combined trade controls', () => {
    const html = renderToStaticMarkup(
      <ActionPicker
        picker={{
          kind: 'trade-combined',
          selectedGive: 'Moons',
          selectedReceive: 'Suns',
          top: 0,
          left: 0,
        }}
        pickerRef={createRef<HTMLElement>()}
        legalActions={TRADE_ACTIONS}
        tradeSourceGroups={buildTradeSourceGroups(TRADE_ACTIONS)}
        onPickerChange={noop}
        onSelectAction={noop}
        onClose={noop}
      />
    );

    expect(html).toContain('Trade resources');
    expect(html).toContain('Give x3');
    expect(html).toContain('Receive x1');
    expect(html.match(/is-selected/g)).toHaveLength(2);
  });

  it('renders combined develop-outright district and payment controls', () => {
    const html = renderToStaticMarkup(
      <ActionPicker
        picker={{
          kind: 'develop-outright-combined',
          cardId: '6',
          selectedDistrictId: 'D1',
          top: 0,
          left: 0,
        }}
        pickerRef={createRef<HTMLElement>()}
        legalActions={OUTRIGHT_ACTIONS}
        tradeSourceGroups={[]}
        onPickerChange={noop}
        onSelectAction={noop}
        onClose={noop}
      />
    );

    expect(html).toContain('District');
    expect(html).toContain('Payment');
    expect(html).toContain('>D1</button>');
    expect(html).toContain('>D2</button>');
    expect(html).toContain('data-token-suit="Moons"');
    expect(html).toContain('data-token-suit="Knots"');
  });
});
