import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { FinalScore } from '../../engine/types';
import type { HumanActionListItem } from '../actionPresentation';
import { ActionsPanel } from './ActionsPanel';

const SCORE: FinalScore = {
  districtPoints: { PlayerA: 2, PlayerB: 1 },
  rankTotals: { PlayerA: 18, PlayerB: 14 },
  resourceTotals: { PlayerA: 6, PlayerB: 4 },
  winner: 'PlayerA',
  decidedBy: 'districts',
};
const noop = () => {};
const ACTION_ITEMS: HumanActionListItem[] = [
  { kind: 'action', action: { type: 'sell-card', cardId: '6' } },
  { kind: 'action', action: { type: 'sell-card', cardId: '7' } },
  {
    kind: 'trade-group',
    give: 'Moons',
    options: [
      { type: 'trade', give: 'Moons', receive: 'Suns' },
      { type: 'trade', give: 'Moons', receive: 'Waves' },
    ],
  },
];

function renderPanel(
  overrides: Partial<Parameters<typeof ActionsPanel>[0]> = {}
): string {
  return renderToStaticMarkup(
    <ActionsPanel
      terminal={false}
      isLastTurn={false}
      score={SCORE}
      wonDistrictsByPlayer={{ PlayerA: ['D1', 'D2'], PlayerB: ['D3'] }}
      activePlayerId="PlayerA"
      humanPlayerId="PlayerA"
      botPlayerId="PlayerB"
      visibleActionItems={ACTION_ITEMS}
      isIncomeChoicePhase={false}
      hasMultipleTradeSources={false}
      actionPicker={null}
      canResetTurn={false}
      botThinking={false}
      showBotThinkingDuringIncomeChoiceLock={false}
      hideBotWaitMessageDuringTurnCycleLock={false}
      humanActionUiBlockedByAnimation={false}
      humanActionUiBlockedByTurnCycleAnimation={false}
      onAction={noop}
      onResetTurn={noop}
      onClosePicker={noop}
      onOpenTradeCombinedPicker={noop}
      onOpenTradePicker={noop}
      onOpenDistrictPicker={noop}
      onOpenDevelopOutrightCombinedPicker={noop}
      onOpenDevelopOutrightDistrictOnlyPicker={noop}
      onOpenDeedPaymentPicker={noop}
      {...overrides}
    />
  );
}

describe('ActionsPanel', () => {
  it('renders terminal summary', () => {
    const html = renderPanel({ terminal: true });

    expect(html).toContain('<h2>Game Over</h2>');
    expect(html).toContain('Winner: <strong>You</strong>');
  });

  it('renders contiguous categories once, grouped submenus, and reset control', () => {
    const html = renderPanel({ canResetTurn: true, isLastTurn: true });

    expect(html.match(/>Sell Card</g)).toHaveLength(1);
    expect(html.match(/>Trade</g)).toHaveLength(1);
    expect(html).toContain('action-button has-submenu');
    expect(html).toContain('Reset turn');
    expect(html).toContain('Last Turn');
  });

  it('renders animation lock and bot status messages', () => {
    expect(
      renderPanel({
        humanActionUiBlockedByAnimation: true,
        humanActionUiBlockedByTurnCycleAnimation: true,
      })
    ).toContain('Resolving income and taxation...');
    expect(
      renderPanel({
        activePlayerId: 'PlayerB',
        botThinking: true,
      })
    ).toContain('Bot is thinking...');
    expect(
      renderPanel({
        activePlayerId: 'PlayerB',
      })
    ).toContain('Waiting for bot...');
  });

  it('hides bot wait text during a non-income turn-cycle lock', () => {
    const html = renderPanel({
      activePlayerId: 'PlayerB',
      hideBotWaitMessageDuringTurnCycleLock: true,
    });

    expect(html).not.toContain('Bot is thinking...');
    expect(html).not.toContain('Waiting for bot...');
  });

  it('renders human income choices during a bot-owned shared income phase', () => {
    const html = renderPanel({
      activePlayerId: 'PlayerB',
      isIncomeChoicePhase: true,
      visibleActionItems: [
        {
          kind: 'action',
          action: {
            type: 'choose-income-suit',
            playerId: 'PlayerA',
            districtId: 'D1',
            cardId: '6',
            suit: 'Moons',
          },
        },
      ],
    });

    expect(html).toContain('Choose Income');
    expect(html).not.toContain('Waiting for bot...');
  });

  it('uses shared income-choice wording after the human has submitted', () => {
    const html = renderPanel({
      isIncomeChoicePhase: true,
      visibleActionItems: [],
    });

    expect(html).toContain('Resolving income choices...');
    expect(html).not.toContain('No legal actions.');
  });
});
