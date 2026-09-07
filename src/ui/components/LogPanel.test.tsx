import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { GameAction, GameLogEntry, PlayerId } from '../../engine/types';
import { makeGameState } from '../../engine/__tests__/fixtures';
import { stepToDecision } from '../../engine/session';
import { transitionLogEntries } from '../logTimeline';
import { formatLogSummary } from '../logPresentation';
import { LogPanel } from './LogPanel';

const LOG: GameLogEntry[] = [
  {
    turn: 0,
    player: 'PlayerA',
    phase: 'StartTurn',
    summary: 'Seed fixed-seed',
  },
  {
    turn: 1,
    player: 'PlayerA',
    phase: 'ActionWindow',
    summary: 'trade Moons for Knots',
  },
  {
    turn: 1,
    player: 'PlayerB',
    phase: 'CollectIncome',
    summary: 'income choice 6:Waves',
  },
];

describe('LogPanel', () => {
  it.each([
    ['PlayerA', 'PlayerB', 0],
    ['PlayerA', 'PlayerB', 1],
    ['PlayerB', 'PlayerA', 0],
    ['PlayerB', 'PlayerA', 1],
  ] as const)(
    'hides opponent choices until every submission resolves (viewer %s, opponent %s, turn owner %s)',
    (humanPlayerId: PlayerId, botPlayerId: PlayerId, activePlayerIndex) => {
      let state = makeGameState({
        turn: 22,
        phase: 'CollectIncome',
        activePlayerIndex,
        incomeChoiceReturnPlayerId:
          activePlayerIndex === 0 ? 'PlayerA' : 'PlayerB',
        pendingIncomeChoices: [
          {
            playerId: botPlayerId,
            districtId: 'D1',
            cardId: '27',
            suits: ['Waves', 'Wyrms'],
          },
          {
            playerId: humanPlayerId,
            districtId: 'D2',
            cardId: '28',
            suits: ['Leaves', 'Knots'],
          },
          {
            playerId: humanPlayerId,
            districtId: 'D3',
            cardId: '29',
            suits: ['Moons', 'Knots'],
          },
        ],
      });
      const timelineLog: GameLogEntry[] = [
        {
          turn: 21,
          phase: 'CollectIncome',
          player: botPlayerId,
          summary: 'income choice 6:Waves',
        },
      ];
      const submit = (action: GameAction) => {
        const next = stepToDecision(state, action);
        timelineLog.push(
          ...transitionLogEntries(state, next, action, humanPlayerId)
        );
        state = next;
      };
      const render = () =>
        renderToStaticMarkup(
          <LogPanel
            timelineLog={timelineLog}
            humanPlayerId={humanPlayerId}
            state={state}
          />
        ).replace(/<[^>]*>/g, '');
      const botChoice = formatLogSummary('income choice 27:Waves');

      submit({
        type: 'choose-income-suit',
        playerId: botPlayerId,
        districtId: 'D1',
        cardId: '27',
        suit: 'Waves',
      });
      // The raw export timeline contains the choice; the rendered panel must not.
      expect(
        timelineLog.some((entry) => entry.summary === 'income choice 27:Waves')
      ).toBe(true);
      expect(render()).not.toContain(botChoice);
      expect(render()).toContain(formatLogSummary('income choice 6:Waves'));

      submit({
        type: 'choose-income-suit',
        playerId: humanPlayerId,
        districtId: 'D2',
        cardId: '28',
        suit: 'Knots',
      });
      expect(render()).not.toContain(botChoice);
      expect(render()).toContain(formatLogSummary('income choice 28:Knots'));

      submit({
        type: 'choose-income-suit',
        playerId: humanPlayerId,
        districtId: 'D3',
        cardId: '29',
        suit: 'Moons',
      });
      expect(state.phase).toBe('ActionWindow');
      expect(render().split(botChoice)).toHaveLength(2);
    }
  );

  it('renders reverse-chronological grouped entries, seed rows, and colored suit codes', () => {
    const html = renderToStaticMarkup(
      <LogPanel
        timelineLog={LOG}
        humanPlayerId="PlayerA"
        state={makeGameState()}
      />
    );

    expect(html.indexOf('T1')).toBeLessThan(html.indexOf('Seed'));
    expect(html).toContain('>You<');
    expect(html).toContain('[Bot] Income choice');
    expect(html).toContain('class="log-suit-code"');
    expect(html).toContain('>wa<');
    expect(html).toContain('fixed-seed');
  });

  it('renders an empty state without entries', () => {
    const html = renderToStaticMarkup(
      <LogPanel
        timelineLog={[]}
        humanPlayerId="PlayerA"
        state={makeGameState()}
      />
    );

    expect(html).toContain('No actions yet.');
  });
});
