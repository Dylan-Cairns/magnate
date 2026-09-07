import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';

import { makeGameState } from '../../engine/__tests__/fixtures';
import type { GameState } from '../../engine/types';
import { useGameController } from './useGameController';

const presentation = vi.hoisted(() => ({
  viewState: null as GameState | null,
  presentedState: null as GameState | null,
}));

vi.mock('../gameControllerModel', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../gameControllerModel')>()),
  createBrowserSession: () => makeGameState({ phase: 'GameOver' }),
}));

vi.mock('../../policies/catalog', () => ({
  DEFAULT_BOT_PROFILE_ID: 'test-bot',
  resolveBotProfile: () => ({ selected: { id: 'test-bot' }, policy: {} }),
}));

vi.mock('./useGameAnimations', () => ({
  useGameAnimations: () => ({
    enabled: true,
    presentationSnapshot: presentation.viewState
      ? { viewState: presentation.viewState }
      : null,
    presentedState: presentation.presentedState,
    presentationPending: Boolean(
      presentation.viewState ?? presentation.presentedState
    ),
  }),
}));

function ControllerHarness() {
  const controller = useGameController({
    humanPlayerId: 'PlayerA',
    botPlayerId: 'PlayerB',
    startupPreloadReady: false,
  });
  return (
    <span>
      {controller.state.phase}:{controller.terminal ? 'winner' : 'playing'}:
      {controller.humanActionsAcceptingInput.length}
    </span>
  );
}

describe('useGameController winner presentation', () => {
  it('keeps the winner hidden during final-turn animation and between queued actions', () => {
    const finalBotTurn = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 1,
      finalTurnsRemaining: 1,
    });
    presentation.viewState = finalBotTurn;
    presentation.presentedState = null;
    expect(renderToStaticMarkup(<ControllerHarness />)).toContain(
      'GameOver:playing:0'
    );

    presentation.viewState = null;
    presentation.presentedState = finalBotTurn;
    expect(renderToStaticMarkup(<ControllerHarness />)).toContain(
      'GameOver:playing:0'
    );

    presentation.presentedState = null;
    expect(renderToStaticMarkup(<ControllerHarness />)).toContain(
      'GameOver:winner:0'
    );
  });
});
