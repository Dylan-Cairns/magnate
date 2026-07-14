import { describe, expect, it } from 'vitest';

import {
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../../engine/__tests__/fixtures';
import { buildAnimationSequence } from '../runtime/animationSequence';
import { buildGameTransaction } from '../runtime/transactions';
import { sequencePresentationSnapshotUpdateTimes } from './useGameAnimations';

describe('useGameAnimations scheduling helpers', () => {
  it('does not schedule a presentation snapshot at the final commit/unlock boundary', () => {
    const transaction = buildGameTransaction({
      previousState: makeGameState({
        players: [
          makePlayer(PLAYER_A, {
            hand: ['6'],
            resources: makeResources({ Moons: 2, Knots: 2 }),
          }),
          makePlayer(PLAYER_B),
        ],
      }),
      action: { type: 'buy-deed', cardId: '6', districtId: 'D1' },
      actingPlayerId: PLAYER_A,
      transactionId: 'tx-buy-deed',
      stepToDecision: () =>
        makeGameState({
          players: [
            makePlayer(PLAYER_A, {
              hand: [],
              resources: makeResources({ Moons: 1, Knots: 1 }),
            }),
            makePlayer(PLAYER_B),
          ],
          districts: [
            makeDistrict('D1', ['Moons'], {
              [PLAYER_A]: {
                developed: [],
                deed: { cardId: '6', progress: 0, tokens: {} },
              },
            }),
          ],
        }),
    });
    const sequence = buildAnimationSequence(transaction);
    const finalBoundaryMs = Math.min(sequence.commitMs, sequence.inputUnlockMs);

    const updateTimes = sequencePresentationSnapshotUpdateTimes(
      sequence,
      sequence.durationMs,
      finalBoundaryMs
    );

    expect(sequence.commitMs).toBe(sequence.inputUnlockMs);
    expect(updateTimes).not.toContain(finalBoundaryMs);
    expect(updateTimes.every((atMs) => atMs < finalBoundaryMs)).toBe(true);
  });
});
