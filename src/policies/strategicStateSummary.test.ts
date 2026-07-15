import { describe, expect, it } from 'vitest';

import { createStrategicPositionCatalogV0 } from '../botEval/strategicPositionCatalog';
import { actionStableKey } from '../engine/actionSurface';
import { legalActionsForDecisionPlayer } from '../engine/decisionActor';
import { scoreGame } from '../engine/scoring';
import type { GameState } from '../engine/types';
import {
  STRATEGIC_STATE_SUMMARY_CONTRACT,
  STRATEGIC_STATE_SUMMARY_VERSION,
  strategicActionDeltasV0,
  strategicStateSummaryV0,
} from './strategicStateSummary';

describe('strategic state summary v0', () => {
  it('emits objective player-view-safe score, deed, income, and card facts', () => {
    const position = requiredPosition('deed-fork-affordable');
    const summary = strategicStateSummaryV0(
      position.state,
      position.perspectivePlayerId
    );

    expect(summary.contract).toBe(STRATEGIC_STATE_SUMMARY_CONTRACT);
    expect(summary.version).toBe(STRATEGIC_STATE_SUMMARY_VERSION);
    expect(summary.visibility).toBe('player-view');
    expect(summary.cards.unknownPropertyCardIds).toHaveLength(
      summary.clock.drawCount + summary.cards.opponentHandCount
    );
    expect(
      summary.players.self.incomeByResult.find((income) => income.result === 9)
        ?.choiceSources
    ).toEqual([
      {
        districtId: 'D4',
        cardId: '29',
        allowedSuits: ['Moons', 'Suns'],
      },
    ]);
    expect(
      summary.players.self.incomeByResult.find((income) => income.result === 10)
        ?.fixedTokensBySuit
    ).toEqual({
      Moons: 1,
      Suns: 0,
      Waves: 1,
      Leaves: 0,
      Wyrms: 0,
      Knots: 1,
    });

    const deed = summary.districts.find(
      (district) => district.districtId === 'D4'
    )?.self.deed;
    expect(deed).toMatchObject({
      cardId: '29',
      progress: 8,
      target: 9,
      remaining: 1,
      matchingLooseResources: 1,
      spendableMatchingResources: 1,
      resourceCompletionShortfall: 0,
      hasResourcesToComplete: true,
    });
  });

  it('reports exact capped tax loss by suit', () => {
    const state = structuredClone(
      requiredPosition('deed-fork-affordable').state
    );
    state.players[0].resources.Moons = 5;

    expect(
      strategicStateSummaryV0(state, 'PlayerA').players.self
        .taxLossIfSuitSelected
    ).toEqual({
      Moons: 4,
      Suns: 0,
      Waves: 0,
      Leaves: 0,
      Wyrms: 0,
      Knots: 0,
    });
  });

  it('rejects malformed resource counts and deed progress', () => {
    const invalidResources = structuredClone(
      requiredPosition('deed-fork-affordable').state
    );
    invalidResources.players[0].resources.Moons = -1;
    expect(() => strategicStateSummaryV0(invalidResources, 'PlayerA')).toThrow(
      'nonnegative safe integer'
    );

    const invalidDeed = structuredClone(
      requiredPosition('deed-fork-affordable').state
    );
    const deed = invalidDeed.districts.find((district) => district.id === 'D4')
      ?.stacks.PlayerA.deed;
    if (!deed) {
      throw new Error('Missing expected D4 deed.');
    }
    deed.progress = 8.5;
    expect(() => strategicStateSummaryV0(invalidDeed, 'PlayerA')).toThrow(
      'Invalid incomplete deed progress'
    );
  });

  it('compares terminal score fields independent of object key order', () => {
    const state = structuredClone(
      requiredPosition('minimum-winning-coalition').state
    );
    const score = scoreGame(state);
    state.phase = 'GameOver';
    state.finalScore = {
      rankTotals: {
        PlayerB: score.rankTotals.PlayerB,
        PlayerA: score.rankTotals.PlayerA,
      },
      resourceTotals: {
        PlayerB: score.resourceTotals.PlayerB,
        PlayerA: score.resourceTotals.PlayerA,
      },
      districtPoints: {
        PlayerB: score.districtPoints.PlayerB,
        PlayerA: score.districtPoints.PlayerA,
      },
      decidedBy: score.decidedBy,
      winner: score.winner,
    };

    expect(() => strategicStateSummaryV0(state, 'PlayerA')).not.toThrow();
  });

  it('is invariant to hidden assignment, draw order, seed, cursor, and log', () => {
    const original = requiredPosition('minimum-winning-coalition').state;
    const hidden = [
      ...original.players[1].hand,
      ...original.deck.draw,
    ].reverse();
    const opponentHandCount = original.players[1].hand.length;
    const reassigned: GameState = {
      ...original,
      seed: 'different-hidden-seed',
      rngCursor: 999,
      players: original.players.map((player) =>
        player.id === 'PlayerB'
          ? { ...player, hand: hidden.slice(0, opponentHandCount) }
          : player
      ),
      deck: {
        ...original.deck,
        draw: hidden.slice(opponentHandCount),
      },
      log: [
        {
          turn: original.turn,
          player: 'PlayerB',
          phase: 'ActionWindow',
          summary: 'hidden assignment should not matter',
        },
      ],
    };

    expect(strategicStateSummaryV0(reassigned, 'PlayerA')).toEqual(
      strategicStateSummaryV0(original, 'PlayerA')
    );
  });

  it('uses Ace-aware district score while keeping raw rank separate', () => {
    const summary = strategicStateSummaryV0(
      requiredPosition('ace-aware-control').state,
      'PlayerA'
    );
    const district = summary.districts.find(
      (candidate) => candidate.districtId === 'D4'
    );

    expect(district?.self).toMatchObject({
      developedRankTotal: 8,
      aceBonus: 2,
      score: 10,
    });
    expect(district?.opponent).toMatchObject({
      developedRankTotal: 9,
      aceBonus: 0,
      score: 9,
    });
    expect(district?.score.control).toBe('self');
  });

  it('reports exact legal one-action swings without assigning action values', () => {
    const position = requiredPosition('rank-tiebreak-conversion');
    const deltas = strategicActionDeltasV0(position.state, 'PlayerA');
    const develop = requiredFocusDelta(position, deltas, 'convert-rank');
    const sell = requiredFocusDelta(position, deltas, 'sell');

    expect(develop).toMatchObject({
      districtPointMarginDelta: 0,
      developedRankMarginDelta: 2,
      currentOutcomeBefore: 'behind',
      currentOutcomeAfter: 'ahead',
      playedCardDestination: 'developed',
    });
    expect(sell).toMatchObject({
      districtPointMarginDelta: 0,
      developedRankMarginDelta: 0,
      currentOutcomeBefore: 'behind',
      currentOutcomeAfter: 'behind',
      playedCardDestination: 'dead-discard',
    });
  });

  it('classifies the same sale on opposite sides of the reshuffle boundary', () => {
    const before = requiredPosition('sale-before-first-reshuffle');
    const after = requiredPosition('sale-after-first-reshuffle');

    expect(
      requiredFocusDelta(
        before,
        strategicActionDeltasV0(before.state, 'PlayerA'),
        'sell'
      ).playedCardDestination
    ).toBe('first-reshuffle-discard');
    expect(
      requiredFocusDelta(
        after,
        strategicActionDeltasV0(after.state, 'PlayerA'),
        'sell'
      ).playedCardDestination
    ).toBe('dead-discard');
  });

  it('aligns action delta keys with canonical legal actions', () => {
    const position = requiredPosition('known-hand-optionality-original');
    const legalKeys = legalActionsForDecisionPlayer(
      position.state,
      'PlayerA'
    ).map(actionStableKey);
    const deltaKeys = strategicActionDeltasV0(position.state, 'PlayerA').map(
      (delta) => delta.actionKey
    );

    expect(deltaKeys).toEqual(
      [...legalKeys].sort((a, b) => a.localeCompare(b))
    );
  });
});

function requiredPosition(id: string) {
  const position = createStrategicPositionCatalogV0().find(
    (candidate) => candidate.id === id
  );
  if (!position) {
    throw new Error(`Missing strategic position ${id}.`);
  }
  return position;
}

function requiredFocusDelta(
  position: ReturnType<typeof requiredPosition>,
  deltas: ReturnType<typeof strategicActionDeltasV0>,
  focusActionId: string
) {
  const focus = position.focusActions.find(
    (candidate) => candidate.id === focusActionId
  );
  const delta = deltas.find(
    (candidate) => candidate.actionKey === focus?.actionKey
  );
  if (!focus || !delta) {
    throw new Error(
      `Missing strategic focus delta ${position.id}:${focusActionId}.`
    );
  }
  return delta;
}
