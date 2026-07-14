import { describe, expect, it } from 'vitest';

import { actionStableKey } from '../engine/actionSurface';
import { PROPERTY_CARDS, type CardId } from '../engine/cards';
import { legalActionsForDecisionPlayer } from '../engine/decisionActor';
import { applyKnownLegalAction } from '../engine/reducer';
import { rngFromSeed } from '../engine/rng';
import { toPlayerView } from '../engine/view';
import { sampleHiddenWorldStates } from '../policies/determinization';
import {
  strategicActionDeltasV0,
  strategicStateSummaryV0,
} from '../policies/strategicStateSummary';
import {
  STRATEGIC_POSITION_CATALOG_VERSION,
  createStrategicPositionCatalogV0,
} from './strategicPositionCatalog';

describe('strategic position catalog v0', () => {
  it('contains unique, complete, legal, determinizable positions', () => {
    const positions = createStrategicPositionCatalogV0();
    expect(positions.length).toBeGreaterThanOrEqual(8);
    expect(new Set(positions.map((position) => position.id)).size).toBe(
      positions.length
    );

    for (const position of positions) {
      expect(position.catalogVersion).toBe(STRATEGIC_POSITION_CATALOG_VERSION);
      expect(position.state.phase).toBe('ActionWindow');
      expect(position.state.players[position.state.activePlayerIndex]?.id).toBe(
        position.perspectivePlayerId
      );
      const legalActions = legalActionsForDecisionPlayer(
        position.state,
        position.perspectivePlayerId
      );
      const legalKeys = new Set(legalActions.map(actionStableKey));
      expect(legalActions.length).toBeGreaterThan(1);
      for (const focus of position.focusActions) {
        expect(legalKeys.has(focus.actionKey), position.id).toBe(true);
      }

      const summary = strategicStateSummaryV0(
        position.state,
        position.perspectivePlayerId
      );
      expect(summary.cards.unknownPropertyCardIds).toHaveLength(
        summary.clock.drawCount + summary.cards.opponentHandCount
      );
      expect(propertyCardPartition(position.state)).toEqual(
        PROPERTY_CARDS.map((card) => card.id)
      );
      expect(() =>
        sampleHiddenWorldStates({
          state: position.state,
          view: toDecisionView(position),
          rootPlayer: position.perspectivePlayerId,
          worldCount: 1,
          random: rngFromSeed(`catalog-check:${position.id}`),
          errorPrefix: position.id,
        })
      ).not.toThrow();
    }
  });

  it('encodes global district and conditional tiebreak relationships', () => {
    const coalition = requiredPosition('minimum-winning-coalition');
    const coalitionDeltas = strategicActionDeltasV0(coalition.state, 'PlayerA');
    expect(
      delta(coalition, coalitionDeltas, 'pivotal').districtPointMarginDelta
    ).toBe(1);
    expect(
      delta(coalition, coalitionDeltas, 'fortress').districtPointMarginDelta
    ).toBe(0);

    const denial = requiredPosition('tie-denial-restores-match');
    const denialDeltas = strategicActionDeltasV0(denial.state, 'PlayerA');
    expect(delta(denial, denialDeltas, 'deny')).toMatchObject({
      districtPointMarginDelta: 1,
      currentOutcomeBefore: 'behind',
      currentOutcomeAfter: 'ahead',
    });
    expect(delta(denial, denialDeltas, 'fortress')).toMatchObject({
      districtPointMarginDelta: 0,
      currentOutcomeBefore: 'behind',
      currentOutcomeAfter: 'behind',
    });
  });

  it('distinguishes endpoint support only after equal immediate actions', () => {
    const position = requiredPosition('endpoint-optionality');
    const deltas = strategicActionDeltasV0(position.state, 'PlayerA');
    const flexible = delta(position, deltas, 'flexible-endpoint');
    const dead = delta(position, deltas, 'dead-endpoint');

    expect({
      district: flexible.districtPointMarginDelta,
      rank: flexible.developedRankMarginDelta,
      resources: flexible.resourceMarginDelta,
      local: flexible.targetDistrictScoreMarginDelta,
    }).toEqual({
      district: dead.districtPointMarginDelta,
      rank: dead.developedRankMarginDelta,
      resources: dead.resourceMarginDelta,
      local: dead.targetDistrictScoreMarginDelta,
    });

    const flexibleAfter = summaryAfterFocus(position, 'flexible-endpoint');
    const deadAfter = summaryAfterFocus(position, 'dead-endpoint');
    const flexibleD4 = district(flexibleAfter, 'D4');
    const deadD4 = district(deadAfter, 'D4');
    expect(flexibleD4.placementSupport.ownHandForSelf).toEqual(['9']);
    expect(deadD4.placementSupport.ownHandForSelf).toEqual([]);
    expect(flexibleD4.placementSupport.unknownPoolForSelf).toHaveLength(8);
    expect(deadD4.placementSupport.unknownPoolForSelf).toEqual(['29']);
  });

  it('separates deed progress from immediate affordability', () => {
    const affordable = strategicStateSummaryV0(
      requiredPosition('deed-fork-affordable').state,
      'PlayerA'
    );
    const inaccessible = strategicStateSummaryV0(
      requiredPosition('deed-fork-inaccessible').state,
      'PlayerA'
    );
    const affordableDeed = district(affordable, 'D4').self.deed;
    const inaccessibleDeed = district(inaccessible, 'D4').self.deed;

    expect(affordableDeed?.progress).toBe(inaccessibleDeed?.progress);
    expect(affordableDeed?.remaining).toBe(inaccessibleDeed?.remaining);
    expect(affordableDeed).toMatchObject({
      hasResourcesToComplete: true,
      resourceCompletionShortfall: 0,
    });
    expect(inaccessibleDeed).toMatchObject({
      hasResourcesToComplete: false,
      resourceCompletionShortfall: 1,
    });

    const completion = delta(
      requiredPosition('deed-fork-affordable'),
      strategicActionDeltasV0(
        requiredPosition('deed-fork-affordable').state,
        'PlayerA'
      ),
      'complete-deed'
    );
    expect(completion).toMatchObject({
      districtPointMarginDelta: 2,
      currentOutcomeBefore: 'behind',
      currentOutcomeAfter: 'ahead',
      cardPlayAvailableAfterAction: true,
    });
  });

  it('isolates reshuffle status in the paired clock positions', () => {
    const before = requiredPosition('sale-before-first-reshuffle');
    const after = requiredPosition('sale-after-first-reshuffle');
    const beforeSummary = strategicStateSummaryV0(before.state, 'PlayerA');
    const afterSummary = strategicStateSummaryV0(after.state, 'PlayerA');

    expect({ ...beforeSummary.clock, reshuffles: null }).toEqual({
      ...afterSummary.clock,
      reshuffles: null,
    });
    expect(beforeSummary.clock.reshuffles).toBe(0);
    expect(afterSummary.clock.reshuffles).toBe(1);
    expect(beforeSummary.cards).toEqual(afterSummary.cards);
    expect(beforeSummary.districts).toEqual(afterSummary.districts);
    expect(beforeSummary.players).toEqual(afterSummary.players);
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

function delta(
  position: ReturnType<typeof requiredPosition>,
  deltas: ReturnType<typeof strategicActionDeltasV0>,
  focusActionId: string
) {
  const actionKey = position.focusActions.find(
    (focus) => focus.id === focusActionId
  )?.actionKey;
  const match = deltas.find((candidate) => candidate.actionKey === actionKey);
  if (!match) {
    throw new Error(`Missing delta ${position.id}:${focusActionId}.`);
  }
  return match;
}

function summaryAfterFocus(
  position: ReturnType<typeof requiredPosition>,
  focusActionId: string
) {
  const focus = position.focusActions.find(
    (candidate) => candidate.id === focusActionId
  );
  const action = legalActionsForDecisionPlayer(
    position.state,
    position.perspectivePlayerId
  ).find((candidate) => actionStableKey(candidate) === focus?.actionKey);
  if (!focus || !action) {
    throw new Error(`Missing focus action ${position.id}:${focusActionId}.`);
  }
  return strategicStateSummaryV0(
    applyKnownLegalAction(position.state, action, { recordLog: false }),
    position.perspectivePlayerId
  );
}

function district(
  summary: ReturnType<typeof strategicStateSummaryV0>,
  districtId: string
) {
  const match = summary.districts.find(
    (candidate) => candidate.districtId === districtId
  );
  if (!match) {
    throw new Error(`Missing district ${districtId}.`);
  }
  return match;
}

function toDecisionView(position: ReturnType<typeof requiredPosition>) {
  return toPlayerView(position.state, position.perspectivePlayerId);
}

function propertyCardPartition(
  state: ReturnType<typeof requiredPosition>['state']
): CardId[] {
  const cardIndex = new Map(
    PROPERTY_CARDS.map((card, index) => [card.id, index])
  );
  return [
    ...state.players.flatMap((player) => player.hand),
    ...state.deck.draw,
    ...state.deck.discard,
    ...state.districts.flatMap((district) =>
      (['PlayerA', 'PlayerB'] as const).flatMap((playerId) => {
        const stack = district.stacks[playerId];
        return [...stack.developed, ...(stack.deed ? [stack.deed.cardId] : [])];
      })
    ),
  ].sort(
    (left, right) =>
      (cardIndex.get(left) ?? Number.MAX_SAFE_INTEGER) -
      (cardIndex.get(right) ?? Number.MAX_SAFE_INTEGER)
  );
}
