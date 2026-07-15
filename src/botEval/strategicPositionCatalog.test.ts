import { describe, expect, it } from 'vitest';

import { actionStableKey } from '../engine/actionSurface';
import { PROPERTY_CARDS, type CardId } from '../engine/cards';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
} from '../engine/decisionActor';
import { applyKnownLegalAction } from '../engine/reducer';
import { rngFromSeed } from '../engine/rng';
import { stepKnownLegalActionToDecision } from '../engine/session';
import type { GameAction, GameState } from '../engine/types';
import { toPlayerView } from '../engine/view';
import { sampleHiddenWorldStates } from '../policies/determinization';
import {
  strategicActionDeltasV0,
  strategicStateSummaryV0,
} from '../policies/strategicStateSummary';
import {
  STRATEGIC_POSITION_CATALOG_VERSION,
  createStrategicPositionCatalogV0,
  isStrategicOptionalityPositionV0,
} from './strategicPositionCatalog';

describe('strategic position catalog v1', () => {
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

  it('isolates same-card endpoint optionality in original and mirrored lanes', () => {
    const families = [
      {
        prefix: 'known-hand-optionality',
        stableHandCardOccurrences: [['0', 0]] as const,
        changingHandCardIds: ['6'] as const,
        stableUnknownCardIds: ['8', '26', '9', '18', '13', '25'] as const,
        changingUnknownCardIds: [] as const,
        stableUnknownOpponentOccurrences: 0,
        changingUnknownOpponentOccurrences: 0,
      },
      {
        prefix: 'unknown-pool-optionality',
        stableHandCardOccurrences: [
          ['25', 1],
          ['29', 1],
        ] as const,
        changingHandCardIds: [] as const,
        stableUnknownCardIds: ['8', '26', '18'] as const,
        changingUnknownCardIds: ['20'] as const,
        stableUnknownOpponentOccurrences: 0,
        changingUnknownOpponentOccurrences: 2,
      },
    ];

    for (const family of families) {
      const original = requiredPosition(`${family.prefix}-original`);
      const mirror = requiredPosition(`${family.prefix}-mirror`);
      expect(actionForFocus(original, 'preserve-option')).toMatchObject({
        type: 'develop-outright',
        cardId: '14',
        districtId: 'D4',
        payment: { Waves: 2, Leaves: 2 },
      });
      expect(actionForFocus(original, 'overwrite-option')).toMatchObject({
        type: 'develop-outright',
        cardId: '14',
        districtId: 'D1',
        payment: { Waves: 2, Leaves: 2 },
      });
      expect(actionForFocus(mirror, 'preserve-option')).toMatchObject({
        type: 'develop-outright',
        cardId: '14',
        districtId: 'D1',
        payment: { Waves: 2, Leaves: 2 },
      });
      expect(actionForFocus(mirror, 'overwrite-option')).toMatchObject({
        type: 'develop-outright',
        cardId: '14',
        districtId: 'D4',
        payment: { Waves: 2, Leaves: 2 },
      });

      expect(normalizeDistrictPermutation(summary(original))).toEqual(
        normalizeDistrictPermutation(summary(mirror))
      );

      for (const position of [original, mirror]) {
        const deltas = strategicActionDeltasV0(position.state, 'PlayerA');
        const preserve = delta(position, deltas, 'preserve-option');
        const overwrite = delta(position, deltas, 'overwrite-option');
        const expectedImmediateDelta = {
          districtPointMarginDelta: 2,
          developedRankMarginDelta: 4,
          resourceMarginDelta: -4,
          targetDistrictScoreMarginDelta: 4,
          currentOutcomeBefore: 'behind',
          currentOutcomeAfter: 'behind',
          playedCardDestination: 'developed',
        } as const;
        expect(preserve).toMatchObject(expectedImmediateDelta);
        expect(overwrite).toMatchObject(expectedImmediateDelta);

        const preserveAfter = summaryAfterFocus(position, 'preserve-option');
        const overwriteAfter = summaryAfterFocus(position, 'overwrite-option');
        expect(preserveAfter.players).toEqual(overwriteAfter.players);
        expect(preserveAfter.cards).toEqual(overwriteAfter.cards);
        expect(preserveAfter.score).toEqual(overwriteAfter.score);
        expect(developedCardMultiset(preserveAfter)).toEqual(
          developedCardMultiset(overwriteAfter)
        );
        for (const districtId of ['D1', 'D4']) {
          expect(district(preserveAfter, districtId).self.aceBonus).toBe(0);
          expect(district(overwriteAfter, districtId).self.aceBonus).toBe(0);
          expect(district(preserveAfter, districtId).opponent.aceBonus).toBe(0);
          expect(district(overwriteAfter, districtId).opponent.aceBonus).toBe(
            0
          );
        }

        for (const [cardId, occurrences] of family.stableHandCardOccurrences) {
          expect(
            supportOccurrences(preserveAfter, cardId, 'ownHandForSelf')
          ).toBe(occurrences);
          expect(
            supportOccurrences(overwriteAfter, cardId, 'ownHandForSelf')
          ).toBe(occurrences);
        }
        for (const cardId of family.changingHandCardIds) {
          expect(
            supportOccurrences(preserveAfter, cardId, 'ownHandForSelf')
          ).toBe(1);
          expect(
            supportOccurrences(overwriteAfter, cardId, 'ownHandForSelf')
          ).toBe(0);
        }
        for (const cardId of family.stableUnknownCardIds) {
          expect(
            supportOccurrences(overwriteAfter, cardId, 'unknownPoolForSelf')
          ).toBe(
            supportOccurrences(preserveAfter, cardId, 'unknownPoolForSelf')
          );
          expect(
            supportOccurrences(preserveAfter, cardId, 'unknownPoolForOpponent')
          ).toBe(family.stableUnknownOpponentOccurrences);
          expect(
            supportOccurrences(overwriteAfter, cardId, 'unknownPoolForOpponent')
          ).toBe(family.stableUnknownOpponentOccurrences);
        }
        for (const cardId of family.changingUnknownCardIds) {
          expect(
            supportOccurrences(preserveAfter, cardId, 'unknownPoolForSelf')
          ).toBe(1);
          expect(
            supportOccurrences(overwriteAfter, cardId, 'unknownPoolForSelf')
          ).toBe(0);
          expect(
            supportOccurrences(preserveAfter, cardId, 'unknownPoolForOpponent')
          ).toBe(family.changingUnknownOpponentOccurrences);
          expect(
            supportOccurrences(overwriteAfter, cardId, 'unknownPoolForOpponent')
          ).toBe(family.changingUnknownOpponentOccurrences);
        }
        expect(
          preserveAfter.cards.unknownPropertyCardIds.every(
            (cardId) => requiredProperty(cardId).rank !== 1
          )
        ).toBe(true);
      }

      expect(
        normalizeDistrictPermutation(
          summaryAfterFocus(original, 'preserve-option')
        )
      ).toEqual(
        normalizeDistrictPermutation(
          summaryAfterFocus(mirror, 'preserve-option')
        )
      );
      expect(
        normalizeDistrictPermutation(
          summaryAfterFocus(original, 'overwrite-option')
        )
      ).toEqual(
        normalizeDistrictPermutation(
          summaryAfterFocus(mirror, 'overwrite-option')
        )
      );
    }
  });

  it('isolates independent optionality holdouts with new cards, payments, and lanes', () => {
    const families = [
      {
        prefix: 'known-hand-optionality-holdout',
        family: 'known-hand',
        rootCardId: '7',
        targetCardId: '8',
        originalValuableDistrictId: 'D0',
        originalAlternativeDistrictId: 'D3',
        payment: { Suns: 1, Wyrms: 1 },
        developedRankMarginDelta: 2,
        resourceMarginDelta: -2,
        stableOwnHandCardIds: ['1'] as const,
        stableUnknownCardIds: ['12', '15', '16', '17', '23', '24'] as const,
      },
      {
        prefix: 'unknown-pool-optionality-holdout',
        family: 'unknown-pool',
        rootCardId: '13',
        targetCardId: '19',
        originalValuableDistrictId: 'D2',
        originalAlternativeDistrictId: 'D3',
        payment: { Moons: 2, Suns: 2 },
        developedRankMarginDelta: 4,
        resourceMarginDelta: -4,
        stableOwnHandCardIds: ['17', '24'] as const,
        stableUnknownCardIds: ['8', '14', '26'] as const,
      },
    ] as const;

    expect(
      createStrategicPositionCatalogV0()
        .filter(isStrategicOptionalityPositionV0)
        .map((position) => position.id)
    ).toEqual([
      'known-hand-optionality-original',
      'known-hand-optionality-mirror',
      'unknown-pool-optionality-original',
      'unknown-pool-optionality-mirror',
      'known-hand-optionality-holdout-original',
      'known-hand-optionality-holdout-mirror',
      'unknown-pool-optionality-holdout-original',
      'unknown-pool-optionality-holdout-mirror',
    ]);

    for (const family of families) {
      const original = requiredPosition(`${family.prefix}-original`);
      const mirror = requiredPosition(`${family.prefix}-mirror`);
      expect(normalizeDistrictPermutation(summary(original))).toEqual(
        normalizeDistrictPermutation(summary(mirror))
      );

      for (const [position, valuableDistrictId, alternativeDistrictId] of [
        [
          original,
          family.originalValuableDistrictId,
          family.originalAlternativeDistrictId,
        ],
        [
          mirror,
          family.originalAlternativeDistrictId,
          family.originalValuableDistrictId,
        ],
      ] as const) {
        expect(position.optionalityTrace).toEqual({
          family: family.family,
          targetCardId: family.targetCardId,
          valuableDistrictId,
          alternativeDistrictId,
          preserveFocusActionId: 'preserve-option',
          overwriteFocusActionId: 'overwrite-option',
        });

        const rootDevelopments = legalActionsForDecisionPlayer(
          position.state,
          'PlayerA'
        ).filter(
          (
            action
          ): action is Extract<GameAction, { type: 'develop-outright' }> =>
            action.type === 'develop-outright' &&
            action.cardId === family.rootCardId
        );
        expect(rootDevelopments).toHaveLength(2);
        expect(
          new Set(rootDevelopments.map((action) => action.districtId))
        ).toEqual(new Set([valuableDistrictId, alternativeDistrictId]));
        expect(actionForFocus(position, 'preserve-option')).toMatchObject({
          type: 'develop-outright',
          cardId: family.rootCardId,
          districtId: alternativeDistrictId,
          payment: family.payment,
        });
        expect(actionForFocus(position, 'overwrite-option')).toMatchObject({
          type: 'develop-outright',
          cardId: family.rootCardId,
          districtId: valuableDistrictId,
          payment: family.payment,
        });

        const deltas = strategicActionDeltasV0(position.state, 'PlayerA');
        const preserveDelta = delta(position, deltas, 'preserve-option');
        const overwriteDelta = delta(position, deltas, 'overwrite-option');
        const expectedImmediateDelta = {
          districtPointMarginDelta: 2,
          developedRankMarginDelta: family.developedRankMarginDelta,
          resourceMarginDelta: family.resourceMarginDelta,
          targetDistrictScoreMarginDelta: family.developedRankMarginDelta,
          currentOutcomeBefore: 'behind',
          currentOutcomeAfter: 'behind',
          playedCardDestination: 'developed',
        } as const;
        expect(preserveDelta).toMatchObject(expectedImmediateDelta);
        expect(overwriteDelta).toMatchObject(expectedImmediateDelta);

        const preserveAfter = summaryAfterFocus(position, 'preserve-option');
        const overwriteAfter = summaryAfterFocus(position, 'overwrite-option');
        expect(preserveAfter.players).toEqual(overwriteAfter.players);
        expect(preserveAfter.cards).toEqual(overwriteAfter.cards);
        expect(preserveAfter.score).toEqual(overwriteAfter.score);
        expect(developedCardMultiset(preserveAfter)).toEqual(
          developedCardMultiset(overwriteAfter)
        );

        const targetSupportField =
          family.family === 'known-hand'
            ? 'ownHandForSelf'
            : 'unknownPoolForSelf';
        expect(
          supportOccurrences(
            preserveAfter,
            family.targetCardId,
            targetSupportField,
            [valuableDistrictId, alternativeDistrictId]
          )
        ).toBe(1);
        expect(
          supportOccurrences(
            overwriteAfter,
            family.targetCardId,
            targetSupportField,
            [valuableDistrictId, alternativeDistrictId]
          )
        ).toBe(0);
        expect(
          supportByDistrict(
            preserveAfter,
            family.targetCardId,
            targetSupportField,
            valuableDistrictId
          )
        ).toBe(true);
        expect(
          supportByDistrict(
            preserveAfter,
            family.targetCardId,
            targetSupportField,
            alternativeDistrictId
          )
        ).toBe(false);
        expect(
          [valuableDistrictId, alternativeDistrictId].map((districtId) =>
            supportByDistrict(
              overwriteAfter,
              family.targetCardId,
              targetSupportField,
              districtId
            )
          )
        ).toEqual([false, false]);

        for (const cardId of family.stableOwnHandCardIds) {
          const preserveSupport = [
            valuableDistrictId,
            alternativeDistrictId,
          ].map((districtId) =>
            supportByDistrict(
              preserveAfter,
              cardId,
              'ownHandForSelf',
              districtId
            )
          );
          const overwriteSupport = [
            valuableDistrictId,
            alternativeDistrictId,
          ].map((districtId) =>
            supportByDistrict(
              overwriteAfter,
              cardId,
              'ownHandForSelf',
              districtId
            )
          );
          if (family.family === 'unknown-pool') {
            expect(preserveSupport).toEqual([true, false]);
            expect(overwriteSupport).toEqual([false, true]);
          } else {
            expect(preserveSupport).toEqual(overwriteSupport);
          }
        }
        for (const cardId of family.stableUnknownCardIds) {
          for (const field of [
            'unknownPoolForSelf',
            'unknownPoolForOpponent',
          ] as const) {
            expect(
              supportOccurrences(preserveAfter, cardId, field, [
                valuableDistrictId,
                alternativeDistrictId,
              ])
            ).toBe(
              supportOccurrences(overwriteAfter, cardId, field, [
                valuableDistrictId,
                alternativeDistrictId,
              ])
            );
            expect(
              [valuableDistrictId, alternativeDistrictId].map((districtId) =>
                supportByDistrict(preserveAfter, cardId, field, districtId)
              )
            ).toEqual(
              [valuableDistrictId, alternativeDistrictId].map((districtId) =>
                supportByDistrict(overwriteAfter, cardId, field, districtId)
              )
            );
            for (const after of [preserveAfter, overwriteAfter]) {
              const laneSupport = [
                valuableDistrictId,
                alternativeDistrictId,
              ].map((districtId) =>
                supportByDistrict(after, cardId, field, districtId)
              );
              expect(laneSupport[0]).toBe(laneSupport[1]);
            }
          }
        }
        if (family.family === 'unknown-pool') {
          expect(
            supportOccurrences(
              preserveAfter,
              family.targetCardId,
              'unknownPoolForOpponent',
              [valuableDistrictId, alternativeDistrictId]
            )
          ).toBe(2);
          expect(
            supportOccurrences(
              overwriteAfter,
              family.targetCardId,
              'unknownPoolForOpponent',
              [valuableDistrictId, alternativeDistrictId]
            )
          ).toBe(2);
          for (const after of [preserveAfter, overwriteAfter]) {
            expect(
              [valuableDistrictId, alternativeDistrictId].map((districtId) =>
                supportByDistrict(
                  after,
                  family.targetCardId,
                  'unknownPoolForOpponent',
                  districtId
                )
              )
            ).toEqual([true, true]);
          }
        }
      }

      expect(
        normalizeDistrictPermutation(
          summaryAfterFocus(original, 'preserve-option')
        )
      ).toEqual(
        normalizeDistrictPermutation(
          summaryAfterFocus(mirror, 'preserve-option')
        )
      );
      expect(
        normalizeDistrictPermutation(
          summaryAfterFocus(original, 'overwrite-option')
        )
      ).toEqual(
        normalizeDistrictPermutation(
          summaryAfterFocus(mirror, 'overwrite-option')
        )
      );
    }
  });

  it('keeps each holdout continuation executable before terminal scoring', () => {
    for (const mirrored of [false, true]) {
      const role = mirrored ? 'mirror' : 'original';

      for (const focusActionId of ['preserve-option', 'overwrite-option']) {
        const known = requiredPosition(
          `known-hand-optionality-holdout-${role}`
        );
        const knownValuableDistrictId =
          known.optionalityTrace?.valuableDistrictId;
        if (!knownValuableDistrictId) {
          throw new Error(`Missing optionality metadata for ${known.id}.`);
        }
        let knownState = stateAfterFocusTurn(known, focusActionId);
        knownState = playSaleTurn(knownState);
        knownState = playSaleTurn(knownState, '1');
        knownState = playSaleTurn(knownState);
        expect(decisionPlayerIdForState(knownState)).toBe('PlayerA');
        expect(knownState.phase).toBe('ActionWindow');
        expect(knownState.finalTurnsRemaining).toBe(2);
        expect(knownState.players[0].resources.Waves).toBeGreaterThanOrEqual(1);
        expect(knownState.players[0].resources.Leaves).toBeGreaterThanOrEqual(
          1
        );
        const originDevelopments = targetDevelopments(knownState, '8');
        if (focusActionId === 'preserve-option') {
          expect(originDevelopments.length).toBeGreaterThan(0);
          expect(
            new Set(originDevelopments.map((action) => action.districtId))
          ).toEqual(new Set([knownValuableDistrictId]));
          expect(
            summaryAfterAction(knownState, originDevelopments[0]).score
          ).toMatchObject({
            districts: { self: 3, opponent: 1 },
            currentLexicographicOutcome: 'ahead',
          });
        } else {
          expect(originDevelopments).toHaveLength(0);
        }

        const unknown = requiredPosition(
          `unknown-pool-optionality-holdout-${role}`
        );
        const unknownValuableDistrictId =
          unknown.optionalityTrace?.valuableDistrictId;
        if (!unknownValuableDistrictId) {
          throw new Error(`Missing optionality metadata for ${unknown.id}.`);
        }
        let unknownState = stateAfterFocusTurn(unknown, focusActionId);
        expect(unknownState.players[0].hand).toContain('19');
        unknownState = playSaleTurn(unknownState);
        expect(decisionPlayerIdForState(unknownState)).toBe('PlayerA');
        expect(unknownState.phase).toBe('ActionWindow');
        expect(unknownState.finalTurnsRemaining).toBe(2);
        expect(unknownState.players[0].resources.Leaves).toBe(5);
        expect(unknownState.players[0].resources.Knots).toBe(5);
        const marketDevelopments = targetDevelopments(unknownState, '19');
        if (focusActionId === 'preserve-option') {
          expect(marketDevelopments.length).toBeGreaterThan(0);
          expect(
            new Set(marketDevelopments.map((action) => action.districtId))
          ).toEqual(new Set([unknownValuableDistrictId]));
          expect(
            summaryAfterAction(unknownState, marketDevelopments[0]).score
          ).toMatchObject({
            districts: { self: 3, opponent: 1 },
            currentLexicographicOutcome: 'ahead',
          });
        } else {
          expect(marketDevelopments).toHaveLength(0);
        }
      }
    }
  });

  it('keeps each optionality continuation executable before terminal scoring', () => {
    for (const mirrored of [false, true]) {
      const role = mirrored ? 'mirror' : 'original';
      const valuableDistrictId = mirrored ? 'D4' : 'D1';

      for (const focusActionId of ['preserve-option', 'overwrite-option']) {
        const known = requiredPosition(`known-hand-optionality-${role}`);
        let knownState = stateAfterFocusTurn(known, focusActionId);
        knownState = playSaleTurn(knownState);
        knownState = playSaleTurn(knownState, '0');
        knownState = playSaleTurn(knownState);
        expect(decisionPlayerIdForState(knownState)).toBe('PlayerA');
        expect(knownState.phase).toBe('ActionWindow');
        expect(knownState.finalTurnsRemaining).toBe(2);
        expect(knownState.players[0].resources.Moons).toBeGreaterThanOrEqual(1);
        expect(knownState.players[0].resources.Knots).toBeGreaterThanOrEqual(1);

        const authorDevelopments = legalActionsForDecisionPlayer(
          knownState,
          'PlayerA'
        ).filter(
          (
            action
          ): action is Extract<GameAction, { type: 'develop-outright' }> =>
            action.type === 'develop-outright' && action.cardId === '6'
        );
        if (focusActionId === 'preserve-option') {
          expect(authorDevelopments.length).toBeGreaterThan(0);
          expect(
            new Set(authorDevelopments.map((action) => action.districtId))
          ).toEqual(new Set([valuableDistrictId]));
          const afterAuthor = strategicStateSummaryV0(
            applyKnownLegalAction(knownState, authorDevelopments[0], {
              recordLog: false,
            }),
            'PlayerA'
          );
          expect(afterAuthor.score.districts).toMatchObject({
            self: 2,
            opponent: 1,
          });
          expect(afterAuthor.score.currentLexicographicOutcome).toBe('ahead');
        } else {
          expect(authorDevelopments).toHaveLength(0);
        }

        const unknown = requiredPosition(`unknown-pool-optionality-${role}`);
        let unknownState = stateAfterFocusTurn(unknown, focusActionId);
        expect(unknownState.players[0].hand).toContain('20');
        unknownState = playSaleTurn(unknownState);
        expect(decisionPlayerIdForState(unknownState)).toBe('PlayerA');
        expect(unknownState.phase).toBe('ActionWindow');
        expect(unknownState.finalTurnsRemaining).toBe(2);
        expect(unknownState.players[0].resources.Suns).toBeGreaterThanOrEqual(
          1
        );
        expect(unknownState.players[0].resources.Wyrms).toBeGreaterThanOrEqual(
          1
        );

        const penitentDevelopments = legalActionsForDecisionPlayer(
          unknownState,
          'PlayerA'
        ).filter(
          (
            action
          ): action is Extract<GameAction, { type: 'develop-outright' }> =>
            action.type === 'develop-outright' && action.cardId === '20'
        );
        if (focusActionId === 'preserve-option') {
          expect(penitentDevelopments.length).toBeGreaterThan(0);
          expect(
            new Set(penitentDevelopments.map((action) => action.districtId))
          ).toEqual(new Set([valuableDistrictId]));
          const afterPenitent = strategicStateSummaryV0(
            applyKnownLegalAction(unknownState, penitentDevelopments[0], {
              recordLog: false,
            }),
            'PlayerA'
          );
          expect(afterPenitent.score.districts).toMatchObject({
            self: 3,
            opponent: 1,
          });
          expect(afterPenitent.score.currentLexicographicOutcome).toBe('ahead');
        } else {
          expect(penitentDevelopments).toHaveLength(0);
        }
      }
    }
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

function summary(position: ReturnType<typeof requiredPosition>) {
  return strategicStateSummaryV0(position.state, position.perspectivePlayerId);
}

function normalizeDistrictPermutation(
  value: ReturnType<typeof strategicStateSummaryV0>
) {
  const districts = value.districts
    .map((district) =>
      structuredClone({
        markerSuitMask: district.markerSuitMask,
        score: district.score,
        self: district.self,
        opponent: district.opponent,
        placementSupport: district.placementSupport,
      })
    )
    .sort((left, right) =>
      JSON.stringify(left).localeCompare(JSON.stringify(right))
    );
  return { ...value, districts };
}

function actionForFocus(
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
  return action;
}

function stateAfterFocusTurn(
  position: ReturnType<typeof requiredPosition>,
  focusActionId: string
): GameState {
  const afterFocus = stepKnownLegalActionToDecision(
    position.state,
    actionForFocus(position, focusActionId)
  );
  return finishPlayedCardTurn(afterFocus);
}

function playSaleTurn(state: GameState, cardId?: CardId): GameState {
  const sale = requiredLegalAction(
    state,
    (action) =>
      action.type === 'sell-card' &&
      (cardId === undefined || action.cardId === cardId)
  );
  return finishPlayedCardTurn(stepKnownLegalActionToDecision(state, sale));
}

function finishPlayedCardTurn(state: GameState): GameState {
  expect(state.phase).toBe('ActionWindow');
  expect(state.cardPlayedThisTurn).toBe(true);
  return stepKnownLegalActionToDecision(
    state,
    requiredLegalAction(state, (action) => action.type === 'end-turn')
  );
}

function requiredLegalAction(
  state: GameState,
  predicate: (action: GameAction) => boolean
): GameAction {
  const playerId = decisionPlayerIdForState(state);
  if (!playerId) {
    throw new Error(`State ${state.seed} has no decision player.`);
  }
  const action = legalActionsForDecisionPlayer(state, playerId).find(predicate);
  if (!action) {
    throw new Error(`Missing required action in state ${state.seed}.`);
  }
  return action;
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

function developedCardMultiset(
  summary: ReturnType<typeof strategicStateSummaryV0>
) {
  return summary.districts
    .flatMap((entry) => [
      ...entry.self.developedCardIds,
      ...entry.opponent.developedCardIds,
    ])
    .sort((left, right) => Number(left) - Number(right));
}

function supportOccurrences(
  summary: ReturnType<typeof strategicStateSummaryV0>,
  cardId: CardId,
  field: 'ownHandForSelf' | 'unknownPoolForSelf' | 'unknownPoolForOpponent',
  districtIds: readonly string[] = ['D1', 'D4']
): number {
  return summary.districts
    .filter((entry) => districtIds.includes(entry.districtId))
    .filter((entry) => entry.placementSupport[field].includes(cardId)).length;
}

function supportByDistrict(
  summary: ReturnType<typeof strategicStateSummaryV0>,
  cardId: CardId,
  field: 'ownHandForSelf' | 'unknownPoolForSelf' | 'unknownPoolForOpponent',
  districtId: string
): boolean {
  return district(summary, districtId).placementSupport[field].includes(cardId);
}

function targetDevelopments(state: GameState, cardId: CardId) {
  return legalActionsForDecisionPlayer(state, 'PlayerA').filter(
    (action): action is Extract<GameAction, { type: 'develop-outright' }> =>
      action.type === 'develop-outright' && action.cardId === cardId
  );
}

function summaryAfterAction(state: GameState, action: GameAction) {
  return strategicStateSummaryV0(
    applyKnownLegalAction(state, action, { recordLog: false }),
    'PlayerA'
  );
}

function requiredProperty(cardId: CardId) {
  const card = PROPERTY_CARDS.find((candidate) => candidate.id === cardId);
  if (!card) {
    throw new Error(`Missing property ${cardId}.`);
  }
  return card;
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
