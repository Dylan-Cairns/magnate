import { describe, expect, it } from 'vitest';

import { CARD_BY_ID, CROWN_CARDS, EXCUSE_CARD, PAWN_CARDS, PROPERTY_CARDS } from './cards';
import { drawOne, initialSetup, type DrawContext } from './deck';

function sumResourcePool(pool: Record<string, number>): number {
  return Object.values(pool).reduce((sum, value) => sum + value, 0);
}

function drawSequence(context: DrawContext): string[] {
  const sequence: string[] = [];
  let current = context;
  for (let i = 0; i < 30; i += 1) {
    const result = drawOne(current);
    if (!result.cardId) {
      break;
    }
    sequence.push(result.cardId);
    current = {
      deck: result.deck,
      seed: context.seed,
      rngCursor: result.rngCursor,
      exhaustionStage: result.exhaustionStage,
      finalTurnsRemaining: result.finalTurnsRemaining,
    };
  }
  return sequence;
}

describe('initialSetup', () => {
  it('is deterministic for a fixed seed', () => {
    const first = initialSetup('seed-1');
    const second = initialSetup('seed-1');
    expect(first).toEqual(second);
  });

  it('deals exactly 3 crowns to each player', () => {
    const setup = initialSetup('seed-2');
    expect(setup.crownsByPlayer.PlayerA).toHaveLength(3);
    expect(setup.crownsByPlayer.PlayerB).toHaveLength(3);
  });

  it('deals exactly 3 property cards to each player', () => {
    const setup = initialSetup('seed-3');
    expect(setup.handsByPlayer.PlayerA).toHaveLength(3);
    expect(setup.handsByPlayer.PlayerB).toHaveLength(3);
  });

  it('leaves the correct number of cards in the draw deck after dealing', () => {
    const setup = initialSetup('seed-4');
    expect(setup.deck.draw).toHaveLength(PROPERTY_CARDS.length - 6);
  });

  it('assigns starting resources based on dealt crown suits', () => {
    const setup = initialSetup('seed-5');

    (['PlayerA', 'PlayerB'] as const).forEach((playerId) => {
      const crowns = setup.crownsByPlayer[playerId];
      const resources = setup.startingResourcesByPlayer[playerId];
      expect(sumResourcePool(resources)).toBe(3);

      crowns.forEach((cardId) => {
        const card = CARD_BY_ID[cardId];
        expect(card.kind).toBe('Crown');
        if (card.kind === 'Crown') {
          expect(resources[card.suits[0]]).toBeGreaterThanOrEqual(1);
        }
      });
    });
  });

  it('returns district markers of four pawns plus the excuse', () => {
    const setup = initialSetup('seed-6');
    const expected = [...PAWN_CARDS.map((card) => card.id), EXCUSE_CARD.id].sort();
    expect([...setup.districts].sort()).toEqual(expected);
    expect(setup.districts).toHaveLength(5);
  });
});

describe('drawOne', () => {
  it('throws when exhaustionStage does not match deck reshuffle state', () => {
    expect(() =>
      drawOne({
        deck: { draw: ['6'], discard: [], reshuffles: 0 },
        seed: 'seed-mismatch',
        rngCursor: 0,
        exhaustionStage: 1,
      })
    ).toThrow(/Draw context mismatch/);
  });

  it('draws from the top of draw pile when cards are available', () => {
    const result = drawOne({
      deck: { draw: ['6', '7'], discard: [], reshuffles: 0 },
      seed: 'seed-a',
      rngCursor: 0,
      exhaustionStage: 0,
    });

    expect(result.cardId).toBe('6');
    expect(result.deck.draw).toEqual(['7']);
    expect(result.deck.reshuffles).toBe(0);
  });

  it('reshuffles discard on first exhaustion and returns a card', () => {
    const discardCards = ['6', '7', '8', '9'];
    const result = drawOne({
      deck: { draw: [], discard: discardCards, reshuffles: 0 },
      seed: 'seed-b',
      rngCursor: 3,
      exhaustionStage: 0,
    });

    expect(result.cardId).toBeDefined();
    expect(discardCards).toContain(result.cardId);
    expect(result.deck.discard).toEqual([]);
    expect(result.deck.reshuffles).toBe(1);
    expect(result.exhaustionStage).toBe(1);
    expect(result.rngCursor).toBe(4);
  });

  it('marks second exhaustion and no card draw when no reshuffle is available', () => {
    const result = drawOne({
      deck: { draw: [], discard: [], reshuffles: 1 },
      seed: 'seed-c',
      rngCursor: 10,
      exhaustionStage: 1,
    });

    expect(result.cardId).toBeUndefined();
    expect(result.exhaustionStage).toBe(2);
    expect(result.deck.reshuffles).toBe(2);
  });

  it('sets finalTurnsRemaining to 2 on second exhaustion when absent', () => {
    const result = drawOne({
      deck: { draw: [], discard: [], reshuffles: 1 },
      seed: 'seed-d',
      rngCursor: 2,
      exhaustionStage: 1,
    });
    expect(result.finalTurnsRemaining).toBe(2);
  });

  it('preserves finalTurnsRemaining when already present', () => {
    const result = drawOne({
      deck: { draw: [], discard: [], reshuffles: 1 },
      seed: 'seed-e',
      rngCursor: 2,
      exhaustionStage: 1,
      finalTurnsRemaining: 1,
    });
    expect(result.finalTurnsRemaining).toBe(1);
  });

  it('uses seed + rngCursor for deterministic reshuffle ordering', () => {
    const baseContext: DrawContext = {
      deck: { draw: [], discard: ['6', '7', '8', '9', '10', '11'], reshuffles: 0 },
      seed: 'seed-f',
      rngCursor: 4,
      exhaustionStage: 0,
    };

    const seqA = drawSequence(baseContext);
    const seqB = drawSequence(baseContext);
    const seqC = drawSequence({ ...baseContext, rngCursor: 5 });

    expect(seqA).toEqual(seqB);
    expect(seqA).not.toEqual(seqC);
  });
});

describe('issue regressions', () => {
  it('issue 6: setup now deals crowns/hands/resources instead of returning undealt crowns', () => {
    const setup = initialSetup('seed-reg-6');
    const totalDealtCrowns =
      setup.crownsByPlayer.PlayerA.length + setup.crownsByPlayer.PlayerB.length;
    expect(totalDealtCrowns).toBe(CROWN_CARDS.length);
    expect(setup.handsByPlayer.PlayerA).toHaveLength(3);
    expect(setup.handsByPlayer.PlayerB).toHaveLength(3);
    expect(sumResourcePool(setup.startingResourcesByPlayer.PlayerA)).toBe(3);
    expect(sumResourcePool(setup.startingResourcesByPlayer.PlayerB)).toBe(3);
  });

  it('issue 7: reshuffle path changes when rngCursor changes', () => {
    const base: DrawContext = {
      deck: { draw: [], discard: ['6', '7', '8', '9', '10', '11'], reshuffles: 0 },
      seed: 'seed-reg-7',
      rngCursor: 1,
      exhaustionStage: 0,
    };

    const one = drawSequence(base);
    const two = drawSequence({ ...base, rngCursor: 2 });
    expect(one).not.toEqual(two);
  });

  it('issue 9: second exhaustion emits final-turn state marker', () => {
    const result = drawOne({
      deck: { draw: [], discard: [], reshuffles: 1 },
      seed: 'seed-reg-9',
      rngCursor: 0,
      exhaustionStage: 1,
    });
    expect(result.exhaustionStage).toBe(2);
    expect(result.finalTurnsRemaining).toBe(2);
  });
});
