import { describe, expect, it } from 'vitest';

import { CARD_BY_ID, type CardId } from '../engine/cards';
import {
  CARD_BACK_IMAGE,
  CARD_BACK_IMAGE_FILE,
  CARD_IMAGE_BY_ID,
  CARD_IMAGE_FILE_BY_ID,
  getCardImage,
  getCardImageFile,
} from './cardImages';

const EXPECTED_CARD_IMAGE_FILE_BY_ID: Record<CardId, string> = {
  '0': '1_ace_knots.png',
  '1': '1_ace_leaves.png',
  '2': '1_ace_moons.png',
  '3': '1_ace_suns.png',
  '4': '1_ace_waves.png',
  '5': '1_ace_wyrms.png',
  '6': '2_author.png',
  '7': '2_desert.png',
  '8': '2_origin.png',
  '9': '3_journey.png',
  '10': '3_painter.png',
  '11': '3_savage.png',
  '12': '4_battle.png',
  '13': '4_mountain.png',
  '14': '4_sailor.png',
  '15': '5_discovery.png',
  '16': '5_forest.png',
  '17': '5_soldier.png',
  '18': '6_lunactic.png',
  '19': '6_market.png',
  '20': '6_penitent.png',
  '21': '7_castle.png',
  '22': '7_cave.png',
  '23': '7_chance_meeting.png',
  '24': '8_betrayal.png',
  '25': '8_diplomat.png',
  '26': '8_mill.png',
  '27': '9_darkness.png',
  '28': '9_merchant.png',
  '29': '9_pact.png',
  '30': 'crown_knots.png',
  '31': 'crown_leaves.png',
  '32': 'crown_moons.png',
  '33': 'crown_suns.png',
  '34': 'crown_waves.png',
  '35': 'crown_wyrms.png',
  '36': 'excuse.png',
  '37': 'pawn_borderland.png',
  '38': 'pawn_harvest.png',
  '39': 'pawn_light_keeper.png',
  '40': 'pawn_watchman.png',
};

describe('cardImages', () => {
  it('matches the canonical Adaman card-to-image relationships', () => {
    expect(CARD_IMAGE_FILE_BY_ID).toEqual(EXPECTED_CARD_IMAGE_FILE_BY_ID);
  });

  it('covers every engine card id exactly once', () => {
    const mappedIds = Object.keys(CARD_IMAGE_FILE_BY_ID).sort();
    const engineIds = Object.keys(CARD_BY_ID).sort();
    expect(mappedIds).toEqual(engineIds);
  });

  it('resolves every mapped image file to an asset URL', () => {
    for (const cardId of Object.keys(CARD_IMAGE_FILE_BY_ID) as CardId[]) {
      const fileName = CARD_IMAGE_FILE_BY_ID[cardId];
      const imageUrl = CARD_IMAGE_BY_ID[cardId];

      expect(getCardImageFile(cardId)).toBe(fileName);
      expect(getCardImage(cardId)).toBe(imageUrl);
      expect(imageUrl.length).toBeGreaterThan(0);
      expect(imageUrl).toContain(fileName);
    }
  });

  it('resolves the card back image URL', () => {
    expect(CARD_BACK_IMAGE_FILE).toBe('back.png');
    expect(CARD_BACK_IMAGE).toContain(CARD_BACK_IMAGE_FILE);
  });
});
