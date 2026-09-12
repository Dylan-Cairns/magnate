import { describe, expect, it, vi } from 'vitest';

import { ALL_CARDS, CARD_BY_ID, type CardId } from '../engine/cards';
import {
  ALL_CARD_IMAGE_URLS,
  CARD_IMAGE_BY_ID,
  CARD_IMAGE_FILE_BY_ID,
  cardImageFileName,
  getCardImage,
  getCardImageFile,
  preloadCardImageUrl,
} from './cardImages';

const ASSET_MODULES = import.meta.glob('../assets/decktet-card-art/*.webp', {
  eager: true,
  import: 'default',
}) as Record<string, string>;

const fileNameFromModuleKey = (key: string): string =>
  key.slice(key.lastIndexOf('/') + 1);

const EXPECTED_PLAYABLE_FILE_NAMES = ALL_CARDS.map((card) =>
  cardImageFileName(card.name)
);

describe('cardImages', () => {
  it('covers every engine card id exactly once', () => {
    const mappedIds = Object.keys(CARD_IMAGE_FILE_BY_ID).sort();
    const engineIds = Object.keys(CARD_BY_ID).sort();
    expect(mappedIds).toEqual(engineIds);
  });

  it('derives filenames from the normalized card-name convention', () => {
    for (const card of ALL_CARDS) {
      const fileName = CARD_IMAGE_FILE_BY_ID[card.id as CardId];
      expect(fileName).toBe(cardImageFileName(card.name));
      expect(fileName).toMatch(/^decktet-card-[a-z0-9]+(?:-[a-z0-9]+)*\.webp$/);
    }
    expect(cardImageFileName('The Chance Meeting')).toBe(
      'decktet-card-the-chance-meeting.webp'
    );
    expect(cardImageFileName('The Light Keeper')).toBe(
      'decktet-card-the-light-keeper.webp'
    );
  });

  it('maps each playable filename to exactly one engine card', () => {
    const fileNames = Object.values(CARD_IMAGE_FILE_BY_ID);
    expect(new Set(fileNames).size).toBe(fileNames.length);
    expect(new Set(EXPECTED_PLAYABLE_FILE_NAMES).size).toBe(
      EXPECTED_PLAYABLE_FILE_NAMES.length
    );
  });

  it('resolves every mapped image file to an asset URL', () => {
    for (const cardId of Object.keys(CARD_IMAGE_FILE_BY_ID)) {
      const fileName = CARD_IMAGE_FILE_BY_ID[cardId];
      const imageUrl = CARD_IMAGE_BY_ID[cardId];

      expect(getCardImageFile(cardId)).toBe(fileName);
      expect(getCardImage(cardId)).toBe(imageUrl);
      expect(imageUrl.length).toBeGreaterThan(0);
      expect(imageUrl).toContain(fileName);
    }
  });

  it('contains exactly the playable card assets', () => {
    const actualFileNames = Object.keys(ASSET_MODULES)
      .map(fileNameFromModuleKey)
      .sort();
    expect(actualFileNames).toEqual([...EXPECTED_PLAYABLE_FILE_NAMES].sort());
    expect(actualFileNames).toHaveLength(ALL_CARDS.length);
  });

  it('maps the four extended-deck Court assets to engine cards', () => {
    const mappedFileNames = new Set(Object.values(CARD_IMAGE_FILE_BY_ID));
    for (const courtId of ['41', '42', '43', '44'] as CardId[]) {
      expect(mappedFileNames.has(CARD_IMAGE_FILE_BY_ID[courtId])).toBe(true);
    }
    expect(getCardImageFile('41')).toBe('decktet-card-the-consul.webp');
    expect(getCardImageFile('44')).toBe('decktet-card-the-window.webp');
  });

  it('exposes preload URL list for all card art', () => {
    const uniqueUrls = new Set(ALL_CARD_IMAGE_URLS);
    expect(uniqueUrls.size).toBe(ALL_CARD_IMAGE_URLS.length);
    expect(uniqueUrls.size).toBe(ALL_CARDS.length);
    for (const imageUrl of Object.values(CARD_IMAGE_BY_ID)) {
      expect(ALL_CARD_IMAGE_URLS).toContain(imageUrl);
    }
  });

  it('resolves representative cards to the expected new filenames', () => {
    expect(getCardImageFile('0')).toBe('decktet-card-ace-of-knots.webp');
    expect(getCardImageFile('6')).toBe('decktet-card-the-author.webp');
    expect(getCardImageFile('30')).toBe('decktet-card-the-windfall.webp');
    expect(getCardImageFile('36')).toBe('decktet-card-the-excuse.webp');
    expect(getCardImageFile('37')).toBe('decktet-card-the-borderland.webp');
  });

  it('rejects a decode failure and permits a subsequent preload attempt', async () => {
    const url = 'test-decode-failure.webp';
    const images: TestImage[] = [];
    vi.stubGlobal(
      'Image',
      class TestImage {
        complete = false;
        naturalWidth = 1;
        naturalHeight = 1;
        onload: (() => void) | null = null;
        onerror: (() => void) | null = null;

        constructor() {
          images.push(this);
        }

        set src(_value: string) {
          queueMicrotask(() => this.onload?.());
        }

        decode(): Promise<void> {
          return Promise.reject(new Error('decode failed'));
        }
      }
    );

    try {
      await expect(preloadCardImageUrl(url)).rejects.toThrow(
        'Failed to preload card image'
      );
      await expect(preloadCardImageUrl(url)).rejects.toThrow(
        'Failed to preload card image'
      );
      expect(images).toHaveLength(2);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

type TestImage = {
  complete: boolean;
  naturalWidth: number;
  naturalHeight: number;
  onload: (() => void) | null;
  onerror: (() => void) | null;
};
