import { describe, expect, it } from 'vitest';

import {
  preloadStartupAssets,
  type StartupPreloadProgress,
} from './startupPreload';

describe('startupPreload', () => {
  it('preloads images and reports progress', async () => {
    const loadedImages: string[] = [];
    const progressEvents: StartupPreloadProgress[] = [];

    await preloadStartupAssets({
      cardImageUrls: ['image-a.png', 'image-b.png'],
      suitIconUrls: ['suit-a.svg'],
      preloadImage: async (url) => {
        loadedImages.push(url);
      },
      onProgress: (progress) => {
        progressEvents.push(progress);
      },
    });

    expect([...loadedImages].sort()).toEqual([
      'image-a.png',
      'image-b.png',
      'suit-a.svg',
    ]);
    expect(progressEvents).toHaveLength(5);
    expect(progressEvents[0]).toMatchObject({
      completed: 0,
      total: 3,
      percent: 0,
    });
    expect(progressEvents.at(-1)).toMatchObject({
      completed: 3,
      total: 3,
      percent: 100,
      message: 'Assets are ready.',
    });
  });

  it('propagates preload failures', async () => {
    await expect(
      preloadStartupAssets({
        cardImageUrls: ['broken.png'],
        suitIconUrls: [],
        preloadImage: async () => {
          throw new Error('image failed');
        },
      })
    ).rejects.toThrow('image failed');
  });
});
