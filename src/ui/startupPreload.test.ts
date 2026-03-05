import { describe, expect, it } from 'vitest';

import {
  preloadStartupAssets,
  type StartupPreloadProgress,
} from './startupPreload';

describe('startupPreload', () => {
  it('preloads images and models and reports progress', async () => {
    const loadedImages: string[] = [];
    let tdValueLoads = 0;
    let tdSearchLoads = 0;
    const progressEvents: StartupPreloadProgress[] = [];

    await preloadStartupAssets({
      cardImageUrls: ['image-a.png', 'image-b.png'],
      preloadImage: async (url) => {
        loadedImages.push(url);
      },
      preloadTdValueModel: async () => {
        tdValueLoads += 1;
      },
      preloadTdSearchModel: async () => {
        tdSearchLoads += 1;
      },
      onProgress: (progress) => {
        progressEvents.push(progress);
      },
    });

    expect([...loadedImages].sort()).toEqual(['image-a.png', 'image-b.png']);
    expect(tdValueLoads).toBe(1);
    expect(tdSearchLoads).toBe(1);
    expect(progressEvents).toHaveLength(6);
    expect(progressEvents[0]).toMatchObject({
      completed: 0,
      total: 4,
      percent: 0,
    });
    expect(progressEvents.at(-1)).toMatchObject({
      completed: 4,
      total: 4,
      percent: 100,
      message: 'Assets are ready.',
    });
  });

  it('propagates preload failures', async () => {
    await expect(
      preloadStartupAssets({
        cardImageUrls: [],
        preloadImage: async () => undefined,
        preloadTdValueModel: async () => undefined,
        preloadTdSearchModel: async () => {
          throw new Error('search model failed');
        },
      })
    ).rejects.toThrow('search model failed');
  });
});
