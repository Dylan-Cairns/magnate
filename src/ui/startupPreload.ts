import {
  preloadTdSearchBrowserModel,
  preloadTdValueBrowserModel,
} from '../policies/modelRuntimeCache';
import { ALL_CARD_IMAGE_URLS, preloadCardImageUrl } from './cardImages';

const STARTUP_PRELOAD_LOADING_MESSAGE = 'Loading card images and bot models...';
const STARTUP_PRELOAD_READY_MESSAGE = 'Assets are ready.';

export interface StartupPreloadProgress {
  completed: number;
  total: number;
  percent: number;
  message: string;
}

export interface StartupPreloadOptions {
  onProgress?: (progress: StartupPreloadProgress) => void;
  cardImageUrls?: readonly string[];
  preloadImage?: (url: string) => Promise<void>;
  preloadTdValueModel?: () => Promise<unknown>;
  preloadTdSearchModel?: () => Promise<unknown>;
}

export async function preloadStartupAssets(
  options: StartupPreloadOptions = {}
): Promise<void> {
  const cardImageUrls = options.cardImageUrls ?? ALL_CARD_IMAGE_URLS;
  const preloadImage = options.preloadImage ?? preloadCardImageUrl;
  const preloadTdValueModel =
    options.preloadTdValueModel ?? preloadTdValueBrowserModel;
  const preloadTdSearchModel =
    options.preloadTdSearchModel ?? preloadTdSearchBrowserModel;

  const tasks: Array<() => Promise<unknown>> = [
    ...cardImageUrls.map((url) => () => preloadImage(url)),
    preloadTdValueModel,
    preloadTdSearchModel,
  ];

  const total = tasks.length;
  let completed = 0;
  emitProgress(options.onProgress, {
    completed,
    total,
    percent: progressPercent(completed, total),
    message: STARTUP_PRELOAD_LOADING_MESSAGE,
  });

  await Promise.all(
    tasks.map(async (task) => {
      await task();
      completed += 1;
      emitProgress(options.onProgress, {
        completed,
        total,
        percent: progressPercent(completed, total),
        message: STARTUP_PRELOAD_LOADING_MESSAGE,
      });
    })
  );

  emitProgress(options.onProgress, {
    completed: total,
    total,
    percent: 100,
    message: STARTUP_PRELOAD_READY_MESSAGE,
  });
}

function emitProgress(
  onProgress: StartupPreloadOptions['onProgress'],
  progress: StartupPreloadProgress
): void {
  if (onProgress) {
    onProgress(progress);
  }
}

function progressPercent(completed: number, total: number): number {
  if (total <= 0) {
    return 100;
  }
  return Math.round((completed / total) * 100);
}
