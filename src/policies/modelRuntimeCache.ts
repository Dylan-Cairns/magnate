import {
  loadTdValueModelFromIndexUrl,
  resolvePublicAssetUrl,
  type TdValueScorer,
} from './tdValueModelPack';

export const DEFAULT_TD_VALUE_MODEL_INDEX_PATH = 'model-packs/index.json';

const tdValueModelByIndexUrl = new Map<string, Promise<TdValueScorer>>();

export function preloadTdValueBrowserModel(
  indexPath = DEFAULT_TD_VALUE_MODEL_INDEX_PATH
): Promise<TdValueScorer> {
  const indexUrl = resolvePublicAssetUrl(indexPath);
  return cachePromise(tdValueModelByIndexUrl, indexUrl, async () => {
    const loaded = await loadTdValueModelFromIndexUrl(indexUrl);
    return loaded.scorer;
  });
}

export function clearModelRuntimeCachesForTests(): void {
  tdValueModelByIndexUrl.clear();
}

function cachePromise<T>(
  cache: Map<string, Promise<T>>,
  key: string,
  loader: () => Promise<T>
): Promise<T> {
  const existing = cache.get(key);
  if (existing) {
    return existing;
  }
  const created = loader();
  cache.set(key, created);
  created.catch(() => {
    if (cache.get(key) === created) {
      cache.delete(key);
    }
  });
  return created;
}
