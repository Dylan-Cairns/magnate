import {
  loadTdValueModelFromIndexUrl,
  resolvePublicAssetUrl,
  type TdValueScorer,
} from './tdValueModelPack';
import {
  loadTdSearchModelFromIndexUrl,
  type LoadedTdSearchModel,
} from './tdSearchModelPack';

export const DEFAULT_TD_VALUE_MODEL_INDEX_PATH = 'model-packs/index.json';
export const DEFAULT_TD_SEARCH_MODEL_INDEX_PATH = 'model-packs/index.json';

const tdValueModelByIndexUrl = new Map<string, Promise<TdValueScorer>>();
const tdSearchModelByIndexUrl = new Map<string, Promise<LoadedTdSearchModel>>();

export function preloadTdValueBrowserModel(
  indexPath = DEFAULT_TD_VALUE_MODEL_INDEX_PATH
): Promise<TdValueScorer> {
  const indexUrl = resolvePublicAssetUrl(indexPath);
  return cachePromise(tdValueModelByIndexUrl, indexUrl, async () => {
    const loaded = await loadTdValueModelFromIndexUrl(indexUrl);
    return loaded.scorer;
  });
}

export function preloadTdSearchBrowserModel(
  indexPath = DEFAULT_TD_SEARCH_MODEL_INDEX_PATH
): Promise<LoadedTdSearchModel> {
  const indexUrl = resolvePublicAssetUrl(indexPath);
  return cachePromise(tdSearchModelByIndexUrl, indexUrl, () =>
    loadTdSearchModelFromIndexUrl(indexUrl)
  );
}

export function clearModelRuntimeCachesForTests(): void {
  tdValueModelByIndexUrl.clear();
  tdSearchModelByIndexUrl.clear();
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
