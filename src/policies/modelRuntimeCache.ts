import {
  loadTdValueModelFromIndexUrl,
  resolvePublicAssetUrl,
  type TdValueScorer,
} from './tdValueModelPack';
import { loadTdRootModelFromIndexUrl } from './tdRootModelPack';
import type { LoadedTdGuidanceModel } from './tdGuidanceModel';

export const DEFAULT_TD_VALUE_MODEL_INDEX_PATH = 'model-packs/index.json';
export const DEFAULT_TD_ROOT_MODEL_INDEX_PATH = 'model-packs/index.json';

const tdValueModelByIndexUrl = new Map<string, Promise<TdValueScorer>>();
const tdRootModelByIndexUrl = new Map<string, Promise<LoadedTdGuidanceModel>>();

export function preloadTdValueBrowserModel(
  indexPath = DEFAULT_TD_VALUE_MODEL_INDEX_PATH
): Promise<TdValueScorer> {
  const indexUrl = resolvePublicAssetUrl(indexPath);
  return cachePromise(tdValueModelByIndexUrl, indexUrl, async () => {
    const loaded = await loadTdValueModelFromIndexUrl(indexUrl);
    return loaded.scorer;
  });
}

export function preloadTdRootBrowserModel(
  indexPath = DEFAULT_TD_ROOT_MODEL_INDEX_PATH
): Promise<LoadedTdGuidanceModel> {
  const indexUrl = resolvePublicAssetUrl(indexPath);
  return cachePromise(tdRootModelByIndexUrl, indexUrl, () =>
    loadTdRootModelFromIndexUrl(indexUrl)
  );
}

export function clearModelRuntimeCachesForTests(): void {
  tdValueModelByIndexUrl.clear();
  tdRootModelByIndexUrl.clear();
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
