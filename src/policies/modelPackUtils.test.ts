import { afterEach, describe, expect, it, vi } from 'vitest';

import { fetchJsonCached, SHARED_JSON_CACHE_NAME } from './modelPackUtils';

class MemoryCache {
  private readonly store = new Map<string, Response>();

  async match(key: string): Promise<Response | undefined> {
    return this.store.get(key)?.clone();
  }

  async put(key: string, response: Response): Promise<void> {
    this.store.set(key, response.clone());
  }

  get size(): number {
    return this.store.size;
  }
}

class MemoryCacheStorage {
  private readonly caches = new Map<string, MemoryCache>();

  async open(name: string): Promise<MemoryCache> {
    let cache = this.caches.get(name);
    if (!cache) {
      cache = new MemoryCache();
      this.caches.set(name, cache);
    }
    return cache;
  }

  cacheNamed(name: string): MemoryCache | undefined {
    return this.caches.get(name);
  }
}

const originalCaches = Object.getOwnPropertyDescriptor(globalThis, 'caches');
const originalFetch = globalThis.fetch;

function installCaches(): MemoryCacheStorage {
  const storage = new MemoryCacheStorage();
  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    value: storage,
  });
  return storage;
}

afterEach(() => {
  if (originalCaches) {
    Object.defineProperty(globalThis, 'caches', originalCaches);
  } else {
    delete (globalThis as { caches?: unknown }).caches;
  }
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe('fetchJsonCached', () => {
  it('downloads once and serves later reads of the same version from cache', async () => {
    const storage = installCaches();
    let calls = 0;
    const fetchSpy = vi.fn(async () => {
      calls += 1;
      return new Response(JSON.stringify({ ok: true, calls }), { status: 200 });
    });
    globalThis.fetch = fetchSpy as unknown as typeof fetch;

    const first = await fetchJsonCached(
      'https://example.test/weights.json',
      'pack@v1'
    );
    const second = await fetchJsonCached(
      'https://example.test/weights.json',
      'pack@v1'
    );

    expect(first).toEqual({ ok: true, calls: 1 });
    expect(second).toEqual({ ok: true, calls: 1 });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(storage.cacheNamed(SHARED_JSON_CACHE_NAME)?.size).toBe(1);
  });

  it('refetches when the version changes so a re-exported pack is not stale', async () => {
    installCaches();
    let calls = 0;
    globalThis.fetch = vi.fn(
      async () => new Response(JSON.stringify({ calls: ++calls }), { status: 200 })
    ) as unknown as typeof fetch;

    const first = await fetchJsonCached(
      'https://example.test/weights.json',
      'pack@v1'
    );
    const second = await fetchJsonCached(
      'https://example.test/weights.json',
      'pack@v2'
    );

    expect(first).toEqual({ calls: 1 });
    expect(second).toEqual({ calls: 2 });
  });

  it('falls back to a plain fetch when Cache Storage is unavailable', async () => {
    delete (globalThis as { caches?: unknown }).caches;
    const fetchSpy = vi.fn(
      async () => new Response(JSON.stringify({ ok: true }), { status: 200 })
    );
    globalThis.fetch = fetchSpy as unknown as typeof fetch;

    await fetchJsonCached('https://example.test/weights.json', 'pack@v1');
    await fetchJsonCached('https://example.test/weights.json', 'pack@v1');

    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it('surfaces fetch failures with the url and status', async () => {
    installCaches();
    globalThis.fetch = vi.fn(
      async () =>
        new Response('nope', { status: 404, statusText: 'Not Found' })
    ) as unknown as typeof fetch;

    await expect(
      fetchJsonCached('https://example.test/weights.json', 'pack@v1')
    ).rejects.toThrow(
      'Failed to fetch JSON from https://example.test/weights.json: status=404 Not Found'
    );
  });
});
