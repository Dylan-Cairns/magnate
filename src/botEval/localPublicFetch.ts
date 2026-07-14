import { readFile } from 'node:fs/promises';
import path from 'node:path';

const LOCAL_PUBLIC_HOST = 'localhost';
const TD_PACK_ID_QUERY_PARAM = 'tdPackId';

let installed = false;

export function installLocalPublicFetch(): void {
  if (installed) {
    return;
  }
  installed = true;
  const fallbackFetch = globalThis.fetch?.bind(globalThis);
  globalThis.fetch = async (input: Parameters<typeof fetch>[0], init) => {
    const url = new URL(String(input));
    if (url.protocol === 'http:' && url.hostname === LOCAL_PUBLIC_HOST) {
      return localPublicJsonResponse(url);
    }
    if (!fallbackFetch) {
      throw new Error(`No fetch implementation is available for ${url.toString()}.`);
    }
    return fallbackFetch(input, init);
  };
}

export function localPublicUrl(publicRelativePath: string): string {
  return `http://${LOCAL_PUBLIC_HOST}/${publicRelativePath.replace(/^\/+/, '')}`;
}

export function tdModelIndexPath(tdPackId: string | undefined): string {
  if (!tdPackId) {
    return 'model-packs/index.json';
  }
  return `model-packs/index.json?${TD_PACK_ID_QUERY_PARAM}=${encodeURIComponent(tdPackId)}`;
}

async function localPublicJsonResponse(url: URL): Promise<Response> {
  const relativePath = url.pathname.replace(/^\/+/, '');
  const payload = JSON.parse(
    await readFile(path.join(process.cwd(), 'public', relativePath), 'utf8')
  ) as unknown;
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    async json() {
      return maybeOverrideDefaultTdPack(payload, url);
    },
  } as Response;
}

function maybeOverrideDefaultTdPack(payload: unknown, url: URL): unknown {
  const selectedPackId = url.searchParams.get(TD_PACK_ID_QUERY_PARAM);
  if (!selectedPackId) {
    return payload;
  }
  return {
    ...(payload as object),
    defaultPackId: selectedPackId,
  };
}
