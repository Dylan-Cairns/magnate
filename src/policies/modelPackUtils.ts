export interface ModelPackIndexEntryBase {
  id: string;
  modelType: string;
  manifestPath: string;
}

export interface TensorPayload {
  shape: number[];
  values: number[];
}

export function resolvePublicAssetUrl(relativePath: string): string {
  if (/^[a-zA-Z]+:\/\//.test(relativePath)) {
    return relativePath;
  }
  const normalizedPath = relativePath.replace(/^\/+/, '');
  const workerBase = runtimeWorkerAppBaseHref();
  if (workerBase) {
    return new URL(normalizedPath, workerBase).toString();
  }
  const base = import.meta.env.BASE_URL ?? '/';
  const normalizedBase = base.endsWith('/') ? base : `${base}/`;
  const rootedPath = `${normalizedBase}${normalizedPath}`;
  return toAbsoluteUrl(rootedPath);
}

export function resolveManifestUrl(
  indexUrl: string,
  manifestPath: string
): string {
  if (/^[a-zA-Z]+:\/\//.test(manifestPath)) {
    return manifestPath;
  }
  if (manifestPath.startsWith('./') || manifestPath.startsWith('../')) {
    return new URL(manifestPath, indexUrl).toString();
  }
  return resolvePublicAssetUrl(manifestPath);
}

export function toAbsoluteUrl(url: string): string {
  if (/^[a-zA-Z]+:\/\//.test(url)) {
    return url;
  }
  const runtimeBase = runtimeBaseHrefForUrl(url) ?? 'http://localhost/';
  return new URL(url, runtimeBase).toString();
}

export async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch JSON from ${url}: status=${String(response.status)} ${response.statusText}`
    );
  }
  return response.json();
}

export function selectPack<TEntry extends ModelPackIndexEntryBase>(
  packs: readonly TEntry[],
  defaultPackId: string | null
): TEntry {
  if (packs.length === 0) {
    throw new Error('selectPack requires at least one pack.');
  }
  if (typeof defaultPackId === 'string' && defaultPackId.length > 0) {
    const match = packs.find((entry) => entry.id === defaultPackId);
    if (match) {
      return match;
    }
  }
  return packs[0];
}

export function requiredTensorRecord(
  value: unknown,
  label: string
): Record<string, TensorPayload> {
  const raw = requiredRecord(value, label);
  const out: Record<string, TensorPayload> = {};
  for (const [key, entry] of Object.entries(raw)) {
    const row = requiredRecord(entry, `${label}.${key}`);
    out[key] = {
      shape: requiredIntegerArray(row.shape, `${label}.${key}.shape`),
      values: requiredNumberArray(row.values, `${label}.${key}.values`),
    };
  }
  return out;
}

export function parseTensor(
  tensors: Record<string, TensorPayload>,
  key: string,
  expectedShape: readonly number[],
  errorPrefix: string
): Float32Array {
  const payload = tensors[key];
  if (!payload) {
    throw new Error(`${errorPrefix} weights are missing tensor ${key}.`);
  }
  if (
    payload.shape.length !== expectedShape.length ||
    payload.shape.some((value, index) => value !== expectedShape[index])
  ) {
    throw new Error(
      `${errorPrefix} tensor shape mismatch for ${key}. expected=[${expectedShape.join(',')}] actual=[${payload.shape.join(',')}]`
    );
  }
  const expectedLength = expectedShape.reduce(
    (product, current) => product * current,
    1
  );
  if (payload.values.length !== expectedLength) {
    throw new Error(
      `${errorPrefix} tensor length mismatch for ${key}. expected=${String(expectedLength)} actual=${String(payload.values.length)}`
    );
  }
  const out = new Float32Array(expectedLength);
  for (let index = 0; index < payload.values.length; index += 1) {
    out[index] = payload.values[index];
  }
  return out;
}

export function requiredRecord(
  value: unknown,
  label: string
): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

export function optionalRecord(
  value: unknown,
  label: string
): Record<string, unknown> | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  return requiredRecord(value, label);
}

export function requiredString(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

export function optionalStringOrNull(
  value: unknown,
  label: string
): string | null {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    throw new Error(`${label} must be a string or null.`);
  }
  return value;
}

export function requiredInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value)) {
    throw new Error(`${label} must be an integer.`);
  }
  return value as number;
}

export function requiredStringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array of strings.`);
  }
  const out: string[] = [];
  for (const [index, entry] of value.entries()) {
    if (typeof entry !== 'string' || entry.length === 0) {
      throw new Error(`${label}[${String(index)}] must be a non-empty string.`);
    }
    out.push(entry);
  }
  return out;
}

export function requiredIntegerArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array of integers.`);
  }
  const out: number[] = [];
  for (const [index, entry] of value.entries()) {
    if (!Number.isInteger(entry) || entry <= 0) {
      throw new Error(`${label}[${String(index)}] must be a positive integer.`);
    }
    out.push(entry);
  }
  return out;
}

export function requiredNumberArray(value: unknown, label: string): number[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array of numbers.`);
  }
  const out: number[] = [];
  for (const [index, entry] of value.entries()) {
    if (typeof entry !== 'number' || !Number.isFinite(entry)) {
      throw new Error(`${label}[${String(index)}] must be a finite number.`);
    }
    out.push(entry);
  }
  return out;
}

function runtimeBaseHrefForUrl(url: string): string | undefined {
  if (
    typeof window !== 'undefined' &&
    typeof window.location?.href === 'string'
  ) {
    return window.location.href;
  }
  const location = (globalThis as { location?: { href?: unknown } }).location;
  if (typeof location?.href !== 'string') {
    return undefined;
  }
  if (url.startsWith('/')) {
    return location.href;
  }
  return appBaseHrefFromWorkerLocation(location.href);
}

function runtimeWorkerAppBaseHref(): string | undefined {
  if (typeof window !== 'undefined') {
    return undefined;
  }
  const location = (globalThis as { location?: { href?: unknown } }).location;
  return typeof location?.href === 'string'
    ? appBaseHrefFromWorkerLocation(location.href)
    : undefined;
}

function appBaseHrefFromWorkerLocation(href: string): string {
  const parsed = new URL(href);
  const assetsIndex = parsed.pathname.lastIndexOf('/assets/');
  if (assetsIndex >= 0) {
    parsed.pathname = parsed.pathname.slice(0, assetsIndex + 1);
    parsed.search = '';
    parsed.hash = '';
    return parsed.toString();
  }
  const sourceIndex = parsed.pathname.indexOf('/src/');
  if (sourceIndex >= 0) {
    parsed.pathname = '/';
    parsed.search = '';
    parsed.hash = '';
    return parsed.toString();
  }
  return href;
}
