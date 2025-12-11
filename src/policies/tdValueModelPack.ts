import { ENCODING_VERSION, OBSERVATION_DIM } from './trainingEncoding';

export const TD_VALUE_MODEL_PACK_INDEX_SCHEMA_VERSION = 1;
export const TD_VALUE_MODEL_PACK_MANIFEST_SCHEMA_VERSION = 1;
export const TD_VALUE_MODEL_PACK_WEIGHTS_SCHEMA_VERSION = 1;
export const TD_VALUE_MODEL_TYPE = 'td-value-v1';
export const TD_VALUE_CHECKPOINT_TYPE = 'magnate_td_value_v1';

export const TD_VALUE_REQUIRED_TENSOR_KEYS = [
  'encoder.0.weight',
  'encoder.0.bias',
  'encoder.2.weight',
  'encoder.2.bias',
  'encoder.4.weight',
  'encoder.4.bias',
] as const;

type TdValueTensorKey = (typeof TD_VALUE_REQUIRED_TENSOR_KEYS)[number];

export interface TdValueModelPackIndexEntry {
  id: string;
  label: string;
  modelType: string;
  manifestPath: string;
  createdAtUtc: string;
  sourceRunId?: string | null;
  sourceValueCheckpoint?: string | null;
}

export interface TdValueModelPackIndex {
  schemaVersion: number;
  generatedAtUtc: string;
  defaultPackId: string | null;
  packs: TdValueModelPackIndexEntry[];
}

export interface TdValueModelPackManifest {
  schemaVersion: number;
  packId: string;
  label: string;
  createdAtUtc: string;
  model: {
    modelType: string;
    checkpointType: string;
    encodingVersion: number;
    observationDim: number;
    hiddenDim: number;
    weightsPath: string;
    requiredStateDictKeys: string[];
  };
  source: {
    runId?: string | null;
    valueCheckpoint?: string | null;
    checkpointMetadata?: Record<string, unknown>;
  };
}

export interface TdValueTensorPayload {
  shape: number[];
  values: number[];
}

export interface TdValueWeightsFile {
  schemaVersion: number;
  tensors: Record<string, TdValueTensorPayload>;
}

export interface TdValueScorer {
  readonly observationDim: number;
  predict(observation: readonly number[]): number;
}

export interface LoadedTdValueModel {
  manifest: TdValueModelPackManifest;
  scorer: TdValueScorer;
}

export interface TdValueNetworkTensors {
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
  b2: Float32Array;
  w3: Float32Array;
  b3: Float32Array;
  observationDim: number;
  hiddenDim: number;
}

export class TdValueNetwork implements TdValueScorer {
  readonly observationDim: number;
  readonly hiddenDim: number;

  private readonly w1: Float32Array;
  private readonly b1: Float32Array;
  private readonly w2: Float32Array;
  private readonly b2: Float32Array;
  private readonly w3: Float32Array;
  private readonly b3: Float32Array;

  constructor(tensors: TdValueNetworkTensors) {
    this.observationDim = tensors.observationDim;
    this.hiddenDim = tensors.hiddenDim;
    this.w1 = tensors.w1;
    this.b1 = tensors.b1;
    this.w2 = tensors.w2;
    this.b2 = tensors.b2;
    this.w3 = tensors.w3;
    this.b3 = tensors.b3;
  }

  predict(observation: readonly number[]): number {
    if (observation.length !== this.observationDim) {
      throw new Error(
        `TD value network observation length mismatch. expected=${String(this.observationDim)} actual=${String(observation.length)}.`
      );
    }

    const hidden1 = new Float32Array(this.hiddenDim);
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.b1[row] ?? 0;
      const rowOffset = row * this.observationDim;
      for (let col = 0; col < this.observationDim; col += 1) {
        sum += (this.w1[rowOffset + col] ?? 0) * observation[col];
      }
      hidden1[row] = Math.tanh(sum);
    }

    const hidden2 = new Float32Array(this.hiddenDim);
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.b2[row] ?? 0;
      const rowOffset = row * this.hiddenDim;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        sum += (this.w2[rowOffset + col] ?? 0) * hidden1[col];
      }
      hidden2[row] = Math.tanh(sum);
    }

    let output = this.b3[0] ?? 0;
    for (let index = 0; index < this.hiddenDim; index += 1) {
      output += (this.w3[index] ?? 0) * hidden2[index];
    }
    return output;
  }
}

export function resolvePublicAssetUrl(relativePath: string): string {
  if (/^[a-zA-Z]+:\/\//.test(relativePath)) {
    return relativePath;
  }
  const normalizedPath = relativePath.replace(/^\/+/, '');
  const base = import.meta.env.BASE_URL ?? '/';
  const normalizedBase = base.endsWith('/') ? base : `${base}/`;
  const rootedPath = `${normalizedBase}${normalizedPath}`;
  return toAbsoluteUrl(rootedPath);
}

export async function loadTdValueModelFromIndexUrl(
  indexUrl: string
): Promise<LoadedTdValueModel> {
  const absoluteIndexUrl = toAbsoluteUrl(indexUrl);
  const indexPayload = await fetchJson(absoluteIndexUrl);
  const index = parseModelPackIndex(indexPayload);

  if (index.packs.length === 0) {
    throw new Error('TD value model-pack index does not include any packs.');
  }

  let selected = index.packs[0];
  if (typeof index.defaultPackId === 'string' && index.defaultPackId.length > 0) {
    const match = index.packs.find((entry) => entry.id === index.defaultPackId);
    if (!match) {
      throw new Error(
        `TD value model-pack index defaultPackId was not found in packs: ${index.defaultPackId}`
      );
    }
    selected = match;
  }

  const manifestUrl = resolveManifestUrl(absoluteIndexUrl, selected.manifestPath);
  return loadTdValueModelFromManifestUrl(manifestUrl);
}

export async function loadTdValueModelFromManifestUrl(
  manifestUrl: string
): Promise<LoadedTdValueModel> {
  const absoluteManifestUrl = toAbsoluteUrl(manifestUrl);
  const manifestPayload = await fetchJson(absoluteManifestUrl);
  const manifest = parseModelPackManifest(manifestPayload);
  const weightsUrl = new URL(
    manifest.model.weightsPath,
    absoluteManifestUrl
  ).toString();
  const weightsPayload = await fetchJson(weightsUrl);
  const scorer = createTdValueNetworkFromWeights(
    manifest,
    parseWeightsFile(weightsPayload)
  );
  return {
    manifest,
    scorer,
  };
}

export function createTdValueNetworkFromWeights(
  manifest: TdValueModelPackManifest,
  weightsFile: TdValueWeightsFile
): TdValueNetwork {
  const hiddenDim = manifest.model.hiddenDim;
  const observationDim = manifest.model.observationDim;
  const tensors = weightsFile.tensors;

  const w1 = parseTensor(tensors, 'encoder.0.weight', [hiddenDim, observationDim]);
  const b1 = parseTensor(tensors, 'encoder.0.bias', [hiddenDim]);
  const w2 = parseTensor(tensors, 'encoder.2.weight', [hiddenDim, hiddenDim]);
  const b2 = parseTensor(tensors, 'encoder.2.bias', [hiddenDim]);
  const w3 = parseTensor(tensors, 'encoder.4.weight', [1, hiddenDim]);
  const b3 = parseTensor(tensors, 'encoder.4.bias', [1]);

  return new TdValueNetwork({
    w1,
    b1,
    w2,
    b2,
    w3,
    b3,
    observationDim,
    hiddenDim,
  });
}

export function parseModelPackIndex(value: unknown): TdValueModelPackIndex {
  const source = requiredRecord(value, 'TD value model-pack index');
  const schemaVersion = requiredInteger(source.schemaVersion, 'index.schemaVersion');
  if (schemaVersion !== TD_VALUE_MODEL_PACK_INDEX_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported TD value model-pack index schemaVersion=${String(schemaVersion)}.`
    );
  }

  const generatedAtUtc = requiredString(source.generatedAtUtc, 'index.generatedAtUtc');
  const defaultPackId = optionalStringOrNull(source.defaultPackId, 'index.defaultPackId');
  const packsRaw = source.packs;
  if (!Array.isArray(packsRaw)) {
    throw new Error('TD value model-pack index packs must be an array.');
  }
  const packs = packsRaw.map((entry, index): TdValueModelPackIndexEntry => {
    const row = requiredRecord(entry, `index.packs[${String(index)}]`);
    return {
      id: requiredString(row.id, `index.packs[${String(index)}].id`),
      label: requiredString(row.label, `index.packs[${String(index)}].label`),
      modelType: requiredString(
        row.modelType,
        `index.packs[${String(index)}].modelType`
      ),
      manifestPath: requiredString(
        row.manifestPath,
        `index.packs[${String(index)}].manifestPath`
      ),
      createdAtUtc: requiredString(
        row.createdAtUtc,
        `index.packs[${String(index)}].createdAtUtc`
      ),
      sourceRunId: optionalStringOrNull(
        row.sourceRunId,
        `index.packs[${String(index)}].sourceRunId`
      ),
      sourceValueCheckpoint: optionalStringOrNull(
        row.sourceValueCheckpoint,
        `index.packs[${String(index)}].sourceValueCheckpoint`
      ),
    };
  });
  return {
    schemaVersion,
    generatedAtUtc,
    defaultPackId,
    packs,
  };
}

export function parseModelPackManifest(value: unknown): TdValueModelPackManifest {
  const source = requiredRecord(value, 'TD value model-pack manifest');
  const schemaVersion = requiredInteger(source.schemaVersion, 'manifest.schemaVersion');
  if (schemaVersion !== TD_VALUE_MODEL_PACK_MANIFEST_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported TD value model-pack manifest schemaVersion=${String(schemaVersion)}.`
    );
  }

  const modelRaw = requiredRecord(source.model, 'manifest.model');
  const modelType = requiredString(modelRaw.modelType, 'manifest.model.modelType');
  if (modelType !== TD_VALUE_MODEL_TYPE) {
    throw new Error(`Unsupported modelType in manifest: ${modelType}`);
  }
  const checkpointType = requiredString(
    modelRaw.checkpointType,
    'manifest.model.checkpointType'
  );
  if (checkpointType !== TD_VALUE_CHECKPOINT_TYPE) {
    throw new Error(`Unsupported checkpointType in manifest: ${checkpointType}`);
  }
  const encodingVersion = requiredInteger(
    modelRaw.encodingVersion,
    'manifest.model.encodingVersion'
  );
  if (encodingVersion !== ENCODING_VERSION) {
    throw new Error(
      `Encoding version mismatch in manifest. expected=${String(ENCODING_VERSION)} actual=${String(encodingVersion)}.`
    );
  }
  const observationDim = requiredInteger(
    modelRaw.observationDim,
    'manifest.model.observationDim'
  );
  if (observationDim !== OBSERVATION_DIM) {
    throw new Error(
      `Observation dim mismatch in manifest. expected=${String(OBSERVATION_DIM)} actual=${String(observationDim)}.`
    );
  }
  const hiddenDim = requiredInteger(modelRaw.hiddenDim, 'manifest.model.hiddenDim');
  if (hiddenDim <= 0) {
    throw new Error('Manifest hiddenDim must be > 0.');
  }
  const weightsPath = requiredString(modelRaw.weightsPath, 'manifest.model.weightsPath');
  const requiredStateDictKeys = requiredStringArray(
    modelRaw.requiredStateDictKeys,
    'manifest.model.requiredStateDictKeys'
  );
  for (const key of TD_VALUE_REQUIRED_TENSOR_KEYS) {
    if (!requiredStateDictKeys.includes(key)) {
      throw new Error(`Manifest requiredStateDictKeys is missing key: ${key}`);
    }
  }

  const sourceRaw = requiredRecord(source.source, 'manifest.source');
  return {
    schemaVersion,
    packId: requiredString(source.packId, 'manifest.packId'),
    label: requiredString(source.label, 'manifest.label'),
    createdAtUtc: requiredString(source.createdAtUtc, 'manifest.createdAtUtc'),
    model: {
      modelType,
      checkpointType,
      encodingVersion,
      observationDim,
      hiddenDim,
      weightsPath,
      requiredStateDictKeys,
    },
    source: {
      runId: optionalStringOrNull(sourceRaw.runId, 'manifest.source.runId'),
      valueCheckpoint: optionalStringOrNull(
        sourceRaw.valueCheckpoint,
        'manifest.source.valueCheckpoint'
      ),
      checkpointMetadata: optionalRecord(
        sourceRaw.checkpointMetadata,
        'manifest.source.checkpointMetadata'
      ),
    },
  };
}

export function parseWeightsFile(value: unknown): TdValueWeightsFile {
  const source = requiredRecord(value, 'TD value model-pack weights');
  const schemaVersion = requiredInteger(source.schemaVersion, 'weights.schemaVersion');
  if (schemaVersion !== TD_VALUE_MODEL_PACK_WEIGHTS_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported TD value model-pack weights schemaVersion=${String(schemaVersion)}.`
    );
  }

  const tensorsRaw = requiredRecord(source.tensors, 'weights.tensors');
  const tensors: Record<string, TdValueTensorPayload> = {};
  for (const [key, tensorValue] of Object.entries(tensorsRaw)) {
    const tensor = requiredRecord(tensorValue, `weights.tensors.${key}`);
    const shape = requiredIntegerArray(tensor.shape, `weights.tensors.${key}.shape`);
    const values = requiredNumberArray(tensor.values, `weights.tensors.${key}.values`);
    tensors[key] = { shape, values };
  }
  return {
    schemaVersion,
    tensors,
  };
}

function parseTensor(
  tensors: Record<string, TdValueTensorPayload>,
  key: TdValueTensorKey,
  expectedShape: readonly number[]
): Float32Array {
  const payload = tensors[key];
  if (!payload) {
    throw new Error(`TD value model weights are missing tensor ${key}.`);
  }
  if (
    payload.shape.length !== expectedShape.length
    || payload.shape.some((value, index) => value !== expectedShape[index])
  ) {
    throw new Error(
      `Tensor shape mismatch for ${key}. expected=[${expectedShape.join(',')}] actual=[${payload.shape.join(',')}]`
    );
  }
  const expectedLength = expectedShape.reduce(
    (product, current) => product * current,
    1
  );
  if (payload.values.length !== expectedLength) {
    throw new Error(
      `Tensor length mismatch for ${key}. expected=${String(expectedLength)} actual=${String(payload.values.length)}`
    );
  }
  const out = new Float32Array(expectedLength);
  for (let index = 0; index < payload.values.length; index += 1) {
    out[index] = payload.values[index];
  }
  return out;
}

function resolveManifestUrl(indexUrl: string, manifestPath: string): string {
  if (/^[a-zA-Z]+:\/\//.test(manifestPath)) {
    return manifestPath;
  }
  if (manifestPath.startsWith('./') || manifestPath.startsWith('../')) {
    return new URL(manifestPath, indexUrl).toString();
  }
  return resolvePublicAssetUrl(manifestPath);
}

function toAbsoluteUrl(url: string): string {
  if (/^[a-zA-Z]+:\/\//.test(url)) {
    return url;
  }
  const runtimeBase =
    typeof window !== 'undefined' && typeof window.location?.href === 'string'
      ? window.location.href
      : 'http://localhost/';
  return new URL(url, runtimeBase).toString();
}

async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch JSON from ${url}: status=${String(response.status)} ${response.statusText}`
    );
  }
  return response.json();
}

function requiredRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function optionalRecord(
  value: unknown,
  label: string
): Record<string, unknown> | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  return requiredRecord(value, label);
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function optionalStringOrNull(value: unknown, label: string): string | null {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    throw new Error(`${label} must be a string or null.`);
  }
  return value;
}

function requiredInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value)) {
    throw new Error(`${label} must be an integer.`);
  }
  return value as number;
}

function requiredStringArray(value: unknown, label: string): string[] {
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

function requiredIntegerArray(value: unknown, label: string): number[] {
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

function requiredNumberArray(value: unknown, label: string): number[] {
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
