import { ACTION_FEATURE_DIM, ENCODING_VERSION, OBSERVATION_DIM } from './trainingEncoding';
import {
  TD_VALUE_CHECKPOINT_TYPE,
  TD_VALUE_MODEL_PACK_WEIGHTS_SCHEMA_VERSION,
  TD_VALUE_REQUIRED_TENSOR_KEYS,
  TdValueNetwork,
  parseModelPackIndex,
  resolvePublicAssetUrl,
  type TdValueScorer,
  type TdValueModelPackIndexEntry,
} from './tdValueModelPack';

export const TD_SEARCH_MODEL_TYPE = 'td-search-v1';
export const TD_SEARCH_OPPONENT_CHECKPOINT_TYPE = 'magnate_td_opponent_v1';
export const TD_SEARCH_MODEL_PACK_MANIFEST_SCHEMA_VERSION = 1;

export const TD_SEARCH_OPPONENT_REQUIRED_TENSOR_KEYS = [
  'obs_encoder.0.weight',
  'obs_encoder.0.bias',
  'obs_encoder.2.weight',
  'obs_encoder.2.bias',
  'action_encoder.0.weight',
  'action_encoder.0.bias',
  'policy_head.0.weight',
  'policy_head.0.bias',
  'policy_head.2.weight',
  'policy_head.2.bias',
] as const;

type OpponentTensorKey = (typeof TD_SEARCH_OPPONENT_REQUIRED_TENSOR_KEYS)[number];
type ValueTensorKey = (typeof TD_VALUE_REQUIRED_TENSOR_KEYS)[number];

interface TdSearchTensorPayload {
  shape: number[];
  values: number[];
}

interface TdSearchWeightsFile {
  schemaVersion: number;
  valueTensors: Record<string, TdSearchTensorPayload>;
  opponentTensors: Record<string, TdSearchTensorPayload>;
}

export interface TdSearchModelPackManifest {
  schemaVersion: number;
  packId: string;
  label: string;
  createdAtUtc: string;
  model: {
    modelType: string;
    weightsPath: string;
    value: {
      checkpointType: string;
      encodingVersion: number;
      observationDim: number;
      hiddenDim: number;
      requiredStateDictKeys: string[];
    };
    opponent: {
      checkpointType: string;
      encodingVersion: number;
      observationDim: number;
      actionFeatureDim: number;
      hiddenDim: number;
      requiredStateDictKeys: string[];
    };
  };
  source: {
    runId?: string | null;
    valueCheckpoint?: string | null;
    opponentCheckpoint?: string | null;
    checkpointMetadata?: Record<string, unknown>;
  };
}

export interface TdSearchOpponentScorer {
  readonly observationDim: number;
  readonly actionFeatureDim: number;
  logits(
    observation: readonly number[],
    actionFeatures: readonly number[][]
  ): Float32Array;
}

export interface LoadedTdSearchModel {
  manifest: TdSearchModelPackManifest;
  valueScorer: TdValueScorer;
  opponentScorer: TdSearchOpponentScorer;
}

interface TdSearchOpponentNetworkTensors {
  obsW1: Float32Array;
  obsB1: Float32Array;
  obsW2: Float32Array;
  obsB2: Float32Array;
  actionW: Float32Array;
  actionB: Float32Array;
  headW1: Float32Array;
  headB1: Float32Array;
  headW2: Float32Array;
  headB2: Float32Array;
  observationDim: number;
  actionFeatureDim: number;
  hiddenDim: number;
}

export class TdSearchOpponentNetwork implements TdSearchOpponentScorer {
  readonly observationDim: number;
  readonly actionFeatureDim: number;
  readonly hiddenDim: number;

  private readonly obsW1: Float32Array;
  private readonly obsB1: Float32Array;
  private readonly obsW2: Float32Array;
  private readonly obsB2: Float32Array;
  private readonly actionW: Float32Array;
  private readonly actionB: Float32Array;
  private readonly headW1: Float32Array;
  private readonly headB1: Float32Array;
  private readonly headW2: Float32Array;
  private readonly headB2: Float32Array;

  constructor(tensors: TdSearchOpponentNetworkTensors) {
    this.observationDim = tensors.observationDim;
    this.actionFeatureDim = tensors.actionFeatureDim;
    this.hiddenDim = tensors.hiddenDim;
    this.obsW1 = tensors.obsW1;
    this.obsB1 = tensors.obsB1;
    this.obsW2 = tensors.obsW2;
    this.obsB2 = tensors.obsB2;
    this.actionW = tensors.actionW;
    this.actionB = tensors.actionB;
    this.headW1 = tensors.headW1;
    this.headB1 = tensors.headB1;
    this.headW2 = tensors.headW2;
    this.headB2 = tensors.headB2;
  }

  logits(
    observation: readonly number[],
    actionFeatures: readonly number[][]
  ): Float32Array {
    if (observation.length !== this.observationDim) {
      throw new Error(
        `TD search opponent model observation length mismatch. expected=${String(this.observationDim)} actual=${String(observation.length)}.`
      );
    }
    if (actionFeatures.length === 0) {
      return new Float32Array(0);
    }
    for (let index = 0; index < actionFeatures.length; index += 1) {
      const features = actionFeatures[index];
      if (features.length !== this.actionFeatureDim) {
        throw new Error(
          `TD search opponent model action feature length mismatch at index ${String(index)}. expected=${String(this.actionFeatureDim)} actual=${String(features.length)}.`
        );
      }
    }

    const obsEmbedding = this.encodeObservation(observation);
    const logits = new Float32Array(actionFeatures.length);
    for (let actionIndex = 0; actionIndex < actionFeatures.length; actionIndex += 1) {
      const actionEmbedding = this.encodeAction(actionFeatures[actionIndex]);
      logits[actionIndex] = this.policyLogit(obsEmbedding, actionEmbedding);
    }
    return logits;
  }

  private encodeObservation(observation: readonly number[]): Float32Array {
    const hidden1 = new Float32Array(this.hiddenDim);
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.obsB1[row] ?? 0;
      const rowOffset = row * this.observationDim;
      for (let col = 0; col < this.observationDim; col += 1) {
        sum += (this.obsW1[rowOffset + col] ?? 0) * observation[col];
      }
      hidden1[row] = Math.tanh(sum);
    }

    const hidden2 = new Float32Array(this.hiddenDim);
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.obsB2[row] ?? 0;
      const rowOffset = row * this.hiddenDim;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        sum += (this.obsW2[rowOffset + col] ?? 0) * hidden1[col];
      }
      hidden2[row] = Math.tanh(sum);
    }
    return hidden2;
  }

  private encodeAction(actionFeatures: readonly number[]): Float32Array {
    const embedding = new Float32Array(this.hiddenDim);
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.actionB[row] ?? 0;
      const rowOffset = row * this.actionFeatureDim;
      for (let col = 0; col < this.actionFeatureDim; col += 1) {
        sum += (this.actionW[rowOffset + col] ?? 0) * actionFeatures[col];
      }
      embedding[row] = Math.tanh(sum);
    }
    return embedding;
  }

  private policyLogit(
    observationEmbedding: Float32Array,
    actionEmbedding: Float32Array
  ): number {
    const pair = new Float32Array(this.hiddenDim * 3);
    for (let index = 0; index < this.hiddenDim; index += 1) {
      const obs = observationEmbedding[index] ?? 0;
      const act = actionEmbedding[index] ?? 0;
      pair[index] = obs;
      pair[this.hiddenDim + index] = act;
      pair[(this.hiddenDim * 2) + index] = obs * act;
    }

    const headHidden = new Float32Array(this.hiddenDim);
    const pairWidth = this.hiddenDim * 3;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.headB1[row] ?? 0;
      const rowOffset = row * pairWidth;
      for (let col = 0; col < pairWidth; col += 1) {
        sum += (this.headW1[rowOffset + col] ?? 0) * pair[col];
      }
      headHidden[row] = Math.tanh(sum);
    }

    let output = this.headB2[0] ?? 0;
    for (let col = 0; col < this.hiddenDim; col += 1) {
      output += (this.headW2[col] ?? 0) * headHidden[col];
    }
    return output;
  }
}

export async function loadTdSearchModelFromIndexUrl(
  indexUrl: string
): Promise<LoadedTdSearchModel> {
  const absoluteIndexUrl = toAbsoluteUrl(indexUrl);
  const indexPayload = await fetchJson(absoluteIndexUrl);
  const index = parseModelPackIndex(indexPayload);

  const tdSearchPacks = index.packs.filter(
    (entry) => entry.modelType === TD_SEARCH_MODEL_TYPE
  );
  if (tdSearchPacks.length === 0) {
    throw new Error(
      'TD search model-pack index does not include any td-search-v1 packs.'
    );
  }
  const selected = selectPack(tdSearchPacks, index.defaultPackId);
  const manifestUrl = resolveManifestUrl(absoluteIndexUrl, selected.manifestPath);
  return loadTdSearchModelFromManifestUrl(manifestUrl);
}

export async function loadTdSearchModelFromManifestUrl(
  manifestUrl: string
): Promise<LoadedTdSearchModel> {
  const absoluteManifestUrl = toAbsoluteUrl(manifestUrl);
  const manifestPayload = await fetchJson(absoluteManifestUrl);
  const manifest = parseTdSearchModelPackManifest(manifestPayload);
  const weightsUrl = new URL(
    manifest.model.weightsPath,
    absoluteManifestUrl
  ).toString();
  const weightsPayload = await fetchJson(weightsUrl);
  const weights = parseTdSearchWeightsFile(weightsPayload);
  return {
    manifest,
    valueScorer: createTdSearchValueNetwork(manifest, weights),
    opponentScorer: createTdSearchOpponentNetwork(manifest, weights),
  };
}

export function parseTdSearchModelPackManifest(
  value: unknown
): TdSearchModelPackManifest {
  const source = requiredRecord(value, 'TD search model-pack manifest');
  const schemaVersion = requiredInteger(source.schemaVersion, 'manifest.schemaVersion');
  if (schemaVersion !== TD_SEARCH_MODEL_PACK_MANIFEST_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported TD search model-pack manifest schemaVersion=${String(schemaVersion)}.`
    );
  }
  const modelRaw = requiredRecord(source.model, 'manifest.model');
  const modelType = requiredString(modelRaw.modelType, 'manifest.model.modelType');
  if (modelType !== TD_SEARCH_MODEL_TYPE) {
    throw new Error(`Unsupported TD search modelType: ${modelType}`);
  }
  const weightsPath = requiredString(modelRaw.weightsPath, 'manifest.model.weightsPath');

  const valueRaw = requiredRecord(modelRaw.value, 'manifest.model.value');
  const valueEncoding = requiredInteger(
    valueRaw.encodingVersion,
    'manifest.model.value.encodingVersion'
  );
  const valueObservationDim = requiredInteger(
    valueRaw.observationDim,
    'manifest.model.value.observationDim'
  );
  const valueHiddenDim = requiredInteger(
    valueRaw.hiddenDim,
    'manifest.model.value.hiddenDim'
  );
  const valueCheckpointType = requiredString(
    valueRaw.checkpointType,
    'manifest.model.value.checkpointType'
  );
  const valueKeys = requiredStringArray(
    valueRaw.requiredStateDictKeys,
    'manifest.model.value.requiredStateDictKeys'
  );
  if (valueCheckpointType !== TD_VALUE_CHECKPOINT_TYPE) {
    throw new Error(
      `Unsupported TD search value checkpointType: ${valueCheckpointType}.`
    );
  }
  if (valueEncoding !== ENCODING_VERSION) {
    throw new Error(
      `TD search value encoding mismatch. expected=${String(ENCODING_VERSION)} actual=${String(valueEncoding)}.`
    );
  }
  if (valueObservationDim !== OBSERVATION_DIM) {
    throw new Error(
      `TD search value observation dim mismatch. expected=${String(OBSERVATION_DIM)} actual=${String(valueObservationDim)}.`
    );
  }
  if (valueHiddenDim <= 0) {
    throw new Error('TD search value hiddenDim must be > 0.');
  }
  for (const key of TD_VALUE_REQUIRED_TENSOR_KEYS) {
    if (!valueKeys.includes(key)) {
      throw new Error(`TD search manifest value keys missing ${key}.`);
    }
  }

  const opponentRaw = requiredRecord(modelRaw.opponent, 'manifest.model.opponent');
  const opponentEncoding = requiredInteger(
    opponentRaw.encodingVersion,
    'manifest.model.opponent.encodingVersion'
  );
  const opponentObservationDim = requiredInteger(
    opponentRaw.observationDim,
    'manifest.model.opponent.observationDim'
  );
  const opponentActionFeatureDim = requiredInteger(
    opponentRaw.actionFeatureDim,
    'manifest.model.opponent.actionFeatureDim'
  );
  const opponentHiddenDim = requiredInteger(
    opponentRaw.hiddenDim,
    'manifest.model.opponent.hiddenDim'
  );
  const opponentCheckpointType = requiredString(
    opponentRaw.checkpointType,
    'manifest.model.opponent.checkpointType'
  );
  const opponentKeys = requiredStringArray(
    opponentRaw.requiredStateDictKeys,
    'manifest.model.opponent.requiredStateDictKeys'
  );
  if (opponentCheckpointType !== TD_SEARCH_OPPONENT_CHECKPOINT_TYPE) {
    throw new Error(
      `Unsupported TD search opponent checkpointType: ${opponentCheckpointType}.`
    );
  }
  if (opponentEncoding !== ENCODING_VERSION) {
    throw new Error(
      `TD search opponent encoding mismatch. expected=${String(ENCODING_VERSION)} actual=${String(opponentEncoding)}.`
    );
  }
  if (opponentObservationDim !== OBSERVATION_DIM) {
    throw new Error(
      `TD search opponent observation dim mismatch. expected=${String(OBSERVATION_DIM)} actual=${String(opponentObservationDim)}.`
    );
  }
  if (opponentActionFeatureDim !== ACTION_FEATURE_DIM) {
    throw new Error(
      `TD search opponent action feature dim mismatch. expected=${String(ACTION_FEATURE_DIM)} actual=${String(opponentActionFeatureDim)}.`
    );
  }
  if (opponentHiddenDim <= 0) {
    throw new Error('TD search opponent hiddenDim must be > 0.');
  }
  for (const key of TD_SEARCH_OPPONENT_REQUIRED_TENSOR_KEYS) {
    if (!opponentKeys.includes(key)) {
      throw new Error(`TD search manifest opponent keys missing ${key}.`);
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
      weightsPath,
      value: {
        checkpointType: valueCheckpointType,
        encodingVersion: valueEncoding,
        observationDim: valueObservationDim,
        hiddenDim: valueHiddenDim,
        requiredStateDictKeys: valueKeys,
      },
      opponent: {
        checkpointType: opponentCheckpointType,
        encodingVersion: opponentEncoding,
        observationDim: opponentObservationDim,
        actionFeatureDim: opponentActionFeatureDim,
        hiddenDim: opponentHiddenDim,
        requiredStateDictKeys: opponentKeys,
      },
    },
    source: {
      runId: optionalStringOrNull(sourceRaw.runId, 'manifest.source.runId'),
      valueCheckpoint: optionalStringOrNull(
        sourceRaw.valueCheckpoint,
        'manifest.source.valueCheckpoint'
      ),
      opponentCheckpoint: optionalStringOrNull(
        sourceRaw.opponentCheckpoint,
        'manifest.source.opponentCheckpoint'
      ),
      checkpointMetadata: optionalRecord(
        sourceRaw.checkpointMetadata,
        'manifest.source.checkpointMetadata'
      ),
    },
  };
}

function parseTdSearchWeightsFile(value: unknown): TdSearchWeightsFile {
  const source = requiredRecord(value, 'TD search model-pack weights');
  const schemaVersion = requiredInteger(source.schemaVersion, 'weights.schemaVersion');
  if (schemaVersion !== TD_VALUE_MODEL_PACK_WEIGHTS_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported TD search weights schemaVersion=${String(schemaVersion)}.`
    );
  }
  const valueTensors = requiredTensorRecord(source.valueTensors, 'weights.valueTensors');
  const opponentTensors = requiredTensorRecord(
    source.opponentTensors,
    'weights.opponentTensors'
  );
  return {
    schemaVersion,
    valueTensors,
    opponentTensors,
  };
}

function createTdSearchValueNetwork(
  manifest: TdSearchModelPackManifest,
  weights: TdSearchWeightsFile
): TdValueNetwork {
  const hiddenDim = manifest.model.value.hiddenDim;
  const observationDim = manifest.model.value.observationDim;
  return new TdValueNetwork({
    observationDim,
    hiddenDim,
    w1: parseTensor(
      weights.valueTensors,
      'encoder.0.weight',
      [hiddenDim, observationDim]
    ),
    b1: parseTensor(weights.valueTensors, 'encoder.0.bias', [hiddenDim]),
    w2: parseTensor(weights.valueTensors, 'encoder.2.weight', [hiddenDim, hiddenDim]),
    b2: parseTensor(weights.valueTensors, 'encoder.2.bias', [hiddenDim]),
    w3: parseTensor(weights.valueTensors, 'encoder.4.weight', [1, hiddenDim]),
    b3: parseTensor(weights.valueTensors, 'encoder.4.bias', [1]),
  });
}

function createTdSearchOpponentNetwork(
  manifest: TdSearchModelPackManifest,
  weights: TdSearchWeightsFile
): TdSearchOpponentNetwork {
  const hiddenDim = manifest.model.opponent.hiddenDim;
  const observationDim = manifest.model.opponent.observationDim;
  const actionFeatureDim = manifest.model.opponent.actionFeatureDim;
  return new TdSearchOpponentNetwork({
    observationDim,
    actionFeatureDim,
    hiddenDim,
    obsW1: parseTensor(
      weights.opponentTensors,
      'obs_encoder.0.weight',
      [hiddenDim, observationDim]
    ),
    obsB1: parseTensor(weights.opponentTensors, 'obs_encoder.0.bias', [hiddenDim]),
    obsW2: parseTensor(
      weights.opponentTensors,
      'obs_encoder.2.weight',
      [hiddenDim, hiddenDim]
    ),
    obsB2: parseTensor(weights.opponentTensors, 'obs_encoder.2.bias', [hiddenDim]),
    actionW: parseTensor(
      weights.opponentTensors,
      'action_encoder.0.weight',
      [hiddenDim, actionFeatureDim]
    ),
    actionB: parseTensor(
      weights.opponentTensors,
      'action_encoder.0.bias',
      [hiddenDim]
    ),
    headW1: parseTensor(
      weights.opponentTensors,
      'policy_head.0.weight',
      [hiddenDim, hiddenDim * 3]
    ),
    headB1: parseTensor(weights.opponentTensors, 'policy_head.0.bias', [hiddenDim]),
    headW2: parseTensor(weights.opponentTensors, 'policy_head.2.weight', [1, hiddenDim]),
    headB2: parseTensor(weights.opponentTensors, 'policy_head.2.bias', [1]),
  });
}

function requiredTensorRecord(
  value: unknown,
  label: string
): Record<string, TdSearchTensorPayload> {
  const raw = requiredRecord(value, label);
  const out: Record<string, TdSearchTensorPayload> = {};
  for (const [key, entry] of Object.entries(raw)) {
    const row = requiredRecord(entry, `${label}.${key}`);
    out[key] = {
      shape: requiredIntegerArray(row.shape, `${label}.${key}.shape`),
      values: requiredNumberArray(row.values, `${label}.${key}.values`),
    };
  }
  return out;
}

function parseTensor(
  tensors: Record<string, TdSearchTensorPayload>,
  key: OpponentTensorKey | ValueTensorKey,
  expectedShape: readonly number[]
): Float32Array {
  const payload = tensors[key];
  if (!payload) {
    throw new Error(`TD search model weights are missing tensor ${key}.`);
  }
  if (
    payload.shape.length !== expectedShape.length
    || payload.shape.some((value, index) => value !== expectedShape[index])
  ) {
    throw new Error(
      `TD search tensor shape mismatch for ${key}. expected=[${expectedShape.join(',')}] actual=[${payload.shape.join(',')}]`
    );
  }
  const expectedLength = expectedShape.reduce((product, current) => product * current, 1);
  if (payload.values.length !== expectedLength) {
    throw new Error(
      `TD search tensor length mismatch for ${key}. expected=${String(expectedLength)} actual=${String(payload.values.length)}`
    );
  }
  const out = new Float32Array(expectedLength);
  for (let index = 0; index < payload.values.length; index += 1) {
    out[index] = payload.values[index];
  }
  return out;
}

function selectPack(
  packs: readonly TdValueModelPackIndexEntry[],
  defaultPackId: string | null
): TdValueModelPackIndexEntry {
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
