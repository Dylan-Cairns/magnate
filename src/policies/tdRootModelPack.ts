import {
  ACTION_FEATURE_DIM,
  ENCODING_VERSION,
  OBSERVATION_DIM,
  encodeActionInto,
} from './trainingEncoding';
import type { GameAction } from '../engine/types';
import {
  TD_VALUE_CHECKPOINT_TYPE,
  TD_VALUE_MODEL_PACK_WEIGHTS_SCHEMA_VERSION,
  TD_VALUE_REQUIRED_TENSOR_KEYS,
  TdValueNetwork,
} from './tdValueModelPack';
import {
  fetchJson,
  optionalRecord,
  optionalStringOrNull,
  parseTensor,
  requiredInteger,
  requiredRecord,
  requiredString,
  requiredStringArray,
  requiredTensorRecord,
  resolveManifestUrl,
  selectPack,
  toAbsoluteUrl,
  type TensorPayload,
} from './modelPackUtils';
import type {
  LoadedTdGuidanceModel,
  TdGuidanceActionScorer,
} from './tdGuidanceModel';

export const TD_ROOT_MODEL_TYPE = 'td-root-search-v1';
export const TD_ROOT_OPPONENT_CHECKPOINT_TYPE = 'magnate_td_opponent_v1';
export const TD_ROOT_MODEL_PACK_MANIFEST_SCHEMA_VERSION = 1;

export const TD_ROOT_OPPONENT_REQUIRED_TENSOR_KEYS = [
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

type OpponentTensorKey = (typeof TD_ROOT_OPPONENT_REQUIRED_TENSOR_KEYS)[number];
type ValueTensorKey = (typeof TD_VALUE_REQUIRED_TENSOR_KEYS)[number];
type TdRootTensorKey = OpponentTensorKey | ValueTensorKey;

interface TdRootWeightsFile {
  schemaVersion: number;
  valueTensors: Record<string, TensorPayload>;
  opponentTensors: Record<string, TensorPayload>;
}

export interface TdRootModelPackManifest {
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

interface TdRootOpponentNetworkTensors {
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

export class TdRootOpponentNetwork implements TdGuidanceActionScorer {
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
  private readonly observationHidden1: Float32Array;
  private readonly observationHidden2: Float32Array;
  private readonly actionEmbedding: Float32Array;
  private readonly actionFeatureScratch: Float32Array;
  private readonly interactionEmbedding: Float32Array;
  private readonly observationHeadBase: Float32Array;
  private readonly policyHidden: Float32Array;
  private readonly pairObservationHidden1A: Float32Array;
  private readonly pairObservationHidden1B: Float32Array;
  private readonly pairObservationHidden2A: Float32Array;
  private readonly pairObservationHidden2B: Float32Array;
  private readonly pairActionEmbeddingA: Float32Array;
  private readonly pairActionEmbeddingB: Float32Array;
  private readonly pairActionFeatureScratchA: Float32Array;
  private readonly pairActionFeatureScratchB: Float32Array;
  private readonly pairInteractionEmbeddingA: Float32Array;
  private readonly pairInteractionEmbeddingB: Float32Array;
  private readonly pairObservationHeadBaseA: Float32Array;
  private readonly pairObservationHeadBaseB: Float32Array;
  private readonly pairPolicyHiddenA: Float32Array;
  private readonly pairPolicyHiddenB: Float32Array;

  constructor(tensors: TdRootOpponentNetworkTensors) {
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
    this.observationHidden1 = new Float32Array(this.hiddenDim);
    this.observationHidden2 = new Float32Array(this.hiddenDim);
    this.actionEmbedding = new Float32Array(this.hiddenDim);
    this.actionFeatureScratch = new Float32Array(this.actionFeatureDim);
    this.interactionEmbedding = new Float32Array(this.hiddenDim);
    this.observationHeadBase = new Float32Array(this.hiddenDim);
    this.policyHidden = new Float32Array(this.hiddenDim);
    this.pairObservationHidden1A = new Float32Array(this.hiddenDim);
    this.pairObservationHidden1B = new Float32Array(this.hiddenDim);
    this.pairObservationHidden2A = new Float32Array(this.hiddenDim);
    this.pairObservationHidden2B = new Float32Array(this.hiddenDim);
    this.pairActionEmbeddingA = new Float32Array(this.hiddenDim);
    this.pairActionEmbeddingB = new Float32Array(this.hiddenDim);
    this.pairActionFeatureScratchA = new Float32Array(this.actionFeatureDim);
    this.pairActionFeatureScratchB = new Float32Array(this.actionFeatureDim);
    this.pairInteractionEmbeddingA = new Float32Array(this.hiddenDim);
    this.pairInteractionEmbeddingB = new Float32Array(this.hiddenDim);
    this.pairObservationHeadBaseA = new Float32Array(this.hiddenDim);
    this.pairObservationHeadBaseB = new Float32Array(this.hiddenDim);
    this.pairPolicyHiddenA = new Float32Array(this.hiddenDim);
    this.pairPolicyHiddenB = new Float32Array(this.hiddenDim);
  }

  logits(
    observation: readonly number[],
    actionFeatures: readonly number[][]
  ): Float32Array {
    this.validateObservation(observation);
    if (actionFeatures.length === 0) {
      return new Float32Array(0);
    }
    for (let index = 0; index < actionFeatures.length; index += 1) {
      const features = actionFeatures[index];
      if (features.length !== this.actionFeatureDim) {
        throw new Error(
          `TD root opponent model action feature length mismatch at index ${String(index)}. expected=${String(this.actionFeatureDim)} actual=${String(features.length)}.`
        );
      }
    }

    const obsEmbedding = this.encodeObservation(observation);
    this.encodeObservationHeadBase(obsEmbedding);
    const logits = new Float32Array(actionFeatures.length);
    for (let index = 0; index < actionFeatures.length; index += 1) {
      const actionEmbedding = this.encodeAction(actionFeatures[index]);
      const interactionEmbedding = this.encodeInteraction(
        obsEmbedding,
        actionEmbedding
      );
      logits[index] = this.policyLogit(actionEmbedding, interactionEmbedding);
    }
    return logits;
  }

  logitsForActions(
    observation: readonly number[],
    actions: readonly GameAction[]
  ): Float32Array {
    this.validateObservation(observation);
    if (actions.length === 0) {
      return new Float32Array(0);
    }

    const obsEmbedding = this.encodeObservation(observation);
    this.encodeObservationHeadBase(obsEmbedding);
    const logits = new Float32Array(actions.length);
    for (let index = 0; index < actions.length; index += 1) {
      encodeActionInto(actions[index], this.actionFeatureScratch);
      const actionEmbedding = this.encodeAction(this.actionFeatureScratch);
      const interactionEmbedding = this.encodeInteraction(
        obsEmbedding,
        actionEmbedding
      );
      logits[index] = this.policyLogit(actionEmbedding, interactionEmbedding);
    }
    return logits;
  }

  /**
   * Benchmark candidate for the browser's natural two-rollout worker chunk.
   *
   * This method deliberately keeps each lane's arithmetic in the same order as
   * logitsForActions while interleaving the lanes so weight reads can be shared
   * by the JavaScript engine. It is not wired into production search scheduling.
   */
  logitsForActionPair(
    observationA: readonly number[],
    actionsA: readonly GameAction[],
    observationB: readonly number[],
    actionsB: readonly GameAction[]
  ): readonly [Float32Array, Float32Array] {
    this.validateObservation(observationA);
    this.validateObservation(observationB);
    if (actionsA.length === 0 || actionsB.length === 0) {
      return [
        this.logitsForActions(observationA, actionsA),
        this.logitsForActions(observationB, actionsB),
      ];
    }

    this.encodeObservationPair(observationA, observationB);
    this.encodeObservationHeadBasePair();

    const logitsA = new Float32Array(actionsA.length);
    const logitsB = new Float32Array(actionsB.length);
    const sharedCount = Math.min(actionsA.length, actionsB.length);
    for (let index = 0; index < sharedCount; index += 1) {
      encodeActionInto(actionsA[index], this.pairActionFeatureScratchA);
      encodeActionInto(actionsB[index], this.pairActionFeatureScratchB);
      this.encodeActionPair();
      this.encodeInteractionPair();
      const [logitA, logitB] = this.policyLogitPair();
      logitsA[index] = logitA;
      logitsB[index] = logitB;
    }

    for (let index = sharedCount; index < actionsA.length; index += 1) {
      encodeActionInto(actionsA[index], this.pairActionFeatureScratchA);
      this.encodeActionLane(
        this.pairActionFeatureScratchA,
        this.pairActionEmbeddingA
      );
      this.encodeInteractionLane(
        this.pairObservationHidden2A,
        this.pairActionEmbeddingA,
        this.pairInteractionEmbeddingA
      );
      logitsA[index] = this.policyLogitLane(
        this.pairObservationHeadBaseA,
        this.pairActionEmbeddingA,
        this.pairInteractionEmbeddingA,
        this.pairPolicyHiddenA
      );
    }
    for (let index = sharedCount; index < actionsB.length; index += 1) {
      encodeActionInto(actionsB[index], this.pairActionFeatureScratchB);
      this.encodeActionLane(
        this.pairActionFeatureScratchB,
        this.pairActionEmbeddingB
      );
      this.encodeInteractionLane(
        this.pairObservationHidden2B,
        this.pairActionEmbeddingB,
        this.pairInteractionEmbeddingB
      );
      logitsB[index] = this.policyLogitLane(
        this.pairObservationHeadBaseB,
        this.pairActionEmbeddingB,
        this.pairInteractionEmbeddingB,
        this.pairPolicyHiddenB
      );
    }
    return [logitsA, logitsB];
  }

  private validateObservation(observation: readonly number[]): void {
    if (observation.length !== this.observationDim) {
      throw new Error(
        `TD root opponent model observation length mismatch. expected=${String(this.observationDim)} actual=${String(observation.length)}.`
      );
    }
  }

  private encodeObservation(observation: readonly number[]): Float32Array {
    const hidden1 = this.observationHidden1;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.obsB1[row];
      const offset = row * this.observationDim;
      for (let col = 0; col < this.observationDim; col += 1) {
        sum += this.obsW1[offset + col] * observation[col];
      }
      hidden1[row] = Math.tanh(sum);
    }

    const hidden2 = this.observationHidden2;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.obsB2[row];
      const offset = row * this.hiddenDim;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        sum += this.obsW2[offset + col] * hidden1[col];
      }
      hidden2[row] = Math.tanh(sum);
    }
    return hidden2;
  }

  private encodeObservationPair(
    observationA: readonly number[],
    observationB: readonly number[]
  ): void {
    const hidden1A = this.pairObservationHidden1A;
    const hidden1B = this.pairObservationHidden1B;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sumA = this.obsB1[row];
      let sumB = this.obsB1[row];
      const offset = row * this.observationDim;
      for (let col = 0; col < this.observationDim; col += 1) {
        const weight = this.obsW1[offset + col];
        sumA += weight * observationA[col];
        sumB += weight * observationB[col];
      }
      hidden1A[row] = Math.tanh(sumA);
      hidden1B[row] = Math.tanh(sumB);
    }

    const hidden2A = this.pairObservationHidden2A;
    const hidden2B = this.pairObservationHidden2B;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sumA = this.obsB2[row];
      let sumB = this.obsB2[row];
      const offset = row * this.hiddenDim;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        const weight = this.obsW2[offset + col];
        sumA += weight * hidden1A[col];
        sumB += weight * hidden1B[col];
      }
      hidden2A[row] = Math.tanh(sumA);
      hidden2B[row] = Math.tanh(sumB);
    }
  }

  private encodeAction(actionFeatures: ArrayLike<number>): Float32Array {
    const embedding = this.actionEmbedding;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.actionB[row];
      const offset = row * this.actionFeatureDim;
      for (let col = 0; col < this.actionFeatureDim; col += 1) {
        sum += this.actionW[offset + col] * actionFeatures[col];
      }
      embedding[row] = Math.tanh(sum);
    }
    return embedding;
  }

  private encodeActionPair(): void {
    const embeddingA = this.pairActionEmbeddingA;
    const embeddingB = this.pairActionEmbeddingB;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sumA = this.actionB[row];
      let sumB = this.actionB[row];
      const offset = row * this.actionFeatureDim;
      for (let col = 0; col < this.actionFeatureDim; col += 1) {
        const weight = this.actionW[offset + col];
        sumA += weight * this.pairActionFeatureScratchA[col];
        sumB += weight * this.pairActionFeatureScratchB[col];
      }
      embeddingA[row] = Math.tanh(sumA);
      embeddingB[row] = Math.tanh(sumB);
    }
  }

  private encodeActionLane(
    actionFeatures: ArrayLike<number>,
    embedding: Float32Array
  ): void {
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.actionB[row];
      const offset = row * this.actionFeatureDim;
      for (let col = 0; col < this.actionFeatureDim; col += 1) {
        sum += this.actionW[offset + col] * actionFeatures[col];
      }
      embedding[row] = Math.tanh(sum);
    }
  }

  private encodeInteraction(
    observationEmbedding: Float32Array,
    actionEmbedding: Float32Array
  ): Float32Array {
    const interaction = this.interactionEmbedding;
    for (let index = 0; index < this.hiddenDim; index += 1) {
      interaction[index] = observationEmbedding[index] * actionEmbedding[index];
    }
    return interaction;
  }

  private encodeInteractionPair(): void {
    for (let index = 0; index < this.hiddenDim; index += 1) {
      this.pairInteractionEmbeddingA[index] =
        this.pairObservationHidden2A[index] * this.pairActionEmbeddingA[index];
      this.pairInteractionEmbeddingB[index] =
        this.pairObservationHidden2B[index] * this.pairActionEmbeddingB[index];
    }
  }

  private encodeInteractionLane(
    observationEmbedding: Float32Array,
    actionEmbedding: Float32Array,
    interactionEmbedding: Float32Array
  ): void {
    for (let index = 0; index < this.hiddenDim; index += 1) {
      interactionEmbedding[index] =
        observationEmbedding[index] * actionEmbedding[index];
    }
  }

  private encodeObservationHeadBase(
    observationEmbedding: Float32Array
  ): Float32Array {
    const headBase = this.observationHeadBase;
    const width = this.hiddenDim * 3;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.headB1[row];
      const offset = row * width;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        sum += this.headW1[offset + col] * observationEmbedding[col];
      }
      headBase[row] = sum;
    }
    return headBase;
  }

  private encodeObservationHeadBasePair(): void {
    const width = this.hiddenDim * 3;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sumA = this.headB1[row];
      let sumB = this.headB1[row];
      const offset = row * width;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        const weight = this.headW1[offset + col];
        sumA += weight * this.pairObservationHidden2A[col];
        sumB += weight * this.pairObservationHidden2B[col];
      }
      this.pairObservationHeadBaseA[row] = sumA;
      this.pairObservationHeadBaseB[row] = sumB;
    }
  }

  private policyLogit(
    actionEmbedding: Float32Array,
    interactionEmbedding: Float32Array
  ): number {
    const hidden = this.policyHidden;
    const width = this.hiddenDim * 3;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = this.observationHeadBase[row];
      const offset = row * width;
      const actionOffset = offset + this.hiddenDim;
      const productOffset = actionOffset + this.hiddenDim;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        sum += this.headW1[actionOffset + col] * actionEmbedding[col];
        sum += this.headW1[productOffset + col] * interactionEmbedding[col];
      }
      hidden[row] = Math.tanh(sum);
    }

    let output = this.headB2[0];
    for (let col = 0; col < this.hiddenDim; col += 1) {
      output += this.headW2[col] * hidden[col];
    }
    return output;
  }

  private policyLogitPair(): readonly [number, number] {
    const width = this.hiddenDim * 3;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sumA = this.pairObservationHeadBaseA[row];
      let sumB = this.pairObservationHeadBaseB[row];
      const offset = row * width;
      const actionOffset = offset + this.hiddenDim;
      const productOffset = actionOffset + this.hiddenDim;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        const actionWeight = this.headW1[actionOffset + col];
        const productWeight = this.headW1[productOffset + col];
        sumA += actionWeight * this.pairActionEmbeddingA[col];
        sumA += productWeight * this.pairInteractionEmbeddingA[col];
        sumB += actionWeight * this.pairActionEmbeddingB[col];
        sumB += productWeight * this.pairInteractionEmbeddingB[col];
      }
      this.pairPolicyHiddenA[row] = Math.tanh(sumA);
      this.pairPolicyHiddenB[row] = Math.tanh(sumB);
    }

    let outputA = this.headB2[0];
    let outputB = this.headB2[0];
    for (let col = 0; col < this.hiddenDim; col += 1) {
      const weight = this.headW2[col];
      outputA += weight * this.pairPolicyHiddenA[col];
      outputB += weight * this.pairPolicyHiddenB[col];
    }
    return [outputA, outputB];
  }

  private policyLogitLane(
    observationHeadBase: Float32Array,
    actionEmbedding: Float32Array,
    interactionEmbedding: Float32Array,
    policyHidden: Float32Array
  ): number {
    const width = this.hiddenDim * 3;
    for (let row = 0; row < this.hiddenDim; row += 1) {
      let sum = observationHeadBase[row];
      const offset = row * width;
      const actionOffset = offset + this.hiddenDim;
      const productOffset = actionOffset + this.hiddenDim;
      for (let col = 0; col < this.hiddenDim; col += 1) {
        sum += this.headW1[actionOffset + col] * actionEmbedding[col];
        sum += this.headW1[productOffset + col] * interactionEmbedding[col];
      }
      policyHidden[row] = Math.tanh(sum);
    }

    let output = this.headB2[0];
    for (let col = 0; col < this.hiddenDim; col += 1) {
      output += this.headW2[col] * policyHidden[col];
    }
    return output;
  }
}

export async function loadTdRootModelFromIndexUrl(
  indexUrl: string
): Promise<LoadedTdGuidanceModel> {
  const absoluteIndexUrl = toAbsoluteUrl(indexUrl);
  const indexPayload = await fetchJson(absoluteIndexUrl);
  const index = parseTdRootModelPackIndex(indexPayload);
  const packs = index.packs.filter(
    (entry) => entry.modelType === TD_ROOT_MODEL_TYPE
  );
  if (packs.length === 0) {
    throw new Error(
      'TD root model-pack index does not include any td-root-search-v1 packs.'
    );
  }
  const selected = selectPack(packs, index.defaultPackId);
  const manifestUrl = resolveManifestUrl(
    absoluteIndexUrl,
    selected.manifestPath
  );
  return loadTdRootModelFromManifestUrl(manifestUrl);
}

export async function loadTdRootModelFromManifestUrl(
  manifestUrl: string
): Promise<LoadedTdGuidanceModel> {
  const absoluteManifestUrl = toAbsoluteUrl(manifestUrl);
  const manifestPayload = await fetchJson(absoluteManifestUrl);
  const manifest = parseTdRootModelPackManifest(manifestPayload);
  const weightsUrl = new URL(
    manifest.model.weightsPath,
    absoluteManifestUrl
  ).toString();
  const weightsPayload = await fetchJson(weightsUrl);
  const weights = parseTdRootWeightsFile(weightsPayload);
  return {
    valueScorer: createTdRootValueNetwork(manifest, weights),
    opponentScorer: createTdRootOpponentNetwork(manifest, weights),
  };
}

export function parseTdRootModelPackIndex(value: unknown): {
  schemaVersion: number;
  generatedAtUtc: string;
  defaultPackId: string | null;
  packs: Array<{
    id: string;
    label: string;
    modelType: string;
    manifestPath: string;
    createdAtUtc: string;
  }>;
} {
  const source = requiredRecord(value, 'TD root model-pack index');
  const schemaVersion = requiredInteger(
    source.schemaVersion,
    'index.schemaVersion'
  );
  if (schemaVersion !== 1) {
    throw new Error(
      `Unsupported TD root model-pack index schemaVersion=${String(schemaVersion)}.`
    );
  }
  const packsRaw = source.packs;
  if (!Array.isArray(packsRaw)) {
    throw new Error('TD root model-pack index packs must be an array.');
  }
  return {
    schemaVersion,
    generatedAtUtc: requiredString(
      source.generatedAtUtc,
      'index.generatedAtUtc'
    ),
    defaultPackId: optionalStringOrNull(
      source.defaultPackId,
      'index.defaultPackId'
    ),
    packs: packsRaw.map((entry, index) => {
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
      };
    }),
  };
}

export function parseTdRootModelPackManifest(
  value: unknown
): TdRootModelPackManifest {
  const source = requiredRecord(value, 'TD root model-pack manifest');
  const schemaVersion = requiredInteger(
    source.schemaVersion,
    'manifest.schemaVersion'
  );
  if (schemaVersion !== TD_ROOT_MODEL_PACK_MANIFEST_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported TD root model-pack manifest schemaVersion=${String(schemaVersion)}.`
    );
  }
  const modelRaw = requiredRecord(source.model, 'manifest.model');
  const modelType = requiredString(
    modelRaw.modelType,
    'manifest.model.modelType'
  );
  if (modelType !== TD_ROOT_MODEL_TYPE) {
    throw new Error(`Unsupported TD root modelType: ${modelType}`);
  }
  const weightsPath = requiredString(
    modelRaw.weightsPath,
    'manifest.model.weightsPath'
  );

  const valueRaw = requiredRecord(modelRaw.value, 'manifest.model.value');
  const valueCheckpointType = requiredString(
    valueRaw.checkpointType,
    'manifest.model.value.checkpointType'
  );
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
  const valueKeys = requiredStringArray(
    valueRaw.requiredStateDictKeys,
    'manifest.model.value.requiredStateDictKeys'
  );
  if (valueCheckpointType !== TD_VALUE_CHECKPOINT_TYPE) {
    throw new Error(
      `Unsupported TD root value checkpointType: ${valueCheckpointType}.`
    );
  }
  if (valueEncoding !== ENCODING_VERSION) {
    throw new Error(
      `TD root value encoding mismatch. expected=${String(ENCODING_VERSION)} actual=${String(valueEncoding)}.`
    );
  }
  if (valueObservationDim !== OBSERVATION_DIM) {
    throw new Error(
      `TD root value observation dim mismatch. expected=${String(OBSERVATION_DIM)} actual=${String(valueObservationDim)}.`
    );
  }
  if (valueHiddenDim <= 0) {
    throw new Error('TD root value hiddenDim must be > 0.');
  }
  for (const key of TD_VALUE_REQUIRED_TENSOR_KEYS) {
    if (!valueKeys.includes(key)) {
      throw new Error(`TD root manifest value keys missing ${key}.`);
    }
  }

  const opponentRaw = requiredRecord(
    modelRaw.opponent,
    'manifest.model.opponent'
  );
  const opponentCheckpointType = requiredString(
    opponentRaw.checkpointType,
    'manifest.model.opponent.checkpointType'
  );
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
  const opponentKeys = requiredStringArray(
    opponentRaw.requiredStateDictKeys,
    'manifest.model.opponent.requiredStateDictKeys'
  );
  if (opponentCheckpointType !== TD_ROOT_OPPONENT_CHECKPOINT_TYPE) {
    throw new Error(
      `Unsupported TD root opponent checkpointType: ${opponentCheckpointType}.`
    );
  }
  if (opponentEncoding !== ENCODING_VERSION) {
    throw new Error(
      `TD root opponent encoding mismatch. expected=${String(ENCODING_VERSION)} actual=${String(opponentEncoding)}.`
    );
  }
  if (opponentObservationDim !== OBSERVATION_DIM) {
    throw new Error(
      `TD root opponent observation dim mismatch. expected=${String(OBSERVATION_DIM)} actual=${String(opponentObservationDim)}.`
    );
  }
  if (opponentActionFeatureDim !== ACTION_FEATURE_DIM) {
    throw new Error(
      `TD root opponent action feature dim mismatch. expected=${String(ACTION_FEATURE_DIM)} actual=${String(opponentActionFeatureDim)}.`
    );
  }
  if (opponentHiddenDim <= 0) {
    throw new Error('TD root opponent hiddenDim must be > 0.');
  }
  for (const key of TD_ROOT_OPPONENT_REQUIRED_TENSOR_KEYS) {
    if (!opponentKeys.includes(key)) {
      throw new Error(`TD root manifest opponent keys missing ${key}.`);
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

function parseTdRootWeightsFile(value: unknown): TdRootWeightsFile {
  const source = requiredRecord(value, 'TD root model-pack weights');
  const schemaVersion = requiredInteger(
    source.schemaVersion,
    'weights.schemaVersion'
  );
  if (schemaVersion !== TD_VALUE_MODEL_PACK_WEIGHTS_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported TD root weights schemaVersion=${String(schemaVersion)}.`
    );
  }
  return {
    schemaVersion,
    valueTensors: requiredTensorRecord(
      source.valueTensors,
      'weights.valueTensors'
    ),
    opponentTensors: requiredTensorRecord(
      source.opponentTensors,
      'weights.opponentTensors'
    ),
  };
}

function createTdRootValueNetwork(
  manifest: TdRootModelPackManifest,
  weights: TdRootWeightsFile
): TdValueNetwork {
  const hiddenDim = manifest.model.value.hiddenDim;
  const observationDim = manifest.model.value.observationDim;
  return new TdValueNetwork({
    observationDim,
    hiddenDim,
    w1: parseTdRootTensor(weights.valueTensors, 'encoder.0.weight', [
      hiddenDim,
      observationDim,
    ]),
    b1: parseTdRootTensor(weights.valueTensors, 'encoder.0.bias', [hiddenDim]),
    w2: parseTdRootTensor(weights.valueTensors, 'encoder.2.weight', [
      hiddenDim,
      hiddenDim,
    ]),
    b2: parseTdRootTensor(weights.valueTensors, 'encoder.2.bias', [hiddenDim]),
    w3: parseTdRootTensor(weights.valueTensors, 'encoder.4.weight', [
      1,
      hiddenDim,
    ]),
    b3: parseTdRootTensor(weights.valueTensors, 'encoder.4.bias', [1]),
  });
}

function createTdRootOpponentNetwork(
  manifest: TdRootModelPackManifest,
  weights: TdRootWeightsFile
): TdRootOpponentNetwork {
  const hiddenDim = manifest.model.opponent.hiddenDim;
  const observationDim = manifest.model.opponent.observationDim;
  const actionFeatureDim = manifest.model.opponent.actionFeatureDim;
  return new TdRootOpponentNetwork({
    observationDim,
    actionFeatureDim,
    hiddenDim,
    obsW1: parseTdRootTensor(weights.opponentTensors, 'obs_encoder.0.weight', [
      hiddenDim,
      observationDim,
    ]),
    obsB1: parseTdRootTensor(weights.opponentTensors, 'obs_encoder.0.bias', [
      hiddenDim,
    ]),
    obsW2: parseTdRootTensor(weights.opponentTensors, 'obs_encoder.2.weight', [
      hiddenDim,
      hiddenDim,
    ]),
    obsB2: parseTdRootTensor(weights.opponentTensors, 'obs_encoder.2.bias', [
      hiddenDim,
    ]),
    actionW: parseTdRootTensor(
      weights.opponentTensors,
      'action_encoder.0.weight',
      [hiddenDim, actionFeatureDim]
    ),
    actionB: parseTdRootTensor(
      weights.opponentTensors,
      'action_encoder.0.bias',
      [hiddenDim]
    ),
    headW1: parseTdRootTensor(weights.opponentTensors, 'policy_head.0.weight', [
      hiddenDim,
      hiddenDim * 3,
    ]),
    headB1: parseTdRootTensor(weights.opponentTensors, 'policy_head.0.bias', [
      hiddenDim,
    ]),
    headW2: parseTdRootTensor(weights.opponentTensors, 'policy_head.2.weight', [
      1,
      hiddenDim,
    ]),
    headB2: parseTdRootTensor(
      weights.opponentTensors,
      'policy_head.2.bias',
      [1]
    ),
  });
}

function parseTdRootTensor(
  tensors: Record<string, TensorPayload>,
  key: TdRootTensorKey,
  expectedShape: readonly number[]
): Float32Array {
  return parseTensor(tensors, key, expectedShape, 'TD root model');
}
