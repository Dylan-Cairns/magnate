import type { GameAction } from '../engine/types';

import { encodeActionCandidates, encodeObservation } from './trainingEncoding';
import type { ActionPolicy } from './types';

interface BrowserPpoWeights {
  checkpointType: string;
  observationDim: number;
  actionFeatureDim: number;
  hiddenDim: number;
  weights: {
    obsEncoder0Weight: number[][];
    obsEncoder0Bias: number[];
    obsEncoder2Weight: number[][];
    obsEncoder2Bias: number[];
    actionEncoder0Weight: number[][];
    actionEncoder0Bias: number[];
    policyHead0Weight: number[][];
    policyHead0Bias: number[];
    policyHead2Weight: number[][];
    policyHead2Bias: number[];
  };
}

const BROWSER_CHECKPOINT_TYPE = 'magnate_ppo_browser_v1';
const CHECKPOINT_CACHE = new Map<string, Promise<BrowserPpoModel>>();

export interface PpoBrowserPolicyOptions {
  modelUrl: string;
  temperature?: number;
}

export function createPpoBrowserPolicy(options: PpoBrowserPolicyOptions): ActionPolicy {
  const { modelUrl } = options;
  const temperature = options.temperature ?? 1.0;
  if (temperature <= 0) {
    throw new Error('PPO browser policy temperature must be > 0.');
  }

  return {
    async selectAction({ view, legalActions }): Promise<GameAction | undefined> {
      if (legalActions.length === 0) {
        return undefined;
      }

      const model = await loadBrowserPpoModel(modelUrl);
      const observation = encodeObservation(view);
      const actionFeatures = encodeActionCandidates(legalActions);
      const actionIndex = model.chooseActionIndex(observation, actionFeatures, temperature);
      return legalActions[actionIndex];
    },
  };
}

async function loadBrowserPpoModel(modelUrl: string): Promise<BrowserPpoModel> {
  const cached = CHECKPOINT_CACHE.get(modelUrl);
  if (cached) {
    return cached;
  }

  const loadPromise = (async () => {
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch PPO checkpoint ${modelUrl}: HTTP ${response.status}`);
    }
    const payload = (await response.json()) as BrowserPpoWeights;
    return BrowserPpoModel.fromPayload(payload);
  })();

  CHECKPOINT_CACHE.set(modelUrl, loadPromise);
  try {
    return await loadPromise;
  } catch (error) {
    CHECKPOINT_CACHE.delete(modelUrl);
    throw error;
  }
}

class BrowserPpoModel {
  private readonly observationDim: number;
  private readonly actionFeatureDim: number;
  private readonly hiddenDim: number;
  private readonly obsEncoder0Weight: number[][];
  private readonly obsEncoder0Bias: number[];
  private readonly obsEncoder2Weight: number[][];
  private readonly obsEncoder2Bias: number[];
  private readonly actionEncoder0Weight: number[][];
  private readonly actionEncoder0Bias: number[];
  private readonly policyHead0Weight: number[][];
  private readonly policyHead0Bias: number[];
  private readonly policyHead2WeightRow: number[];
  private readonly policyHead2Bias: number;

  private constructor(payload: BrowserPpoWeights) {
    this.observationDim = payload.observationDim;
    this.actionFeatureDim = payload.actionFeatureDim;
    this.hiddenDim = payload.hiddenDim;
    this.obsEncoder0Weight = payload.weights.obsEncoder0Weight;
    this.obsEncoder0Bias = payload.weights.obsEncoder0Bias;
    this.obsEncoder2Weight = payload.weights.obsEncoder2Weight;
    this.obsEncoder2Bias = payload.weights.obsEncoder2Bias;
    this.actionEncoder0Weight = payload.weights.actionEncoder0Weight;
    this.actionEncoder0Bias = payload.weights.actionEncoder0Bias;
    this.policyHead0Weight = payload.weights.policyHead0Weight;
    this.policyHead0Bias = payload.weights.policyHead0Bias;
    this.policyHead2WeightRow = payload.weights.policyHead2Weight[0];
    this.policyHead2Bias = payload.weights.policyHead2Bias[0];
  }

  static fromPayload(payload: BrowserPpoWeights): BrowserPpoModel {
    validatePayload(payload);
    return new BrowserPpoModel(payload);
  }

  chooseActionIndex(
    observation: readonly number[],
    actionFeatures: readonly number[][],
    temperature: number
  ): number {
    if (observation.length !== this.observationDim) {
      throw new Error(
        `Observation length mismatch. expected=${this.observationDim}, actual=${observation.length}`
      );
    }
    if (actionFeatures.length === 0) {
      throw new Error('At least one legal action is required.');
    }
    for (const features of actionFeatures) {
      if (features.length !== this.actionFeatureDim) {
        throw new Error(
          `Action feature length mismatch. expected=${this.actionFeatureDim}, actual=${features.length}`
        );
      }
    }

    const obsEmbed = this.forwardObservation(observation);
    let bestIndex = 0;
    let bestLogit = -Infinity;
    for (let index = 0; index < actionFeatures.length; index += 1) {
      const logits = this.forwardPolicyLogit(obsEmbed, actionFeatures[index]) / temperature;
      if (logits > bestLogit) {
        bestLogit = logits;
        bestIndex = index;
      }
    }
    return bestIndex;
  }

  private forwardObservation(observation: readonly number[]): number[] {
    const first = tanhVector(linear(this.obsEncoder0Weight, this.obsEncoder0Bias, observation));
    return tanhVector(linear(this.obsEncoder2Weight, this.obsEncoder2Bias, first));
  }

  private forwardPolicyLogit(obsEmbed: readonly number[], actionFeatures: readonly number[]): number {
    const actionEmbed = tanhVector(
      linear(this.actionEncoder0Weight, this.actionEncoder0Bias, actionFeatures)
    );
    const pairFeatures = new Array<number>(this.hiddenDim * 3);
    for (let i = 0; i < this.hiddenDim; i += 1) {
      pairFeatures[i] = obsEmbed[i];
      pairFeatures[this.hiddenDim + i] = actionEmbed[i];
      pairFeatures[(this.hiddenDim * 2) + i] = obsEmbed[i] * actionEmbed[i];
    }

    const hidden = tanhVector(linear(this.policyHead0Weight, this.policyHead0Bias, pairFeatures));
    return dot(this.policyHead2WeightRow, hidden) + this.policyHead2Bias;
  }
}

function validatePayload(payload: BrowserPpoWeights): void {
  if (!payload || typeof payload !== 'object') {
    throw new Error('Invalid PPO browser checkpoint payload.');
  }
  if (payload.checkpointType !== BROWSER_CHECKPOINT_TYPE) {
    throw new Error(
      `Unsupported PPO browser checkpoint type ${String(payload.checkpointType)}; expected ${BROWSER_CHECKPOINT_TYPE}.`
    );
  }
  if (payload.observationDim <= 0 || payload.actionFeatureDim <= 0 || payload.hiddenDim <= 0) {
    throw new Error('PPO browser checkpoint dimensions must be positive.');
  }
  if (payload.weights.policyHead2Weight.length !== 1) {
    throw new Error('Expected policyHead2Weight to have one row.');
  }
}

function linear(
  weight: readonly number[][],
  bias: readonly number[],
  input: readonly number[]
): number[] {
  const out = new Array<number>(weight.length);
  for (let rowIndex = 0; rowIndex < weight.length; rowIndex += 1) {
    const row = weight[rowIndex];
    if (row.length !== input.length) {
      throw new Error(
        `Linear layer input mismatch. expected=${row.length}, actual=${input.length}`
      );
    }
    let sum = bias[rowIndex] ?? 0.0;
    for (let colIndex = 0; colIndex < row.length; colIndex += 1) {
      sum += row[colIndex] * input[colIndex];
    }
    out[rowIndex] = sum;
  }
  return out;
}

function tanhVector(values: readonly number[]): number[] {
  const out = new Array<number>(values.length);
  for (let index = 0; index < values.length; index += 1) {
    out[index] = Math.tanh(values[index]);
  }
  return out;
}

function dot(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) {
    throw new Error(`Dot product dimension mismatch. left=${left.length}, right=${right.length}`);
  }
  let sum = 0.0;
  for (let index = 0; index < left.length; index += 1) {
    sum += left[index] * right[index];
  }
  return sum;
}
