export interface TdGuidanceValueScorer {
  readonly observationDim: number;
  predict(observation: readonly number[]): number;
}

export interface TdGuidanceActionScorer {
  readonly observationDim: number;
  readonly actionFeatureDim: number;
  logits(
    observation: readonly number[],
    actionFeatures: readonly number[][]
  ): Float32Array;
}

export interface LoadedTdGuidanceModel {
  readonly valueScorer: TdGuidanceValueScorer;
  readonly opponentScorer: TdGuidanceActionScorer;
}
