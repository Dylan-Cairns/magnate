import { createReadStream } from 'node:fs';
import { readdir } from 'node:fs/promises';
import path from 'node:path';
import { createInterface } from 'node:readline';

import { ACTION_IDS } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import type { LoadedTdGuidanceModel } from '../policies/tdGuidanceModel';
import {
  ACTION_DISTRICT_ID_FEATURE_INDEX,
  ACTION_FEATURE_DIM,
  ACTION_HAS_DISTRICT_FEATURE_INDEX,
  OBSERVATION_DIM,
  OBSERVATION_DISTRICT_COUNT,
  OBSERVATION_DISTRICT_FEATURE_DIM,
  OBSERVATION_GLOBAL_FEATURE_DIM,
} from '../policies/trainingEncoding';
import type { TdReplayOpponentSamplePayload } from './types';

export const TD_SYMMETRY_AUDIT_SCHEMA_VERSION = 1 as const;
export const TD_SYMMETRY_PERMUTATION_SCHEME =
  'pawn-district-s4-d3-fixed-v1' as const;
export const PAWN_DISTRICT_NUMBERS = [1, 2, 4, 5] as const;
const IDENTITY_DESTINATION_BY_SOURCE = [0, 1, 2, 3, 4, 5] as const;
const EPSILON = 1e-9;

export interface PawnDistrictPermutation {
  readonly id: string;
  readonly destinationBySource: readonly number[];
}

export interface SymmetryReplaySample {
  readonly sourcePath: string;
  readonly sourceLine: number;
  readonly row: TdReplayOpponentSamplePayload;
}

export interface SymmetryReplaySampleSet {
  readonly files: readonly string[];
  readonly rowsScanned: number;
  readonly samples: readonly SymmetryReplaySample[];
}

export interface TdSymmetryAuditOptions {
  readonly samples: readonly SymmetryReplaySample[];
  readonly rowsScanned: number;
  readonly sourceFiles: readonly string[];
  readonly samplingSeed: string;
  readonly requestedSampleSize: number;
  readonly modelPackId: string;
  readonly modelIndexPath: string;
  readonly model: LoadedTdGuidanceModel;
  readonly worstCaseLimit?: number;
  readonly onProgress?: (progress: TdSymmetryAuditProgress) => void;
}

export interface TdSymmetryAuditProgress {
  readonly completedSamples: number;
  readonly totalSamples: number;
}

export interface TdSymmetryMetricSummary {
  readonly mean: number;
  readonly max: number;
}

export interface TdSymmetryStratum {
  readonly samples: number;
  readonly comparisons: number;
  readonly topActionMatchRate: number;
  readonly meanJensenShannonDivergence: number;
  readonly meanMaxProbabilityDelta: number;
}

export interface TdSymmetrySlotEffect {
  readonly districtId: 'D1' | 'D2' | 'D4' | 'D5';
  readonly observations: number;
  readonly meanProbability: number;
  readonly meanCenteredLogit: number;
  readonly sourceFilesWithObservations: number;
  readonly sourceFilesUniquelyHighestCenteredLogit: number;
}

export interface TdSymmetryWorstCase {
  readonly sourcePath: string;
  readonly sourceLine: number;
  readonly permutationId: string;
  readonly candidateCount: number;
  readonly baselineTopActionIndex: number;
  readonly permutedTopActionIndex: number;
  readonly baselineTopActionType: string;
  readonly permutedTopActionType: string;
  readonly baselineTopProbabilityMargin: number;
  readonly jensenShannonDivergence: number;
  readonly maxProbabilityDelta: number;
  readonly maxCenteredLogitDelta: number;
  readonly valueAbsoluteDelta: number;
}

export interface TdSymmetryAuditRun {
  readonly schemaVersion: typeof TD_SYMMETRY_AUDIT_SCHEMA_VERSION;
  readonly permutationScheme: typeof TD_SYMMETRY_PERMUTATION_SCHEME;
  readonly model: {
    readonly packId: string;
    readonly indexPath: string;
  };
  readonly replay: {
    readonly sourceFiles: readonly string[];
    readonly rowsScanned: number;
    readonly samplingSeed: string;
    readonly requestedSampleSize: number;
    readonly sampledRows: number;
  };
  readonly permutations: readonly string[];
  readonly aggregate: {
    readonly samples: number;
    readonly samplesWithAnyTopActionFlip: number;
    readonly candidateRows: number;
    readonly nonIdentityComparisons: number;
    readonly topActionMatchRate: number;
    readonly jensenShannonDivergence: TdSymmetryMetricSummary;
    readonly maxProbabilityDelta: TdSymmetryMetricSummary;
    readonly maxCenteredLogitDelta: TdSymmetryMetricSummary;
    readonly valueAbsoluteDelta: TdSymmetryMetricSummary;
  };
  readonly byBaselineTopActionType: Readonly<Record<string, TdSymmetryStratum>>;
  readonly byBaselineTopProbabilityMargin: Readonly<
    Record<string, TdSymmetryStratum>
  >;
  readonly byDistrictActionAvailability: Readonly<
    Record<string, TdSymmetryStratum>
  >;
  readonly byPermutation: Readonly<Record<string, TdSymmetryStratum>>;
  readonly pawnSlotEffects: readonly TdSymmetrySlotEffect[];
  readonly worstCases: readonly TdSymmetryWorstCase[];
}

interface MetricAccumulator {
  count: number;
  sum: number;
  max: number;
}

interface StratumAccumulator {
  samples: Set<string>;
  comparisons: number;
  topMatches: number;
  jsSum: number;
  probabilityDeltaSum: number;
}

interface SlotAccumulator {
  count: number;
  probabilitySum: number;
  centeredLogitSum: number;
}

export function createPawnDistrictPermutations(): PawnDistrictPermutation[] {
  return permutations([...PAWN_DISTRICT_NUMBERS]).map((destinations) => {
    const destinationBySource: number[] = [...IDENTITY_DESTINATION_BY_SOURCE];
    PAWN_DISTRICT_NUMBERS.forEach((source, index) => {
      destinationBySource[source] = destinations[index];
    });
    return {
      id: PAWN_DISTRICT_NUMBERS.map(
        (source) => `D${String(source)}>D${String(destinationBySource[source])}`
      ).join(','),
      destinationBySource,
    };
  });
}

export function inversePawnDistrictPermutation(
  permutation: PawnDistrictPermutation
): PawnDistrictPermutation {
  validatePermutation(permutation);
  const destinationBySource = [...IDENTITY_DESTINATION_BY_SOURCE];
  for (const source of PAWN_DISTRICT_NUMBERS) {
    destinationBySource[permutation.destinationBySource[source]] = source;
  }
  return {
    id: PAWN_DISTRICT_NUMBERS.map(
      (source) => `D${String(source)}>D${String(destinationBySource[source])}`
    ).join(','),
    destinationBySource,
  };
}

export function permuteEncodedObservation(
  observation: readonly number[],
  permutation: PawnDistrictPermutation
): number[] {
  validateObservation(observation);
  validatePermutation(permutation);
  const result = [...observation];
  for (let source = 1; source <= OBSERVATION_DISTRICT_COUNT; source += 1) {
    const destination = permutation.destinationBySource[source];
    const sourceOffset =
      OBSERVATION_GLOBAL_FEATURE_DIM +
      (source - 1) * OBSERVATION_DISTRICT_FEATURE_DIM;
    const destinationOffset =
      OBSERVATION_GLOBAL_FEATURE_DIM +
      (destination - 1) * OBSERVATION_DISTRICT_FEATURE_DIM;
    for (
      let offset = 0;
      offset < OBSERVATION_DISTRICT_FEATURE_DIM;
      offset += 1
    ) {
      result[destinationOffset + offset] = observation[sourceOffset + offset];
    }
  }
  return result;
}

export function permuteEncodedActionFeatures(
  actionFeatures: readonly number[],
  permutation: PawnDistrictPermutation
): number[] {
  validateActionFeatures(actionFeatures);
  validatePermutation(permutation);
  const result = [...actionFeatures];
  if (approximatelyEqual(result[ACTION_HAS_DISTRICT_FEATURE_INDEX], 0)) {
    return result;
  }
  if (!approximatelyEqual(result[ACTION_HAS_DISTRICT_FEATURE_INDEX], 1)) {
    throw new Error('Action has-district feature must be encoded as 0 or 1.');
  }
  const source = Math.round(
    result[ACTION_DISTRICT_ID_FEATURE_INDEX] * OBSERVATION_DISTRICT_COUNT
  );
  if (
    source < 1 ||
    source > OBSERVATION_DISTRICT_COUNT ||
    !approximatelyEqual(
      result[ACTION_DISTRICT_ID_FEATURE_INDEX],
      source / OBSERVATION_DISTRICT_COUNT
    )
  ) {
    throw new Error('Action district feature does not encode D1-D5.');
  }
  result[ACTION_DISTRICT_ID_FEATURE_INDEX] =
    permutation.destinationBySource[source] / OBSERVATION_DISTRICT_COUNT;
  return result;
}

export function permuteOpponentSample(
  row: TdReplayOpponentSamplePayload,
  permutation: PawnDistrictPermutation
): TdReplayOpponentSamplePayload {
  validateOpponentSample(row);
  return {
    ...structuredClone(row),
    observation: permuteEncodedObservation(row.observation, permutation),
    actionFeatures: row.actionFeatures.map((features) =>
      permuteEncodedActionFeatures(features, permutation)
    ),
  };
}

export async function sampleOpponentReplayDirectory(
  replayDirectory: string,
  sampleSize: number,
  samplingSeed: string
): Promise<SymmetryReplaySampleSet> {
  if (!Number.isSafeInteger(sampleSize) || sampleSize <= 0) {
    throw new Error(
      'TD symmetry audit sample size must be a positive integer.'
    );
  }
  if (samplingSeed.trim() === '') {
    throw new Error('TD symmetry audit sampling seed must be non-empty.');
  }
  const files = (await listOpponentReplayFiles(replayDirectory)).sort((a, b) =>
    a.localeCompare(b)
  );
  if (files.length === 0) {
    throw new Error(
      `TD symmetry audit found no *.opponent.jsonl files under ${replayDirectory}.`
    );
  }
  const random = rngFromSeed(samplingSeed);
  const samples: SymmetryReplaySample[] = [];
  let rowsScanned = 0;
  for (const file of files) {
    const lines = createInterface({
      input: createReadStream(file, { encoding: 'utf8' }),
      crlfDelay: Number.POSITIVE_INFINITY,
    });
    let sourceLine = 0;
    for await (const line of lines) {
      sourceLine += 1;
      if (line.trim() === '') {
        continue;
      }
      const row = JSON.parse(line) as TdReplayOpponentSamplePayload;
      validateOpponentSample(row);
      const sample = { sourcePath: file, sourceLine, row };
      rowsScanned += 1;
      if (samples.length < sampleSize) {
        samples.push(sample);
        continue;
      }
      const replacementIndex = Math.floor(random() * rowsScanned);
      if (replacementIndex < sampleSize) {
        samples[replacementIndex] = sample;
      }
    }
  }
  samples.sort(
    (left, right) =>
      left.sourcePath.localeCompare(right.sourcePath) ||
      left.sourceLine - right.sourceLine
  );
  return { files, rowsScanned, samples };
}

export function runTdSymmetryAudit(
  options: TdSymmetryAuditOptions
): TdSymmetryAuditRun {
  if (options.samples.length === 0) {
    throw new Error('TD symmetry audit requires at least one replay sample.');
  }
  const permutations = createPawnDistrictPermutations();
  const identity = permutations.find((candidate) =>
    isIdentityPermutation(candidate)
  );
  if (!identity) {
    throw new Error('TD symmetry audit is missing the identity permutation.');
  }
  const js = metricAccumulator();
  const probabilityDelta = metricAccumulator();
  const centeredLogitDelta = metricAccumulator();
  const valueDelta = metricAccumulator();
  const strata = new Map<string, StratumAccumulator>();
  const marginStrata = new Map<string, StratumAccumulator>();
  const availabilityStrata = new Map<string, StratumAccumulator>();
  const permutationStrata = new Map<string, StratumAccumulator>();
  const slots = new Map<number, SlotAccumulator>(
    PAWN_DISTRICT_NUMBERS.map((district) => [
      district,
      { count: 0, probabilitySum: 0, centeredLogitSum: 0 },
    ])
  );
  const slotsBySourceFile = new Map<string, Map<number, SlotAccumulator>>();
  const worstCases: TdSymmetryWorstCase[] = [];
  let candidateRows = 0;
  let topMatches = 0;
  let comparisons = 0;
  let samplesWithAnyTopActionFlip = 0;

  options.samples.forEach((sample, sampleIndex) => {
    validateOpponentSample(sample.row);
    const baselineLogits = Array.from(
      options.model.opponentScorer.logits(
        sample.row.observation,
        sample.row.actionFeatures
      )
    );
    const baselineProbabilities = softmax(baselineLogits);
    const baselineValue = options.model.valueScorer.predict(
      sample.row.observation
    );
    const baselineTopIndex = argmax(baselineProbabilities);
    const baselineTopProbabilityMargin = topProbabilityMargin(
      baselineProbabilities
    );
    const baselineTopType = actionType(
      sample.row.actionFeatures[baselineTopIndex]
    );
    const baselineCenteredLogits = center(baselineLogits);
    const marginStratum = probabilityMarginStratum(
      baselineTopProbabilityMargin
    );
    const availabilityStratum = districtActionAvailability(
      sample.row.actionFeatures
    );
    const sampleKey = `${sample.sourcePath}:${String(sample.sourceLine)}`;
    candidateRows += sample.row.actionFeatures.length;
    let sampleFlipped = false;

    for (const permutation of permutations) {
      const transformed = permuteOpponentSample(sample.row, permutation);
      const logits = Array.from(
        options.model.opponentScorer.logits(
          transformed.observation,
          transformed.actionFeatures
        )
      );
      const probabilities = softmax(logits);
      const centeredLogits = center(logits);
      for (let actionIndex = 0; actionIndex < logits.length; actionIndex += 1) {
        const district = actionDistrictNumber(
          transformed.actionFeatures[actionIndex]
        );
        const slot = district === null ? undefined : slots.get(district);
        if (slot && district !== null) {
          addSlotObservation(
            slot,
            probabilities[actionIndex],
            centeredLogits[actionIndex]
          );
          const sourceSlots =
            slotsBySourceFile.get(sample.sourcePath) ??
            new Map<number, SlotAccumulator>();
          const sourceSlot = sourceSlots.get(district) ?? {
            count: 0,
            probabilitySum: 0,
            centeredLogitSum: 0,
          };
          addSlotObservation(
            sourceSlot,
            probabilities[actionIndex],
            centeredLogits[actionIndex]
          );
          sourceSlots.set(district, sourceSlot);
          slotsBySourceFile.set(sample.sourcePath, sourceSlots);
        }
      }
      if (permutation === identity) {
        continue;
      }
      const topIndex = argmax(probabilities);
      const topMatch = topIndex === baselineTopIndex;
      if (!topMatch) {
        sampleFlipped = true;
      }
      const comparisonJs = jensenShannon(baselineProbabilities, probabilities);
      const comparisonProbabilityDelta = maxAbsoluteDelta(
        baselineProbabilities,
        probabilities
      );
      const comparisonCenteredLogitDelta = maxAbsoluteDelta(
        baselineCenteredLogits,
        centeredLogits
      );
      const transformedValue = options.model.valueScorer.predict(
        transformed.observation
      );
      const comparisonValueDelta = Math.abs(transformedValue - baselineValue);
      addMetric(js, comparisonJs);
      addMetric(probabilityDelta, comparisonProbabilityDelta);
      addMetric(centeredLogitDelta, comparisonCenteredLogitDelta);
      addMetric(valueDelta, comparisonValueDelta);
      comparisons += 1;
      topMatches += topMatch ? 1 : 0;
      addStratum(
        strata,
        baselineTopType,
        sampleKey,
        topMatch,
        comparisonJs,
        comparisonProbabilityDelta
      );
      addStratum(
        permutationStrata,
        permutation.id,
        sampleKey,
        topMatch,
        comparisonJs,
        comparisonProbabilityDelta
      );
      addStratum(
        marginStrata,
        marginStratum,
        sampleKey,
        topMatch,
        comparisonJs,
        comparisonProbabilityDelta
      );
      addStratum(
        availabilityStrata,
        availabilityStratum,
        sampleKey,
        topMatch,
        comparisonJs,
        comparisonProbabilityDelta
      );
      retainWorstCase(
        worstCases,
        {
          sourcePath: sample.sourcePath,
          sourceLine: sample.sourceLine,
          permutationId: permutation.id,
          candidateCount: sample.row.actionFeatures.length,
          baselineTopActionIndex: baselineTopIndex,
          permutedTopActionIndex: topIndex,
          baselineTopActionType: baselineTopType,
          permutedTopActionType: actionType(
            transformed.actionFeatures[topIndex]
          ),
          baselineTopProbabilityMargin,
          jensenShannonDivergence: comparisonJs,
          maxProbabilityDelta: comparisonProbabilityDelta,
          maxCenteredLogitDelta: comparisonCenteredLogitDelta,
          valueAbsoluteDelta: comparisonValueDelta,
        },
        options.worstCaseLimit ?? 25
      );
    }
    if (sampleFlipped) {
      samplesWithAnyTopActionFlip += 1;
    }
    if (
      options.onProgress &&
      ((sampleIndex + 1) % 25 === 0 ||
        sampleIndex + 1 === options.samples.length)
    ) {
      options.onProgress({
        completedSamples: sampleIndex + 1,
        totalSamples: options.samples.length,
      });
    }
  });

  const worstCaseLimit = options.worstCaseLimit ?? 25;
  worstCases.sort(
    (left, right) =>
      right.jensenShannonDivergence - left.jensenShannonDivergence ||
      right.maxProbabilityDelta - left.maxProbabilityDelta ||
      left.sourcePath.localeCompare(right.sourcePath) ||
      left.sourceLine - right.sourceLine ||
      left.permutationId.localeCompare(right.permutationId)
  );

  return {
    schemaVersion: TD_SYMMETRY_AUDIT_SCHEMA_VERSION,
    permutationScheme: TD_SYMMETRY_PERMUTATION_SCHEME,
    model: { packId: options.modelPackId, indexPath: options.modelIndexPath },
    replay: {
      sourceFiles: [...options.sourceFiles],
      rowsScanned: options.rowsScanned,
      samplingSeed: options.samplingSeed,
      requestedSampleSize: options.requestedSampleSize,
      sampledRows: options.samples.length,
    },
    permutations: permutations.map((permutation) => permutation.id),
    aggregate: {
      samples: options.samples.length,
      samplesWithAnyTopActionFlip,
      candidateRows,
      nonIdentityComparisons: comparisons,
      topActionMatchRate: safeDivide(topMatches, comparisons),
      jensenShannonDivergence: summarizeMetric(js),
      maxProbabilityDelta: summarizeMetric(probabilityDelta),
      maxCenteredLogitDelta: summarizeMetric(centeredLogitDelta),
      valueAbsoluteDelta: summarizeMetric(valueDelta),
    },
    byBaselineTopActionType: renderStrata(strata),
    byBaselineTopProbabilityMargin: renderStrata(marginStrata),
    byDistrictActionAvailability: renderStrata(availabilityStrata),
    byPermutation: renderStrata(permutationStrata),
    pawnSlotEffects: PAWN_DISTRICT_NUMBERS.map((district) => {
      const slot = slots.get(district);
      if (!slot) {
        throw new Error(
          `Missing symmetry slot accumulator D${String(district)}.`
        );
      }
      return {
        districtId:
          `D${String(district)}` as TdSymmetrySlotEffect['districtId'],
        observations: slot.count,
        meanProbability: safeDivide(slot.probabilitySum, slot.count),
        meanCenteredLogit: safeDivide(slot.centeredLogitSum, slot.count),
        sourceFilesWithObservations: countSourceFilesWithSlot(
          slotsBySourceFile,
          district
        ),
        sourceFilesUniquelyHighestCenteredLogit:
          countSourceFilesWithUniqueHighestSlot(slotsBySourceFile, district),
      };
    }),
    worstCases: worstCases.slice(0, worstCaseLimit),
  };
}

function addSlotObservation(
  accumulator: SlotAccumulator,
  probability: number,
  centeredLogit: number
): void {
  accumulator.count += 1;
  accumulator.probabilitySum += probability;
  accumulator.centeredLogitSum += centeredLogit;
}

function countSourceFilesWithSlot(
  slotsBySourceFile: ReadonlyMap<string, ReadonlyMap<number, SlotAccumulator>>,
  district: number
): number {
  return [...slotsBySourceFile.values()].filter(
    (slots) => (slots.get(district)?.count ?? 0) > 0
  ).length;
}

function countSourceFilesWithUniqueHighestSlot(
  slotsBySourceFile: ReadonlyMap<string, ReadonlyMap<number, SlotAccumulator>>,
  district: number
): number {
  let count = 0;
  for (const slots of slotsBySourceFile.values()) {
    const means = PAWN_DISTRICT_NUMBERS.map((candidate) => {
      const slot = slots.get(candidate);
      return {
        district: candidate,
        mean: slot
          ? safeDivide(slot.centeredLogitSum, slot.count)
          : Number.NEGATIVE_INFINITY,
      };
    });
    const best = Math.max(...means.map((entry) => entry.mean));
    const winners = means.filter(
      (entry) =>
        Number.isFinite(entry.mean) && Math.abs(entry.mean - best) <= EPSILON
    );
    if (winners.length === 1 && winners[0]?.district === district) {
      count += 1;
    }
  }
  return count;
}

function retainWorstCase(
  cases: TdSymmetryWorstCase[],
  candidate: TdSymmetryWorstCase,
  limit: number
): void {
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new Error('TD symmetry audit worst-case limit must be nonnegative.');
  }
  if (limit === 0) {
    return;
  }
  if (cases.length < limit) {
    cases.push(candidate);
    return;
  }
  let leastIndex = 0;
  for (let index = 1; index < cases.length; index += 1) {
    if (compareWorstCases(cases[index], cases[leastIndex]) > 0) {
      leastIndex = index;
    }
  }
  if (compareWorstCases(candidate, cases[leastIndex]) < 0) {
    cases[leastIndex] = candidate;
  }
}

function compareWorstCases(
  left: TdSymmetryWorstCase,
  right: TdSymmetryWorstCase
): number {
  return (
    right.jensenShannonDivergence - left.jensenShannonDivergence ||
    right.maxProbabilityDelta - left.maxProbabilityDelta ||
    left.sourcePath.localeCompare(right.sourcePath) ||
    left.sourceLine - right.sourceLine ||
    left.permutationId.localeCompare(right.permutationId)
  );
}

function validatePermutation(permutation: PawnDistrictPermutation): void {
  if (permutation.destinationBySource.length !== 6) {
    throw new Error('Pawn district permutation must index D1-D5.');
  }
  if (permutation.destinationBySource[3] !== 3) {
    throw new Error('Pawn district permutation must keep D3 fixed.');
  }
  const destinations = PAWN_DISTRICT_NUMBERS.map(
    (source) => permutation.destinationBySource[source]
  );
  if (
    [...destinations].sort((left, right) => left - right).join(',') !==
    PAWN_DISTRICT_NUMBERS.join(',')
  ) {
    throw new Error(
      'Pawn district permutation must be a bijection over D1,D2,D4,D5.'
    );
  }
}

function isIdentityPermutation(permutation: PawnDistrictPermutation): boolean {
  return permutation.destinationBySource.every(
    (destination, source) => destination === source
  );
}

function permutations(values: readonly number[]): number[][] {
  if (values.length === 0) {
    return [[]];
  }
  return values.flatMap((value, index) =>
    permutations([...values.slice(0, index), ...values.slice(index + 1)]).map(
      (tail) => [value, ...tail]
    )
  );
}

async function listOpponentReplayFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const entryPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        return listOpponentReplayFiles(entryPath);
      }
      return entry.isFile() && entry.name.endsWith('.opponent.jsonl')
        ? [entryPath]
        : [];
    })
  );
  return nested.flat();
}

function validateOpponentSample(row: TdReplayOpponentSamplePayload): void {
  validateObservation(row.observation);
  if (row.actionFeatures.length === 0) {
    throw new Error('Opponent replay sample must contain action features.');
  }
  if (row.actionFeatures.length !== row.actionProbs.length) {
    throw new Error(
      'Opponent replay sample actionFeatures/actionProbs mismatch.'
    );
  }
  if (
    !Number.isSafeInteger(row.actionIndex) ||
    row.actionIndex < 0 ||
    row.actionIndex >= row.actionFeatures.length
  ) {
    throw new Error('Opponent replay sample actionIndex is out of bounds.');
  }
  row.actionFeatures.forEach(validateActionFeatures);
}

function validateObservation(observation: readonly number[]): void {
  if (observation.length !== OBSERVATION_DIM) {
    throw new Error(
      `Symmetry observation length mismatch. expected=${String(OBSERVATION_DIM)} actual=${String(observation.length)}.`
    );
  }
  if (
    OBSERVATION_GLOBAL_FEATURE_DIM +
      OBSERVATION_DISTRICT_COUNT * OBSERVATION_DISTRICT_FEATURE_DIM !==
    OBSERVATION_DIM
  ) {
    throw new Error(
      'Symmetry observation layout constants do not match encoding dimension.'
    );
  }
  if (!observation.every(Number.isFinite)) {
    throw new Error('Symmetry observation must contain finite values.');
  }
}

function validateActionFeatures(actionFeatures: readonly number[]): void {
  if (actionFeatures.length !== ACTION_FEATURE_DIM) {
    throw new Error(
      `Symmetry action feature length mismatch. expected=${String(ACTION_FEATURE_DIM)} actual=${String(actionFeatures.length)}.`
    );
  }
  if (!actionFeatures.every(Number.isFinite)) {
    throw new Error('Symmetry action features must contain finite values.');
  }
}

function actionDistrictNumber(features: readonly number[]): number | null {
  if (!approximatelyEqual(features[ACTION_HAS_DISTRICT_FEATURE_INDEX], 1)) {
    return null;
  }
  const district = Math.round(
    features[ACTION_DISTRICT_ID_FEATURE_INDEX] * OBSERVATION_DISTRICT_COUNT
  );
  return PAWN_DISTRICT_NUMBERS.includes(
    district as (typeof PAWN_DISTRICT_NUMBERS)[number]
  )
    ? district
    : null;
}

function actionType(features: readonly number[]): string {
  let bestIndex = 0;
  for (let index = 1; index < ACTION_IDS.length; index += 1) {
    if (features[index] > features[bestIndex]) {
      bestIndex = index;
    }
  }
  return ACTION_IDS[bestIndex] ?? 'unknown';
}

function districtActionAvailability(
  actionFeatures: readonly (readonly number[])[]
): string {
  const districts = actionFeatures
    .map(actionDistrictNumber)
    .filter((district): district is number => district !== null);
  if (districts.length === 0) {
    return 'no-pawn-district-actions';
  }
  return 'has-pawn-district-actions';
}

function softmax(logits: readonly number[]): number[] {
  if (logits.length === 0) {
    throw new Error('Symmetry audit cannot softmax empty logits.');
  }
  const max = Math.max(...logits);
  const exponentials = logits.map((logit) => Math.exp(logit - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error('Symmetry audit logits produced invalid softmax total.');
  }
  return exponentials.map((value) => value / total);
}

function center(values: readonly number[]): number[] {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.map((value) => value - mean);
}

function argmax(values: readonly number[]): number {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) {
      bestIndex = index;
    }
  }
  return bestIndex;
}

function topProbabilityMargin(values: readonly number[]): number {
  if (values.length < 2) {
    return 1;
  }
  let highest = Number.NEGATIVE_INFINITY;
  let second = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (value > highest) {
      second = highest;
      highest = value;
    } else if (value > second) {
      second = value;
    }
  }
  return highest - second;
}

function probabilityMarginStratum(margin: number): string {
  if (margin < 0.01) {
    return '[0,0.01)';
  }
  if (margin < 0.05) {
    return '[0.01,0.05)';
  }
  if (margin < 0.1) {
    return '[0.05,0.10)';
  }
  return '[0.10,1]';
}

function jensenShannon(
  left: readonly number[],
  right: readonly number[]
): number {
  if (left.length !== right.length) {
    throw new Error('Jensen-Shannon inputs must have equal length.');
  }
  let divergence = 0;
  for (let index = 0; index < left.length; index += 1) {
    const midpoint = (left[index] + right[index]) / 2;
    if (left[index] > 0) {
      divergence += 0.5 * left[index] * Math.log(left[index] / midpoint);
    }
    if (right[index] > 0) {
      divergence += 0.5 * right[index] * Math.log(right[index] / midpoint);
    }
  }
  return divergence;
}

function maxAbsoluteDelta(
  left: readonly number[],
  right: readonly number[]
): number {
  if (left.length !== right.length) {
    throw new Error('Metric inputs must have equal length.');
  }
  let max = 0;
  for (let index = 0; index < left.length; index += 1) {
    max = Math.max(max, Math.abs(left[index] - right[index]));
  }
  return max;
}

function metricAccumulator(): MetricAccumulator {
  return { count: 0, sum: 0, max: 0 };
}

function addMetric(accumulator: MetricAccumulator, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(
      `Symmetry metric must be finite and nonnegative: ${String(value)}.`
    );
  }
  accumulator.count += 1;
  accumulator.sum += value;
  accumulator.max = Math.max(accumulator.max, value);
}

function summarizeMetric(
  accumulator: MetricAccumulator
): TdSymmetryMetricSummary {
  return {
    mean: safeDivide(accumulator.sum, accumulator.count),
    max: accumulator.max,
  };
}

function addStratum(
  strata: Map<string, StratumAccumulator>,
  key: string,
  sampleKey: string,
  topMatch: boolean,
  js: number,
  probabilityDelta: number
): void {
  const accumulator = strata.get(key) ?? {
    samples: new Set<string>(),
    comparisons: 0,
    topMatches: 0,
    jsSum: 0,
    probabilityDeltaSum: 0,
  };
  accumulator.samples.add(sampleKey);
  accumulator.comparisons += 1;
  accumulator.topMatches += topMatch ? 1 : 0;
  accumulator.jsSum += js;
  accumulator.probabilityDeltaSum += probabilityDelta;
  strata.set(key, accumulator);
}

function renderStrata(
  strata: ReadonlyMap<string, StratumAccumulator>
): Record<string, TdSymmetryStratum> {
  return Object.fromEntries(
    [...strata.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, value]) => [
        key,
        {
          samples: value.samples.size,
          comparisons: value.comparisons,
          topActionMatchRate: safeDivide(value.topMatches, value.comparisons),
          meanJensenShannonDivergence: safeDivide(
            value.jsSum,
            value.comparisons
          ),
          meanMaxProbabilityDelta: safeDivide(
            value.probabilityDeltaSum,
            value.comparisons
          ),
        },
      ])
  );
}

function safeDivide(numerator: number, denominator: number): number {
  return denominator > 0 ? numerator / denominator : 0;
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= EPSILON;
}
