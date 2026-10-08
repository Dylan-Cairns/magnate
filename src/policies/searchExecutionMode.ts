import { BotKind } from './values';
import type { BotSpec } from './botSpec';
import { SearchWorkerExecutionMode } from './searchWorkerProtocol';

export function validateSearchExecutionMode(
  spec: BotSpec,
  mode: unknown,
  workerCount: number
): asserts mode is SearchWorkerExecutionMode | undefined {
  if (
    mode !== undefined &&
    mode !== SearchWorkerExecutionMode.Legacy &&
    mode !== SearchWorkerExecutionMode.ResumableScalar &&
    mode !== SearchWorkerExecutionMode.ResumablePairedTd
  ) {
    throw new Error(`Unsupported search execution mode: ${String(mode)}.`);
  }
  if (mode === undefined || mode === SearchWorkerExecutionMode.Legacy) {
    return;
  }
  if (spec.kind !== BotKind.TdRootSearch) {
    throw new Error(
      `Search execution mode ${mode} requires a TD-root search policy.`
    );
  }
  if (workerCount <= 1) {
    throw new Error(
      `Search execution mode ${mode} requires parallel search workers.`
    );
  }
}

export function resolveEffectiveSearchExecutionMode(
  spec: BotSpec,
  requestedMode: unknown,
  workerCount: number
): SearchWorkerExecutionMode | undefined {
  validateSearchExecutionMode(spec, requestedMode, workerCount);
  if (requestedMode !== undefined) {
    return requestedMode;
  }
  if (spec.kind !== BotKind.TdRootSearch || workerCount <= 1) {
    return undefined;
  }
  return SearchWorkerExecutionMode.ResumablePairedTd;
}

export function searchWorkerPoolConfigurationMatches(
  currentWorkerCount: number,
  currentMode: SearchWorkerExecutionMode | undefined,
  requestedWorkerCount: number,
  requestedMode: SearchWorkerExecutionMode | undefined
): boolean {
  return (
    currentWorkerCount === requestedWorkerCount && currentMode === requestedMode
  );
}
