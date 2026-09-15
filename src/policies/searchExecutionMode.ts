import type { BotSpec } from './botSpec';
import type { SearchWorkerExecutionMode } from './searchWorkerProtocol';

export function validateSearchExecutionMode(
  spec: BotSpec,
  mode: unknown,
  workerCount: number
): asserts mode is SearchWorkerExecutionMode | undefined {
  if (
    mode !== undefined &&
    mode !== 'legacy' &&
    mode !== 'resumable-scalar' &&
    mode !== 'resumable-paired-td'
  ) {
    throw new Error(`Unsupported search execution mode: ${String(mode)}.`);
  }
  if (mode === undefined || mode === 'legacy') {
    return;
  }
  if (spec.kind !== 'td-root-search') {
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
  if (spec.kind !== 'td-root-search' || workerCount <= 1) {
    return undefined;
  }
  return 'resumable-paired-td';
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
