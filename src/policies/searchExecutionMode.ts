import type { BotSpec } from './botSpec';
import type { SearchWorkerExecutionMode } from './searchWorkerProtocol';
import { resolveTdRootSearchGuidanceConfig } from './tdRootGuidanceConfig';

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
  const guidance = resolveTdRootSearchGuidanceConfig(spec.guidance);
  if (guidance.rollout !== 'td') {
    throw new Error(
      `Search execution mode ${mode} requires TD rollout guidance.`
    );
  }
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
