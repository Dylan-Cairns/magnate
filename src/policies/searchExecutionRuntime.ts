import type { SearchWorkerExecutionMode } from './searchWorkerProtocol';

export const TD_SEARCH_EXECUTOR_QUERY_PARAMETER = 'tdSearchExecutor';

export function searchExecutionModeOverrideFromSearch(
  search: string
): SearchWorkerExecutionMode | undefined {
  const value = new URLSearchParams(search).get(
    TD_SEARCH_EXECUTOR_QUERY_PARAMETER
  );
  if (value === null || value === '') {
    return undefined;
  }
  if (value === 'legacy') {
    return 'legacy';
  }
  if (value === 'paired') {
    return 'resumable-paired-td';
  }
  throw new Error(
    `${TD_SEARCH_EXECUTOR_QUERY_PARAMETER} must be legacy or paired; received ${value}.`
  );
}

export function browserSearchExecutionModeOverride():
  | SearchWorkerExecutionMode
  | undefined {
  const location = (
    globalThis as typeof globalThis & {
      location?: { search?: string };
    }
  ).location;
  return searchExecutionModeOverrideFromSearch(location?.search ?? '');
}
