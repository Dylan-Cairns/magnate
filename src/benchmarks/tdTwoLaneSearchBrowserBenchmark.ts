interface SearchBenchmarkOptions {
  mode: 'smoke' | 'full';
  states: number;
  repetitions: number;
  warmupDecisions: number;
  workers: number;
  transcriptGames: number;
  minimumSpeedup: number;
}

interface BenchmarkWorkerResponse {
  ok: boolean;
  result?: unknown;
  error?: string;
}

const resultElement = document.querySelector<HTMLPreElement>('#result');
if (!resultElement) {
  throw new Error('TD two-lane search benchmark result element is missing.');
}

try {
  const options = parseOptions(new URLSearchParams(window.location.search));
  const result = await runWorker(options);
  resultElement.dataset.status = 'complete';
  resultElement.textContent = JSON.stringify(result, null, 2);
  document.title = 'Magnate TD two-lane search benchmark complete';
} catch (error) {
  resultElement.dataset.status = 'failed';
  resultElement.textContent = JSON.stringify(
    {
      error:
        error instanceof Error ? (error.stack ?? error.message) : String(error),
    },
    null,
    2
  );
  document.title = 'Magnate TD two-lane search benchmark failed';
}

function parseOptions(params: URLSearchParams): SearchBenchmarkOptions {
  const mode = params.get('mode') === 'smoke' ? 'smoke' : 'full';
  const defaults =
    mode === 'smoke'
      ? {
          states: 1,
          repetitions: 1,
          warmupDecisions: 1,
          workers: 2,
          transcriptGames: 0,
        }
      : {
          states: 24,
          repetitions: 3,
          warmupDecisions: 2,
          workers: 8,
          transcriptGames: 1,
        };
  return {
    mode,
    states: nonnegativeInteger(
      params.get('states'),
      defaults.states,
      'states',
      1
    ),
    repetitions: nonnegativeInteger(
      params.get('repetitions'),
      defaults.repetitions,
      'repetitions',
      1
    ),
    warmupDecisions: nonnegativeInteger(
      params.get('warmupDecisions'),
      defaults.warmupDecisions,
      'warmupDecisions'
    ),
    workers: nonnegativeInteger(
      params.get('workers'),
      defaults.workers,
      'workers',
      1
    ),
    transcriptGames: nonnegativeInteger(
      params.get('transcriptGames'),
      defaults.transcriptGames,
      'transcriptGames'
    ),
    minimumSpeedup: positiveNumber(
      params.get('minimumSpeedup'),
      1.2,
      'minimumSpeedup'
    ),
  };
}

function runWorker(options: SearchBenchmarkOptions): Promise<unknown> {
  const worker = new Worker(
    new URL('./tdTwoLaneSearchBenchmarkWorker.ts', import.meta.url),
    { type: 'module' }
  );
  return new Promise((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<BenchmarkWorkerResponse>) => {
      worker.terminate();
      if (event.data.ok) {
        resolve(event.data.result);
      } else {
        reject(
          new Error(event.data.error ?? 'Search benchmark worker failed.')
        );
      }
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message));
    };
    worker.postMessage(options);
  });
}

function nonnegativeInteger(
  value: string | null,
  fallback: number,
  label: string,
  minimum = 0
): number {
  const parsed = value === null ? fallback : Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum) {
    throw new Error(`${label} must be an integer >= ${String(minimum)}.`);
  }
  return parsed;
}

function positiveNumber(
  value: string | null,
  fallback: number,
  label: string
): number {
  const parsed = value === null ? fallback : Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive number.`);
  }
  return parsed;
}
