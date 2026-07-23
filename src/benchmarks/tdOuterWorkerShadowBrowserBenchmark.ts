interface OuterShadowOptions {
  mode: 'smoke' | 'full';
  states: number;
  repetitions: number;
  warmupDecisions: number;
  shadowGames: number;
  minimumSpeedup: number;
}

interface BenchmarkWorkerResponse {
  ok: boolean;
  result?: unknown;
  error?: string;
}

const resultElement = document.querySelector<HTMLPreElement>('#result');
if (!resultElement) {
  throw new Error('TD outer-worker shadow result element is missing.');
}

try {
  const options = parseOptions(new URLSearchParams(window.location.search));
  const result = await runWorker(options);
  resultElement.dataset.status = 'complete';
  resultElement.textContent = JSON.stringify(result, null, 2);
  document.title = 'Magnate TD outer-worker shadow benchmark complete';
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
  document.title = 'Magnate TD outer-worker shadow benchmark failed';
}

function parseOptions(params: URLSearchParams): OuterShadowOptions {
  const mode = params.get('mode') === 'smoke' ? 'smoke' : 'full';
  const defaults =
    mode === 'smoke'
      ? {
          states: 1,
          repetitions: 1,
          warmupDecisions: 1,
          shadowGames: 0,
        }
      : {
          states: 64,
          repetitions: 2,
          warmupDecisions: 2,
          shadowGames: 4,
        };
  return {
    mode,
    states: integerAtLeast(params.get('states'), defaults.states, 'states', 1),
    repetitions: integerAtLeast(
      params.get('repetitions'),
      defaults.repetitions,
      'repetitions',
      1
    ),
    warmupDecisions: integerAtLeast(
      params.get('warmupDecisions'),
      defaults.warmupDecisions,
      'warmupDecisions',
      0
    ),
    shadowGames: integerAtLeast(
      params.get('shadowGames'),
      defaults.shadowGames,
      'shadowGames',
      0
    ),
    minimumSpeedup: positiveNumber(
      params.get('minimumSpeedup'),
      1.2,
      'minimumSpeedup'
    ),
  };
}

function runWorker(options: OuterShadowOptions): Promise<unknown> {
  const worker = new Worker(
    new URL('./tdOuterWorkerShadowBenchmarkWorker.ts', import.meta.url),
    { type: 'module' }
  );
  return new Promise((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<BenchmarkWorkerResponse>) => {
      worker.terminate();
      if (event.data.ok) {
        resolve(event.data.result);
      } else {
        reject(
          new Error(event.data.error ?? 'Shadow benchmark worker failed.')
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

function integerAtLeast(
  value: string | null,
  fallback: number,
  label: string,
  minimum: number
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
