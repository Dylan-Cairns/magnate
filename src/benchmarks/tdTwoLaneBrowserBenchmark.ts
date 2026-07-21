interface BenchmarkOptions {
  mode: 'smoke' | 'full';
  states: number;
  rounds: number;
  warmupRounds: number;
  minimumSpeedup: number;
}

interface BenchmarkWorkerResponse {
  ok: boolean;
  result?: unknown;
  error?: string;
}

const resultElement = document.querySelector<HTMLPreElement>('#result');
if (!resultElement) {
  throw new Error('TD two-lane benchmark result element is missing.');
}

try {
  const options = parseOptions(new URLSearchParams(window.location.search));
  const result = await runWorker(options);
  resultElement.dataset.status = 'complete';
  resultElement.textContent = JSON.stringify(result, null, 2);
  document.title = 'Magnate TD two-lane benchmark complete';
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
  document.title = 'Magnate TD two-lane benchmark failed';
}

function parseOptions(params: URLSearchParams): BenchmarkOptions {
  const mode = params.get('mode') === 'smoke' ? 'smoke' : 'full';
  const defaults =
    mode === 'smoke'
      ? { states: 8, rounds: 1, warmupRounds: 1 }
      : { states: 128, rounds: 500, warmupRounds: 5 };
  const states = positiveInteger(
    params.get('states'),
    defaults.states,
    'states'
  );
  if (states % 2 !== 0) {
    throw new Error('TD two-lane benchmark states must be even.');
  }
  return {
    mode,
    states,
    rounds: positiveInteger(params.get('rounds'), defaults.rounds, 'rounds'),
    warmupRounds: positiveInteger(
      params.get('warmupRounds'),
      defaults.warmupRounds,
      'warmupRounds'
    ),
    minimumSpeedup: positiveNumber(
      params.get('minimumSpeedup'),
      1.3,
      'minimumSpeedup'
    ),
  };
}

function runWorker(options: BenchmarkOptions): Promise<unknown> {
  const worker = new Worker(
    new URL('./tdTwoLaneBenchmarkWorker.ts', import.meta.url),
    { type: 'module' }
  );
  return new Promise((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<BenchmarkWorkerResponse>) => {
      worker.terminate();
      if (event.data.ok) {
        resolve(event.data.result);
      } else {
        reject(new Error(event.data.error ?? 'Benchmark worker failed.'));
      }
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message));
    };
    worker.postMessage(options);
  });
}

function positiveInteger(
  value: string | null,
  fallback: number,
  label: string
): number {
  if (value === null) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function positiveNumber(
  value: string | null,
  fallback: number,
  label: string
): number {
  if (value === null) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive number.`);
  }
  return parsed;
}
