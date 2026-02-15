import readline from 'node:readline';

import { MagnateBridgeRuntime } from './runtime';

const runtime = new MagnateBridgeRuntime();

const io = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
  terminal: false,
});

io.on('line', (line: string) => {
  const trimmed = line.trim();
  if (trimmed === '') {
    return;
  }

  let request: unknown;
  try {
    request = JSON.parse(trimmed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stdout.write(
      JSON.stringify({
        requestId: 'unknown',
        ok: false,
        error: {
          code: 'INVALID_PAYLOAD',
          message: `Invalid JSON request: ${message}`,
        },
      }) + '\n'
    );
    return;
  }

  const response = runtime.handleRequest(request);
  process.stdout.write(JSON.stringify(response) + '\n');
});

io.on('close', () => {
  process.exit(0);
});
