import { writeFile } from 'node:fs/promises';
import process from 'node:process';

async function main() {
  const options = parseOptions(process.argv.slice(2));
  const deadline =
    options.timeoutMs === 0
      ? Number.POSITIVE_INFINITY
      : Date.now() + options.timeoutMs;
  const target = await waitForBenchmarkTarget(
    options.port,
    options.pagePath,
    deadline
  );
  const client = await CdpClient.connect(target.webSocketDebuggerUrl);

  try {
    await client.send('Runtime.enable');
    let envelope;
    while (Date.now() < deadline) {
      const evaluation = await client.send('Runtime.evaluate', {
        expression: `(() => {
          const element = document.querySelector('#result');
          return element
            ? { status: element.dataset.status ?? 'missing', payload: element.textContent ?? '' }
            : { status: 'missing', payload: '' };
        })()`,
        returnByValue: true,
      });
      envelope = evaluation.result?.result?.value;
      if (envelope?.status === 'complete' || envelope?.status === 'failed') {
        break;
      }
      await delay(500);
    }
    if (envelope?.status !== 'complete' && envelope?.status !== 'failed') {
      throw new Error(
        `Timed out waiting for benchmark completion after ${options.timeoutMs}ms.`
      );
    }
    await writeFile(
      options.outPath,
      `${JSON.stringify(envelope, null, 2)}\n`,
      'utf8'
    );
    client.sendWithoutWaiting('Browser.close');
  } finally {
    client.close();
  }
}

function parseOptions(args) {
  const values = new Map();
  for (let index = 0; index < args.length; index += 2) {
    values.set(args[index], args[index + 1]);
  }
  const port = Number(values.get('--port'));
  const timeoutMs = Number(values.get('--timeout-ms'));
  const outPath = values.get('--out');
  const pagePath = values.get('--page-path') ?? '/benchmarks/td-two-lane.html';
  if (!Number.isInteger(port) || port <= 0) {
    throw new Error('--port must be a positive integer.');
  }
  if (!Number.isInteger(timeoutMs) || timeoutMs < 0) {
    throw new Error(
      '--timeout-ms must be zero (disabled) or a positive integer.'
    );
  }
  if (!outPath) {
    throw new Error('--out is required.');
  }
  return { port, timeoutMs, outPath, pagePath };
}

async function waitForBenchmarkTarget(port, pagePath, targetDeadline) {
  const listUrl = `http://127.0.0.1:${port}/json/list`;
  while (Date.now() < targetDeadline) {
    try {
      const response = await globalThis.fetch(listUrl);
      if (response.ok) {
        const targets = await response.json();
        const target = targets.find(
          (entry) =>
            entry.type === 'page' &&
            entry.url.includes(pagePath) &&
            typeof entry.webSocketDebuggerUrl === 'string'
        );
        if (target) {
          return target;
        }
      }
    } catch {
      // Edge may not have opened the debugging endpoint yet.
    }
    await delay(250);
  }
  throw new Error(
    `Timed out waiting for Edge debugging target on port ${port}.`
  );
}

class CdpClient {
  static async connect(url) {
    const socket = new globalThis.WebSocket(url);
    await new Promise((resolve, reject) => {
      socket.addEventListener('open', resolve, { once: true });
      socket.addEventListener(
        'error',
        () =>
          reject(
            new Error(`Could not connect to browser debugging target ${url}.`)
          ),
        { once: true }
      );
    });
    return new CdpClient(socket);
  }

  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    socket.addEventListener('message', (event) => {
      const message = JSON.parse(String(event.data));
      if (typeof message.id !== 'number') {
        return;
      }
      const pending = this.pending.get(message.id);
      if (!pending) {
        return;
      }
      this.pending.delete(message.id);
      if (message.error) {
        pending.reject(
          new Error(
            message.error.message ?? 'Browser debugging command failed.'
          )
        );
      } else {
        pending.resolve(message);
      }
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  sendWithoutWaiting(method, params = {}) {
    const id = this.nextId++;
    this.socket.send(JSON.stringify({ id, method, params }));
  }

  close() {
    this.socket.close();
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => globalThis.setTimeout(resolve, milliseconds));
}

await main();
