import { validateHeadToHeadConfig } from './matchup';
import {
  createRuntimePairBots,
  playPairedSeed,
  type RuntimePairBots,
} from './pair';
import type {
  PairWorkerRequest,
  PairWorkerResponse,
} from './pairWorkerProtocol';
import { installLocalPublicFetch } from './localPublicFetch';
import type { HeadToHeadConfig } from './types';

installLocalPublicFetch();

let config: HeadToHeadConfig | undefined;
let bots: RuntimePairBots | undefined;
let progressIntervalMs = 0;
let activePairIndex: number | undefined;

process.on('message', (request: PairWorkerRequest) => {
  void handleRequest(request).catch((error: unknown) => {
    sendError(error, activePairIndex);
  });
});

async function handleRequest(request: PairWorkerRequest): Promise<void> {
  switch (request.type) {
    case 'initialize':
      if (config) {
        throw new Error('Pair worker was initialized more than once.');
      }
      validateHeadToHeadConfig(request.config);
      config = request.config;
      bots = createRuntimePairBots(config);
      progressIntervalMs = request.progressIntervalMs;
      send({ type: 'ready' });
      return;
    case 'run-pair': {
      if (!config || !bots) {
        throw new Error('Pair worker received a job before initialization.');
      }
      if (activePairIndex !== undefined) {
        throw new Error('Pair worker received a job while already busy.');
      }
      activePairIndex = request.job.pairIndex;
      const result = await playPairedSeed({
        config,
        bots,
        job: request.job,
        progressIntervalMs,
        onHeartbeat(heartbeat) {
          send({
            type: 'heartbeat',
            pairIndex: request.job.pairIndex,
            heartbeat,
          });
        },
        onGameCompleted(game) {
          send({
            type: 'game-completed',
            pairIndex: request.job.pairIndex,
            game,
          });
        },
      });
      activePairIndex = undefined;
      send({ type: 'pair-completed', result });
      return;
    }
    case 'shutdown':
      process.disconnect();
      return;
  }
}

function send(response: PairWorkerResponse): void {
  if (!process.send) {
    throw new Error('Pair worker requires an IPC channel.');
  }
  process.send(response);
}

function sendError(error: unknown, pairIndex?: number): void {
  const normalized =
    error instanceof Error ? error : new Error(String(error));
  send({
    type: 'error',
    ...(pairIndex === undefined ? {} : { pairIndex }),
    message: normalized.message,
    ...(normalized.stack ? { stack: normalized.stack } : {}),
  });
}
