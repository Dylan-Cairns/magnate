import { PairWorkerMessageType } from './workerValues';
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
    case PairWorkerMessageType.Initialize:
      if (config) {
        throw new Error('Pair worker was initialized more than once.');
      }
      validateHeadToHeadConfig(request.config);
      config = request.config;
      bots = createRuntimePairBots(config);
      progressIntervalMs = request.progressIntervalMs;
      send({ type: PairWorkerMessageType.Ready });
      return;
    case PairWorkerMessageType.RunPair: {
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
            type: PairWorkerMessageType.Heartbeat,
            pairIndex: request.job.pairIndex,
            heartbeat,
          });
        },
        onGameCompleted(game) {
          send({
            type: PairWorkerMessageType.GameCompleted,
            pairIndex: request.job.pairIndex,
            game,
          });
        },
      });
      activePairIndex = undefined;
      send({ type: PairWorkerMessageType.PairCompleted, result });
      return;
    }
    case PairWorkerMessageType.Shutdown:
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
  const normalized = error instanceof Error ? error : new Error(String(error));
  send({
    type: PairWorkerMessageType.Error,
    ...(pairIndex === undefined ? {} : { pairIndex }),
    message: normalized.message,
    ...(normalized.stack ? { stack: normalized.stack } : {}),
  });
}
