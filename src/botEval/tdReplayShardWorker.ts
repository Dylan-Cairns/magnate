import { collectAndWriteTdReplayArtifacts } from './tdReplayArtifacts';
import type {
  TdReplayShardWorkerRequest,
  TdReplayShardWorkerResponse,
} from './tdReplayShardWorkerProtocol';

let activeShardIndex: number | undefined;

process.on('message', (request: TdReplayShardWorkerRequest) => {
  void handleRequest(request).catch((error: unknown) => {
    sendError(error, activeShardIndex);
  });
});

send({ type: 'ready' });

async function handleRequest(
  request: TdReplayShardWorkerRequest
): Promise<void> {
  switch (request.type) {
    case 'run-shard': {
      if (activeShardIndex !== undefined) {
        throw new Error('TD replay shard worker received a job while busy.');
      }
      activeShardIndex = request.shard.shardIndex;
      const written = await collectAndWriteTdReplayArtifacts(
        {
          ...request.config,
          games: request.shard.games,
        },
        request.outputDirectory,
        {
          generatedAtUtc: request.generatedAtUtc,
          git: request.git,
          nodeVersion: request.nodeVersion,
          runBaseName: shardRunBaseName(request.shard.shardIndex),
          gameIndexStart: request.shard.gameIndexStart,
          gameIndexTotal: request.gameIndexTotal,
          progressIntervalMs: request.progressIntervalMs,
          onProgress(progress) {
            send({
              type: 'progress',
              shardIndex: request.shard.shardIndex,
              progress,
            });
          },
        }
      );
      activeShardIndex = undefined;
      send({
        type: 'shard-completed',
        result: {
          shard: request.shard,
          written,
        },
      });
      return;
    }
    case 'shutdown':
      process.disconnect();
      return;
  }
}

function send(response: TdReplayShardWorkerResponse): void {
  if (!process.send) {
    throw new Error('TD replay shard worker requires an IPC channel.');
  }
  process.send(response);
}

function sendError(error: unknown, shardIndex?: number): void {
  const normalized =
    error instanceof Error ? error : new Error(String(error));
  send({
    type: 'error',
    ...(shardIndex === undefined ? {} : { shardIndex }),
    message: normalized.message,
    ...(normalized.stack ? { stack: normalized.stack } : {}),
  });
}

function shardRunBaseName(shardIndex: number): string {
  return `shard-${String(shardIndex).padStart(3, '0')}`;
}
