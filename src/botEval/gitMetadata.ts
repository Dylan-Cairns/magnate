import { execFileSync } from 'node:child_process';

import type { GitMetadata } from './types';

export function collectGitMetadata(cwd = process.cwd()): GitMetadata {
  try {
    const commit = runGit(['rev-parse', 'HEAD'], cwd);
    const status = runGit(['status', '--porcelain'], cwd);
    return {
      commit,
      dirty: status !== '',
    };
  } catch {
    return {
      commit: null,
      dirty: null,
    };
  }
}

function runGit(args: readonly string[], cwd: string): string {
  return execFileSync('git', args, {
    cwd,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim();
}
