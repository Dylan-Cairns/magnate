import { rngFromSeed } from '../engine/rng';
import type { GameState } from '../engine/types';

export const POLICY_RANDOM_SCHEME_VERSION = 'state-derived-v1';

export function policyRandomSeedForState(
  state: GameState,
  policyId: string
): string {
  return `${state.seed}:bot:${policyId}:turn:${state.turn}:phase:${state.phase}:log:${state.log.length}:actor:${state.activePlayerIndex}`;
}

export function policyRandomForState(
  state: GameState,
  policyId: string
): () => number {
  return rngFromSeed(policyRandomSeedForState(state, policyId));
}
