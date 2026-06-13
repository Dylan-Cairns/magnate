import type { PlayerId, Suit } from '../../engine/types';
import type { AnimationSequence } from './animationSequence';
import type { GamePresentationEvent } from './types';

export type AnimationVisualCommand =
  | {
      type: 'pulse-tax-resources';
      startMs: number;
      endMs: number;
      targets: readonly { playerId: PlayerId; suit: Suit }[];
    }
  | {
      type: 'launch-tax-token-flights';
      atMs: number;
      losses: readonly Extract<
        GamePresentationEvent,
        { type: 'tax-token-lost' }
      >[];
    }
  | {
      type: 'launch-income-token-flights';
      atMs: number;
      gains: readonly Extract<
        GamePresentationEvent,
        { type: 'income-token-gained' }
      >[];
    };

export function deriveAnimationVisualCommands(
  sequence: AnimationSequence
): readonly AnimationVisualCommand[] {
  const commands: AnimationVisualCommand[] = [];
  const taxFlightStep = sequence.steps.find(
    (step) => step.type === 'launch-tax-token-flights'
  );
  const taxPulseStep = sequence.steps.find(
    (step) => step.type === 'pulse-tax-die'
  );
  if (taxPulseStep && taxFlightStep && taxFlightStep.losses.length > 0) {
    commands.push({
      type: 'pulse-tax-resources',
      startMs: taxPulseStep.startMs,
      endMs: taxPulseStep.endMs,
      targets: taxPulseTargets(taxFlightStep.losses),
    });
  }
  if (taxFlightStep && taxFlightStep.losses.length > 0) {
    commands.push({
      type: 'launch-tax-token-flights',
      atMs: taxFlightStep.startMs,
      losses: taxFlightStep.losses,
    });
  }

  const incomeFlightStep = sequence.steps.find(
    (step) => step.type === 'launch-income-token-flights'
  );
  if (incomeFlightStep && incomeFlightStep.gains.length > 0) {
    commands.push({
      type: 'launch-income-token-flights',
      atMs: incomeFlightStep.startMs,
      gains: incomeFlightStep.gains,
    });
  }

  return commands.sort(
    (left, right) => visualCommandStartMs(left) - visualCommandStartMs(right)
  );
}

function taxPulseTargets(
  losses: readonly Extract<GamePresentationEvent, { type: 'tax-token-lost' }>[]
): readonly { playerId: PlayerId; suit: Suit }[] {
  const targets: Array<{ playerId: PlayerId; suit: Suit }> = [];
  const seen = new Set<string>();
  for (const loss of losses) {
    const key = `${loss.playerId}:${loss.suit}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    targets.push({
      playerId: loss.playerId,
      suit: loss.suit,
    });
  }
  return targets;
}

function visualCommandStartMs(command: AnimationVisualCommand): number {
  return command.type === 'pulse-tax-resources'
    ? command.startMs
    : command.atMs;
}
