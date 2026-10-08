import {
  AnimationVisualCommandType,
  GamePresentationEventType,
  AnimationStepType,
} from './values';
import type { CardId } from '../../engine/cards';
import type { PlayerId, Suit } from '../../engine/types';
import type { AnimationSequence } from './animationSequence';
import type { GamePresentationEvent } from './types';

export type AnimationVisualCommand =
  | {
      type: typeof AnimationVisualCommandType.LaunchDrawCardFlight;
      atMs: number;
      landingMs: number;
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      type: typeof AnimationVisualCommandType.LaunchSoldCardFlight;
      atMs: number;
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      type: typeof AnimationVisualCommandType.LaunchSellTokenFlights;
      atMs: number;
      durationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      gains: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.SellResourceGained }
      >[];
    }
  | {
      type: typeof AnimationVisualCommandType.LaunchCardToDistrictFlight;
      atMs: number;
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.CardPlayedToDistrict }
      >;
    }
  | {
      type: typeof AnimationVisualCommandType.LaunchPaymentTokenFlights;
      atMs: number;
      durationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.ResourcePaymentStarted }
      >;
    }
  | {
      type: typeof AnimationVisualCommandType.LaunchTradeTokenFlights;
      atMs: number;
      durationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.TradeResourcesApplied }
      >;
    }
  | {
      type: typeof AnimationVisualCommandType.LaunchDeedTokenFlights;
      atMs: number;
      durationMs: number;
      tokens: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.DeedTokenPaid }
      >[];
    }
  | {
      type: typeof AnimationVisualCommandType.PulseTaxResources;
      startMs: number;
      endMs: number;
      targets: readonly { playerId: PlayerId; suit: Suit }[];
    }
  | {
      type: typeof AnimationVisualCommandType.LaunchTaxTokenFlights;
      atMs: number;
      durationMs: number;
      losses: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.TaxTokenLost }
      >[];
    }
  | {
      type: typeof AnimationVisualCommandType.LaunchIncomeTokenFlights;
      atMs: number;
      durationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      gains: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.IncomeTokenGained }
      >[];
    };

export function deriveAnimationVisualCommands(
  sequence: AnimationSequence
): readonly AnimationVisualCommand[] {
  const commands: AnimationVisualCommand[] = [];
  for (const step of sequence.steps) {
    switch (step.type) {
      case AnimationStepType.DrawCardFlight:
        commands.push({
          type: AnimationVisualCommandType.LaunchDrawCardFlight,
          atMs: step.startMs,
          landingMs: step.endMs,
          playerId: step.playerId,
          cardId: step.cardId,
        });
        break;
      case AnimationStepType.StageSoldCard:
        commands.push({
          type: AnimationVisualCommandType.LaunchSoldCardFlight,
          atMs: step.startMs,
          playerId: step.playerId,
          cardId: step.cardId,
        });
        break;
      case AnimationStepType.LaunchSellTokenFlights:
        commands.push({
          type: AnimationVisualCommandType.LaunchSellTokenFlights,
          atMs: step.startMs,
          durationMs: step.flightSequenceDurationMs,
          flightDurationMs: step.flightDurationMs,
          flightStaggerMs: step.flightStaggerMs,
          gains: step.gains,
        });
        break;
      case AnimationStepType.LaunchCardToDistrictFlight:
        commands.push({
          type: AnimationVisualCommandType.LaunchCardToDistrictFlight,
          atMs: step.startMs,
          durationMs: step.durationMs,
          event: step.event,
        });
        break;
      case AnimationStepType.LaunchPaymentTokenFlights:
        commands.push({
          type: AnimationVisualCommandType.LaunchPaymentTokenFlights,
          atMs: step.startMs,
          durationMs: step.flightSequenceDurationMs,
          flightDurationMs: step.flightDurationMs,
          flightStaggerMs: step.flightStaggerMs,
          event: step.event,
        });
        break;
      case AnimationStepType.LaunchTradeTokenFlights:
        commands.push({
          type: AnimationVisualCommandType.LaunchTradeTokenFlights,
          atMs: step.startMs,
          durationMs: step.flightSequenceDurationMs,
          flightDurationMs: step.flightDurationMs,
          flightStaggerMs: step.flightStaggerMs,
          event: step.event,
        });
        break;
      case AnimationStepType.LaunchDeedTokenFlights:
        commands.push({
          type: AnimationVisualCommandType.LaunchDeedTokenFlights,
          atMs: step.startMs,
          durationMs: step.durationMs,
          tokens: step.tokens,
        });
        break;
    }
  }

  const taxFlightStep = sequence.steps.find(
    (step) => step.type === AnimationStepType.LaunchTaxTokenFlights
  );
  const taxPreFlightHoldStep = sequence.steps.find(
    (step) => step.type === AnimationStepType.HoldBeforeTaxFlights
  );
  if (
    taxPreFlightHoldStep &&
    taxFlightStep &&
    taxFlightStep.losses.length > 0
  ) {
    commands.push({
      type: AnimationVisualCommandType.PulseTaxResources,
      startMs: taxPreFlightHoldStep.startMs,
      endMs: taxPreFlightHoldStep.endMs,
      targets: taxPulseTargets(taxFlightStep.losses),
    });
  }
  if (taxFlightStep && taxFlightStep.losses.length > 0) {
    commands.push({
      type: AnimationVisualCommandType.LaunchTaxTokenFlights,
      atMs: taxFlightStep.startMs,
      durationMs: taxFlightStep.flightSequenceDurationMs,
      losses: taxFlightStep.losses,
    });
  }

  const incomeFlightStep = sequence.steps.find(
    (step) => step.type === AnimationStepType.LaunchIncomeTokenFlights
  );
  if (incomeFlightStep && incomeFlightStep.gains.length > 0) {
    commands.push({
      type: AnimationVisualCommandType.LaunchIncomeTokenFlights,
      atMs: incomeFlightStep.startMs,
      durationMs: incomeFlightStep.flightSequenceDurationMs,
      flightDurationMs: incomeFlightStep.flightDurationMs,
      flightStaggerMs: incomeFlightStep.flightStaggerMs,
      gains: incomeFlightStep.gains,
    });
  }

  return commands.sort(
    (left, right) => visualCommandStartMs(left) - visualCommandStartMs(right)
  );
}

function taxPulseTargets(
  losses: readonly Extract<
    GamePresentationEvent,
    { type: typeof GamePresentationEventType.TaxTokenLost }
  >[]
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
  return command.type === AnimationVisualCommandType.PulseTaxResources
    ? command.startMs
    : command.atMs;
}
