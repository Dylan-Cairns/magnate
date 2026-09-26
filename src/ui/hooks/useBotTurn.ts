import {
  useCallback,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from 'react';

import type { GameAction, GameState, PlayerId } from '../../engine/types';
import type {
  ActionPolicy,
  SearchDecisionDiagnostics,
} from '../../policies/types';
import {
  botDecisionResultIsCurrent,
  botRandomForState,
  botRandomSeedForState,
  errorMessage,
  planBotDecision,
  resolveBotActingPlayerId,
} from '../gameControllerModel';

const DEFAULT_BOT_DELAY_MS = 450;

function logBotSearchDiagnostics(diagnostics: SearchDecisionDiagnostics): void {
  const rootActions = diagnostics.rootActions.map((entry) => ({
    actionKey: entry.actionKey,
    visits: entry.visits,
    meanValue: roundDiagnosticNumber(entry.meanValue),
    terminalRate: roundDiagnosticNumber(entry.terminalRate),
    terminalRollouts: entry.terminalRollouts,
    prior: roundDiagnosticNumber(entry.prior),
  }));
  console.info('[Magnate bot search]', {
    heuristic: diagnostics.heuristic ?? 'v1',
    stochasticSimulation: diagnostics.stochasticSimulation ?? null,
    workers: diagnostics.parallelWorkers ?? 1,
    batches: diagnostics.parallelBatches ?? null,
    batchSize: diagnostics.parallelBatchSize ?? null,
    legalRootActions: diagnostics.legalRootActions,
    expandedRootActions: diagnostics.expandedRootActions,
    rootVisits: diagnostics.rootVisitBudget,
    simulatedActionSteps: diagnostics.simulatedActionSteps,
    maxSimulatedActionSteps: diagnostics.maxSimulatedActionSteps,
    terminalRollouts: diagnostics.terminalRollouts,
    terminalRate: diagnostics.terminalRate,
    selectedActionKey: diagnostics.selectedActionKey,
    selectedActionVisits: diagnostics.selectedActionVisits,
    selectedActionMeanValue: diagnostics.selectedActionMeanValue,
    selectedActionTerminalRate: diagnostics.selectedActionTerminalRate,
    rootActions,
  });
  if (rootActions.length > 0) {
    console.table(rootActions);
  }
}

function roundDiagnosticNumber(value: number): number {
  return Number(value.toFixed(4));
}

export type UseBotTurnOptions = {
  /** Dependency key: the effect only re-runs when the canonical state changes. */
  state: GameState;
  /** Latest canonical state, read after the scheduling delay elapses. */
  stateRef: { current: GameState };
  shouldRunBot: boolean;
  botPlayerId: PlayerId;
  botProfileId: string;
  policy: ActionPolicy;
  turnDelayMs?: number;
  collectDiagnostics: boolean;
  dispatchAction: (
    sourceState: GameState,
    action: GameAction,
    actingPlayerId: PlayerId
  ) => void;
  onError: (message: string) => void;
};

export type UseBotTurnResult = {
  botThinking: boolean;
  /** Invalidate any in-flight decision and stop showing the thinking state. */
  invalidatePendingDecision: () => void;
};

/**
 * Runs the bot's async turn: schedule a delayed decision, guard it against
 * supersession, and dispatch the result through canonical action handling.
 * Generation counting lives here; `invalidatePendingDecision` is the external
 * hook for session/turn resets.
 */
export function useBotTurn({
  state,
  stateRef,
  shouldRunBot,
  botPlayerId,
  botProfileId,
  policy,
  turnDelayMs,
  collectDiagnostics,
  dispatchAction,
  onError,
}: UseBotTurnOptions): UseBotTurnResult {
  const [botThinking, setBotThinking] = useState<boolean>(false);
  const [prevShouldRunBot, setPrevShouldRunBot] = useState(false);
  const decisionGenerationRef = useRef(0);
  // The latest error handler is used without making it an effect dependency.
  const reportError = useEffectEvent(onError);

  if (shouldRunBot !== prevShouldRunBot) {
    setPrevShouldRunBot(shouldRunBot);
    setBotThinking(shouldRunBot);
  }

  const invalidatePendingDecision = useCallback(() => {
    decisionGenerationRef.current += 1;
    setBotThinking(false);
  }, []);

  useEffect(() => {
    const decisionGeneration = decisionGenerationRef.current + 1;
    decisionGenerationRef.current = decisionGeneration;
    if (!shouldRunBot) {
      return;
    }

    let cancelled = false;
    const botTurnDelayMs = turnDelayMs ?? DEFAULT_BOT_DELAY_MS;
    const timerId = window.setTimeout(() => {
      void (async () => {
        const current = stateRef.current;
        const plan = planBotDecision(current, botPlayerId);
        if (
          cancelled ||
          decisionGenerationRef.current !== decisionGeneration ||
          !plan.applicable ||
          plan.view === null
        ) {
          if (decisionGenerationRef.current === decisionGeneration) {
            setBotThinking(false);
          }
          return;
        }

        if (plan.actions.length === 0) {
          reportError('Bot has no legal actions.');
          setBotThinking(false);
          return;
        }

        let choice: GameAction | null | undefined;
        try {
          choice = await policy.selectAction({
            state: current,
            view: plan.view,
            legalActions: plan.actions,
            random: botRandomForState(current, botProfileId),
            randomSeed: botRandomSeedForState(current, botProfileId),
            ...(collectDiagnostics
              ? { onSearchDiagnostics: logBotSearchDiagnostics }
              : {}),
          });
        } catch (err) {
          if (
            botDecisionResultIsCurrent({
              cancelled,
              decisionGeneration,
              currentGeneration: decisionGenerationRef.current,
              decisionState: current,
              currentState: stateRef.current,
            })
          ) {
            reportError(`Bot action failed: ${errorMessage(err)}`);
            setBotThinking(false);
          }
          return;
        }

        if (
          !botDecisionResultIsCurrent({
            cancelled,
            decisionGeneration,
            currentGeneration: decisionGenerationRef.current,
            decisionState: current,
            currentState: stateRef.current,
          })
        ) {
          return;
        }
        if (!choice) {
          reportError('Bot policy could not select an action.');
          setBotThinking(false);
          return;
        }

        try {
          dispatchAction(
            current,
            choice,
            resolveBotActingPlayerId(choice, plan.actingPlayerId, botPlayerId)
          );
        } catch (err) {
          if (
            !cancelled &&
            decisionGenerationRef.current === decisionGeneration
          ) {
            reportError(`Bot action failed: ${errorMessage(err)}`);
          }
        } finally {
          if (
            !cancelled &&
            decisionGenerationRef.current === decisionGeneration &&
            stateRef.current === current
          ) {
            setBotThinking(false);
          }
        }
      })();
    }, botTurnDelayMs);

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [
    botPlayerId,
    botProfileId,
    collectDiagnostics,
    dispatchAction,
    policy,
    shouldRunBot,
    state,
    stateRef,
    turnDelayMs,
  ]);

  return { botThinking, invalidatePendingDecision };
}
