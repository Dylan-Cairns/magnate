import { createHash } from 'node:crypto';

import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import { CARD_BY_ID, type CardId } from '../engine/cards';
import {
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { rngFromSeed } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import type {
  FinalScore,
  GameAction,
  GameState,
  PlayerId,
  ResourcePool,
} from '../engine/types';
import { sampleHiddenWorldStates } from '../policies/determinization';
import { scoreHeuristicV2Actions } from '../policies/heuristicScorerV2';
import {
  DEFAULT_TD_ROOT_MODEL_INDEX_PATH,
  preloadTdRootBrowserModel,
} from '../policies/modelRuntimeCache';
import {
  rolloutSearchScenarioSeeds,
  runRolloutSearchTask,
  type RolloutSearchRuntimeGuidance,
  type RolloutSearchTraceStep,
  type RolloutSearchWorkerTask,
} from '../policies/rolloutSearchCore';
import type { SearchPolicyConfig } from '../policies/searchConfig';
import type { LoadedTdGuidanceModel } from '../policies/tdGuidanceModel';
import { createTdRootSearchRolloutGuidance } from '../policies/tdRootSearchPolicy';
import {
  encodeActionCandidates,
  encodeObservation,
} from '../policies/trainingEncoding';
import {
  STRATEGIC_POSITION_CATALOG_VERSION,
  createStrategicPositionCatalogV0,
  isStrategicOptionalityPositionV0,
  type StrategicOptionalityFamilyV0,
  type StrategicPositionV0,
} from './strategicPositionCatalog';
import {
  STRATEGIC_POSITION_COMPARISON_SEED_SCHEME,
  STRATEGIC_TD_800_VISIT_VARIANT_ID,
  createStrategicComparisonVariantCatalogV0,
  strategicComparisonSeed,
} from './strategicPositionComparison';

export const STRATEGIC_FORCED_ROLLOUT_TRACE_SCHEMA_VERSION = 1 as const;

export type StrategicForcedRolloutGuideV0 = 'td' | 'heuristic-v2';
export type { StrategicOptionalityFamilyV0 } from './strategicPositionCatalog';

export interface StrategicForcedRolloutProposalV0 {
  readonly actionKey: string;
  readonly actionLabel: string;
  /** Zero-based rank among the legal actions. */
  readonly rank: number;
  readonly rawScore: number;
}

export interface StrategicForcedRolloutStepV0 {
  readonly stepIndex: number;
  readonly decisionPlayer: PlayerId;
  readonly turn: number;
  readonly phase: GameState['phase'];
  readonly finalTurnsRemaining: number | null;
  readonly actionKey: string;
  readonly actionLabel: string;
  readonly legalActionKeys: readonly string[];
  readonly proposals: {
    readonly td: StrategicForcedRolloutProposalV0;
    readonly heuristicV2: StrategicForcedRolloutProposalV0;
  } | null;
  readonly playerAHandBefore: readonly CardId[];
  readonly playerAHandAfter: readonly CardId[];
  readonly playerAResourcesBefore: ResourcePool;
  readonly targetLocationBefore: string;
  readonly targetLocationAfter: string;
  readonly targetLegalDistrictsBefore: readonly string[];
}

export type StrategicContinuationStatusV0 =
  | 'realized'
  | 'legal-but-not-used'
  | 'held-but-never-legal'
  | 'opponent-held'
  | 'not-reached-by-player-a';

export interface StrategicForcedRolloutTraceV0 {
  readonly rootFocusActionId: 'preserve-option' | 'overwrite-option';
  readonly guide: StrategicForcedRolloutGuideV0;
  readonly score: number;
  readonly simulatedActionSteps: number;
  readonly terminatedBeforeDepthLimit: boolean;
  readonly finalScore: FinalScore;
  readonly steps: readonly StrategicForcedRolloutStepV0[];
  readonly firstProposalDivergenceStepIndex: number | null;
  readonly firstPlayerAProposalDivergenceStepIndex: number | null;
  readonly targetEverHeldByPlayerA: boolean;
  readonly targetOpportunitySeen: boolean;
  readonly targetDevelopedInValuableLane: boolean;
  readonly continuationStatus: StrategicContinuationStatusV0;
}

export interface StrategicForcedRolloutScenarioV0 {
  readonly scenarioIndex: number;
  readonly worldIndex: number;
  readonly engineSeed: string;
  readonly rolloutRandomSeed: string;
  readonly hiddenAssignmentFingerprint: string;
  readonly targetInitialLocation: string;
  readonly traces: readonly StrategicForcedRolloutTraceV0[];
}

export interface StrategicForcedRolloutRepetitionV0 {
  readonly repetition: number;
  readonly sharedRandomSeed: string;
  readonly scenarios: readonly StrategicForcedRolloutScenarioV0[];
}

export interface StrategicForcedRolloutCaseV0 {
  readonly positionId: string;
  readonly randomGroupId: string;
  readonly family: StrategicOptionalityFamilyV0;
  readonly targetCardId: CardId;
  readonly targetCardName: string;
  readonly valuableDistrictId: string;
  readonly alternativeDistrictId: string;
  readonly focusActions: readonly {
    readonly id: 'preserve-option' | 'overwrite-option';
    readonly actionKey: string;
  }[];
  readonly repetitions: readonly StrategicForcedRolloutRepetitionV0[];
}

export interface StrategicForcedRolloutTraceRunV0 {
  readonly schemaVersion: typeof STRATEGIC_FORCED_ROLLOUT_TRACE_SCHEMA_VERSION;
  readonly catalogVersion: typeof STRATEGIC_POSITION_CATALOG_VERSION;
  readonly seedScheme: typeof STRATEGIC_POSITION_COMPARISON_SEED_SCHEME;
  readonly modelIndexPath: string;
  readonly config: SearchPolicyConfig;
  readonly repetitionIds: readonly number[];
  readonly scenarioIndices: readonly number[];
  readonly positions: readonly StrategicForcedRolloutCaseV0[];
}

export interface StrategicForcedRolloutTraceProgressV0 {
  readonly completedTraces: number;
  readonly totalTraces: number;
  readonly positionId: string;
  readonly repetition: number;
  readonly scenarioIndex: number;
  readonly rootFocusActionId: 'preserve-option' | 'overwrite-option';
  readonly guide: StrategicForcedRolloutGuideV0;
}

export interface StrategicForcedRolloutTraceOptionsV0 {
  readonly positions?: readonly StrategicPositionV0[];
  readonly repetitionIds?: readonly number[];
  readonly scenarioIndices?: readonly number[];
  readonly config?: SearchPolicyConfig;
  readonly modelIndexPath?: string;
  readonly loadModel?: () => Promise<LoadedTdGuidanceModel>;
  readonly onProgress?: (
    progress: StrategicForcedRolloutTraceProgressV0
  ) => void;
}

const TRACE_GUIDES: readonly StrategicForcedRolloutGuideV0[] = [
  'td',
  'heuristic-v2',
];

export async function runStrategicForcedRolloutTraceV0(
  options: StrategicForcedRolloutTraceOptionsV0 = {}
): Promise<StrategicForcedRolloutTraceRunV0> {
  const defaults = defaultTraceRuntime();
  const positions =
    options.positions ??
    createStrategicPositionCatalogV0().filter(isStrategicOptionalityPositionV0);
  const repetitionIds = validateIndices(
    options.repetitionIds ?? [0],
    'repetition'
  );
  const scenarioIndices = validateIndices(
    options.scenarioIndices ?? Array.from({ length: 50 }, (_, index) => index),
    'scenario'
  );
  const config = structuredClone(options.config ?? defaults.config);
  validateTraceConfig(config);
  const modelIndexPath = options.modelIndexPath ?? defaults.modelIndexPath;
  const model = await (options.loadModel
    ? options.loadModel()
    : preloadTdRootBrowserModel(modelIndexPath));
  const tdGuidance = createTdRootSearchRolloutGuidance({
    model,
    guidance: { rollout: 'td', leaf: 'td' },
  });
  if (!tdGuidance.chooseRolloutAction || !tdGuidance.evaluateLeaf) {
    throw new Error(
      'Forced rollout tracing requires TD rollout and leaf guidance.'
    );
  }
  if (positions.length === 0) {
    throw new Error('Forced rollout tracing requires optionality positions.');
  }

  const totalTraces =
    positions.length *
    repetitionIds.length *
    scenarioIndices.length *
    2 *
    TRACE_GUIDES.length;
  let completedTraces = 0;
  const cases: StrategicForcedRolloutCaseV0[] = [];

  for (const position of positions) {
    const context = optionalityContext(position);
    const repetitions: StrategicForcedRolloutRepetitionV0[] = [];
    for (const repetition of repetitionIds) {
      const randomGroupId = position.pairId ?? position.id;
      const sharedRandomSeed = strategicComparisonSeed(
        randomGroupId,
        repetition
      );
      const rootPlayer = position.perspectivePlayerId;
      const view = toDecisionPlayerView(position.state, rootPlayer);
      const worldStates = sampleHiddenWorldStates({
        state: position.state,
        view,
        rootPlayer,
        worldCount: config.worlds,
        random: rngFromSeed(sharedRandomSeed),
        errorPrefix: 'Strategic forced rollout trace',
      });
      if (worldStates.length !== config.worlds) {
        throw new Error(
          `Forced rollout trace expected ${String(config.worlds)} worlds but received ${String(worldStates.length)}.`
        );
      }
      const scenarios: StrategicForcedRolloutScenarioV0[] = [];

      for (const scenarioIndex of scenarioIndices) {
        const worldIndex = scenarioIndex % worldStates.length;
        const world = worldStates[worldIndex];
        const { engineSeed, rolloutRandomSeed } = rolloutSearchScenarioSeeds(
          sharedRandomSeed,
          scenarioIndex
        );
        const traces: StrategicForcedRolloutTraceV0[] = [];

        for (const focus of context.focusActions) {
          const rootAction = actionForKey(
            world,
            rootPlayer,
            focus.actionKey,
            position.id
          );
          for (const guide of TRACE_GUIDES) {
            traces.push(
              runOneTrace({
                position,
                context,
                worldStates,
                worldIndex,
                scenarioIndex,
                engineSeed,
                rolloutRandomSeed,
                rootPlayer,
                rootAction,
                rootFocusActionId: focus.id,
                guide,
                config,
                model,
                tdGuidance,
              })
            );
            completedTraces += 1;
            options.onProgress?.({
              completedTraces,
              totalTraces,
              positionId: position.id,
              repetition,
              scenarioIndex,
              rootFocusActionId: focus.id,
              guide,
            });
          }
        }

        scenarios.push({
          scenarioIndex,
          worldIndex,
          engineSeed,
          rolloutRandomSeed,
          hiddenAssignmentFingerprint: hiddenAssignmentFingerprint(
            world,
            rootPlayer
          ),
          targetInitialLocation: targetLocation(world, context.targetCardId),
          traces,
        });
      }
      repetitions.push({ repetition, sharedRandomSeed, scenarios });
    }
    cases.push({
      positionId: position.id,
      randomGroupId: position.pairId ?? position.id,
      family: context.family,
      targetCardId: context.targetCardId,
      targetCardName: CARD_BY_ID[context.targetCardId].name,
      valuableDistrictId: context.valuableDistrictId,
      alternativeDistrictId: context.alternativeDistrictId,
      focusActions: context.focusActions,
      repetitions,
    });
  }

  return {
    schemaVersion: STRATEGIC_FORCED_ROLLOUT_TRACE_SCHEMA_VERSION,
    catalogVersion: STRATEGIC_POSITION_CATALOG_VERSION,
    seedScheme: STRATEGIC_POSITION_COMPARISON_SEED_SCHEME,
    modelIndexPath,
    config,
    repetitionIds,
    scenarioIndices,
    positions: cases,
  };
}

interface OptionalityContext {
  readonly family: StrategicOptionalityFamilyV0;
  readonly targetCardId: CardId;
  readonly valuableDistrictId: string;
  readonly alternativeDistrictId: string;
  readonly focusActions: readonly {
    readonly id: 'preserve-option' | 'overwrite-option';
    readonly actionKey: string;
  }[];
}

function optionalityContext(position: StrategicPositionV0): OptionalityContext {
  if (position.perspectivePlayerId !== 'PlayerA') {
    throw new Error(
      `Optionality trace ${position.id} must use the PlayerA catalog perspective.`
    );
  }
  const trace = position.optionalityTrace;
  if (!trace) {
    throw new Error(
      `Position ${position.id} is not an endpoint optionality case.`
    );
  }
  const focusById = new Map(
    position.focusActions.map((focus) => [focus.id, focus])
  );
  const preserve = focusById.get(trace.preserveFocusActionId);
  const overwrite = focusById.get(trace.overwriteFocusActionId);
  if (!preserve || !overwrite) {
    throw new Error(
      `Optionality position ${position.id} must define preserve-option and overwrite-option.`
    );
  }
  const legalActions = legalActionsForDecisionPlayer(
    position.state,
    position.perspectivePlayerId
  );
  const preserveAction = actionForStableKey(legalActions, preserve.actionKey);
  const overwriteAction = actionForStableKey(legalActions, overwrite.actionKey);
  if (
    preserveAction.type !== 'develop-outright' ||
    overwriteAction.type !== 'develop-outright'
  ) {
    throw new Error(
      `Optionality position ${position.id} focus actions must be outright developments.`
    );
  }
  if (
    preserveAction.districtId !== trace.alternativeDistrictId ||
    overwriteAction.districtId !== trace.valuableDistrictId ||
    trace.valuableDistrictId === trace.alternativeDistrictId
  ) {
    throw new Error(
      `Optionality position ${position.id} trace metadata does not match its focus lanes.`
    );
  }
  return {
    family: trace.family,
    targetCardId: trace.targetCardId,
    valuableDistrictId: trace.valuableDistrictId,
    alternativeDistrictId: trace.alternativeDistrictId,
    focusActions: [
      { id: 'preserve-option', actionKey: preserve.actionKey },
      { id: 'overwrite-option', actionKey: overwrite.actionKey },
    ],
  };
}

function runOneTrace({
  position,
  context,
  worldStates,
  worldIndex,
  scenarioIndex,
  engineSeed,
  rolloutRandomSeed,
  rootPlayer,
  rootAction,
  rootFocusActionId,
  guide,
  config,
  model,
  tdGuidance,
}: {
  position: StrategicPositionV0;
  context: OptionalityContext;
  worldStates: readonly GameState[];
  worldIndex: number;
  scenarioIndex: number;
  engineSeed: string;
  rolloutRandomSeed: string;
  rootPlayer: PlayerId;
  rootAction: GameAction;
  rootFocusActionId: 'preserve-option' | 'overwrite-option';
  guide: StrategicForcedRolloutGuideV0;
  config: SearchPolicyConfig;
  model: LoadedTdGuidanceModel;
  tdGuidance: RolloutSearchRuntimeGuidance;
}): StrategicForcedRolloutTraceV0 {
  const steps: StrategicForcedRolloutStepV0[] = [];
  let finalState: GameState | undefined;
  let leafCalls = 0;
  const runtimeGuidance: RolloutSearchRuntimeGuidance = {
    ...(guide === 'td'
      ? { chooseRolloutAction: tdGuidance.chooseRolloutAction }
      : {}),
    evaluateLeaf(input) {
      leafCalls += 1;
      return tdGuidance.evaluateLeaf!(input);
    },
  };
  const task: RolloutSearchWorkerTask = {
    kind: 'rollout-search',
    contextId: `${position.id}:forced-trace`,
    visitIndex: scenarioIndex,
    actionVisitIndex: scenarioIndex,
    scenarioIndex,
    worldIndex,
    engineSeed,
    rootPlayer,
    rootAction,
    rootActionKey: actionStableKey(rootAction),
    config,
    randomSeed: rolloutRandomSeed,
  };
  const result = runRolloutSearchTask(
    task,
    worldStates,
    undefined,
    runtimeGuidance,
    {
      onStep(step) {
        finalState = step.stateAfter;
        const compact = compactStep({
          step,
          context,
          model,
        });
        const followedProposal =
          compact.proposals?.[guide === 'td' ? 'td' : 'heuristicV2'];
        if (
          followedProposal &&
          followedProposal.actionKey !== compact.actionKey
        ) {
          throw new Error(
            `Forced rollout trace ${guide} proposal does not match its trajectory action.`
          );
        }
        steps.push(compact);
      },
    }
  );
  if (!finalState || !isTerminal(finalState) || !finalState.finalScore) {
    throw new Error(
      `Forced rollout trace did not reach terminal scoring for ${position.id}, scenario ${String(scenarioIndex)}, ${rootFocusActionId}, ${guide}.`
    );
  }
  if (!result.terminatedBeforeDepthLimit || leafCalls !== 0) {
    throw new Error(
      `Forced rollout trace unexpectedly used its depth-limit leaf for ${position.id}, scenario ${String(scenarioIndex)}, ${rootFocusActionId}, ${guide}.`
    );
  }
  if (steps.length !== result.simulatedActionSteps) {
    throw new Error(
      'Forced rollout trace step count does not match task result.'
    );
  }

  const targetEverHeldByPlayerA = steps.some(
    (step) =>
      step.playerAHandBefore.includes(context.targetCardId) ||
      step.playerAHandAfter.includes(context.targetCardId)
  );
  const targetOpportunitySeen = steps.some((step) =>
    step.targetLegalDistrictsBefore.includes(context.valuableDistrictId)
  );
  const targetDevelopedInValuableLane =
    finalState.districts
      .find((district) => district.id === context.valuableDistrictId)
      ?.stacks.PlayerA.developed.includes(context.targetCardId) ?? false;
  const targetInitiallyHeldByOpponent =
    targetLocation(worldStates[worldIndex], context.targetCardId) ===
    'PlayerB-hand';

  return {
    rootFocusActionId,
    guide,
    score: result.score,
    simulatedActionSteps: result.simulatedActionSteps,
    terminatedBeforeDepthLimit: result.terminatedBeforeDepthLimit,
    finalScore: structuredClone(finalState.finalScore),
    steps,
    firstProposalDivergenceStepIndex:
      steps.find((step) => proposalsDiverge(step))?.stepIndex ?? null,
    firstPlayerAProposalDivergenceStepIndex:
      steps.find(
        (step) => step.decisionPlayer === 'PlayerA' && proposalsDiverge(step)
      )?.stepIndex ?? null,
    targetEverHeldByPlayerA,
    targetOpportunitySeen,
    targetDevelopedInValuableLane,
    continuationStatus: targetDevelopedInValuableLane
      ? 'realized'
      : targetOpportunitySeen
        ? 'legal-but-not-used'
        : targetEverHeldByPlayerA
          ? 'held-but-never-legal'
          : targetInitiallyHeldByOpponent
            ? 'opponent-held'
            : 'not-reached-by-player-a',
  };
}

function compactStep({
  step,
  context,
  model,
}: {
  step: RolloutSearchTraceStep;
  context: OptionalityContext;
  model: LoadedTdGuidanceModel;
}): StrategicForcedRolloutStepV0 {
  const playerABefore = player(step.stateBefore, 'PlayerA');
  const playerAAfter = player(step.stateAfter, 'PlayerA');
  return {
    stepIndex: step.stepIndex,
    decisionPlayer: step.decisionPlayer,
    turn: step.stateBefore.turn,
    phase: step.stateBefore.phase,
    finalTurnsRemaining: step.stateBefore.finalTurnsRemaining ?? null,
    actionKey: step.actionKey,
    actionLabel: actionLabel(step.action, context),
    legalActionKeys: step.legalActionKeys,
    proposals:
      step.stepIndex === 0
        ? null
        : proposalsForState({
            state: step.stateBefore,
            decisionPlayer: step.decisionPlayer,
            context,
            model,
          }),
    playerAHandBefore: [...playerABefore.hand],
    playerAHandAfter: [...playerAAfter.hand],
    playerAResourcesBefore: structuredClone(playerABefore.resources),
    targetLocationBefore: targetLocation(
      step.stateBefore,
      context.targetCardId
    ),
    targetLocationAfter: targetLocation(step.stateAfter, context.targetCardId),
    targetLegalDistrictsBefore: targetLegalDistricts(
      step.stateBefore,
      step.decisionPlayer,
      context.targetCardId
    ),
  };
}

function proposalsForState({
  state,
  decisionPlayer,
  context,
  model,
}: {
  state: GameState;
  decisionPlayer: PlayerId;
  context: OptionalityContext;
  model: LoadedTdGuidanceModel;
}): NonNullable<StrategicForcedRolloutStepV0['proposals']> {
  const actions = legalActionsForDecisionPlayer(state, decisionPlayer);
  const view = toDecisionPlayerView(state, decisionPlayer);
  const keyed = toKeyedActions(actions);
  const observation = encodeObservation(view);
  const logits = model.opponentScorer.logitsForActions
    ? model.opponentScorer.logitsForActions(
        observation,
        keyed.map((candidate) => candidate.action)
      )
    : model.opponentScorer.logits(
        observation,
        encodeActionCandidates(keyed.map((candidate) => candidate.action))
      );
  if (logits.length !== keyed.length) {
    throw new Error('Forced rollout trace TD logits length mismatch.');
  }
  for (const logit of logits) {
    if (!Number.isFinite(logit)) {
      throw new Error('Forced rollout trace TD proposal logit is not finite.');
    }
  }
  const tdRankedIndices = keyed
    .map((_candidate, index) => index)
    .sort((left, right) => {
      const delta = logits[right] - logits[left];
      if (Math.abs(delta) > 1e-9) {
        return delta > 0 ? 1 : -1;
      }
      return keyed[left].actionKey.localeCompare(keyed[right].actionKey);
    });
  const tdIndex = tdRankedIndices[0];
  const tdCandidate = keyed[tdIndex];
  if (!tdCandidate || !Number.isFinite(logits[tdIndex])) {
    throw new Error(
      'Forced rollout trace TD proposal is not a finite legal action.'
    );
  }
  const heuristic = scoreHeuristicV2Actions(actions, {
    state,
    view,
    legalActions: actions,
  });
  const heuristicBest = heuristic[0];
  if (!heuristicBest) {
    throw new Error('Forced rollout trace heuristic has no legal proposal.');
  }
  return {
    td: {
      actionKey: tdCandidate.actionKey,
      actionLabel: actionLabel(tdCandidate.action, context),
      rank: 0,
      rawScore: logits[tdIndex],
    },
    heuristicV2: {
      actionKey: heuristicBest.actionKey,
      actionLabel: actionLabel(heuristicBest.action, context),
      rank: heuristicBest.rank,
      rawScore: heuristicBest.score,
    },
  };
}

function targetLegalDistricts(
  state: GameState,
  decisionPlayer: PlayerId,
  targetCardId: CardId
): string[] {
  if (decisionPlayer !== 'PlayerA') {
    return [];
  }
  return legalActionsForDecisionPlayer(state, decisionPlayer)
    .filter(
      (action): action is Extract<GameAction, { type: 'develop-outright' }> =>
        action.type === 'develop-outright' && action.cardId === targetCardId
    )
    .map((action) => action.districtId)
    .filter((districtId, index, values) => values.indexOf(districtId) === index)
    .sort((left, right) => left.localeCompare(right));
}

function targetLocation(state: GameState, targetCardId: CardId): string {
  for (const entry of state.players) {
    if (entry.hand.includes(targetCardId)) {
      return `${entry.id}-hand`;
    }
    if (entry.crowns.includes(targetCardId)) {
      return `${entry.id}-crowns`;
    }
  }
  for (const district of state.districts) {
    for (const playerId of ['PlayerA', 'PlayerB'] as const) {
      const stack = district.stacks[playerId];
      if (stack.developed.includes(targetCardId)) {
        return `${playerId}-developed:${district.id}`;
      }
      if (stack.deed?.cardId === targetCardId) {
        return `${playerId}-deed:${district.id}`;
      }
    }
  }
  const drawIndex = state.deck.draw.indexOf(targetCardId);
  if (drawIndex >= 0) {
    return `draw:${String(drawIndex)}`;
  }
  if (state.deck.discard.includes(targetCardId)) {
    return 'discard';
  }
  return 'unlocated';
}

function actionLabel(action: GameAction, context: OptionalityContext): string {
  switch (action.type) {
    case 'end-turn':
      return 'End turn';
    case 'trade':
      return `Trade ${action.give} for ${action.receive}`;
    case 'choose-income-suit':
      return `Choose ${action.suit} income for ${CARD_BY_ID[action.cardId].name}`;
    case 'sell-card':
      return `Sell ${CARD_BY_ID[action.cardId].name}`;
    case 'buy-deed':
      return `Buy deed for ${CARD_BY_ID[action.cardId].name} in ${semanticDistrict(action.districtId, context)}`;
    case 'develop-deed':
      return `Develop deed ${CARD_BY_ID[action.cardId].name} in ${semanticDistrict(action.districtId, context)}`;
    case 'develop-outright':
      return `Develop ${CARD_BY_ID[action.cardId].name} in ${semanticDistrict(action.districtId, context)}`;
  }
}

function semanticDistrict(
  districtId: string,
  context: OptionalityContext
): string {
  if (districtId === context.valuableDistrictId) {
    return `valuable lane (${districtId})`;
  }
  if (districtId === context.alternativeDistrictId) {
    return `alternative lane (${districtId})`;
  }
  return districtId;
}

function hiddenAssignmentFingerprint(
  state: GameState,
  rootPlayer: PlayerId
): string {
  const opponent = rootPlayer === 'PlayerA' ? 'PlayerB' : 'PlayerA';
  const payload = JSON.stringify({
    opponentHand: player(state, opponent).hand,
    draw: state.deck.draw,
    discard: state.deck.discard,
  });
  return `sha256:${createHash('sha256').update(payload).digest('hex')}`;
}

function actionForKey(
  state: GameState,
  playerId: PlayerId,
  actionKey: string,
  positionId: string
): GameAction {
  const action = actionForStableKey(
    legalActionsForDecisionPlayer(state, playerId),
    actionKey
  );
  if (actionStableKey(action) !== actionKey) {
    throw new Error(
      `Forced root key mismatch for ${positionId}: ${actionKey}.`
    );
  }
  return action;
}

function actionForStableKey(
  actions: readonly GameAction[],
  actionKey: string
): GameAction {
  const action = actions.find(
    (candidate) => actionStableKey(candidate) === actionKey
  );
  if (!action) {
    throw new Error(`No legal action matches forced key ${actionKey}.`);
  }
  return action;
}

function player(state: GameState, playerId: PlayerId) {
  const found = state.players.find((entry) => entry.id === playerId);
  if (!found) {
    throw new Error(`Game state is missing ${playerId}.`);
  }
  return found;
}

function proposalsDiverge(step: StrategicForcedRolloutStepV0): boolean {
  return (
    step.proposals !== null &&
    step.proposals.td.actionKey !== step.proposals.heuristicV2.actionKey
  );
}

function defaultTraceRuntime(): {
  readonly config: SearchPolicyConfig;
  readonly modelIndexPath: string;
} {
  const variant = createStrategicComparisonVariantCatalogV0().find(
    (entry) => entry.descriptor.id === STRATEGIC_TD_800_VISIT_VARIANT_ID
  );
  if (
    !variant ||
    variant.descriptor.kind !== 'bot-spec' ||
    variant.descriptor.spec.kind !== 'td-root-search'
  ) {
    throw new Error('The 800-visit TD strategic variant is unavailable.');
  }
  return {
    config: {
      ...structuredClone(variant.descriptor.spec.config),
      heuristic: 'v2',
    },
    modelIndexPath:
      variant.descriptor.spec.modelIndexPath ??
      DEFAULT_TD_ROOT_MODEL_INDEX_PATH,
  };
}

function validateTraceConfig(config: SearchPolicyConfig): void {
  if (config.rolloutEpsilon !== 0) {
    throw new Error('Forced rollout tracing requires rolloutEpsilon=0.');
  }
  if (config.rollouts !== 1) {
    throw new Error('Forced rollout tracing requires rollouts=1.');
  }
  if (config.heuristic !== 'v2') {
    throw new Error('Forced rollout tracing requires heuristic v2.');
  }
  for (const [label, value] of [
    ['worlds', config.worlds],
    ['rollouts', config.rollouts],
    ['depth', config.depth],
    ['maxRootActions', config.maxRootActions],
  ] as const) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`Forced rollout trace ${label} must be positive.`);
    }
  }
}

function validateIndices(values: readonly number[], label: string): number[] {
  if (
    values.length === 0 ||
    values.some((value) => !Number.isSafeInteger(value) || value < 0) ||
    new Set(values).size !== values.length
  ) {
    throw new Error(
      `Forced rollout trace ${label} ids must be unique nonnegative safe integers.`
    );
  }
  return [...values];
}
