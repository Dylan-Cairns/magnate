import { actionStableKey } from '../engine/actionSurface';
import { COURT_CARDS, type CardId } from '../engine/cards';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
} from '../engine/decisionActor';
import { rngFromSeed } from '../engine/rng';
import { isTerminal, scoreGame } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import type { GameAction, GameState, PlayerId, Ruleset } from '../engine/types';
import { toPlayerView } from '../engine/view';
import { courtActionBreakdown } from '../policies/courtPotentialV2';
import { sampleHiddenWorldStates } from '../policies/determinization';
import { createHeuristicV2PositionContext } from '../policies/heuristicV2PositionContext';
import {
  bestHeuristicV2Action,
  scoreHeuristicV2Actions,
} from '../policies/heuristicScorerV2';
import { collectGitMetadata } from './gitMetadata';
import type {
  ConfidenceInterval,
  GitMetadata,
  HeadToHeadCheckpoint,
  PlayedGame,
} from './types';

export const COURT_DECISION_EVAL_SCHEMA_VERSION = 1;
export const COURT_DECISION_EVAL_ARTIFACT_TYPE = 'ts-court-decision-eval';
export const DEFAULT_COURT_DECISION_EVAL_WORLDS = 20;
export const DEFAULT_COURT_DECISION_EVAL_MAX_POSITIONS = 150;
export const DEFAULT_COURT_DECISION_EVAL_CONTINUATION_SCALE = 1;
export const DEFAULT_COURT_DECISION_EVAL_SEED = 'court-decision-eval-v1';

const TERM_COURT_VALUE_SCALE = 1;
const MAX_PLAYOUT_STEPS = 1000;
const MARGIN_EPSILON = 1e-9;
const COURT_CARD_IDS = new Set<CardId>(COURT_CARDS.map((card) => card.id));

const ACTION_TYPE_BUCKETS = ['buy-deed', 'develop-deed'] as const;
const SWING_BUCKETS = ['<=0', '0-0.15', '0.15-0.35', '>0.35'] as const;
const FEASIBILITY_BUCKETS = ['<0.25', '0.25-0.5', '0.5-0.75', '>=0.75'] as const;
const PHASE_BUCKETS = ['early (1-14)', 'mid (15-28)', 'late (29+)'] as const;

export interface CourtDecisionEvalInput {
  config: {
    runLabel: string;
    ruleset?: Ruleset;
  };
  games: readonly PlayedGame[];
}

export interface CourtDecisionEvalOptions {
  sourceKind?: 'artifact' | 'checkpoint';
  sourcePath?: string;
  worlds?: number;
  maxPositions?: number;
  continuationScale?: number;
  seed?: string;
  generatedAtUtc?: string;
  git?: GitMetadata;
  onProgress?: (evaluated: number, total: number) => void;
}

export interface CourtDecisionEvalSummary {
  positions: number;
  worldsPerPosition: number;
  meanMarginDelta: number;
  marginDeltaCi95: ConfidenceInterval;
  courtBetterPositions: number;
  nonCourtBetterPositions: number;
  tiedPositions: number;
  meanCourtMargin: number;
  meanNonCourtMargin: number;
  meanCourtWinRate: number;
  meanNonCourtWinRate: number;
  meanRankTotalMarginDelta: number;
}

export interface CourtDecisionEvalBucket {
  key: string;
  summary: CourtDecisionEvalSummary;
}

export interface CourtDecisionEvalReport {
  schemaVersion: typeof COURT_DECISION_EVAL_SCHEMA_VERSION;
  artifactType: typeof COURT_DECISION_EVAL_ARTIFACT_TYPE;
  generatedAtUtc: string;
  git: GitMetadata;
  sourceKind?: 'artifact' | 'checkpoint';
  sourcePath?: string;
  runLabel: string;
  ruleset: Ruleset;
  config: {
    worlds: number;
    maxPositions: number;
    continuationScale: number;
    seed: string;
  };
  positions: {
    collected: number;
    evaluated: number;
    courtRecommended: number;
    courtRejected: number;
    skippedNoNonCourtAction: number;
    skippedNoBreakdown: number;
  };
  overall: CourtDecisionEvalSummary;
  recommended: CourtDecisionEvalSummary;
  rejected: CourtDecisionEvalSummary;
  byActionType: CourtDecisionEvalBucket[];
  byActionTypeRecommended: CourtDecisionEvalBucket[];
  byActionTypeRejected: CourtDecisionEvalBucket[];
  bySwing: CourtDecisionEvalBucket[];
  byFeasibility: CourtDecisionEvalBucket[];
  byPhase: CourtDecisionEvalBucket[];
  bySwingRecommended: CourtDecisionEvalBucket[];
  bySwingRejected: CourtDecisionEvalBucket[];
  byFeasibilityRecommended: CourtDecisionEvalBucket[];
  byFeasibilityRejected: CourtDecisionEvalBucket[];
}

export interface CourtDecisionOutcome {
  marginDelta: number;
  courtMargin: number;
  nonCourtMargin: number;
  courtWinRate: number;
  nonCourtWinRate: number;
  rankTotalMarginDelta: number;
}

interface CourtDecisionPosition {
  gameId: string;
  decisionIndex: number;
  botId: string;
  playerId: PlayerId;
  turn: number;
  state: GameState;
  courtAction: GameAction;
  nonCourtAction: GameAction;
  recommended: boolean;
  courtActionType: (typeof ACTION_TYPE_BUCKETS)[number];
  swing: number;
  feasibility: number;
}

interface CourtDecisionResult {
  position: CourtDecisionPosition;
  outcome: CourtDecisionOutcome;
}

interface TerminalArmOutcome {
  districtPointMargin: number;
  rankTotalMargin: number;
  winIndicator: number;
}

export function courtDecisionEvalInputFromCheckpoint(
  checkpoint: HeadToHeadCheckpoint
): CourtDecisionEvalInput {
  return {
    config: checkpoint.config,
    games: checkpoint.results.flatMap((result) => result.games),
  };
}

export function isTermCourtAction(action: GameAction): boolean {
  if (action.type !== 'buy-deed' && action.type !== 'develop-deed') {
    return false;
  }
  return COURT_CARD_IDS.has(action.cardId);
}

export function buildCourtDecisionEvalReport(
  input: CourtDecisionEvalInput,
  options: CourtDecisionEvalOptions = {}
): CourtDecisionEvalReport {
  const worlds = requiredPositiveIntegerOption(
    options.worlds,
    DEFAULT_COURT_DECISION_EVAL_WORLDS,
    'Court decision eval worlds'
  );
  const maxPositions = requiredPositiveIntegerOption(
    options.maxPositions,
    DEFAULT_COURT_DECISION_EVAL_MAX_POSITIONS,
    'Court decision eval maxPositions'
  );
  const continuationScale = requiredNonnegativeNumberOption(
    options.continuationScale,
    DEFAULT_COURT_DECISION_EVAL_CONTINUATION_SCALE,
    'Court decision eval continuationScale'
  );
  const seed = options.seed ?? DEFAULT_COURT_DECISION_EVAL_SEED;
  if (seed.trim() === '') {
    throw new Error('Court decision eval seed must be a non-empty string.');
  }
  const ruleset = input.config.ruleset ?? 'standard';

  const collection = collectCourtDecisionPositions(input, ruleset);
  const evaluated = selectPositions(collection.positions, maxPositions);
  const results = evaluated.map((position, index) => {
    options.onProgress?.(index + 1, evaluated.length);
    return {
      position,
      outcome: evaluateCourtDecisionPosition(
        position,
        worlds,
        continuationScale,
        seed
      ),
    };
  });
  const recommended = results.filter((result) => result.position.recommended);
  const rejected = results.filter((result) => !result.position.recommended);

  return {
    schemaVersion: COURT_DECISION_EVAL_SCHEMA_VERSION,
    artifactType: COURT_DECISION_EVAL_ARTIFACT_TYPE,
    generatedAtUtc: options.generatedAtUtc ?? new Date().toISOString(),
    git: options.git ?? collectGitMetadata(),
    ...(options.sourcePath
      ? {
          sourceKind: options.sourceKind ?? 'artifact',
          sourcePath: options.sourcePath,
        }
      : {}),
    runLabel: input.config.runLabel,
    ruleset,
    config: {
      worlds,
      maxPositions,
      continuationScale,
      seed,
    },
    positions: {
      collected: collection.positions.length,
      evaluated: evaluated.length,
      courtRecommended: recommended.length,
      courtRejected: rejected.length,
      skippedNoNonCourtAction: collection.skippedNoNonCourtAction,
      skippedNoBreakdown: collection.skippedNoBreakdown,
    },
    overall: summarizeCourtDecisionOutcomes(
      results.map((result) => result.outcome),
      worlds
    ),
    recommended: summarizeCourtDecisionOutcomes(
      recommended.map((result) => result.outcome),
      worlds
    ),
    rejected: summarizeCourtDecisionOutcomes(
      rejected.map((result) => result.outcome),
      worlds
    ),
    byActionType: buildBuckets(
      results,
      (result) => result.position.courtActionType,
      ACTION_TYPE_BUCKETS,
      worlds
    ),
    byActionTypeRecommended: buildBuckets(
      recommended,
      (result) => result.position.courtActionType,
      ACTION_TYPE_BUCKETS,
      worlds
    ),
    byActionTypeRejected: buildBuckets(
      rejected,
      (result) => result.position.courtActionType,
      ACTION_TYPE_BUCKETS,
      worlds
    ),
    bySwing: buildBuckets(results, swingBucket, SWING_BUCKETS, worlds),
    byFeasibility: buildBuckets(
      results,
      feasibilityBucket,
      FEASIBILITY_BUCKETS,
      worlds
    ),
    byPhase: buildBuckets(results, phaseBucket, PHASE_BUCKETS, worlds),
    bySwingRecommended: buildBuckets(
      recommended,
      swingBucket,
      SWING_BUCKETS,
      worlds
    ),
    bySwingRejected: buildBuckets(rejected, swingBucket, SWING_BUCKETS, worlds),
    byFeasibilityRecommended: buildBuckets(
      recommended,
      feasibilityBucket,
      FEASIBILITY_BUCKETS,
      worlds
    ),
    byFeasibilityRejected: buildBuckets(
      rejected,
      feasibilityBucket,
      FEASIBILITY_BUCKETS,
      worlds
    ),
  };
}

export function summarizeCourtDecisionOutcomes(
  outcomes: readonly CourtDecisionOutcome[],
  worldsPerPosition: number
): CourtDecisionEvalSummary {
  if (outcomes.length === 0) {
    return {
      positions: 0,
      worldsPerPosition,
      meanMarginDelta: 0,
      marginDeltaCi95: { low: 0, high: 0 },
      courtBetterPositions: 0,
      nonCourtBetterPositions: 0,
      tiedPositions: 0,
      meanCourtMargin: 0,
      meanNonCourtMargin: 0,
      meanCourtWinRate: 0,
      meanNonCourtWinRate: 0,
      meanRankTotalMarginDelta: 0,
    };
  }
  const deltas = outcomes.map((outcome) => outcome.marginDelta);
  const meanMarginDelta = mean(deltas);
  const variance =
    outcomes.length > 1
      ? deltas.reduce(
          (sum, delta) => sum + (delta - meanMarginDelta) ** 2,
          0
        ) /
        (outcomes.length - 1)
      : 0;
  const standardError = Math.sqrt(variance / outcomes.length);

  return {
    positions: outcomes.length,
    worldsPerPosition,
    meanMarginDelta,
    marginDeltaCi95: {
      low: meanMarginDelta - 1.96 * standardError,
      high: meanMarginDelta + 1.96 * standardError,
    },
    courtBetterPositions: deltas.filter((delta) => delta > MARGIN_EPSILON)
      .length,
    nonCourtBetterPositions: deltas.filter((delta) => delta < -MARGIN_EPSILON)
      .length,
    tiedPositions: deltas.filter(
      (delta) => Math.abs(delta) <= MARGIN_EPSILON
    ).length,
    meanCourtMargin: mean(outcomes.map((outcome) => outcome.courtMargin)),
    meanNonCourtMargin: mean(
      outcomes.map((outcome) => outcome.nonCourtMargin)
    ),
    meanCourtWinRate: mean(outcomes.map((outcome) => outcome.courtWinRate)),
    meanNonCourtWinRate: mean(
      outcomes.map((outcome) => outcome.nonCourtWinRate)
    ),
    meanRankTotalMarginDelta: mean(
      outcomes.map((outcome) => outcome.rankTotalMarginDelta)
    ),
  };
}

function collectCourtDecisionPositions(
  input: CourtDecisionEvalInput,
  ruleset: Ruleset
): {
  positions: CourtDecisionPosition[];
  skippedNoNonCourtAction: number;
  skippedNoBreakdown: number;
} {
  const positions: CourtDecisionPosition[] = [];
  let skippedNoNonCourtAction = 0;
  let skippedNoBreakdown = 0;

  for (const game of input.games) {
    let state = createSession(game.seed, game.firstPlayer, ruleset);
    for (const decision of game.transcript) {
      const decisionPlayer = decisionPlayerIdForState(state);
      if (decisionPlayer !== decision.activePlayerId) {
        throw new Error(
          `Court decision eval replay divergence in game ${game.gameId}: decision ${String(decision.decisionIndex)} expected ${decisionPlayer} but recorded ${decision.activePlayerId}.`
        );
      }
      const actions = legalActionsForDecisionPlayer(state, decisionPlayer);
      const action = actions.find(
        (candidate) => actionStableKey(candidate) === decision.actionKey
      );
      if (!action) {
        throw new Error(
          `Court decision eval replay divergence in game ${game.gameId}: decision ${String(decision.decisionIndex)} action ${decision.actionKey} is not legal.`
        );
      }
      if (actions.some(isTermCourtAction)) {
        const view = toPlayerView(state, decisionPlayer);
        const scored = scoreHeuristicV2Actions(actions, {
          state,
          view,
          courtValueScale: TERM_COURT_VALUE_SCALE,
        });
        const topAction = scored[0];
        const topCourt = scored.find((candidate) =>
          isTermCourtAction(candidate.action)
        );
        const topNonCourt = scored.find(
          (candidate) => !isTermCourtAction(candidate.action)
        );
        if (!topAction || !topCourt || !topNonCourt) {
          skippedNoNonCourtAction += 1;
        } else {
          const courtActionType = topCourt.action.type;
          const breakdown = courtActionBreakdown(
            topCourt.action,
            state,
            decisionPlayer,
            createHeuristicV2PositionContext(state, decisionPlayer),
            TERM_COURT_VALUE_SCALE
          );
          if (
            !breakdown ||
            (courtActionType !== 'buy-deed' &&
              courtActionType !== 'develop-deed')
          ) {
            skippedNoBreakdown += 1;
          } else {
            positions.push({
              gameId: game.gameId,
              decisionIndex: decision.decisionIndex,
              botId: decision.botId,
              playerId: decisionPlayer,
              turn: state.turn,
              state,
              courtAction: topCourt.action,
              nonCourtAction: topNonCourt.action,
              recommended: isTermCourtAction(topAction.action),
              courtActionType,
              swing: breakdown.swing,
              feasibility: breakdown.feasibility,
            });
          }
        }
      }
      state = stepToDecision(state, action);
    }
  }

  return { positions, skippedNoNonCourtAction, skippedNoBreakdown };
}

function selectPositions(
  positions: readonly CourtDecisionPosition[],
  maxPositions: number
): CourtDecisionPosition[] {
  if (positions.length <= maxPositions) {
    return [...positions];
  }
  const stride = positions.length / maxPositions;
  return Array.from({ length: maxPositions }, (_, index) => {
    return positions[Math.floor(index * stride)];
  });
}

function evaluateCourtDecisionPosition(
  position: CourtDecisionPosition,
  worlds: number,
  continuationScale: number,
  seed: string
): CourtDecisionOutcome {
  const view = toPlayerView(position.state, position.playerId);
  const random = rngFromSeed(
    `${seed}:${position.gameId}:${String(position.decisionIndex)}`
  );
  const worldStates = sampleHiddenWorldStates({
    state: position.state,
    view,
    rootPlayer: position.playerId,
    worldCount: worlds,
    random,
    errorPrefix: `Court decision eval ${position.gameId}#${String(position.decisionIndex)}`,
  });

  let courtMarginSum = 0;
  let nonCourtMarginSum = 0;
  let courtWinSum = 0;
  let nonCourtWinSum = 0;
  let rankTotalDeltaSum = 0;
  for (const world of worldStates) {
    const court = playForcedActionToTerminal(
      structuredClone(world),
      position.courtAction,
      position.playerId,
      continuationScale
    );
    const nonCourt = playForcedActionToTerminal(
      structuredClone(world),
      position.nonCourtAction,
      position.playerId,
      continuationScale
    );
    courtMarginSum += court.districtPointMargin;
    nonCourtMarginSum += nonCourt.districtPointMargin;
    courtWinSum += court.winIndicator;
    nonCourtWinSum += nonCourt.winIndicator;
    rankTotalDeltaSum +=
      court.rankTotalMargin - nonCourt.rankTotalMargin;
  }
  const count = Math.max(1, worldStates.length);

  return {
    marginDelta: (courtMarginSum - nonCourtMarginSum) / count,
    courtMargin: courtMarginSum / count,
    nonCourtMargin: nonCourtMarginSum / count,
    courtWinRate: courtWinSum / count,
    nonCourtWinRate: nonCourtWinSum / count,
    rankTotalMarginDelta: rankTotalDeltaSum / count,
  };
}

function playForcedActionToTerminal(
  state: GameState,
  forcedAction: GameAction,
  rootPlayer: PlayerId,
  continuationScale: number
): TerminalArmOutcome {
  let current = stepToDecision(state, forcedAction);
  let steps = 0;
  while (!isTerminal(current)) {
    if (steps >= MAX_PLAYOUT_STEPS) {
      throw new Error(
        `Court decision eval playout exceeded ${String(MAX_PLAYOUT_STEPS)} steps.`
      );
    }
    const player = decisionPlayerIdForState(current);
    if (!player) {
      throw new Error('Court decision eval playout found no decision player.');
    }
    const actions = legalActionsForDecisionPlayer(current, player);
    const next = bestHeuristicV2Action(actions, {
      state: current,
      view: toPlayerView(current, player),
      legalActions: actions,
      courtValueScale: continuationScale,
    });
    if (!next) {
      throw new Error('Court decision eval playout found no legal action.');
    }
    current = stepToDecision(current, next.action);
    steps += 1;
  }

  const finalScore = current.finalScore ?? scoreGame(current);
  const opponent = rootPlayer === 'PlayerA' ? 'PlayerB' : 'PlayerA';
  return {
    districtPointMargin:
      finalScore.districtPoints[rootPlayer] -
      finalScore.districtPoints[opponent],
    rankTotalMargin:
      finalScore.rankTotals[rootPlayer] - finalScore.rankTotals[opponent],
    winIndicator:
      finalScore.winner === rootPlayer
        ? 1
        : finalScore.winner === 'Draw'
          ? 0.5
          : 0,
  };
}

function buildBuckets(
  results: readonly CourtDecisionResult[],
  keyFor: (result: CourtDecisionResult) => string,
  order: readonly string[],
  worldsPerPosition: number
): CourtDecisionEvalBucket[] {
  const buckets: CourtDecisionEvalBucket[] = [];
  for (const key of order) {
    const outcomes = results
      .filter((result) => keyFor(result) === key)
      .map((result) => result.outcome);
    if (outcomes.length === 0) {
      continue;
    }
    buckets.push({
      key,
      summary: summarizeCourtDecisionOutcomes(outcomes, worldsPerPosition),
    });
  }
  return buckets;
}

function swingBucket(result: CourtDecisionResult): string {
  const swing = result.position.swing;
  if (swing <= 0) {
    return SWING_BUCKETS[0];
  }
  if (swing <= 0.15) {
    return SWING_BUCKETS[1];
  }
  if (swing <= 0.35) {
    return SWING_BUCKETS[2];
  }
  return SWING_BUCKETS[3];
}

function feasibilityBucket(result: CourtDecisionResult): string {
  const feasibility = result.position.feasibility;
  if (feasibility < 0.25) {
    return FEASIBILITY_BUCKETS[0];
  }
  if (feasibility < 0.5) {
    return FEASIBILITY_BUCKETS[1];
  }
  if (feasibility < 0.75) {
    return FEASIBILITY_BUCKETS[2];
  }
  return FEASIBILITY_BUCKETS[3];
}

function phaseBucket(result: CourtDecisionResult): string {
  const turn = result.position.turn;
  if (turn <= 14) {
    return PHASE_BUCKETS[0];
  }
  if (turn <= 28) {
    return PHASE_BUCKETS[1];
  }
  return PHASE_BUCKETS[2];
}

function mean(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function requiredPositiveIntegerOption(
  value: number | undefined,
  fallback: number,
  label: string
): number {
  const resolved = value ?? fallback;
  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return resolved;
}

function requiredNonnegativeNumberOption(
  value: number | undefined,
  fallback: number,
  label: string
): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved < 0) {
    throw new Error(`${label} must be a finite number >= 0.`);
  }
  return resolved;
}

export function renderCourtDecisionEvalMarkdown(
  report: CourtDecisionEvalReport
): string {
  const lines: string[] = [
    `# Court Decision Evaluation: ${report.runLabel}`,
    '',
    `- Ruleset: ${report.ruleset}`,
    `- Git: ${report.git.commit ?? 'unknown'}${report.git.dirty ? ' (dirty)' : ''}`,
    ...(report.sourcePath
      ? [`- Source: ${report.sourceKind ?? 'artifact'} ${report.sourcePath}`]
      : []),
    `- Config: worlds=${String(report.config.worlds)} maxPositions=${String(report.config.maxPositions)} continuationScale=${formatNumber(report.config.continuationScale)} seed=${report.config.seed}`,
    '',
    'Each position is a recorded decision with a term-owned Court option (buy-deed or develop-deed on a Court). The term-owned Court action is compared against the best non-Court action under matched hidden worlds and matched engine seeds, with the heuristic v2 continuation policy. Positive delta means the Court action reached a higher district-point margin for the acting player.',
    '',
    '## Positions',
    '',
    `Collected ${String(report.positions.collected)}, evaluated ${String(report.positions.evaluated)} (recommended ${String(report.positions.courtRecommended)}, rejected ${String(report.positions.courtRejected)}); skipped no non-Court action ${String(report.positions.skippedNoNonCourtAction)}, skipped no breakdown ${String(report.positions.skippedNoBreakdown)}.`,
    '',
    '## Overall',
    '',
    ...renderSummaryTable([{ key: 'all positions', summary: report.overall }]),
    '## Term Recommendations',
    '',
    ...renderSummaryTable([
      { key: 'recommended', summary: report.recommended },
    ]),
    '## Term Rejections',
    '',
    ...renderSummaryTable([{ key: 'rejected', summary: report.rejected }]),
    '## By Action Type',
    '',
    ...renderSummaryTable(report.byActionType),
    '## By Action Type: Term Recommendations',
    '',
    ...renderSummaryTable(report.byActionTypeRecommended),
    '## By Action Type: Term Rejections',
    '',
    ...renderSummaryTable(report.byActionTypeRejected),
    '## By Swing',
    '',
    ...renderSummaryTable(report.bySwing),
    '## By Feasibility',
    '',
    ...renderSummaryTable(report.byFeasibility),
    '## By Phase',
    '',
    ...renderSummaryTable(report.byPhase),
    '## By Swing: Term Recommendations',
    '',
    ...renderSummaryTable(report.bySwingRecommended),
    '## By Swing: Term Rejections',
    '',
    ...renderSummaryTable(report.bySwingRejected),
    '## By Feasibility: Term Recommendations',
    '',
    ...renderSummaryTable(report.byFeasibilityRecommended),
    '## By Feasibility: Term Rejections',
    '',
    ...renderSummaryTable(report.byFeasibilityRejected),
  ];
  return `${lines.join('\n')}\n`;
}

function renderSummaryTable(
  buckets: readonly CourtDecisionEvalBucket[]
): string[] {
  const lines = [
    '| bucket | positions | mean delta | ci95 | court better | non-Court better | tied | court margin | non-Court margin | court win rate | non-Court win rate | rank delta |',
    '|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|',
  ];
  for (const bucket of buckets) {
    const summary = bucket.summary;
    lines.push(
      `| ${bucket.key} | ${String(summary.positions)} | ${formatSigned(summary.meanMarginDelta)} | [${formatSigned(summary.marginDeltaCi95.low)}, ${formatSigned(summary.marginDeltaCi95.high)}] | ${String(summary.courtBetterPositions)} | ${String(summary.nonCourtBetterPositions)} | ${String(summary.tiedPositions)} | ${formatNumber(summary.meanCourtMargin)} | ${formatNumber(summary.meanNonCourtMargin)} | ${formatNumber(summary.meanCourtWinRate)} | ${formatNumber(summary.meanNonCourtWinRate)} | ${formatSigned(summary.meanRankTotalMarginDelta)} |`
    );
  }
  lines.push('');
  return lines;
}

function formatNumber(value: number): string {
  return value.toFixed(3);
}

function formatSigned(value: number): string {
  return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3);
}
