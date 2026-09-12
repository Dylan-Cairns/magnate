import { actionStableKey } from '../engine/actionSurface';
import { COURT_CARDS, type CardId } from '../engine/cards';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
} from '../engine/decisionActor';
import { isTerminal } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import {
  developmentCost,
  findDevelopableCard,
  sumTokens,
} from '../engine/stateHelpers';
import type {
  GameAction,
  GameState,
  PlayerId,
  Ruleset,
} from '../engine/types';
import { toPlayerView } from '../engine/view';
import type { BotSpec } from '../policies/botSpec';
import {
  courtActionBreakdown,
  isCourtCard,
} from '../policies/courtPotentialV2';
import { createHeuristicV2PositionContext } from '../policies/heuristicV2PositionContext';
import {
  scoreHeuristicV2Actions,
  type HeuristicV2ScoredAction,
} from '../policies/heuristicScorerV2';
import { DEFAULT_COURT_VALUE_SCALE } from '../policies/searchConfig';
import { pairedDiscordantSummary, type PairedDiscordantSummary } from './stats';
import type { HeadToHeadArtifact, PlayedGame } from './types';

export const STANDARD_NONINFERIORITY_MARGIN = 0.07;
export const MIN_DECISION_PAIRS = 30;
export const DEED_BUY_RATE_RELATIVE_BAND = 0.2;
export const COURT_FOLLOW_THROUGH_PASS_RATIO = 0.5;
export const COURT_FOLLOW_THROUGH_OBSERVE_RATIO = 0.25;

const COURT_CARD_IDS = new Set<CardId>(COURT_CARDS.map((card) => card.id));

export type CourtValueGateStatus = 'pass' | 'fail' | 'observe';

export interface CourtValueGateResult {
  readonly id: string;
  readonly status: CourtValueGateStatus;
  readonly detail: string;
}

export interface CourtValuePairRecord {
  readonly pairId: string;
  readonly seed: string;
  readonly candidateWins: number;
  readonly opponentWins: number;
  readonly draws: number;
  readonly margin: number;
}

export interface CourtValueUsageSummary {
  readonly botId: string;
  readonly decisions: number;
  readonly buys: number;
  readonly buyRate: number;
  readonly sells: number;
  readonly courtBuys: number;
  readonly courtOutrights: number;
  readonly courtDeedDevelops: number;
  readonly courtCompletions: number;
  readonly courtSells: number;
  readonly courtSellsWithLegalCourtBuild: number;
}

export interface CourtDecisionDiagnostics {
  readonly botId: string;
  readonly decisions: number;
  readonly decisionsWithCourtOption: number;
  readonly chosenCourtActions: number;
  readonly bestCourtRankTop1: number;
  readonly bestCourtRankTop4: number;
  readonly bestCourtRankTop16: number;
  readonly meanBestCourtSwing: number;
  readonly meanBestCourtFeasibility: number;
  readonly meanBestCourtDelta: number;
}

export interface CourtValueReport {
  readonly artifactPath: string | null;
  readonly runLabel: string;
  readonly ruleset: Ruleset;
  readonly gitCommit: string | null;
  readonly gitDirty: boolean | null;
  readonly candidate: { readonly id: string; readonly courtValueScale: string };
  readonly opponent: { readonly id: string; readonly courtValueScale: string };
  readonly totals: {
    readonly games: number;
    readonly pairs: number;
    readonly candidateWins: number;
    readonly opponentWins: number;
    readonly draws: number;
    readonly candidateWinRate: number;
    readonly candidateWinRateCi95: { readonly low: number; readonly high: number };
    readonly sideGap: number;
    readonly averageTurns: number;
  };
  readonly paired: PairedDiscordantSummary;
  readonly perPair: readonly CourtValuePairRecord[];
  readonly usageByBotId: Readonly<Record<string, CourtValueUsageSummary>>;
  readonly courtDecisionsByBotId: Readonly<Record<string, CourtDecisionDiagnostics>>;
  readonly gates: readonly CourtValueGateResult[];
}

interface MutableCourtUsage {
  botId: string;
  decisions: number;
  buys: number;
  sells: number;
  courtBuys: number;
  courtOutrights: number;
  courtDeedDevelops: number;
  courtCompletions: number;
  courtSells: number;
  courtSellsWithLegalCourtBuild: number;
}

interface MutableCourtDecisionDiagnostics {
  botId: string;
  courtValueScale: number;
  decisions: number;
  decisionsWithCourtOption: number;
  chosenCourtActions: number;
  bestCourtRankTop1: number;
  bestCourtRankTop4: number;
  bestCourtRankTop16: number;
  swingSum: number;
  feasibilitySum: number;
  deltaSum: number;
  samples: number;
}

export function buildCourtValueReport(
  artifact: HeadToHeadArtifact,
  artifactPath?: string
): CourtValueReport {
  const ruleset: Ruleset = artifact.config.ruleset ?? 'standard';
  const perPair = collectPairRecords(artifact);
  const paired = pairedDiscordantSummary(
    perPair.map((record) => record.margin)
  );
  const usageByBotId = collectUsageByBotId(artifact, ruleset);
  const courtDecisionsByBotId = collectCourtDecisionDiagnostics(
    artifact,
    ruleset
  );
  const candidateUsage = requiredSummaryUsage(
    usageByBotId,
    artifact.config.candidate.id
  );
  const opponentUsage = requiredSummaryUsage(
    usageByBotId,
    artifact.config.opponent.id
  );
  const gates = evaluateCourtValueGates({
    ruleset,
    paired,
    candidateUsage,
    opponentUsage,
  });

  return {
    artifactPath: artifactPath ?? null,
    runLabel: artifact.config.runLabel,
    ruleset,
    gitCommit: artifact.git.commit,
    gitDirty: artifact.git.dirty,
    candidate: {
      id: artifact.config.candidate.id,
      courtValueScale: describeCourtValueScale(artifact.config.candidate),
    },
    opponent: {
      id: artifact.config.opponent.id,
      courtValueScale: describeCourtValueScale(artifact.config.opponent),
    },
    totals: {
      games: artifact.summary.totalGames,
      pairs: perPair.length,
      candidateWins: artifact.summary.candidateWins,
      opponentWins: artifact.summary.opponentWins,
      draws: artifact.summary.draws,
      candidateWinRate: artifact.summary.candidateWinRate,
      candidateWinRateCi95: artifact.summary.candidateWinRateCi95,
      sideGap: artifact.summary.sideGap,
      averageTurns: artifact.summary.averageTurns,
    },
    paired,
    perPair,
    usageByBotId,
    courtDecisionsByBotId,
    gates: [
      {
        id: 'replay-integrity',
        status: 'pass',
        detail: `${String(artifact.games.length)} games replayed through canonical legal actions`,
      },
      ...gates,
    ],
  };
}

export function evaluateCourtValueGates(input: {
  ruleset: Ruleset;
  paired: PairedDiscordantSummary;
  candidateUsage: CourtValueUsageSummary;
  opponentUsage: CourtValueUsageSummary;
}): CourtValueGateResult[] {
  const gates: CourtValueGateResult[] = [];
  const { paired, candidateUsage, opponentUsage } = input;

  if (input.ruleset === 'standard') {
    if (paired.pairs < MIN_DECISION_PAIRS) {
      gates.push({
        id: 'standard-paired-noninferiority',
        status: 'observe',
        detail: `only ${String(paired.pairs)} pairs; at least ${String(MIN_DECISION_PAIRS)} required for a gate call`,
      });
    } else if (
      paired.meanWinMarginCi95.low > -STANDARD_NONINFERIORITY_MARGIN
    ) {
      gates.push({
        id: 'standard-paired-noninferiority',
        status: 'pass',
        detail: `paired margin ${formatNumber(paired.meanWinMargin)} ci95 [${formatNumber(paired.meanWinMarginCi95.low)}, ${formatNumber(paired.meanWinMarginCi95.high)}] above -${formatNumber(STANDARD_NONINFERIORITY_MARGIN)}`,
      });
    } else {
      gates.push({
        id: 'standard-paired-noninferiority',
        status: 'fail',
        detail: `paired margin ${formatNumber(paired.meanWinMargin)} ci95 [${formatNumber(paired.meanWinMarginCi95.low)}, ${formatNumber(paired.meanWinMarginCi95.high)}] breaches -${formatNumber(STANDARD_NONINFERIORITY_MARGIN)}`,
      });
    }
    gates.push(deedBuyRateGate(candidateUsage, opponentUsage));
    return gates;
  }

  if (paired.pairs < MIN_DECISION_PAIRS) {
    gates.push({
      id: 'extended-paired-improvement',
      status: 'observe',
      detail: `only ${String(paired.pairs)} pairs; at least ${String(MIN_DECISION_PAIRS)} required for a gate call`,
    });
  } else if (paired.meanWinMarginCi95.low > 0) {
    gates.push({
      id: 'extended-paired-improvement',
      status: 'pass',
      detail: `paired margin ${formatNumber(paired.meanWinMargin)} ci95 [${formatNumber(paired.meanWinMarginCi95.low)}, ${formatNumber(paired.meanWinMarginCi95.high)}] above 0`,
    });
  } else if (paired.meanWinMargin > 0) {
    gates.push({
      id: 'extended-paired-improvement',
      status: 'observe',
      detail: `paired margin ${formatNumber(paired.meanWinMargin)} positive but ci95 [${formatNumber(paired.meanWinMarginCi95.low)}, ${formatNumber(paired.meanWinMarginCi95.high)}] includes 0; extend the series`,
    });
  } else {
    gates.push({
      id: 'extended-paired-improvement',
      status: 'fail',
      detail: `paired margin ${formatNumber(paired.meanWinMargin)} does not improve on the control`,
    });
  }

  const acquisitions =
    candidateUsage.courtBuys + candidateUsage.courtOutrights;
  if (acquisitions > 0 && candidateUsage.courtCompletions > 0) {
    gates.push({
      id: 'extended-court-utilization',
      status: 'pass',
      detail: `candidate bought/developed ${String(acquisitions)} courts and completed ${String(candidateUsage.courtCompletions)}`,
    });
  } else if (acquisitions > 0) {
    gates.push({
      id: 'extended-court-utilization',
      status: 'observe',
      detail: `candidate acquired ${String(acquisitions)} courts but completed none`,
    });
  } else {
    gates.push({
      id: 'extended-court-utilization',
      status: 'fail',
      detail: 'candidate never acquired a court',
    });
  }

  gates.push(courtFollowThroughGate(candidateUsage));

  if (candidateUsage.courtSellsWithLegalCourtBuild === 0) {
    gates.push({
      id: 'extended-court-dump',
      status: 'pass',
      detail: 'candidate never sold a court while a court build was legal',
    });
  } else if (candidateUsage.courtSellsWithLegalCourtBuild <= 2) {
    gates.push({
      id: 'extended-court-dump',
      status: 'observe',
      detail: `candidate sold a court with a legal court build ${String(candidateUsage.courtSellsWithLegalCourtBuild)} times`,
    });
  } else {
    gates.push({
      id: 'extended-court-dump',
      status: 'fail',
      detail: `candidate sold a court with a legal court build ${String(candidateUsage.courtSellsWithLegalCourtBuild)} times`,
    });
  }
  return gates;
}

function courtFollowThroughGate(
  candidateUsage: CourtValueUsageSummary
): CourtValueGateResult {
  const acquisitions =
    candidateUsage.courtBuys + candidateUsage.courtOutrights;
  if (acquisitions === 0) {
    return {
      id: 'extended-court-follow-through',
      status: 'fail',
      detail: 'candidate never acquired a court to follow through',
    };
  }
  const ratio = candidateUsage.courtCompletions / acquisitions;
  if (ratio >= COURT_FOLLOW_THROUGH_PASS_RATIO) {
    return {
      id: 'extended-court-follow-through',
      status: 'pass',
      detail: `candidate completed ${formatNumber(ratio)} of acquired courts`,
    };
  }
  if (ratio >= COURT_FOLLOW_THROUGH_OBSERVE_RATIO) {
    return {
      id: 'extended-court-follow-through',
      status: 'observe',
      detail: `candidate completed only ${formatNumber(ratio)} of acquired courts`,
    };
  }
  return {
    id: 'extended-court-follow-through',
    status: 'fail',
    detail: `candidate completed only ${formatNumber(ratio)} of acquired courts`,
  };
}

function deedBuyRateGate(
  candidateUsage: CourtValueUsageSummary,
  opponentUsage: CourtValueUsageSummary
): CourtValueGateResult {
  if (opponentUsage.buyRate <= 0.001) {
    const status: CourtValueGateStatus =
      candidateUsage.buyRate <= 0.02 ? 'pass' : 'observe';
    return {
      id: 'standard-deed-buy-behavior',
      status,
      detail: `control buy rate ${formatNumber(opponentUsage.buyRate)}; candidate ${formatNumber(candidateUsage.buyRate)}`,
    };
  }
  const ratio = candidateUsage.buyRate / opponentUsage.buyRate;
  const withinBand =
    ratio >= 1 - DEED_BUY_RATE_RELATIVE_BAND &&
    ratio <= 1 + DEED_BUY_RATE_RELATIVE_BAND;
  return {
    id: 'standard-deed-buy-behavior',
    status: withinBand ? 'pass' : 'fail',
    detail: `candidate/control deed buy-rate ratio ${formatNumber(ratio)} (band ${formatNumber(1 - DEED_BUY_RATE_RELATIVE_BAND)}-${formatNumber(1 + DEED_BUY_RATE_RELATIVE_BAND)})`,
  };
}

function collectPairRecords(
  artifact: HeadToHeadArtifact
): CourtValuePairRecord[] {
  const candidateId = artifact.config.candidate.id;
  const byPairId = new Map<
    string,
    { seed: string; games: number; candidateWins: number; opponentWins: number; draws: number }
  >();

  for (const game of artifact.games) {
    const pairId = pairIdForGame(game);
    const candidateSeat = seatForBot(game, candidateId);
    const record = byPairId.get(pairId) ?? {
      seed: game.seed,
      games: 0,
      candidateWins: 0,
      opponentWins: 0,
      draws: 0,
    };
    record.games += 1;
    if (game.finalScore.winner === 'Draw') {
      record.draws += 1;
    } else if (game.finalScore.winner === candidateSeat) {
      record.candidateWins += 1;
    } else {
      record.opponentWins += 1;
    }
    byPairId.set(pairId, record);
  }

  return [...byPairId.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([pairId, record]) => {
      if (record.games !== 2) {
        throw new Error(
          `Pair ${pairId} expected exactly 2 games; received ${String(record.games)}.`
        );
      }
      return {
        pairId,
        seed: record.seed,
        candidateWins: record.candidateWins,
        opponentWins: record.opponentWins,
        draws: record.draws,
        margin: (record.candidateWins - record.opponentWins) / 2,
      };
    });
}

function collectUsageByBotId(
  artifact: HeadToHeadArtifact,
  ruleset: Ruleset
): Record<string, CourtValueUsageSummary> {
  const usageByBotId = new Map<string, MutableCourtUsage>();
  for (const botId of [
    artifact.config.candidate.id,
    artifact.config.opponent.id,
  ]) {
    usageByBotId.set(botId, createMutableUsage(botId));
  }

  for (const game of artifact.games) {
    replayGame(game, ruleset, usageByBotId, undefined);
  }

  return Object.fromEntries(
    [...usageByBotId.entries()].map(([botId, usage]) => [
      botId,
      {
        botId,
        decisions: usage.decisions,
        buys: usage.buys,
        buyRate: usage.decisions > 0 ? usage.buys / usage.decisions : 0,
        sells: usage.sells,
        courtBuys: usage.courtBuys,
        courtOutrights: usage.courtOutrights,
        courtDeedDevelops: usage.courtDeedDevelops,
        courtCompletions: usage.courtCompletions,
        courtSells: usage.courtSells,
        courtSellsWithLegalCourtBuild: usage.courtSellsWithLegalCourtBuild,
      },
    ])
  );
}

function collectCourtDecisionDiagnostics(
  artifact: HeadToHeadArtifact,
  ruleset: Ruleset
): Record<string, CourtDecisionDiagnostics> {
  const specsById = new Map<string, BotSpec>([
    [artifact.config.candidate.id, artifact.config.candidate],
    [artifact.config.opponent.id, artifact.config.opponent],
  ]);
  const diagnosticsByBotId = new Map<
    string,
    MutableCourtDecisionDiagnostics
  >();
  for (const [botId, spec] of specsById) {
    if (isHeuristicV2SearchSpec(spec)) {
      diagnosticsByBotId.set(
        botId,
        createMutableDiagnostics(
          botId,
          spec.config.courtValueScale ?? DEFAULT_COURT_VALUE_SCALE
        )
      );
    }
  }

  for (const game of artifact.games) {
    replayGame(game, ruleset, undefined, diagnosticsByBotId);
  }

  return Object.fromEntries(
    [...diagnosticsByBotId.entries()].map(([botId, diagnostics]) => [
      botId,
      {
        botId,
        decisions: diagnostics.decisions,
        decisionsWithCourtOption: diagnostics.decisionsWithCourtOption,
        chosenCourtActions: diagnostics.chosenCourtActions,
        bestCourtRankTop1: diagnostics.bestCourtRankTop1,
        bestCourtRankTop4: diagnostics.bestCourtRankTop4,
        bestCourtRankTop16: diagnostics.bestCourtRankTop16,
        meanBestCourtSwing: safeMean(
          diagnostics.swingSum,
          diagnostics.samples
        ),
        meanBestCourtFeasibility: safeMean(
          diagnostics.feasibilitySum,
          diagnostics.samples
        ),
        meanBestCourtDelta: safeMean(
          diagnostics.deltaSum,
          diagnostics.samples
        ),
      },
    ])
  );
}

function replayGame(
  game: PlayedGame,
  ruleset: Ruleset,
  usageByBotId: Map<string, MutableCourtUsage> | undefined,
  diagnosticsByBotId:
    | Map<string, MutableCourtDecisionDiagnostics>
    | undefined
): void {
  let state = createSession(game.seed, game.firstPlayer, ruleset);
  for (const decision of game.transcript) {
    const decisionPlayer = decisionPlayerIdForState(state);
    if (decisionPlayer !== decision.activePlayerId) {
      throw new Error(
        `Report replay divergence in game ${game.gameId}: decision ${String(decision.decisionIndex)} expected ${decisionPlayer} but recorded ${decision.activePlayerId}.`
      );
    }
    const actions = legalActionsForDecisionPlayer(state, decisionPlayer);
    const action = actions.find(
      (candidate) => actionStableKey(candidate) === decision.actionKey
    );
    if (!action) {
      throw new Error(
        `Report replay divergence in game ${game.gameId}: decision ${String(decision.decisionIndex)} action ${decision.actionKey} is not legal.`
      );
    }
    if (usageByBotId) {
      const usage = requiredMutableUsage(usageByBotId, decision.botId);
      usage.decisions += 1;
      recordActionUsage(state, decisionPlayer, action, actions, usage);
    }
    if (diagnosticsByBotId) {
      recordCourtDecisionDiagnostics(
        diagnosticsByBotId,
        decision.botId,
        state,
        decisionPlayer,
        actions,
        action
      );
    }
    state = stepToDecision(state, action);
  }
  if (!isTerminal(state)) {
    throw new Error(
      `Report replay for game ${game.gameId} did not reach a terminal state.`
    );
  }
}

function recordCourtDecisionDiagnostics(
  diagnosticsByBotId: Map<string, MutableCourtDecisionDiagnostics>,
  botId: string,
  state: GameState,
  decisionPlayer: PlayerId,
  legalActions: readonly GameAction[],
  chosenAction: GameAction
): void {
  const diagnostics = diagnosticsByBotId.get(botId);
  if (!diagnostics) {
    return;
  }
  diagnostics.decisions += 1;
  const hasCourtOption = legalActions.some(isCourtBuildAction);
  if (hasCourtOption) {
    diagnostics.decisionsWithCourtOption += 1;
  }
  if (isCourtBuildAction(chosenAction)) {
    diagnostics.chosenCourtActions += 1;
  }
  if (!hasCourtOption) {
    return;
  }

  const scaled = scoreHeuristicV2Actions(legalActions, {
    state,
    view: toPlayerView(state, decisionPlayer),
    courtValueScale: diagnostics.courtValueScale,
  });
  const bestCourt = scaled.find((candidate) =>
    isCourtBuildAction(candidate.action)
  );
  if (!bestCourt) {
    return;
  }
  const rank = bestCourt.rank;
  if (rank === 0) {
    diagnostics.bestCourtRankTop1 += 1;
  }
  if (rank < 4) {
    diagnostics.bestCourtRankTop4 += 1;
  }
  if (rank < 16) {
    diagnostics.bestCourtRankTop16 += 1;
  }
  accumulateCourtBreakdown(diagnostics, bestCourt, state, decisionPlayer);
}

function accumulateCourtBreakdown(
  diagnostics: MutableCourtDecisionDiagnostics,
  bestCourt: HeuristicV2ScoredAction,
  state: GameState,
  decisionPlayer: PlayerId
): void {
  const card = findDevelopableCard(
    'cardId' in bestCourt.action ? bestCourt.action.cardId : ''
  );
  if (!isCourtCard(card)) {
    return;
  }
  const breakdown = courtActionBreakdown(
    bestCourt.action,
    state,
    decisionPlayer,
    createHeuristicV2PositionContext(state, decisionPlayer),
    diagnostics.courtValueScale
  );
  if (!breakdown) {
    return;
  }
  diagnostics.swingSum += breakdown.swing;
  diagnostics.feasibilitySum += breakdown.feasibility;
  diagnostics.deltaSum += breakdown.delta;
  diagnostics.samples += 1;
}

function recordActionUsage(
  state: GameState,
  playerId: PlayerId,
  action: GameAction,
  legalActions: readonly GameAction[],
  usage: MutableCourtUsage
): void {
  switch (action.type) {
    case 'buy-deed':
      usage.buys += 1;
      if (COURT_CARD_IDS.has(action.cardId)) {
        usage.courtBuys += 1;
      }
      return;
    case 'sell-card':
      usage.sells += 1;
      if (COURT_CARD_IDS.has(action.cardId)) {
        usage.courtSells += 1;
        if (legalActions.some(isCourtBuildAction)) {
          usage.courtSellsWithLegalCourtBuild += 1;
        }
      }
      return;
    case 'develop-deed': {
      if (!COURT_CARD_IDS.has(action.cardId)) {
        return;
      }
      usage.courtDeedDevelops += 1;
      const card = findDevelopableCard(action.cardId);
      const deed = districtDeed(state, action.districtId, playerId);
      if (
        card &&
        deed &&
        deed.progress + sumTokens(action.tokens) >= developmentCost(card)
      ) {
        usage.courtCompletions += 1;
      }
      return;
    }
    case 'develop-outright':
      if (COURT_CARD_IDS.has(action.cardId)) {
        usage.courtOutrights += 1;
        usage.courtCompletions += 1;
      }
      return;
    default:
      return;
  }
}

function isCourtBuildAction(action: GameAction): boolean {
  if (
    action.type !== 'buy-deed' &&
    action.type !== 'develop-outright' &&
    action.type !== 'develop-deed'
  ) {
    return false;
  }
  return COURT_CARD_IDS.has(action.cardId);
}

export function renderCourtValueReportMarkdown(
  report: CourtValueReport
): string {
  const lines: string[] = [
    `# Court Value Benchmark Report: ${report.runLabel}`,
    '',
    `- Ruleset: ${report.ruleset}`,
    `- Candidate: ${report.candidate.id} (courtValueScale=${report.candidate.courtValueScale})`,
    `- Opponent: ${report.opponent.id} (courtValueScale=${report.opponent.courtValueScale})`,
    `- Git: ${report.gitCommit ?? 'unknown'}${report.gitDirty ? ' (dirty)' : ''}`,
    ...(report.artifactPath ? [`- Artifact: ${report.artifactPath}`] : []),
    '',
    '## Head-to-head',
    '',
    `Games ${String(report.totals.games)} across ${String(report.totals.pairs)} pairs; candidate ${String(report.totals.candidateWins)}W-${String(report.totals.opponentWins)}L-${String(report.totals.draws)}D, win rate ${formatNumber(report.totals.candidateWinRate)} ci95 [${formatNumber(report.totals.candidateWinRateCi95.low)}, ${formatNumber(report.totals.candidateWinRateCi95.high)}], side gap ${formatNumber(report.totals.sideGap)}, average turns ${formatNumber(report.totals.averageTurns)}.`,
    '',
    '## Paired analysis',
    '',
    '| pairs | candidate-more | opponent-more | tied | mean margin | ci95 | mcnemar p |',
    '|:---|---:|---:|---:|---:|:---|---:|',
    `| ${String(report.paired.pairs)} | ${String(report.paired.candidateWinsMorePairs)} | ${String(report.paired.opponentWinsMorePairs)} | ${String(report.paired.tiedPairs)} | ${formatNumber(report.paired.meanWinMargin)} | [${formatNumber(report.paired.meanWinMarginCi95.low)}, ${formatNumber(report.paired.meanWinMarginCi95.high)}] | ${formatNumber(report.paired.mcnemarTwoSidedP)} |`,
    '',
    '## Deed and court usage',
    '',
    '| bot | decisions | buys | buy rate | sells | court buys | court outrights | court deed develops | court completions | court sells | court sells with legal build |',
    '|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
  ];
  for (const usage of Object.values(report.usageByBotId)) {
    lines.push(
      `| ${usage.botId} | ${String(usage.decisions)} | ${String(usage.buys)} | ${formatNumber(usage.buyRate)} | ${String(usage.sells)} | ${String(usage.courtBuys)} | ${String(usage.courtOutrights)} | ${String(usage.courtDeedDevelops)} | ${String(usage.courtCompletions)} | ${String(usage.courtSells)} | ${String(usage.courtSellsWithLegalCourtBuild)} |`
    );
  }
  lines.push(
    '',
    '## Court decision diagnostics',
    '',
    '| bot | decisions | court-option decisions | chosen court actions | best court top-1 | top-4 | top-16 | mean swing | mean feasibility | mean delta |',
    '|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
  );
  for (const diagnostics of Object.values(report.courtDecisionsByBotId)) {
    lines.push(
      `| ${diagnostics.botId} | ${String(diagnostics.decisions)} | ${String(diagnostics.decisionsWithCourtOption)} | ${String(diagnostics.chosenCourtActions)} | ${String(diagnostics.bestCourtRankTop1)} | ${String(diagnostics.bestCourtRankTop4)} | ${String(diagnostics.bestCourtRankTop16)} | ${formatNumber(diagnostics.meanBestCourtSwing)} | ${formatNumber(diagnostics.meanBestCourtFeasibility)} | ${formatNumber(diagnostics.meanBestCourtDelta)} |`
    );
  }
  lines.push('', '## Gates', '', '| gate | status | detail |', '|:---|:---|:---|');
  for (const gate of report.gates) {
    lines.push(`| ${gate.id} | ${gate.status} | ${gate.detail} |`);
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
}

function pairIdForGame(game: PlayedGame): string {
  const match = /^pair-(\d+)-candidate-as-[ab]$/u.exec(game.gameId);
  if (!match) {
    throw new Error(
      `Head-to-head game id ${game.gameId} does not match the paired-seed scheme.`
    );
  }
  return match[1];
}

function seatForBot(game: PlayedGame, botId: string): PlayerId {
  if (game.botBySeat.PlayerA === botId) {
    return 'PlayerA';
  }
  if (game.botBySeat.PlayerB === botId) {
    return 'PlayerB';
  }
  throw new Error(`Game ${game.gameId} does not include bot ${botId}.`);
}

function districtDeed(
  state: GameState,
  districtId: string,
  playerId: PlayerId
): GameState['districts'][number]['stacks'][PlayerId]['deed'] {
  const district = state.districts.find(
    (candidate) => candidate.id === districtId
  );
  return district?.stacks[playerId].deed;
}

function createMutableUsage(botId: string): MutableCourtUsage {
  return {
    botId,
    decisions: 0,
    buys: 0,
    sells: 0,
    courtBuys: 0,
    courtOutrights: 0,
    courtDeedDevelops: 0,
    courtCompletions: 0,
    courtSells: 0,
    courtSellsWithLegalCourtBuild: 0,
  };
}

function createMutableDiagnostics(
  botId: string,
  courtValueScale: number
): MutableCourtDecisionDiagnostics {
  return {
    botId,
    courtValueScale,
    decisions: 0,
    decisionsWithCourtOption: 0,
    chosenCourtActions: 0,
    bestCourtRankTop1: 0,
    bestCourtRankTop4: 0,
    bestCourtRankTop16: 0,
    swingSum: 0,
    feasibilitySum: 0,
    deltaSum: 0,
    samples: 0,
  };
}

function isHeuristicV2SearchSpec(
  spec: BotSpec
): spec is Extract<BotSpec, { kind: 'search' }> {
  return spec.kind === 'search' && spec.config.heuristic === 'v2';
}

function requiredMutableUsage(
  usageByBotId: ReadonlyMap<string, MutableCourtUsage>,
  botId: string
): MutableCourtUsage {
  const usage = usageByBotId.get(botId);
  if (!usage) {
    throw new Error(`Court value report is missing bot ${botId}.`);
  }
  return usage;
}

function requiredSummaryUsage(
  usageByBotId: Readonly<Record<string, CourtValueUsageSummary>>,
  botId: string
): CourtValueUsageSummary {
  const usage = usageByBotId[botId];
  if (!usage) {
    throw new Error(`Court value report is missing bot ${botId}.`);
  }
  return usage;
}

function describeCourtValueScale(spec: BotSpec): string {
  if (spec.kind !== 'search' && spec.kind !== 'td-root-search') {
    return 'n/a';
  }
  return spec.config.courtValueScale === undefined
    ? `default (${String(DEFAULT_COURT_VALUE_SCALE)})`
    : String(spec.config.courtValueScale);
}

function safeMean(sum: number, count: number): number {
  return count > 0 ? sum / count : 0;
}

function formatNumber(value: number): string {
  return value.toFixed(4);
}
