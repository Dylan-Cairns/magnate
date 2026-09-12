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
import type { BotSpec } from '../policies/botSpec';
import { DEFAULT_DEED_POTENTIAL_BASE } from '../policies/searchConfig';
import { pairedDiscordantSummary, type PairedDiscordantSummary } from './stats';
import type { HeadToHeadArtifact, PlayedGame } from './types';

export const STANDARD_NONINFERIORITY_MARGIN = 0.07;
export const MIN_DECISION_PAIRS = 30;
export const DEED_BUY_RATE_RELATIVE_BAND = 0.2;

const COURT_CARD_IDS = new Set<CardId>(COURT_CARDS.map((card) => card.id));

export type DeedGateStatus = 'pass' | 'fail' | 'observe';

export interface DeedGateResult {
  readonly id: string;
  readonly status: DeedGateStatus;
  readonly detail: string;
}

export interface DeedPairRecord {
  readonly pairId: string;
  readonly seed: string;
  readonly candidateWins: number;
  readonly opponentWins: number;
  readonly draws: number;
  readonly margin: number;
}

export interface DeedUsageSummary {
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

export interface DeedPotentialReport {
  readonly artifactPath: string | null;
  readonly runLabel: string;
  readonly ruleset: Ruleset;
  readonly gitCommit: string | null;
  readonly gitDirty: boolean | null;
  readonly candidate: { readonly id: string; readonly deedPotentialBase: string };
  readonly opponent: { readonly id: string; readonly deedPotentialBase: string };
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
  readonly perPair: readonly DeedPairRecord[];
  readonly usageByBotId: Readonly<Record<string, DeedUsageSummary>>;
  readonly gates: readonly DeedGateResult[];
}

interface MutableDeedUsage {
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

export function buildDeedPotentialReport(
  artifact: HeadToHeadArtifact,
  artifactPath?: string
): DeedPotentialReport {
  const ruleset: Ruleset = artifact.config.ruleset ?? 'standard';
  const perPair = collectPairRecords(artifact);
  const paired = pairedDiscordantSummary(
    perPair.map((record) => record.margin)
  );
  const usageByBotId = collectUsageByBotId(artifact, ruleset);
  const candidateUsage = requiredSummaryUsage(
    usageByBotId,
    artifact.config.candidate.id
  );
  const opponentUsage = requiredSummaryUsage(
    usageByBotId,
    artifact.config.opponent.id
  );
  const gates = evaluateDeedGates({
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
      deedPotentialBase: describeDeedPotentialBase(artifact.config.candidate),
    },
    opponent: {
      id: artifact.config.opponent.id,
      deedPotentialBase: describeDeedPotentialBase(artifact.config.opponent),
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

export function evaluateDeedGates(input: {
  ruleset: Ruleset;
  paired: PairedDiscordantSummary;
  candidateUsage: DeedUsageSummary;
  opponentUsage: DeedUsageSummary;
}): DeedGateResult[] {
  const gates: DeedGateResult[] = [];
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

  const engagement = candidateUsage.courtBuys + candidateUsage.courtOutrights;
  if (engagement > 0 && candidateUsage.courtCompletions > 0) {
    gates.push({
      id: 'extended-court-utilization',
      status: 'pass',
      detail: `candidate bought/developed ${String(engagement)} courts and completed ${String(candidateUsage.courtCompletions)}`,
    });
  } else if (engagement > 0) {
    gates.push({
      id: 'extended-court-utilization',
      status: 'observe',
      detail: `candidate acquired ${String(engagement)} courts but completed none`,
    });
  } else {
    gates.push({
      id: 'extended-court-utilization',
      status: 'fail',
      detail: 'candidate never acquired a court',
    });
  }

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

function deedBuyRateGate(
  candidateUsage: DeedUsageSummary,
  opponentUsage: DeedUsageSummary
): DeedGateResult {
  if (opponentUsage.buyRate <= 0.001) {
    const status: DeedGateStatus =
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
): DeedPairRecord[] {
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
): Record<string, DeedUsageSummary> {
  const usageByBotId = new Map<string, MutableDeedUsage>();
  for (const botId of [
    artifact.config.candidate.id,
    artifact.config.opponent.id,
  ]) {
    usageByBotId.set(botId, createMutableUsage(botId));
  }

  for (const game of artifact.games) {
    replayGameUsage(game, ruleset, usageByBotId);
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

function replayGameUsage(
  game: PlayedGame,
  ruleset: Ruleset,
  usageByBotId: Map<string, MutableDeedUsage>
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
    const usage = requiredMutableUsage(usageByBotId, decision.botId);
    usage.decisions += 1;
    recordActionUsage(state, decisionPlayer, action, actions, usage);
    state = stepToDecision(state, action);
  }
  if (!isTerminal(state)) {
    throw new Error(
      `Report replay for game ${game.gameId} did not reach a terminal state.`
    );
  }
}

function recordActionUsage(
  state: GameState,
  playerId: PlayerId,
  action: GameAction,
  legalActions: readonly GameAction[],
  usage: MutableDeedUsage
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

export function renderDeedPotentialReportMarkdown(
  report: DeedPotentialReport
): string {
  const lines: string[] = [
    `# Deed Potential Benchmark Report: ${report.runLabel}`,
    '',
    `- Ruleset: ${report.ruleset}`,
    `- Candidate: ${report.candidate.id} (deedPotentialBase=${report.candidate.deedPotentialBase})`,
    `- Opponent: ${report.opponent.id} (deedPotentialBase=${report.opponent.deedPotentialBase})`,
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

function createMutableUsage(botId: string): MutableDeedUsage {
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

function requiredMutableUsage(
  usageByBotId: ReadonlyMap<string, MutableDeedUsage>,
  botId: string
): MutableDeedUsage {
  const usage = usageByBotId.get(botId);
  if (!usage) {
    throw new Error(`Deed potential report is missing bot ${botId}.`);
  }
  return usage;
}

function requiredSummaryUsage(
  usageByBotId: Readonly<Record<string, DeedUsageSummary>>,
  botId: string
): DeedUsageSummary {
  const usage = usageByBotId[botId];
  if (!usage) {
    throw new Error(`Deed potential report is missing bot ${botId}.`);
  }
  return usage;
}

function describeDeedPotentialBase(spec: BotSpec): string {
  if (spec.kind !== 'search' && spec.kind !== 'td-root-search') {
    return 'n/a';
  }
  return spec.config.deedPotentialBase === undefined
    ? `default (${String(DEFAULT_DEED_POTENTIAL_BASE)})`
    : String(spec.config.deedPotentialBase);
}

function formatNumber(value: number): string {
  return value.toFixed(4);
}
