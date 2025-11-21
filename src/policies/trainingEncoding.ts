import type { GameAction, PlayerId, PlayerView, Suit } from '../engine/types';

const SUITS: readonly Suit[] = ['Moons', 'Suns', 'Waves', 'Leaves', 'Wyrms', 'Knots'];
const SUIT_INDEX = new Map<Suit, number>(SUITS.map((suit, index) => [suit, index]));

const PHASES = [
  'StartTurn',
  'TaxCheck',
  'CollectIncome',
  'ActionWindow',
  'DrawCard',
  'GameOver',
] as const;
const PHASE_INDEX = new Map<string, number>(PHASES.map((phase, index) => [phase, index]));

const ACTION_IDS = [
  'buy-deed',
  'choose-income-suit',
  'develop-deed',
  'develop-outright',
  'end-turn',
  'sell-card',
  'trade',
] as const;
const ACTION_ID_INDEX = new Map<string, number>(
  ACTION_IDS.map((actionId, index) => [actionId, index])
);

const PLAYER_IDS: readonly PlayerId[] = ['PlayerA', 'PlayerB'];
const PLAYER_INDEX = new Map<PlayerId, number>(PLAYER_IDS.map((id, index) => [id, index]));

const CROWN_SUIT_BY_CARD_ID: Readonly<Record<string, Suit>> = {
  '30': 'Knots',
  '31': 'Leaves',
  '32': 'Moons',
  '33': 'Suns',
  '34': 'Waves',
  '35': 'Wyrms',
};

const MAX_CARD_ID = 40.0;
const MAX_TURN = 40.0;
const MAX_DECK_SIZE = 41.0;
const MAX_RESOURCES = 20.0;
const MAX_HAND_COUNT = 10.0;
const MAX_DISTRICT_STACK = 9.0;
const MAX_DISTRICT_RANK_SUM = 60.0;
const MAX_DEED_PROGRESS = 9.0;
const MAX_TOKEN_COUNT = 9.0;

export const ENCODING_VERSION = 2;
export const OBSERVATION_DIM = 206;
export const ACTION_FEATURE_DIM = 40;

const PROPERTY_SUITS_BY_CARD_ID: Readonly<Record<string, readonly Suit[]>> = {
  '0': ['Knots'],
  '1': ['Leaves'],
  '2': ['Moons'],
  '3': ['Suns'],
  '4': ['Waves'],
  '5': ['Wyrms'],
  '6': ['Moons', 'Knots'],
  '7': ['Suns', 'Wyrms'],
  '8': ['Waves', 'Leaves'],
  '9': ['Moons', 'Waves'],
  '10': ['Suns', 'Knots'],
  '11': ['Leaves', 'Wyrms'],
  '12': ['Wyrms', 'Knots'],
  '13': ['Moons', 'Suns'],
  '14': ['Waves', 'Leaves'],
  '15': ['Suns', 'Waves'],
  '16': ['Moons', 'Leaves'],
  '17': ['Wyrms', 'Knots'],
  '18': ['Moons', 'Waves'],
  '19': ['Leaves', 'Knots'],
  '20': ['Suns', 'Wyrms'],
  '21': ['Suns', 'Knots'],
  '22': ['Waves', 'Wyrms'],
  '23': ['Moons', 'Leaves'],
  '24': ['Wyrms', 'Knots'],
  '25': ['Moons', 'Suns'],
  '26': ['Waves', 'Leaves'],
  '27': ['Waves', 'Wyrms'],
  '28': ['Leaves', 'Knots'],
  '29': ['Moons', 'Suns'],
};

export function encodeObservation(view: PlayerView): number[] {
  const activePlayerId = view.activePlayerId;
  const opponentId: PlayerId = activePlayerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
  const playersById = new Map(view.players.map((player) => [player.id, player]));
  const activePlayer = playersById.get(activePlayerId);
  const opponentPlayer = playersById.get(opponentId);

  const vector: number[] = [];
  vector.push(...oneHot(PHASE_INDEX.get(view.phase) ?? -1, PHASES.length));
  vector.push(norm(view.turn, MAX_TURN));
  vector.push(view.cardPlayedThisTurn ? 1.0 : 0.0);
  vector.push(norm(view.finalTurnsRemaining ?? 0, 2.0));
  vector.push(norm(view.deck.drawCount, MAX_DECK_SIZE));
  vector.push(norm(view.deck.discard.length, MAX_DECK_SIZE));
  vector.push(norm(view.deck.reshuffles, 2.0));
  vector.push(norm(view.lastIncomeRoll?.die1 ?? 0, 10.0));
  vector.push(norm(view.lastIncomeRoll?.die2 ?? 0, 10.0));

  vector.push(...suitOneHot(view.lastTaxSuit));
  vector.push(...resourceVector(activePlayer?.resources));
  vector.push(...resourceVector(opponentPlayer?.resources));

  vector.push(norm(activePlayer?.handCount ?? 0, MAX_HAND_COUNT));
  vector.push(norm(opponentPlayer?.handCount ?? 0, MAX_HAND_COUNT));
  vector.push(...crownSuitCounts(activePlayer?.crowns));
  vector.push(...crownSuitCounts(opponentPlayer?.crowns));
  vector.push(...handSuitHistogram(activePlayer?.hand));
  vector.push(...handRankHistogram(activePlayer?.hand));
  vector.push(
    ...endgameTiebreakFeatures({
      view,
      activePlayerId,
      opponentId,
      activePlayer,
      opponentPlayer,
    })
  );

  const districts = [...view.districts].sort((a, b) => a.id.localeCompare(b.id));
  for (const district of districts) {
    vector.push(...suitCountVector(district.markerSuitMask, 3.0));
    vector.push(...districtStackFeatures(district.stacks[activePlayerId]));
    vector.push(...districtStackFeatures(district.stacks[opponentId]));
  }

  if (vector.length !== OBSERVATION_DIM) {
    throw new Error(
      `Observation vector length mismatch. expected=${OBSERVATION_DIM}, actual=${vector.length}`
    );
  }
  return vector;
}

export function encodeActionCandidates(actions: readonly GameAction[]): number[][] {
  return actions.map((action) => encodeAction(action));
}

export function encodeAction(action: GameAction): number[] {
  const payload = action as unknown as Partial<Record<string, unknown>>;
  const vector: number[] = [];

  vector.push(...oneHot(ACTION_ID_INDEX.get(action.type) ?? -1, ACTION_IDS.length));

  const cardId = asString(payload.cardId);
  const cardRank = cardRankFromId(cardId);
  vector.push(norm(cardNumericId(cardId), MAX_CARD_ID));
  vector.push(norm(cardRank, 10.0));

  const districtId = asString(payload.districtId);
  vector.push(norm(districtIndex(districtId), 5.0));

  const playerId = payload.playerId;
  const playerIndex =
    playerId === 'PlayerA' || playerId === 'PlayerB' ? PLAYER_INDEX.get(playerId) ?? -1 : -1;
  vector.push(...oneHot(playerIndex, 2));

  vector.push(...suitOneHot(suitOrUndefined(payload.suit)));
  vector.push(...suitOneHot(suitOrUndefined(payload.give)));
  vector.push(...suitOneHot(suitOrUndefined(payload.receive)));

  const tokenMap = mapFromUnknown(payload.payment) ?? mapFromUnknown(payload.tokens) ?? {};
  const tokenVector = resourceVector(tokenMap, MAX_TOKEN_COUNT);
  vector.push(...tokenVector);
  vector.push(norm(sumResourceMap(tokenMap), MAX_TOKEN_COUNT));

  vector.push(cardId ? 1.0 : 0.0);
  vector.push(districtId ? 1.0 : 0.0);
  vector.push(isPropertyCard(cardId) ? 1.0 : 0.0);

  if (vector.length !== ACTION_FEATURE_DIM) {
    throw new Error(
      `Action feature length mismatch. expected=${ACTION_FEATURE_DIM}, actual=${vector.length}`
    );
  }
  return vector;
}

function districtStackFeatures(stack: {
  developed: string[];
  deed?: { cardId: string; progress: number; tokens: Partial<Record<Suit, number>> };
}): number[] {
  const developedRanks = stack.developed
    .filter((cardId) => isPropertyCard(cardId))
    .map((cardId) => cardRankFromId(cardId));
  const developedCount = stack.developed.length;
  const developedRankSum = developedRanks.reduce((sum, rank) => sum + rank, 0);

  const deedCardId = stack.deed?.cardId ?? '';
  const deedProgress = stack.deed?.progress ?? 0;
  const deedTarget = developmentTarget(deedCardId);
  const deedTokens = stack.deed?.tokens ?? {};

  const features: number[] = [
    norm(developedCount, MAX_DISTRICT_STACK),
    norm(developedRankSum, MAX_DISTRICT_RANK_SUM),
    stack.deed ? 1.0 : 0.0,
    norm(deedProgress, MAX_DEED_PROGRESS),
    norm(deedTarget, MAX_DEED_PROGRESS),
  ];
  features.push(...resourceVector(deedTokens, MAX_TOKEN_COUNT));
  return features;
}

function crownSuitCounts(crowns: readonly string[] | undefined): number[] {
  const counts: Record<Suit, number> = {
    Moons: 0,
    Suns: 0,
    Waves: 0,
    Leaves: 0,
    Wyrms: 0,
    Knots: 0,
  };
  for (const cardId of crowns ?? []) {
    const suit = CROWN_SUIT_BY_CARD_ID[cardId];
    if (suit) {
      counts[suit] += 1;
    }
  }
  return SUITS.map((suit) => norm(counts[suit], 3.0));
}

function handSuitHistogram(hand: readonly string[] | undefined): number[] {
  const counts: Record<Suit, number> = {
    Moons: 0,
    Suns: 0,
    Waves: 0,
    Leaves: 0,
    Wyrms: 0,
    Knots: 0,
  };
  for (const cardId of hand ?? []) {
    for (const suit of PROPERTY_SUITS_BY_CARD_ID[cardId] ?? []) {
      counts[suit] += 1;
    }
  }
  return SUITS.map((suit) => norm(counts[suit], MAX_HAND_COUNT * 2.0));
}

function handRankHistogram(hand: readonly string[] | undefined): number[] {
  const counts = Array(10).fill(0);
  for (const cardId of hand ?? []) {
    const rank = cardRankFromId(cardId);
    if (rank >= 1 && rank <= 10) {
      counts[rank - 1] += 1;
    }
  }
  return counts.map((value) => norm(value, MAX_HAND_COUNT));
}

function endgameTiebreakFeatures({
  view,
  activePlayerId,
  opponentId,
  activePlayer,
  opponentPlayer,
}: {
  view: PlayerView;
  activePlayerId: PlayerId;
  opponentId: PlayerId;
  activePlayer:
    | {
        resources: Partial<Record<Suit, number>>;
      }
    | undefined;
  opponentPlayer:
    | {
        resources: Partial<Record<Suit, number>>;
      }
    | undefined;
}): number[] {
  const endgameFlag =
    view.deck.reshuffles >= 2 || (view.finalTurnsRemaining ?? 0) > 0 ? 1.0 : 0.0;

  let districtPointDiff = 0.0;
  let developedRankDiff = 0.0;
  for (const district of view.districts) {
    const activeStack = district.stacks[activePlayerId];
    const opponentStack = district.stacks[opponentId];
    const activeRank = developedRankSum(activeStack);
    const opponentRank = developedRankSum(opponentStack);
    developedRankDiff += activeRank - opponentRank;
    if (activeRank > opponentRank) {
      districtPointDiff += 1.0;
    } else if (activeRank < opponentRank) {
      districtPointDiff -= 1.0;
    }
  }

  const districtTerm = districtPointDiff / Math.max(1, view.districts.length);
  const rankTerm = Math.tanh(developedRankDiff / 18.0);
  const resourceTerm = Math.tanh(
    (resourceTotal(activePlayer?.resources) - resourceTotal(opponentPlayer?.resources)) / 10.0
  );
  return [
    endgameFlag,
    signedToUnitInterval(districtTerm),
    signedToUnitInterval(rankTerm),
    signedToUnitInterval(resourceTerm),
  ];
}

function developedRankSum(stack: { developed: string[] }): number {
  let total = 0;
  for (const cardId of stack.developed) {
    total += cardRankFromId(cardId);
  }
  return total;
}

function resourceVector(
  resourceMap: Partial<Record<Suit, number>> | Record<string, unknown> | undefined,
  normalizeBy: number = MAX_RESOURCES
): number[] {
  return SUITS.map((suit) => norm(asInt(resourceMap?.[suit]), normalizeBy));
}

function suitCountVector(suits: readonly Suit[], normalizeBy: number): number[] {
  const counts: Record<Suit, number> = {
    Moons: 0,
    Suns: 0,
    Waves: 0,
    Leaves: 0,
    Wyrms: 0,
    Knots: 0,
  };
  for (const suit of suits) {
    counts[suit] += 1;
  }
  return SUITS.map((suit) => norm(counts[suit], normalizeBy));
}

function suitOneHot(suit: Suit | undefined): number[] {
  const index = suit ? SUIT_INDEX.get(suit) ?? -1 : -1;
  return oneHot(index, SUITS.length);
}

function districtIndex(districtId: string): number {
  if (districtId.startsWith('D')) {
    const maybeNumber = Number.parseInt(districtId.slice(1), 10);
    if (Number.isFinite(maybeNumber)) {
      return maybeNumber;
    }
  }
  return 0;
}

function developmentTarget(cardId: string): number {
  if (!isPropertyCard(cardId)) {
    return 0;
  }
  const rank = cardRankFromId(cardId);
  return rank === 1 ? 3 : rank;
}

function isPropertyCard(cardId: string): boolean {
  const numeric = cardNumericId(cardId);
  return numeric >= 0 && numeric <= 29;
}

function cardNumericId(cardId: string): number {
  if (/^\d+$/.test(cardId)) {
    return Number.parseInt(cardId, 10);
  }
  return 0;
}

function cardRankFromId(cardId: string): number {
  const numeric = cardNumericId(cardId);
  if (numeric >= 0 && numeric <= 5) {
    return 1;
  }
  if (numeric >= 6 && numeric <= 29) {
    return 2 + Math.floor((numeric - 6) / 3);
  }
  if (numeric >= 30 && numeric <= 35) {
    return 10;
  }
  return 0;
}

function norm(value: number, ceiling: number): number {
  if (ceiling <= 0) {
    return 0.0;
  }
  const clipped = Math.max(0.0, Math.min(Number(value), ceiling));
  return clipped / ceiling;
}

function signedToUnitInterval(value: number): number {
  const clipped = Math.max(-1.0, Math.min(value, 1.0));
  return (clipped + 1.0) * 0.5;
}

function oneHot(index: number, length: number): number[] {
  const out = Array(length).fill(0.0);
  if (index >= 0 && index < length) {
    out[index] = 1.0;
  }
  return out;
}

function asInt(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === 'boolean') {
    return value ? 1 : 0;
  }
  return 0;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function suitOrUndefined(value: unknown): Suit | undefined {
  if (
    value === 'Moons' ||
    value === 'Suns' ||
    value === 'Waves' ||
    value === 'Leaves' ||
    value === 'Wyrms' ||
    value === 'Knots'
  ) {
    return value;
  }
  return undefined;
}

function mapFromUnknown(value: unknown): Record<string, unknown> | undefined {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return undefined;
}

function sumResourceMap(resourceMap: Record<string, unknown>): number {
  let total = 0;
  for (const value of Object.values(resourceMap)) {
    total += asInt(value);
  }
  return total;
}

function resourceTotal(resourceMap: Partial<Record<Suit, number>> | undefined): number {
  if (!resourceMap) {
    return 0;
  }
  let total = 0;
  for (const suit of SUITS) {
    total += asInt(resourceMap[suit]);
  }
  return total;
}
