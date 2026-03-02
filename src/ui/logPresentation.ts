import { CARD_BY_ID, type CardId } from '../engine/cards';
import type { GameLogEntry, PlayerId, Suit } from '../engine/types';

export type SuitLogCode = 'mo' | 'su' | 'wa' | 'le' | 'wy' | 'kn';

const SUIT_LOG_CODE: Record<Suit, SuitLogCode> = {
  Moons: 'mo',
  Suns: 'su',
  Waves: 'wa',
  Leaves: 'le',
  Wyrms: 'wy',
  Knots: 'kn',
};
const SUIT_NAME_PATTERN = /\b(Moons|Suns|Waves|Leaves|Wyrms|Knots)\b/g;
const CARD_ACTION_PATTERN = /\b(buy deed|sell|advance|develop)\s+(\d+)\b/gi;
const INCOME_CHOICE_PATTERN = /\bincome choice\s+(\d+):([A-Za-z]+)\b/gi;
export const SUIT_CODE_PATTERN = /\b(mo|su|wa|le|wy|kn)\b/g;

export function groupLogEntriesByTurn(
  entries: ReadonlyArray<GameLogEntry>
): ReadonlyArray<{
  turn: number;
  player: PlayerId;
  entries: ReadonlyArray<GameLogEntry>;
}> {
  const groups: Array<{
    turn: number;
    player: PlayerId;
    entries: GameLogEntry[];
  }> = [];

  for (const entry of entries) {
    const current = groups[groups.length - 1];
    if (!current || current.turn !== entry.turn) {
      groups.push({
        turn: entry.turn,
        player: entry.player,
        entries: [entry],
      });
      continue;
    }
    current.entries.push(entry);
    // `entries` are already reverse-chronological; keep header player aligned
    // to the oldest entry in the turn (turn-owner context) instead of the
    // newest cross-player income-choice action.
    current.player = entry.player;
  }

  return groups;
}

export function formatLogSummary(summary: string): string {
  let next = summary;

  next = next.replace(
    INCOME_CHOICE_PATTERN,
    (_match, rawCardId: string, rawSuit: string) => {
      const cardLabel = formatCardIdForLog(rawCardId);
      const suitCode = suitNameToCode(rawSuit);
      return `income choice ${cardLabel}:${suitCode ?? rawSuit}`;
    }
  );

  next = next.replace(
    CARD_ACTION_PATTERN,
    (_match, verb: string, rawCardId: string) =>
      `${verb} ${formatCardIdForLog(rawCardId)}`
  );

  next = next.replace(SUIT_NAME_PATTERN, (_match, suitName: string) => {
    const suit = suitName as Suit;
    return SUIT_LOG_CODE[suit] ?? suitName;
  });

  return sentenceCaseSummary(next);
}

export function seedSummaryValue(summary: string): string | null {
  const prefix = 'Seed ';
  if (!summary.startsWith(prefix)) {
    return null;
  }
  return summary.slice(prefix.length);
}

export function suitCodeToSuit(value: SuitLogCode): Suit {
  switch (value) {
    case 'mo':
      return 'Moons';
    case 'su':
      return 'Suns';
    case 'wa':
      return 'Waves';
    case 'le':
      return 'Leaves';
    case 'wy':
      return 'Wyrms';
    case 'kn':
      return 'Knots';
  }
}

function sentenceCaseSummary(summary: string): string {
  if (!summary) {
    return summary;
  }
  const prefixMatch = summary.match(/^(\[[^\]]+\]\s*)/);
  const prefix = prefixMatch?.[1] ?? '';
  const rest = summary.slice(prefix.length);
  if (rest.length === 0) {
    return summary;
  }
  return `${prefix}${rest.slice(0, 1).toUpperCase()}${rest.slice(1)}`;
}

function formatCardIdForLog(rawCardId: string): string {
  const card = CARD_BY_ID[rawCardId as CardId];
  if (!card) {
    return rawCardId;
  }
  if (card.kind !== 'Property' && card.kind !== 'Crown') {
    return rawCardId;
  }
  const suitCodes = card.suits.map((suit) => SUIT_LOG_CODE[suit]).join(' ');
  return `${card.rank} ${suitCodes} (${rawCardId})`;
}

function suitNameToCode(value: string): SuitLogCode | null {
  if (
    value !== 'Moons' &&
    value !== 'Suns' &&
    value !== 'Waves' &&
    value !== 'Leaves' &&
    value !== 'Wyrms' &&
    value !== 'Knots'
  ) {
    return null;
  }
  return SUIT_LOG_CODE[value];
}
