import type { CardId } from '../../engine/cards';
import type { Suit } from '../../engine/types';

export type DeedTokenPerspective = 'human' | 'bot';

export type DeedTokenEntry = { suit: Suit; count: number };
export type DeedTokenSide = 'left' | 'right';
type SideLayout = {
  left: DeedTokenEntry[];
  right: DeedTokenEntry[];
};

type LayoutMemory = {
  sideBySuit: Partial<Record<Suit, DeedTokenSide>>;
  orderBySuit: Partial<Record<Suit, number>>;
  nextOrder: number;
};

const LAYOUT_MEMORY_BY_CARD = new Map<string, LayoutMemory>();

function memoryKey(cardId: CardId, perspective: DeedTokenPerspective): string {
  return `${perspective}:${cardId}`;
}

function defaultSide(perspective: DeedTokenPerspective): DeedTokenSide {
  return perspective === 'bot' ? 'right' : 'left';
}

function tieBreakSide(perspective: DeedTokenPerspective): DeedTokenSide {
  return perspective === 'bot' ? 'right' : 'left';
}

function ensureLayoutMemory(cardId: CardId, perspective: DeedTokenPerspective): LayoutMemory {
  const key = memoryKey(cardId, perspective);
  const existing = LAYOUT_MEMORY_BY_CARD.get(key);
  if (existing) {
    return existing;
  }

  const created: LayoutMemory = {
    sideBySuit: {},
    orderBySuit: {},
    nextOrder: 0,
  };
  LAYOUT_MEMORY_BY_CARD.set(key, created);
  return created;
}

export function resetDeedTokenLayout(cardId: CardId, perspective: DeedTokenPerspective): void {
  LAYOUT_MEMORY_BY_CARD.delete(memoryKey(cardId, perspective));
}

export function clearAllDeedTokenLayouts(): void {
  LAYOUT_MEMORY_BY_CARD.clear();
}

function assignedCountsForEntries(
  entries: readonly DeedTokenEntry[],
  sideBySuit: Partial<Record<Suit, DeedTokenSide>>
): { left: number; right: number } {
  let left = 0;
  let right = 0;
  for (const entry of entries) {
    const side = sideBySuit[entry.suit];
    if (side === 'left') {
      left += 1;
    } else if (side === 'right') {
      right += 1;
    }
  }
  return { left, right };
}

function assignSideForNewSuit(
  entries: readonly DeedTokenEntry[],
  memory: LayoutMemory,
  perspective: DeedTokenPerspective
): DeedTokenSide {
  const assignedCounts = assignedCountsForEntries(entries, memory.sideBySuit);
  if (assignedCounts.left === 0 && assignedCounts.right === 0) {
    return defaultSide(perspective);
  }
  if (assignedCounts.left === 0) {
    return 'left';
  }
  if (assignedCounts.right === 0) {
    return 'right';
  }
  if (assignedCounts.left < assignedCounts.right) {
    return 'left';
  }
  if (assignedCounts.right < assignedCounts.left) {
    return 'right';
  }
  return tieBreakSide(perspective);
}

export function layoutDeedTokensBySide(
  cardId: CardId,
  perspective: DeedTokenPerspective,
  entries: readonly DeedTokenEntry[],
  options?: { resetWhenEmpty?: boolean }
): SideLayout {
  if (options?.resetWhenEmpty && entries.length === 0) {
    resetDeedTokenLayout(cardId, perspective);
  }

  const memory = ensureLayoutMemory(cardId, perspective);

  for (const entry of entries) {
    if (!memory.sideBySuit[entry.suit]) {
      memory.sideBySuit[entry.suit] = assignSideForNewSuit(entries, memory, perspective);
    }
    if (memory.orderBySuit[entry.suit] === undefined) {
      memory.orderBySuit[entry.suit] = memory.nextOrder;
      memory.nextOrder += 1;
    }
  }

  const sortedByFirstSeen = [...entries].sort(
    (a, b) => (memory.orderBySuit[a.suit] ?? 0) - (memory.orderBySuit[b.suit] ?? 0)
  );

  return {
    left: sortedByFirstSeen.filter((entry) => memory.sideBySuit[entry.suit] === 'left'),
    right: sortedByFirstSeen.filter((entry) => memory.sideBySuit[entry.suit] === 'right'),
  };
}
