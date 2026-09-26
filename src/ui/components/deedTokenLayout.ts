import type { CardId } from '../../engine/cards';
import type { Suit } from '../../engine/types';

export type DeedTokenPerspective = 'human' | 'bot';

export type DeedTokenEntry = { suit: Suit; count: number };
export type DeedTokenSide = 'left' | 'right';
export type DeedTokenLayout = {
  left: DeedTokenEntry[];
  right: DeedTokenEntry[];
};

type LayoutMemory = {
  sideBySuit: Partial<Record<Suit, DeedTokenSide>>;
  orderBySuit: Partial<Record<Suit, number>>;
  nextOrder: number;
};

type LayoutOptions = { resetWhenEmpty?: boolean };

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

function emptyLayoutMemory(): LayoutMemory {
  return { sideBySuit: {}, orderBySuit: {}, nextOrder: 0 };
}

function readLayoutMemory(
  cardId: CardId,
  perspective: DeedTokenPerspective
): LayoutMemory | undefined {
  return LAYOUT_MEMORY_BY_CARD.get(memoryKey(cardId, perspective));
}

function ensureLayoutMemory(
  cardId: CardId,
  perspective: DeedTokenPerspective
): LayoutMemory {
  const key = memoryKey(cardId, perspective);
  const existing = LAYOUT_MEMORY_BY_CARD.get(key);
  if (existing) {
    return existing;
  }

  const created = emptyLayoutMemory();
  LAYOUT_MEMORY_BY_CARD.set(key, created);
  return created;
}

function cloneLayoutMemory(memory: LayoutMemory): LayoutMemory {
  return {
    sideBySuit: { ...memory.sideBySuit },
    orderBySuit: { ...memory.orderBySuit },
    nextOrder: memory.nextOrder,
  };
}

export function resetDeedTokenLayout(
  cardId: CardId,
  perspective: DeedTokenPerspective
): void {
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

function applyEntriesToMemory(
  memory: LayoutMemory,
  perspective: DeedTokenPerspective,
  entries: readonly DeedTokenEntry[]
): void {
  for (const entry of entries) {
    if (!memory.sideBySuit[entry.suit]) {
      memory.sideBySuit[entry.suit] = assignSideForNewSuit(
        entries,
        memory,
        perspective
      );
    }
    if (memory.orderBySuit[entry.suit] === undefined) {
      memory.orderBySuit[entry.suit] = memory.nextOrder;
      memory.nextOrder += 1;
    }
  }
}

function layoutFromMemory(
  memory: LayoutMemory,
  entries: readonly DeedTokenEntry[]
): DeedTokenLayout {
  const sortedByFirstSeen = [...entries].sort(
    (a, b) =>
      (memory.orderBySuit[a.suit] ?? 0) - (memory.orderBySuit[b.suit] ?? 0)
  );

  return {
    left: sortedByFirstSeen.filter(
      (entry) => memory.sideBySuit[entry.suit] === 'left'
    ),
    right: sortedByFirstSeen.filter(
      (entry) => memory.sideBySuit[entry.suit] === 'right'
    ),
  };
}

/**
 * Pure layout for render: reads the shared side/order memory but never writes
 * it. New suits are assigned against a copy so React can render repeatedly
 * (including discarded or double-invoked renders) without corrupting the
 * memory that animation flight planning also reads.
 */
export function planDeedTokenLayout(
  cardId: CardId,
  perspective: DeedTokenPerspective,
  entries: readonly DeedTokenEntry[],
  options?: LayoutOptions
): DeedTokenLayout {
  if (options?.resetWhenEmpty && entries.length === 0) {
    return { left: [], right: [] };
  }

  const stored = readLayoutMemory(cardId, perspective);
  const memory = stored ? cloneLayoutMemory(stored) : emptyLayoutMemory();
  applyEntriesToMemory(memory, perspective, entries);
  return layoutFromMemory(memory, entries);
}

/**
 * Record any newly seen suits in the shared memory and return the layout.
 * Callable outside render (effects, animation planning); component render uses
 * `planDeedTokenLayout` plus `useDeedTokenLayout`'s post-render commit.
 */
export function commitDeedTokenLayout(
  cardId: CardId,
  perspective: DeedTokenPerspective,
  entries: readonly DeedTokenEntry[],
  options?: LayoutOptions
): DeedTokenLayout {
  if (options?.resetWhenEmpty && entries.length === 0) {
    resetDeedTokenLayout(cardId, perspective);
    return { left: [], right: [] };
  }

  const memory = ensureLayoutMemory(cardId, perspective);
  applyEntriesToMemory(memory, perspective, entries);
  return layoutFromMemory(memory, entries);
}

/**
 * Mutating entry point for non-render callers that need the assignments
 * recorded immediately (animation flight planning).
 */
export function layoutDeedTokensBySide(
  cardId: CardId,
  perspective: DeedTokenPerspective,
  entries: readonly DeedTokenEntry[],
  options?: LayoutOptions
): DeedTokenLayout {
  return commitDeedTokenLayout(cardId, perspective, entries, options);
}
