import { useEffect } from 'react';

import type { CardId } from '../../engine/cards';
import {
  commitDeedTokenLayout,
  planDeedTokenLayout,
  type DeedTokenEntry,
  type DeedTokenLayout,
  type DeedTokenPerspective,
} from './deedTokenLayout';

/**
 * Render-safe deed token layout. The returned layout is a pure read of the
 * shared side/order memory; newly seen suits are recorded in a post-render
 * effect so component render never mutates module state.
 */
export function useDeedTokenLayout(
  cardId: CardId,
  perspective: DeedTokenPerspective,
  entries: readonly DeedTokenEntry[],
  options?: { resetWhenEmpty?: boolean }
): DeedTokenLayout {
  const layout = planDeedTokenLayout(cardId, perspective, entries, options);

  useEffect(() => {
    commitDeedTokenLayout(cardId, perspective, entries, options);
  });

  return layout;
}
