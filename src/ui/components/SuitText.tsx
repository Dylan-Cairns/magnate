import type { ReactNode } from 'react';

import { SUIT_TOKEN_REGEX, SUIT_TOKEN_TO_SUIT } from '../suitIcons';
import { TokenChip } from './TokenComponents';

export function SuitText({ text }: { text: string }) {
  if (!text) {
    return text;
  }

  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(SUIT_TOKEN_REGEX)) {
    const index = match.index ?? 0;
    const token = match[0];
    const suit = SUIT_TOKEN_TO_SUIT[token];

    if (index > cursor) {
      nodes.push(text.slice(cursor, index));
    }

    if (suit) {
      nodes.push(
        <TokenChip
          key={`suit-${index}-${suit}`}
          suit={suit}
          count={1}
          compact
          className="inline-token-chip"
        />
      );
    } else {
      nodes.push(token);
    }

    cursor = index + token.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes.length > 0 ? <>{nodes}</> : text;
}
