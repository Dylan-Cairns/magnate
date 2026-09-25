import type { Card } from '../engine/types';

// The Darkness is canonically a blank Decktet card. A blank face reads as a
// broken image in the digital UI, so certain cards carry a presentational face
// effect instead of relying on their flat artwork. Keep this catalog keyed by
// the stable card name; nothing here changes engine behavior.
export type CardFaceEffect = 'darkness';

const FACE_EFFECT_BY_CARD_NAME: ReadonlyMap<string, CardFaceEffect> = new Map([
  ['The Darkness', 'darkness'],
]);

export function cardFaceEffect(
  card: Pick<Card, 'name'>
): CardFaceEffect | null {
  return FACE_EFFECT_BY_CARD_NAME.get(card.name) ?? null;
}
