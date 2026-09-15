export type HandCardPosition = {
  left: number;
};

export type HandCardSlide = {
  cardId: string;
  dx: number;
};

const MIN_SLIDE_DISTANCE_PX = 0.5;

export function readHandCardPositions(
  elements: Iterable<HTMLElement>
): Map<string, HandCardPosition> {
  const positions = new Map<string, HandCardPosition>();
  for (const element of elements) {
    const cardId = element.getAttribute('data-hand-card-id');
    if (!cardId) {
      continue;
    }
    positions.set(cardId, { left: element.getBoundingClientRect().left });
  }
  return positions;
}

export function deriveHandCardSlides(
  previousPositions: ReadonlyMap<string, HandCardPosition>,
  nextPositions: ReadonlyMap<string, HandCardPosition>
): HandCardSlide[] {
  const slides: HandCardSlide[] = [];
  for (const [cardId, position] of nextPositions) {
    const previous = previousPositions.get(cardId);
    if (!previous) {
      continue;
    }
    const dx = previous.left - position.left;
    if (Math.abs(dx) < MIN_SLIDE_DISTANCE_PX) {
      continue;
    }
    slides.push({ cardId, dx });
  }
  return slides;
}
