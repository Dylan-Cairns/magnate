import { useEffect, useLayoutEffect, useRef } from 'react';
import type { RefObject } from 'react';

import {
  deriveHandCardSlides,
  readHandCardPositions,
  type HandCardPosition,
} from '../animations/handSlide';
import { HAND_SLIDE_DURATION_MS } from '../animations/timing';

export type HandSlideAnimationOptions = {
  containerRef: RefObject<HTMLElement | null>;
  cardIds: readonly string[];
  enabled: boolean;
  resetKey?: string;
  durationMs?: number;
};

export function useHandSlideAnimation({
  containerRef,
  cardIds,
  enabled,
  resetKey,
  durationMs = HAND_SLIDE_DURATION_MS,
}: HandSlideAnimationOptions): void {
  const positionsRef = useRef<Map<string, HandCardPosition> | null>(null);
  const resetKeyRef = useRef(resetKey);
  const animationsRef = useRef(new Map<string, Animation>());
  const handKey = cardIds.join('|');

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) {
      positionsRef.current = null;
      return;
    }

    const elements = Array.from(
      container.querySelectorAll<HTMLElement>('[data-hand-card-id]')
    );
    const measured = readHandCardPositions(elements);
    const isNewGame = resetKeyRef.current !== resetKey;
    resetKeyRef.current = resetKey;
    if (isNewGame) {
      for (const animation of animationsRef.current.values()) {
        animation.cancel();
      }
      animationsRef.current.clear();
    }
    const previous = isNewGame ? null : positionsRef.current;
    positionsRef.current = measured;

    if (!enabled || previous === null) {
      return;
    }

    const elementByCardId = new Map<string, HTMLElement>();
    for (const element of elements) {
      const cardId = element.getAttribute('data-hand-card-id');
      if (cardId) {
        elementByCardId.set(cardId, element);
      }
    }

    for (const slide of deriveHandCardSlides(previous, measured)) {
      const element = elementByCardId.get(slide.cardId);
      if (!element || typeof element.animate !== 'function') {
        continue;
      }
      animationsRef.current.get(slide.cardId)?.cancel();
      const animation = element.animate(
        [
          { transform: `translateX(${slide.dx}px)` },
          { transform: 'translateX(0)' },
        ],
        {
          duration: durationMs,
          easing: 'ease-out',
        }
      );
      animationsRef.current.set(slide.cardId, animation);
      animation.addEventListener('finish', () => {
        if (animationsRef.current.get(slide.cardId) === animation) {
          animationsRef.current.delete(slide.cardId);
        }
      });
    }
  }, [containerRef, durationMs, enabled, handKey, resetKey]);

  useEffect(() => {
    const animations = animationsRef.current;
    const refreshPositions = () => {
      const container = containerRef.current;
      if (!container) {
        return;
      }
      positionsRef.current = readHandCardPositions(
        container.querySelectorAll<HTMLElement>('[data-hand-card-id]')
      );
    };
    window.addEventListener('resize', refreshPositions);
    return () => {
      window.removeEventListener('resize', refreshPositions);
      for (const animation of animations.values()) {
        animation.cancel();
      }
      animations.clear();
    };
  }, [containerRef]);
}
