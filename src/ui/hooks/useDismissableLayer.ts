import { useEffect, useEffectEvent, type RefObject } from 'react';

type UseDismissableLayerOptions = {
  enabled: boolean;
  onDismiss: () => void;
  insideRefs: ReadonlyArray<RefObject<Element | null>>;
  closeOnScroll?: boolean;
};

function containsNode(ref: RefObject<Element | null>, target: Node): boolean {
  return Boolean(ref.current?.contains(target));
}

export function useDismissableLayer({
  enabled,
  onDismiss,
  insideRefs,
  closeOnScroll = false,
}: UseDismissableLayerOptions): void {
  // The listeners are registered once per enabled/guard change; the event
  // callback always sees the latest `onDismiss` without re-subscribing.
  const dismiss = useEffectEvent(onDismiss);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }

      if (insideRefs.some((ref) => containsNode(ref, target))) {
        return;
      }

      dismiss();
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        dismiss();
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleEscape);

    if (closeOnScroll) {
      window.addEventListener('scroll', dismiss, true);
    }

    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleEscape);

      if (closeOnScroll) {
        window.removeEventListener('scroll', dismiss, true);
      }
    };
  }, [closeOnScroll, enabled, insideRefs]);
}
