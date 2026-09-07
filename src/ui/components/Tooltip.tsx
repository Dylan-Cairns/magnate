import {
  autoUpdate,
  flip,
  FloatingPortal,
  offset,
  shift,
  useFloating,
} from '@floating-ui/react';
import { useEffect, useId, useState, type ReactNode } from 'react';

/**
 * The parent must have the `tooltip-trigger` class. The component uses a
 * zero-size anchor to find that parent, then portals the visible bubble to
 * the document root so it cannot expand the page's scrollable area.
 */
export function Tooltip({
  children,
  placement = 'above',
}: {
  children: ReactNode;
  placement?: 'above' | 'below' | 'below-left';
}) {
  const [anchor, setAnchor] = useState<HTMLSpanElement | null>(null);
  const [floating, setFloating] = useState<HTMLSpanElement | null>(null);
  const [open, setOpen] = useState(false);
  const tooltipId = useId();
  const { floatingStyles, refs } = useFloating({
    open,
    onOpenChange: setOpen,
    placement:
      placement === 'below-left'
        ? 'bottom-end'
        : placement === 'below'
          ? 'bottom'
          : 'top',
    strategy: 'fixed',
    middleware: [offset(7), flip({ padding: 8 }), shift({ padding: 8 })],
    whileElementsMounted: autoUpdate,
  });

  useEffect(() => {
    refs.setFloating(floating);
  }, [floating, refs]);

  useEffect(() => {
    const trigger = anchor?.parentElement;
    if (!trigger) return;

    refs.setReference(trigger);
    const previousDescribedBy = trigger.getAttribute('aria-describedby');
    const describedBy = previousDescribedBy
      ? `${previousDescribedBy} ${tooltipId}`
      : tooltipId;
    trigger.setAttribute('aria-describedby', describedBy);

    const show = () => setOpen(true);
    const hide = () => setOpen(false);
    trigger.addEventListener('mouseenter', show);
    trigger.addEventListener('mouseleave', hide);
    trigger.addEventListener('focusin', show);
    trigger.addEventListener('focusout', hide);

    return () => {
      trigger.removeEventListener('mouseenter', show);
      trigger.removeEventListener('mouseleave', hide);
      trigger.removeEventListener('focusin', show);
      trigger.removeEventListener('focusout', hide);
      if (previousDescribedBy === null) {
        trigger.removeAttribute('aria-describedby');
      } else {
        trigger.setAttribute('aria-describedby', previousDescribedBy);
      }
      refs.setReference(null);
    };
  }, [anchor, refs, tooltipId]);

  return (
    <>
      <span
        ref={setAnchor}
        className={`tooltip-anchor${placement !== 'above' ? ' tooltip-below' : ''}`}
      />
      {open ? (
        <FloatingPortal>
          <span
            ref={setFloating}
            id={tooltipId}
            className="app-tooltip"
            role="tooltip"
            style={floatingStyles}
          >
            {children}
          </span>
        </FloatingPortal>
      ) : null}
    </>
  );
}
