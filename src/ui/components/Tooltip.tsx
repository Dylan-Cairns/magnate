import type { ReactNode } from 'react';

/**
 * The shared, in-app tooltip bubble. Its parent must have the
 * `tooltip-trigger` class so the bubble can be positioned without adding a
 * layout-affecting wrapper around the trigger.
 */
export function Tooltip({
  children,
  placement = 'above',
}: {
  children: ReactNode;
  placement?: 'above' | 'below';
}) {
  return (
    <span
      className={`app-tooltip${placement === 'below' ? ' tooltip-below' : ''}`}
      role="tooltip"
    >
      {children}
    </span>
  );
}
