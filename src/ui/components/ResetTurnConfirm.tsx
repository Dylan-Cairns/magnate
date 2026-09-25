import { useEffect, useRef } from 'react';

/**
 * Confirmation for the turn reset. The reset rewinds the game state to the
 * anchor captured at the start of the human action window, so it undoes the
 * turn's actions and returns any card played to the hand; the confirm keeps a
 * misclick from doing all of that silently.
 */
export function ResetTurnConfirm({
  open,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const cancelRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onCancel]);

  // Cancel takes focus: the destructive choice should never be one Enter away.
  useEffect(() => {
    if (open) cancelRef.current?.focus();
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="reset-turn-confirm-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="reset-turn-confirm-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="panel reset-turn-confirm">
        <h2 id="reset-turn-confirm-title">Reset turn?</h2>
        <p className="reset-turn-confirm-message">
          This undoes everything you have done this turn and returns the board
          to the start of your turn.
        </p>
        <div className="reset-turn-confirm-actions">
          <button
            ref={cancelRef}
            type="button"
            className="reset-turn-confirm-cancel"
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            type="button"
            className="reset-turn-confirm-reset"
            onClick={onConfirm}
          >
            Reset turn
          </button>
        </div>
      </div>
    </div>
  );
}
