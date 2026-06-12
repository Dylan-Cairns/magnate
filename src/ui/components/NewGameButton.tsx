import type { RefObject } from 'react';

import { BOT_PROFILES, type BotProfileId } from '../../policies/catalog';

export function NewGameButton({
  expanded,
  panelRef,
  buttonRef,
  seedInputRef,
  botProfileId,
  botStatusText,
  onToggle,
  onBotProfileChange,
}: {
  expanded: boolean;
  panelRef: RefObject<HTMLElement | null>;
  buttonRef: RefObject<HTMLButtonElement | null>;
  seedInputRef: RefObject<HTMLInputElement | null>;
  botProfileId: BotProfileId;
  botStatusText: string;
  onToggle: () => void;
  onBotProfileChange: (id: BotProfileId) => void;
}) {
  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className={`new-game-btn${expanded ? ' is-ready' : ''}`}
        aria-expanded={expanded}
        aria-controls="new-game-panel"
        onClick={onToggle}
      >
        {expanded ? 'Start' : 'New Game'}
      </button>
      {expanded && (
        <section
          id="new-game-panel"
          ref={panelRef as RefObject<HTMLElement>}
          className="new-game-panel"
          aria-label="New game options"
        >
          <div className="brand-controls">
            <input
              id="seed-input"
              aria-label="Seed"
              className="seed-input"
              ref={seedInputRef}
              autoComplete="off"
              defaultValue=""
              placeholder="seed (blank=random)"
            />
          </div>
          <div className="bot-profile-controls">
            <label htmlFor="bot-profile-select">Bot Profile</label>
            <select
              id="bot-profile-select"
              className="bot-profile-select"
              value={botProfileId}
              onChange={(e) => onBotProfileChange(e.target.value as BotProfileId)}
            >
              {BOT_PROFILES.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.label}
                </option>
              ))}
            </select>
            <p className="bot-profile-note">{botStatusText}</p>
          </div>
        </section>
      )}
    </>
  );
}
