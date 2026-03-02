import type { RefObject } from 'react';

import { BOT_PROFILES, type BotProfileId } from '../../policies/catalog';

export function OptionsMenu({
  open,
  botProfileId,
  botStatusText,
  animationsEnabled,
  menuRef,
  buttonRef,
  seedInputRef,
  onToggle,
  onReset,
  onBotProfileChange,
  onAnimationsEnabledChange,
}: {
  open: boolean;
  botProfileId: BotProfileId;
  botStatusText: string;
  animationsEnabled: boolean;
  menuRef: RefObject<HTMLElement | null>;
  buttonRef: RefObject<HTMLButtonElement | null>;
  seedInputRef: RefObject<HTMLInputElement | null>;
  onToggle: () => void;
  onReset: () => void;
  onBotProfileChange: (id: BotProfileId) => void;
  onAnimationsEnabledChange: (enabled: boolean) => void;
}) {
  return (
    <div className="corner-options-anchor">
      <button
        ref={buttonRef}
        type="button"
        className={`hamburger-button${open ? ' is-open' : ''}`}
        aria-label="Game options"
        aria-controls="brand-options-menu"
        aria-expanded={open}
        onClick={onToggle}
      >
        <span />
        <span />
        <span />
      </button>

      {open ? (
        <section
          id="brand-options-menu"
          ref={menuRef}
          className="brand-options-menu"
          aria-label="Game options"
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
            <button className="reset-button" type="button" onClick={onReset}>
              New Game
            </button>
          </div>
          <div className="bot-profile-controls">
            <label htmlFor="bot-profile-select">Bot Profile</label>
            <select
              id="bot-profile-select"
              className="bot-profile-select"
              value={botProfileId}
              onChange={(event) =>
                onBotProfileChange(event.target.value as BotProfileId)
              }
            >
              {BOT_PROFILES.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.label}
                </option>
              ))}
            </select>
            <p className="bot-profile-note">{botStatusText}</p>
          </div>
          <div className="bot-profile-controls animation-controls">
            <label className="animation-toggle-row" htmlFor="animations-toggle">
              <span>Animations</span>
              <input
                id="animations-toggle"
                type="checkbox"
                checked={animationsEnabled}
                onChange={(event) =>
                  onAnimationsEnabledChange(event.target.checked)
                }
              />
            </label>
          </div>
        </section>
      ) : null}
    </div>
  );
}

export function OptionsBackdrop({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  return open ? (
    <div className="options-backdrop" aria-hidden="true" onClick={onClose} />
  ) : null;
}
