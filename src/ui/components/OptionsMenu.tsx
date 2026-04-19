import type { RefObject } from 'react';

import { BOT_PROFILES, type BotProfileId } from '../../policies/catalog';
import { BugReportModal } from './BugReportModal';

export function OptionsMenu({
  open,
  botProfileId,
  botStatusText,
  animationsEnabled,
  menuRef,
  buttonRef,
  seedInputRef,
  onBugReport,
  onToggle,
  onReset,
  onBotProfileChange,
  onAnimationsEnabledChange,
  bugReportOpen,
  bugReportIssueUrl,
  onBugReportDownload,
}: {
  open: boolean;
  botProfileId: BotProfileId;
  botStatusText: string;
  animationsEnabled: boolean;
  menuRef: RefObject<HTMLElement | null>;
  buttonRef: RefObject<HTMLButtonElement | null>;
  seedInputRef: RefObject<HTMLInputElement | null>;
  onBugReport: () => void;
  onToggle: () => void;
  onReset: () => void;
  onBotProfileChange: (id: BotProfileId) => void;
  onAnimationsEnabledChange: (enabled: boolean) => void;
  bugReportOpen: boolean;
  bugReportIssueUrl: string;
  onBugReportDownload: () => void;
}) {
  return (
    <div className="corner-options-anchor">
      <BugReportModal
        open={bugReportOpen}
        issueUrl={bugReportIssueUrl}
        onDownload={onBugReportDownload}
      />
      <button
        type="button"
        className={`bug-report-button${bugReportOpen ? ' is-open' : ''}`}
        aria-label={bugReportOpen ? 'Close bug report' : 'Report a bug'}
        onClick={onBugReport}
      >
        <BugIcon />
        <span className="close-x" aria-hidden="true"><span /><span /></span>
      </button>
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
        <span className="close-x" aria-hidden="true"><span /><span /></span>
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

function BugIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className="bug-report-button-icon"
    >
      <path d="M8.4 7.5A3.8 3.8 0 0 1 12 5a3.8 3.8 0 0 1 3.6 2.5" />
      <path d="M9 5 7.5 3.2" />
      <path d="m15 5 1.5-1.8" />
      <path d="M7.5 10h9" />
      <path d="M7.8 14.5h8.4" />
      <path d="M8 10.2C8 8.4 9.8 7 12 7s4 1.4 4 3.2V15c0 2.2-1.8 4-4 4s-4-1.8-4-4z" />
      <path d="M6 11H3.8" />
      <path d="M6.6 15.2 4.4 16.4" />
      <path d="M18 11h2.2" />
      <path d="m17.4 15.2 2.2 1.2" />
    </svg>
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
