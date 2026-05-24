import type { RefObject } from 'react';

import type { BotProfileId } from '../../policies/catalog';
import { BugReportModal } from './BugReportModal';
import { NewGameButton } from './NewGameButton';

export function OptionsMenu({
  open,
  botProfileId,
  botStatusText,
  animationsEnabled,
  menuRef,
  buttonRef,
  seedInputRef,
  newGameExpanded,
  newGamePanelRef,
  newGameButtonRef,
  onBugReport,
  onToggle,
  onNewGameToggle,
  onBotProfileChange,
  onAnimationsEnabledChange,
  bugReportOpen,
  bugReportIssueUrl,
  onBugReportDownload,
  logVisible,
  onToggleLog,
  mapVisible,
  onToggleMap,
  deckMapInteractive,
  onDeckMapInteractiveChange,
  onHistoryOpen,
}: {
  open: boolean;
  botProfileId: BotProfileId;
  botStatusText: string;
  animationsEnabled: boolean;
  menuRef: RefObject<HTMLElement | null>;
  buttonRef: RefObject<HTMLButtonElement | null>;
  seedInputRef: RefObject<HTMLInputElement | null>;
  newGameExpanded: boolean;
  newGamePanelRef: RefObject<HTMLElement | null>;
  newGameButtonRef: RefObject<HTMLButtonElement | null>;
  onBugReport: () => void;
  onToggle: () => void;
  onNewGameToggle: () => void;
  onBotProfileChange: (id: BotProfileId) => void;
  onAnimationsEnabledChange: (enabled: boolean) => void;
  bugReportOpen: boolean;
  bugReportIssueUrl: string;
  onBugReportDownload: () => void;
  logVisible: boolean;
  onToggleLog: () => void;
  mapVisible: boolean;
  onToggleMap: () => void;
  deckMapInteractive: boolean;
  onDeckMapInteractiveChange: (enabled: boolean) => void;
  onHistoryOpen: () => void;
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
        className={`log-toggle-button${!mapVisible ? ' is-inactive' : ''}`}
        aria-label={mapVisible ? 'Hide deck map' : 'Show deck map'}
        aria-pressed={mapVisible}
        onClick={onToggleMap}
      >
        <MapIcon />
      </button>
      <button
        type="button"
        className={`log-toggle-button${!logVisible ? ' is-inactive' : ''}`}
        aria-label={logVisible ? 'Hide game log' : 'Show game log'}
        aria-pressed={logVisible}
        onClick={onToggleLog}
      >
        <LogIcon />
      </button>
      <button
        type="button"
        className="history-button"
        aria-label="Game history"
        onClick={onHistoryOpen}
      >
        <TrophyIcon />
      </button>
      <button
        type="button"
        className={`bug-report-button${bugReportOpen ? ' is-open' : ''}`}
        aria-label={bugReportOpen ? 'Close bug report' : 'Report a bug'}
        onClick={onBugReport}
      >
        <BugIcon />
        <span className="close-x" aria-hidden="true">
          <span />
          <span />
        </span>
      </button>
      <button
        ref={buttonRef}
        type="button"
        className={`hamburger-button${open ? ' is-open' : ''}`}
        aria-label="Settings"
        aria-controls="brand-options-menu"
        aria-expanded={open}
        onClick={onToggle}
      >
        <GearIcon />
        <span className="close-x" aria-hidden="true">
          <span />
          <span />
        </span>
      </button>
      <NewGameButton
        expanded={newGameExpanded}
        panelRef={newGamePanelRef}
        buttonRef={newGameButtonRef}
        seedInputRef={seedInputRef}
        botProfileId={botProfileId}
        botStatusText={botStatusText}
        onToggle={onNewGameToggle}
        onBotProfileChange={onBotProfileChange}
      />

      {open ? (
        <section
          id="brand-options-menu"
          ref={menuRef}
          className="brand-options-menu"
          aria-label="Settings"
        >
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
            <label
              className="animation-toggle-row"
              htmlFor="deck-map-interactive-toggle"
            >
              <span>Interactive deck map</span>
              <input
                id="deck-map-interactive-toggle"
                type="checkbox"
                checked={deckMapInteractive}
                onChange={(event) =>
                  onDeckMapInteractiveChange(event.target.checked)
                }
              />
            </label>
          </div>
        </section>
      ) : null}
    </div>
  );
}

function TrophyIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="history-button-icon">
      <path d="M6 9H4a2 2 0 0 1-2-2V5h4" />
      <path d="M18 9h2a2 2 0 0 0 2-2V5h-4" />
      <path d="M9 21h6" />
      <path d="M12 17v4" />
      <path d="M6 5h12v6a6 6 0 0 1-12 0z" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className="hamburger-button-icon"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function MapIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className="log-toggle-button-icon"
    >
      <path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3V6z" />
      <path d="M9 3v15" />
      <path d="M15 6v15" />
    </svg>
  );
}

function LogIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className="log-toggle-button-icon"
    >
      <path d="M3 3v5h5" />
      <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8" />
      <path d="M12 7v5l3 2" />
    </svg>
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
