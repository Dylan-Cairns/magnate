import type { RefObject } from 'react';

import type { Ruleset } from '../../engine/types';
import type { BotProfileId } from '../../policies/catalog';
import { BugReportInstructions } from './BugReportInstructions';
import { GameCredits } from './GameCredits';
import { NewGameButton } from './NewGameButton';
import { Tooltip } from './Tooltip';

export function OptionsMenu({
  open,
  botProfileId,
  botStatusText,
  ruleset,
  animationsEnabled,
  menuRef,
  buttonRef,
  seedInputRef,
  newGameExpanded,
  newGamePanelRef,
  newGameButtonRef,
  onToggle,
  onNewGameToggle,
  onBotProfileChange,
  onRulesetChange,
  onAnimationsEnabledChange,
  bugReportIssueUrl,
  onBugReportDownload,
  logVisible,
  onToggleLog,
  mapVisible,
  onToggleMap,
  onHistoryOpen,
}: {
  open: boolean;
  botProfileId: BotProfileId;
  botStatusText: string;
  ruleset: Ruleset;
  animationsEnabled: boolean;
  menuRef: RefObject<HTMLElement | null>;
  buttonRef: RefObject<HTMLButtonElement | null>;
  seedInputRef: RefObject<HTMLInputElement | null>;
  newGameExpanded: boolean;
  newGamePanelRef: RefObject<HTMLElement | null>;
  newGameButtonRef: RefObject<HTMLButtonElement | null>;
  onToggle: () => void;
  onNewGameToggle: () => void;
  onBotProfileChange: (id: BotProfileId) => void;
  onRulesetChange: (ruleset: Ruleset) => void;
  onAnimationsEnabledChange: (enabled: boolean) => void;
  bugReportIssueUrl: string;
  onBugReportDownload: () => void;
  logVisible: boolean;
  onToggleLog: () => void;
  mapVisible: boolean;
  onToggleMap: () => void;
  onHistoryOpen: () => void;
}) {
  return (
    <div className="corner-options-anchor">
      <button
        type="button"
        className={`log-toggle-button tooltip-trigger${!mapVisible ? ' is-inactive' : ''}`}
        aria-label={mapVisible ? 'Hide deck map' : 'Show deck map'}
        aria-pressed={mapVisible}
        onClick={onToggleMap}
      >
        <MapIcon />
        <Tooltip>{mapVisible ? 'Hide deck map' : 'Show deck map'}</Tooltip>
      </button>
      <button
        type="button"
        className={`log-toggle-button tooltip-trigger${!logVisible ? ' is-inactive' : ''}`}
        aria-label={logVisible ? 'Hide game log' : 'Show game log'}
        aria-pressed={logVisible}
        onClick={onToggleLog}
      >
        <LogIcon />
        <Tooltip>{logVisible ? 'Hide game log' : 'Show game log'}</Tooltip>
      </button>
      <button
        type="button"
        className="history-button tooltip-trigger"
        aria-label="History"
        onClick={onHistoryOpen}
      >
        <TrophyIcon />
        <Tooltip>View history</Tooltip>
      </button>
      <button
        ref={buttonRef}
        type="button"
        className={`hamburger-button tooltip-trigger${open ? ' is-open' : ''}`}
        aria-label="Info"
        aria-controls="brand-options-menu"
        aria-expanded={open}
        onClick={onToggle}
      >
        <QuestionMarkIcon />
        <span className="close-x" aria-hidden="true">
          <span />
          <span />
        </span>
        <Tooltip>{open ? 'Close info' : 'Open info'}</Tooltip>
      </button>
      <NewGameButton
        expanded={newGameExpanded}
        panelRef={newGamePanelRef}
        buttonRef={newGameButtonRef}
        seedInputRef={seedInputRef}
        botProfileId={botProfileId}
        botStatusText={botStatusText}
        ruleset={ruleset}
        animationsEnabled={animationsEnabled}
        onToggle={onNewGameToggle}
        onBotProfileChange={onBotProfileChange}
        onRulesetChange={onRulesetChange}
        onAnimationsEnabledChange={onAnimationsEnabledChange}
      />

      {open ? (
        <section
          id="brand-options-menu"
          ref={menuRef}
          className="brand-options-menu"
          aria-label="Info"
        >
          <BugReportInstructions
            issueUrl={bugReportIssueUrl}
            onDownload={onBugReportDownload}
          />
          <GameCredits />
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

function QuestionMarkIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className="hamburger-button-icon"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <path d="M12 17h.01" />
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
